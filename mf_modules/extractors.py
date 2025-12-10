import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from copy import deepcopy

# ImageDream paths
sys.path.append(os.path.abspath("submodules/ImageDream/extern/ImageDream"))
from omegaconf import OmegaConf
from imagedream.ldm.interface import LatentDiffusionInterface
from imagedream.ldm.util import add_random_background


class ImageDreamGenerator(pl.LightningModule):
    """ImageDream inference wrapper with CFG sampling."""

    guidance_scale: float = 7.5  # classifier‑free guidance strength

    def __init__(self, finetune: bool = False):
        super().__init__()
        base = "submodules/ImageDream/extern/ImageDream"
        cfg = OmegaConf.load(os.path.join(base, "imagedream/configs/sd_v2_base_ipmv.yaml"))
        ckpt = torch.load(os.path.join(base, "release_models/sd-v2.1-base-4view-ipmv.pt"), map_location="cpu")

        self.guidance = LatentDiffusionInterface(
            cfg.model.params.unet_config,
            cfg.model.params.clip_config,
            cfg.model.params.vae_config,
        )
        self.guidance.load_state_dict(ckpt, strict=False)
        self.guidance.cuda()
        if finetune:
            self.guidance.train(); self.guidance.requires_grad_(True)
        else:
            self.guidance.eval();  self.guidance.requires_grad_(False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, shape, cond, steps=200, init_latent=None):
        """DDPM sampling with classifier‑free guidance."""
        bsz = shape[0]
        device = next(self.parameters()).device
        z = init_latent.clone() if init_latent is not None else torch.randn(shape, device=device)
        ts = torch.linspace(self.guidance.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
        for i, t in enumerate(ts):
            # duplicate latent for cond / uncond
            z_in   = torch.cat([z, z], dim=0)          # (2B,4,32,32)
            t_batch = t.repeat(2*bsz)                  # (2B,)

            eps_all = self.guidance.apply_model(z_in, t_batch, cond)  # (2B,4,32,32)
            eps_cond, eps_uncond = eps_all.chunk(2, dim=0)
            eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)

            x0 = self.guidance.predict_start_from_noise(z, t.repeat(bsz), eps)
            if i < steps - 1:
                noise = torch.randn_like(z)
                z = self.guidance.q_sample(x0, t.repeat(bsz) - 1, noise)
            else:
                z = x0
        return self.guidance.decode_first_stage(z)

    # ------------------------------------------------------------------
    def forward(self, image, *, prompt=None, camera=None, ip=None):
        bsz = image.shape[0]
        device = image.device

        # 1. text conditioning ------------------------------------------------
        if prompt is None:
            text_cond = torch.zeros((bsz, 77, 1024), device=device)
        else:
            text_cond = self.guidance.get_learned_conditioning(prompt).repeat(bsz, 1, 1)
        cond = {"context": torch.cat([text_cond, text_cond], dim=0)}  # duplicate for CFG

        # 2. camera conditioning --------------------------------------------
        if camera is None:
            raise ValueError("camera is required")
        cam_vec = camera["camera_vec"]
        cond["camera"] = torch.cat([cam_vec, cam_vec], dim=0)

        # 3. ip conditioning --------------------------------------------------
        if ip is not None:
            if isinstance(ip, torch.Tensor):
                ip_np = (ip.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                ip_pil = Image.fromarray(ip_np).convert("RGBA")
            elif isinstance(ip, np.ndarray):
                ip_pil = Image.fromarray(ip.astype(np.uint8)).convert("RGBA")
            elif isinstance(ip, Image.Image):
                ip_pil = ip.convert("RGBA")
            else:
                raise TypeError(f"unsupported ip type {type(ip)}")
            ip_rgb = add_random_background(ip_pil, bg_color=(255,255,255))
            ip_list = [deepcopy(ip_rgb) for _ in range(2*bsz)]  # duplicated
            cond["ip"] = self.guidance.get_learned_image_conditioning(ip_list)

        # 4. init latent ------------------------------------------------------
        image_norm = image * 2 - 1
        z0 = self.guidance.get_first_stage_encoding(self.guidance.encode_first_stage(image_norm))

        # 5. sample -----------------------------------------------------------
        return self.sample((bsz,4,32,32), cond, steps=200, init_latent=z0)
