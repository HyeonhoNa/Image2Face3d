import os
import sys
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import hparams as hp

ARC2FACE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'Arc2Face'))
if ARC2FACE_PATH not in sys.path:
    sys.path.append(ARC2FACE_PATH)

LIVEPORTRAIT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'LivePortrait', 'src'))
if LIVEPORTRAIT_PATH not in sys.path:
    sys.path.append(LIVEPORTRAIT_PATH)

SMIRK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'smirk', 'src'))
if SMIRK_PATH not in sys.path:
    sys.path.append(SMIRK_PATH)

antelope_root = os.path.join(os.path.dirname(__file__), '..', 'submodules', 'Arc2Face')

from arc2face.models import CLIPTextModelWrapper
from transformers import CLIPTokenizer
from modules.convnextv2 import convnextv2_tiny
from smirk_encoder import SmirkEncoder

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim=512, out_dim=768):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.model(x)


class ProjectFaceEmbedder(nn.Module):
    def __init__(self, text_encoder, tokenizer, embed_dim=768):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.mlp = ProjectionMLP(512, embed_dim)
        self.embed_dim = embed_dim

        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def forward(self, face_embs):
        return self.mlp(face_embs)  


class Arc2FaceIDEncoder(nn.Module):
    def __init__(self, text_encoder=None, tokenizer=None,
                 device=None, finetune: bool=False, ort_device_id=None):
        super().__init__()

        # ① 랭크별 디바이스 고정
        if torch.cuda.is_available():
            if ort_device_id is None and device is None:
                raise ValueError("DDP에서는 ort_device_id(=local_rank)를 반드시 넘겨주세요.")
            dev = torch.device(f"cuda:{int(ort_device_id)}") if ort_device_id is not None else torch.device(device)
        else:
            dev = torch.device("cpu")
        self.device = dev
        self.ort_device_id = (int(ort_device_id) if (ort_device_id is not None and torch.cuda.is_available()) else None)

        # ② ONNXRuntime용 providers (device_id 지정)
        if self.device.type == "cuda":
            providers = [
                ("CUDAExecutionProvider", {"device_id": self.ort_device_id}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        # ③ FaceAnalysis도 해당 GPU만 쓰도록 고정
        self.face_app = FaceAnalysis(name="antelopev2", root=antelope_root, providers=providers)
        self.face_app.prepare(ctx_id=(self.ort_device_id if self.device.type == "cuda" else -1),
                              det_size=(640, 640))

        encoder_path = os.path.join(os.path.dirname(__file__), '..', 'submodules', 'Arc2Face', 'models', 'encoder')
        self.text_encoder = text_encoder or CLIPTextModelWrapper.from_pretrained(encoder_path)
        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        self.face_projector = ProjectFaceEmbedder(self.text_encoder, self.tokenizer).to(self.device)

        if finetune:
            self.train()
            self.face_projector.mlp.requires_grad_(True)
        else:
            self.eval()
            self.face_projector.mlp.requires_grad_(False)

    def extract_arcface_feature(self, img_np: np.ndarray) -> torch.Tensor:
        faces = self.face_app.get(img_np)
        if len(faces) == 0:
            raise ValueError("No face detected.")
        emb = torch.tensor(faces[0]['embedding'], dtype=torch.float32)  # CPU 생성
        emb = emb / torch.norm(emb)
        return emb

    def forward(self, imgs: np.ndarray) -> torch.Tensor:
        """
        imgs: (B, V, H, W, 3) numpy array
        returns: (B, V, D) id embedding
        """
        B, V = imgs.shape[:2]
        results = []
        for b in range(B):
            per_view_emb = []
            for v in range(V):
                emb = self.extract_arcface_feature(imgs[b, v])  # CPU 텐서
                per_view_emb.append(emb)
            per_view_emb = torch.stack(per_view_emb, dim=0).to(self.device)  # 랭크 GPU로 이동
            proj = self.face_projector(per_view_emb)  # (V, Dproj)
            results.append(proj)
        return torch.stack(results)  # (B, V, Dproj)



class LivePortraitEXPEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, device="cuda", finetune=False):
        super().__init__()

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'submodules', 'LivePortrait', 'pretrained_weights', 'liveportrait', 'base_models', 'motion_extractor.pth'
            )

        self.model = convnextv2_tiny()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.device = device

        if finetune:
            self.model.train()
            self.model.requires_grad_(True)
        else:
            self.model.eval()
            self.model.requires_grad_(False)

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        from torchvision.transforms import Resize
        from torchvision.transforms.functional import to_pil_image, to_tensor
        
        """
        imgs: (B, F, H, W, 3) torch.Tensor, dtype = uint8 or float32 (0~1)
        returns dict with keys:
        - 'kp'   : (B, F, 2)
        - 'pose' : (B, F, 3) [pitch,yaw,roll]
        - 'exp'  : (B, F, D_exp)
        - 'scale': (B, F, 1)
        """
        B, F = imgs.shape[:2]
        tensor_list = []

        resize = Resize((256, 256))  # torchvision.transforms

        for b in range(B):
            for f in range(F):
                img = imgs[b, f]  # (H, W, 3)

                if img.dtype != torch.float32:
                    img = img.float() / 255.0  # if uint8

                img = img.permute(2, 0, 1)  # (3, H, W)
                img = resize(img)           # (3, 256, 256)
                tensor_list.append(img)

        batch_tensor = torch.stack(tensor_list).to(self.device)  # (B*F, 3, 256, 256)
        output = self.model(batch_tensor)

        def unflatten(t: torch.Tensor, feat_dim: list[int]):
            return t.view(B, F, *feat_dim)

        kp    = unflatten(output['kp'],    feat_dim=[-1, 2])
        pitch = output['pitch'].view(B, F, -1)
        yaw   = output['yaw'  ].view(B, F, -1)
        roll  = output['roll' ].view(B, F, -1)
        pose  = torch.cat([pitch, yaw, roll], dim=-1)

        exp   = unflatten(output['exp'],   feat_dim=[output['exp'].shape[-1]])
        scale = unflatten(output['scale'], feat_dim=[1])

        return {
            'kp':    kp,        # (B, F, 2)
            'pose':  pose,      # (B, F, 3)
            'exp':   exp,       # (B, F, D_exp)
            'scale': scale,     # (B, F, 1)
        }

class SmirkEXPEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, device="cuda", finetune=False):
        super().__init__()
        CUR  = os.path.dirname(os.path.abspath(__file__))
        ROOT = os.path.abspath(os.path.join(CUR, ".."))
        SUBM = os.path.join(ROOT, "submodules", "smirk")
        if SUBM not in sys.path:
            sys.path.append(SUBM)

        try:
            from src.smirk_encoder import SmirkEncoder
        except ImportError:
            from smirk.src.smirk_encoder import SmirkEncoder

        self.model = SmirkEncoder().to(device)
        self.device = device

        if checkpoint_path is None:
            checkpoint_path = os.path.join(ROOT, "submodules", "smirk", "pretrained_models", "SMIRK_em1.pt")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        enc_sd = {}
        for k, v in sd.items():
            if k.startswith("smirk_encoder."):
                enc_sd[k.replace("smirk_encoder.", "", 1)] = v
        if not enc_sd:
            enc_sd = sd
        self.model.load_state_dict(enc_sd, strict=False)

        if finetune:
            self.model.train()
            self.model.requires_grad_(True)
        else:
            self.model.eval()
            self.model.requires_grad_(False)

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        imgs: (B, F, H, W, 3) torch.Tensor, dtype uint8 or float32
        return dict:
            - "expression_params": (B, F, D_exp)
        """
        from torchvision.transforms import Resize

        B, F = imgs.shape[:2]
        resize = Resize((224, 224))   # smirk input size
        tensor_list = []

        for b in range(B):
            for f in range(F):
                img = imgs[b, f]
                if img.dtype != torch.float32:
                    img = img.float() / 255.0
                img = img.permute(2, 0, 1)  # (3,H,W)
                img = resize(img)
                tensor_list.append(img)

        batch_tensor = torch.stack(tensor_list).to(self.device)  # (B*F,3,224,224)

        with torch.no_grad():
            out = self.model(batch_tensor)

        if isinstance(out, dict) and "expression_params" in out:
            exp = out["expression_params"]
        elif isinstance(out, torch.Tensor):
            exp = out
        else:
            raise RuntimeError(f"Unexpected SMIRK output: {type(out)}")

        exp = exp.view(B, F, -1)
        return {"expression_params": exp}

# === embeddings.py: AppearanceEncoder ========================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class AppearanceEncoder(nn.Module):
    """
    Appearance(머리/의복/피부톤/악세서리 + 제한적 실루엣 단서) 전용 임베딩 인코더.
    - 입력: (B,3,H,W) 또는 (B,V,3,H,W)  [RGB, 0..1 또는 uint8]
    - 출력: (B,app_dim) 또는 (B,V,app_dim)
    - 백본 우선순위: timm dinov2 -> torchvision vit_b_16 -> torchvision resnet50
    """
    def __init__(
        self,
        app_dim: int = 768,
        backbone: str = "auto",     # "auto" | "dinov2_s14" | "vit_b_16" | "resnet50"
        finetune: bool = False,
        l2norm: bool = True,
        dropout_p: float = 0.0,
        input_size: int = 224,
        device: str = "cuda",
    ):
        super().__init__()
        self.app_dim   = int(app_dim)
        self.l2norm    = bool(l2norm)
        self.input_sz  = int(input_size)
        self.device    = device

        self.backbone, feat_dim = self._build_backbone(backbone)
        self.proj = nn.Linear(feat_dim, self.app_dim)

        self.pool = _MixPool()                      # GAP + GMP 혼합(텍스처/하이라이트 견고)
        self.drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.to(self.device)
        if not finetune:
            self.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ImageNet 정규화
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    # ---------------- public ----------------
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B,3,H,W) or (B,V,3,H,W)
        returns: (B,app_dim) or (B,V,app_dim)
        """
        if imgs.dim() == 5:
            B, V = imgs.shape[:2]
            x = imgs.view(B*V, *imgs.shape[2:])
            emb = self._encode_one(x)                     # (B*V, D)
            return emb.view(B, V, -1)
        else:
            return self._encode_one(imgs)

    # ---------------- helpers ---------------
    @torch.no_grad()
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:   # (3,H,W) -> (1,3,H,W)
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.float().div(255.0)
        if x.shape[-2:] != (self.input_sz, self.input_sz):
            x = F.interpolate(x, size=(self.input_sz, self.input_sz),
                              mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return x

    def _encode_one(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(imgs).to(self.device)        # (B,3,S,S)
        feats = self._forward_backbone(x)                 # (B,C,H',W') or (B,D)
        feats = self.pool(feats)                          # (B,C) or (B,D)
        feats = self.drop(feats)
        emb = self.proj(feats)                            # (B,app_dim)
        if self.l2norm:
            emb = F.normalize(emb, dim=-1, eps=1e-6)
        return emb

    # --------------- backbone builders -------
    def _build_backbone(self, backbone: str):
        # 1) timm dinov2 우선 시도
        if backbone in ("auto", "dinov2_s14", "dinov2"):
            try:
                import timm
                model_name = "vit_small_patch14_dinov2"
                net = timm.create_model(model_name, pretrained=True, num_classes=0)  # (B,D)
                feat_dim = net.num_features
                self._forward_backbone = lambda x: net(x)   # (B,D)
                return net, feat_dim
            except Exception:
                if backbone not in ("auto",):
                    raise

        # 2) torchvision vit_b_16
        if backbone in ("auto", "vit_b_16"):
            try:
                from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
                vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
                vit.heads = nn.Identity()
                feat_dim = 768
                def vit_forward(x):
                    return vit(x)                           # (B,768)
                self._forward_backbone = vit_forward
                return vit, feat_dim
            except Exception:
                if backbone == "vit_b_16":
                    raise

        # 3) fallback: torchvision resnet50
        from torchvision.models import resnet50, ResNet50_Weights
        rn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = rn.fc.in_features  # 2048
        rn.fc = nn.Identity()
        def rn_forward(x):
            return rn(x)                                   # (B,2048)
        self._forward_backbone = rn_forward
        return rn, feat_dim


class _MixPool(nn.Module):
    """
    CNN/ViT 모두 호환되는 혼합 풀링:
    - (B,C,H,W) -> GAP+GMP concat
    - (B,D)     -> 그대로 반환 (ViT)
    """
    def __init__(self): super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:   # (B,D)
            return x
        gap = x.mean(dim=[-2,-1])                            # (B,C)
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)         # (B,C)
        return torch.cat([gap, gmp], dim=1)                  # (B,2C)
