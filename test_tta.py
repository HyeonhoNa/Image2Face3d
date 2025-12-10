# test_tta.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime

import torch
import torch.nn.functional as F

from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
from lpips import LPIPS
import pytorch_ssim

# Repository
from mf_modules import (
    Arc2FaceIDEncoder,
    AppearanceEncoder,
    UVDecoder2,
)
from utils import load_cam
from utils.ckpt_utils import save_checkpoint2, load_checkpoint2
import hparams as hp
from loader.dataset_all import CustomDataset
from PIL import Image
import argparse
import csv

from glob import glob

def _sanitize_for_raster(s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac):
    def _ff(x): return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    s_pos   = _ff(s_pos)
    s_quat  = torch.nn.functional.normalize(_ff(s_quat), dim=-1, eps=1e-6)
    s_scale = _ff(s_scale).clamp(1e-4, 1e2)
    s_shdc  = _ff(s_shdc)
    s_shr   = _ff(s_shr)
    s_opac  = _ff(s_opac).clamp(1e-6, 0.995)
    return s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac

def compute_psnr(pred, target, eps=1e-8):
    """pred, target: (..., 3, H, W) in [0,1]"""
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))

class InceptionFeatureExtractor(torch.nn.Module):
    """Inception-v3 pool3(2048-d) feature"""
    def __init__(self, device="cuda"):
        super().__init__()
        w = models.Inception_V3_Weights.IMAGENET1K_V1
        net = models.inception_v3(weights=w, aux_logits=True).eval().to(device)
        self.extractor = create_feature_extractor(net, return_nodes={"avgpool": "feat"}).to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1))

    @torch.no_grad()
    def forward(self, x):
        if x.device != self.mean.device:
            x = x.to(self.mean.device)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        out = self.extractor(x)["feat"]        # (N,2048,1,1)
        return out.squeeze(-1).squeeze(-1)     # (N,2048)

def _matrix_sqrt_sym(A, eps=1e-6):
    A = 0.5 * (A + A.T)
    vals, vecs = torch.linalg.eigh(A)
    vals = torch.clamp(vals, min=eps)
    return (vecs @ torch.diag(torch.sqrt(vals)) @ vecs.T)

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = (mu1 - mu2)
    cov_prod = sigma1 @ sigma2
    covmean = _matrix_sqrt_sym(cov_prod, eps=eps)
    tr = torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    return (diff @ diff) + tr

@torch.no_grad()
def compute_fid_from_features(feats_real, feats_fake, eps=1e-6):
    fr = feats_real.to(torch.float64)
    ff = feats_fake.to(torch.float64)
    mu_r = fr.mean(dim=0)
    mu_f = ff.mean(dim=0)
    Sr = torch.cov(fr.T)
    Sf = torch.cov(ff.T)
    fid = frechet_distance(mu_r, Sr, mu_f, Sf, eps=eps)
    return float(fid.item())

# -------------------------
# ID embedding helpers & batch filter
# -------------------------
def get_id_embs_from_batch(batch, device, arc2face_fallback=None):
    if "source_id_emb" in batch and batch["source_id_emb"] is not None:
        return batch["source_id_emb"].to(device).float()
    for k in ("id_embs", "source_arcface_emb", "arcface_id_emb"):
        if k in batch and batch[k] is not None:
            return batch[k].to(device).float()
    if arc2face_fallback is None:
        raise RuntimeError("No precomputed ID embeddings in batch and no fallback encoder provided.")
    src_imgs = batch["source_images"]  # (B,3,H,W)
    id_list = []
    with torch.no_grad():
        for src in src_imgs:
            src_np = (src.detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
            src_np = np.clip(src_np, 0, 255).astype(np.uint8)
            input_img = src_np[None, None, ...]  # (1,1,H,W,3)
            id_emb = arc2face_fallback(input_img)  # (1,1,D)
            id_emb = id_emb.squeeze(0).squeeze(0)  # (D,)
            id_list.append(id_emb)
    return torch.stack(id_list, dim=0).to(device).float()  # (B,D)

def _first_dim_equals(x, B):
    return torch.is_tensor(x) and x.dim() >= 1 and x.shape[0] == B

def filter_batch_by_valid_id(batch, device, arc2face_fallback=None, eps=1e-6):
    id_embs = get_id_embs_from_batch(batch, device, arc2face_fallback)  # (B,512)
    B = id_embs.shape[0]
    ok = torch.isfinite(id_embs).all(dim=1) & (id_embs.norm(dim=1) > eps)
    if "source_id_ok" in batch and batch["source_id_ok"] is not None:
        ok = ok & batch["source_id_ok"].to(ok.device).bool()

    keep_idx = ok.nonzero(as_tuple=False).squeeze(-1)
    n_keep = int(keep_idx.numel())

    if n_keep == 0:
        return None, None, keep_idx
    if n_keep == B:
        return batch, id_embs, keep_idx

    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if _first_dim_equals(v, B):
                new_batch[k] = v.index_select(0, keep_idx)
            else:
                new_batch[k] = v
        elif isinstance(v, (list, tuple)):
            if len(v) == B:
                new_batch[k] = type(v)([v[i] for i in keep_idx.tolist()])
            else:
                new_batch[k] = v
        else:
            new_batch[k] = v

    id_embs = id_embs.index_select(0, keep_idx)
    return new_batch, id_embs, keep_idx

# -------------------------
# Save .ply
# -------------------------

import struct

def _infer_sh_degree_from_rest(C_rest_per_rgb: int) -> int:
    # per-channel rest = (d+1)^2 - 1  ->  d = sqrt(rest+1) - 1
    import math
    return int(round(math.sqrt(C_rest_per_rgb + 1) - 1))

def save_gaussians_to_ply(
    path,
    s_pos,      # (..., N, 3)
    s_quat,     # (..., N, 4)  quaternion
    s_scale,    # (..., N, 3)
    s_shdc,     # (..., N, 3)  DC color in [0,1]
    s_shr=None, # (..., N, C)  
    s_opac=None,# (..., N, 1)  opacity in [0,1]
    *,
    quat_is_wxyz=True
):
    """
    Accepts both 6-arg and 7-arg calls. s_shr는 넘어오면 무시(PLY에는 미사용).
    쿼터니언은 기본 (w,x,y,z) 순서로 저장. 필요 시 quat_is_wxyz=False로 (x,y,z,w)->(w,x,y,z) 변환.
    """
    import numpy as np, os, torch

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with torch.no_grad():
        def to_np(x, last_dim):
            if x is None:
                return None
            x = x.detach().cpu().float()
            x = x.view(-1, last_dim)  # (..., last_dim) -> (N, last_dim)
            return x.numpy()

        P = to_np(s_pos,  3)  # (N,3)
        Q = to_np(s_quat, 4)  # (N,4)
        S = to_np(s_scale,3)  # (N,3)
        C = to_np(s_shdc, 3)  # (N,3)
        A = to_np(s_opac, 1)  # (N,1) or None

        if P is None or Q is None or S is None or C is None:
            raise ValueError("save_gaussians_to_ply: required tensors (pos, quat, scale, shdc) are None.")

        if not quat_is_wxyz:
            # (x,y,z,w) -> (w,x,y,z)
            Q = np.concatenate([Q[:, 3:4], Q[:, 0:3]], axis=1)

        if A is None:
            A = np.ones((P.shape[0], 1), dtype=np.float32)

        N = P.shape[0]
        V = np.concatenate([P, S, Q, C, A], axis=1)  # (N, 14)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float scale_x\nproperty float scale_y\nproperty float scale_z\n"
        "property float qw\nproperty float qx\nproperty float qy\nproperty float qz\n"
        "property float r\nproperty float g\nproperty float b\n"
        "property float opacity\n"
        "end_header\n"
    )

    with open(path, "w") as f:
        f.write(header)
        # x y z sx sy sz qw qx qy qz r g b a
        np.savetxt(f, V, fmt="%.6f")


def _append_line(path, line):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _log_skip_seedroot(seed_root_path, reason="no_valid_id_embeddings", split_name=None, log_path=None):
    ts = datetime.now().isoformat(timespec="seconds")
    tag = f" [{split_name}]" if split_name else ""
    lp = log_path or os.path.join(os.getcwd(), "tta_skipped_batch.log")
    _append_line(lp, f"{ts}{tag}  {reason}  {seed_root_path}")

def _log_metrics_csv(seed_root_path, split_name, phase, metrics: dict, log_path: str):
    cols = ["time","split","phase","seed_root","total","SSIM","L2","MaskL1","LPIPS","PSNR","FID","best_step","sh_degree"]
    dirpath = os.path.dirname(log_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    need_header = not os.path.exists(log_path)

    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "split": split_name,
        "phase": phase,  # "test" or "opt"
        "seed_root": seed_root_path,
        "total": metrics.get("loss", metrics.get("total", "")),
        "SSIM":  metrics.get("ssim", metrics.get("SSIM_avg", "")),
        "L2":    metrics.get("L2", ""),
        "MaskL1":metrics.get("maskl1", metrics.get("MaskL1","")),
        "LPIPS": metrics.get("lpips", metrics.get("LPIPS","")),
        "PSNR":  metrics.get("psnr",""),
        "FID":   metrics.get("fid",""),
        "best_step": metrics.get("best_step",""),
        "sh_degree": metrics.get("sh_degree",""),
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if need_header:
            w.writeheader()
        w.writerow(row)


def _any_pt_under(d):
    return os.path.isdir(d) and len(glob(os.path.join(d, "*.pt"))) > 0

def _latest_pt(d):
    pts = glob(os.path.join(d, "*.pt"))
    return max(pts, key=os.path.getmtime) if pts else None

def _exp_frame_dir(root, fi):
    d = os.path.join(root, "exp", f"frame_{fi:04d}")
    os.makedirs(d, exist_ok=True)
    return d

# -------------------------
# Test
# -------------------------
@torch.no_grad()
def run_test(displacementDecoder, arc2FaceIDEncoder, appearanceEncoder,
             lpips_fn, cam_matrix, cam_params,
             loader, max_batches=10, sh_interval=1000, device="cuda",
             save_images=False, timestamp="none", global_step=0, split_tag="eval",
             save_ply=False, ply_path=None,
             seed_root_path=None, split_name=None):

    displacementDecoder.eval()
    displacementDecoder.enable_delta(True)
    appearanceEncoder.eval()

    total_l1 = total_mask_l1 = total_lpips = total_ssim = total_loss = 0.0
    total_psnr = 0.0
    n_seen = 0

    fid_max = int(getattr(hp, "fid_max_samples", 1024))
    inc_extractor = InceptionFeatureExtractor(device=device)
    feats_real_list, feats_fake_list = [], []
    fid_collected = 0

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch_f, id_embs, keep_idx = filter_batch_by_valid_id(batch, device, arc2face_fallback=arc2FaceIDEncoder)
        if batch_f is None:
            _log_skip_seedroot(seed_root_path, reason="no_valid_id_embeddings_in_test",
                       split_name=split_tag, log_path=args.skip_log)
            continue

        source = batch_f["source_images"].to(device)
        gt = batch_f["ground_truth_images"].to(device)
        gt_mask = batch_f["ground_truth_mask_images"].to(device)
        yaw_tensor = batch_f["yaw_tensor"].to(device)

        B, V, T, C, H, W = gt.shape

        code_id = id_embs
        code_id = code_id.unsqueeze(1).expand(-1, T, -1).contiguous()
        with torch.no_grad():
            code_app = appearanceEncoder(source)
        code_app = code_app.unsqueeze(1).expand(-1, T, -1).contiguous()

        # deform
        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = displacementDecoder(code_id, code_app)
        cur_sh = displacementDecoder.active_sh_degree
        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = _sanitize_for_raster(s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac)
        if save_ply and bi == 0:
            out_path = ply_path
            if out_path is None:
                os.makedirs("results", exist_ok=True)
                out_path = os.path.join("results", f"{timestamp}_gaussians.ply")
            save_gaussians_to_ply(
                out_path,
                s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac
            )

        # render & loss
        batch_total = torch.zeros((), device=device)
        b_l1 = torch.zeros((), device=device)
        b_mask_l1 = torch.zeros((), device=device)
        b_lpips = torch.zeros((), device=device)
        b_ssim = torch.zeros((), device=device)
        b_psnr = torch.zeros((), device=device)

        do_save = bool(save_images)
        preds_v = [] if do_save else None

        for v in range(V):
            yaw_b = yaw_tensor[:, v].float()  # (B,)
            preds = []
            for b in range(B):
                preds.append(displacementDecoder.render(
                    s_pos[b:b+1], s_quat[b:b+1], s_scale[b:b+1],
                    s_shdc[b:b+1], s_shr[b:b+1], s_opac[b:b+1],
                    cam_matrix.float(), cam_params, float(yaw_b[b].item()), cur_sh)
                )
            pred = torch.cat(preds, dim=0)

            if do_save:
                preds_v.append(pred.detach())

            gt_v = gt[:, v]
            gt_mask_v = gt_mask[:, v]
            mask = (gt_mask_v > 0.5).float()

            loss_l1 = F.l1_loss(pred, gt_v)
            abs_err = torch.abs(gt_v - pred)
            face_sum = (abs_err * mask).sum()
            bg_sum = (abs_err * (1-mask)).sum()
            faces = mask.sum()
            bgs = (1 - mask).sum()
            loss_face = face_sum / (faces + 1e-6)
            loss_bg = bg_sum / (bgs + 1e-6)
            loss_mask_l1 = hp.w_face * loss_face + hp.w_background * loss_bg

            loss_lpips = lpips_fn((pred.flatten(0,1) * 2.0 - 1.0).float(), (gt_v.flatten(0,1) * 2.0 - 1.0).float()).mean()
            loss_ssim = 1.0 - pytorch_ssim.ssim(pred.flatten(0,1).float(), gt_v.flatten(0,1).float())

            psnr_val = compute_psnr(pred.flatten(0,1), gt_v.flatten(0,1))

            loss = loss_l1 + loss_mask_l1 + loss_lpips + loss_ssim
            batch_total += loss
            b_l1 += loss_l1
            b_mask_l1 += loss_mask_l1
            b_lpips += loss_lpips
            b_ssim += loss_ssim
            b_psnr += psnr_val

            if fid_collected < fid_max:
                p_bt = pred.flatten(0,1).clamp(0,1)
                g_bt = gt_v.flatten(0,1).clamp(0,1)
                take = min(fid_max - fid_collected, p_bt.shape[0])
                if take > 0:
                    fp = inc_extractor(p_bt[:take])
                    fr = inc_extractor(g_bt[:take])
                    feats_fake_list.append(fp.detach().cpu())
                    feats_real_list.append(fr.detach().cpu())
                    fid_collected += take

        # ---- SAVE PREVIEWS (with src) ----
        if do_save and preds_v:
            preds_bvt = torch.stack(preds_v, dim=1)  # (B, V_succ, T, 3, H, W)
            Bp, Vp, Tp = preds_bvt.shape[:3]
            Vg, Tg = gt.shape[1], gt.shape[2]
            Vc = min(Vp, Vg)
            Tc = min(Tp, Tg)

            if bi == 0:
                print(f"[eval/save] shapes gt(B,V,T,3,H,W)={tuple(gt.shape)} "
                      f"preds_bvt(B,V,T,3,H,W)={tuple(preds_bvt.shape)} -> Vc={Vc}, Tc={Tc}")

            if Vc > 0 and Tc > 0:
                for b in range(Bp):
                    seed_text = str(batch_f["src_id"][b])
                    view_nums = batch_f["target_view_nums"][b].tolist()
                    src_tile = None

                    if isinstance(source, torch.Tensor):
                        if source.dim() == 4:  # (B,3,H,W)
                            src_tile = source[b]
                        elif source.dim() == 5:  # (B,T,3,H,W)
                            src_tile = source[b, 0] 

                    displacementDecoder.save_image(
                        gt[b, :Vc, :Tc],
                        preds_bvt[b, :Vc, :Tc],
                        global_step, src=src_tile,
                        timestamp=timestamp,
                        tag=f"test_{split_tag}_b{bi:03d}_s{b:02d}",
                        seed_text=seed_text,
                        view_nums=view_nums
                    )
                    if seed_root_path is not None:
                        render_dir = os.path.join(seed_root_path, "render")
                        if split_name is not None:
                            render_dir = os.path.join(render_dir, str(split_name))
                        os.makedirs(render_dir, exist_ok=True)

                        Bp, Vc, Tc = preds_bvt.shape[:3]
                        for b in range(Bp):
                            for i in range(Vc):
                                img = preds_bvt[b, i, 0].clamp(0, 1)  # (3,H,W)
                                img_u8 = (img * 255).byte().permute(1, 2, 0).cpu().numpy()

                                if isinstance(view_nums, list) and i < len(view_nums):
                                    view_id = int(view_nums[i])
                                else:
                                    view_id = i

                                fname = f"view_{view_id:02d}.png" 
                                Image.fromarray(img_u8).save(os.path.join(render_dir, fname))
                        print(f"[test/render] saved per-view PNGs to {render_dir}")
            del preds_bvt, preds_v

        Vt = float(V)
        total_l1 += (b_l1 / Vt).item()
        total_mask_l1 += (b_mask_l1 / Vt).item()
        total_lpips += (b_lpips / Vt).item()
        total_ssim += (b_ssim / Vt).item()
        total_loss += (batch_total / Vt).item()
        total_psnr += (b_psnr / Vt).item()
        n_seen += 1

    if n_seen == 0:
        return {}

    if len(feats_fake_list) >= 1 and len(feats_real_list) >= 1:
        feats_fake = torch.cat(feats_fake_list, dim=0)
        feats_real = torch.cat(feats_real_list, dim=0)
        fid_val = compute_fid_from_features(feats_real, feats_fake) if (feats_fake.shape[0] >= 32 and feats_real.shape[0] >= 32) else float("nan")
    else:
        fid_val = float("nan")

    return_stats = {
        "l1": total_l1 / n_seen,
        "maskl1": total_mask_l1 / n_seen,
        "lpips": total_lpips / n_seen,
        "ssim": total_ssim / n_seen,
        "psnr": total_psnr / n_seen,
        "fid": fid_val,
        "loss": total_loss / n_seen,
        "batches": n_seen,
    }

    try:
        _log_metrics_csv(seed_root_path, split_tag, "test", return_stats, args.metrics_log)
    except Exception as e:
        print(f"[metrics-log][test][warn] {e}")
    return return_stats


def run_optimize(
    displacementDecoder, arc2FaceIDEncoder, appearanceEncoder,
    lpips_fn, cam_matrix, cam_params, loader,
    device="cuda", save_tag="tta", timestamp="none",
    max_steps=None, ssim_target=0.94,         
    save_every=5000, warmup_steps=500, cosine_min_lr=0.01, view_chunk=4,
    state_save_path=None, state_load_path=None, resume_optimizer=False,
    phase1_steps=1500,                         
    split_name: str = "neutral",
    seed_root_path: str | None = None,
    frame_dir: str | None = None,
):
    import os, math, torch
    import torch.nn.functional as F
    import pytorch_ssim
    import hparams as hp

    dec = displacementDecoder
    dec.enable_delta(True)
    dec.eval(); appearanceEncoder.eval()
    for p in dec.parameters(): p.requires_grad_(False)

    # ---------- Hyperparams ----------
    lr_geo_base = float(getattr(hp, "tta_lr_geo",      1e-2))
    lr_app_base = float(getattr(hp, "tta_lr_app",      1e-2))   # shdc
    lr_shr_base = float(getattr(hp, "tta_lr_sh_rest",  5e-3))   # shr
    lr_opa_base = float(getattr(hp, "tta_lr_opa",      5e-3))

    lam_geo  = float(getattr(hp, "tta_reg_geo",     1e-5))
    lam_shdc = float(getattr(hp, "tta_reg_sh",      1e-4))
    lam_shr  = float(getattr(hp, "tta_reg_sh_rest", 1e-4))
    lam_opa  = float(getattr(hp, "tta_reg_opa",     1e-4))

    max_steps = int(max_steps if max_steps is not None else getattr(hp, "tta_max_steps", 2000))

    # ---- Loss weights ----
    wl2      = 1.0
    wmaskl1  = 4.0           
    wlpips   = 0.0           
    wssim    = 0.0           
    opa_floor = 0.03        

    w_face = float(getattr(hp, "w_face",        0.9))
    w_bg   = float(getattr(hp, "w_background",  0.1))

    # Delta gains
    r_pos      = float(getattr(hp, "tta_r_pos",      0.10))
    k_rot      = float(getattr(hp, "tta_k_rot",      1.0))
    k_scl      = float(getattr(hp, "tta_k_scl",      0.3))
    k_col_dc   = float(getattr(hp, "tta_k_col",      0.15))
    k_col_rest = float(getattr(hp, "tta_k_col_rest", 0.15))

    # ---------- Load one sample ----------
    batch = next(iter(loader))
    for k, v in list(batch.items()):
        if torch.is_tensor(v): batch[k] = v.to(device)

    from __main__ import filter_batch_by_valid_id
    batch_f, id_embs, _ = filter_batch_by_valid_id(batch, device, arc2face_fallback=arc2FaceIDEncoder)
    if batch_f is None:
        _log_skip_seedroot(seed_root_path, reason="no_valid_id_embeddings_in_optimize",
                           split_name=split_name, log_path=args.skip_log)
        print(f"[TTA][SKIP] {split_name}: no valid ID embeddings — skipping this seed.")
        try:
            _log_metrics_csv(seed_root_path, split_name, "opt",
                             {"loss":"","SSIM_avg":"","L2":"","MaskL1":"","LPIPS":"",
                              "psnr":"","fid":"","best_step":"skipped","sh_degree":displacementDecoder.active_sh_degree},
                             args.metrics_log)
        except Exception as e:
            print(f"[metrics-log][opt-skip][warn] {e}")
        return {"skipped": True, "reason": "no_valid_id_embeddings"}
    with torch.no_grad():
        code_app = appearanceEncoder(batch_f["source_images"]).unsqueeze(1)  # (B,1,Dapp)
    code_id = id_embs.unsqueeze(1)                                           # (B,1,Did)

    # ---------- Trainables(d_*) ----------
    delta_down_default = 16
    with torch.no_grad():
        b_pos, b_quat, b_scale, b_shdc, b_shr, b_opa = dec.export_UV(
            code_id, code_app, delta_down=delta_down_default, t_chunk=1, amp=True, return_delta=False
        )
        b_pos  = b_pos.detach().contiguous()
        b_quat = b_quat.detach().contiguous()
        b_scale= b_scale.detach().contiguous()
        b_shdc = b_shdc.detach().contiguous()
        b_shr  = b_shr.detach().contiguous()
        b_opa  = b_opa.detach().contiguous()
    _, _, _, H, W = b_pos.shape
    C_shr = int(b_shr.shape[2])

    d_pos  = torch.nn.Parameter(torch.zeros(3,     H, W, device=device))
    d_rot  = torch.nn.Parameter(torch.zeros(3,     H, W, device=device))
    d_slog = torch.nn.Parameter(torch.zeros(3,     H, W, device=device))
    d_shdc = torch.nn.Parameter(torch.zeros(3,     H, W, device=device))
    d_shr  = torch.nn.Parameter(torch.zeros(C_shr, H, W, device=device))
    d_olog = torch.nn.Parameter(torch.zeros(1,     H, W, device=device))

    # ---------- Optimizer ----------
    opt = torch.optim.Adam([
        {"params": [d_pos, d_rot, d_slog], "lr": lr_geo_base, "name": "geo"},
        {"params": [d_shdc],               "lr": 0.0,         "name": "shdc"},  # Phase1 off
        {"params": [d_shr],                "lr": 0.0,         "name": "shr"},   # Phase1 off
        {"params": [d_olog],               "lr": 0.0,         "name": "opa"},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.startswith("cuda")))
    base_lrs = [g["lr"] for g in opt.param_groups]

    loaded_meta = {}
    if state_load_path and os.path.isfile(state_load_path):
        ck = torch.load(state_load_path, map_location=device)
        for k, t in [("d_pos", d_pos), ("d_rot", d_rot), ("d_slog", d_slog),
                     ("d_shdc", d_shdc), ("d_shr", d_shr), ("d_olog", d_olog)]:
            if k in ck:
                with torch.no_grad(): t.copy_(ck[k].to(device))
        if "code_id" in ck:
            code_id = ck["code_id"].to(device)
            if code_id.dim() == 2: code_id = code_id.unsqueeze(1)
        if "code_app" in ck:
            code_app = ck["code_app"].to(device)
            if code_app.dim() == 2: code_app = code_app.unsqueeze(1)
        if resume_optimizer and ("opt" in ck):
            try:
                opt.load_state_dict(ck["opt"]); print(f"[TTA] optimizer state resumed from {state_load_path}")
            except Exception as e:
                print(f"[TTA] optimizer resume failed: {e}")
        if "meta" in ck and isinstance(ck["meta"], dict):
            loaded_meta = dict(ck["meta"])
        print(f"[TTA] deltas & codes resumed from {state_load_path}")

    with torch.no_grad():
        ddn = int(loaded_meta.get("delta_down", delta_down_default))
        b_pos, b_quat, b_scale, b_shdc, b_shr, b_opa = dec.export_UV(
            code_id, code_app, delta_down=ddn, t_chunk=1, amp=True, return_delta=False
        )
        b_pos  = b_pos.detach().contiguous()
        b_quat = b_quat.detach().contiguous()
        b_scale= b_scale.detach().contiguous()
        b_shdc = b_shdc.detach().contiguous()
        b_shr  = b_shr.detach().contiguous()
        b_opa  = b_opa.detach().contiguous()
    _, _, _, H, W = b_pos.shape  

    def lr_multiplier(step):
        if step < warmup_steps:
            return max(1e-4, (step + 1) / float(warmup_steps))
        t = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        return cosine_min_lr + 0.5 * (1 - cosine_min_lr) * (1 + math.cos(math.pi * min(1.0, t)))

    def build_splats():
        b_pos_, b_quat_, b_scale_, b_shdc_, b_shr_, b_opa_ = b_pos, b_quat, b_scale, b_shdc, b_shr, b_opa

        pos  = b_pos_ + torch.tanh(d_pos)[None, None] * r_pos
        rv   = (k_rot * torch.tanh(d_rot))[None, None]
        dq   = dec._rotvec_to_quat(rv[:, 0]).unsqueeze(1)
        quat = F.normalize(dec._quat_mul(dq.squeeze(1), b_quat_.squeeze(1)), dim=1, eps=1e-8).unsqueeze(1)

        smin, smax = float(dec.s_min), float(dec.s_max)
        p = (b_scale_ - smin) / max(1e-8, (smax - smin))
        p = p.clamp(1e-6, 1 - 1e-6)
        slog  = torch.logit(p) + (k_scl * d_slog)[None, None]
        scale = smin + (smax - smin) * torch.sigmoid(slog)

        sh_dc = b_shdc_ + k_col_dc   * torch.tanh(d_shdc)[None, None]
        sh_rs =  b_shr_ + k_col_rest * torch.tanh(d_shr )[None, None]

        uv_valid = dec.uv_valid.to(b_opa_.dtype)
        # of   = ((b_opa_.clamp(0,1) - opa_floor) / max(1e-6, (1 - opa_floor))).clamp(1e-6, 1 - 1e-6)
        # olog = torch.logit(of) + d_olog[None, None]
        # opacity = opa_floor + (1.0 - opa_floor) * (torch.sigmoid(olog) * uv_valid)
        eps = 1e-6
        uv_valid = dec.uv_valid.to(b_opa_.dtype)
        p = b_opa_.clamp(eps, 1 - eps)                   
        olog = torch.logit(p) + d_olog[None, None]        
        opacity = torch.sigmoid(olog) * uv_valid        
        opacity = opacity.clamp_min(opa_floor)  

        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = dec.export_gaussian_splats(
            pos, quat, scale, sh_dc, sh_rs, torch.nan_to_num(opacity, nan=opa_floor).clamp(0.0, 1.0),
            N=dec.num_points, weighted=True, replacement=False, temperature=0.7
        )
        return _sanitize_for_raster(s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac)

    def forward_loss_chunk(splats, view_idxes):
        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = splats
        l2_sum  = torch.zeros((), device=device)
        ml1_sum = torch.zeros((), device=device)
        ssim_s  = torch.zeros((), device=device)
        lpip_s  = torch.zeros((), device=device)
        for v in view_idxes:
            yaw_deg = float(batch["yaw_tensor"][:, v].item())
            with torch.amp.autocast('cuda', enabled=False):
                pred_v = dec.render(
                    s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac,
                    cam_matrix.float(), cam_params, yaw_deg, dec.active_sh_degree
                )
            if pred_v.dim() == 5: pred_v = pred_v[:, 0]
            gt_v   = batch["ground_truth_images"][:, v, 0]
            gmsk_v = batch["ground_truth_mask_images"][:, v, 0]
            if gmsk_v.shape[1] > 1: gmsk_v = gmsk_v.mean(1, keepdim=True)
            m_v = (gmsk_v > 0.5).float()

            face_cnt = m_v.sum(); bg_cnt = (1.0 - m_v).sum()
            sq = (pred_v - gt_v) ** 2
            l2_face = (sq * m_v).sum() / (face_cnt + 1e-6)
            l2_bg   = (sq * (1.0 - m_v)).sum() / (bg_cnt + 1e-6)
            l2_sum  += (w_face * l2_face + w_bg * l2_bg)
            l1_abs   = (pred_v - gt_v).abs()
            ml1_face = (l1_abs * m_v).sum() / (face_cnt + 1e-6)
            ml1_bg   = (l1_abs * (1.0 - m_v)).sum() / (bg_cnt + 1e-6)
            ml1_sum  += (w_face * ml1_face + w_bg * ml1_bg)
            
            pv_m = pred_v * m_v
            gv_m = gt_v * m_v

            ssim_s += 1.0 - pytorch_ssim.ssim(pv_m.clamp(0,1), gv_m.clamp(0,1))
            lpip_s += lpips_fn((pv_m.clamp(0,1) * 2 - 1).float(),
                               (gv_m.clamp(0,1) * 2 - 1).float()).mean()

        Vt = float(len(view_idxes))
        loss_l2     = l2_sum / Vt
        loss_maskl1 = ml1_sum/ Vt
        loss_ssim   = ssim_s/ Vt
        loss_lpips  = lpip_s/ Vt
        reg_geo  = (d_pos.pow(2).mean() + d_rot.pow(2).mean() + d_slog.pow(2).mean())
        reg_shdc = d_shdc.pow(2).mean()
        reg_shr  = d_shr.pow(2).mean()
        reg_opa  = d_olog.pow(2).mean()
        reg      = lam_geo * reg_geo + lam_shdc * reg_shdc + lam_shr * reg_shr + lam_opa * reg_opa
        total = (wl2 * loss_l2 + wmaskl1 * loss_maskl1 + wlpips * loss_lpips + reg)  # SSIM 제외
        return total, loss_ssim, loss_l2, loss_maskl1, loss_lpips

    # -------- Optimize (chunked) --------
    V_all = int(batch["ground_truth_images"].shape[1])
    view_idx_all = list(range(V_all))
    chunk_size = max(1, min(view_chunk, V_all))
    cursor = 0
    phase2 = False  

     with torch.no_grad():
        splats0 = build_splats()
        t0, _, l20, m10, lp0 = forward_loss_chunk(splats0, view_idx_all[:chunk_size])
        _, ssim0, _, _, _ = forward_loss_chunk(splats0, [0])
    best_total = float(t0.item())
    best_maps  = splats0
    best_step  = 0
    best_snapshot = dict(
        total=best_total,
        SSIM_avg=1.0 - float(ssim0.item()),
        L2=float(l20.item()),
        MaskL1=float(m10.item()),
        LPIPS=float(lp0.item()),
    )

    step = 0
    LOG_EVERY = 100
    while step < max_steps:
        mult = lr_multiplier(step)
        for g, base in zip(opt.param_groups, base_lrs): g["lr"] = base * mult

        chunk_idx = view_idx_all[cursor:cursor+chunk_size]
        if len(chunk_idx) < chunk_size:
            chunk_idx += view_idx_all[:(chunk_size - len(chunk_idx))]
        cursor = (cursor + chunk_size) % V_all

        splats = build_splats()
        with torch.amp.autocast('cuda', enabled=(device.startswith("cuda"))):
            total, _, l2, ml1, lp = forward_loss_chunk(splats, chunk_idx)

        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        torch.nn.utils.clip_grad_norm_([d_pos, d_rot, d_slog, d_shdc, d_shr, d_olog], 1.0)
        scaler.step(opt); scaler.update()

        if (not phase2) and (step >= int(phase1_steps)):
            phase2 = True
            wmaskl1 = 1.0      
            wlpips  = 1.0      
            opa_floor = 0.01     
            for g in opt.param_groups:
                if g.get("name") == "geo":  g["lr"] = g["lr"] * 0.5
                if g.get("name") == "shdc": g["lr"] = lr_app_base
                if g.get("name") == "shr":  g["lr"] = lr_shr_base
                if g.get("name") == "opa":  g["lr"] = lr_opa_base
            base_lrs = [g["lr"] for g in opt.param_groups]
            print(f"[TTA] Phase2 start @ step {step}")

        with torch.no_grad():
            _, ssim_eval, _, _, _ = forward_loss_chunk(splats, [0])
        ssim_chunk = 1.0 - float(ssim_eval.item())

        cur_total = float(total.item())
        if cur_total < best_total:
            best_total = cur_total; best_step = step
            with torch.no_grad():
                best_maps = build_splats()
                _, ssim_b, l2b, ml1b, lpb = forward_loss_chunk(best_maps, [0])
            best_snapshot = dict(
                total=best_total,
                SSIM_avg=1.0 - float(ssim_b.item()),
                L2=float(l2b.item()),
                MaskL1=float(ml1b.item()),
                LPIPS=float(lpb.item()),
            )

        if ssim_chunk >= float(ssim_target):
            print(f"[TTA] Stop: SSIM(chunk) {ssim_chunk:.4f} ≥ target {ssim_target:.4f}")
            break

        if (step % LOG_EVERY) == 0:
            lrs = "/".join(f"{g['name']}={g['lr']:.2e}" for g in opt.param_groups)
            print(f"[TTA] step={step:05d} total={cur_total:.6f} | "
                  f"L2={float(l2.item()):.5f} MaskL1={float(ml1.item()):.5f} "
                  f"LPIPS={float(lp.item()):.5f} SSIM(chunk)={ssim_chunk:.6f} | "
                  f"phase2={'Y' if phase2 else 'N'} lr[{lrs}]")

        if save_every and (step % int(save_every) == 0) and step > 0:
            with torch.no_grad():
                maps_now = build_splats()
                s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = maps_now
                preds = []
                for v in range(V_all):
                    yaw_deg = float(batch["yaw_tensor"][:, v].item())
                    with torch.amp.autocast('cuda', enabled=False):
                        p = dec.render(s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac,
                                       cam_matrix.float(), cam_params, yaw_deg, dec.active_sh_degree)
                    if p.dim() == 5: p = p[:, 0]
                    preds.append(p)
                preds_bvt = torch.stack(preds, 1).unsqueeze(2)  # (1,V_all,1,3,H,W)
                gt_bvt    = batch["ground_truth_images"][:, :V_all, :1]
                gmsk_bvt  = (batch["ground_truth_mask_images"][:, :V_all, :1] > 0.5).float()

                preds_bvt_m = preds_bvt * gmsk_bvt
                gt_bvt_m    = gt_bvt * gmsk_bvt

                src_tile  = batch["source_images"][0]
                seed_text = str(batch.get("src_id", ["custom"])[0])
                view_nums = batch.get("target_view_nums", torch.arange(V_all)[None, :]).tolist()[0]
                dec.save_image(gt_bvt_m[0], preds_bvt_m[0], step,
                               timestamp=timestamp, tag=f"{save_tag}_snap_all",
                               src=src_tile, seed_text=seed_text, view_nums=view_nums)
                del preds_bvt, preds, preds_bvt_m, gt_bvt_m
        step += 1

    print(f"[TTA] BEST @ step {best_step}: total={best_snapshot['total']:.6f} | "
          f"SSIM(avg≈chunk)={best_snapshot['SSIM_avg']:.6f} | "
          f"L2={best_snapshot['L2']:.6f} MaskL1={best_snapshot['MaskL1']:.6f} "
          f"LPIPS={best_snapshot['LPIPS']:.6f}")
    
    try:
        to_log = dict(best_snapshot)
        to_log["best_step"] = best_step
        to_log["sh_degree"] = displacementDecoder.active_sh_degree
        _log_metrics_csv(seed_root_path, split_name, "opt", to_log, args.metrics_log)
    except Exception as e:
        print(f"[metrics-log][opt][warn] {e}")

    with torch.no_grad():
        if best_maps is None: best_maps = build_splats()
        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = best_maps
        preds = []
        for v in range(V_all):
            yaw_deg = float(batch["yaw_tensor"][:, v].item())
            with torch.amp.autocast('cuda', enabled=False):
                p = dec.render(s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac,
                               cam_matrix.float(), cam_params, yaw_deg, dec.active_sh_degree)
            if p.dim() == 5: p = p[:, 0]
            preds.append(p)
        preds_bvt = torch.stack(preds, 1).unsqueeze(2)  # (1, V_all, 1, 3, H, W)
        gt_bvt    = batch["ground_truth_images"][:, :V_all, :1]
        gmsk_bvt  = (batch["ground_truth_mask_images"][:, :V_all, :1] > 0.5).float()

        preds_bvt_m = preds_bvt * gmsk_bvt
        gt_bvt_m    = gt_bvt    * gmsk_bvt

        src_tile  = batch["source_images"][0]
        view_nums_all = batch.get("target_view_nums", torch.arange(V_all)[None, :]).tolist()[0]
        dec.save_image(
            gt_bvt_m[0], preds_bvt_m[0], 0,
            timestamp=timestamp, tag=f"{save_tag}_multiview_all",
            src=src_tile, seed_text=str(batch.get("src_id", ['custom'])[0]),
            view_nums=view_nums_all
        )
        if seed_root_path is not None:
            import os
            from PIL import Image
            import numpy as np

            if split_name.lower().startswith("neut"):
                out_dir = os.path.join(seed_root_path, "neutral")
            else:
                out_dir = frame_dir if frame_dir is not None else os.path.join(seed_root_path, "exp")

            os.makedirs(out_dir, exist_ok=True)

            seed_root_name = os.path.basename(seed_root_path.rstrip("/"))

            # preds_bvt: (1, V_all, 1, 3, H, W)
            V_save = preds_bvt_m.shape[1]
            for i in range(V_save):
                img = preds_bvt_m[0, i, 0].clamp(0, 1)                 # (3,H,W)
                img_u8 = (img * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H,W,3)

                if isinstance(view_nums_all, list):
                    view_id = int(view_nums_all[i])
                else:
                    view_id = i

                fname = f"view_{view_id:02d}.png"
                save_path = os.path.join(out_dir, fname)
                Image.fromarray(img_u8).save(save_path)
            print(f"[TTA] saved per-view PNGs to {out_dir}")

    if state_save_path:
        tosave = {
            "d_pos":  d_pos.detach().cpu(),
            "d_rot":  d_rot.detach().cpu(),
            "d_slog": d_slog.detach().cpu(),
            "d_shdc": d_shdc.detach().cpu(),
            "d_shr":  d_shr.detach().cpu(),
            "d_olog": d_olog.detach().cpu(),
            "code_id":  code_id.detach().cpu(),   # (B,1,Did)
            "code_app": code_app.detach().cpu(),  # (B,1,Dapp)
            "meta": {
                "H": int(H), "W": int(W),
                "active_sh_degree": int(displacementDecoder.active_sh_degree),
                "delta_down": int(loaded_meta.get("delta_down", delta_down_default)),
                "timestamp": str(timestamp),
            },
        }
        if resume_optimizer:
            tosave["opt"] = opt.state_dict()
        os.makedirs(os.path.dirname(state_save_path), exist_ok=True)
        torch.save(tosave, state_save_path)
        print(f"[TTA] saved TTA state to {state_save_path}")

    return best_snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-run", action="store_true")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--anchor-pt", type=str, default=None)
    parser.add_argument("--chunk-tag", type=str, default="")
    parser.add_argument("--seed-root", type=str, required=True, help="개별 시드 폴더 경로 (예: /data/.../seed002569_E01)")
    parser.add_argument("--skip-log", type=str, default=os.path.join(os.getcwd(), "tta_skipped_batch.log"),
                        help="스킵 로그를 누적 기록할 경로(기본: 현재 작업 디렉터리/tta_skipped_batch.log)")    
    parser.add_argument("--metrics-log", type=str, default=os.path.join(os.getcwd(), "tta_metrics_batch.csv"),
                        help="메트릭 로그 누적 경로 (기본: CWD/tta_metrics_batch.csv)")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    arc2FaceIDEncoder = Arc2FaceIDEncoder(finetune=False, ort_device_id=0).to(device).eval()
    for p in arc2FaceIDEncoder.parameters(): p.requires_grad = False

    appearanceEncoder = AppearanceEncoder(
        app_dim=getattr(hp, "app_dim", 768),
        backbone=getattr(hp, "app_backbone", "auto"),
        finetune=False,
        l2norm=getattr(hp, "app_l2norm", True),
        dropout_p=getattr(hp, "app_dropout", 0.0),
        input_size=getattr(hp, "app_input_size", 518),
        device=device,
    ).to(device).eval()

    decoder = UVDecoder2(texture_path=hp.texture_path, obj_path=hp.template_mesh_path).to(device).eval()

    resume_path = getattr(hp, "resume_path", "")
    _, global_step, best_metric, ckpt = load_checkpoint2(
        resume_path,
        arc2face=arc2FaceIDEncoder,
        appearance=appearanceEncoder,
        decoder=decoder,
        optimizer=None,
        strict=True,
        map_location=device,
    )
    if hasattr(decoder, "set_global_step"):
        decoder.set_global_step(global_step)
        print("[DEBUG] global_step =", global_step)
    
    print("[DEBUG] active_sh_degree =", decoder.active_sh_degree)

    cam_matrix, cam_params = load_cam(hp.single_cam_path)
    cam_matrix = cam_matrix.to(device).float()
    if isinstance(cam_params, dict):
        cam_params = {k: (v.to(device).float() if torch.is_tensor(v) else v) for k, v in cam_params.items()}

    # seed_root = getattr(hp, "seed_root")
    seed_root = args.seed_root
  
    neutral_dir = os.path.join(seed_root, "neutral")
    exp_dir     = os.path.join(seed_root, "exp")
    os.makedirs(neutral_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    neutral_set = CustomDataset(seed_root, sample_view=8, mode="neutral")
    neutral_loader = torch.utils.data.DataLoader(neutral_set, batch_size=1, shuffle=False)
    expr_set = CustomDataset(seed_root, sample_view=8, mode="expression")
    expr_loader = torch.utils.data.DataLoader(expr_set, batch_size=1, shuffle=False)

    lpips_fn = LPIPS(net='vgg').to(device).eval()
    fname = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.seq_run:
        has_neu = _any_pt_under(neutral_dir)
        has_exp = _any_pt_under(exp_dir)
        if has_neu and has_exp:
            _log_skip_seedroot(seed_root, reason="already_done_both_pt", split_name="both", log_path=args.skip_log)
            print(f"[SKIP] {seed_root} : neutral & exp .pt 둘 다 존재 → 전체 스킵")
            raise SystemExit(0)

        stats = run_test(
            displacementDecoder=decoder,
            arc2FaceIDEncoder=arc2FaceIDEncoder,
            appearanceEncoder=appearanceEncoder,
            lpips_fn=lpips_fn,
            cam_matrix=cam_matrix,
            cam_params=cam_params,
            loader=neutral_loader,
            max_batches=getattr(hp, "eval_max_batches", 1),
            device=device,
            save_images=True,
            timestamp=fname,
            global_step=global_step,
            split_tag="neutral",
            save_ply=False,
            ply_path=None,
            seed_root_path=seed_root,
        )
        print("[test] stats:", stats)

        run_optimize(
            displacementDecoder=decoder,
            arc2FaceIDEncoder=arc2FaceIDEncoder,
            appearanceEncoder=appearanceEncoder,
            cam_matrix=cam_matrix,
            cam_params=cam_params,
            loader=neutral_loader,
            lpips_fn=lpips_fn,
            device=device,
            max_steps=getattr(hp, "tta_max_steps", 110000),
            save_tag="tta",
            timestamp=fname,
            ssim_target=0.89,
            save_every=10000,
            warmup_steps=1500,
            cosine_min_lr=0.01,
            view_chunk=8,
            state_save_path=os.path.join(neutral_dir, f"{fname}_neutral.pt"),
            resume_optimizer=False,
            phase1_steps=3000,
            split_name="neutral",
            seed_root_path=seed_root,
            frame_dir=None, 
        )

        run_optimize(
            displacementDecoder=decoder,
            arc2FaceIDEncoder=arc2FaceIDEncoder,
            appearanceEncoder=appearanceEncoder,
            cam_matrix=cam_matrix,
            cam_params=cam_params,
            loader=expr_loader,
            lpips_fn=lpips_fn,
            device=device,
            max_steps=getattr(hp, "tta_max_steps", 110000),
            save_tag="tta",
            timestamp=fname,
            ssim_target=0.94,
            save_every=10000,
            warmup_steps=1500,
            cosine_min_lr=0.01,
            view_chunk=8,
            state_save_path=os.path.join(exp_dir, f"{fname}_exp.pt"),
            state_load_path=os.path.join(neutral_dir, f"{fname}_neutral.pt"),
            resume_optimizer=False,
            phase1_steps=5000,
            split_name="exp",
            seed_root_path=seed_root,
            frame_dir=None,  
        )
    
    else:
        if args.frame_start is None or args.frame_end is None or args.anchor_pt is None:
            raise SystemExit("--seq-run에는 --frame-start --frame-end --anchor-pt 모두 필요합니다.")

        f0, f1 = int(args.frame_start), int(args.frame_end)
        if f0 > f1:
            raise SystemExit("frame-start <= frame-end 여야 합니다.")

        prev_pt = args.anchor_pt

        for fi in range(f0, f1 + 1):
            fixed_name = f"frame_{fi:04d}.png"
            expr_set_f = CustomDataset(
                seed_root, sample_view=8, mode="expression",
                fixed_frame_name=fixed_name         
            )
            expr_loader_f = torch.utils.data.DataLoader(expr_set_f, batch_size=1, shuffle=False)

            frame_out_dir = _exp_frame_dir(seed_root, fi)

            done_pt = _latest_pt(frame_out_dir)
            if done_pt is not None:
                _log_skip_seedroot(seed_root, reason=f"frame_{fi:04d}_already_done",
                                split_name="exp", log_path=args.skip_log)
                print(f"[SKIP] frame_{fi:04d}: {done_pt}")
                prev_pt = done_pt  
                continue

            save_pt = os.path.join(frame_out_dir, f"{fname}_frame{fi:04d}.pt")
            print(f"[SEQ] frame={fi:04d} | load={prev_pt} → save={save_pt}")

            out_stats = run_optimize(
                displacementDecoder=decoder,
                arc2FaceIDEncoder=arc2FaceIDEncoder,
                appearanceEncoder=appearanceEncoder,
                cam_matrix=cam_matrix,
                cam_params=cam_params,
                loader=expr_loader_f,                 
                lpips_fn=lpips_fn,
                device=device,
                max_steps=getattr(hp, "tta_max_steps", 110000),
                save_tag=f"tta{('_' + args.chunk_tag) if args.chunk_tag else ''}",
                timestamp=fname,
                ssim_target=0.94,
                save_every=10000,
                warmup_steps=1500,
                cosine_min_lr=0.01,
                view_chunk=8,
                state_save_path=save_pt,
                state_load_path=prev_pt,             
                resume_optimizer=False,
                phase1_steps=5000,
                split_name="exp",
                seed_root_path=seed_root,
                frame_dir=frame_out_dir,              
            )

            prev_pt = save_pt