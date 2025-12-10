# trainer_ddp.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from collections import deque
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.utils import save_image, make_grid
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
from tqdm import tqdm
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
from loader.dataset_all import build_neutral_datasets

# ---------------- DDP Helpers ----------------
def ddp_is_available() -> bool:
    return torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_setup():
    if ddp_is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
        world = dist.get_world_size()
        is_main = (rank == 0)
    else:
        local_rank, rank, world = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_main = True
    return device, local_rank, rank, world, is_main

def unwrap(m):
    return m.module if hasattr(m, "module") else m

def sync_internal_device(m):
    d = next(unwrap(m).parameters()).device
    setattr(unwrap(m), "device", d)

def _tstats(t):
    if not torch.is_tensor(t): return "N/A"
    t = t.float()
    return f"shape={tuple(t.shape)} min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f} finite={bool(torch.isfinite(t).all())}"

def inspect_batch(batch, step, tag="train"):
    src   = batch["source_images"]
    drv   = batch["driving_images"]
    gt    = batch["ground_truth_images"]
    gmsk  = batch["ground_truth_mask_images"]
    yaw   = batch["yaw_tensor"]
    print(
        f"[{tag}][step={step}] "
        f"source({_tstats(src)}) | "
        f"driving({_tstats(drv)}) | "
        f"gt({_tstats(gt)}) | "
        f"gt_mask({_tstats(gmsk)}) | "
        f"yaw({_tstats(yaw)})"
    )

def save_debug_images(batch, outdir, step, max_v=2):
    os.makedirs(outdir, exist_ok=True)
    src   = batch["source_images"].detach().cpu().clamp(0,1)          # (B,3,H,W)
    drv   = batch["driving_images"].detach().cpu().clamp(0,1)         # (B,T,3,H,W)
    gt    = batch["ground_truth_images"].detach().cpu().clamp(0,1)    # (B,V,T,3,H,W)

    grid_src = make_grid(src[:4], nrow=min(4, src.shape[0]))
    save_image(grid_src, os.path.join(outdir, f"step{step:06d}_src_grid.png"))

    grid_drv = make_grid(drv[0, :min(4, drv.shape[1])], nrow=min(4, drv.shape[1]))
    save_image(grid_drv, os.path.join(outdir, f"step{step:06d}_drv_t0-3.png"))

    B, V, T = gt.shape[:3]
    tiles = [gt[0, v, 0] for v in range(min(V, max_v))]
    grid_gt = make_grid(torch.stack(tiles, 0), nrow=len(tiles))
    save_image(grid_gt, os.path.join(outdir, f"step{step:06d}_gt_v0-{len(tiles)-1}_t0.png"))

def _to_color_viz(t: torch.Tensor) -> torch.Tensor:
    """
    t: (..., C, H, W) 또는 (C, H, W)
    반환: (3, H, W) [0,1]  ← RGB
    """
    if t.dim() == 5:    # (B,T,C,H,W) -> 첫 샘플/첫 타임만
        t = t[0, 0]     # (C,H,W)
    elif t.dim() == 4:  # (T,C,H,W)
        t = t[0]        # (C,H,W)

    if t.dim() == 3:
        C, H, W = t.shape
        if C == 1:
            # 단일채널은 3채널로 복제
            out = t.repeat(3, 1, 1).float()
        elif C == 2:
            # 2채널이면 마지막에 제로 채널 추가해서 RGB 맞춤
            out = torch.cat([t, torch.zeros(1, H, W, device=t.device)], dim=0).float()
        else:
            # 3채널 이상이면 앞 3개만 사용
            out = t[:3].float()

        # 정규화 [0,1]
        mmin, mmax = float(out.min()), float(out.max())
        if mmax > mmin:
            out = (out - mmin) / (mmax - mmin)
        else:
            out = torch.zeros_like(out)
        return out.clamp(0, 1)  # (3,H,W)

    elif t.dim() == 2:
        # (H,W)면 RGB로 복제
        out = t.unsqueeze(0).repeat(3,1,1).float()
        mmin, mmax = float(out.min()), float(out.max())
        if mmax > mmin:
            out = (out - mmin) / (mmax - mmin)
        else:
            out = torch.zeros_like(out)
        return out.clamp(0,1)

    else:
        # (1,1,C)처럼 공간 없는 벡터 → None
        return None
    
def save_aux_deltas(aux: dict, outdir: str, tag: str):
    """
    aux: {"d_id": (B,T,C,H,W), "d_app": ..., "d_exp": ...}
    - 원본은 .npz로 저장
    - 요약 이미지는 채널-노름 그레이 PNG로 저장 (첫 샘플/첫 타임 기준)
    """
    os.makedirs(outdir, exist_ok=True)
    # 1) 수치 전체 저장
    try:
        pack = {}
        for k, v in aux.items():
            if torch.is_tensor(v):
                pack[k] = v.detach().cpu().to(torch.float16).numpy()
            else:
                pack[k] = v
        np.savez_compressed(os.path.join(outdir, f"{tag}_aux_deltas.npz"), **pack)
    except Exception as ex:
        print(f"[warn] save_aux_deltas npz failed: {ex}")

    # 2) 시각화
    for k, v in aux.items():
        try:
            if not torch.is_tensor(v):
                continue
            viz = _to_color_viz(v.detach().cpu())
            if viz is None:
                # 벡터만 남은 경우: 간단한 벡터 플롯 대체로 PNG 만들고 싶으면 여기서 처리
                # 혹은 텍스트로 저장:
                vec = v.flatten().detach().cpu().float().numpy()
                np.savetxt(os.path.join(outdir, f"{tag}_{k}_vector.txt"), vec, fmt="%.6f")
                continue
            # (1,H,W) -> save_image가 알아서 그레이 PNG로 저장
            save_image(viz, os.path.join(outdir, f"{tag}_{k}_mag.png"))
        except Exception as ex:
            print(f"[warn] save_aux_deltas viz failed for {k}: {ex}")


# -------------------------
# Helpers
# -------------------------
def guard_and_sanitize(step, tensors: dict, do_print_every=1000):
    bad = False
    for k, v in tensors.items():
        if not torch.isfinite(v).all():
            print(f"[WARN][{step:06d}] non-finite detected in {k} -> nan_to_num")
            tensors[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            bad = True
    if (step % do_print_every) == 0 or bad:
        for k, v in tensors.items():
            t = v.detach()
            print(f"[{step:06d}] {k} "
                  f"min={t.min().item():.5f} max={t.max().item():.5f} "
                  f"finite={torch.isfinite(t).all().item()}")
    return tensors

def _sanitize_for_raster(s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac):
    def _ff(x): return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    s_pos   = _ff(s_pos)
    s_quat  = torch.nn.functional.normalize(_ff(s_quat), dim=-1, eps=1e-6)
    s_scale = _ff(s_scale).clamp(1e-4, 1e2)
    s_shdc  = _ff(s_shdc)
    s_shr   = _ff(s_shr)
    s_opac  = _ff(s_opac).clamp(1e-6, 0.995)
    return s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac

# =========================
# Metrics helpers (PSNR / FID)
# =========================
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

def cosine_loss(x, y):
    x = F.normalize(x, dim=-1); y = F.normalize(y, dim=-1)
    return (1.0 - (x * y).sum(dim=-1)).mean()

def info_nce_streaming(emb_q, emb_k, id_labels, tau=0.07, chunk_size=256):
    q = F.normalize(emb_q, dim=-1)
    k = F.normalize(emb_k, dim=-1)
    B = q.size(0)
    device = q.device

    # 자기 자신 제외한 양성 마스크(전체 크기를 만들지 않고 행 단위로 슬라이싱해서 씁니다)
    eye = None  # 필요할 때만 만들기
    # 먼저 유효 행 판단: 배치에 같은 ID가 최소 한 개라도 있는지
    id_eq = (id_labels[:, None] == id_labels[None, :])   # (B,B) (int형일 때는 bool로 알아서 브로드캐스팅)
    if B == 1 or (id_eq.sum(dim=1) <= 1).all():
        return q.new_zeros(())  # 전부 자기 자신뿐이면 스킵

    # 1) 행별 max(안정적 logsumexp용)
    m = torch.full((B, 1), float('-inf'), device=device)
    for j in range(0, B, chunk_size):
        kb = k[j:j+chunk_size]                             # (C,D)
        block = (q @ kb.t()) / tau                         # (B,C)
        m = torch.maximum(m, block.max(dim=1, keepdim=True).values)

    # 2) 분모 누적(∑ exp(block - m))  -- 자기 자신은 제외
    denom = torch.zeros(B, 1, device=device)
    for j in range(0, B, chunk_size):
        kb = k[j:j+chunk_size]
        block = (q @ kb.t()) / tau                         # (B,C)
        if eye is None:
            eye = torch.eye(B, dtype=torch.bool, device=device)
        mask_self = eye[:, j:j+chunk_size]                 # (B,C)
        block = block.masked_fill(mask_self, float('-inf'))
        denom += torch.exp(block - m).sum(dim=1, keepdim=True)
    lse = m + torch.log(denom.clamp_min(1e-12))            # (B,1)

    # 3) 양성 로그확률 합 및 개수
    pos_sum = torch.zeros(B, device=device)
    pos_cnt = torch.zeros(B, device=device)
    for j in range(0, B, chunk_size):
        kb = k[j:j+chunk_size]
        block = (q @ kb.t()) / tau                         # (B,C)
        if eye is None:
            eye = torch.eye(B, dtype=torch.bool, device=device)
        # 같은 ID & 자기 자신 제외
        pos_mask = (id_eq[:, j:j+chunk_size]) & (~eye[:, j:j+chunk_size])  # (B,C)
        logprob_block = block - lse                        # (B,C)
        pos_sum += (logprob_block * pos_mask.float()).sum(dim=1)
        pos_cnt += pos_mask.sum(dim=1).float()

    valid = pos_cnt > 0
    if not valid.any():
        return q.new_zeros(())
    loss = -(pos_sum[valid] / pos_cnt[valid]).mean()
    return loss

def nce_diag(emb_q, emb_k, tau=0.07):
    # emb_q: y_id, emb_k: front_emb
    q = F.normalize(emb_q, dim=-1, eps=1e-6)
    k = F.normalize(emb_k, dim=-1, eps=1e-6)
    logits = q @ k.t() / tau            # (B,B)
    target = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, target)


# -------------------------
# ID embedding helpers & batch filter
# -------------------------
def get_id_embs_from_batch(batch, device, arc2face_fallback=None):
    """
    우선순위:
      1) batch['source_id_emb']      # 권장 (B,512)
      2) batch['id_embs'] / 'source_arcface_emb' / 'arcface_id_emb'  # 호환
      3) fallback: Arc2Face로 source_images에서 추출
    반환: (B,512) float tensor (device)
    """
    # 1) 권장 키
    if "source_id_emb" in batch and batch["source_id_emb"] is not None:
        return batch["source_id_emb"].to(device).float()
    # 2) 호환 키들
    for k in ("id_embs", "source_arcface_emb", "arcface_id_emb"):
        if k in batch and batch[k] is not None:
            return batch[k].to(device).float()
    # 3) 폴백
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
    """
    - batch에서 ID 임베딩을 얻고(ok 마스크 계산),
    - ok==True 인 샘플만 남긴 batch와 임베딩을 반환.
    - ok 기준: isfinite(all) & norm > eps & (옵션) batch['source_id_ok']==True
    반환: (filtered_batch or None, id_embs or None, keep_idx(LongTensor))
    """
    id_embs = get_id_embs_from_batch(batch, device, arc2face_fallback)  # (B,512)
    B = id_embs.shape[0]
    ok = torch.isfinite(id_embs).all(dim=1) & (id_embs.norm(dim=1) > eps)
    if "source_id_ok" in batch and batch["source_id_ok"] is not None:
        ok = ok & batch["source_id_ok"].to(ok.device).bool()

    keep_idx = ok.nonzero(as_tuple=False).squeeze(-1)
    n_keep = int(keep_idx.numel())

    if n_keep == 0:
        return None, None, keep_idx  # ← 이 배치는 통째로 스킵

    if n_keep == B:
        return batch, id_embs, keep_idx  # ← 그대로 사용

    # 일부만 유효 → 0번째 차원을 따라 슬라이스
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
# Eval loop (with optional image saving)
# -------------------------
@torch.no_grad()
def run_eval(displacementDecoder, arc2FaceIDEncoder, appearanceEncoder,
             lpips_fn, cam_matrix, cam_params,
             loader, max_batches=10, metric_downsample=0, sh_interval=1000, device="cuda",
             save_images=False, timestamp="none", global_step=0, split_tag="eval"):

    displacementDecoder.eval()
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

        batch_f, id_embs, keep_idx = filter_batch_by_valid_id(
            batch, device, arc2face_fallback=arc2FaceIDEncoder
        )
        if batch_f is None: 
            continue

        source   = batch_f["source_images"].to(device)
        driving  = batch_f["driving_images"].to(device)
        gt       = batch_f["ground_truth_images"].to(device)
        gt_mask  = batch_f["ground_truth_mask_images"].to(device)
        yaw_tensor = batch_f["yaw_tensor"].to(device)

        B, V, T, C, H, W = gt.shape

        code_id     = id_embs
        code_id     = code_id.unsqueeze(1).expand(-1, T, -1).contiguous()
        with torch.no_grad():
            code_app = appearanceEncoder(source)
        code_app = code_app.unsqueeze(1).expand(-1, T, -1).contiguous()


        # deform
        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = displacementDecoder(
            code_id,
            code_app
        )
        cur_sh = displacementDecoder.active_sh_degree
        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = _sanitize_for_raster(
            s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac
        )

        # render & loss
        batch_total = torch.zeros((), device=device)
        b_l1       = torch.zeros((), device=device)
        b_mask_l1  = torch.zeros((), device=device)
        b_lpips    = torch.zeros((), device=device)
        b_ssim     = torch.zeros((), device=device)
        b_psnr     = torch.zeros((), device=device)

        do_save = bool(save_images)
        preds_v = [] if do_save else None

        for v in range(V):
            yaw_b = yaw_tensor[:, v].float()   # (B,)
            preds = []
            for b in range(B):
                preds.append(unwrap(displacementDecoder).render(
                    s_pos[b:b+1], s_quat[b:b+1], s_scale[b:b+1],
                    s_shdc[b:b+1], s_shr[b:b+1], s_opac[b:b+1],
                    cam_matrix.float(), cam_params, float(yaw_b[b].item()), cur_sh
                ))
            pred = torch.cat(preds, dim=0)

            if do_save:
                preds_v.append(pred.detach())

            gt_v = gt[:, v]
            gt_mask_v = gt_mask[:, v]
            mask = (gt_mask_v > 0.5).float()

            loss_l1 = F.l1_loss(pred, gt_v)
            abs_err = torch.abs(gt_v - pred)
            face_sum = (abs_err * mask).sum()
            bg_sum   = (abs_err * (1-mask)).sum()
            faces    = mask.sum()
            bgs      = (1 - mask).sum()
            loss_face = face_sum / (faces + 1e-6)
            loss_bg   = bg_sum   / (bgs   + 1e-6)
            loss_mask_l1 = hp.w_face * loss_face + hp.w_background * loss_bg

            loss_lpips = lpips_fn(pred.flatten(0, 1) * 2 - 1, gt_v.flatten(0, 1) * 2 - 1).mean()
            loss_ssim = 1.0 - pytorch_ssim.ssim(pred.flatten(0, 1), gt_v.flatten(0, 1))

            psnr_val = compute_psnr(pred.flatten(0,1), gt_v.flatten(0,1))

            loss = loss_l1 + loss_mask_l1 + loss_lpips + loss_ssim
            batch_total += loss
            b_l1       += loss_l1
            b_mask_l1  += loss_mask_l1
            b_lpips    += loss_lpips
            b_ssim     += loss_ssim
            b_psnr     += psnr_val

            # FID 수집
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
            preds_bvt = torch.stack(preds_v, dim=1)   # (B, V_succ, T, 3, H, W)
            Bp, Vp = preds_bvt.shape[0], preds_bvt.shape[1]
            Vg, Tg = gt.shape[1], gt.shape[2]
            Vc = min(Vp, Vg)
            Tc = min(preds_bvt.shape[2], Tg)

            if bi == 0:
                print(f"[eval/save] shapes gt(B,V,T,3,H,W)={tuple(gt.shape)} "
                      f"preds_bvt(B,V,T,3,H,W)={tuple(preds_bvt.shape)} -> Vc={Vc}, Tc={Tc}")

            if Vc > 0 and Tc > 0:
                for b in range(Bp):
                    # --- src 구성: (B,3,H,W) 또는 (B,T,3,H,W) -> (V,T,3,H,W)
                    seed_text = str(batch_f["src_id"][b])
                    view_nums = batch_f["target_view_nums"][b].tolist()
                    src_tile = None
                    if isinstance(source, torch.Tensor):
                        if source.dim() == 4:   # (B,3,H,W)
                            src_tile = source[b]
                        elif source.dim() == 5: # (B,T,3,H,W)
                            src_tile = source[b, 0]   # 첫 타임만 대표로

                    displacementDecoder.save_image(
                        gt[b,      :Vc, :Tc],
                        preds_bvt[b, :Vc, :Tc],
                        global_step,
                        src=src_tile,
                        timestamp=timestamp,
                        tag=f"eval_{split_tag}_b{bi:03d}_s{b:02d}",
                        seed_text=seed_text,                   # ← 전체 좌상단에 1회 표기
                        view_nums=view_nums           # ← 각 뷰 행 좌상단에 view=xx 표기
                    )
            del preds_bvt, preds_v

        Vt = float(V)
        total_l1      += (b_l1 / Vt).item()
        total_mask_l1 += (b_mask_l1 / Vt).item()
        total_lpips   += (b_lpips / Vt).item()
        total_ssim    += (b_ssim / Vt).item()
        total_loss    += (batch_total / Vt).item()
        total_psnr    += (b_psnr / Vt).item()
        n_seen += 1

    if n_seen == 0:
        return {}

    if len(feats_fake_list) >= 1 and len(feats_real_list) >= 1:
        feats_fake = torch.cat(feats_fake_list, dim=0)
        feats_real = torch.cat(feats_real_list, dim=0)
        fid_val = compute_fid_from_features(feats_real, feats_fake) if (feats_fake.shape[0] >= 32 and feats_real.shape[0] >= 32) else float("nan")
    else:
        fid_val = float("nan")

    return {
        "l1":        total_l1 / n_seen,
        "maskl1":    total_mask_l1 / n_seen,
        "lpips":     total_lpips / n_seen,
        "ssim":      total_ssim / n_seen,
        "psnr":      total_psnr / n_seen,
        "fid":       fid_val,
        "loss":      total_loss / n_seen,
        "batches":   n_seen,
    }

# -------------------------
# Training
# -------------------------
def run_train():
    import rich.traceback
    rich.traceback.install()

    # ---- DDP setup ----
    device, local_rank, rank, world_size, is_main = ddp_setup()
    torch.backends.cudnn.benchmark = True

    now = datetime.now()
    fname = now.strftime("%Y%m%d_%H%M%S")

    # iters
    stage1_iters = int(getattr(hp, "stage1_iters", 10000))
    max_iters    = stage1_iters

    # ---------------- WandB (옵션; rank0만) ----------------
    use_wandb = bool(getattr(hp, "use_wandb", True)) and is_main
    if use_wandb:
        import wandb
        # wandb.login(key="7a520b2a0871a3ecf4a95c9f834313ea6c62e937", relogin=False) #NHH
        wandb.login(key="d627f98f023bdd7e5f22df3bb60ff5d58de71ee2", relogin=False) #LCM
        wandb.init(project=hp.simple_name, name=fname)
        wandb.define_metric("global_step")
        wandb.define_metric("train/*",  step_metric="global_step")
        wandb.define_metric("eval/*",   step_metric="global_step")
        wandb.define_metric("stat/*",   step_metric="global_step")
        wandb.define_metric("data/*", step_metric="global_step")

    if is_main:
        print(f"== MonoFace Training :: {fname} ==")
        if ddp_is_available():
            print(f"[DDP] world_size={world_size} rank={rank} local_rank={local_rank}")

    metric_downsample = int(getattr(hp, "metric_downsample", 512))
    eval_interval     = int(getattr(hp, "eval_interval", 1000))
    preview_interval  = int(getattr(hp, "preview_interval", 1000))
    preview_burst     = int(getattr(hp, "preview_burst", 1))

    lam_geo_inv       = float(getattr(hp, "lam_geo_inv", 1.0))
    lam_id_app_ortho  = float(getattr(hp, "lam_id_app_ortho", 0.1))
    lam_app_inv       = float(getattr(hp, "lam_app_inv", 0.0))

    # ---------------- Models ----------------
    id_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    arc2FaceIDEncoder = Arc2FaceIDEncoder(finetune=False, ort_device_id=local_rank).to(f"cuda:{local_rank}").eval()
    for p in arc2FaceIDEncoder.parameters():
        p.requires_grad = False

    appearanceEncoder = AppearanceEncoder(    
        app_dim=getattr(hp, "app_dim", 768),
        backbone=getattr(hp, "app_backbone", "auto"),
        finetune=False,                 
        l2norm=getattr(hp, "app_l2norm", True),
        dropout_p=getattr(hp, "app_dropout", 0.0),
        input_size=getattr(hp, "app_input_size", 224),
        device=device,
    ).to(device).eval()
    sync_internal_device(appearanceEncoder)
    decoder = UVDecoder2(texture_path=hp.texture_path, obj_path=hp.template_mesh_path).to(device)

    base_sh_rest_params, other_params = [], []
    for name, p in decoder.named_parameters():
        if name == "img_sh_rest":
            base_sh_rest_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.Adam([
        {'params': other_params,                  'lr': hp.lr['dd'],    'weight_decay': 0.0},
        {'params': base_sh_rest_params,           'lr': hp.lr['dd'] * 0.05},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=("cuda" in device))

    # ---------------- Camera ----------------
    if is_main:
        print("Loading camera params...")
    cam_matrix, cam_params = load_cam(hp.single_cam_path)
    cam_matrix = cam_matrix.to(device).float()
    if isinstance(cam_params, dict):
        cam_params = {k: (v.to(device).float() if torch.is_tensor(v) else v) for k, v in cam_params.items()}

    # ---------------- Datasets / Dataloaders ----------------
    neutral_sets = build_neutral_datasets(hp.dataset_root)

    num_workers = int(getattr(hp, "num_workers", 0))
    pin_mem = bool(getattr(hp, "pin_memory", True))
    persistent = (num_workers > 0)

    # 분산 샘플러 (train만)
    neutral_sampler = DistributedSampler(neutral_sets["train"], shuffle=True, drop_last=True) if ddp_is_available() else None

    neutral_loader = DataLoader(
        neutral_sets["train"],
        batch_size=hp.batch_size,
        shuffle=(neutral_sampler is None),
        sampler=neutral_sampler,
        num_workers=num_workers, persistent_workers=persistent,
        pin_memory=pin_mem, drop_last=True,
    )
    neutral_val_loader = DataLoader(
        neutral_sets["val"],
        batch_size=getattr(hp, "eval_batch", 1), shuffle=False,
        num_workers=num_workers, persistent_workers=persistent,
        pin_memory=True, drop_last=False,
    )

    # ---------------- After building loaders & before training loop ----------------
    neutral_iter = iter(neutral_loader)
    def next_batch(global_step):
        nonlocal neutral_iter
        if global_step < stage1_iters:
            try:
                return next(neutral_iter), 1
            except StopIteration:
                if neutral_sampler is not None:
                    neutral_sampler.set_epoch(global_step)
                neutral_iter = iter(neutral_loader)
                return next(neutral_iter), 1
        else:
            pass

    # ---------------- Loss / Metrics ----------------
    lpips_fn = LPIPS(net='vgg').to(device).eval()
    

    # ---------------- Save/Resume ----------------
    os.makedirs(hp.save_path, exist_ok=True)
    save_interval   = int(getattr(hp, "save_interval", 10000))
    recent_ckpts    = deque()
    best_metric     = float("inf")

    global_step = 0
    ckpt = {}
    resume_path = getattr(hp, "resume_path", "")
    if resume_path and os.path.isfile(resume_path):
        # 래핑 전 로드
        _, global_step, best_metric, ckpt = load_checkpoint2(
            resume_path,
            arc2FaceIDEncoder,
            appearanceEncoder,
            decoder,
            optimizer=optimizer,
            strict=True,
            map_location=device
        )
        if is_main:
            print(f"[Resume] step={global_step}, best={best_metric}")

    if "decoder_active_sh_degree" in ckpt:
        unwrap(decoder).active_sh_degree = int(ckpt["decoder_active_sh_degree"])
    else:
        unwrap(decoder).active_sh_degree = int(
            min(unwrap(decoder).max_sh_degree, global_step // getattr(hp, "sh_interval", 1000))
        )

    # ---- DDP wrap (학습되는 모델만 래핑) ----
    if ddp_is_available():
        decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # ---------------- Train loop ----------------
    if is_main:
        print("-- Stage1(Neutral) -> Stage2(Expressive) Training Start --")
    pbar = tqdm(total=max_iters, initial=global_step, dynamic_ncols=True) if is_main else None

    while global_step < max_iters:
        batch, stage_now = next_batch(global_step)

        dec = unwrap(decoder)
        dec.train()
        dec.train_stage1_id()
        
        # ------- unpack -------
        source    = batch["source_images"].to(device)
        driving   = batch["driving_images"].to(device)
        gt        = batch["ground_truth_images"].to(device)
        gt_mask   = batch["ground_truth_mask_images"].to(device)
        yaw_tensor = batch["yaw_tensor"].to(device)

        # ===== ID 실패 샘플 필터링 =====
        batch_f, id_embs, keep_idx = filter_batch_by_valid_id(
            {
                "source_images": source,
                "driving_images": driving,
                "ground_truth_images": gt,
                "ground_truth_mask_images": gt_mask,
                "yaw_tensor": yaw_tensor,
                "source_id_emb": batch.get("source_id_emb", None),
                "source_id_ok":  batch.get("source_id_ok",  None),
                "frontal_id_emb": batch.get("frontal_id_emb", None),
                "identity_index": batch.get("identity_index", None),
                "id_source_rel": batch.get("id_source_rel", None),
                "src_id": batch.get("src_id", None),
                "target_view_nums": batch.get("target_view_nums", None),
            },
            device,
            arc2face_fallback=arc2FaceIDEncoder
        )
        if batch_f is None:
            if is_main:
                print(f"[skip] all samples in batch had invalid ID embeddings at step {global_step}")
            continue

        source     = batch_f["source_images"]
        driving    = batch_f["driving_images"]
        gt         = batch_f["ground_truth_images"]
        gt_mask    = batch_f["ground_truth_mask_images"]
        yaw_tensor = batch_f["yaw_tensor"]
        front_emb  = batch_f["frontal_id_emb"].to(device).float()     # (B,D)
        id_labels  = batch_f["identity_index"].to(device).long()      # (B,)

        # ---- 소스 뷰 통계 (view_00만 쓰는지 확인) ----
        src_view_idx = batch_f.get("src_view_idx", None)  # (B,) long or None
        if src_view_idx is not None:
            if torch.is_tensor(src_view_idx):
                view_ids = src_view_idx.detach().cpu().tolist()
            else:
                view_ids = list(map(int, src_view_idx))  # 안전장치
        else:
            # 폴백: 문자열 경로에서 view_?? 파싱
            src_rel_list = batch_f.get("id_source_rel", None)  # list[str]
            view_ids = []
            if src_rel_list is not None:
                import re
                pat = re.compile(r'view[_-]?(\d{2})', re.IGNORECASE)
                for s in src_rel_list:
                    m = pat.search(str(s))
                    view_ids.append(int(m.group(1)) if m else -1)

        # 통계/로깅
        if view_ids:
            front_frac = sum(1 for v in view_ids if v == 0) / max(1, len(view_ids))
            if use_wandb:
                import wandb
                wandb.log({
                    "data/src_view_front_frac": front_frac,
                    "data/src_view_hist": wandb.Histogram(view_ids),
                    "global_step": int(global_step),
                })

        B, V, T, C, H, W = gt.shape

        # ------- forward -------
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            code_app = appearanceEncoder(source)

        y_id = id_embs
        id_app_cos = F.cosine_similarity(
            F.normalize(y_id, dim=-1),
            F.normalize(code_app, dim=-1), dim=-1
        )
        L_id_app_ortho = id_app_cos.abs().mean()

        if lam_app_inv > 0:
            src2 = torch.flip(source, dims=[-1])
            with torch.no_grad():
                code_app_2 = appearanceEncoder(src2)
            L_app_inv = 1.0 - F.cosine_similarity(
                F.normalize(code_app,  dim=-1, eps=1e-6),
                F.normalize(code_app_2,dim=-1, eps=1e-6),
                dim=-1
            ).mean()
        else:
            L_app_inv = torch.zeros((), device=device)

        code_id  = y_id.unsqueeze(1).expand(-1, T, -1).contiguous()
        code_app = code_app.unsqueeze(1).expand(-1, T, -1).contiguous()

        if hasattr(unwrap(decoder), "set_global_step"):
            unwrap(decoder).set_global_step(global_step)

        # 1) 디코더는 배치당 1번만 forward
        with torch.amp.autocast('cuda', enabled=("cuda" in device)):
            s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = unwrap(decoder)(code_id, code_app, return_aux=False)[:6]
            # app shuffle로 geo-invariance
            out_s = unwrap(decoder)(code_id, code_app[torch.randperm(code_app.shape[0], device=code_app.device)], return_aux=False)
            s_pos_s, s_quat_s, s_scale_s = out_s[0], out_s[1], out_s[2]

        L_geo_inv = (
            F.l1_loss(s_pos_s,  s_pos.detach()) +
            F.l1_loss(s_quat_s, s_quat.detach()) +
            F.l1_loss(s_scale_s, s_scale.detach())
        )

        # 2) 렌더에 쓸 파라미터 sanitize (한 번만)
        with torch.amp.autocast('cuda', enabled=False):
            s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = _sanitize_for_raster(
                s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac
            )
        cur_sh = unwrap(decoder).active_sh_degree

        # 3) 뷰 서브샘플링 (기본 8)
        V = gt.shape[1]
        V_sel = int(getattr(hp, "lpips_views", min(8, V)))
        V_sel = max(1, min(V, V_sel))
        v_indices = list(range(V)) if V_sel >= V else torch.randperm(V, device=device)[:V_sel].tolist()

        acc_l1      = torch.zeros((), device=device)
        acc_maskl1  = torch.zeros((), device=device)
        acc_lpips   = torch.zeros((), device=device)
        acc_ssim    = torch.zeros((), device=device)

        do_preview = (global_step % preview_interval) < preview_burst
        burst_idx = global_step % preview_interval
        vis_preds = [] if (do_preview and is_main) else None
        
        for v in v_indices:
            with torch.amp.autocast('cuda', enabled=False):
                yaw_b = yaw_tensor[:, v].float()   # (B,)
                preds = []
                for b in range(B):
                    preds.append(unwrap(decoder).render(
                        s_pos[b:b+1], s_quat[b:b+1], s_scale[b:b+1],
                        s_shdc[b:b+1], s_shr[b:b+1], s_opac[b:b+1],
                        cam_matrix.float(), cam_params, float(yaw_b[b].item()), cur_sh
                    ))
                pred = torch.cat(preds, dim=0)     # (B,T,3,H,W)

            gt_v      = gt[:, v]
            gt_mask_v = gt_mask[:, v]
            mask = (gt_mask_v > 0.5).float()

            # L1 & Mask L1
            loss_l1 = F.l1_loss(pred, gt_v)
            abs_err = torch.abs(gt_v - pred)
            face_sum = (abs_err * mask).sum()
            bg_sum   = (abs_err * (1-mask)).sum()
            faces    = mask.sum()
            bgs      = (1 - mask).sum()
            loss_face = face_sum / (faces + 1e-6)
            loss_bg   = bg_sum   / (bgs   + 1e-6)
            loss_mask_l1 = hp.w_face * loss_face + hp.w_background * loss_bg

            loss_lpips = lpips_fn(pred.flatten(0, 1) * 2 - 1, gt_v.flatten(0, 1) * 2 - 1).mean()
            loss_ssim = 1.0 - pytorch_ssim.ssim(pred.flatten(0, 1), gt_v.flatten(0, 1))

            # 누적
            acc_l1      += loss_l1
            acc_maskl1  += loss_mask_l1
            acc_lpips   += loss_lpips
            acc_ssim    += loss_ssim

            if vis_preds is not None:
                vis_preds.append(pred.detach())

            del pred, gt_v, gt_mask_v, mask, abs_err, loss_l1, loss_mask_l1, loss_lpips, loss_ssim
            torch.cuda.empty_cache()

        # 5) 선택된 뷰 평균(≒ 전체 평균의 unbiased estimator) + 정규화 항
        mean_l1      = acc_l1     / float(V_sel)
        mean_maskl1  = acc_maskl1 / float(V_sel)
        mean_lpips   = acc_lpips  / float(V_sel)
        mean_ssim    = acc_ssim   / float(V_sel)

        total_loss = (mean_l1 + mean_maskl1 + mean_lpips + mean_ssim
                      + lam_id_app_ortho * L_id_app_ortho
                      + lam_geo_inv     * L_geo_inv
                      + lam_app_inv     * L_app_inv)

        # 6) 단 한 번 backward/step
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 로깅용 값
        total_l1      = mean_l1.detach()
        total_mask_l1 = mean_maskl1.detach()
        total_lpips   = mean_lpips.detach()
        total_ssim    = mean_ssim.detach()

        global_step += 1
        
        if pbar is not None:
            pbar.update(1)
        
        if (global_step % hp.sh_interval == 0):
            unwrap(decoder).oneupSHdegree()
            
        if use_wandb:
            wandb.log({
                "train/loss": float(total_loss.item()),
                "train/l1": float(total_l1.item()),
                "train/maskl1": float(total_mask_l1.item()),
                "train/lpips": float(total_lpips.item()),
                "train/ssim": float(total_ssim.item()),
                "train/L_id_app_ortho": float(L_id_app_ortho.item()),
                "train/lam_id_app_ortho": float(lam_id_app_ortho),
                "train/L_geo_inv": float(L_geo_inv.item()),
                "train/lam_geo_inv": float(lam_geo_inv),
                "train/L_app_inv": float(L_app_inv.item()),
                "train/lam_app_inv": float(lam_app_inv),
                "train/stage": 1 if stage_now == 1 else 2,
                "global_step": int(global_step),
            })
        
        # --- preview (rank0) ---
        if vis_preds is not None and len(vis_preds) > 0 and is_main:
            preds_bvt = torch.stack(vis_preds, dim=1)  # (B, V_sel, T, 3, H, W)

            Bp = min(preds_bvt.shape[0], 2)
            Vc = preds_bvt.shape[1]
            Tc = preds_bvt.shape[2]

            # v_indices: 선택된 뷰 인덱스 (list[int])
            v_idx = v_indices[:Vc]

            for b in range(Bp):
                # GT도 같은 순서로 인덱싱!
                gt_sel = gt[b, v_idx, :Tc].detach().cpu()          # (Vc,Tc,3,H,W)
                pred_sel = preds_bvt[b, :Vc, :Tc].detach().cpu()   # (Vc,Tc,3,H,W)

                # 라벨도 같은 순서로 맞추기
                full_view_nums = batch_f["target_view_nums"][b].tolist()  # 길이 V
                view_nums_sel = [full_view_nums[i] for i in v_idx]

                # src 타일
                src_tile = None
                if isinstance(source, torch.Tensor):
                    src_tile = source[b] if source.dim()==4 else source[b,0]

                seed_text = str(batch_f["src_id"][b])

                unwrap(decoder).save_image(
                    gt_sel, pred_sel, global_step,
                    timestamp=fname,
                    tag=f"burst{(global_step % preview_interval):02d}_b{b:02d}",
                    src=src_tile.detach().cpu() if torch.is_tensor(src_tile) else None,
                    seed_text=seed_text,
                    view_nums=view_nums_sel
                )

            del preds_bvt, vis_preds

        # ckpt (rank0)
        if ((global_step % save_interval) == 0 or global_step == max_iters) and is_main:
            tag = f"iter{global_step:06d}"
            path = save_checkpoint2(
                epoch=global_step,
                global_step=global_step,
                save_dir=hp.save_path,
                tag=tag,
                arc2face=arc2FaceIDEncoder,
                appearance=appearanceEncoder,
                decoder=unwrap(decoder),
                optimizer=optimizer,
                best_metric=best_metric,
                fname=fname,
            )
            recent_ckpts.append(path)

        # eval (rank0)
        if ((global_step % eval_interval) == 0 or global_step == max_iters) and is_main:
            which = "neutral"
            loader = neutral_val_loader
            stats = run_eval(
                unwrap(decoder), arc2FaceIDEncoder, appearanceEncoder, 
                lpips_fn, cam_matrix, cam_params, loader,
                max_batches=getattr(hp, "eval_max_batches", 10),
                metric_downsample=metric_downsample,
                device=device,
                save_images=True,
                timestamp=fname,
                global_step=global_step,
                split_tag=which
            )
            if stats:
                print(f"[eval/{which}] step={global_step} :: {stats}")
                if use_wandb:
                    log_eval = {f"eval/{which}/{k}": v for k, v in stats.items()}
                    log_eval["global_step"] = int(global_step)
                    wandb.log(log_eval)
            cur_metric = stats["loss"] if stats else float("inf")
            if stats and (cur_metric < best_metric):
                best_metric = cur_metric
                save_checkpoint2(
                    epoch=global_step,
                    global_step=global_step,
                    save_dir=hp.save_path,
                    tag=f"best_{which}",
                    arc2face=arc2FaceIDEncoder,
                    appearance=appearanceEncoder,
                    decoder=unwrap(decoder),
                    optimizer=optimizer,
                    best_metric=best_metric,
                    fname=fname,
                )

    # last save (rank0)
    if is_main:
        save_checkpoint2(
            epoch=global_step,
            global_step=global_step,
            save_dir=hp.save_path,
            tag="last",
            arc2face=arc2FaceIDEncoder,
            appearance=appearanceEncoder,
            decoder=unwrap(decoder),
            optimizer=optimizer,
            best_metric=best_metric,
            fname=fname,
        )
        print(f"[Saved] last checkpoint at {hp.save_path}")

    # 종료 정리
    if ddp_is_available():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_train()
