import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import convert_image_dtype

# torch.backends.cuda.matmul.allow_tf32 = True # NHH Revise
# torch.backends.cudnn.allow_tf32 = True # NHH Revise
# torch.set_float32_matmul_precision("high") # NHH Revise

import os
import json
import math
import numpy as np
import imageio
from PIL import Image
from datetime import datetime
from kaolin.ops.mesh import sample_points

from utils import load_obj_as_mesh, save_pointcloud_image_offscreen, convert_c2w_to_w2c, load_cam
from utils.sh_utils import RGB2SH, SH2RGB
from renderer import render_gaussian
from mf_modules.visualize_camera import visualize_camera_and_points
from mf_modules.img2img import ConditionalUNetDelta
import hparams as hp

class CrossAttnDisplacementTransformer(nn.Module):
    def __init__(self, code_dim=(hp.id_dim+hp.exp_dim), hidden_dim=hp.hidden_dim, emb_dim=hp.emb_dim,
                 n_heads=8, n_layers=6, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len

        # 1) 분기형 프로젝션 + 정규화
        self.id_proj  = nn.Linear(hp.id_dim, hidden_dim)
        self.exp_proj = nn.Linear(hp.exp_dim, hidden_dim)
        self.id_ln  = nn.LayerNorm(hidden_dim)
        self.exp_ln = nn.LayerNorm(hidden_dim)

        # 선택: 게이팅(학습 가능한 스칼라)
        self.id_gate  = nn.Parameter(torch.tensor(1.0))
        self.exp_gate = nn.Parameter(torch.tensor(1.0))

        # 디코더 쪽
        self.seq_proj   = nn.Linear(emb_dim, hidden_dim)
        self.output_proj= nn.Linear(hidden_dim, emb_dim)

        # self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len + 1, hidden_dim))
        self.register_buffer("pos_enc", torch.randn(1, max_seq_len + 1, hidden_dim)) # NHH Revise

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

    def forward(self, identity, expression_seq, inference_steps):
        B, T = expression_seq.shape[0], expression_seq.shape[1]
        assert inference_steps == T

        identity_token = identity.unsqueeze(1).expand(-1, T, -1)

        # (그대로) 분기 투영 + 정규화
        id_h  = self.id_ln(self.id_proj(identity_token))     # (B,T,H)
        exp_h = self.exp_ln(self.exp_proj(expression_seq))   # (B,T,H)

        encoder_input = self.id_gate * id_h + self.exp_gate * exp_h
        encoder_input = encoder_input + self.pos_enc[:, :T, :]
        memory = self.encoder(encoder_input)                 # (B,T,H)

        preds = []
        prev_disp = expression_seq.new_zeros(B, 1, self.emb_dim)
        for t in range(inference_steps):
            # ★ 쿼리에 현재 프레임의 exp 임베딩(exp_h[:, t])을 직접 주입
            q = self.seq_proj(prev_disp) + exp_h[:, t:t+1, :] + self.pos_enc[:, t+1:t+2, :]

            # ★ 메모리도 현재 시점 t만 보도록 슬라이스 (시간 정렬 강제)
            mem_t = memory[:, t:t+1, :]

            dec_out = self.decoder(q, memory=mem_t)
            pred_disp = self.output_proj(dec_out)            # (B,1,emb_dim)
            preds.append(pred_disp)
            prev_disp = pred_disp

        return torch.cat(preds, dim=1)  # (B,T,emb_dim)
        
class UVDecoder(nn.Module):
    """
    Builds learnable UV-space base maps from a texture and an OBJ (with UVs),
    predicts delta maps conditioned on a code, and outputs final maps as base + delta.

    Final channel layout per pixel:
        pos(3), quat(4), scale(3), sh_dc(3), sh_rest(3*(max_coeffs-1)), opacity(1)
    """
    def __init__(self, texture_path: str, obj_path: str,
                 code_dim=hp.emb_dim, num_points=hp.num_points,
                 init_sh_degree=0, max_sh_degree=3):
        super().__init__()

        # Texture (float32 in [0,1])
        tex = convert_image_dtype(read_image(texture_path, mode=ImageReadMode.RGB), dtype=torch.float32)
        self.tex = tex
        self.H, self.W = int(tex.shape[1]), int(tex.shape[2])
        id_dim = getattr(hp, "id_dim")
        exp_dim = getattr(hp, "exp_dim")
        app_dim = getattr(hp, "app_dim")
        if app_dim is None:
            app_dim = int(getattr(hp, "app_dim", 768))
        self.app_dim = app_dim

        # Minimal OBJ parser and UV rasterizer (raw XYZ → UV position map)
        def _parse_obj(path: str):
            V, VT, F_v, F_vt = [], [], [], []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line or line.startswith("#"): continue
                    t = line.strip().split()
                    if not t: continue
                    if t[0] == "v" and len(t) >= 4:
                        V.append([float(t[1]), float(t[2]), float(t[3])])
                    elif t[0] == "vt" and len(t) >= 3:
                        VT.append([float(t[1]), float(t[2])])
                    elif t[0] == "f" and len(t) >= 4:
                        def parse_face_token(tok):
                            parts = tok.split("/")
                            v_idx = int(parts[0]) if parts[0] else 0
                            vt_idx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                            return v_idx, vt_idx
                        vts = [parse_face_token(tok) for tok in t[1:]]
                        for i in range(1, len(vts) - 1):
                            a, b, c = vts[0], vts[i], vts[i+1]
                            F_v.append([a[0]-1, b[0]-1, c[0]-1])
                            F_vt.append([a[1]-1, b[1]-1, c[1]-1])
            V = np.asarray(V, dtype=np.float32)
            VT = np.asarray(VT, dtype=np.float32)
            F_v = np.asarray(F_v, dtype=np.int32)
            F_vt = np.asarray(F_vt, dtype=np.int32)
            if VT.size == 0 or (F_vt < 0).any():
                raise ValueError("OBJ missing UVs (vt) or face UV indices.")
            return V, F_v, VT, F_vt

        def _rasterize_uv_position(verts_raw, faces, uvs01, faces_uv, H, W):
            img = np.zeros((H, W, 3), dtype=np.float32)
            zbuf = np.full((H, W), -np.inf, dtype=np.float32)
            UV = uvs01.copy()
            UV[:, 0] = UV[:, 0] * (W - 1)
            UV[:, 1] = (1.0 - UV[:, 1]) * (H - 1)
            F = faces.shape[0]
            for f in range(F):
                vi = faces[f]; ti = faces_uv[f]
                if (ti < 0).any(): continue
                tri_uv = UV[ti]; tri_p = verts_raw[vi]
                xmin = max(int(np.floor(tri_uv[:,0].min())), 0)
                xmax = min(int(np.ceil (tri_uv[:,0].max())), W-1)
                ymin = max(int(np.floor(tri_uv[:,1].min())), 0)
                ymax = min(int(np.ceil (tri_uv[:,1].max())), H-1)
                if xmax < xmin or ymax < ymin: continue
                p0, p1, p2 = tri_uv[0], tri_uv[1], tri_uv[2]
                area = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])
                if abs(area) < 1e-8: continue
                for y in range(ymin, ymax+1):
                    py = y + 0.5
                    for x in range(xmin, xmax+1):
                        px = x + 0.5
                        w0 = ((p1[0]-px)*(p2[1]-py) - (p2[0]-px)*(p1[1]-py)) / area
                        w1 = ((p2[0]-px)*(p0[1]-py) - (p0[0]-px)*(p2[1]-py)) / area
                        w2 = 1.0 - w0 - w1
                        if (w0 >= -1e-6) and (w1 >= -1e-6) and (w2 >= -1e-6):
                            p = w0*tri_p[0] + w1*tri_p[1] + w2*tri_p[2]
                            z = p[2]
                            if z >= zbuf[y, x]:
                                img[y, x, :] = p
                                zbuf[y, x] = z
            return img

        def _rasterize_uv_position_fast(verts_raw, faces, uvs01, faces_uv, H, W):
            import numpy as np
            img  = np.ones((H, W, 3), dtype=np.float32)/2.0
            zbuf = np.full((H, W), -np.inf, dtype=np.float32)

            UV = uvs01.copy().astype(np.float32)
            UV[:, 0] = UV[:, 0] * (W - 1)
            UV[:, 1] = (1.0 - UV[:, 1]) * (H - 1)

            F = faces.shape[0]
            for f in range(F):
                vi = faces[f]; ti = faces_uv[f]
                if (ti < 0).any(): 
                    continue

                tri_uv = UV[ti]            # (3,2)
                tri_p  = verts_raw[vi]     # (3,3)

                xmin = max(int(np.floor(tri_uv[:,0].min())), 0)
                xmax = min(int(np.ceil (tri_uv[:,0].max())), W-1)
                ymin = max(int(np.floor(tri_uv[:,1].min())), 0)
                ymax = min(int(np.ceil (tri_uv[:,1].max())), H-1)
                if xmax < xmin or ymax < ymin: 
                    continue

                p0, p1, p2 = tri_uv[0], tri_uv[1], tri_uv[2]
                area = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])
                if abs(area) < 1e-8:
                    continue

                xs = np.arange(xmin, xmax+1, dtype=np.float32)
                ys = np.arange(ymin, ymax+1, dtype=np.float32)
                xx, yy = np.meshgrid(xs + 0.5, ys + 0.5)  # (h,w)

                w0 = ((p1[0]-xx)*(p2[1]-yy) - (p2[0]-xx)*(p1[1]-yy)) / area
                w1 = ((p2[0]-xx)*(p0[1]-yy) - (p0[0]-xx)*(p2[1]-yy)) / area
                w2 = 1.0 - w0 - w1
                mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
                if not mask.any():
                    continue

                p_interp = (w0[...,None]*tri_p[0] + w1[...,None]*tri_p[1] + w2[...,None]*tri_p[2])  # (h,w,3)
                z = p_interp[..., 2]

                ys_idx, xs_idx = np.where(mask)
                yi = (ys_idx + ymin).astype(np.int32)
                xi = (xs_idx + xmin).astype(np.int32)

                z_old = zbuf[yi, xi]
                z_new = z[ys_idx, xs_idx]
                keep  = z_new >= z_old
                if not np.any(keep):
                    continue

                yi = yi[keep]; xi = xi[keep]
                vals = p_interp[ys_idx[keep], xs_idx[keep], :]
                img[yi, xi, :]  = vals
                zbuf[yi, xi]    = z_new[keep]
            
            valid = (zbuf > -np.inf).astype(np.float32)
            return img, valid

        # Bake UV position map
        verts_np, faces_np, uvs_np, faces_uv_np = _parse_obj(obj_path)
        verts = torch.from_numpy(verts_np).float()
        faces = torch.from_numpy(faces_np.astype(np.int64))
        uvs   = torch.from_numpy(uvs_np).float()
        faces_uv = torch.from_numpy(faces_uv_np.astype(np.int64))

        # Optional FLAME-style translation
        flame_trans = -verts.mean(dim=0)
        verts_trans = verts + flame_trans

        pitch0_deg = float(getattr(hp, "init_pitch_deg", 8.0)) 
        theta = torch.deg2rad(torch.tensor(pitch0_deg))
        Rx = torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta), -torch.sin(theta)],
                        [0, torch.sin(theta),  torch.cos(theta)]], dtype=torch.float32)
        verts_trans = (verts_trans @ Rx.T)
        pos_map_np, valid_np = _rasterize_uv_position_fast(
            verts_trans.cpu().numpy(), faces.cpu().numpy(),
            uvs.cpu().numpy(), faces_uv.cpu().numpy(),
            self.H, self.W
        )
        pos_map = torch.from_numpy(pos_map_np).permute(2, 0, 1).contiguous()  # (3,H,W)
        uv_valid = torch.from_numpy(valid_np).unsqueeze(0).contiguous()
        self.register_buffer("uv_valid", uv_valid)

        # SH configuration
        self.num_points = int(num_points)
        self.active_sh_degree = int(init_sh_degree)
        self.max_sh_degree = int(max_sh_degree)
        self.max_coeffs = (self.max_sh_degree + 1) ** 2
        self.max_rest_dim = self.max_coeffs - 1  # exclude DC

        self.register_parameter("img_position", nn.Parameter(pos_map))
        quat0 = torch.zeros(4, self.H, self.W); quat0[3].fill_(1.0)
        self.register_parameter("img_quat", nn.Parameter(quat0))
        self.s_min = float(getattr(hp, "scale_min", 2e-4))
        self.s_max = float(getattr(hp, "scale_max", 0.004))
        init_scale = float(getattr(hp, "init_scale", 6e-4))      # [s_min, s_max] 내부

        p_scale = (init_scale - self.s_min) / (self.s_max - self.s_min)  # 0~1로 정규화
        scale_logit0 = _logit(p_scale)
        self.register_parameter(
            "img_scale",
            nn.Parameter(torch.full((3, self.H, self.W), float(scale_logit0)))
        )
        self.register_parameter("img_sh_dc", nn.Parameter(tex))
        self.register_parameter("img_sh_rest", nn.Parameter(torch.zeros(3 * self.max_rest_dim, self.H, self.W)))
        init_opa = float(getattr(hp, "init_opacity", 0.20))
        opa_logit0 = _logit(init_opa)
        self.register_parameter("img_opacity", nn.Parameter(opa_logit0 * uv_valid))  

        # Delta U-Net (in/out channels = total base channels)
        self.C_base = 3 + 4 + 3 + 3 + 3 * self.max_rest_dim + 1
        
        self.delta_id  = ConditionalUNetDelta(in_ch=self.C_base, out_ch=self.C_base, z_dim=id_dim, base=64, depth=2)
        self.delta_exp = ConditionalUNetDelta(in_ch=self.C_base, out_ch=self.C_base, z_dim=exp_dim, base=64, depth=2)
        self.delta_app = ConditionalUNetDelta(in_ch=self.C_base, out_ch=self.C_base, z_dim=app_dim, base=64, depth=2)
        
        self.w_id  = nn.Parameter(torch.tensor(1.0))
        self.w_exp = nn.Parameter(torch.tensor(1.0))
        self.w_app = nn.Parameter(torch.tensor(1.0))

        self._save_image_dir = None
        self._delta_enabled = True

        self._prev_rest_dim = (self.active_sh_degree + 1)**2 - 1
        self._curr_rest_dim = self._prev_rest_dim
        self._zero_new_rest_once = False
        self._global_step = 0

    def enable_delta(self, flag: bool):
        self._delta_enabled = bool(flag)
    
    def is_delta_enabled(self) -> bool:
        return getattr(self, "_delta_enabled", True)
    
    def set_global_step(self, step: int):
        self._global_step = int(step)

    # -------------------------------------------------------------------------
    # Forward: export_UV -> export_gaussian_splats
    # -------------------------------------------------------------------------
    def forward(self, code_id, code_exp, code_app, return_aux: bool = False):
        out = self.export_UV(
            code_id, code_exp, code_app, delta_down=1, t_chunk=1, return_delta=return_aux
        )
        if return_aux:
            pos, quat, scale, sh_dc, sh_rest, opacity, aux = out
        else:
            pos, quat, scale, sh_dc, sh_rest, opacity = out

        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = self.export_gaussian_splats(
            pos, quat, scale, sh_dc, sh_rest, opacity,
            N=self.num_points, weighted=True, replacement=False, temperature=0.7
        )

        if return_aux:
            aux_maps = {"uv_pos": pos, "uv_scale": scale, "uv_opacity": opacity,
                        "d_id": aux["d_id"], "d_app": aux["d_app"], "d_exp": aux["d_exp"]}
            return (s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac), aux_maps

        return s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac
    
    def _quat_mul(self, qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
        # qa ⊗ qb, 채널 순서 (x,y,z,w)
        ax, ay, az, aw = qa[:,0:1], qa[:,1:2], qa[:,2:3], qa[:,3:4]
        bx, by, bz, bw = qb[:,0:1], qb[:,1:2], qb[:,2:3], qb[:,3:4]
        x = aw*bx + ax*bw + ay*bz - az*by
        y = aw*by - ax*bz + ay*bw + az*bx
        z = aw*bz + ax*by - ay*bx + az*bw
        w = aw*bw - ax*bx - ay*by - az*bz
        return torch.cat([x, y, z, w], dim=1)

    def _rotvec_to_quat(self, rv: torch.Tensor) -> torch.Tensor:
        # rv: (B,3,H,W) 축-각(라디안) 벡터 → quat(x,y,z,w)
        angle = rv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        axis  = rv / angle
        half  = 0.5 * angle
        s, c  = torch.sin(half), torch.cos(half)
        return torch.cat([axis * s, c], dim=1)

    # -------------------------------------------------------------------------
    # code (B,D) or (B,T,D) → base + delta; returns (B,T,·,H,W)
    # -------------------------------------------------------------------------
    def export_UV(self, code_id, code_exp, code_app, delta_down: int | None = 1, t_chunk: int | None = None, amp: bool = True, return_delta: bool = False):
        """
        Args:
            code_id : (B,T,D)
            code_exp: (B,T,D)
        Returns:
            pos, quat, scale, sh_dc, sh_rest, opacity: (B,T,·,H,W)
        """
        B, T, Di = code_id.shape
        _, _, De = code_exp.shape
        _, _, Da = code_app.shape
        H, W = self.H, self.W
        df = 1 if (delta_down is None or int(delta_down) <= 1) else int(delta_down)
        t_chunk = T if (t_chunk is None or t_chunk <= 0) else int(t_chunk)

        base_stack_B = torch.cat([
            self.img_position, self.img_quat, self.img_scale,
            self.img_sh_dc, self.img_sh_rest, self.img_opacity
        ], dim=0).unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # (B,C,H,W)

        chunks = []
        delta_id_chunks  = [] if return_delta else None
        delta_app_chunks = [] if return_delta else None
        delta_exp_chunks = [] if return_delta else None
        for t0 in range(0, T, t_chunk):
            t1 = min(T, t0 + t_chunk)
            z_id  = code_id[:, t0:t1, :].reshape(B * (t1 - t0), Di)
            z_exp = code_exp[:, t0:t1, :].reshape(B * (t1 - t0), De)
            z_app = code_app[:, t0:t1, :].reshape(B * (t1 - t0), Da)

            bs = base_stack_B[:, None, ...].expand(B, (t1 - t0), base_stack_B.shape[1], H, W)
            bs = bs.reshape(B * (t1 - t0), base_stack_B.shape[1], H, W)
        
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                idx = 0
                b_pos   = bs[:, idx:idx+3];                         idx += 3
                b_quat  = bs[:, idx:idx+4];                         idx += 4
                b_scale = bs[:, idx:idx+3];                         idx += 3   # ← 로짓 공간
                b_shdc  = bs[:, idx:idx+3];                         idx += 3
                b_shr   = bs[:, idx:idx+3*self.max_rest_dim];       idx += 3*self.max_rest_dim
                b_opac  = bs[:, idx:idx+1]
            
                if not getattr(self, "_delta_enabled", True):
                    delta = torch.zeros(bs.shape[0], self.C_base, H, W, device=bs.device, dtype=bs.dtype)
                    d_id = d_exp = d_app = torch.zeros_like(delta)
                else:
                    if df > 1:
                        bs_small = F.interpolate(bs, scale_factor=1.0/df, mode="bilinear", align_corners=False)
                        d_id_small  = self.delta_id (bs_small, z_id)
                        d_exp_small = self.delta_exp(bs_small, z_exp)
                        d_app_small = self.delta_app(bs_small, z_app)
                        delta_small = self.w_id * d_id_small + self.w_exp * d_exp_small + self.w_app * d_app_small
                        delta = F.interpolate(delta_small, size=(H, W), mode="bilinear", align_corners=False)
                        d_id  = F.interpolate(d_id_small,  size=(H, W), mode="bilinear", align_corners=False)
                        d_exp = F.interpolate(d_exp_small, size=(H, W), mode="bilinear", align_corners=False)
                        d_app = F.interpolate(d_app_small, size=(H, W), mode="bilinear", align_corners=False)
                    else:
                        d_id  = self.delta_id (bs, z_id)
                        d_exp = self.delta_exp(bs, z_exp)
                        d_app = self.delta_app(bs, z_app)
                        delta = self.w_id * d_id + self.w_exp * d_exp + self.w_app * d_app
                        
                idx = 0
                d_pos   = delta[:, idx:idx+3];                      idx += 3
                d_quat  = delta[:, idx:idx+4];                      idx += 4
                d_scale = delta[:, idx:idx+3];                      idx += 3
                d_shdc  = delta[:, idx:idx+3];                      idx += 3
                d_shr   = delta[:, idx:idx+3*self.max_rest_dim];    idx += 3*self.max_rest_dim
                d_opac  = delta[:, idx:idx+1]

                geom_warmup = getattr(hp, "geom_warmup_iters", 1)
                r_pos = getattr(hp, "max_pos_offset", 0.10)
                k_rot = getattr(hp, "delta_rot_scale", 1.0)
                k_scl = getattr(hp, "delta_scale_gain", 0.3)   # ← 0.5 → 0.05
                k_col = getattr(hp, "delta_color_scale", 0.15)
                k_opa = getattr(hp, "delta_opacity_scale", 0.50)

                step = getattr(self, "_global_step", 0)
                alpha = 1.0 if geom_warmup <= 0 else min(1.0, step / float(geom_warmup))

                # pos (동일, 제한)
                pos = b_pos + torch.tanh(d_pos) * (r_pos * alpha)

                # quat (소각 회전 곱)
                rv   = (k_rot * alpha) * torch.tanh(d_quat[:, :3, ...])
                dq   = self._rotvec_to_quat(rv)
                quat = F.normalize(self._quat_mul(dq, b_quat), dim=1, eps=1e-8)
                # quat = F.normalize(self._quat_mul(F.normalize(d_quat), b_quat), dim=1, eps=1e-8)

                # (중요) scale: 로짓공간에서 합 → sigmoid → [s_min,s_max]
                scale_logits = b_scale + (k_scl * alpha) * d_scale        # b_scale가 로짓이므로 그대로 더함
                scale = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(scale_logits)

                # 색/불투명도
                sh_dc = (b_shdc + k_col * torch.tanh(d_shdc)).clamp_(0.0, 1.0) if self.active_sh_degree == 0 else (b_shdc + d_shdc)
                if getattr(self, "_zero_new_rest_once", False):
                    s = 3 * getattr(self, "_prev_rest_dim", 0)
                    e = 3 * getattr(self, "_curr_rest_dim", 0)
                    if e > s:   # 새로 열린 계수 구간이 있을 때만
                        d_shr[:, s:e, ...] = 0
                    self._zero_new_rest_once = False
                sh_rest = b_shr + d_shr
                opacity = torch.sigmoid(b_opac + k_opa * d_opac)
                opacity = opacity * self.uv_valid.to(opacity.dtype)
                opa_floor = float(getattr(hp, "opacity_floor", 0.1))
                opacity = opa_floor + (1.0 - opa_floor) * opacity
                opacity = torch.nan_to_num(opacity, nan=opa_floor, posinf=1.0, neginf=opa_floor).clamp(0.0, 1.0)

                out_cat = torch.cat([pos, quat, scale, sh_dc, sh_rest, opacity], dim=1)
                out_chunk = out_cat.view(B, (t1 - t0), -1, H, W)
                chunks.append(out_chunk)
                
                if return_delta:
                    delta_id_chunks.append(d_id.view(B, (t1 - t0), self.C_base, H, W))
                    delta_app_chunks.append(d_app.view(B, (t1 - t0), self.C_base, H, W))
                    delta_exp_chunks.append(d_exp.view(B, (t1 - t0), self.C_base, H, W))
            
        out = torch.cat(chunks, dim=1)                                    # (B,T,C,H,W)

        # ---- slice channels ----
        idx = 0
        pos     = out[:, :, idx:idx+3];                   idx += 3
        quat    = out[:, :, idx:idx+4];                   idx += 4
        scale   = out[:, :, idx:idx+3];                   idx += 3
        sh_dc   = out[:, :, idx:idx+3];                   idx += 3
        sh_rest = out[:, :, idx:idx+3*self.max_rest_dim]; idx += 3*self.max_rest_dim
        opacity = out[:, :, idx:idx+1]

        if return_delta:
            d_id_full  = torch.cat(delta_id_chunks,  dim=1)  # (B,T,C,H,W)
            d_app_full = torch.cat(delta_app_chunks, dim=1)
            d_exp_full = torch.cat(delta_exp_chunks, dim=1)
            aux = {"d_id": d_id_full.contiguous(),
                "d_app": d_app_full.contiguous(),
                "d_exp": d_exp_full.contiguous()}
            return (pos.contiguous(), quat.contiguous(), scale.contiguous(),
                    sh_dc.contiguous(), sh_rest.contiguous(), opacity.contiguous(), aux)

        return (pos.contiguous(), quat.contiguous(), scale.contiguous(),
                sh_dc.contiguous(), sh_rest.contiguous(), opacity.contiguous())

    # -------------------------------------------------------------------------
    # UV images → N splats (GPU-parallel random sampling in UV)
    # -------------------------------------------------------------------------
    # @torch.no_grad()
    def export_gaussian_splats(
        self,
        pos, quat, scale, sh_dc, sh_rest, opacity,
        N: int | None = None,
        weighted: bool = True,
        replacement: bool = False,
        temperature: float = 1.0,
        sh_chunk: int = 32
    ):
        """
        Args:
            pos, quat, scale, sh_dc, sh_rest, opacity: (B, T, C, H, W)
        Returns:
            splat_pos      : (B, T, N, 3)
            splat_quat     : (B, T, N, 4)
            splat_scale    : (B, T, N, 3)
            splat_sh_dc    : (B, T, N, 3)
            splat_sh_rest  : (B, T, N, 3*max_rest_dim)
            splat_opacity  : (B, T, N, 1)
        """
        B, T, _, H, W = pos.shape
        BT, HW = B * T, H * W
        N = self.num_points if N is None else int(N)
        device = pos.device

        def flatten(x):  # (B,T,C,H,W) -> (BT,C,H,W)
            return x.reshape(B * T, *x.shape[2:])

        pos_f, quat_f, scale_f = flatten(pos), flatten(quat), flatten(scale)
        shdc_f, shr_f, opac_f  = flatten(sh_dc), flatten(sh_rest), flatten(opacity)

        # Sampling indices in UV with opacity-weighted multinomial (GPU)
        if weighted:
            w = opac_f.reshape(BT, HW).float()
            uvv = self.uv_valid.to(w.dtype).view(1, HW).expand(BT, HW)
            w = w * uvv

            # 안전장치
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w.clamp_(min=0.0)

            if temperature != 1.0:
                w = (w + 1e-12).pow(1.0 / temperature)

            # 완전 영행 방지용 uniform mix
            uniform = torch.full_like(w, 1.0 / HW)
            mix = 0.05
            w = (1.0 - mix) * w + mix * uniform

            row_sum = w.sum(dim=1, keepdim=True)
            bad = (row_sum <= 0) | (~torch.isfinite(row_sum))
            w = torch.where(bad, uniform, w / row_sum.clamp_min(1e-12))

            # 무중복 샘플링(겹침 방지)
            k = min(N, HW)
            idx = torch.multinomial(w, num_samples=k, replacement=replacement)
        else:
            idx = torch.randint(0, HW, (BT, min(N, HW)), device=device)

        # Gather utility: (BT,C,H,W) -> (BT,N,C) with optional channel chunking
        def gather_map(x: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
            BT_, C, H_, W_ = x.shape
            xf = x.reshape(BT_, C, H_ * W_)
            if (chunk is None) or (C <= (chunk or 1_000_000_000)):
                idx_exp = idx.unsqueeze(1).expand(BT_, C, N)
                out = torch.gather(xf, 2, idx_exp)             # (BT,C,N)
                return out.transpose(1, 2).contiguous()        # (BT,N,C)
            outs = []
            for s in range(0, C, chunk):
                e = min(s + chunk, C)
                idx_exp = idx.unsqueeze(1).expand(BT_, e - s, N)
                out_ch = torch.gather(xf[:, s:e, :], 2, idx_exp)
                outs.append(out_ch)
            out = torch.cat(outs, dim=1)                       # (BT,C,N)
            return out.transpose(1, 2).contiguous()            # (BT,N,C)

        splat_pos     = gather_map(pos_f,   None)
        splat_quat    = gather_map(quat_f,  None)
        splat_scale   = gather_map(scale_f, None)
        splat_sh_dc   = gather_map(shdc_f,  None)
        splat_sh_rest = gather_map(shr_f,   sh_chunk)
        splat_opacity = gather_map(opac_f,  None)

        def unflatten(xC): return xC.view(B, T, N, xC.shape[-1]).contiguous()
        return (unflatten(splat_pos), unflatten(splat_quat), unflatten(splat_scale),
                unflatten(splat_sh_dc), unflatten(splat_sh_rest), unflatten(splat_opacity))

            
    def train_stage1_id(self, freeze_posquat_first_k=1500, step=0):
        # Stage1: base + delta 학습 (초반엔 pos/quat만 잠가 안정화)
        for name, p in self.named_parameters():
            if name.startswith("delta_id"):
                p.requires_grad = True
            elif name.startswith("delta_app"):
                p.requires_grad = True
                
            elif name.startswith("delta_exp"):
                p.requires_grad = False
            elif name.startswith("img_position") or name.startswith("img_quat"):
                p.requires_grad = (step >= freeze_posquat_first_k)
            elif name.startswith("img_"):
                p.requires_grad = True

            elif name in ("w_id", "w_app"):
                p.requires_grad = True
            elif name in ("w_exp", ):
                p.requires_grad = False
                
            else:
                p.requires_grad = False
        self.enable_delta(True)

    def train_stage2_exp(self):
        # Stage2: delta만 학습 (base 전부 고정)
        for name, p in self.named_parameters():
            if name.startswith("delta_exp"):
                p.requires_grad = True
            elif name.startswith("delta_id"):
                p.requires_grad = True
            elif name.startswith("img_position" or "img_quat"):
                p.requires_grad = False
            elif name.startswith("img_"):
                p.requires_grad = True
            # elif name.startswith("img_"):
            #     p.requires_grad = False
            elif name in ("w_id", "w_exp", "w_app"):
                p.requires_grad = False
            else:
                p.requires_grad = False
        self.enable_delta(True)


    # -------------------------------------------------------------------------
    # Increase active SH degree; convert DC map RGB→SH once when stepping from 0→1
    # -------------------------------------------------------------------------
    def oneupSHdegree(self):
        """Increment active SH degree; apply RGB2SH to img_sh_dc when leaving degree 0."""
        if self.active_sh_degree == 0:
            with torch.no_grad():
                self.img_sh_dc.copy_(RGB2SH(self.img_sh_dc.clamp(0, 1)))
        if self.active_sh_degree < self.max_sh_degree:
            old = (self.active_sh_degree + 1)**2 - 1
            self.active_sh_degree += 1
            new = (self.active_sh_degree + 1)**2 - 1
            self._prev_rest_dim = old
            self._curr_rest_dim = new
            self._zero_new_rest_once = True
            print("active_sh_degree:", self.active_sh_degree)
    # -------------------------------------------------------------------------
    # Render sequence: code → maps → splats → differentiable renderer
    # -------------------------------------------------------------------------
    def render(self, s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac, cam_matrix, cam_params, yaw_deg, sh_degree):
        """
        Args:
            gs_params
            cam_matrix: (B, 3, 4)
            cam_params: dict with intrinsics
            yaw_deg: float or list-like supported by your renderer
            sh_degree: int, SH degree for rendering
        Returns:
            images: (B, T, 3, H, W)
        """
        B, T = s_pos.shape[:2]
        if sh_degree == 0:
            color_arg = s_shdc.view(B * T, self.num_points, 3).clamp(0, 1)
            shs_arg = None
        else:
            coeffs = (sh_degree + 1) ** 2
            cur_rest_dim = coeffs - 1
            shr_flat = s_shr[..., :3 * cur_rest_dim]
            f_rest  = shr_flat.view(B, T, self.num_points, 3, cur_rest_dim)
            shs_arg = torch.cat([s_shdc.unsqueeze(-1), f_rest], dim=-1)  # (B,T,N,3,coeffs)
            shs_arg = shs_arg.permute(0, 1, 2, 4, 3).contiguous()
            shs_arg = shs_arg.view(B * T, self.num_points, coeffs, 3)
            #shs_arg = shs_arg.view(B * T, self.num_points, coeffs, 3).contiguous()
            color_arg = None

        xyz   = s_pos.view(B * T, self.num_points, 3)
        rots  = s_quat.view(B * T, self.num_points, 4)
        scals = s_scale.view(B * T, self.num_points, 3)
        opacs = s_opac.view(B * T, self.num_points, 1)

        cam_mat_BT = cam_matrix.repeat_interleave(T, dim=0)

        gs_params = {
            'xyz': xyz,
            'rotations': rots,
            'scales': scals,
            'colors': color_arg,
            'opacities': opacs,
            'shs': shs_arg,
        }

        rendered = render_gaussian(gs_params, cam_mat_BT, cam_params=cam_params, yaw_deg=yaw_deg, sh_degree=sh_degree)
        images = rendered["images"].view(B, T, *rendered["images"].shape[1:])

        return images

    # -------------------------------------------------------------------------
    # Quick base-only render (delta=0) for a list of yaws
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def render_base(
        self,
        cam_json=None,
        yaw_list=None,
        save_dir=None,           # e.g. os.path.join(hp.save_path, "render_base")
        prefix="yaw",            # filename prefix
        save_video=False,        # also export an MP4
        fps=25,
        overwrite=False,
        epoch=0
    ):
        """
        Returns:
            outs:       list of (1,3,H,W) tensors in [0,1]
            paths:      list of saved png paths (if save_dir)
            video_path: str | None
        """
        json_path = cam_json if cam_json is not None else hp.single_cam_path
        cam_matrix, cam_params = load_cam(json_path)  # (1,3,4), dict

        B, T = 1, 1
        pos   = self.img_position.unsqueeze(0).unsqueeze(0)
        quat  = F.normalize(self.img_quat, dim=0, eps=1e-8).unsqueeze(0).unsqueeze(0)
        scale_map = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.img_scale)
        scale = scale_map.unsqueeze(0).unsqueeze(0)
        sh_dc = self.img_sh_dc.unsqueeze(0).unsqueeze(0)
        sh_r  = self.img_sh_rest.unsqueeze(0).unsqueeze(0)
        opac  = torch.sigmoid(self.img_opacity).unsqueeze(0).unsqueeze(0)

        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = self.export_gaussian_splats(
            pos, quat, scale, sh_dc, sh_r, opac, N=self.num_points, weighted=True, replacement=False, temperature=0.7
        )

        if yaw_list is None:
            yaw_list = [i * (360 / 32) for i in range(32)]

        save_dir = os.path.join(save_dir, f"{epoch:06d}")
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # if overwrite:
            #     for f in os.listdir(save_dir):
            #         try: os.remove(os.path.join(save_dir, f))
            #         except: pass

        outs, paths = [], []

        def to_u8(img_1x3xHxW):
            x = img_1x3xHxW[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            return x

        for i, yaw in enumerate(yaw_list):
            coeffs = (self.active_sh_degree + 1) ** 2
            if self.active_sh_degree == 0:
                color_arg = s_shdc.view(B * T, self.num_points, 3).clamp(0, 1)
                shs_arg = None
            else:
                cur_rest_dim = coeffs - 1
                shr_flat = s_shr[..., :3 * cur_rest_dim]
                f_rest = shr_flat.view(B, T, self.num_points, 3, cur_rest_dim)
                shs_arg = torch.cat([s_shdc.unsqueeze(-1), f_rest], dim=-1)
                shs_arg = shs_arg.permute(0, 1, 2, 4, 3).contiguous()
                shs_arg = shs_arg.view(B * T, self.num_points, coeffs, 3)
                #shs_arg = shs_arg.view(B * T, self.num_points, coeffs, 3).contiguous()
                color_arg = None

            gs_params = {
                'xyz': s_pos.view(B * T, self.num_points, 3),
                'rotations': s_quat.view(B * T, self.num_points, 4),
                'scales': s_scale.view(B * T, self.num_points, 3),
                'colors': color_arg,
                'opacities': s_opac.view(B * T, self.num_points, 1),
                'shs': shs_arg,
            }
            render_out = render_gaussian(gs_params, cam_matrix, cam_params, yaw_deg=yaw, sh_degree=self.active_sh_degree)
            img = render_out["images"]  # (1,3,H,W)
            outs.append(img)

            if save_dir is not None:
                fname = f"{prefix}_{i:03d}_yaw{yaw:06.2f}.png"
                path = os.path.join(save_dir, fname)
                Image.fromarray(to_u8(img)).save(path)
                paths.append(path)

        video_path = None
        if save_video and save_dir is not None and len(outs) > 0:
            # NOTE: needs "import imageio" at top of file
            video_path = os.path.join(save_dir, f"{prefix}.mp4")
            writer = imageio.get_writer(video_path, fps=fps)
            try:
                for img in outs:
                    writer.append_data(to_u8(img))
            finally:
                writer.close()

        return outs, paths, video_path

    @property
    def sh_degree(self):
        """Current active spherical harmonics degree."""
        return self.active_sh_degree
    
    @property
    def texture(self):
        return self.tex

    # -------------------------------------------------------------------------
    # Export a single timestep of Gaussians from code (B,T,D)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def export_gaussians(self, code: torch.Tensor, step_idx: int = 0):
        """
        Args:
            code: (B, T, D)
        Returns:
            dict with means, scales, rotations, opacity, sh, sh_degree
        """
        if code is None or code.dim() != 3:
            raise ValueError(f"export_gaussians expects code (B,T,D), got {None if code is None else tuple(code.shape)}")
        B, T, D = code.shape
        step = int(step_idx) % T
        code_one = code[:, step:step+1, :]  # (B,1,D)

        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = self.forward(code_one)  # (B,1,·,H,W)

        means     = s_pos[:, 0]      # (B,N,3)
        scales    = s_scale[:, 0]    # (B,N,3)
        rotations = s_quat[:, 0]     # (B,N,4)
        opacity   = s_opac[:, 0]     # (B,N,1)

        sd = self.active_sh_degree
        if sd == 0:
            sh = s_shdc[:, 0].unsqueeze(-1)  # (B,N,3,1)
        else:
            coeffs = (sd + 1) ** 2
            rest_dim = coeffs - 1
            f_rest = s_shr[:, 0, :, :3 * rest_dim].view(B, self.num_points, 3, rest_dim)
            sh = torch.cat([s_shdc[:, 0].unsqueeze(-1), f_rest], dim=-1)  # (B,N,3,coeffs)

        return {
            "means": means, "scales": scales, "rotations": rotations,
            "opacity": opacity, "sh": sh, "sh_degree": sd
        }

    def _minmax_norm(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Per-channel min-max normalization.
        x: (C,H,W) on any device. Returns (C,H,W) in [0,1].
        """
        xc = x.detach().float()
        C, H, W = xc.shape
        flat = xc.view(C, -1)
        xmin = flat.min(dim=1, keepdim=True).values
        xmax = flat.max(dim=1, keepdim=True).values
        denom = (xmax - xmin).clamp_min(eps)
        norm = ((flat - xmin) / denom).view(C, H, W)
        return norm

    def _to_uint8_image(self, x: torch.Tensor) -> Image.Image:
        """
        x: (1,H,W) or (3,H,W) in [0,1] float tensor → PIL Image.
        """
        xc = x.detach().clamp(0, 1).mul(255).byte().cpu()
        if xc.shape[0] == 1:
            return Image.fromarray(xc[0].numpy(), mode="L")
        elif xc.shape[0] == 3:
            return Image.fromarray(xc.permute(1, 2, 0).numpy(), mode="RGB")
        else:
            raise ValueError(f"_to_uint8_image expects 1 or 3 channels, got {x.shape[0]}")

    def _save_tensor_image(self, x: torch.Tensor, path: str, do_minmax: bool = True):
        """
        x: (1,H,W) or (3,H,W). Optionally min-max normalize per channel, then save.
        """
        if do_minmax:
            x = self._minmax_norm(x)
        img = self._to_uint8_image(x)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)

    def _save_quat_as_images(self, quat_chw: torch.Tensor, dirpath: str, prefix: str):
        """
        quat_chw: (4,H,W). Visualize as:
        - vec RGB: map (x,y,z) in [-1,1] → [0,1]
        - w  Gray: map w in [-1,1] → [0,1]
        Uses unit quaternions with sign fix so w >= 0.
        """
        q = quat_chw.detach().float()
        # Normalize per-pixel across channel dim
        norm = (q.pow(2).sum(0, keepdim=True).sqrt().clamp_min(1e-8))
        q = q / norm
        # Sign fix: q and -q represent the same rotation → enforce w>=0
        sign = torch.where(q[3:4] >= 0, torch.tensor(1.0, device=q.device), torch.tensor(-1.0, device=q.device))
        q = q * sign
        # Map to display
        vec_rgb = ((q[0:3] + 1.0) * 0.5).clamp(0, 1)  # (3,H,W)
        w_gray  = ((q[3:4] + 1.0) * 0.5).clamp(0, 1)  # (1,H,W)
        os.makedirs(dirpath, exist_ok=True)
        self._save_tensor_image(vec_rgb, os.path.join(dirpath, f"{prefix}_quat_vec_rgb.png"), do_minmax=False)
        self._save_tensor_image(w_gray,  os.path.join(dirpath, f"{prefix}_quat_w_gray.png"),  do_minmax=False)

    def _save_sh_rest_grid(self, sh_rest_chw: torch.Tensor, path: str, cols: int = 8):
        """
        sh_rest_chw: (3*R,H,W) where R = max_rest_dim.
        Tiles R coefficient triplets (RGB each) into a grid image.
        """
        C, H, W = sh_rest_chw.shape
        assert C % 3 == 0, f"sh_rest has {C} channels, not multiple of 3"
        R = C // 3
        tiles = sh_rest_chw.view(R, 3, H, W)  # (R,3,H,W)

        rows = (R + cols - 1) // cols
        canvas = torch.zeros(3, rows * H, cols * W, dtype=tiles.dtype, device=tiles.device)

        for i in range(R):
            r, c = divmod(i, cols)
            tile = self._minmax_norm(tiles[i])  # per-coefficient minmax
            canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = tile

        self._save_tensor_image(canvas, path, do_minmax=False)

    def export_base_img(self, save_root: str = "results/uv_base", sh_cols: int = 8, use_minmax_for_all: bool = True):
        """
        Saves the learnable base maps as images.
        - position_rgb.png : (3,H,W), per-channel min-max
        - quat_vec_rgb.png : (x,y,z) mapped to RGB, w>=0 sign fix
        - quat_w_gray.png  : w component as grayscale
        - scale_rgb.png    : (3,H,W), per-channel min-max
        - sh_dc_rgb.png    : (3,H,W), assumes already in [0,1]
        - sh_rest_grid.png : tiled coefficients (each 3ch) in a grid
        - opacity_gray.png : (1,H,W), min-max
        """
        os.makedirs(save_root, exist_ok=True)

        # Position
        self._save_tensor_image(self.img_position, os.path.join(save_root, "position_rgb.png"), do_minmax=use_minmax_for_all)

        # Quaternion (vec RGB + w Gray)
        self._save_quat_as_images(self.img_quat, save_root, prefix="base")

        # Scale
        scale_vis = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.img_scale)
        self._save_tensor_image(scale_vis, os.path.join(save_root, "scale_rgb.png"), do_minmax=use_minmax_for_all)

        # SH DC (texture)
        # If you require strict [0,1], clamp; otherwise preserve as is.
        if self.active_sh_degree == 0:
            sh_dc_vis = self.img_sh_dc.detach().float().clamp(0, 1)
        else:
            # SH 계수 → RGB로 환산 후 저장
            sh_dc_vis = SH2RGB(self.img_sh_dc.detach().float()).clamp(0, 1)
        self._save_tensor_image(sh_dc_vis, os.path.join(save_root, "sh_dc_rgb.png"), do_minmax=False)

        # SH Rest grid
        self._save_sh_rest_grid(self.img_sh_rest, os.path.join(save_root, "sh_rest_grid.png"), cols=sh_cols)

        # Opacity
        opa_vis = torch.sigmoid(self.img_opacity)
        self._save_tensor_image(opa_vis, os.path.join(save_root, "opacity_gray.png"), do_minmax=use_minmax_for_all)

    def export_result_img(
        self,
        pos: torch.Tensor,       # (B,T,3,H,W)
        quat: torch.Tensor,      # (B,T,4,H,W)
        scale: torch.Tensor,     # (B,T,3,H,W)
        sh_dc: torch.Tensor,     # (B,T,3,H,W)
        sh_rest: torch.Tensor,   # (B,T,3*R,H,W)
        opacity: torch.Tensor,   # (B,T,1,H,W)
        save_root: str = "results/uv_result",
        sh_cols: int = 8,
        use_minmax_for_all: bool = True
    ):
        """
        Saves forward outputs for each batch/time step.
        Directory structure:
        save_root/
            b{b:02d}_t{t:04d}/
            position_rgb.png
            quat_vec_rgb.png
            quat_w_gray.png
            scale_rgb.png
            sh_dc_rgb.png
            sh_rest_grid.png
            opacity_gray.png
        """
        B, T = pos.shape[0], pos.shape[1]
        os.makedirs(save_root, exist_ok=True)

        for b in range(B):
            for t in range(T):
                out_dir = os.path.join(save_root, f"b{b:02d}_t{t:04d}")
                os.makedirs(out_dir, exist_ok=True)

                # Position
                self._save_tensor_image(pos[b, t], os.path.join(out_dir, "position_rgb.png"), do_minmax=use_minmax_for_all)

                # Quaternion (vec RGB + w Gray)
                self._save_quat_as_images(quat[b, t], out_dir, prefix="result")

                # Scale
                self._save_tensor_image(scale[b, t], os.path.join(out_dir, "scale_rgb.png"), do_minmax=use_minmax_for_all)

                # SH DC
                if self.active_sh_degree == 0:
                    sh_dc_vis = sh_dc[b, t].detach().float().clamp(0, 1)
                else:
                    sh_dc_vis = SH2RGB(sh_dc[b, t].detach().float()).clamp(0, 1)
                self._save_tensor_image(sh_dc_vis, os.path.join(out_dir, "sh_dc_rgb.png"), do_minmax=False)

                # SH Rest grid
                self._save_sh_rest_grid(sh_rest[b, t], os.path.join(out_dir, "sh_rest_grid.png"), cols=sh_cols)

                # Opacity
                self._save_tensor_image(opacity[b, t], os.path.join(out_dir, "opacity_gray.png"), do_minmax=use_minmax_for_all)
    
    def save_image(
        self, gt, pred, epoch, save_root="results",
        src=None, tag=None, timestamp="none",
        seed_text: str | None = None,
        view_nums: list[int] | None = None
    ):
        """
        Layout:
        [ SID 라벨 (중앙 정렬, 크게) ]
        [ SRC 한 장 ]
        |  (왼쪽 고정 컬럼)
        |  (오른쪽 본문: 각 열은 [VIEW 라벨; GT; PRED])
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import os, torch

        def tensor_to_img(x):
            return (x.detach().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

        # ---------- shapes ----------
        V = min(gt.shape[0], pred.shape[0])
        T = min(gt.shape[1], pred.shape[1])
        C, H, W = gt.shape[2], gt.shape[3], gt.shape[4]
        assert C == 3, f"Expect 3-channel images, got {C}"

        # ---------- sizes ----------
        LABEL_H = max(24, H // 8)   # 라벨 줄 높이 (여백)

        # ---------- 폰트: 함수 내부에서 자동 결정 ----------
        # H에 비례해 글자 픽셀 크기 계산 (원하면 계수 0.24를 조절)
        font_px = max(24, int(H * 0.09))

        def _load_truetype(px: int):
            for name in ["DejaVuSans.ttf", "Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                try:
                    return ImageFont.truetype(name, size=px)
                except Exception:
                    pass
            return None

        _TTF = _load_truetype(font_px)         # 있을 때: 실제 큰 글자
        _DEF = ImageFont.load_default()         # 없을 때: 작은 비트맵 폰트
        _DEF_BASE_H = 11                        # 기본폰트 대략 높이(대충 11px)
        _DEF_SCALE = max(1, int(round(font_px / _DEF_BASE_H)))

        # ---------- left column ----------
        if (src is not None) and isinstance(src, torch.Tensor):
            if src.dim() == 5:      # (V,T,3,H,W)
                src_one = src[0, 0]
            elif src.dim() == 4:    # (T,3,H,W)
                src_one = src[0]
            elif src.dim() == 3:    # (3,H,W)
                src_one = src
            else:
                raise ValueError(f"Unsupported src shape: {tuple(src.shape)}")
            src_np = tensor_to_img(src_one)
        else:
            src_np = np.zeros((H, W, 3), dtype=np.uint8)

        left_col_h = LABEL_H + H
        left_col_w = W
        body_col_h = LABEL_H + 2 * H
        body_col_w = W
        canvas_h = max(left_col_h, body_col_h)

        left_np = np.zeros((canvas_h, left_col_w, 3), dtype=np.uint8)
        left_np[LABEL_H:LABEL_H+H, :, :] = src_np
        left_img = Image.fromarray(left_np).convert("RGBA")

        # ---------- 중앙 정렬 텍스트 유틸 (TTF 있으면 그대로, 없으면 확대 렌더링) ----------
        def draw_center_text(img: Image.Image, text: str, box_xywh, fg=(255,255,255), bg=(0,0,0)):
            x, y, w, h = box_xywh
            if _TTF is not None:
                d = ImageDraw.Draw(img)
                bx0, by0, bx1, by1 = d.textbbox((0, 0), text, font=_TTF)
                tw, th = (bx1 - bx0), (by1 - by0)
                tx = x + (w - tw) / 2
                ty = y + (h - th) / 2
                pad = 8
                d.rectangle((tx - pad, ty - pad, tx + tw + pad, ty + th + pad), fill=bg)
                d.text((tx, ty), text, fill=fg, font=_TTF)
            else:
                # 기본 폰트로 그린 후 비례 확대해서 합성 (글자 자체가 커짐)
                tmp = Image.new("RGBA", (w // _DEF_SCALE or 1, h // _DEF_SCALE or 1), (0,0,0,0))
                d = ImageDraw.Draw(tmp)
                bx0, by0, bx1, by1 = d.textbbox((0, 0), text, font=_DEF)
                tw, th = (bx1 - bx0), (by1 - by0)
                tx = max(0, (tmp.width - tw) // 2)
                ty = max(0, (tmp.height - th) // 2)
                # 작은 캔버스에 먼저 그림
                d.rectangle((tx-2, ty-2, tx+tw+2, ty+th+2), fill=(0,0,0,255))
                d.text((tx, ty), text, fill=fg + (255,), font=_DEF)
                # 원하는 크기로 확대
                big = tmp.resize((w, h), resample=Image.NEAREST)
                # 박스 위치에 합성
                patch = Image.new("RGBA", img.size, (0,0,0,0))
                patch.paste(big, (x, y))
                img.alpha_composite(patch)

        # 헤더 라벨: seed_text가 있으면 그대로 사용
        if seed_text:
            draw_center_text(left_img, seed_text, (0, 0, W, LABEL_H))

        # ---------- body columns ----------
        body_cols = []
        for i in range(V):
            vtxt = f"view_{int(view_nums[i]):02d}" if (view_nums is not None and i < len(view_nums)) else f"view_{i:02d}"
            for j in range(T):
                col_np = np.zeros((canvas_h, body_col_w, 3), dtype=np.uint8)
                col_np[LABEL_H:LABEL_H+H, :, :] = tensor_to_img(gt[i, j])
                col_np[LABEL_H+H:LABEL_H+2*H, :, :] = tensor_to_img(pred[i, j])
                col_img = Image.fromarray(col_np).convert("RGBA")
                draw_center_text(col_img, vtxt, (0, 0, W, LABEL_H))
                body_cols.append(np.array(col_img.convert("RGB")))

        # ---------- concat ----------
        if body_cols:
            body_np = np.concatenate(body_cols, axis=1)
            out_np = np.concatenate([np.array(left_img.convert("RGB")), body_np], axis=1)
        else:
            out_np = np.array(left_img.convert("RGB"))

        out_img = Image.fromarray(out_np)

        # ---------- save ----------
        if getattr(self, "_save_image_dir", None) is None:
            self._save_image_dir = os.path.join(save_root, timestamp)
            os.makedirs(self._save_image_dir, exist_ok=True)

        fname = f"{epoch:05d}_{tag}.png" if tag else f"{epoch:05d}.png"
        save_path = os.path.join(self._save_image_dir, fname)
        out_img.save(save_path)
        print(f"Saved Images: {save_path}")

        
class IdExpMLP(nn.Module):
    """
    간단한 MLP로 identity(정적) + expression(시계열)을 융합해
    프레임별 displacement code 시퀀스(B, T, emb_dim)를 출력합니다.

    Args:
        id_dim, exp_dim, emb_dim: hp에서 가져오되 오버라이드 가능
        hidden: 내부 히든 채널(Default: hp.hidden_dim)
        n_layers: MLP 층 수 (입/출 포함하면 n_layers>=2 권장)
        dropout: 드롭아웃 비율
        use_posenc: 프레임 인덱스에 대한 간단한 사인/코사인 포지셔널 인코딩 사용
        pos_dim: 포지셔널 인코딩 차원(짝수 권장)

    입출력:
        forward(identity, expression_seq) ->
            identity:       (B, Di)
            expression_seq: (B, T, De)
            returns:        (B, T, emb_dim)
    """
    def __init__(
        self,
        id_dim: int = getattr(hp, "id_dim", 512),
        exp_dim: int = getattr(hp, "exp_dim", 512),
        emb_dim: int = getattr(hp, "emb_dim", 128),
        hidden:  int = getattr(hp, "hidden_dim", 512),
        n_layers: int = 3,
        dropout: float = 0.1,
        use_posenc: bool = True,
        pos_dim: int = 16,
    ):
        super().__init__()
        assert n_layers >= 2, "n_layers는 2 이상이어야 합니다."

        self.id_dim   = id_dim
        self.exp_dim  = exp_dim
        self.emb_dim  = emb_dim
        self.hidden   = hidden
        self.use_pos  = bool(use_posenc)
        self.pos_dim  = int(pos_dim // 2) * 2  # 짝수로 정렬

        # 1) 독립 선형투영 + LayerNorm + 게이팅(학습 스칼라)
        self.id_proj  = nn.Linear(id_dim,  hidden)
        self.exp_proj = nn.Linear(exp_dim, hidden)
        self.id_ln    = nn.LayerNorm(hidden)
        self.exp_ln   = nn.LayerNorm(hidden)
        self.id_gate  = nn.Parameter(torch.tensor(1.0))
        self.exp_gate = nn.Parameter(torch.tensor(1.0))

        # 2) (선택) 프레임 포지셔널 인코딩 → hidden으로 사상
        if self.use_pos:
            self.pos_fc = nn.Linear(self.pos_dim, hidden)

        # 3) 융합 입력: [exp_h, id_h, exp_h * id_h, (pos_h)]
        in_hidden = hidden * 3 + (hidden if self.use_pos else 0)

        mlp = []
        dims = [in_hidden] + [hidden] * (n_layers - 1)
        for i in range(n_layers - 1):
            mlp += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*mlp)

        # 4) 출력 투영 + 잔차(residual) (exp → emb_dim)
        self.out = nn.Linear(hidden, emb_dim)
        self.exp_res = nn.Linear(exp_dim, emb_dim)
        self.res_scale = nn.Parameter(torch.tensor(0.1))  # 초기엔 잔차 영향 작게

        self.id_to_emb = nn.Sequential(
            nn.Linear(id_dim, hidden), nn.GELU(),
            nn.Linear(hidden, exp_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier 계열 초기화로 수렴 안정화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _posenc_sin_cos(self, T: int, device: torch.device) -> torch.Tensor:
        """
        간단한 사인/코사인 포지셔널 인코딩 (T, pos_dim)
        """
        if self.pos_dim == 0:
            return None
        t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div_term = torch.exp(
            torch.arange(0, self.pos_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, self.pos_dim))
        )  # (pos_dim/2,)
        pe = torch.zeros(T, self.pos_dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(t * div_term)
        pe[:, 1::2] = torch.cos(t * div_term)
        return pe  # (T, pos_dim)

    def forward(self, identity: torch.Tensor, expression_seq: torch.Tensor) -> torch.Tensor:
        """
        identity:       (B, Di)
        expression_seq: (B, T, De)
        returns:        (B, T, emb_dim)
        """
        B, T, De = expression_seq.shape
        assert identity.dim() == 2 and identity.shape[0] == B, \
            f"identity shape은 (B, Di)여야 합니다. got {tuple(identity.shape)}"

        # 1) 투영 + 정규화 + 게이팅
        id_h  = self.id_ln(self.id_proj(identity))            # (B,H)
        exp_h = self.exp_ln(self.exp_proj(expression_seq))    # (B,T,H)

        # broadcast id_h over time
        id_h_t = id_h.unsqueeze(1).expand(-1, T, -1)          # (B,T,H)

        id_h_t = self.id_gate  * id_h_t
        exp_h  = self.exp_gate * exp_h

        # 2) (선택) 포지셔널 인코딩
        if self.use_pos:
            pe = self._posenc_sin_cos(T, expression_seq.device)          # (T,pos_dim)
            pos_h = self.pos_fc(pe).unsqueeze(0).expand(B, -1, -1)       # (B,T,H)
            fuse_in = torch.cat([exp_h, id_h_t, exp_h * id_h_t, pos_h], dim=-1)
        else:
            fuse_in = torch.cat([exp_h, id_h_t, exp_h * id_h_t], dim=-1) # (B,T,3H)

        # 3) 프레임별 MLP
        x = self.mlp(fuse_in)                                            # (B,T,H)

        # 4) 출력 + 잔차(exp → emb_dim)
        out = self.out(x)                                                # (B,T,D)
        res = self.exp_res(expression_seq)                               # (B,T,D)
        code_exp = out + self.res_scale * res
        
        code_id_l = self.id_to_emb(identity)
        code_id = code_id_l.unsqueeze(1).expand(B, T, -1).contiguous()
        
        return code_id, code_exp                                                         # (B,T,emb_dim)

class DisplacementDecoder(nn.Module):
    def __init__(self, code_dim=hp.emb_dim, num_points=hp.num_points, init_sh_degree=0, max_sh_degree=3):
        super().__init__()
        self.num_points = num_points
        self.active_sh_degree = init_sh_degree
        self.max_sh_degree = max_sh_degree
        self.coeffs = (self.active_sh_degree + 1) ** 2
        self.max_coeffs = (self.max_sh_degree + 1) ** 2
        self.max_rest_dim = self.max_coeffs - 1

        out_dim = 3 + 4 + 3 + 3 + 3 * self.max_rest_dim + 1 # pos, quat, scale, sh_dc, sh_rest, opacity
        self.mlp = nn.Sequential(
            nn.Linear(code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_points * out_dim)  
        )

        last_linear: nn.Linear = self.mlp[-1]
        with torch.no_grad():
            w = last_linear.weight.view(self.num_points, out_dim, -1)
            b = last_linear.bias.view(self.num_points, out_dim)

            dc_start = 3 + 4 + 3
            dc_size = 3
            w[:, dc_start:dc_start+dc_size, :].zero_()
            b[:, dc_start:dc_start+dc_size].zero_()

            rest_start = dc_start + dc_size
            rest_size = 3 * self.max_rest_dim
            w[:, rest_start:rest_start+rest_size, :].zero_()
            b[:, rest_start:rest_start+rest_size].zero_()

            last_linear.weight.copy_(w.view(self.num_points * out_dim, -1))
            last_linear.bias.copy_(b.view(self.num_points * out_dim))

        # Learnable base parameters (1, N, D)
        self.register_parameter("base_pos", nn.Parameter(torch.zeros(1, 1, num_points, 3)))
        self.register_parameter("base_quat", nn.Parameter(F.normalize(torch.rand(1, 1, num_points, 4), dim=-1)))
        self.register_parameter("base_scale", nn.Parameter(torch.zeros(1, 1, num_points, 3)))
        self.register_parameter("base_sh_dc", nn.Parameter(torch.ones(1, 1, num_points, 3) * 0.5))
        self.register_parameter("base_sh_rest", nn.Parameter(torch.zeros(1, 1, num_points, 3 * self.max_rest_dim)))
        self.register_parameter("base_opacity", nn.Parameter(torch.ones(1, 1, num_points, 1) * 0.9))

        self._save_image_dir = None

    def initialize(self, mesh):
        vertices, faces = load_obj_as_mesh(mesh)
        vertices = vertices.cuda().unsqueeze(0)
        faces = faces.cuda()

        ##############
        import random
        import numpy as np
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # <- GPU 기반 샘플링에도 필요

        # #change
        # ############### === Load FLAME vertices from SMIRK ===
        # print(vertices.shape)
        # vertices_path = "/source/Hyeonho/Research/MonoFace/submodules/smirk/results/cam_params/flame_vertices.pt"
        # vertices = torch.load(vertices_path).cuda()  # shape: (1, N, 3)
        # print(vertices.shape)
        # exit()

        # print(vertices.mean(dim=1))
        # vertices -= vertices.mean(dim=1).unsqueeze(1) # alignment
        flame_trans = torch.tensor([[ -0.00888654, -0.00508765,  -0.26269796]], dtype=vertices.dtype, device=vertices.device).unsqueeze(1)
        print(flame_trans)
        vertices += flame_trans


        points, face_indices = sample_points(vertices, faces, num_samples=self.num_points)
        print("-"*30)
        print("Mesh Info")
        print("-"*30)
        print(f'vertices.shape: {vertices.shape}')
        print(f'faces.shape: {faces.shape}')
        print(f'faces.max(): {faces.max()}')
        print(f'vertices.shape[0]: {vertices.shape[0]}')
        print(f'points.shape: {points.shape}')
        print("-"*30)

        points = points.unsqueeze(1)

        with torch.no_grad():
            self.base_pos.data.copy_(points)
            self.base_scale.data.fill_(0.001)
            self.base_sh_dc.data.fill_(0.5)
            self.base_sh_rest.data.zero_()
            self.base_opacity.data.fill_(0.9)

    def forward(self, code, app_emb=None):
        """
        code: (B, N, 128)
        Returns:
            pos: (B, N, 3)
            quat: (B, N, 4)
            scale: (B, N, 3)
            color: (B, N, 3)
            opacity: (B, N, 1)
        """
        out = self.mlp(code)  # (B, N, 14)
        out = out.reshape(out.shape[0], out.shape[1], self.num_points, -1)

        delta_pos = out[..., :3] * 0.01
        # delta_quat = F.normalize(out[..., 3:7], dim=-1)
        delta_quat = out[..., 3:7] * 0.01
        delta_scale = out[..., 7:10]
        delta_sh_dc = out[..., 10:13]
        delta_sh_rest = out[..., 13:13 + 3 * self.max_rest_dim]
        delta_opacity = out[..., 13 + 3 * self.max_rest_dim:]

        if app_emb is not None:
            B, T = code.shape[:2]
            geom = self.app_geom_head(app_emb).view(B, 1, self.num_points, -1)
            sh   = self.app_sh_head(app_emb).view(B, 1, self.num_points, -1)

            app_pos_bias, app_scale_bias, app_opa_bias = torch.split(geom, [3, 3, 1], dim=-1)
            app_dc_bias = sh[..., :3]
            app_rest_bias = sh[..., 3:]

            app_pos_bias = app_pos_bias.expand(B, T, -1, -1)
            app_scale_bias = app_scale_bias.expand(B, T, -1, -1)
            app_opa_bias = app_opa_bias.expand(B, T, -1, -1)
            app_dc_bias = app_dc_bias.expand(B, T, -1, -1)
            app_rest_bias = app_rest_bias.expand(B, T, -1, -1)

            delta_pos = delta_pos + app_pos_bias * 0.01
            delta_scale = delta_scale + app_scale_bias
            delta_opacity = delta_opacity + app_opa_bias
            delta_sh_dc = delta_sh_dc + app_dc_bias
            delta_sh_rest = delta_sh_rest + app_rest_bias
            
        # Base parameters will be broadcasted to (B, N, D)
        pos = self.base_pos + delta_pos
        quat = F.normalize(self.base_quat + delta_quat, dim=-1)
        scale = self.base_scale * torch.exp(0.1 * delta_scale)
        sh_dc = self.base_sh_dc + delta_sh_dc
        sh_rest = self.base_sh_rest + delta_sh_rest
        opacity = torch.sigmoid(self.base_opacity + delta_opacity)

        # return pos, quat, scale, color, opacity
        return pos, quat, scale, sh_dc, sh_rest, opacity 

    def train_base(self):
        for name, p in self.named_parameters():
            if name.startswith("base_"):
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    def train_delta(self):
        for name, p in self.named_parameters():
            if name.startswith("base_"):
                p.requires_grad = False
            else:
                p.requires_grad = True

    def oneupSHdegree(self):
        if self.active_sh_degree == 0:
            with torch.no_grad():
                self.base_sh_dc.copy_(RGB2SH(self.base_sh_dc))
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.coeffs = (self.active_sh_degree + 1) ** 2
            print("active_sh_degree: ", self.active_sh_degree)
    
    def render(self, code, cam_matrix, cam_params, yaw_deg, sh_degree, app_emb=None):
        """
        Differentiable rendering with displacement decoder.

        Args:
            code: (B, N, 128)
            cam_matrix: (B, 3, 4)
            cam_params: dict with fx, fy, cx, cy, etc.

        Returns:
            images: (B, 3, H, W) - differentiable
        """
        #pos, quat, scale, color, opacity = self(code)
        pos, quat, scale, sh_dc, sh_rest, opacity = self(code, app_emb=app_emb)
        B, T, N = pos.shape[0], pos.shape[1], pos.shape[2]

        pos = pos.view(B*T, N, -1)
        quat = quat.view(B*T, N, -1)
        scale = scale.view(B*T, N, -1)
        sh_dc = sh_dc.view(B*T, N, 3)
        coeffs = (sh_degree + 1) ** 2
        sh_rest = sh_rest.view(B*T, N, 3, self.max_rest_dim)
        opacity = opacity.view(B*T, N, -1)

        coeffs = (sh_degree + 1) ** 2
        cur_rest_dim = coeffs - 1
        active_sh_rest = sh_rest[..., :cur_rest_dim]

        shs = torch.cat([sh_dc.unsqueeze(-1), active_sh_rest], dim=-1).permute(0, 1, 3, 2).contiguous()
        
        if sh_degree == 0:
            color_arg = sh_dc.view(B*T, N, 3)
            gs_params = {
                'xyz': pos,
                'rotations': quat,
                'scales': scale,
                'colors': color_arg,
                'opacities': opacity,
                'shs': None,
            }
        else:
            gs_params = {
                'xyz': pos,
                'rotations': quat,
                'scales': scale,
                'colors': None,
                'opacities': opacity,
                'shs': shs,
            }

        rendered = render_gaussian(gs_params, cam_matrix, cam_params=cam_params, yaw_deg=yaw_deg, sh_degree=sh_degree)  # differentiable
        #images = torch.flip(rendered['images'], dims=[-1, -2])  # (B, 3, H, W) Flip Y and X to match correct orientation
        images = rendered['images'].view(B, T, *rendered['images'].shape[1:])

        return images

    def save_image(self, gt, pred, epoch, save_root="results", tag=None):
        """
        Save side-by-side GT and predicted image as PNG.

        Args:
            gt: (3, H, W) float tensor in [0, 1]
            pred: (3, H, W) float tensor in [0, 1]
            epoch: int, current epoch number (used in file name)
            save_root: top-level directory (default: "results")
        """
        def tensor_to_img(x):
            return (x.detach().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

        concat_np = None
        for i in range(gt.shape[0]):
            _concat_np = None
            for j in range(gt.shape[1]):
                gt_np = tensor_to_img(gt[i, j])
                pred_np = tensor_to_img(pred[i, j])
                if _concat_np is None: _concat_np = np.concatenate([gt_np, pred_np], axis=0)  # (2H, W, 3)'
                else: _concat_np = np.concatenate([_concat_np, np.concatenate([gt_np, pred_np], axis=0)], axis=1)  # (H, 2W, 3)
            if concat_np is None: concat_np = _concat_np
            else: concat_np = np.concatenate([concat_np, _concat_np], axis=0)  # (2H, W, 3)
        img = Image.fromarray(concat_np)
        
        if self._save_image_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._save_image_dir = os.path.join(save_root, timestamp)
            os.makedirs(self._save_image_dir, exist_ok=True)
        
        if tag:
            fname = f"{epoch:05d}_{tag}.png"
        else:
            fname = f"{epoch:05d}.png"
        
        save_path = os.path.join(self._save_image_dir, fname)
        img.save(save_path)
    
    
    def render_base(self):
        import torch
        import numpy as np
        import imageio
        from torchvision.utils import save_image

        # 1. Load 25D camera
        json_path = hp.single_cam_path
        cam_matrix, cam_params = load_cam(json_path)  # cam_matrix: (1, 3, 4)
        
        print('cam_matrix: ', cam_matrix, ' cam_params: ', cam_params)
        
        # # === Load camera parameters saved from SMIRK ===
        # cam_param_dir = "/source/Hyeonho/Research/MonoFace/submodules/smirk/results/cam_params"
        # cam_matrix = torch.load(os.path.join(cam_param_dir, "cam_matrix.pt")).to(self.base_pos.device)
        # cam_params = torch.load(os.path.join(cam_param_dir, "cam_params.pt"))

        # print("Loaded camera_matrix from SMIRK:", cam_matrix)
        # print("Loaded cam_params from SMIRK:", cam_params)
        
        # device = self.base_pos.device
        # cam_matrix = cam_matrix.to(device)        

        # import scipy.io
        # mat_path = "/source/Hyeonho/Research/MonoFace/submodules/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/input_image/epoch_20_000000/39798.mat"
        # data = scipy.io.loadmat(mat_path)
        # angle = data['angle'].squeeze()
        # trans = data['trans'].squeeze()
        # T = torch.tensor(trans, dtype=self.base_pos.dtype).cuda()

        # with torch.no_grad():
        #     self.base_pos.data = self.base_pos.data + T.unsqueeze(0).unsqueeze(0)

        coeffs = (self.sh_degree + 1) ** 2
        f_rest = self.base_sh_rest.view(-1, 3, coeffs - 1)
        shs = torch.cat([self.base_sh_dc.unsqueeze(-1), f_rest], dim=-1)

        gs_params = {
            'xyz': self.base_pos,               # (B, N, 3)
            'shs': shs,          # (B, N, 3)
            'opacities': self.base_opacity,     # (B, N, 1)
            'scales': self.base_scale,          # (B, N, 3)
            'rotations': self.base_quat,        # (B, N, 4)
        }        

        print("Checking gs_params:")
        for k, v in gs_params.items():
            print(f"{k}: {v.shape}, nan={torch.isnan(v).any().item()}, inf={torch.isinf(v).any().item()}, min={v.min().item()}, max={v.max().item()}")
        
        # Render
        yaw_deg = {(i * (360 / 32)):f"view{i:02d}" for i in range(32)}
        for yaw in yaw_deg.keys():
            render_out = render_gaussian(gs_params, cam_matrix, cam_params, yaw_deg=yaw)
            images  = render_out["images"]
            cam_pos = render_out["cam_pos"]
            view_mat= render_out["view_mat"]
            fovx = render_out["fovx"];  fovy = render_out["fovy"]

            output_dir = './output'
            os.makedirs(output_dir, exist_ok=True)

            batch_size = cam_matrix.shape[0]
            for i in range(batch_size):
                image = images[i, :3]  # Take RGB channels
                image = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu()  # (H, W, 3)
                img = Image.fromarray(image.numpy())

                from datetime import datetime

                now = datetime.now()
                fname = now.strftime("%Y%m%d_%H%M%S")
                
                img.save(f'{output_dir}/{fname}_{i}_{yaw_deg[yaw]}.png')

            print("Rendering complete and saved to ./output/")
            visualize_camera_and_points(gs_params["xyz"],
                                cam_pos, view_mat,
                                fovx, fovy,
                                save_path=f"./output/camera_fov_correct_{yaw_deg[yaw]}.png")

    @torch.no_grad()
    def save_multiview_gtpred_video(preds, gts, save_dir, basename, fps=25, save_frames=False):
        """
        preds: (V, T, 3, H, W) in [0,1]  (루프 끝의 preds)
        gts  : (1, V, T, 3, H, W) or (V, T, 3, H, W)
        """
        assert preds.dim() == 5, f"preds shape must be (V,T,3,H,W), got {tuple(preds.shape)}"
        if gts.dim() == 6: gts = gts[0]  # (V,T,3,H,W)
        assert gts.shape[:3] == preds.shape[:3], f"gt/pred mismatch: gt{tuple(gts.shape)} pred{tuple(preds.shape)}"

        V, T, C, H, W = preds.shape
        os.makedirs(save_dir, exist_ok=True)
        video_path = os.path.join(save_dir, f"{basename}.mp4")
        frames_dir = os.path.join(save_dir, f"{basename}_frames")
        if save_frames: os.makedirs(frames_dir, exist_ok=True)

        def to_u8(img):  # (3,H,W)->(H,W,3)
            return (img.clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()

        writer = imageio.get_writer(video_path, fps=fps)
        try:
            for t in range(T):
                rows = []
                for v in range(V):
                    gt_f   = to_u8(gts[v, t])
                    pred_f = to_u8(preds[v, t])
                    row = np.concatenate([gt_f, pred_f], axis=1)   # (H, 2W, 3)
                    rows.append(row)
                frame = np.concatenate(rows, axis=0)                # (V*H, 2W, 3)
                writer.append_data(frame)
                if save_frames:
                    Image.fromarray(frame).save(os.path.join(frames_dir, f"{t:05d}.png"))
        finally:
            writer.close()
        return {"video_path": video_path, "frames_dir": frames_dir if save_frames else None, "V": V, "T": T}

    @property
    def sh_degree(self):
        return self.active_sh_degree

    @torch.no_grad()
    def export_gaussians(self, code: torch.Tensor, step_idx: int = 0):
        """
        현재 학습된 상태에서, 주어진 코드에 따라 Gaussian 정보 추출.

        Args:
            code: (B, T, D) Tensor (ex: disp_code)
            step_idx: 인퍼런스에 사용할 time step index (default=0)
        """
        if code is None:
            raise ValueError("export_gaussians requires `code` input (deformation code).")

        if code.dim() != 3:
            raise ValueError(f"Expected code shape (B, T, D), but got {code.shape}")

        B, T, D = code.shape
        b_idx = 0
        step_idx = int(step_idx) % T
        code_one = code[b_idx, step_idx].view(1, 1, -1)

        # Forward pass
        pos, quat, scale, sh_dc, sh_rest, opacity = self.forward(code_one)

        means = pos[0, 0]
        scales = scale[0, 0]
        rotations = quat[0, 0]
        opacity = opacity[0, 0]

        if self.sh_degree == 0:
            dc = sh_dc[0, 0]
            sh = dc.unsqueeze(-1)
        else:
            coeffs = (self.sh_degree + 1) ** 2
            N = sh_rest.shape[2]
            rest_dim = coeffs - 1
            flat = sh_rest[0, 0]
            flat = flat[:, :3*rest_dim].contiguous()
            f_rest = flat.view(N, 3, rest_dim)
            sh = torch.cat([sh_dc[0, 0].unsqueeze(-1), f_rest], dim=-1)

        return {
            "means": means,
            "scales": scales,
            "rotations": rotations,
            "opacity": opacity,
            "sh": sh,
            "sh_degree": self.sh_degree
        }

class DisplacementDecoder_orig(nn.Module):
    def __init__(self, code_dim=hp.emb_dim, num_points=hp.num_points, sh_degree=3):
        super().__init__()
        self.num_points = num_points

        out_dim = 3 + 4 + 3 + 3 + 1 # pos, quat, scale, sh_dc, sh_rest, opacity
        self.mlp = nn.Sequential(
            nn.Linear(code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_points * out_dim)  
        )

        # Learnable base parameters (1, N, D)
        self.register_parameter("base_pos", nn.Parameter(torch.zeros(1, 1, num_points, 3)))
        self.register_parameter("base_quat", nn.Parameter(F.normalize(torch.rand(1, 1, num_points, 4), dim=-1)))
        self.register_parameter("base_scale", nn.Parameter(torch.ones(1, 1, num_points, 3) * 0.005))
        # self.register_parameter("base_sh_dc", nn.Parameter(torch.ones(1, 1, num_points, 3) * 0.5))
        # self.register_parameter("base_sh_rest", nn.Parameter(torch.zeros(1, 1, num_points, 3 * (self.coeffs - 1))))
        self.register_parameter("base_color", nn.Parameter(torch.ones(1, 1, num_points, 3) * 0.5))
        self.register_parameter("base_opacity", nn.Parameter(torch.ones(1, 1, num_points, 1) * 0.5))

        self._save_image_dir = None

    def initialize(self, mesh):
        vertices, faces = load_obj_as_mesh(mesh)
        vertices = vertices.cuda().unsqueeze(0)
        faces = faces.cuda()

        ##############
        import random
        import numpy as np
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # <- GPU 기반 샘플링에도 필요

        # #change
        # ############### === Load FLAME vertices from SMIRK ===
        # print(vertices.shape)
        # vertices_path = "/source/Hyeonho/Research/MonoFace/submodules/smirk/results/cam_params/flame_vertices.pt"
        # vertices = torch.load(vertices_path).cuda()  # shape: (1, N, 3)
        # print(vertices.shape)
        # exit()

        # print(vertices.mean(dim=1))
        # vertices -= vertices.mean(dim=1).unsqueeze(1) # alignment
        flame_trans = torch.tensor([[ -0.00888654, -0.00508765,  -0.26269796]], dtype=vertices.dtype, device=vertices.device).unsqueeze(1)
        print(flame_trans)
        vertices += flame_trans


        points, face_indices = sample_points(vertices, faces, num_samples=self.num_points)
        print("-"*30)
        print("Mesh Info")
        print("-"*30)
        print(f'vertices.shape: {vertices.shape}')
        print(f'faces.shape: {faces.shape}')
        print(f'faces.max(): {faces.max()}')
        print(f'vertices.shape[0]: {vertices.shape[0]}')
        print(f'points.shape: {points.shape}')
        print("-"*30)

        points = points.unsqueeze(1)

        with torch.no_grad():
            self.base_pos.data.copy_(points)
            self.base_scale.data.fill_(0.001)
            # self.base_sh_dc.data.fill_(0.5)
            # self.base_sh_rest.data.zero_()
            self.base_color.data.fill_(0.5)
            self.base_opacity.data.fill_(0.9)
    
    def render(self, code, cam_matrix, cam_params, yaw_deg):
        """
        Differentiable rendering with displacement decoder.

        Args:
            code: (B, N, 128)
            cam_matrix: (B, 3, 4)
            cam_params: dict with fx, fy, cx, cy, etc.

        Returns:
            images: (B, 3, H, W) - differentiable
        """
        pos, quat, scale, color, opacity = self(code)
        # pos, quat, scale, sh_dc, sh_rest, opacity = self(code)
        # B, T, N = pos.shape[0], pos.shape[1], pos.shape[2]

        # pos = pos.view(B*T, N, -1)
        # quat = quat.view(B*T, N, -1)
        # scale = scale.view(B*T, N, -1)
        # # color = color.view(B*T, N, -1)
        # sh_dc = sh_dc.view(B*T, N, 3)
        # sh_rest = sh_rest.view(B*T, N, 3, self.coeffs - 1)
        # opacity = opacity.view(B*T, N, -1)

        B = pos.shape[0]
        T = pos.shape[1]
        pos = pos.view(B*T, *pos.shape[2:])
        quat = quat.view(B*T, *quat.shape[2:])
        scale = scale.view(B*T, *scale.shape[2:])
        color = color.view(B*T, *color.shape[2:])
        opacity = opacity.view(B*T, *opacity.shape[2:])

        # shs = torch.cat([sh_dc.unsqueeze(-1), sh_rest], dim=-1)

        gs_params = {
            'xyz': pos,
            'rotations': quat,
            'scales': scale,
            'colors': color,
            'opacities': opacity,
            # 'shs': shs,
        }

        rendered = render_gaussian(gs_params, cam_matrix, cam_params=cam_params, yaw_deg=yaw_deg)  # differentiable
        #images = torch.flip(rendered['images'], dims=[-1, -2])  # (B, 3, H, W) Flip Y and X to match correct orientation
        images = rendered['images'].view(B, T, *rendered['images'].shape[1:])

        return images

    def save_image(self, gt, pred, epoch, save_root="results"):
        """
        Save side-by-side GT and predicted image as PNG.

        Args:
            gt: (3, H, W) float tensor in [0, 1]
            pred: (3, H, W) float tensor in [0, 1]
            epoch: int, current epoch number (used in file name)
            save_root: top-level directory (default: "results")
        """
        def tensor_to_img(x):
            return (x.detach().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

        concat_np = None
        for i in range(gt.shape[0]):
            _concat_np = None
            for j in range(gt.shape[1]):
                gt_np = tensor_to_img(gt[i, j])
                pred_np = tensor_to_img(pred[i, j])
                if _concat_np is None: _concat_np = np.concatenate([gt_np, pred_np], axis=0)  # (2H, W, 3)'
                else: _concat_np = np.concatenate([_concat_np, np.concatenate([gt_np, pred_np], axis=0)], axis=1)  # (H, 2W, 3)
            if concat_np is None: concat_np = _concat_np
            else: concat_np = np.concatenate([concat_np, _concat_np], axis=0)  # (2H, W, 3)
        img = Image.fromarray(concat_np)
        
        if self._save_image_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._save_image_dir = os.path.join(save_root, timestamp)
            os.makedirs(self._save_image_dir, exist_ok=True)
        
        fname = f"{epoch:05d}.png"
        
        save_path = os.path.join(self._save_image_dir, fname)
        img.save(save_path)

    def forward(self, code):
        """
        code: (B, N, 128)
        Returns:
            pos: (B, N, 3)
            quat: (B, N, 4)
            scale: (B, N, 3)
            color: (B, N, 3)
            opacity: (B, N, 1)
        """
        out = self.mlp(code)  # (B, N, 14)
        out = out.reshape(out.shape[0], out.shape[1], self.num_points, -1)

        delta_pos = torch.tanh(out[..., :3]) * 0.01
        delta_quat = F.normalize(out[..., 3:7], dim=-1)
        # delta_quat = 0.01 * torch.tanh(out[..., 3:7])
        delta_scale = torch.clamp(out[..., 7:10], -3, 3)

        delta_color = out[..., 10:13]
        # delta_sh_dc = out[..., 10:13]
        # delta_sh_rest = out[..., 13:13 + 3 * (self.coeffs - 1)]

        delta_opacity = out[..., 13:14]
        # delta_opacity = out[..., 13 + 3 * (self.coeffs - 1):]

        # Base parameters will be broadcasted to (B, N, D)
        pos = self.base_pos + delta_pos
        quat = F.normalize(self.base_quat + delta_quat, dim=-1)
        scale = self.base_scale * torch.exp(0.1 * delta_scale)
        color = torch.sigmoid(self.base_color + delta_color)
        # sh_dc = self.base_sh_dc + delta_sh_dc
        # sh_rest = self.base_sh_rest + delta_sh_rest
        opacity = torch.sigmoid(self.base_opacity + delta_opacity)

        return pos, quat, scale, color, opacity
        # return self.base_pos.repeat(1, 4, 1, 1), self.base_quat.repeat(1, 4, 1, 1), self.base_scale.repeat(1, 4, 1, 1), self.base_color.repeat(1, 4, 1, 1), self.base_opacity.repeat(1, 4, 1, 1)
        # return pos, quat, scale, sh_dc, sh_rest, opacity 
    
    def render_base(self):
        import torch
        import numpy as np
        import imageio
        from torchvision.utils import save_image

        # 1. Load 25D camera
        json_path = hp.single_cam_path
        cam_matrix, cam_params = load_cam(json_path)  # cam_matrix: (1, 3, 4)
        
        print('cam_matrix: ', cam_matrix, ' cam_params: ', cam_params)
        
        # # === Load camera parameters saved from SMIRK ===
        # cam_param_dir = "/source/Hyeonho/Research/MonoFace/submodules/smirk/results/cam_params"
        # cam_matrix = torch.load(os.path.join(cam_param_dir, "cam_matrix.pt")).to(self.base_pos.device)
        # cam_params = torch.load(os.path.join(cam_param_dir, "cam_params.pt"))

        # print("Loaded camera_matrix from SMIRK:", cam_matrix)
        # print("Loaded cam_params from SMIRK:", cam_params)
        
        # device = self.base_pos.device
        # cam_matrix = cam_matrix.to(device)        

        # import scipy.io
        # mat_path = "/source/Hyeonho/Research/MonoFace/submodules/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/input_image/epoch_20_000000/39798.mat"
        # data = scipy.io.loadmat(mat_path)
        # angle = data['angle'].squeeze()
        # trans = data['trans'].squeeze()
        # T = torch.tensor(trans, dtype=self.base_pos.dtype).cuda()

        # with torch.no_grad():
        #     self.base_pos.data = self.base_pos.data + T.unsqueeze(0).unsqueeze(0)

        gs_params = {
            'xyz': self.base_pos,               # (B, N, 3)
            'colors': self.base_color,          # (B, N, 3)
            'opacities': self.base_opacity,     # (B, N, 1)
            'scales': self.base_scale,          # (B, N, 3)
            'rotations': self.base_quat,        # (B, N, 4)
        }        

        print("Checking gs_params:")
        for k, v in gs_params.items():
            print(f"{k}: {v.shape}, nan={torch.isnan(v).any().item()}, inf={torch.isinf(v).any().item()}, min={v.min().item()}, max={v.max().item()}")
        
        # Render
        yaw_deg = {(i * (360 / 32)):f"view{i:02d}" for i in range(32)}
        for yaw in yaw_deg.keys():
            render_out = render_gaussian(gs_params, cam_matrix, cam_params, yaw_deg=yaw)
            images  = render_out["images"]
            cam_pos = render_out["cam_pos"]
            view_mat= render_out["view_mat"]
            fovx = render_out["fovx"];  fovy = render_out["fovy"]

            output_dir = './output'
            os.makedirs(output_dir, exist_ok=True)

            batch_size = cam_matrix.shape[0]
            for i in range(batch_size):
                image = images[i, :3]  # Take RGB channels
                image = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu()  # (H, W, 3)
                img = Image.fromarray(image.numpy())

                from datetime import datetime

                now = datetime.now()
                fname = now.strftime("%Y%m%d_%H%M%S")
                
                img.save(f'{output_dir}/{fname}_{i}_{yaw_deg[yaw]}.png')

            print("Rendering complete and saved to ./output/")
            visualize_camera_and_points(gs_params["xyz"],
                                cam_pos, view_mat,
                                fovx, fovy,
                                save_path=f"./output/camera_fov_correct_{yaw_deg[yaw]}.png")

    @torch.no_grad()
    def export_gaussians(self, mode: str = "base", code: torch.Tensor = None, step_idx: int = 0):
        if mode not in ("base", "deformed"):
            raise ValueError(f"mode must be 'base or 'deformed', got {mode}")

        if mode == "base":
            means = self.base_pos[0, 0]
            scales = self.base_scale[0, 0]
            rotations = self.base_quat[0, 0]
            opacity = self.base_opacity[0, 0]
            rgb = self.base_color[0, 0]
            return {
                "means": means, "scales": scales, "rotations": rotations,
                "opacity": opacity, "rgb": rgb
            }
        
        if code is None:
            raise ValueError
        if code.dim() != 3:
            raise ValueError
        
        B, T, D = code.shape
        b_idx = 0
        step_idx = int(step_idx) % T
        code_one = code[b_idx, step_idx].view(1, 1, -1)

        pos, quat, scale, color, opacity = self.forward(code_one)
        means = pos[0, 0]
        scales = scale[0, 0]
        rotations = quat[0, 0]
        opacity = opacity[0, 0]
        rgb = color[0, 0]

        return {
            "means": means, "scales": scales, "rotations": rotations,
            "opacity": opacity, "rgb": rgb
        }

def _logit(p, eps=1e-6):
    p = torch.clamp(torch.as_tensor(p).float(), eps, 1-eps)
    return torch.log(p) - torch.log1p(-p)


class IDMapper(nn.Module):
    def __init__(self, dim=768, width=1024, depth=2, dropout=0.0, normalize_out=True):
        super().__init__()
        layers = []
        d_in = dim
        for i in range(depth):
            layers += [nn.Linear(d_in, width), nn.GELU(), nn.Dropout(dropout)]
            d_in = width
        layers += [nn.Linear(d_in, dim)]
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(dim)
        self.normalize_out = normalize_out
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        y = self.net(self.ln(x))
        y = x + 0.5 * y
        if self.normalize_out:
            y = F.normalize(y, dim=-1, eps=1e-6)
        return y


################### MV Generator ###################
def fourier_pe_deg(yaw_deg, L=6):  # yaw_deg: (B,V)
    yaw = torch.deg2rad(yaw_deg.float())
    freqs = torch.tensor([2**k for k in range(L)], device=yaw.device).float()
    s = torch.sin(yaw[...,None]*freqs); c = torch.cos(yaw[...,None]*freqs)
    return torch.cat([s,c], dim=-1)  # (B,V,2L)

class AdaIN(nn.Module):
    def __init__(self, ch, cond_dim): super().__init__(); self.aff = nn.Linear(cond_dim, ch*2)
    def forward(self, x, cond):
        g,b = self.aff(cond).chunk(2, dim=1); g,b = g[...,None,None], b[...,None,None]
        x = F.group_norm(x, 32); return x*(1+g) + b

class Block(nn.Module):
    def __init__(self, ci, co, cond_dim):
        super().__init__()
        # self.c1 = nn.Conv2d(ci, co, 3,1,1); self.n1 = AdaIN(co, cond_dim) # NHH Revise
        # self.c2 = nn.Conv2d(co, co, 3,1,1); self.n2 = AdaIN(co, cond_dim) # NHH Revise
        self.c1 = nn.Conv2d(ci, co, 3,1,1, bias=False); self.n1 = AdaIN(co, cond_dim)
        self.c2 = nn.Conv2d(co, co, 3,1,1, bias=False); self.n2 = AdaIN(co, cond_dim)
    def forward(self, x, c):
        x = self.c1(x); x = self.n1(x,c); x = F.silu(x)
        x = self.c2(x); x = self.n2(x,c); x = F.silu(x); return x

class MVGenMVP(nn.Module):
    """ 입력(B,V,3,H,W) -> 출력(B,V,3,H,W); cond=z_id||z_app(1536), view_pe(각도조건) """
    def __init__(self, base=64, cond_dim=1536, view_dim=12, views=8):
        super().__init__()
        self.views = views
        self.view_proj = nn.Linear(view_dim, view_dim)
        self.cond_proj = nn.Linear(cond_dim + view_dim, cond_dim)
        C = base
        self.e1 = Block(3,   C,   cond_dim); self.d1 = nn.Conv2d(C,   C*2,4,2,1)
        self.e2 = Block(C*2, C*2, cond_dim); self.d2 = nn.Conv2d(C*2, C*4,4,2,1)
        self.mid= Block(C*4, C*4, cond_dim)
        self.u2 = nn.ConvTranspose2d(C*4, C*2,4,2,1); self.g2 = Block(C*4, C*2, cond_dim)
        self.u1 = nn.ConvTranspose2d(C*2, C,  4,2,1); self.g1 = Block(C*2, C,   cond_dim)
        self.out= nn.Conv2d(C, 3, 3,1,1)
        self.to(memory_format=torch.channels_last) # NHH Revise
    def forward(self, x, cond, view_pe):
        B,V,C,H,W = x.shape; assert V==self.views
        # x = x.view(B*V, C, H, W)
        x = x.contiguous(memory_format=torch.channels_last).view(B*V, C, H, W) # NHH Revise
        v = self.view_proj(view_pe).reshape(B*V, -1)
        c = self.cond_proj(torch.cat([cond[:,None,:].expand(B,V,-1).reshape(B*V,-1), v], dim=1))
        e1 = self.e1(x,c); d1 = self.d1(e1)
        e2 = self.e2(d1,c); d2 = self.d2(e2)
        m  = self.mid(d2,c)
        u2 = self.u2(m);  g2 = self.g2(torch.cat([u2,e2],1), c)
        u1 = self.u1(g2); g1 = self.g1(torch.cat([u1,e1],1), c)
        y  = self.out(g1).tanh()
        return y.view(B,V,3,H,W)  # [-1,1]
    

class UVDecoder2(nn.Module):
    """
    Builds learnable UV-space base maps from a texture and an OBJ (with UVs),
    predicts delta maps conditioned on a code, and outputs final maps as base + delta.

    Final channel layout per pixel:
        pos(3), quat(4), scale(3), sh_dc(3), sh_rest(3*(max_coeffs-1)), opacity(1)

    No expression code input
    Only id emb + appearance emb
    """
    def __init__(self, texture_path: str, obj_path: str,
                 code_dim=hp.emb_dim, num_points=hp.num_points,
                 init_sh_degree=0, max_sh_degree=3):
        super().__init__()

        # Texture (float32 in [0,1])
        tex = convert_image_dtype(read_image(texture_path, mode=ImageReadMode.RGB), dtype=torch.float32)
        self.tex = tex
        self.H, self.W = int(tex.shape[1]), int(tex.shape[2])
        id_dim = getattr(hp, "id_dim")
        app_dim = getattr(hp, "app_dim")
        if app_dim is None:
            app_dim = int(getattr(hp, "app_dim", 768))
        self.app_dim = app_dim

        # Minimal OBJ parser and UV rasterizer (raw XYZ → UV position map)
        def _parse_obj(path: str):
            V, VT, F_v, F_vt = [], [], [], []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line or line.startswith("#"): continue
                    t = line.strip().split()
                    if not t: continue
                    if t[0] == "v" and len(t) >= 4:
                        V.append([float(t[1]), float(t[2]), float(t[3])])
                    elif t[0] == "vt" and len(t) >= 3:
                        VT.append([float(t[1]), float(t[2])])
                    elif t[0] == "f" and len(t) >= 4:
                        def parse_face_token(tok):
                            parts = tok.split("/")
                            v_idx = int(parts[0]) if parts[0] else 0
                            vt_idx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                            return v_idx, vt_idx
                        vts = [parse_face_token(tok) for tok in t[1:]]
                        for i in range(1, len(vts) - 1):
                            a, b, c = vts[0], vts[i], vts[i+1]
                            F_v.append([a[0]-1, b[0]-1, c[0]-1])
                            F_vt.append([a[1]-1, b[1]-1, c[1]-1])
            V = np.asarray(V, dtype=np.float32)
            VT = np.asarray(VT, dtype=np.float32)
            F_v = np.asarray(F_v, dtype=np.int32)
            F_vt = np.asarray(F_vt, dtype=np.int32)
            if VT.size == 0 or (F_vt < 0).any():
                raise ValueError("OBJ missing UVs (vt) or face UV indices.")
            return V, F_v, VT, F_vt

        def _rasterize_uv_position(verts_raw, faces, uvs01, faces_uv, H, W):
            img = np.zeros((H, W, 3), dtype=np.float32)
            zbuf = np.full((H, W), -np.inf, dtype=np.float32)
            UV = uvs01.copy()
            UV[:, 0] = UV[:, 0] * (W - 1)
            UV[:, 1] = (1.0 - UV[:, 1]) * (H - 1)
            F = faces.shape[0]
            for f in range(F):
                vi = faces[f]; ti = faces_uv[f]
                if (ti < 0).any(): continue
                tri_uv = UV[ti]; tri_p = verts_raw[vi]
                xmin = max(int(np.floor(tri_uv[:,0].min())), 0)
                xmax = min(int(np.ceil (tri_uv[:,0].max())), W-1)
                ymin = max(int(np.floor(tri_uv[:,1].min())), 0)
                ymax = min(int(np.ceil (tri_uv[:,1].max())), H-1)
                if xmax < xmin or ymax < ymin: continue
                p0, p1, p2 = tri_uv[0], tri_uv[1], tri_uv[2]
                area = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])
                if abs(area) < 1e-8: continue
                for y in range(ymin, ymax+1):
                    py = y + 0.5
                    for x in range(xmin, xmax+1):
                        px = x + 0.5
                        w0 = ((p1[0]-px)*(p2[1]-py) - (p2[0]-px)*(p1[1]-py)) / area
                        w1 = ((p2[0]-px)*(p0[1]-py) - (p0[0]-px)*(p2[1]-py)) / area
                        w2 = 1.0 - w0 - w1
                        if (w0 >= -1e-6) and (w1 >= -1e-6) and (w2 >= -1e-6):
                            p = w0*tri_p[0] + w1*tri_p[1] + w2*tri_p[2]
                            z = p[2]
                            if z >= zbuf[y, x]:
                                img[y, x, :] = p
                                zbuf[y, x] = z
            return img

        def _rasterize_uv_position_fast(verts_raw, faces, uvs01, faces_uv, H, W):
            import numpy as np
            img  = np.ones((H, W, 3), dtype=np.float32)/2.0
            zbuf = np.full((H, W), -np.inf, dtype=np.float32)

            UV = uvs01.copy().astype(np.float32)
            UV[:, 0] = UV[:, 0] * (W - 1)
            UV[:, 1] = (1.0 - UV[:, 1]) * (H - 1)

            F = faces.shape[0]
            for f in range(F):
                vi = faces[f]; ti = faces_uv[f]
                if (ti < 0).any(): 
                    continue

                tri_uv = UV[ti]            # (3,2)
                tri_p  = verts_raw[vi]     # (3,3)

                xmin = max(int(np.floor(tri_uv[:,0].min())), 0)
                xmax = min(int(np.ceil (tri_uv[:,0].max())), W-1)
                ymin = max(int(np.floor(tri_uv[:,1].min())), 0)
                ymax = min(int(np.ceil (tri_uv[:,1].max())), H-1)
                if xmax < xmin or ymax < ymin: 
                    continue

                p0, p1, p2 = tri_uv[0], tri_uv[1], tri_uv[2]
                area = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])
                if abs(area) < 1e-8:
                    continue

                xs = np.arange(xmin, xmax+1, dtype=np.float32)
                ys = np.arange(ymin, ymax+1, dtype=np.float32)
                xx, yy = np.meshgrid(xs + 0.5, ys + 0.5)  # (h,w)

                w0 = ((p1[0]-xx)*(p2[1]-yy) - (p2[0]-xx)*(p1[1]-yy)) / area
                w1 = ((p2[0]-xx)*(p0[1]-yy) - (p0[0]-xx)*(p2[1]-yy)) / area
                w2 = 1.0 - w0 - w1
                mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
                if not mask.any():
                    continue

                p_interp = (w0[...,None]*tri_p[0] + w1[...,None]*tri_p[1] + w2[...,None]*tri_p[2])  # (h,w,3)
                z = p_interp[..., 2]

                ys_idx, xs_idx = np.where(mask)
                yi = (ys_idx + ymin).astype(np.int32)
                xi = (xs_idx + xmin).astype(np.int32)

                z_old = zbuf[yi, xi]
                z_new = z[ys_idx, xs_idx]
                keep  = z_new >= z_old
                if not np.any(keep):
                    continue

                yi = yi[keep]; xi = xi[keep]
                vals = p_interp[ys_idx[keep], xs_idx[keep], :]
                img[yi, xi, :]  = vals
                zbuf[yi, xi]    = z_new[keep]
            
            valid = (zbuf > -np.inf).astype(np.float32)
            return img, valid

        # Bake UV position map
        verts_np, faces_np, uvs_np, faces_uv_np = _parse_obj(obj_path)
        verts = torch.from_numpy(verts_np).float()
        faces = torch.from_numpy(faces_np.astype(np.int64))
        uvs   = torch.from_numpy(uvs_np).float()
        faces_uv = torch.from_numpy(faces_uv_np.astype(np.int64))

        # Optional FLAME-style translation
        flame_trans = -verts.mean(dim=0)
        verts_trans = verts + flame_trans

        pitch0_deg = float(getattr(hp, "init_pitch_deg", 8.0)) 
        theta = torch.deg2rad(torch.tensor(pitch0_deg))
        Rx = torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta), -torch.sin(theta)],
                        [0, torch.sin(theta),  torch.cos(theta)]], dtype=torch.float32)
        verts_trans = (verts_trans @ Rx.T)
        pos_map_np, valid_np = _rasterize_uv_position_fast(
            verts_trans.cpu().numpy(), faces.cpu().numpy(),
            uvs.cpu().numpy(), faces_uv.cpu().numpy(),
            self.H, self.W
        )
        pos_map = torch.from_numpy(pos_map_np).permute(2, 0, 1).contiguous()  # (3,H,W)
        uv_valid = torch.from_numpy(valid_np).unsqueeze(0).contiguous()
        self.register_buffer("uv_valid", uv_valid)

        # SH configuration
        self.num_points = int(num_points)
        self.active_sh_degree = int(init_sh_degree)
        self.max_sh_degree = int(max_sh_degree)
        self.max_coeffs = (self.max_sh_degree + 1) ** 2
        self.max_rest_dim = self.max_coeffs - 1  # exclude DC

        self.register_parameter("img_position", nn.Parameter(pos_map))
        quat0 = torch.zeros(4, self.H, self.W); quat0[3].fill_(1.0)
        self.register_parameter("img_quat", nn.Parameter(quat0))
        self.s_min = float(getattr(hp, "scale_min", 2e-4))
        self.s_max = float(getattr(hp, "scale_max", 0.004))
        init_scale = float(getattr(hp, "init_scale", 6e-4))      # [s_min, s_max] 내부

        p_scale = (init_scale - self.s_min) / (self.s_max - self.s_min)  # 0~1로 정규화
        scale_logit0 = _logit(p_scale)
        self.register_parameter(
            "img_scale",
            nn.Parameter(torch.full((3, self.H, self.W), float(scale_logit0)))
        )
        self.register_parameter("img_sh_dc", nn.Parameter(tex))
        self.register_parameter("img_sh_rest", nn.Parameter(torch.zeros(3 * self.max_rest_dim, self.H, self.W)))
        init_opa = float(getattr(hp, "init_opacity", 0.20))
        opa_logit0 = _logit(init_opa)
        self.register_parameter("img_opacity", nn.Parameter(opa_logit0 * uv_valid))  

        # Delta U-Net (in/out channels = total base channels)
        self.C_base = 3 + 4 + 3 + 3 + 3 * self.max_rest_dim + 1
        
        self.delta_id  = ConditionalUNetDelta(in_ch=self.C_base, out_ch=self.C_base, z_dim=id_dim, base=64, depth=2)
        self.delta_app = ConditionalUNetDelta(in_ch=self.C_base, out_ch=self.C_base, z_dim=app_dim, base=64, depth=2)
        
        self.w_id  = nn.Parameter(torch.tensor(1.0))
        self.w_app = nn.Parameter(torch.tensor(1.0))

        self._save_image_dir = None
        self._delta_enabled = True

        self._prev_rest_dim = (self.active_sh_degree + 1)**2 - 1
        self._curr_rest_dim = self._prev_rest_dim
        self._zero_new_rest_once = False
        self._global_step = 0

    def enable_delta(self, flag: bool):
        self._delta_enabled = bool(flag)
    
    def is_delta_enabled(self) -> bool:
        return getattr(self, "_delta_enabled", True)
    
    def set_global_step(self, step: int):
        self._global_step = int(step)

    # -------------------------------------------------------------------------
    # Forward: export_UV -> export_gaussian_splats
    # -------------------------------------------------------------------------
    def forward(self, code_id, code_app, return_aux: bool = False):
        out = self.export_UV(
            code_id, code_app, delta_down=16, t_chunk=1, return_delta=return_aux
        )
        if return_aux:
            pos, quat, scale, sh_dc, sh_rest, opacity, aux = out
        else:
            pos, quat, scale, sh_dc, sh_rest, opacity = out

        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = self.export_gaussian_splats(
            pos, quat, scale, sh_dc, sh_rest, opacity,
            N=self.num_points, weighted=True, replacement=False, temperature=0.7
        )

        if return_aux:
            aux_maps = {"uv_pos": pos, "uv_scale": scale, "uv_opacity": opacity,
                        "d_id": aux["d_id"], "d_app": aux["d_app"]}
            return (s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac), aux_maps

        return s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac
    
    def _quat_mul(self, qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
        # qa ⊗ qb, 채널 순서 (x,y,z,w)
        ax, ay, az, aw = qa[:,0:1], qa[:,1:2], qa[:,2:3], qa[:,3:4]
        bx, by, bz, bw = qb[:,0:1], qb[:,1:2], qb[:,2:3], qb[:,3:4]
        x = aw*bx + ax*bw + ay*bz - az*by
        y = aw*by - ax*bz + ay*bw + az*bx
        z = aw*bz + ax*by - ay*bx + az*bw
        w = aw*bw - ax*bx - ay*by - az*bz
        return torch.cat([x, y, z, w], dim=1)

    def _rotvec_to_quat(self, rv: torch.Tensor) -> torch.Tensor:
        # rv: (B,3,H,W) 축-각(라디안) 벡터 → quat(x,y,z,w)
        angle = rv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        axis  = rv / angle
        half  = 0.5 * angle
        s, c  = torch.sin(half), torch.cos(half)
        return torch.cat([axis * s, c], dim=1)

    # -------------------------------------------------------------------------
    # code (B,D) or (B,T,D) → base + delta; returns (B,T,·,H,W)
    # -------------------------------------------------------------------------
    def export_UV(self, code_id, code_app, delta_down: int | None = 1, t_chunk: int | None = None, amp: bool = True, return_delta: bool = False):
        """
        Args:
            code_id : (B,T,D)
            code_app: (B,T,D)
        Returns:
            pos, quat, scale, sh_dc, sh_rest, opacity: (B,T,·,H,W)
        """
        B, T, Di = code_id.shape
        _, _, Da = code_app.shape
        H, W = self.H, self.W
        df = 1 if (delta_down is None or int(delta_down) <= 1) else int(delta_down)
        t_chunk = T if (t_chunk is None or t_chunk <= 0) else int(t_chunk)

        base_stack_B = torch.cat([
            self.img_position, self.img_quat, self.img_scale,
            self.img_sh_dc, self.img_sh_rest, self.img_opacity
        ], dim=0).unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # (B,C,H,W)

        chunks = []
        delta_id_chunks  = [] if return_delta else None
        delta_app_chunks = [] if return_delta else None
        for t0 in range(0, T, t_chunk):
            t1 = min(T, t0 + t_chunk)
            z_id  = code_id[:, t0:t1, :].reshape(B * (t1 - t0), Di)
            z_app = code_app[:, t0:t1, :].reshape(B * (t1 - t0), Da)

            bs = base_stack_B[:, None, ...].expand(B, (t1 - t0), base_stack_B.shape[1], H, W)
            bs = bs.reshape(B * (t1 - t0), base_stack_B.shape[1], H, W)
        
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                idx = 0
                b_pos   = bs[:, idx:idx+3];                         idx += 3
                b_quat  = bs[:, idx:idx+4];                         idx += 4
                b_scale = bs[:, idx:idx+3];                         idx += 3   # ← 로짓 공간
                b_shdc  = bs[:, idx:idx+3];                         idx += 3
                b_shr   = bs[:, idx:idx+3*self.max_rest_dim];       idx += 3*self.max_rest_dim
                b_opac  = bs[:, idx:idx+1]
            
                if not getattr(self, "_delta_enabled", True):
                    delta = torch.zeros(bs.shape[0], self.C_base, H, W, device=bs.device, dtype=bs.dtype)
                    d_id = d_app = torch.zeros_like(delta)
                else:
                    if df > 1:
                        bs_small = F.interpolate(bs, scale_factor=1.0/df, mode="bilinear", align_corners=False)
                        d_id_small  = self.delta_id (bs_small, z_id)
                        d_app_small = self.delta_app(bs_small, z_app)
                        delta_small = self.w_id * d_id_small + self.w_app * d_app_small
                        delta = F.interpolate(delta_small, size=(H, W), mode="bilinear", align_corners=False)
                        d_id  = F.interpolate(d_id_small,  size=(H, W), mode="bilinear", align_corners=False)
                        d_app = F.interpolate(d_app_small, size=(H, W), mode="bilinear", align_corners=False)
                    else:
                        d_id  = self.delta_id (bs, z_id)
                        d_app = self.delta_app(bs, z_app)
                        delta = self.w_id * d_id + self.w_app * d_app
                        
                idx = 0
                d_pos   = delta[:, idx:idx+3];                      idx += 3
                d_quat  = delta[:, idx:idx+4];                      idx += 4
                d_scale = delta[:, idx:idx+3];                      idx += 3
                d_shdc  = delta[:, idx:idx+3];                      idx += 3
                d_shr   = delta[:, idx:idx+3*self.max_rest_dim];    idx += 3*self.max_rest_dim
                d_opac  = delta[:, idx:idx+1]

                geom_warmup = getattr(hp, "geom_warmup_iters", 1)
                r_pos = getattr(hp, "max_pos_offset", 0.10)
                k_rot = getattr(hp, "delta_rot_scale", 1.0)
                k_scl = getattr(hp, "delta_scale_gain", 0.3)   # ← 0.5 → 0.05
                k_col = getattr(hp, "delta_color_scale", 0.15)
                k_opa = getattr(hp, "delta_opacity_scale", 0.50)

                step = getattr(self, "_global_step", 0)
                alpha = 1.0 if geom_warmup <= 0 else min(1.0, step / float(geom_warmup))

                # pos (동일, 제한)
                pos = b_pos + torch.tanh(d_pos) * (r_pos * alpha)

                # quat (소각 회전 곱)
                rv   = (k_rot * alpha) * torch.tanh(d_quat[:, :3, ...])
                dq   = self._rotvec_to_quat(rv)
                quat = F.normalize(self._quat_mul(dq, b_quat), dim=1, eps=1e-8)
                # quat = F.normalize(self._quat_mul(F.normalize(d_quat), b_quat), dim=1, eps=1e-8)

                # (중요) scale: 로짓공간에서 합 → sigmoid → [s_min,s_max]
                scale_logits = b_scale + (k_scl * alpha) * d_scale        # b_scale가 로짓이므로 그대로 더함
                scale = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(scale_logits)

                # 색/불투명도
                sh_dc = (b_shdc + k_col * torch.tanh(d_shdc)).clamp_(0.0, 1.0) if self.active_sh_degree == 0 else (b_shdc + d_shdc)
                if getattr(self, "_zero_new_rest_once", False):
                    s = 3 * getattr(self, "_prev_rest_dim", 0)
                    e = 3 * getattr(self, "_curr_rest_dim", 0)
                    if e > s:   # 새로 열린 계수 구간이 있을 때만
                        d_shr[:, s:e, ...] = 0
                    self._zero_new_rest_once = False
                sh_rest = b_shr + d_shr
                opacity = torch.sigmoid(b_opac + k_opa * d_opac)
                opacity = opacity * self.uv_valid.to(opacity.dtype)
                opa_floor = float(getattr(hp, "opacity_floor", 0.1))
                opacity = opa_floor + (1.0 - opa_floor) * opacity
                opacity = torch.nan_to_num(opacity, nan=opa_floor, posinf=1.0, neginf=opa_floor).clamp(0.0, 1.0)

                out_cat = torch.cat([pos, quat, scale, sh_dc, sh_rest, opacity], dim=1)
                out_chunk = out_cat.view(B, (t1 - t0), -1, H, W)
                chunks.append(out_chunk)
                
                if return_delta:
                    delta_id_chunks.append(d_id.view(B, (t1 - t0), self.C_base, H, W))
                    delta_app_chunks.append(d_app.view(B, (t1 - t0), self.C_base, H, W))
            
        out = torch.cat(chunks, dim=1)                                    # (B,T,C,H,W)

        # ---- slice channels ----
        idx = 0
        pos     = out[:, :, idx:idx+3];                   idx += 3
        quat    = out[:, :, idx:idx+4];                   idx += 4
        scale   = out[:, :, idx:idx+3];                   idx += 3
        sh_dc   = out[:, :, idx:idx+3];                   idx += 3
        sh_rest = out[:, :, idx:idx+3*self.max_rest_dim]; idx += 3*self.max_rest_dim
        opacity = out[:, :, idx:idx+1]

        if return_delta:
            d_id_full  = torch.cat(delta_id_chunks,  dim=1)  # (B,T,C,H,W)
            d_app_full = torch.cat(delta_app_chunks, dim=1)
            aux = {"d_id": d_id_full.contiguous(),
                "d_app": d_app_full.contiguous()}
            return (pos.contiguous(), quat.contiguous(), scale.contiguous(),
                    sh_dc.contiguous(), sh_rest.contiguous(), opacity.contiguous(), aux)

        return (pos.contiguous(), quat.contiguous(), scale.contiguous(),
                sh_dc.contiguous(), sh_rest.contiguous(), opacity.contiguous())

    # -------------------------------------------------------------------------
    # UV images → N splats (GPU-parallel random sampling in UV)
    # -------------------------------------------------------------------------
    # @torch.no_grad()
    def export_gaussian_splats(
        self,
        pos, quat, scale, sh_dc, sh_rest, opacity,
        N: int | None = None,
        weighted: bool = True,
        replacement: bool = False,
        temperature: float = 1.0,
        sh_chunk: int = 32
    ):
        """
        Args:
            pos, quat, scale, sh_dc, sh_rest, opacity: (B, T, C, H, W)
        Returns:
            splat_pos      : (B, T, N, 3)
            splat_quat     : (B, T, N, 4)
            splat_scale    : (B, T, N, 3)
            splat_sh_dc    : (B, T, N, 3)
            splat_sh_rest  : (B, T, N, 3*max_rest_dim)
            splat_opacity  : (B, T, N, 1)
        """
        B, T, _, H, W = pos.shape
        BT, HW = B * T, H * W
        N = self.num_points if N is None else int(N)
        device = pos.device

        def flatten(x):  # (B,T,C,H,W) -> (BT,C,H,W)
            return x.reshape(B * T, *x.shape[2:])

        pos_f, quat_f, scale_f = flatten(pos), flatten(quat), flatten(scale)
        shdc_f, shr_f, opac_f  = flatten(sh_dc), flatten(sh_rest), flatten(opacity)

        # Sampling indices in UV with opacity-weighted multinomial (GPU)
        if weighted:
            w = opac_f.reshape(BT, HW).float()
            uvv = self.uv_valid.to(w.dtype).view(1, HW).expand(BT, HW)
            w = w * uvv

            # 안전장치
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w.clamp_(min=0.0)

            if temperature != 1.0:
                w = (w + 1e-12).pow(1.0 / temperature)

            # 완전 영행 방지용 uniform mix
            uniform = torch.full_like(w, 1.0 / HW)
            mix = 0.05
            w = (1.0 - mix) * w + mix * uniform

            row_sum = w.sum(dim=1, keepdim=True)
            bad = (row_sum <= 0) | (~torch.isfinite(row_sum))
            w = torch.where(bad, uniform, w / row_sum.clamp_min(1e-12))

            # 무중복 샘플링(겹침 방지)
            k = min(N, HW)
            idx = torch.multinomial(w, num_samples=k, replacement=replacement)
        else:
            idx = torch.randint(0, HW, (BT, min(N, HW)), device=device)

        # Gather utility: (BT,C,H,W) -> (BT,N,C) with optional channel chunking
        def gather_map(x: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
            BT_, C, H_, W_ = x.shape
            xf = x.reshape(BT_, C, H_ * W_)
            if (chunk is None) or (C <= (chunk or 1_000_000_000)):
                idx_exp = idx.unsqueeze(1).expand(BT_, C, N)
                out = torch.gather(xf, 2, idx_exp)             # (BT,C,N)
                return out.transpose(1, 2).contiguous()        # (BT,N,C)
            outs = []
            for s in range(0, C, chunk):
                e = min(s + chunk, C)
                idx_exp = idx.unsqueeze(1).expand(BT_, e - s, N)
                out_ch = torch.gather(xf[:, s:e, :], 2, idx_exp)
                outs.append(out_ch)
            out = torch.cat(outs, dim=1)                       # (BT,C,N)
            return out.transpose(1, 2).contiguous()            # (BT,N,C)

        splat_pos     = gather_map(pos_f,   None)
        splat_quat    = gather_map(quat_f,  None)
        splat_scale   = gather_map(scale_f, None)
        splat_sh_dc   = gather_map(shdc_f,  None)
        splat_sh_rest = gather_map(shr_f,   sh_chunk)
        splat_opacity = gather_map(opac_f,  None)

        def unflatten(xC): return xC.view(B, T, N, xC.shape[-1]).contiguous()
        return (unflatten(splat_pos), unflatten(splat_quat), unflatten(splat_scale),
                unflatten(splat_sh_dc), unflatten(splat_sh_rest), unflatten(splat_opacity))

    ## float sampling
    # def export_gaussian_splats(
    #     self,
    #     pos, quat, scale, sh_dc, sh_rest, opacity,
    #     N: int | None = None,
    #     weighted: bool = True,
    #     replacement: bool = False,
    #     temperature: float = 1.0,
    #     sh_chunk: int = 32
    # ):
    #     """
    #     Args:
    #         pos, quat, scale, sh_dc, sh_rest, opacity: (B, T, C, H, W)  # 모두 UV 이미지(텍스처 맵)
    #     Returns:
    #         splat_pos      : (B, T, N, 3)
    #         splat_quat     : (B, T, N, 4)
    #         splat_scale    : (B, T, N, 3)
    #         splat_sh_dc    : (B, T, N, 3)
    #         splat_sh_rest  : (B, T, N, 3*max_rest_dim)
    #         splat_opacity  : (B, T, N, 1)
    #     """
    #     B, T, _, H, W = pos.shape
    #     BT, HW = B * T, H * W
    #     N = self.num_points if N is None else int(N)
    #     device = pos.device

    #     def flatten(x):  # (B,T,C,H,W) -> (BT,C,H,W)
    #         return x.reshape(B * T, *x.shape[2:])

    #     pos_f, quat_f, scale_f = flatten(pos), flatten(quat), flatten(scale)
    #     shdc_f, shr_f, opac_f  = flatten(sh_dc), flatten(sh_rest), flatten(opacity)

    #     # --- 1) UV 픽셀 인덱스 샘플링(가중치: opacity * uv_valid) ---
    #     if weighted:
    #         w = opac_f.reshape(BT, HW).float()
    #         uvv = self.uv_valid.to(w.dtype).view(1, HW).expand(BT, HW)
    #         w = w * uvv

    #         # 안정화
    #         w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0.0)

    #         if temperature != 1.0:
    #             w = (w + 1e-12).pow(1.0 / temperature)

    #         # 완전 영행 방지용 uniform mix
    #         uniform = torch.full_like(w, 1.0 / HW)
    #         mix = 0.05
    #         w = (1.0 - mix) * w + mix * uniform

    #         row_sum = w.sum(dim=1, keepdim=True)
    #         bad = (row_sum <= 0) | (~torch.isfinite(row_sum))
    #         w = torch.where(bad, uniform, w / row_sum.clamp_min(1e-12))

    #         k = min(N, HW)
    #         idx = torch.multinomial(w, num_samples=k, replacement=replacement)  # (BT, N)
    #     else:
    #         idx = torch.randint(0, HW, (BT, min(N, HW)), device=device)         # (BT, N)

    #     # 실제 사용할 N은 idx 기준으로 확정
    #     N = idx.shape[1]

    #     # --- 2) 인덱스 -> 실수 UV 좌표 변환 (bilinear용 grid 생성) ---
    #     # 이미지 좌표 기준: u ∈ [0, W-1], v ∈ [0, H-1], v=0이 "위".
    #     u = (idx % W).float()     # (BT,N)
    #     v = (idx // W).float()    # (BT,N)

    #     # 필요하면 서브픽셀 지터를 적용할 수 있으나, 인자 추가 없이 기본 0으로 둠.
    #     # jitter = 0.0
    #     # u = u + (torch.rand_like(u) - 0.5) * jitter
    #     # v = v + (torch.rand_like(v) - 0.5) * jitter

    #     # align_corners=True 기준 정규화: 정수 0..W-1, 0..H-1 ↔ [-1,1]
    #     grid_x = (u / max(W - 1, 1)) * 2 - 1        # (BT,N)
    #     grid_y = (v / max(H - 1, 1)) * 2 - 1        # (BT,N)  # ← v 축 뒤집기 없음
    #     grid = torch.stack([grid_x, grid_y], dim=-1).view(BT, N, 1, 2)  # (BT,N,1,2)

    #     # --- 3) bilinear 보간 유틸 ---
    #     def bilinear_map(x: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
    #         # x: (BT,C,H,W) -> (BT,N,C)
    #         BT_, C, H_, W_ = x.shape
    #         if (chunk is None) or (C <= (chunk or 10**9)):
    #             vals = F.grid_sample(x, grid, mode='bilinear', align_corners=True)   # (BT,C,N,1)
    #             vals = vals.squeeze(-1).transpose(1, 2).contiguous()                 # (BT,N,C)
    #             return vals
    #         outs = []
    #         for s in range(0, C, chunk):
    #             e = min(s + chunk, C)
    #             vals = F.grid_sample(x[:, s:e, ...], grid, mode='bilinear', align_corners=True)
    #             vals = vals.squeeze(-1).transpose(1, 2).contiguous()                 # (BT,N,c)
    #             outs.append(vals)
    #         return torch.cat(outs, dim=-1)  # (BT,N,C)

    #     # --- 4) 실제 보간 샘플링 ---
    #     splat_pos     = bilinear_map(pos_f,   None)
    #     splat_quat    = bilinear_map(quat_f,  None)
    #     splat_scale   = bilinear_map(scale_f, None)
    #     splat_sh_dc   = bilinear_map(shdc_f,  None)
    #     splat_sh_rest = bilinear_map(shr_f,   sh_chunk)
    #     splat_opacity = bilinear_map(opac_f,  None)

    #     # --- 5) (B,T,N,C)로 형태 복원 ---
    #     def unflatten(xC): return xC.view(B, T, N, xC.shape[-1]).contiguous()
    #     return (unflatten(splat_pos), unflatten(splat_quat), unflatten(splat_scale),
    #             unflatten(splat_sh_dc), unflatten(splat_sh_rest), unflatten(splat_opacity))

            
    def train_stage1_id(self, freeze_posquat_first_k=1500, step=0):
        # Stage1: base + delta 학습 (초반엔 pos/quat만 잠가 안정화)
        for name, p in self.named_parameters():
            if name.startswith("delta_id"):
                p.requires_grad = True
            elif name.startswith("delta_app"):
                p.requires_grad = True
                
            elif name.startswith("img_position") or name.startswith("img_quat"):
                p.requires_grad = (step >= freeze_posquat_first_k)
            elif name.startswith("img_"):
                p.requires_grad = True
                # p.requires_grad = False

            elif name in ("w_id", "w_app"):
                p.requires_grad = True
                
            else:
                p.requires_grad = False
        self.enable_delta(True)

    # -------------------------------------------------------------------------
    # Increase active SH degree; convert DC map RGB→SH once when stepping from 0→1
    # -------------------------------------------------------------------------
    def oneupSHdegree(self):
        """Increment active SH degree; apply RGB2SH to img_sh_dc when leaving degree 0."""
        if self.active_sh_degree == 0:
            with torch.no_grad():
                self.img_sh_dc.copy_(RGB2SH(self.img_sh_dc.clamp(0, 1)))
        if self.active_sh_degree < self.max_sh_degree:
            old = (self.active_sh_degree + 1)**2 - 1
            self.active_sh_degree += 1
            new = (self.active_sh_degree + 1)**2 - 1
            self._prev_rest_dim = old
            self._curr_rest_dim = new
            self._zero_new_rest_once = True
            print("active_sh_degree:", self.active_sh_degree)
    # -------------------------------------------------------------------------
    # Render sequence: code → maps → splats → differentiable renderer
    # -------------------------------------------------------------------------
    def render(self, s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac, cam_matrix, cam_params, yaw_deg, sh_degree):
        """
        Args:
            gs_params
            cam_matrix: (B, 3, 4)
            cam_params: dict with intrinsics
            yaw_deg: float or list-like supported by your renderer
            sh_degree: int, SH degree for rendering
        Returns:
            images: (B, T, 3, H, W)
        """
        B, T = s_pos.shape[:2]
        if sh_degree == 0:
            color_arg = s_shdc.view(B * T, self.num_points, 3).clamp(0, 1)
            shs_arg = None
        else:
            coeffs = (sh_degree + 1) ** 2
            cur_rest_dim = coeffs - 1
            shr_flat = s_shr[..., :3 * cur_rest_dim]
            f_rest  = shr_flat.view(B, T, self.num_points, 3, cur_rest_dim)
            shs_arg = torch.cat([s_shdc.unsqueeze(-1), f_rest], dim=-1)  # (B,T,N,3,coeffs)
            shs_arg = shs_arg.permute(0, 1, 2, 4, 3).contiguous()
            shs_arg = shs_arg.view(B * T, self.num_points, coeffs, 3)
            #shs_arg = shs_arg.view(B * T, self.num_points, coeffs, 3).contiguous()
            color_arg = None

        xyz   = s_pos.view(B * T, self.num_points, 3)
        rots  = s_quat.view(B * T, self.num_points, 4)
        scals = s_scale.view(B * T, self.num_points, 3)
        opacs = s_opac.view(B * T, self.num_points, 1)

        cam_mat_BT = cam_matrix.repeat_interleave(T, dim=0)

        gs_params = {
            'xyz': xyz,
            'rotations': rots,
            'scales': scals,
            'colors': color_arg,
            'opacities': opacs,
            'shs': shs_arg,
        }

        rendered = render_gaussian(gs_params, cam_mat_BT, cam_params=cam_params, yaw_deg=yaw_deg, sh_degree=sh_degree)
        images = rendered["images"].view(B, T, *rendered["images"].shape[1:])

        return images

    # -------------------------------------------------------------------------
    # Quick base-only render (delta=0) for a list of yaws
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def render_base(
        self,
        cam_json=None,
        yaw_list=None,
        save_dir=None,           # e.g. os.path.join(hp.save_path, "render_base")
        prefix="yaw",            # filename prefix
        save_video=False,        # also export an MP4
        fps=25,
        overwrite=False,
        epoch=0
    ):
        """
        Returns:
            outs:       list of (1,3,H,W) tensors in [0,1]
            paths:      list of saved png paths (if save_dir)
            video_path: str | None
        """
        json_path = cam_json if cam_json is not None else hp.single_cam_path
        cam_matrix, cam_params = load_cam(json_path)  # (1,3,4), dict

        B, T = 1, 1
        pos   = self.img_position.unsqueeze(0).unsqueeze(0)
        quat  = F.normalize(self.img_quat, dim=0, eps=1e-8).unsqueeze(0).unsqueeze(0)
        scale_map = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.img_scale)
        scale = scale_map.unsqueeze(0).unsqueeze(0)
        sh_dc = self.img_sh_dc.unsqueeze(0).unsqueeze(0)
        sh_r  = self.img_sh_rest.unsqueeze(0).unsqueeze(0)
        opac  = torch.sigmoid(self.img_opacity).unsqueeze(0).unsqueeze(0)

        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = self.export_gaussian_splats(
            pos, quat, scale, sh_dc, sh_r, opac, N=self.num_points, weighted=True, replacement=False, temperature=0.7
        )

        if yaw_list is None:
            yaw_list = [i * (360 / 32) for i in range(32)]

        save_dir = os.path.join(save_dir, f"{epoch:06d}")
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # if overwrite:
            #     for f in os.listdir(save_dir):
            #         try: os.remove(os.path.join(save_dir, f))
            #         except: pass

        outs, paths = [], []

        def to_u8(img_1x3xHxW):
            x = img_1x3xHxW[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            return x

        for i, yaw in enumerate(yaw_list):
            coeffs = (self.active_sh_degree + 1) ** 2
            if self.active_sh_degree == 0:
                color_arg = s_shdc.view(B * T, self.num_points, 3).clamp(0, 1)
                shs_arg = None
            else:
                cur_rest_dim = coeffs - 1
                shr_flat = s_shr[..., :3 * cur_rest_dim]
                f_rest = shr_flat.view(B, T, self.num_points, 3, cur_rest_dim)
                shs_arg = torch.cat([s_shdc.unsqueeze(-1), f_rest], dim=-1)
                shs_arg = shs_arg.permute(0, 1, 2, 4, 3).contiguous()
                shs_arg = shs_arg.view(B * T, self.num_points, coeffs, 3)
                #shs_arg = shs_arg.view(B * T, self.num_points, coeffs, 3).contiguous()
                color_arg = None

            gs_params = {
                'xyz': s_pos.view(B * T, self.num_points, 3),
                'rotations': s_quat.view(B * T, self.num_points, 4),
                'scales': s_scale.view(B * T, self.num_points, 3),
                'colors': color_arg,
                'opacities': s_opac.view(B * T, self.num_points, 1),
                'shs': shs_arg,
            }
            render_out = render_gaussian(gs_params, cam_matrix, cam_params, yaw_deg=yaw, sh_degree=self.active_sh_degree)
            img = render_out["images"]  # (1,3,H,W)
            outs.append(img)

            if save_dir is not None:
                fname = f"{prefix}_{i:03d}_yaw{yaw:06.2f}.png"
                path = os.path.join(save_dir, fname)
                Image.fromarray(to_u8(img)).save(path)
                paths.append(path)

        video_path = None
        if save_video and save_dir is not None and len(outs) > 0:
            # NOTE: needs "import imageio" at top of file
            video_path = os.path.join(save_dir, f"{prefix}.mp4")
            writer = imageio.get_writer(video_path, fps=fps)
            try:
                for img in outs:
                    writer.append_data(to_u8(img))
            finally:
                writer.close()

        return outs, paths, video_path

    @property
    def sh_degree(self):
        """Current active spherical harmonics degree."""
        return self.active_sh_degree
    
    @property
    def texture(self):
        return self.tex

    # -------------------------------------------------------------------------
    # Export a single timestep of Gaussians from code (B,T,D)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def export_gaussians(self, code: torch.Tensor, step_idx: int = 0):
        """
        Args:
            code: (B, T, D)
        Returns:
            dict with means, scales, rotations, opacity, sh, sh_degree
        """
        if code is None or code.dim() != 3:
            raise ValueError(f"export_gaussians expects code (B,T,D), got {None if code is None else tuple(code.shape)}")
        B, T, D = code.shape
        step = int(step_idx) % T
        code_one = code[:, step:step+1, :]  # (B,1,D)

        s_pos, s_quat, s_scale, s_shdc, s_shr, s_opac = self.forward(code_one)  # (B,1,·,H,W)

        means     = s_pos[:, 0]      # (B,N,3)
        scales    = s_scale[:, 0]    # (B,N,3)
        rotations = s_quat[:, 0]     # (B,N,4)
        opacity   = s_opac[:, 0]     # (B,N,1)

        sd = self.active_sh_degree
        if sd == 0:
            sh = s_shdc[:, 0].unsqueeze(-1)  # (B,N,3,1)
        else:
            coeffs = (sd + 1) ** 2
            rest_dim = coeffs - 1
            f_rest = s_shr[:, 0, :, :3 * rest_dim].view(B, self.num_points, 3, rest_dim)
            sh = torch.cat([s_shdc[:, 0].unsqueeze(-1), f_rest], dim=-1)  # (B,N,3,coeffs)

        return {
            "means": means, "scales": scales, "rotations": rotations,
            "opacity": opacity, "sh": sh, "sh_degree": sd
        }

    def _minmax_norm(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Per-channel min-max normalization.
        x: (C,H,W) on any device. Returns (C,H,W) in [0,1].
        """
        xc = x.detach().float()
        C, H, W = xc.shape
        flat = xc.view(C, -1)
        xmin = flat.min(dim=1, keepdim=True).values
        xmax = flat.max(dim=1, keepdim=True).values
        denom = (xmax - xmin).clamp_min(eps)
        norm = ((flat - xmin) / denom).view(C, H, W)
        return norm

    def _to_uint8_image(self, x: torch.Tensor) -> Image.Image:
        """
        x: (1,H,W) or (3,H,W) in [0,1] float tensor → PIL Image.
        """
        xc = x.detach().clamp(0, 1).mul(255).byte().cpu()
        if xc.shape[0] == 1:
            return Image.fromarray(xc[0].numpy(), mode="L")
        elif xc.shape[0] == 3:
            return Image.fromarray(xc.permute(1, 2, 0).numpy(), mode="RGB")
        else:
            raise ValueError(f"_to_uint8_image expects 1 or 3 channels, got {x.shape[0]}")

    def _save_tensor_image(self, x: torch.Tensor, path: str, do_minmax: bool = True):
        """
        x: (1,H,W) or (3,H,W). Optionally min-max normalize per channel, then save.
        """
        if do_minmax:
            x = self._minmax_norm(x)
        img = self._to_uint8_image(x)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)

    def _save_quat_as_images(self, quat_chw: torch.Tensor, dirpath: str, prefix: str):
        """
        quat_chw: (4,H,W). Visualize as:
        - vec RGB: map (x,y,z) in [-1,1] → [0,1]
        - w  Gray: map w in [-1,1] → [0,1]
        Uses unit quaternions with sign fix so w >= 0.
        """
        q = quat_chw.detach().float()
        # Normalize per-pixel across channel dim
        norm = (q.pow(2).sum(0, keepdim=True).sqrt().clamp_min(1e-8))
        q = q / norm
        # Sign fix: q and -q represent the same rotation → enforce w>=0
        sign = torch.where(q[3:4] >= 0, torch.tensor(1.0, device=q.device), torch.tensor(-1.0, device=q.device))
        q = q * sign
        # Map to display
        vec_rgb = ((q[0:3] + 1.0) * 0.5).clamp(0, 1)  # (3,H,W)
        w_gray  = ((q[3:4] + 1.0) * 0.5).clamp(0, 1)  # (1,H,W)
        os.makedirs(dirpath, exist_ok=True)
        self._save_tensor_image(vec_rgb, os.path.join(dirpath, f"{prefix}_quat_vec_rgb.png"), do_minmax=False)
        self._save_tensor_image(w_gray,  os.path.join(dirpath, f"{prefix}_quat_w_gray.png"),  do_minmax=False)

    def _save_sh_rest_grid(self, sh_rest_chw: torch.Tensor, path: str, cols: int = 8):
        """
        sh_rest_chw: (3*R,H,W) where R = max_rest_dim.
        Tiles R coefficient triplets (RGB each) into a grid image.
        """
        C, H, W = sh_rest_chw.shape
        assert C % 3 == 0, f"sh_rest has {C} channels, not multiple of 3"
        R = C // 3
        tiles = sh_rest_chw.view(R, 3, H, W)  # (R,3,H,W)

        rows = (R + cols - 1) // cols
        canvas = torch.zeros(3, rows * H, cols * W, dtype=tiles.dtype, device=tiles.device)

        for i in range(R):
            r, c = divmod(i, cols)
            tile = self._minmax_norm(tiles[i])  # per-coefficient minmax
            canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = tile

        self._save_tensor_image(canvas, path, do_minmax=False)

    def export_base_img(self, save_root: str = "results/uv_base", sh_cols: int = 8, use_minmax_for_all: bool = True):
        """
        Saves the learnable base maps as images.
        - position_rgb.png : (3,H,W), per-channel min-max
        - quat_vec_rgb.png : (x,y,z) mapped to RGB, w>=0 sign fix
        - quat_w_gray.png  : w component as grayscale
        - scale_rgb.png    : (3,H,W), per-channel min-max
        - sh_dc_rgb.png    : (3,H,W), assumes already in [0,1]
        - sh_rest_grid.png : tiled coefficients (each 3ch) in a grid
        - opacity_gray.png : (1,H,W), min-max
        """
        os.makedirs(save_root, exist_ok=True)

        # Position
        self._save_tensor_image(self.img_position, os.path.join(save_root, "position_rgb.png"), do_minmax=use_minmax_for_all)

        # Quaternion (vec RGB + w Gray)
        self._save_quat_as_images(self.img_quat, save_root, prefix="base")

        # Scale
        scale_vis = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.img_scale)
        self._save_tensor_image(scale_vis, os.path.join(save_root, "scale_rgb.png"), do_minmax=use_minmax_for_all)

        # SH DC (texture)
        # If you require strict [0,1], clamp; otherwise preserve as is.
        if self.active_sh_degree == 0:
            sh_dc_vis = self.img_sh_dc.detach().float().clamp(0, 1)
        else:
            # SH 계수 → RGB로 환산 후 저장
            sh_dc_vis = SH2RGB(self.img_sh_dc.detach().float()).clamp(0, 1)
        self._save_tensor_image(sh_dc_vis, os.path.join(save_root, "sh_dc_rgb.png"), do_minmax=False)

        # SH Rest grid
        self._save_sh_rest_grid(self.img_sh_rest, os.path.join(save_root, "sh_rest_grid.png"), cols=sh_cols)

        # Opacity
        opa_vis = torch.sigmoid(self.img_opacity)
        self._save_tensor_image(opa_vis, os.path.join(save_root, "opacity_gray.png"), do_minmax=use_minmax_for_all)

    def export_result_img(
        self,
        pos: torch.Tensor,       # (B,T,3,H,W)
        quat: torch.Tensor,      # (B,T,4,H,W)
        scale: torch.Tensor,     # (B,T,3,H,W)
        sh_dc: torch.Tensor,     # (B,T,3,H,W)
        sh_rest: torch.Tensor,   # (B,T,3*R,H,W)
        opacity: torch.Tensor,   # (B,T,1,H,W)
        save_root: str = "results/uv_result",
        sh_cols: int = 8,
        use_minmax_for_all: bool = True
    ):
        """
        Saves forward outputs for each batch/time step.
        Directory structure:
        save_root/
            b{b:02d}_t{t:04d}/
            position_rgb.png
            quat_vec_rgb.png
            quat_w_gray.png
            scale_rgb.png
            sh_dc_rgb.png
            sh_rest_grid.png
            opacity_gray.png
        """
        B, T = pos.shape[0], pos.shape[1]
        os.makedirs(save_root, exist_ok=True)

        for b in range(B):
            for t in range(T):
                out_dir = os.path.join(save_root, f"b{b:02d}_t{t:04d}")
                os.makedirs(out_dir, exist_ok=True)

                # Position
                self._save_tensor_image(pos[b, t], os.path.join(out_dir, "position_rgb.png"), do_minmax=use_minmax_for_all)

                # Quaternion (vec RGB + w Gray)
                self._save_quat_as_images(quat[b, t], out_dir, prefix="result")

                # Scale
                self._save_tensor_image(scale[b, t], os.path.join(out_dir, "scale_rgb.png"), do_minmax=use_minmax_for_all)

                # SH DC
                if self.active_sh_degree == 0:
                    sh_dc_vis = sh_dc[b, t].detach().float().clamp(0, 1)
                else:
                    sh_dc_vis = SH2RGB(sh_dc[b, t].detach().float()).clamp(0, 1)
                self._save_tensor_image(sh_dc_vis, os.path.join(out_dir, "sh_dc_rgb.png"), do_minmax=False)

                # SH Rest grid
                self._save_sh_rest_grid(sh_rest[b, t], os.path.join(out_dir, "sh_rest_grid.png"), cols=sh_cols)

                # Opacity
                self._save_tensor_image(opacity[b, t], os.path.join(out_dir, "opacity_gray.png"), do_minmax=use_minmax_for_all)
    
    def save_image(
        self, gt, pred, epoch, save_root="results",
        src=None, tag=None, timestamp="none",
        seed_text: str | None = None,
        view_nums: list[int] | None = None
    ):
        """
        Layout:
        [ SID 라벨 (중앙 정렬, 크게) ]
        [ SRC 한 장 ]
        |  (왼쪽 고정 컬럼)
        |  (오른쪽 본문: 각 열은 [VIEW 라벨; GT; PRED])
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import os, torch

        def tensor_to_img(x):
            return (x.detach().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

        # ---------- shapes ----------
        V = min(gt.shape[0], pred.shape[0])
        T = min(gt.shape[1], pred.shape[1])
        C, H, W = gt.shape[2], gt.shape[3], gt.shape[4]
        assert C == 3, f"Expect 3-channel images, got {C}"

        # ---------- sizes ----------
        LABEL_H = max(24, H // 8)   # 라벨 줄 높이 (여백)

        # ---------- 폰트: 함수 내부에서 자동 결정 ----------
        # H에 비례해 글자 픽셀 크기 계산 (원하면 계수 0.24를 조절)
        font_px = max(24, int(H * 0.09))

        def _load_truetype(px: int):
            for name in ["DejaVuSans.ttf", "Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                try:
                    return ImageFont.truetype(name, size=px)
                except Exception:
                    pass
            return None

        _TTF = _load_truetype(font_px)         # 있을 때: 실제 큰 글자
        _DEF = ImageFont.load_default()         # 없을 때: 작은 비트맵 폰트
        _DEF_BASE_H = 11                        # 기본폰트 대략 높이(대충 11px)
        _DEF_SCALE = max(1, int(round(font_px / _DEF_BASE_H)))

        # ---------- left column ----------
        if (src is not None) and isinstance(src, torch.Tensor):
            if src.dim() == 5:      # (V,T,3,H,W)
                src_one = src[0, 0]
            elif src.dim() == 4:    # (T,3,H,W)
                src_one = src[0]
            elif src.dim() == 3:    # (3,H,W)
                src_one = src
            else:
                raise ValueError(f"Unsupported src shape: {tuple(src.shape)}")
            src_np = tensor_to_img(src_one)
        else:
            src_np = np.zeros((H, W, 3), dtype=np.uint8)

        left_col_h = LABEL_H + H
        left_col_w = W
        body_col_h = LABEL_H + 2 * H
        body_col_w = W
        canvas_h = max(left_col_h, body_col_h)

        left_np = np.zeros((canvas_h, left_col_w, 3), dtype=np.uint8)
        left_np[LABEL_H:LABEL_H+H, :, :] = src_np
        left_img = Image.fromarray(left_np).convert("RGBA")

        # ---------- 중앙 정렬 텍스트 유틸 (TTF 있으면 그대로, 없으면 확대 렌더링) ----------
        def draw_center_text(img: Image.Image, text: str, box_xywh, fg=(255,255,255), bg=(0,0,0)):
            x, y, w, h = box_xywh
            if _TTF is not None:
                d = ImageDraw.Draw(img)
                bx0, by0, bx1, by1 = d.textbbox((0, 0), text, font=_TTF)
                tw, th = (bx1 - bx0), (by1 - by0)
                tx = x + (w - tw) / 2
                ty = y + (h - th) / 2
                pad = 8
                d.rectangle((tx - pad, ty - pad, tx + tw + pad, ty + th + pad), fill=bg)
                d.text((tx, ty), text, fill=fg, font=_TTF)
            else:
                # 기본 폰트로 그린 후 비례 확대해서 합성 (글자 자체가 커짐)
                tmp = Image.new("RGBA", (w // _DEF_SCALE or 1, h // _DEF_SCALE or 1), (0,0,0,0))
                d = ImageDraw.Draw(tmp)
                bx0, by0, bx1, by1 = d.textbbox((0, 0), text, font=_DEF)
                tw, th = (bx1 - bx0), (by1 - by0)
                tx = max(0, (tmp.width - tw) // 2)
                ty = max(0, (tmp.height - th) // 2)
                # 작은 캔버스에 먼저 그림
                d.rectangle((tx-2, ty-2, tx+tw+2, ty+th+2), fill=(0,0,0,255))
                d.text((tx, ty), text, fill=fg + (255,), font=_DEF)
                # 원하는 크기로 확대
                big = tmp.resize((w, h), resample=Image.NEAREST)
                # 박스 위치에 합성
                patch = Image.new("RGBA", img.size, (0,0,0,0))
                patch.paste(big, (x, y))
                img.alpha_composite(patch)

        # 헤더 라벨: seed_text가 있으면 그대로 사용
        if seed_text:
            draw_center_text(left_img, seed_text, (0, 0, W, LABEL_H))

        # ---------- body columns ----------
        body_cols = []
        for i in range(V):
            vtxt = f"view_{int(view_nums[i]):02d}" if (view_nums is not None and i < len(view_nums)) else f"view_{i:02d}"
            for j in range(T):
                col_np = np.zeros((canvas_h, body_col_w, 3), dtype=np.uint8)
                col_np[LABEL_H:LABEL_H+H, :, :] = tensor_to_img(gt[i, j])
                col_np[LABEL_H+H:LABEL_H+2*H, :, :] = tensor_to_img(pred[i, j])
                col_img = Image.fromarray(col_np).convert("RGBA")
                draw_center_text(col_img, vtxt, (0, 0, W, LABEL_H))
                body_cols.append(np.array(col_img.convert("RGB")))

        # ---------- concat ----------
        if body_cols:
            body_np = np.concatenate(body_cols, axis=1)
            out_np = np.concatenate([np.array(left_img.convert("RGB")), body_np], axis=1)
        else:
            out_np = np.array(left_img.convert("RGB"))

        out_img = Image.fromarray(out_np)

        # ---------- save ----------
        if getattr(self, "_save_image_dir", None) is None:
            self._save_image_dir = os.path.join(save_root, timestamp)
            os.makedirs(self._save_image_dir, exist_ok=True)

        fname = f"{epoch:05d}_{tag}.png" if tag else f"{epoch:05d}.png"
        save_path = os.path.join(self._save_image_dir, fname)
        out_img.save(save_path)
        print(f"Saved Images: {save_path}")