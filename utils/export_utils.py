# export_utils.py
import os
import struct
import numpy as np
import torch
from datetime import datetime
from utils.sh_utils import RGB2SH

# -----------------------------
# Utilities
# -----------------------------
def _to_numpy(x):
    """Tensor/array -> contiguous float32 numpy"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x).astype(np.float32)

def _flatten_2d(x):
    """
    Any (..., C) -> (N, C)
    If x is None -> None
    """
    if x is None:
        return None
    x = _to_numpy(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)  # (N,1)
    return x.reshape(-1, x.shape[-1])

def _flatten_1d(x):
    """
    Any (..., 1) or (...) -> (N,)
    If x is None -> None
    """
    if x is None:
        return None
    x = _to_numpy(x)
    if x.ndim == 0:
        return x.reshape(1,)
    if x.ndim >= 2 and x.shape[-1] == 1:
        x = x[..., 0]
    return x.reshape(-1)

def _flatten_sh(sh):
    """
    sh: (B,N,3,C) or (N,3,C) -> (BN,3,C)
    If None -> None
    """
    if sh is None:
        return None
    sh = _to_numpy(sh)
    if sh.ndim == 4:  # (B,N,3,C)
        B, N, _, C = sh.shape
        sh = sh.reshape(B * N, 3, C)
    elif sh.ndim == 3:
        # already (N,3,C)
        pass
    else:
        raise ValueError(f"Unexpected SH shape: {sh.shape}")
    return sh

# -----------------------------
# PLY writer
# -----------------------------
def save_gaussians_ply(
    path,
    means,           # (N,3)
    f_dc=None,       # (N,3)
    f_rest=None,     # (N,M)
    opacity=None,    # (N,)
    scales=None,     # (N,3)
    rotations=None,  # (N,4)
    binding=None     # (N,) or None
):
    """
    Write binary_little_endian PLY with exactly this header layout:
      x, y, z,
      nx, ny, nz,
      f_dc_0..2,
      f_rest_0..M-1,
      opacity,
      scale_0..2,
      rot_0..3,
      binding_0
    """
    means = _to_numpy(means).reshape(-1, 3)
    N = means.shape[0]

    # normals placeholder
    normals = np.zeros_like(means, dtype=np.float32)

    # f_dc
    if f_dc is None:
        f_dc = np.zeros((N, 3), dtype=np.float32)
    else:
        f_dc = _to_numpy(f_dc).reshape(N, 3)

    # f_rest
    if f_rest is None:
        # default length to keep viewer compatibility (optional)
        M = 45
        f_rest = np.zeros((N, M), dtype=np.float32)
    else:
        f_rest = _to_numpy(f_rest).reshape(N, -1)
        M = f_rest.shape[1]

    # opacity
    if opacity is None:
        opacity = np.zeros((N,), dtype=np.float32)
    else:
        opacity = _to_numpy(opacity).reshape(N,)

    # scales
    if scales is None:
        scales = np.ones((N, 3), dtype=np.float32)
    else:
        scales = _to_numpy(scales).reshape(N, 3)

    # rotations
    if rotations is None:
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 3] = 1.0
    else:
        rotations = _to_numpy(rotations).reshape(N, 4)

    # binding_0
    if binding is None:
        binding = np.zeros((N,), dtype=np.float32)
    else:
        binding = _to_numpy(binding).reshape(N,)

    # --- build header ---
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(M):
        header.append(f"property float f_rest_{i}")
    header += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property float binding_0",
        "end_header"
    ]
    header_str = "\n".join(header) + "\n"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        # write header
        f.write(header_str.encode("ascii"))

        # prepare struct
        total_floats = 3 + 3 + 3 + M + 1 + 3 + 4 + 1
        fmt = "<" + "f" * total_floats

        # write vertex data
        for i in range(N):
            vals = []
            vals.extend(means[i].tolist())
            vals.extend(normals[i].tolist())
            vals.extend(f_dc[i].tolist())
            vals.extend(f_rest[i].tolist())
            vals.append(float(opacity[i]))
            vals.extend(scales[i].tolist())
            vals.extend(rotations[i].tolist())
            vals.append(float(binding[i]))
            f.write(struct.pack(fmt, *vals))

# -----------------------------
# High-level export (fixes BN flatten)
# -----------------------------
def export_ply(
    displacementDecoder,
    out_dir,
    tag,
    disp_code=None,
    step_idx=0,
    mode="standard",      # "standard" or "rgb"
    fname=None,           # subfolder name (e.g., timestamp). If None, omitted.
    max_points=None       # Optional cap on number of points saved (int)
):
    """
    displacementDecoder.export_gaussians(code, step_idx) must return a dict with keys:
      'means', 'sh', 'sh_degree', 'opacity', 'scales', 'rotations', optionally 'binding', 'rgb'
    Shapes may be (B,N,...) or (N,...). We flatten to (BN, ...) consistently.
    """
    if disp_code is None:
        print(f"[WARN] export_ply skipped at tag={tag}: disp_code is None.")
        return

    base_dir = os.path.join(out_dir, "ply_exports")
    export_dir = os.path.join(base_dir, fname) if fname else base_dir
    os.makedirs(export_dir, exist_ok=True)

    g = displacementDecoder.export_gaussians(code=disp_code, step_idx=step_idx)

    # means (… ,3) -> (BN,3)
    means = _flatten_2d(g["means"])  # (BN,3)
    N_flat = means.shape[0]

    # SH -> f_dc/f_rest
    sh = g.get("sh", None)
    deg = int(g.get("sh_degree", 3))
    if sh is not None:
        sh_t = _flatten_sh(sh)     # (BN,3,C)
        C = sh_t.shape[-1]
        f_dc   = sh_t[..., 0]      # (BN,3)
        f_rest = sh_t[..., 1:].reshape(N_flat, -1) if C > 1 else np.zeros((N_flat, 0), dtype=np.float32)
    else:
        # Fallback: build SH from rgb or default white
        rgb = g.get("rgb", None)
        if rgb is None:
            rgb = np.ones((N_flat, 3), dtype=np.float32)
        else:
            rgb = _flatten_2d(rgb)            # (BN,3)
        coeffs = (deg + 1) ** 2
        sh_t = np.zeros((N_flat, 3, coeffs), dtype=np.float32)
        # RGB2SH expects torch tensor in [0,1]
        dc = RGB2SH(torch.from_numpy(rgb)).detach().cpu().numpy()  # (BN,3)
        sh_t[..., 0] = dc
        f_dc   = sh_t[..., 0]                  # (BN,3)
        f_rest = sh_t[..., 1:].reshape(N_flat, -1)

    # other attributes (flatten)
    opacity   = _flatten_1d(g.get("opacity", None))   # (BN,)
    scales    = _flatten_2d(g.get("scales", None))    # (BN,3)
    rotations = _flatten_2d(g.get("rotations", None)) # (BN,4)
    binding   = _flatten_1d(g.get("binding", None))   # (BN,)

    # Optional: cap number of points written
    if isinstance(max_points, int) and max_points > 0 and means.shape[0] > max_points:
        sel = np.random.choice(means.shape[0], max_points, replace=False)
        means    = means[sel]
        f_dc     = f_dc[sel] if f_dc is not None else None
        f_rest   = f_rest[sel] if f_rest is not None else None
        opacity  = opacity[sel] if opacity is not None and opacity.size == means.shape[0] else opacity
        scales   = scales[sel] if scales is not None else None
        rotations= rotations[sel] if rotations is not None else None
        binding  = binding[sel] if binding is not None and binding.size == means.shape[0] else binding
        N_flat   = means.shape[0]

    # --- Save full SH PLY ---
    ply_path = os.path.join(export_dir, f"gaussians_{tag}.ply")
    save_gaussians_ply(
        ply_path,
        means,
        f_dc=f_dc,
        f_rest=f_rest,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        binding=binding,
    )

    # --- Optional RGB-only PLY (points view) ---
    if mode == "rgb":
        pts_path = os.path.join(export_dir, f"points_{tag}.ply")
        rgb8 = (np.clip(f_dc, 0.0, 1.0) * 255.0).astype(np.float32)  # keep float for writer
        save_gaussians_ply(
            pts_path,
            means,
            f_dc=rgb8,
            f_rest=None,
            opacity=None,
            scales=None,
            rotations=None,
            binding=None,
        )
