import math
import torch
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from utils import rotate_view_around_origin

def render_gaussian(gs_params, cam_matrix, cam_params=None, sh_degree=0, bg_color=None, yaw_deg=None):
    batch_size = cam_matrix.shape[0]

    focal_x, focal_y, cam_size = cam_params['focal_x'], cam_params['focal_y'], cam_params['size']
    height, width = map(int, cam_size)

    fovx  = 2 * math.atan(width  * 0.5 / focal_x)   # radians
    fovy  = 2 * math.atan(height * 0.5 / focal_y)   # radians
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    cam_pos = []
    view_matrices = []
    proj_matrices = []

    for b in range(batch_size):
        R = cam_matrix[b, :3, :3].cpu().numpy()
        T = cam_matrix[b, :3, 3].cpu().numpy()
        
        view = torch.tensor(getWorld2View2(R, T, translate=np.array([0, 0, 0]), scale=1.0), dtype=torch.float32, device="cuda").transpose(0,1)        
        view[3, 0] *= -1
        view[1, 1] *= -1
        view[2, 2] *= -1
        view = rotate_view_around_origin(view, yaw_deg=yaw_deg)

        proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).to(view.device).transpose(0,1)
        # print("view: ", view)
        # print("proj: ", proj)
        proj = view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        view_matrices.append(view)
        proj_matrices.append(proj)
        
        cam_center = view.inverse()[3, :3]
        # cam_center = view.inverse()[:3, 3]
        # print(cam_center)
        # exit()
        
        # print("cam_pos: ", cam_center)
        cam_pos.append(cam_center)

    view_matrices = torch.stack(view_matrices, dim=0)
    proj_matrices = torch.stack(proj_matrices, dim=0)
    cam_pos = torch.stack(cam_pos, dim=0)
    
    # background color
    bg_color = cam_matrix.new_ones(batch_size, 3, dtype=torch.float32) if bg_color is None else bg_color

    points = gs_params['xyz']
    colors = gs_params['colors']
    opacities = gs_params['opacities']
    scales = gs_params['scales']
    rotations = gs_params['rotations']
    shs_all = gs_params.get('shs', None)
    colors = gs_params.get('colors', None)

    def _ensure_batched(x):
        if x is None:
            return None
        return x.unsqueeze(0) if x.dim()==2 else x  # (P,*) → (1,P,*)

    points, colors, opacities = map(_ensure_batched, (points, colors, opacities))
    scales, rotations         = map(_ensure_batched, (scales, rotations))
    if shs_all is not None and shs_all.dim()==3:   # (P,3,C) → (1,P,3,C)
        shs_all = shs_all.unsqueeze(0)

    B_pts = points.shape[0]
    if B_pts == 1 and batch_size > 1:
        points    = points.expand(batch_size, -1, -1)
        colors    = colors.expand(batch_size, -1, -1)
        opacities = opacities.expand(batch_size, -1, -1)
        scales    = scales.expand(batch_size, -1, -1)
        rotations = rotations.expand(batch_size, -1, -1)
        if shs_all is not None:
            shs_all = shs_all.expand(batch_size, -1, -1, -1)

    # -----------------------------
    # Rasterize per batch
    # -----------------------------
    all_rendered, all_radii, all_means2D = [], [], []
    for bid in range(batch_size):
        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color[bid],
            scale_modifier=1.0,
            viewmatrix=view_matrices[bid],
            projmatrix=proj_matrices[bid],
            sh_degree=sh_degree,
            campos=cam_pos[bid],
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = points[bid].new_zeros(points[bid].shape[0], 3, requires_grad=True)
        try: means2D.retain_grad()
        except: pass

        if sh_degree > 0:
            assert shs_all is not None, "sh_degree>0인데 shs가 None입니다."
            shs_arg = shs_all[bid]
            coeffs = (sh_degree + 1) ** 2
            shs_arg = shs_arg[..., :coeffs].contiguous().float()
            color_arg = None
        else:
            shs_arg = None
            color_arg = colors[bid].contiguous().float()  # ← 덮어쓰기 버그 제거

        rendered, radii = rasterizer(
            means3D=points[bid].contiguous().float(),
            means2D=means2D,
            shs=shs_arg,
            colors_precomp=color_arg,
            opacities=opacities[bid].contiguous().float(),
            scales=scales[bid].contiguous().float(),
            rotations=rotations[bid].contiguous().float(),
            cov3D_precomp=None
        )

        all_rendered.append(rendered)
        all_radii.append(radii)
        all_means2D.append(means2D)

    return {
        "images": torch.stack(all_rendered, dim=0),
        "radii": torch.stack(all_radii, dim=0),
        "viewspace_points": torch.stack(all_means2D, dim=0),
        "cam_pos" : cam_pos.detach().cpu(),
        "view_mat": view_matrices.detach().cpu(),
        "fovx"    : fovx,
        "fovy"    : fovy,
    }