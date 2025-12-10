import trimesh
import torch
import open3d as o3d
import numpy as np
import math
import json

def load_obj_as_mesh(obj_path):
    mesh = trimesh.load(obj_path, process=False)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    assert faces.max() < vertices.shape[0], "Face index exceeds vertex count!"

    return vertices, faces
    
def save_pointcloud_image_offscreen(points_tensor, out_path="points_render.png", width=800, height=800):
    import open3d as o3d
    import numpy as np

    points_np = points_tensor.squeeze(0).detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # ✔️ 포인트 색상 (파란색)
    colors = np.tile(np.array([[0.0, 0.0, 1.0]]), (points_np.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene

    # ✔️ 배경색
    scene.set_background([1.0, 1.0, 1.0, 1.0])

    # ✔️ 머티리얼에 point size 지정
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 5.0

    scene.add_geometry("pointcloud", pcd, material)

    # ✔️ 카메라 세팅
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = bounds.get_extent().max()
    eye = center + [0, 0, extent * 1.0]
    scene.camera.look_at(center, eye, [0, 1, 0])

    # 렌더링 및 저장
    img = renderer.render_to_image()
    o3d.io.write_image(out_path, img)

    print(f"✅ Point cloud image saved to {out_path}")

def convert_camera_batch(cam_25_batch, image_size=(512, 512), device=None):
    B = cam_25_batch.shape[0]
    cam_matrix = cam_25_batch[:, :16].reshape(B, 4, 4)[:, :3, :4]  # (B, 3, 4)
    intrinsics = cam_25_batch[:, 16:25].reshape(B, 3, 3)  # (B, 3, 3)

    width, height = image_size

    # focal_x, focal_y를 스칼라 텐서로 변환 (device 지정 가능)
    focal_x = torch.tensor((intrinsics[:, 0, 0]).mean().item(), dtype=torch.float32)
    focal_y = torch.tensor((intrinsics[:, 1, 1]).mean().item(), dtype=torch.float32)

    if device is not None:
        cam_matrix = cam_matrix.to(device)
        focal_x = focal_x.to(device)
        focal_y = focal_y.to(device)

    cam_params = {
        'focal_x': focal_x,                           # tensor scalar
        'focal_y': focal_y,                           # tensor scalar
        'size': torch.tensor([height, width]),       # 1D tensor
    }
    return cam_matrix, cam_params

def convert_c2w_to_w2c(cam_matrix):  # cam_matrix: (B, 3, 4)
    R = cam_matrix[:, :3, :3]
    T = cam_matrix[:, :3, 3:]

    R_inv = R.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, T)

    w2c = torch.cat([R_inv, t_inv], dim=2)
    return w2c

def prepare_cam_matrix(cam_25_tensor, image_size=(512, 512)):
    """
    Converts raw 25D camera tensor into processed cam_matrix and cam_params.

    Args:
        cam_25_tensor: (1, 25) torch tensor (e.g., from .json or .npy)
        image_size: (W, H)

    Returns:
        cam_matrix: (1, 3, 4)
        cam_params: dict
    """
    device = cam_25_tensor.device

    cam_matrix, cam_params = convert_camera_batch(cam_25_tensor, image_size=image_size, device=device)

    flip_y = torch.diag(torch.tensor([1, -1, 1, 1], dtype=torch.float32)).to(device)
    cam_matrix = torch.matmul(flip_y[:3, :3], cam_matrix)
    cam_matrix[:, :3, 2] *= -1
    cam_matrix = convert_c2w_to_w2c(cam_matrix)

    return cam_matrix, cam_params


def rotate_camera_around_origin_3x4(extrinsic_3x4: torch.Tensor, yaw_deg=0, pitch_deg=0):
    # extrinsic_3x4: (B, 3, 4)
    assert extrinsic_3x4.ndim == 3 and extrinsic_3x4.shape[1:] == (3, 4)

    B = extrinsic_3x4.shape[0]
    R = extrinsic_3x4[:, :, :3]     # (B, 3, 3)
    T = extrinsic_3x4[:, :, 3]      # (B, 3)

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    Ry = torch.tensor([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0,           1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ], dtype=torch.float32, device=extrinsic_3x4.device)

    Rx = torch.tensor([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ], dtype=torch.float32, device=extrinsic_3x4.device)

    R_rotate = Ry @ Rx  # (3, 3)

    out = []
    for b in range(B):
        # cam position in world coords
        cam_pos = -R[b].T @ T[b]  # (3,)

        # rotate position
        cam_pos_new = R_rotate @ cam_pos

        # look-at direction
        forward = (torch.zeros(3, device=cam_pos.device) - cam_pos_new)
        forward = forward / forward.norm()

        # right / up
        up_guess = torch.tensor([0, 1, 0], dtype=torch.float32, device=cam_pos.device)
        right = torch.cross(up_guess, forward)
        right = right / right.norm()
        up = torch.cross(forward, right)

        # RH: [right, up, -forward]
        R_new = torch.stack([right, up, -forward], dim=0)  # (3,3)
        T_new = -R_new @ cam_pos_new  # (3,)

        extrinsic_new = torch.cat([R_new, T_new[:, None]], dim=1)  # (3,4)
        out.append(extrinsic_new)

    return torch.stack(out, dim=0)  # (B, 3, 4)

def rotate_view_around_origin(view: torch.Tensor, yaw_deg: float):
    # view: (4, 4), column-major style (Blender-style)
    assert view.shape == (4, 4)
    device = view.device
    dtype = view.dtype

    # 1. view → c2w
    c2w = torch.inverse(view)  # (4,4)

    # 2. extract camera position
    cam_pos = c2w[3, :3]  # (3,)

    # 3. rotate position around origin by yaw
    yaw_rad = math.radians(yaw_deg)
    Ry = torch.tensor([
        [ math.cos(yaw_rad), 0, math.sin(yaw_rad)],
        [ 0,                 1, 0                ],
        [-math.sin(yaw_rad), 0, math.cos(yaw_rad)],
    ], dtype=dtype, device=device)

    cam_pos_rotated = Ry @ cam_pos  # (3,)

    # 4. compute new view direction
    forward = -cam_pos_rotated / cam_pos_rotated.norm()  # look-at center at (0,0,0)
    up = torch.tensor([0, -1, 0], dtype=dtype, device=device)

    right = torch.cross(up, forward)
    right = right / right.norm()
    up = torch.cross(forward, right)

    R_new = torch.stack([right, up, forward], dim=1)  # (3,3), column-major
    c2w_new = torch.eye(4, dtype=dtype, device=device)
    c2w_new[:3, :3] = R_new
    c2w_new[3, :3] = cam_pos_rotated

    # 5. back to view matrix
    view_new = torch.inverse(c2w_new)
    return view_new

def load_cam(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        
    cam = np.array([float(x) for x in data["labels"][0][1]], dtype=np.float32)
    cam = torch.tensor(cam, dtype=torch.float32, device="cuda")  # (25,)
    cam = cam.unsqueeze(0)  # (1, 25)
    cam_matrix, cam_params = convert_camera_batch(cam)  # cam_matrix: (1, 3, 4), c2w

    return cam_matrix, cam_params

# def load_cam(json_path):
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     f0 = data["frames"][0]

#     # Intrinsics (prefer per-frame, fallback to global)
#     fx = float(f0.get("fl_x", data.get("fl_x")))
#     fy = float(f0.get("fl_y", data.get("fl_y", fx)))
#     cx = float(f0.get("cx",   data.get("cx")))
#     cy = float(f0.get("cy",   data.get("cy")))
#     H  = int(f0.get("h",      data.get("h", 512)))
#     W  = int(f0.get("w",      data.get("w", 512)))

#     # 4x4 c2w transform matrix (row-major flatten)
#     T = f0["transform_matrix"]
#     ext_flat = [float(v) for row in T for v in row]  # 16 elems

#     # 3x3 intrinsics (row-major flatten)
#     K_flat = [fx, 0.0, cx,
#               0.0, fy, cy,
#               0.0, 0.0, 1.0]  # 9 elems

#     # Pack to the same 25-d layout as before: [4x4 | 3x3]
#     cam_vec = torch.tensor(ext_flat + K_flat, dtype=torch.float32, device="cuda").unsqueeze(0)  # (1,25)

#     cam_matrix, cam_params = convert_camera_batch(cam_vec)  # cam_matrix: (1,3,4) c2w

#     # Ensure size is available downstream
#     if isinstance(cam_params, dict) and "size" not in cam_params:
#         cam_params["size"] = (H, W)

#     return cam_matrix, cam_params


# def generate_sphere(N: int,
#                     radius: float = 0.1,
#                     center: float | torch.Tensor = 0.0,
#                     device: str = "cuda"):
#     # center → (1, 1, 3)
#     center = torch.as_tensor(center, dtype=torch.float32, device=device).view(-1)
#     if center.numel() == 1:                 # scalar → broadcast
#         center = center.repeat(3)
#     elif center.numel() != 3:               # wrong shape
#         raise ValueError("center must be scalar or length-3")
#     center = center.view(1, 1, 3)

#     xyz = torch.randn(1, N, 3, device=device)
#     xyz = xyz / xyz.norm(dim=-1, keepdim=True)  # unit sphere
#     xyz = xyz * radius + center                # scale & shift

#     color  = torch.zeros(1, N, 3, device=device)
#     color[..., 2] = 1.0                        # blue
#     alpha  = torch.full((1, N, 1), 0.8, device=device)
#     scale  = torch.full((1, N, 3), 0.003, device=device)
#     quat   = torch.tensor([0., 0., 0., 1.], device=device) \
#                 .repeat(1, N, 1)

#     return xyz, quat, scale, color, alpha