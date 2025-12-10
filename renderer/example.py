import os
import torch
from PIL import Image
from utils import render_gaussian
import trimesh
from kaolin.ops.mesh import sample_points
import torch.nn.functional as F


def load_obj_as_mesh(obj_path):
    mesh = trimesh.load(obj_path, process=False)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    assert faces.max() < vertices.shape[0], "Face index exceeds vertex count!"

    return vertices, faces

if __name__ == '__main__':
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load params
    gs_params = torch.load('./data/0.pt', map_location=device, weights_only=True)

    vertices, faces = load_obj_as_mesh("/source/Hyeonho/Research/MonoFace/data/bfm.obj")
    vertices = vertices.cuda().unsqueeze(0)
    faces = faces.cuda()
    points, face_indices = sample_points(vertices, faces, num_samples=100)
    
    gs_params = {
        'xyz': points.cuda(),               # (B, N, 3)
        'colors': torch.ones_like(points, device="cuda:0")*0.5,          # (B, N, 3)
        'opacities': torch.ones_like(points, device="cuda:0")[:,:,:1]*0.9,     # (B, N, 1)
        'scales': torch.ones_like(points, device="cuda:0")*0.001,          # (B, N, 3)
        'rotations': F.normalize(torch.rand(1, 100, 4), dim=-1).cuda(),        # (B, N, 4)
    }        

    cam_matrix = torch.tensor([
        [
            [-0.9976,  0.0662,  0.0215, -0.0179],
            [ 0.0634,  0.9918, -0.1105, -0.0498],
            [-0.0286, -0.1089, -0.9936, 10.2306]
        ]
    ], device='cuda:0')

    cam_params = {
        'focal_x': 12.0,
        'focal_y': 12.0,
        'size': (512, 512)
    }
    
    print("Checking gs_params:")
    for k, v in gs_params.items():
        print(f"{k}: {v.shape}, nan={torch.isnan(v).any().item()}, inf={torch.isinf(v).any().item()}, min={v.min().item()}, max={v.max().item()}")
    
    print(f"camera_matrix: {cam_matrix}")
    print(cam_matrix.shape)
    print(f"camera_params: {cam_params}")

    # Render
    with torch.no_grad():
        images = render_gaussian(gs_params, cam_matrix, cam_params=cam_params)['images']

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    batch_size = cam_matrix.shape[0]
    for i in range(batch_size):
        image = images[i, :3]  # Take RGB channels
        image = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu()  # (H, W, 3)
        img = Image.fromarray(image.numpy())
        img.save(f'{output_dir}/{i}.png')

    print("Rendering complete and saved to ./output/")