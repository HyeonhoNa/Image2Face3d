import json
import numpy as np
import trimesh
import torch
from torchvision.utils import save_image
import os
import math
import scipy.io

# 🔧 PyTorch3D 렌더링 구성``
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, FoVOrthographicCameras, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    PointLights, TexturesVertex
)

def euler_to_rot_matrix(angle):
    """오일러 각 (yaw, pitch, roll)을 회전 행렬로 변환"""
    yaw, pitch, roll = angle[0], angle[1], angle[2]

    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)],
    ], dtype=torch.float32)

    Ry = torch.tensor([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)],
    ], dtype=torch.float32)

    Rz = torch.tensor([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1],
    ], dtype=torch.float32)

    R = Rz @ Ry @ Rx
    R = R.permute(0, 1)
    return R

def intrinsics_to_fov(intrinsics, image_size):
    #fx = intrinsics[0, 0]
    #fy = intrinsics[1, 1]
    fx = 1015
    fy = 1015
    w, h = image_size, image_size  # 정사각형 기준

    fov_x = 2 * math.atan(112 / (fx)) * 180 / math.pi
    fov_y = 2 * math.atan(112 / (fy)) * 180 / math.pi
    #fov_x = 2 * math.atan(w / (2 * fx)) * 180 / math.pi
    #fov_y = 2 * math.atan(h / (2 * fy)) * 180 / math.pi
    return fov_x, fov_y

def render_mesh_image(vertices, faces, c2w_matrix, image_size=512, device="cuda"):
    """
    렌더링 함수: 메쉬와 camera-to-world pose로 정지 이미지 렌더링`
    """
    # 1. 메쉬 설정

    face_model_path = '/source/Hyeonho/Research/MonoFace/submodules/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/BFM/BFM_model_front.mat'
    mat_path = "/source/Hyeonho/Research/MonoFace/submodules/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/input_image/epoch_20_000000/39798.mat"
    data = scipy.io.loadmat(mat_path)
    angle = data['angle'].squeeze()
    print(angle)
    trans = data['trans'].squeeze()
    print(trans)
    print(data)

    face_model = scipy.io.loadmat(face_model_path)
    print(face_model.keys())


    #angle = [0.09270443, -0.03940581, 0.00396501]
    #angle = [-0.0528, -0.0843,  0.0154]
    #trans = [0.00373354, -0.0542998, 0.25858536]

    # 정점 설정 및 변환
    verts = torch.tensor(vertices, dtype=torch.float32, device=device)
    R = euler_to_rot_matrix(angle).to(device)
    T = torch.tensor(trans, dtype=torch.float32, device=device)

    # 정점에 회전 및 이동 적용
    verts = verts @ R + T.unsqueeze(0)

    #verts = torch.tensor(vertices, dtype=torch.float32, device=device)
    #verts = verts + torch.tensor(([ 0.00373354, -0.0542998 , 0.25858536]), dtype=torch.float32, device=device).unsqueeze(0) 
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    verts_rgb = torch.ones_like(verts)[None]  # 흰색 텍스처
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    # 2. camera-to-world → world-to-camera 변환
    flip = np.diag([1, -1, 1, 1])
    w2c = flip @ np.linalg.inv(c2w_matrix)
    print(w2c)
    R = torch.tensor(w2c[:3, :3][None], dtype=torch.float32, device=device)  # (1, 3, 3)
    T = torch.tensor(w2c[:3, 3][None], dtype=torch.float32, device=device)   # (1, 3)

    # 3. 카메라 정의
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    print(cameras)

    # 4. 렌더러 구성
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    # 5. 렌더링
    images = renderer(mesh)  # (1, H, W, 3)
    image = images[0, ..., :3].permute(2, 0, 1)  # (3, H, W)
    return image

def main():
    # 🔧 경로 설정
    json_path = "/source/Hyeonho/Research/MonoFace/submodules/Diffportrait360/diffportrait360_release/sample_data/input_image/dataset.json"
    mesh_path = "/source/Hyeonho/Research/MonoFace/submodules/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/input_image/epoch_20_000000/39798.obj"        # 렌더링할 .obj 메쉬 경로
    #mesh_path = '/source/Hyeonho/Research/MonoFace/data/bfm.obj'
    output_path = "rendered.png" # 출력 이미지 경로
    label_index = 0                    # dataset.json에서 사용할 label index

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. JSON 파일 로딩
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2. 25D 카메라 파라미터 추출
    camera_vec = np.array([float(x) for x in data["labels"][label_index][1]], dtype=np.float32)
    c2w = camera_vec[:16].reshape(4, 4)  # camera-to-world
    intrinsics = camera_vec[16:].reshape(3, 3)  # (선택사항)
    print("intrinsics: ", intrinsics)
    
    fov_x, fov_y = intrinsics_to_fov(intrinsics, image_size=1024)
    print(f"fov_x: {fov_x:.2f} degrees")
    print(f"fov_y: {fov_y:.2f} degrees")

    # 3. 메쉬 로딩
    mesh = trimesh.load(mesh_path, process=False)
    vertices = mesh.vertices
    faces = mesh.faces

    # 4. 렌더링
    image = render_mesh_image(vertices, faces, c2w_matrix=c2w, image_size=1024, device=device)

    # 5. 저장
    save_image(image, output_path)
    print(f"✅ 렌더링 완료 → {output_path}")

if __name__ == "__main__":
    main()
