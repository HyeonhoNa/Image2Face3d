name = "MonoFace: Monocular Video-Driven 3DGS Face Animation"
simple_name = "MonoFace"

# File Paths

dataset_root = ""

drv_indexed_root = ""

drv_frame_glob = "*.png"
single_cam_path = "data/converted_labels_25d.json"
save_path = "results"

# Experiments
template_mesh_path = "data/FLAME2023/head_template_mesh.obj"
texture_path = "data/FLAME2023/tex_mean_painted.png"
lr = {'af': 1e-4,
        'lp': 1e-4,
        'dt': 5e-5,
        'dd': 5e-4,
        'idmap': 1e-4}

num_points = 300000

vid_window_size = 1
sample_view = 8
batch_size = 1
drv_stride = 30
metric_downsample = 512

w_face = 0.9
w_background = 0.1
tex_warmup_iters = 5000

stage0_iters = 0
stage1_iters = 200000
stage2_iters = 0

sh_interval = 1000

render_chunk = 8 

# Models
id_dim = 768
exp_dim = 50 #SmirkEXPEncoder
app_dim = 768
hidden_dim = 256
emb_dim = 128

# Appearance
app_backbone = "auto"   # auto|dinov2_s14|vit_b_16|resnet50
app_input_size = 518
app_l2norm = True
app_dropout = 0.0

# Debugging
export_ply = False
save_interval = 10000
save_aux = True
max_keep = 5
preview_interval = 5000
preview_burst = 1
# resume_path = ""
resume_path = "ckpt_iter100000.pt"
eval_interval = 5000

num_workers = 10
prefetch_factor = 2
persistent_workers = True
pin_memory = True
use_amp = True

#id mapping warmup
idmap_warmup_steps = 10000
idmap_warmup_lr = 1e-4
idmap_warmup_lam_in = 0.0     
idmap_warmup_lam_front = 1.0    
idmap_warmup_lam_norm = 0.1

src_mv_random_per_iter = True      
src_mv_front_prob      = 0.25     
src_mv_allow_any_if_no_ok = True   
arcface_banned_mv_views = set()     

exclude_id_patterns_by_split = {
    "val": [r"_(E09|E10)(?:$|_)"], 
    # "train": [r"_(E09|E10)(?:$|_)"],
    "test":  [r"_(E09|E10)(?:$|_)"],
}

lam_leak_id_on_app   = 0.01   
lam_leak_app_on_geom = 0.02
lam_orth_app         = 0.01

mvgen = dict(
    enable=True,         
    img_size=256,
    cond_dim=1536,       
    view_dim=12,          
    lr=2e-4, wd=0.0,
    lambda_id=1.0,
    lambda_var=0.1,
    eval_views=[0,90,180,270],
)
disable_3dgs = True

seed_root = ""
src_use_src_folder = False