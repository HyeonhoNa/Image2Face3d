# utils/ckpt_utils.py
import os
import tempfile
import torch

def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except:
            pass
        raise e


def _get_state_dict(m):
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()

def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def save_checkpoint(
    epoch, global_step, save_dir, tag,
    arc2face, liveportrait, appearance, xattn, decoder, id_mapper, optimizer=None,
    best_metric=None, fname=None,
):
    ckpt_dir = os.path.join(save_dir, "ckpts", fname)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "arc2face": _get_state_dict(arc2face),
        "liveportrait": _get_state_dict(liveportrait),
        "appearance" : _get_state_dict(appearance),
        "xattn": _get_state_dict(xattn),
        "decoder": _get_state_dict(decoder),
        "id_mapper": _get_state_dict(id_mapper),
        "best_metric": best_metric,
        "torch_version": torch.__version__,
        "decoder_active_sh_degree": int(getattr(decoder, "active_sh_degree", 0)),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    path = os.path.join(ckpt_dir, f"ckpt_{tag}.pt")
    _atomic_save(ckpt, path)
    return path


def load_checkpoint(
    path, arc2face, liveportrait, appearance, xattn, decoder, id_mapper, optimizer=None, strict=True, map_location="cuda"
):
    ckpt = torch.load(path, map_location=map_location)
    (arc2face.module if hasattr(arc2face, "module") else arc2face).load_state_dict(ckpt["arc2face"], strict=strict)
    (liveportrait.module if hasattr(liveportrait, "module") else liveportrait).load_state_dict(ckpt["liveportrait"], strict=strict)
    (appearance.module if hasattr(appearance, "module") else appearance).load_state_dict(ckpt["appearance"], strict=strict) 
    (xattn.module if hasattr(xattn, "module") else xattn).load_state_dict(ckpt["xattn"], strict=strict)
    (decoder.module if hasattr(decoder, "module") else decoder).load_state_dict(ckpt["decoder"], strict=strict)
    (id_mapper.module if hasattr(id_mapper, "module") else id_mapper).load_state_dict(ckpt["id_mapper"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    dec = _unwrap(decoder)
    deg = ckpt.get("decoder_active_sh_degree", None)
    if deg is not None:
        dec.active_sh_degree = int(deg)
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_metric = ckpt.get("best_metric", None)
    return epoch, global_step, best_metric, ckpt


def save_checkpoint2(
    epoch, global_step, save_dir, tag,
    arc2face, appearance, decoder, optimizer=None,
    best_metric=None, fname=None,
):
    ckpt_dir = os.path.join(save_dir, "ckpts", fname)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "arc2face": _get_state_dict(arc2face),
        "appearance" : _get_state_dict(appearance),
        "decoder": _get_state_dict(decoder),
        "best_metric": best_metric,
        "torch_version": torch.__version__,
        "decoder_active_sh_degree": int(getattr(decoder, "active_sh_degree", 0)),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    path = os.path.join(ckpt_dir, f"ckpt_{tag}.pt")
    _atomic_save(ckpt, path)
    return path


def load_checkpoint2(
    path, arc2face, appearance, decoder, optimizer=None, strict=True, map_location="cuda"
):
    ckpt = torch.load(path, map_location=map_location)
    (arc2face.module if hasattr(arc2face, "module") else arc2face).load_state_dict(ckpt["arc2face"], strict=strict)
    (appearance.module if hasattr(appearance, "module") else appearance).load_state_dict(ckpt["appearance"], strict=strict) 
    (decoder.module if hasattr(decoder, "module") else decoder).load_state_dict(ckpt["decoder"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    dec = _unwrap(decoder)
    deg = ckpt.get("decoder_active_sh_degree", None)
    if deg is not None:
        dec.active_sh_degree = int(deg)
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_metric = ckpt.get("best_metric", None)
    return epoch, global_step, best_metric, ckpt
