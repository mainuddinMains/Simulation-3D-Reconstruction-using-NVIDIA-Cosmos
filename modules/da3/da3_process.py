#!/usr/bin/env python3

import os
import re
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from depth_anything_3.api import DepthAnything3


def get_config():
    scene = os.environ["SCENE_NAME"]
    data_root = Path(os.environ["DATA_ROOT"])
    output_root = Path(os.environ["OUTPUT_ROOT"])
    return {
        "scene": scene,
        "img_dir": data_root / scene / "images",
        "out_json": data_root / scene / "transforms.json",
        "out_dir": output_root / scene / "da3_out",
        "model_dir": "depth-anything/DA3NESTED-GIANT-LARGE",
        "device": "cuda"
    }


def compute_intrinsics(preds, orig_w, orig_h):
    """Scales intrinsics to original resolution, computes median values, and derives FOV."""
    _, proc_h, proc_w = preds.depth.shape 
    scale_x, scale_y = orig_w / proc_w, orig_h / proc_h

    Ks = preds.intrinsics.astype(np.float32)
    
    fx = Ks[:, 0, 0] * scale_x
    fy = Ks[:, 1, 1] * scale_y
    cx = Ks[:, 0, 2] * scale_x
    cy = Ks[:, 1, 2] * scale_y

    params = np.stack([fx, fy, cx, cy], axis=1)
    
    valid_mask = np.isfinite(params).all(axis=1) & (params[:, 0] > 1e-3)
    if not valid_mask.any():
        raise RuntimeError("No valid intrinsics found.")
        
    final_fx, final_fy, final_cx, final_cy = np.median(params[valid_mask], axis=0)
    angle_x = 2.0 * np.arctan((orig_w / 2.0) / final_fx)

    return final_fx, final_fy, final_cx, final_cy, angle_x


def compute_extrinsics(preds):
    """Converts W2C to C2W"""
    w2c = preds.extrinsics.astype(np.float32)
    n_frames = w2c.shape[0]

    R_w2c = w2c[:, :3, :3]
    t_w2c = w2c[:, :3, 3:4]

    R_c2w = R_w2c.transpose(0, 2, 1)
    t_c2w = -R_c2w @ t_w2c

    c2w = np.zeros((n_frames, 4, 4), dtype=np.float32)
    c2w[:, 3, 3] = 1.0 
    c2w[:, :3, :3] = R_c2w
    c2w[:, :3, 3:4] = t_c2w
    
    floor_fix_matrix = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float32)
    c2w = floor_fix_matrix @ c2w

    c2w[:, :3, 3] -= c2w[:, :3, 3].mean(axis=0)

    return c2w


def normalize_cameras(c2w):
    """
    Centers the scene and aligns the dataset principal axes to World X, Y, Z.
    """
    c2w[:, :3, 3] -= c2w[:, :3, 3].mean(axis=0)

    _, _, vh = np.linalg.svd(c2w[:, :3, 3])
    R_new = vh

    avg_cam_y = c2w[:, :3, 1].mean(axis=0) 
    if np.dot(R_new[2, :], avg_cam_y) > 0: 
        R_new[2, :] *= -1
        R_new[1, :] *= -1

    c2w[:, :3, :3] = R_new @ c2w[:, :3, :3]
    c2w[:, :3, 3] = (R_new @ c2w[:, :3, 3].T).T
    
    return c2w


def main():
    cfg = get_config()
    
    files = sorted(
        [f for f in cfg["img_dir"].iterdir() if not f.name.startswith('.')],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )
    
    with Image.open(files[0]) as img:
        orig_w, orig_h = img.size

    print(f"[DA3] Scene: {cfg['scene']} | Frames: {len(files)} | Res: {orig_w}x{orig_h}")
    
    model = DepthAnything3.from_pretrained(cfg["model_dir"]).to(cfg["device"])
    
    print(f"Running inference...")
    with torch.inference_mode(), torch.autocast(device_type=cfg["device"], dtype=torch.bfloat16):
        preds = model.inference(
            image=[str(f) for f in files],
            export_dir=str(cfg["out_dir"]),
            export_format="glb",
            use_ray_pose=True
        )

    fx, fy, cx, cy, angle_x = compute_intrinsics(preds, orig_w, orig_h)
    c2w_matrices = compute_extrinsics(preds)
    c2w_matrices = normalize_cameras(c2w_matrices)

    frames_json = [
        {"file_path": f"images/{f.name}", "transform_matrix": m.tolist()}
        for f, m in zip(files, c2w_matrices)
    ]

    out_data = {
        "camera_angle_x": float(angle_x),
        "fl_x": float(fx), "fl_y": float(fy),
        "cx": float(cx),   "cy": float(cy),
        "w": int(orig_w),  "h": int(orig_h),
        "frames": frames_json,
    }

    cfg["out_json"].write_text(json.dumps(out_data, indent=4))
    print(f"[DA3] Done. Saved transforms to {cfg['out_json']}")


if __name__ == "__main__":
    main()