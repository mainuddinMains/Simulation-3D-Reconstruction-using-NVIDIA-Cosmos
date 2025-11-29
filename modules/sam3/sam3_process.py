#!/usr/bin/env python3

import os
import re

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor


def main():
    # --- 1. Setup Paths & Config ---
    scene = os.environ.get("SCENE_NAME")
    data_root = Path(os.environ.get("DATA_ROOT"))
    min_score = float(os.environ.get("SAM3_MIN_SCORE"))
    
    img_dir = data_root / scene / "images"
    out_dir = data_root / scene / "instance_mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = [p.strip() for p in Path("prompts.txt").read_text().splitlines() if p.strip()]

    # --- 2. Load & Sort Frames ---
    frame_files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )

    print(f"[SAM3] Scene='{scene}', Frames={len(frame_files)}, Prompts={prompts}")

    video_frames = [Image.open(f).convert("RGB") for f in frame_files]
    img_w, img_h = video_frames[0].size

    # --- 3. Initialize Model ---
    device = "cuda"
    print(f"Loading Model on {device}...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    session = processor.init_video_session(
        video=video_frames, 
        inference_device=device, 
        dtype=torch.bfloat16
    )

    for text in prompts:
        session = processor.add_text_prompt(session, text=text)

    # --- 4. Propagate & Save ---
    id_map = {} 
    next_label = 0

    print("Propagating...")
    for output in model.propagate_in_video_iterator(session):
        processed = processor.postprocess_outputs(session, output, original_sizes=[(img_h, img_w)])
        
        masks = processed["masks"].cpu().numpy()
        scores = processed["scores"].cpu().numpy()
        obj_ids = processed["object_ids"].cpu().numpy()

        final_mask = np.full((img_h, img_w), 255, dtype=np.uint8)

        valid_indices = np.where(scores >= min_score)[0]
        if len(valid_indices) == 0:
            Image.fromarray(final_mask).save(out_dir / frame_files[output.frame_idx].with_suffix(".png").name)
            continue

        current_masks = masks[valid_indices]
        if current_masks.ndim == 4:
            current_masks = current_masks.squeeze(1)
        
        areas = current_masks.reshape(len(valid_indices), -1).sum(axis=1)
        sort_order = np.argsort(areas)[::-1]

        for idx in sort_order:
            real_idx = valid_indices[idx]
            obj_id = int(obj_ids[real_idx])
            
            if obj_id not in id_map:
                if next_label >= 254:
                    continue
                id_map[obj_id] = next_label
                next_label += 1
            
            label = id_map[obj_id]
            mask_bool = current_masks[idx] > 0
            final_mask[mask_bool] = label

        save_name = frame_files[output.frame_idx].with_suffix(".png").name
        Image.fromarray(final_mask).save(out_dir / save_name)

    print(f"[SAM3] Done. Saved to: {out_dir}")


if __name__ == "__main__":
    main()