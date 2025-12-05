#!/usr/bin/env python3

import os
import re
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor


def main():
    # --- 1. Config & Setup ---
    scene = os.environ["SCENE_NAME"]
    data_root = Path(os.environ["DATA_ROOT"])
    min_score = float(os.environ["SAM3_MIN_SCORE"])
    min_percent = float(os.environ["SAM3_MIN_FRAME_PERCENT"]) 
    device = "cuda"
    
    img_dir = data_root / scene / "images"
    out_dir = data_root / scene / "instance_mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = [p.strip() for p in Path("prompts.txt").read_text().splitlines() if p.strip()]
    frame_files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )
    frames = [Image.open(f).convert("RGB") for f in frame_files]
    h, w = frames[0].size[::-1]
    total_frames = len(frames)

    print(f"[SAM3] Processing {total_frames} frames.")
    print(f"[SAM3] Thresholds: Score >= {min_score}, Prompt Duration >= {min_percent}%")

    # --- 2. Run Inference ---
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    for text in prompts:
        session = processor.add_text_prompt(session, text=text)

    cached_results = []
    
    prompt_frame_counts = Counter() 
    id_frame_counts = Counter()     
    id_to_prompt = {}               

    print("Running inference...")
    with torch.inference_mode():
        for output in model.propagate_in_video_iterator(session):
            processed = processor.postprocess_outputs(session, output)
            
            scores = processed["scores"].cpu().numpy()
            ids = processed["object_ids"].cpu().numpy().astype(int)
            masks = processed["masks"].cpu().numpy().astype(np.uint8)
            prompt_map = processed["prompt_to_obj_ids"]
            
            if masks.ndim == 2: masks = masks[np.newaxis, ...]

            valid_indices = [i for i, s in enumerate(scores) if s >= min_score]
            high_conf_ids = set(ids[valid_indices])

            for p_text, p_ids in prompt_map.items():
                active_ids = [oid for oid in p_ids if oid in high_conf_ids]
                
                if active_ids:
                    prompt_frame_counts[p_text] += 1
                    
                    for oid in active_ids:
                        id_frame_counts[oid] += 1
                        id_to_prompt[oid] = p_text

            cached_results.append({
                "masks": masks, "ids": ids, "scores": scores, "file": frame_files[output.frame_idx]
            })

    # --- 3. Analysis ---
    min_frames_prompt = int(np.ceil(total_frames * (min_percent / 100.0)))
    valid_prompts = {p for p, c in prompt_frame_counts.items() if c >= min_frames_prompt}
    final_valid_ids = set()
    
    print("-" * 60)
    print(f"{'PROMPT':<20} | {'FRAMES':<10} | {'STATUS':<10} | {'IDs MERGED'}")
    print("-" * 60)
    
    for p in prompts:
        p_count = prompt_frame_counts[p]
        is_p_valid = p in valid_prompts
        status = "KEEP" if is_p_valid else "DROP"
        
        p_ids = [oid for oid, text in id_to_prompt.items() if text == p]
        p_ids_str = str(sorted(p_ids)) if len(p_ids) < 10 else f"{len(p_ids)} IDs"
        
        print(f"{p:<20} | {p_count}/{total_frames:<4} | {status:<10} | {p_ids_str}")
        
        if is_p_valid:
            final_valid_ids.update(p_ids)

    print("-" * 60)

    if not final_valid_ids:
        print("No valid IDs found. Exiting.")
        return

    id_map = {oid: i for i, oid in enumerate(sorted(final_valid_ids)) if i < 255}

    # --- 4. Generate Masks ---
    for res in cached_results:
        final_mask = np.full((h, w), 255, dtype=np.uint8)
        
        valid_indices = [
            i for i, s in enumerate(res["scores"]) 
            if s >= min_score and res["ids"][i] in id_map
        ]

        if valid_indices:
            curr_masks = res["masks"][valid_indices]
            curr_ids = res["ids"][valid_indices]
            areas = curr_masks.sum(axis=(1, 2))
            
            for idx in np.argsort(areas)[::-1]:
                mask_content = curr_masks[idx] > 0
                label = id_map[curr_ids[idx]]
                final_mask[mask_content] = label

        Image.fromarray(final_mask).save(out_dir / res["file"].name)

    print(f"[SAM3] Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()