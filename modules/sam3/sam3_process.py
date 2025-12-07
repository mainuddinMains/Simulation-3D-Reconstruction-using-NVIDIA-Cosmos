#!/usr/bin/env python3

import os
import re
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter, defaultdict
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor


def get_config():
    scene = os.environ["SCENE_NAME"]
    data_root = Path(os.environ["DATA_ROOT"])
    return {
        "scene": scene,
        "img_dir": data_root / scene / "images",
        "out_dir": data_root / scene / "instance_mask",
        "min_score": float(os.environ["SAM3_MIN_SCORE"]),
        "min_dur": float(os.environ["SAM3_MIN_FRAME_DURATION"]) / 100.0,
        "prompts": [p.strip() for p in Path("prompts.txt").read_text().splitlines() if p.strip()],
        "device": "cuda"
    }


def load_frames(img_dir):
    """Loads and numerically sorts video frames."""
    files = sorted(
        [f for f in img_dir.iterdir() if not f.name.startswith('.')],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )
    return [Image.open(f) for f in files], files


def save_mask(filepath, masks, ids, scores, id_map, h, w, min_score):
    """Composites overlapping masks using Painter's Algorithm (largest drawn first)."""
    final_mask = np.full((h, w), 255, dtype=np.uint8)
    
    valid_indices = [i for i, s in enumerate(scores) if s >= min_score and int(ids[i]) in id_map]
    
    if valid_indices:
        curr_masks = masks[valid_indices]
        curr_ids = ids[valid_indices]
        areas = curr_masks.sum(axis=(1, 2))
        for idx in np.argsort(areas)[::-1]:
            final_mask[curr_masks[idx] > 0] = id_map[int(curr_ids[idx])]

    Image.fromarray(final_mask).save(filepath)


def print_stats_table(prompts, prompt_counts, id_to_prompt, valid_ids, total_frames):
    """Generates the CLI summary table."""
    prompt_to_ids_all = defaultdict(list)
    for oid, p_text in id_to_prompt.items():
        prompt_to_ids_all[p_text].append(oid)

    print("\n" + "-" * 80)
    print(f"{'PROMPT':<25} | {'FRAMES':<12} | {'IDS (KEEP/ALL)':<15} | {'KEPT IDS'}")
    print("-" * 80)

    for p in prompts:
        p_count = prompt_counts.get(p, 0)
        all_ids = sorted(prompt_to_ids_all.get(p, []))
        kept_ids = [oid for oid in all_ids if oid in valid_ids]
        
        ratio_str = f"{len(kept_ids)}/{len(all_ids)}"
        ids_str = str(kept_ids) if len(kept_ids) < 8 else f"{len(kept_ids)} IDs"
        
        print(f"{p:<25} | {f'{p_count}/{total_frames}':<12} | {ratio_str:<15} | {ids_str}")

    print("-" * 80)


def main():
    cfg = get_config()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    frames, frame_files = load_frames(cfg["img_dir"])
    h, w = frames[0].size[::-1]
    total_frames = len(frames)

    model = Sam3VideoModel.from_pretrained("facebook/sam3", config=Sam3VideoConfig(recondition_on_trk_masks=True))
    model.to(cfg["device"], dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    session = processor.init_video_session(
        video=frames, inference_device=cfg["device"], processing_device="cpu", video_storage_device="cpu", dtype=torch.bfloat16
    )
    processor.add_text_prompt(session, cfg["prompts"])

    cached_data = []
    prompt_counts = Counter()
    id_counts = Counter()
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

            high_conf_ids = set(ids[scores >= cfg["min_score"]])
            
            for p_text, p_ids in prompt_map.items():
                active_ids = [oid for oid in p_ids if oid in high_conf_ids]
                if active_ids:
                    prompt_counts[p_text] += 1
                    for oid in active_ids:
                        id_counts[oid] += 1
                        id_to_prompt[oid] = p_text

            cached_data.append({
                "masks": masks, "ids": ids, "scores": scores, "file": frame_files[output.frame_idx]
            })

    min_frames_id = int(np.ceil(total_frames * cfg["min_dur"]))
    final_valid_ids = {oid for oid, c in id_counts.items() if c >= min_frames_id}

    print_stats_table(cfg["prompts"], prompt_counts, id_to_prompt, final_valid_ids, total_frames)

    if not final_valid_ids:
        print("No valid IDs found. Exiting.")
        return

    id_map = {oid: i for i, oid in enumerate(sorted(final_valid_ids)) if i < 255}
    
    print(f"Summary: Processed {total_frames} frames. Score > {cfg['min_score']}, Frame Duration > {cfg['min_dur']*100}%\n")
    print(f"Saving {len(final_valid_ids)} object masks...")
    for data in cached_data:
        save_mask(
            cfg["out_dir"] / data["file"].name, 
            data["masks"], data["ids"], data["scores"], 
            id_map, h, w, cfg["min_score"]
        )

    print(f"[SAM3] Done. Saved to {cfg['out_dir']}")


if __name__ == "__main__":
    main()