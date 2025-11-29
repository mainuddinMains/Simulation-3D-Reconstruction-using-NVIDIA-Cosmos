# Video2Sim

Video2Sim is a Docker-based pipeline for converting raw video (or sensor logs)
into simulation-ready 3D assets using a sequence of modular components.


## Requirements
- NVIDIA GPU (16GB VRAM minimum)
- Docker Engine
- A Hugging Face access token with permission to pull from:
  - `https://huggingface.co/facebook/sam3`


## Pipeline Overview

### Preprocessor

Takes a video or ROS bag and outputs images.

### DA3 (Depth Anything 3)

Reads extracted frames and produces a NeRF‑style `transforms.json`
containing camera intrinsics and per-frame camera poses.

### SAM 3 (Segment Anything Model 3)

Reads extracted frames and produces consistent per‑frame instance masks.
Prompts are defined in `prompts.txt`.

### HoloScene

Uses extracted frames, poses, masks, and Marigold‑generated priors
to reconstruct the 3D scene and export final assets.


## Running the Pipeline

1. **Place input files**  
Place `.mp4` or `.bag` into `data/input`.

2. **Configure environment**  
Fill out `.env` (scene name, Hugging Face token, etc.).

3. **Build and run each module in sequence**  
    ```bash
    docker compose up --build preprocessor
    docker compose up --build da3
    docker compose up --build sam3
    docker compose up --build holoscene
    ```

4. **Retrieve results**  
Results will be generated into `data/output`.

**Note:** You can temporarily set `CMD ["tail", "-f", "/dev/null"]` in a
module’s `Dockerfile` if you need to keep the container running for debugging.


## Citations

```bibtex
@article{depthanything3,
  title         = {Depth Anything 3: Recovering the visual space from any views},
  author        = {Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal       = {arXiv preprint arXiv:2511.10647},
  year          = {2025}
}
```

```bibtex
@inproceedings{sam3_2025,
  title         = {SAM 3: Segment Anything with Concepts},
  author        = {Anonymous authors},
  booktitle     = {Submitted to ICLR 2026},
  year          = {2025},
  url           = {https://openreview.net/forum?id=r35clVtGzw},
  note          = {Paper ID: 4183, under double-blind review}
}
```

```bibtex
@misc{xia2025holoscene,
  title         = {HoloScene: Simulation-Ready Interactive 3D Worlds from a Single Video}, 
  author        = {Hongchi Xia and Chih-Hao Lin and Hao-Yu Hsu and Quentin Leboutet and Katelyn Gao and Michael Paulitsch and Benjamin Ummenhofer and Shenlong Wang},
  year          = {2025},
  eprint        = {2510.05560},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2510.05560}, 
}
```