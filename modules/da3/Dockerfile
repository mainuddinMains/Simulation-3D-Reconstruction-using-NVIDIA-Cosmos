FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-dev \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app/da3
RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git .
RUN sed -i '/xformers/d' pyproject.toml
RUN pip install --break-system-packages .

COPY da3_process.py .