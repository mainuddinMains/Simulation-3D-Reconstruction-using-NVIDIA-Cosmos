FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-dev \
    ninja-build \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages \
    einops \
    decord \
    pycocotools \
    psutil \
    accelerate \
    kernels

RUN pip install --break-system-packages git+https://github.com/huggingface/transformers.git

WORKDIR /app/sam3
RUN git clone https://github.com/facebookresearch/sam3.git .
RUN pip install --break-system-packages .

COPY sam3_process.py .