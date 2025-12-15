FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gpg \
    curl \
    ffmpeg \
    lsb-release \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | gpg --dearmor -o /etc/apt/keyrings/librealsense.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.gpg] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list
RUN apt-get update && apt-get install -y --no-install-recommends librealsense2-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app/preprocessor

COPY preprocess.sh .