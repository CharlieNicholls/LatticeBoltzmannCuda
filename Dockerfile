FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt update -y && apt upgrade -y

RUN apt install -y --allow-change-held-packages \
                   libcublas-12-8 \
                   libcublas-dev-12-8 \
                   gcc \
                   g++ \
                   nvidia-container-toolkit \
                   cmake \
                   cuda-toolkit-12-8 \
                   libcgal-dev

RUN apt-get update && \     
    apt-get install -y x11-apps \
                       libglew-dev \
                       libglfw3 \
                       libglfw3-dev \
                       build-essential \
                       libgl1-mesa-glx \
                       libx11-dev \
                       libxrandr-dev \
                       libxinerama-dev \
                       libxcursor-dev \
                       libxi-dev \
                       mesa-utils \
                       x11-apps \
                       && rm -rf /var/lib/apt/lists/*

ENV DISPLAY=host.docker.internal:0.0