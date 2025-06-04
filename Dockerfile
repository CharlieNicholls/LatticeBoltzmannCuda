FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt update -y && apt upgrade -y

RUN apt install -y gcc \
                   g++ \
                   nvidia-container-toolkit \
                   cmake \
                   cuda-toolkit-12-8 \
                   libcgal-dev

RUN apt-get update && \     
    apt-get install -y x11-apps \
                       libglew-dev \
                       libglfw3 \
                       libglfw3-dev

ENV DISPLAY=host.docker.internal:0.0