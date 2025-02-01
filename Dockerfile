FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt update -y && apt upgrade -y

RUN apt install -y gcc \
                   g++ \
                   nvidia-container-toolkit \
                   cmake \
                   cuda-toolkit-12-8 \
                   libcgal-dev