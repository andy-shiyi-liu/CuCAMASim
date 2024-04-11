# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Environment variable to avoid interactive prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    htop tmux cmake make git gcc g++ gdb libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and install matio
RUN cd ~ && git clone git://git.code.sf.net/p/matio/matio && \
    cd matio && \
    cmake . && \
    cmake --build . && \
    cmake --install .
