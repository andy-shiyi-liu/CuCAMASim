FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    htop tmux cmake make git\
    && rm -rf /var/lib/apt/lists/*