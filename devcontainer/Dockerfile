FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    htop tmux cmake make git gcc g++ gdb\
    && rm -rf /var/lib/apt/lists/*

# config git info
RUN git config --global pull.rebase false && \
    git config --global user.email "<your email>" && \
    git config --global user.name "<your name>" && \
    git config --global core.editor "code --wait"