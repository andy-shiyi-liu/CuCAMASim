# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Environment variable to avoid interactive prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake make git gcc g++ gdb libhdf5-dev \
    python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone and install matio
RUN cd ~ && git clone git://git.code.sf.net/p/matio/matio && \
    cd matio && \
    cmake . && \
    cmake --build . && \
    cmake --install . 

# Install Python packages
RUN pip install --no-cache-dir numpy scipy scikit-learn pyyaml \
    scikit-image pandas openml plotly tqdm redmail matplotlib \
    black pytest ipykernel jupyterlab jupyterlab_code_formatter \
    python-dotenv ucimlrepo

# Install Pytorch for MNIST dataset
RUN pip install --no-cache-dir torch torchvision torchaudio