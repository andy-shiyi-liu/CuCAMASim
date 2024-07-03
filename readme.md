# CuCAMASim
A cpp/CUDA implementation of [CAMASim](https://github.com/menggg22/CAMASim), a comprehensive content-addressable accelerator simulation framework.

> this repo is still under development

## Setting up development environment
### Using dev container / docker
Using dev container is recommended. Steps:
1. install `Dev Containers` extension in vscode
2. install docker and nvidia-docker on host to enable contrainer and GPU access in container
3. copy `./devcontainer` to `./.devcontainer`
4. modify contents in dockerfile and json file, such as custom vscode extensions, etc.
5. run vscode command `Dev Containers: Open Folder in Container` and select the current folder to open it

You can also setup a docker container with dockerfile `devcontainer/Dockerfile` for development environment.

> Since CuCAMASim works with CUDA, you need to install CUDA on host and enable nvidia-docker on host.

### Ubuntu Linux
> The following content just follows the key setps in the dockerfile.

1. Check CUDA and nvcc are installed
```bash
nvidia-smi
nvcc --version
```
2. Install necessary apt packages.
```bash
sudo apt-get update && apt-get install -y cmake make git gcc g++ gdb libhdf5-dev python-is-python3 python3-pip
```
3. Install matio library. The matio library is used to read mat files, which we use to store the dataset. You can check the [matio github repo](https://github.com/tbeu/matio) for more details.
```bash
git clone git://git.code.sf.net/p/matio/matio
cd matio
cmake .
cmake --build .
cmake --install .
```
4. Install python dependencies.
```bash
pip install --no-cache-dir numpy scipy scikit-learn pyyaml \
    scikit-image pandas openml plotly tqdm redmail matplotlib \
    black pytest ipykernel jupyterlab jupyterlab_code_formatter \
    python-dotenv ucimlrepo torch torchvision torchaudio
```

> Please remember to change the `WORKSPACE_DIR` variable in `./.env` to the root path of the repository.

## Building CuCAMASim

> The following procedure is only tested in Ubuntu 20.04.

CuCAMASim uses cmake to manage the project. The `./main.cpp` provides an example on how to use CuCAMASim.
```bash
mkdir build && cd build
cmake ..
make
```
Then run `./build/CuCAMASim_runner` to run the example.

## Usage
CuCAMASim is a simulator for Content-Addressable Memory(CAM) and adopts a similar architecture as CAMASim[1].

We provide an example in which we use CuCAMASim to simulate ACAM-based decision tree inference process. Details could be found in [this doc](./doc/decision_tree_example.md). 

## Todo
- add python - cpp interface

## References
[1] M. Li, S. Liu, M. M. Sharifi, and X. S. Hu, “CAMASim: A Comprehensive Simulation Framework for Content-Addressable Memory based Accelerators.” arXiv, Mar. 05, 2024. Accessed: Mar. 07, 2024. [Online]. Available: http://arxiv.org/abs/2403.03442