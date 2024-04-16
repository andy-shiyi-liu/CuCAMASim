# CuCAMASim
A cpp/CUDA implementation of [CAMASim](https://github.com/menggg22/CAMASim), a comprehensive content-addressable accelerator simulation framework.

> this repo is still under development

## Setting up development environment
Using dev container is recommended. Steps:
1. install `Dev Containers` extension in vscode
2. install docker and nvidia-docker on host to enable contrainer and GPU access in container
3. copy `./devcontainer` to `./.devcontainer`
4. modify contents in dockerfile and json file, such as custom vscode extensions, etc.
5. run vscode command `Dev Containers: Open Folder in Container` and select the current folder to open it

## Todo
- finish CuCAMASim::query()
  - save results to simResult
  - change data allocation and free location
  - change error code copy location
- init PerformanceEvaluator