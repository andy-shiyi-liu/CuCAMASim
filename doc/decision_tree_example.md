# Using CuCAMASim for Decision Tree Inference Simulation

## Introduction

This document provides a step-by-step guide on how to use CuCAMASim for decision tree inference simulation. CuCAMASim is a CUDA-based implementation of CAMASim, a decision tree inference simulator.

## Quick Start

### Setup Environment and Build CuCAMASim
Please refer to [README](../readme.md).

### Run Simulation

Assume the built executable is at `build/CuCAMASim_runner`, then simply run the simulation by `./build/CuCAMASim_runner`.

#### Choosing Task
You can choose from a list of implemented tasks for simulation. Currently, we have implemented the following tasks:
- **CAM-based decision tree inference**: In this task, we use CuCAMASim to simulate the decision tree inference with CAM. You can change the setting in the config file to simulate CAM inference accuracy under different noise setups and mapping strategies. A noise-free software-based inference is also performed at the begining of the task for comparing the inference accuracy with CAM-based inference. You can use `--task CAM_inference` to perform this task. This is the default task.
- **Software-based decision tree inference**: In this task, we perform software-based decision tree inference without CAM. The inference process can be noisy, according to the content in the config file. You can use `--task software_inference` to perform this task. Please also explicitly select an corresponding software inference config file for this task.
- **Statistics for error distribution**: In this task, we do the statistics for the error distribution of the noisy inference process. We calculate the proportion of each error type, e.g. no-match error, one-match error, etc. You can use `--task errDistr` to perform this task. Since randomness is involved, you can also specify a `--sample_time <number of times>` argument to repeat the simulation for a given number of times and give a final statistical result.
- **Print information**: In this task, we print basic informations of the dataset, trained tree, etc. You can use `--task print_info` to perform this task.

#### Choosing Config File
You can specify a config file for either CAM-based or software-based inference as an argument when running the executable. For example, to run the simulation with a specific config file, you would run:
```bash
./build/CuCAMASim_runner --config <config_file_path>
```

The config file type (CAM config/software inference config) should match the task type.

#### Choosing Dataset
You can choose from a list of implemented datasets for simulation. Currently, we have implemented the following datasets:
- Belgium TSC
- Breast Cancer Wisconsin (Diagnostic)
- Eye-movements
- gas_concentrations
- Gesture Phase Segmentation
- iris
- MNIST
- Haberman's Survival

> The dataset details including how they are loaded and stored can be found in `data/datasets` folder.

To run the simulation with a specific dataset, you need to specify the dataset name as an argument when running the executable. For example, to run the simulation with the MNIST dataset, you would run:
```bash
./build/CuCAMASim_runner --dataset MNIST
```

> The dataset name passed as an argument might be different with its original name. Generally, the dataset name passed as argument should be the same with the dataset folder name in `data/datasets`. You can check the `datasetPath` variable in `loadDataset()` function in `CuCAMASim/util/data.cpp` to check all avaliable datasets.

For most datasets, we provide two versions: original and normalized. Both versions use a 70%-30% train-test split. The original version is not pre-processed. For the normalized version, we pre-process by normalizing all features to the range [0, 1]. This means the minimum and maximum values of each feature are scaled to 0 and 1, respectively. This normalization aims to enhance robustness against ACAM device variation. To use the normalized version, add suffix `_normalized` to the dataset name for most datasets. For example, you would run the following command for the normalized version of MNIST dataset:
```bash
./build/CuCAMASim_runner --dataset MNIST_normalized
```

By default, the adapted version of `Belgium TSC` dataset is used (equivalently `--datset BTSC_adapted`).

#### Choosing Tree Text
You can choose a different decision tree than the default one for simulation. The usage is as following:
```bash
./build/CuCAMASim_runner --use_trained_tree <tree_text_file_path>
```

> If this argument is not specified, the default tree text for each dataset will be used. The default tree text for each dataset can be found in `getTreeTextPath()` function in `applications/DecisionTree/dt2cam.cpp`

## Extending the Example
### Adding New Datasets
When simulating a decision tree inference task, two elements are required: dataset and decision tree text. Therefore, adding new dataset is a matter of preparing these two elements and correponding configurations for CuCAMASim simulation.

#### Preparing Dataset
The dataset should be in a format that CuCAMASim can read. This format is a `.mat` file which was originally used by MATLAB to store matrix data. We use [matio](https://github.com/tbeu/matio) library for reading the `.mat` file. The `.mat` file should contain 4 matrices, namely `trainInputs`, `testInputs`, `trainLabels`, and `testLabels`.

For the implemented datasets, we use a seperate python script to generate the `.mat` file. The following content uses the preparation of "Breast Cancer Wisconsin (Diagnostic)" dataset as an example.

The first step is to load dataset from the UCI ML repo, making a train/test split and formatting the data. In the end, 4 numpy arrays should be given, which contains `trainInputs`, `testInputs`, `trainLabels`, and `testLabels`. The `trainInputs` and `testInputs` are 2-dimensional arrays, where each row is a sample and each column is a feature. The `trainLabels` and `testLabels` are 1-dimensional arrays, where each element is the label of the corresponding sample. This step is done in `load_original_dataset()` in `data/datasets/breast_cancer/breast_cancer.py`.

The second step is to store the datasets in a `.mat` file. We recommand using `scipy.io.savemat()` function with the following code. Please note that we use the transpose of `trainInputs` and `testInputs` to make the format consistent with the original `.mat` file. 
```py
scipy.io.savemat(
    outputPath,
    {
        "trainInputs": trainInputs.T.astype(np.float64),
        "testInputs": testInputs.T.astype(np.float64),
        "trainLabels": trainLabels.astype(np.uint64),
        "testLabels": testLabels.astype(np.uint64),
    },
)
```
For the second step, we provide a wrapper function, i.e. `convert.datasetName2matFile()` in `run_script/util.py`. Before using this function, you also need to add support for the new dataset in `util.loadDataset()` function in `run_script/util.py`. This step may seems redundant but recommanded, since the `util.loadDataset()` function will be used for training a new decision tree on the new dataset. Also, a similar function provides dataset loading support for `CAMASim`(`EVACAMx`) so it may save some time when you want to run the new datset on both `CAMASim` and `CuCAMASim`. After adding support in `util.loadDataset()`, you can simply provide the name for the dataset and the output path for the `.mat` file in the wrapper function and let the script do the rest. You may also normalize the values of each feature in the dataset by specifying `normalize=True` argument when calling `convert.datasetName2matFile()`. This step is done in `data/datasets/breast_cancer/save2mat.py`

In the end, you should get a `.mat` file which contains `trainInputs`, `testInputs`, `trainLabels`, and `testLabels`.

#### Preparing Tree Text
Another thing is to train a new decision tree and prepare the tree text for inference. We recommand using `scikit-learn` library for training the decision tree and save the tree text. For "Breast Cancer Wisconsin (Diagnostic)" dataset, this is done by `data/treeText/breast_cancer/saveTreeText.py` script.

#### Adding Support for New Dataset in CPP Code
After preparing the `.mat` dataset file and the tree text, you need to add relative information in the code. To be specific, you need to add information about the dataset name to the `.mat` file path/the tree text path to the code.

The dataset name to the `.mat` file path mapping is done in `loadDataset()` function in `CuCAMASim/util/data.cpp`, and the dataset name to the tree text path mapping is done in `getTreeTextPath()` function in `applications/DecisionTree/dt2cam.cpp`. You need to add a new key-value pair in the correspinding variable in both functions.

For the exemplar "Breast Cancer Wisconsin (Diagnostic)" dataset, you need to add something like
```cpp
{"breast_cancer", "/workspaces/CuCAMASim/data/datasets/breast_cancer/breast_cancer.mat"}
```
in `datasetPath` variable in `loadDataset()` and add something like
```cpp
{"breast_cancer", "/workspaces/CuCAMASim/data/treeText/breast_cancer/breast_cancer.txt"}
```
in `treeTextPath` variable in `getTreeTextPath()`.

#### Run Simulation on New Dataset

After adding support for the new dataset in the code, you need to re-compile the code. Then, you can simply run the simulation on the new dataset by providing the dataset name.

```bash
./build/CuCAMASim_runner --dataset breast_cancer
```