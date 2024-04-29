import os
import urllib.request
import gzip
import numpy as np
from pathlib import Path
from scipy.io import loadmat

scriptDir = Path(__file__).parent
rawDataDir = scriptDir / "MNIST/raw"


def download_mnist():
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    rawDataDir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if (rawDataDir / file).exists():
            # print(f"{file} already exists.")
            continue
        print(f"Downloading {rawDataDir / file}...")
        urllib.request.urlretrieve(base_url + file, rawDataDir / file)
    # print("Download complete.")


def loadImages(filename) -> np.ndarray:
    with gzip.open(filename, "rb") as f:
        # Skip the magic number and dimensions information
        f.read(16)
        # Each image is 28x28, and there are 60,000 training images and 10,000 test images
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28 * 28)


def loadLabels(filename) -> np.ndarray:
    with gzip.open(filename, "rb") as f:
        # Skip the magic number
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_original_dataset():
    download_mnist()
    trainInputs = loadImages(rawDataDir / "train-images-idx3-ubyte.gz")
    trainLabels = loadLabels(rawDataDir / "train-labels-idx1-ubyte.gz")
    testInputs = loadImages(rawDataDir / "t10k-images-idx3-ubyte.gz")
    testLabels = loadLabels(rawDataDir / "t10k-labels-idx1-ubyte.gz")

    return trainInputs, testInputs, trainLabels, testLabels

def load_original_dataset_small_test():
    download_mnist()
    trainInputs = loadImages(rawDataDir / "train-images-idx3-ubyte.gz")
    trainLabels = loadLabels(rawDataDir / "train-labels-idx1-ubyte.gz")
    testInputs = loadImages(rawDataDir / "t10k-images-idx3-ubyte.gz")
    testLabels = loadLabels(rawDataDir / "t10k-labels-idx1-ubyte.gz")

    # shuffle testInputs and testLabels in the same way
    np.random.seed(0)
    idx = np.random.permutation(len(testInputs))
    testInputs = testInputs[idx]
    testLabels = testLabels[idx]

    # take only 1000 samples from testInputs and testLabels
    testInputs = testInputs[:1000]
    testLabels = testLabels[:1000]

    return trainInputs, testInputs, trainLabels, testLabels


def download_original_dataset_torch():
    import torch
    from torchvision import datasets, transforms
    import numpy as np

    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download and load the training data
    trainset = datasets.MNIST(
        scriptDir,
        download=True,
        train=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False
    )

    # Download and load the test data
    testset = datasets.MNIST(
        scriptDir,
        download=True,
        train=False,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False
    )

    # Convert train dataset to NumPy arrays
    trainInputs, trainLabels = next(iter(trainloader))
    trainInputs = trainInputs.numpy()
    trainLabels = trainLabels.numpy()

    # Convert test dataset to NumPy arrays
    testInputs, testLabels = next(iter(testloader))
    testInputs = testInputs.numpy()
    testLabels = testLabels.numpy()

    assert isinstance(trainInputs, np.ndarray)
    assert isinstance(trainLabels, np.ndarray)
    assert isinstance(testInputs, np.ndarray)
    assert isinstance(testLabels, np.ndarray)

    print(f"Training set images shape: {trainInputs.shape}")
    print(f"Training set labels shape: {trainLabels.shape}")
    print(f"Test set images shape: {testInputs.shape}")
    print(f"Test set labels shape: {testLabels.shape}")

    # assert len(trainInputs.shape) == 4
    # trainInputs.reshape(
    #     trainInputs.shape[0], trainInputs[1] * trainInputs[2] * trainInputs[3]
    # )
    # assert len(testInputs.shape) == 4
    # testInputs.reshape(
    #     testInputs.shape[0], testInputs[1] * testInputs[2] * testInputs[3]
    # )

    # print(f"Training set images shape: {trainInputs.shape}")
    # print(f"Training set labels shape: {trainLabels.shape}")
    # print(f"Test set images shape: {testInputs.shape}")
    # print(f"Test set labels shape: {testLabels.shape}")

def load_normalized_dataset():
    mat_load = loadmat(scriptDir / "MNIST_normalized.mat")

    trainInputs = mat_load["trainInputs"]
    testInputs = mat_load["testInputs"]

    trainLabels = mat_load["trainLabels"]
    testLabels = mat_load["testLabels"]

    trainInputs, testInputs, trainLabels, testLabels = (
        trainInputs.T,
        testInputs.T,
        trainLabels.flatten(),
        testLabels.flatten(),
    )

    return trainInputs, testInputs, trainLabels, testLabels

def load_dataset():
    mat_load = loadmat(scriptDir / "MNIST.mat")

    trainInputs = mat_load["trainInputs"]
    testInputs = mat_load["testInputs"]

    trainLabels = mat_load["trainLabels"]
    testLabels = mat_load["testLabels"]

    trainInputs, testInputs, trainLabels, testLabels = (
        trainInputs.T,
        testInputs.T,
        trainLabels.flatten(),
        testLabels.flatten(),
    )

    return trainInputs, testInputs, trainLabels, testLabels

def load_normalized_dataset_small_test():
    mat_load = loadmat(scriptDir / "MNIST_small_normalized.mat")

    trainInputs = mat_load["trainInputs"]
    testInputs = mat_load["testInputs"]

    trainLabels = mat_load["trainLabels"]
    testLabels = mat_load["testLabels"]

    trainInputs, testInputs, trainLabels, testLabels = (
        trainInputs.T,
        testInputs.T,
        trainLabels.flatten(),
        testLabels.flatten(),
    )

    return trainInputs, testInputs, trainLabels, testLabels

def load_dataset_small_test():
    mat_load = loadmat(scriptDir / "MNIST_small.mat")

    trainInputs = mat_load["trainInputs"]
    testInputs = mat_load["testInputs"]

    trainLabels = mat_load["trainLabels"]
    testLabels = mat_load["testLabels"]

    trainInputs, testInputs, trainLabels, testLabels = (
        trainInputs.T,
        testInputs.T,
        trainLabels.flatten(),
        testLabels.flatten(),
    )

    return trainInputs, testInputs, trainLabels, testLabels


if __name__ == "__main__":
    # download_original_dataset_torch()
    load_original_dataset()
