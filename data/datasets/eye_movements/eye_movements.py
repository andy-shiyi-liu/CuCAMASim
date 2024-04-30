import openml
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path

scriptDir = Path(__file__).parent

def load_original_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = openml.datasets.get_dataset(1044)

    X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

    # print(X.shape)
    # print(y.shape)

    # split dataset into train and test set
    trainInputs, testInputs, trainLabels, testLabels = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    assert isinstance(trainInputs, pd.DataFrame)
    assert isinstance(testInputs, pd.DataFrame)
    assert isinstance(trainLabels, pd.Series)
    assert isinstance(testLabels, pd.Series)

    trainInputs = trainInputs.to_numpy().astype(np.float64)
    testInputs = testInputs.to_numpy().astype(np.float64)
    trainLabels = trainLabels.to_numpy().astype(np.int64)
    testLabels = testLabels.to_numpy().astype(np.int64)
    
    return trainInputs, testInputs, trainLabels, testLabels


def load_normalized_dataset():
    mat_load = loadmat(scriptDir / "eye_movements_normalized.mat")

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
    mat_load = loadmat(scriptDir / "eye_movements.mat")

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

# if __name__ == "__main__":
#     load_original_dataset()
