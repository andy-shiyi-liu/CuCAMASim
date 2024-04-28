from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np
from pathlib import Path
from scipy.io import loadmat

scriptDir = Path(__file__).parent


def load_original_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target variable (species)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


def load_data_2feature():
    dataset = np.load(scriptDir.joinpath(f"iris_2feature.npz"))
    (
        train_inputs,
        train_classes,
        validation_inputs,
        validation_classes,
        test_inputs,
        test_classes,
    ) = (
        dataset["train_inputs"],
        dataset["train_classes"],
        dataset["validation_inputs"],
        dataset["validation_classes"],
        dataset["test_inputs"],
        dataset["test_classes"],
    )

    return (
        train_inputs,
        train_classes,
        validation_inputs,
        validation_classes,
        test_inputs,
        test_classes,
    )


def save2mat():
    trainInputs, testInputs, trainLabels, testLabels = load_original_data()


def load_data():
    mat_load = loadmat(scriptDir / "iris.mat")

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

def load_data_normalized():
    mat_load = loadmat(scriptDir / "iris_normalized.mat")

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
