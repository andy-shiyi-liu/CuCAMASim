from scipy.io import loadmat
from typing import Tuple
import numpy as np
from pathlib import Path

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mat_load = loadmat("data/datasets/BelgiumTSC/dataset_adapted.mat")

    image_size = 16
    train_inputs = mat_load["train_inputs"]
    test_inputs = mat_load["test_inputs"]

    train_inputs = train_inputs.reshape(
        (train_inputs.shape[0], image_size * image_size)
    )
    test_inputs = test_inputs.reshape((test_inputs.shape[0], image_size * image_size))

    train_classes = mat_load["train_classes"]
    test_classes = mat_load["test_classes"]

    train_inputs, test_inputs, train_classes, test_classes = (
        train_inputs,
        test_inputs,
        train_classes.flatten(),
        test_classes.flatten(),
    )

    return train_inputs, test_inputs, train_classes, test_classes

def load_rand_data(N_TRAIN:int, N_VALIDATION:int, N_TEST:int):
    scriptDir = Path(__file__).parent
    dataset = np.load(scriptDir.joinpath(f"{N_TRAIN}train_{N_VALIDATION}validation_{N_TEST}test.npz"))

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