from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

scriptDir = Path(__file__).parent

def load_original_dataset():
    # fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    
    # metadata 
    # print(breast_cancer_wisconsin_diagnostic.metadata) 
    
    # variable information 
    # print(breast_cancer_wisconsin_diagnostic.variables) 

    # convert string labels to numerical labels
    le = LabelEncoder()
    y = le.fit_transform(y.values.ravel())
    
    X = X.to_numpy()

    trainInputs, testInputs, trainLabels, testLabels = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return trainInputs, testInputs, trainLabels, testLabels


def load_normalized_dataset():
    mat_load = loadmat(scriptDir / "breast_cancer_normalized.mat")

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
    mat_load = loadmat(scriptDir / "breast_cancer.mat")

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
    load_original_dataset()

