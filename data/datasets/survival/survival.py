from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from pathlib import Path

scriptDir = Path(__file__).parent

def load_original_dataset():
    # fetch dataset
    haberman_s_survival = fetch_ucirepo(id=43)

    # data (as pandas dataframes)
    X = haberman_s_survival.data.features
    y = haberman_s_survival.data.targets

    # metadata
    # print(haberman_s_survival.metadata)

    # variable information
    # print(haberman_s_survival.variables)


    X = X.to_numpy()
    y = y.to_numpy()


    trainInputs, testInputs, trainLabels, testLabels = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return trainInputs, testInputs, trainLabels, testLabels

def load_normalized_dataset():
    mat_load = loadmat(scriptDir / "survival_normalized.mat")

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
    mat_load = loadmat(scriptDir / "survival.mat")

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