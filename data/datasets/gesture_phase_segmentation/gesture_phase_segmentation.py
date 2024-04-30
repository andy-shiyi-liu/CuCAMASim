from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from pathlib import Path
import urllib.request
import pandas as pd

scriptDir = Path(__file__).parent

rawDataDir = scriptDir / "raw"


def download_dataset():
    base_url = "https://archive.ics.uci.edu/static/public/302/"
    files = [
        "gesture+phase+segmentation.zip",
    ]
    rawDataDir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if (rawDataDir / file).exists():
            # print(f"{file} already exists.")
            continue
        print(f"Downloading {rawDataDir / file}...")
        urllib.request.urlretrieve(base_url + file, rawDataDir / file)
        # unzip the dataset
        import zipfile

        for file in files:
            with zipfile.ZipFile(rawDataDir / file, "r") as zip_ref:
                zip_ref.extractall(rawDataDir)
    # print("Download complete.")


def load_original_dataset():
    download_dataset()

    # load data in all csv files
    data = []
    for file in rawDataDir.glob("*_va3.csv"):
        data.append(pd.read_csv(file))
    data = pd.concat(data)
    # print(data)

    # convert data to X and y
    X = data.drop(columns=["Phase"])
    y = data["Phase"]

    # print(X)
    # print(y)

    # change str label to int label
    labelUID = 0
    for label in y.unique():
        y = y.replace(label, labelUID)
        labelUID += 1

    # convert to numpy array and do train test split
    X = X.to_numpy()
    y = y.to_numpy()
    trainInputs, testInputs, trainLabels, testLabels = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return trainInputs, testInputs, trainLabels, testLabels


def load_normalized_dataset():
    mat_load = loadmat(scriptDir / "gesture_phase_segmentation_normalized.mat")

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
    mat_load = loadmat(scriptDir / "gesture_phase_segmentation.mat")

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

