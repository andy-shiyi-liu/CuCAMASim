import sys
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

scriptDir = Path(__file__).parent
workspaceDir = Path(os.getenv("WORKSPACE_DIR"))

if not f"{workspaceDir}" in sys.path:
    sys.path.append(f"{workspaceDir}")

from run_script.util import *


def saveTreeText(datasetName: str):
    outputPath = scriptDir / f"{datasetName}.txt"
    trainInputs, testInputs, trainLabels, testLabels = util.loadDataset(datasetName)

    print("trainInputs.shape: ", trainInputs.shape)
    print("testInputs.shape: ", testInputs.shape)
    print("trainLabels.shape: ", trainLabels.shape)
    print("testLabels.shape: ", testLabels.shape)

    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    clf = DecisionTreeClassifier(
        # max_depth=5,
        criterion="entropy",
    )
    # clf = RandomForestClassifier(n_estimators=15, max_depth=10)
    clf.fit(trainInputs, trainLabels)
    pred = clf.predict(testInputs)
    accuracy_original = accuracy_score(testLabels, pred)
    print("DT Accuracy (original): ", accuracy_original)

    # save tree text trained by DecisionTreeClassifier
    with open(outputPath, "w") as f:
        f.write(tree.export_text(clf, max_depth=10000, decimals=20))
        print(f"Decision tree text saved to {outputPath}")


if __name__ == "__main__":
    saveTreeText("MNIST")
    saveTreeText("MNIST_normalized")
    saveTreeText("MNIST_small")
    saveTreeText("MNIST_small_normalized")
