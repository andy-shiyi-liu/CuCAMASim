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

# datasetName = "BTSC_adapted_rand"
# datasetName = "gas"
# datasetName = "gas_normalized"
# datasetName = 'iris'
# datasetName = 'iris_normalized'
# datasetName = "survival"
datasetName = "survival_normalized"
outputPath = workspaceDir / "data/treeText" / f"{datasetName}.txt"



trainInputs, testInputs, trainLabels, testLabels = util.loadDataset(datasetName)

print(trainInputs.shape)
print(testInputs.shape)
print(trainLabels.shape)
print(testLabels.shape)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(
    max_depth=10,
    criterion="entropy",
)
# clf = RandomForestClassifier(n_estimators=15, max_depth=10)
clf.fit(trainInputs, trainLabels)
pred = clf.predict(testInputs)
accuracy_original = accuracy_score(testLabels, pred)
print("DT Accuracy (original): ", accuracy_original)


# save tree text trained by DecisionTreeClassifier
with open(outputPath, "w") as f:
    f.write(tree.export_text(clf, max_depth=10, decimals=20))
    print(f"Decision tree text saved to {outputPath}")
