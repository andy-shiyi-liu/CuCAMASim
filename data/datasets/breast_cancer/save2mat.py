import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

scriptDir = Path(__file__).parent
workspaceDir = Path(os.getenv("WORKSPACE_DIR"))

if not f"{workspaceDir}" in sys.path:
    sys.path.append(f"{workspaceDir}")

from run_script.util import *

convert.datasetName2matFile(
    "breast_cancer_original",
    workspaceDir / "data/datasets/breast_cancer/breast_cancer.mat",
    normalize=False,
)

convert.datasetName2matFile(
    "breast_cancer_original",
    workspaceDir / "data/datasets/breast_cancer/breast_cancer_normalized.mat",
    normalize=True,
)