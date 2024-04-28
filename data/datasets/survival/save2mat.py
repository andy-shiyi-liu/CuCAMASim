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
    "survival_original",
    workspaceDir / "data/datasets/survival/survival.mat",
    normalize=False,
)

convert.datasetName2matFile(
    "survival_original",
    workspaceDir / "data/datasets/survival/survival_normalized.mat",
    normalize=True,
)
