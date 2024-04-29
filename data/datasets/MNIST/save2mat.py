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
    "MNIST_original",
    workspaceDir / "data/datasets/MNIST/MNIST.mat",
    normalize=False,
)

convert.datasetName2matFile(
    "MNIST_original",
    workspaceDir / "data/datasets/MNIST/MNIST_normalized.mat",
    normalize=True,
)

convert.datasetName2matFile(
    "MNIST_small_original",
    workspaceDir / "data/datasets/MNIST/MNIST_small.mat",
    normalize=False,
)

convert.datasetName2matFile(
    "MNIST_small_original",
    workspaceDir / "data/datasets/MNIST/MNIST_small_normalized.mat",
    normalize=True,
)