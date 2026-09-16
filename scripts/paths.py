import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
LOG_DIR = os.path.join(ROOT_DIR, "runs")
DOCUMENT_DIR = os.path.join(ROOT_DIR, "documents")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
