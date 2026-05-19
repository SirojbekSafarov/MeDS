import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))               # dinomaly/ for dataset.py, utils.py, models/
sys.path.insert(0, os.path.join(_here, "..", "..", "shared"))  # MeDS/shared/ for dinov1, dinov2, beit, optimizers
