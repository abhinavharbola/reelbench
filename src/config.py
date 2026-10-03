import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", "data/processed"))
RESULTS_DIR = Path(os.environ.get("RECSYS_RESULTS_DIR", "results"))
MODELS_DIR = DATA_DIR / "models"
