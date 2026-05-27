import re
import random
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------ Load Data ------------------
def resolve_base_dir():
    """Find the folder that contains both Data and Master Data."""
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent]
    for candidate in candidates:
        if (candidate / "Data").exists() and (candidate / "Master Data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find 'Data' and 'Master Data' folders next to test.py or one level above."
    )


BASE_DIR = resolve_base_dir()
training = pd.read_csv(BASE_DIR / "Data" / "Training.csv")
testing  = pd.read_csv(BASE_DIR / "Data" / "Testing.csv")

