from pathlib import Path
import sys
import gc
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.data_utils import load_s1_fast_slice
from src.metrics import load_ground_truth
from src.normalization import normalize_text, first_token

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

s1_dev, slice_ids = load_s1_fast_slice(
    TRAIN_DIR / "train_source1.tsv",
    OUTPUT_DIR / "s1_splits.tsv",
)

s1_dev["name_norm"] = s1_dev["business_name"].map(normalize_text)
s1_dev["first_token"] = s1_dev["name_norm"].map(first_token)

dev_gt_dict = load_ground_truth(
    TRAIN_DIR / "train_ground_truth.tsv"
)

dev_gt_dict = {
    k: v for k, v in dev_gt_dict.items()
    if k in slice_ids
}

if len(s1_dev) != 50_000:
    raise RuntimeError(f"Expected 50,000 S1 rows, found {len(s1_dev):,}")

if len(dev_gt_dict) != 50_000:
    raise RuntimeError(
        f"Expected 50,000 GT rows, found {len(dev_gt_dict):,}"
    )

total_links = sum(len(v) for v in dev_gt_dict.values())
singletons = sum(len(v) == 0 for v in dev_gt_dict.values())
matched = sum(len(v) > 0 for v in dev_gt_dict.values())

print("FAST DEV SLICE READY")
print(f"S1 rows: {len(s1_dev):,}")
print(f"True links: {total_links:,}")
print(f"Matched S1 entities: {matched:,}")
print(f"Singleton S1 entities: {singletons:,}")

# This script is intended as a reusable preparation/checkpoint utility.
# Do not persist huge in-memory DataFrames; save only small metadata needed
# by downstream scripts.
