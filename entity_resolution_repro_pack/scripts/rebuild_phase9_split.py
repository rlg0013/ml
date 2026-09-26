from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.splitting import build_grouped_split

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

df = build_grouped_split(
    gt_path=TRAIN_DIR / "train_ground_truth.tsv",
    s1_path=TRAIN_DIR / "train_source1.tsv",
    output_path=OUTPUT_DIR / "s1_splits.tsv",
    holdout_frac=0.20,
    fast_slice_size=50_000,
    random_state=42,
)

print("Split metadata rebuilt.")
print(f"Total S1: {len(df):,}")
print(f"Development: {(df['split'] == 'dev').sum():,}")
print(f"Holdout: {(df['split'] == 'holdout').sum():,}")
print(f"Fast-dev: {df['fast_slice'].sum():,}")
print(f"Saved: {OUTPUT_DIR / 's1_splits.tsv'}")
