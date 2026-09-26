import time
from pathlib import Path

import pandas as pd
import numpy as np


# ==============================================================================
# PATHS
# ==============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
S1_PATH = TRAIN_DIR / "train_source1.tsv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("REBUILDING PHASE 9 SPLIT METADATA")
print("=" * 80)
t0_start = time.time()
t0 = time.time()


# ==============================================================================
# 1. LOAD GROUND TRUTH
# ==============================================================================

print("\n[1/4] Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
)

print(f"GT rows: {len(gt):,}")


# ==============================================================================
# 2. LOAD ONLY S1 COUNTRY
# ==============================================================================

print("\n[2/4] Loading S1 country metadata...")

s1_meta = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=[
        "entity_id",
        "country",
    ],
)

s1_meta = s1_meta.rename(
    columns={
        "entity_id": "source1_entity_id"
    }
)

print(f"S1 rows: {len(s1_meta):,}")


# ==============================================================================
# 3. BUILD STRATIFICATION VARIABLES
# ==============================================================================

print("\n[3/4] Building stratification groups...")

df = gt.merge(
    s1_meta,
    on="source1_entity_id",
    how="inner",
    validate="one_to_one",
)


def get_cardinality_bucket(value):
    value = value.strip()

    if not value:
        return "0"

    count = len(value.split(","))

    if count == 1:
        return "1"

    if count in (2, 3):
        return "2-3"

    return "4+"


df["card_bucket"] = (
    df["matched_entity_ids"]
    .map(get_cardinality_bucket)
)

df["strat_key"] = (
    df["country"] +
    "_" +
    df["card_bucket"]
)


print("\nStratification groups:")
print(
    df["strat_key"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ==============================================================================
# 4. 80/20 HOLDOUT
# ==============================================================================

holdout = (
    df.groupby(
        "strat_key",
        group_keys=False
    )
    .sample(
        frac=0.20,
        random_state=42
    )
)

df["split"] = "dev"

df.loc[
    holdout.index,
    "split"
] = "holdout"


dev_df = df[
    df["split"] == "dev"
].copy()


# ==============================================================================
# 5. EXACT 50,000 FAST-DEV SLICE
# ==============================================================================

FAST_SLICE_SIZE = 50_000

group_sizes = (
    dev_df["strat_key"]
    .value_counts()
    .sort_index()
)

# Initial proportional allocation.
raw_counts = (
    group_sizes /
    len(dev_df) *
    FAST_SLICE_SIZE
)

allocation = np.floor(raw_counts).astype(int)

# Distribute remaining rows according to largest fractional remainders.
remaining = (
    FAST_SLICE_SIZE -
    allocation.sum()
)

remainders = (
    raw_counts -
    allocation
)

for key in remainders.sort_values(
    ascending=False
).index[:remaining]:

    allocation.loc[key] += 1


# Sample exactly the allocated number from each stratum.
fast_parts = []

for key, n in allocation.items():

    group = dev_df[
        dev_df["strat_key"] == key
    ]

    sampled = group.sample(
        n=int(n),
        random_state=42
    )

    fast_parts.append(sampled)


fast_slice = pd.concat(
    fast_parts,
    axis=0
)


assert len(fast_slice) == FAST_SLICE_SIZE


# ==============================================================================
# 6. SAVE METADATA
# ==============================================================================

df["fast_slice"] = False

df.loc[
    fast_slice.index,
    "fast_slice"
] = True


output_columns = [
    "source1_entity_id",
    "country",
    "card_bucket",
    "split",
    "fast_slice",
]


output_path = OUTPUT_DIR / "s1_splits.tsv"

df[
    output_columns
].to_csv(
    output_path,
    sep="\t",
    index=False,
)


# ==============================================================================
# 7. VALIDATION
# ==============================================================================

print("\n" + "=" * 80)
print("SPLIT SUMMARY")
print("=" * 80)

print(
    f"Total S1 entities:       {len(df):,}"
)

print(
    f"Development Set:         "
    f"{(df['split'] == 'dev').sum():,}"
)

print(
    f"Holdout Set:             "
    f"{(df['split'] == 'holdout').sum():,}"
)

print(
    f"Fast Dev Slice:          "
    f"{df['fast_slice'].sum():,}"
)

print(
    f"Saved: {output_path}"
)

print(
    f"Execution Time: "
    f"{time.time() - t0:.2f}s"
)

print("=" * 80)
