# ==============================================================================
# PHASE 10 (STEP 2): S2 CHEAP BLOCKING BASELINE EVALUATION
#
# Standalone version
#
# Evaluates:
#   Key A: (country, exact_normalized_name)
#   Key B: (country, first_token)
#   Key A ∪ Key B
#
# Reconstructs:
#   - 50,000 fast-dev S1 slice
#   - Ground-truth dictionary
#
# Does NOT depend on previous notebook variables.
# ==============================================================================

import gc
import re
import time
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ==============================================================================
# 0. PROJECT PATHS
# ==============================================================================

# This script is expected to live in:
# MLchallenge/notebooks/phase10_step2.py

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SPLIT_PATH = OUTPUT_DIR / "s1_splits.tsv"


print("=" * 80)
print("PHASE 10 (STEP 2): S2 CHEAP BLOCKING RETRIEVAL BENCHMARK")
print("=" * 80)

print(f"\nProject directory: {PROJECT_DIR}")
print(f"S1: {S1_PATH}")
print(f"S2: {S2_PATH}")
print(f"GT: {GT_PATH}")
print(f"Splits: {SPLIT_PATH}")


# ==============================================================================
# 1. VERIFY FILES
# ==============================================================================

required_files = [
    S1_PATH,
    S2_PATH,
    GT_PATH,
    SPLIT_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

print("\nAll required files found.")


# ==============================================================================
# 2. NORMALIZATION
# ==============================================================================

def normalize_text(text):
    """
    Unicode-safe normalization established during Phase 7.

    Important:
    - NFKC
    - casefold
    - punctuation/symbol -> spaces
    - preserve Unicode letters/numbers
    - collapse whitespace
    """

    if text is None or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text).casefold()

    cleaned = []

    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith(("P", "S")):
            cleaned.append(" ")
        elif ch.isspace():
            cleaned.append(" ")
        else:
            cleaned.append(ch)

    text = "".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_first_token(text):
    tokens = text.split()
    return tokens[0] if tokens else ""


# ==============================================================================
# 3. LOAD FAST-DEV SLICE METADATA
# ==============================================================================

print("\n[1/5] Loading fast-dev split metadata...")

splits_df = pd.read_csv(
    SPLIT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

if "source1_entity_id" not in splits_df.columns:
    raise ValueError("s1_splits.tsv does not contain source1_entity_id")

if "fast_slice" not in splits_df.columns:
    raise ValueError("s1_splits.tsv does not contain fast_slice")

# Handle both True/False strings and boolean values.
fast_mask = splits_df["fast_slice"].astype(str).str.lower().eq("true")

slice_s1_ids = set(
    splits_df.loc[fast_mask, "source1_entity_id"]
)

print(f"Fast-dev S1 entities: {len(slice_s1_ids):,}")

if len(slice_s1_ids) != 50_000:
    raise ValueError(
        f"Expected 50,000 fast-dev S1 entities, "
        f"found {len(slice_s1_ids):,}"
    )


# ==============================================================================
# 4. LOAD ONLY THE 50K S1 RECORDS
# ==============================================================================

print("\n[2/5] Loading 50,000 fast-dev S1 records...")

CHUNK_SIZE = 250_000

s1_rows = []

s1_reader = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ],
)

for chunk in s1_reader:

    matched = chunk[
        chunk["entity_id"].isin(slice_s1_ids)
    ]

    if not matched.empty:
        s1_rows.append(matched.copy())

    del chunk
    gc.collect()


if not s1_rows:
    raise RuntimeError("Could not find any fast-dev S1 records.")

s1_dev = pd.concat(
    s1_rows,
    ignore_index=True
)

del s1_rows
gc.collect()

if len(s1_dev) != 50_000:
    raise ValueError(
        f"Expected 50,000 S1 records, found {len(s1_dev):,}"
    )

print(f"Loaded S1 records: {len(s1_dev):,}")


# Normalize names.
print("Normalizing S1 names...")

s1_dev["name_norm"] = (
    s1_dev["business_name"]
    .map(normalize_text)
)

s1_dev["first_token"] = (
    s1_dev["name_norm"]
    .map(get_first_token)
)


# ==============================================================================
# 5. LOAD GROUND TRUTH
# ==============================================================================

print("\n[3/5] Loading ground truth...")

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

gt_slice = gt[
    gt["source1_entity_id"].isin(slice_s1_ids)
].copy()

del gt
gc.collect()


if len(gt_slice) != 50_000:
    raise ValueError(
        f"Expected 50,000 GT rows, found {len(gt_slice):,}"
    )


def parse_targets(value):
    value = value.strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


gt_slice["target_set"] = (
    gt_slice["matched_entity_ids"]
    .map(parse_targets)
)

dev_gt_dict = dict(
    zip(
        gt_slice["source1_entity_id"],
        gt_slice["target_set"]
    )
)

del gt_slice
gc.collect()


# ==============================================================================
# 6. PREPARE S2 TARGETS
# ==============================================================================

dev_gt_s2 = {}

total_s2_true_links = 0
s1_with_s2_matches = []

for s1_id in s1_dev["entity_id"].values:

    targets = dev_gt_dict.get(
        s1_id,
        set()
    )

    s2_targets = {
        target
        for target in targets
        if target.startswith("S2-")
    }

    dev_gt_s2[s1_id] = s2_targets

    total_s2_true_links += len(s2_targets)

    if s2_targets:
        s1_with_s2_matches.append(s1_id)


print(f"S1 entities: {len(s1_dev):,}")
print(
    f"S1 entities with >=1 S2 match: "
    f"{len(s1_with_s2_matches):,}"
)
print(
    f"Total true S2 links: "
    f"{total_s2_true_links:,}"
)


# ==============================================================================
# 7. BUILD QUERY-SIDE BLOCKING INDEXES
# ==============================================================================

print("\n[4/5] Building S1 blocking indexes...")

# Key A:
# (country, exact normalized name) -> S1 IDs

index_s1_exact = defaultdict(list)

# Key B:
# (country, first token) -> S1 IDs

index_s1_token1 = defaultdict(list)


for row in s1_dev[
    [
        "entity_id",
        "country",
        "name_norm",
        "first_token",
    ]
].itertuples(index=False):

    s1_id = row.entity_id
    country = row.country
    name = row.name_norm
    token1 = row.first_token

    if name:
        index_s1_exact[
            (country, name)
        ].append(s1_id)

    if token1:
        index_s1_token1[
            (country, token1)
        ].append(s1_id)


print(
    f"Unique exact-name keys: "
    f"{len(index_s1_exact):,}"
)

print(
    f"Unique first-token keys: "
    f"{len(index_s1_token1):,}"
)


# ==============================================================================
# 8. STREAM S2
# ==============================================================================

print("\n[5/5] Streaming S2...")

CHUNK_SIZE = 250_000
TOTAL_S2_RECORDS = 5_034_616

cand_exact = defaultdict(set)
cand_token1 = defaultdict(set)

rows_processed = 0
chunk_num = 0

t0_stream = time.time()

s2_reader = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE,
    usecols=[
        "entity_id",
        "business_name",
        "country",
    ],
)


for chunk in s2_reader:

    chunk_num += 1
    rows_processed += len(chunk)

    # Normalize names.
    chunk["name_norm"] = (
        chunk["business_name"]
        .map(normalize_text)
    )

    chunk["first_token"] = (
        chunk["name_norm"]
        .map(get_first_token)
    )

    # ------------------------------------------------------------------
    # KEY A: COUNTRY + EXACT NORMALIZED NAME
    # ------------------------------------------------------------------

    # Create lookup key only for non-empty names.
    exact_mask = chunk["name_norm"].ne("")

    exact_chunk = chunk.loc[
        exact_mask,
        [
            "entity_id",
            "country",
            "name_norm",
        ],
    ]

    for country, name, s2_id in exact_chunk[
        ["country", "name_norm", "entity_id"]
    ].itertuples(index=False):

        matching_s1 = index_s1_exact.get(
            (country, name)
        )

        if matching_s1:

            for s1_id in matching_s1:
                cand_exact[s1_id].add(s2_id)


    # ------------------------------------------------------------------
    # KEY B: COUNTRY + FIRST TOKEN
    # ------------------------------------------------------------------

    token_mask = chunk["first_token"].ne("")

    token_chunk = chunk.loc[
        token_mask,
        [
            "entity_id",
            "country",
            "first_token",
        ],
    ]

    for country, token1, s2_id in token_chunk[
        ["country", "first_token", "entity_id"]
    ].itertuples(index=False):

        matching_s1 = index_s1_token1.get(
            (country, token1)
        )

        if matching_s1:

            for s1_id in matching_s1:
                cand_token1[s1_id].add(s2_id)


    elapsed = time.time() - t0_stream
    percent = (
        rows_processed /
        TOTAL_S2_RECORDS
    ) * 100

    print(
        f"Chunk {chunk_num:02d} | "
        f"{rows_processed:,}/{TOTAL_S2_RECORDS:,} "
        f"({percent:.1f}%) | "
        f"Elapsed: {elapsed:.1f}s"
    )

    del chunk
    del exact_chunk
    del token_chunk

    gc.collect()


# Free query indexes.
del index_s1_exact
del index_s1_token1

gc.collect()


# ==============================================================================
# 9. EVALUATION
# ==============================================================================

print("\nEvaluating retrieval metrics...")


def evaluate_retriever(
    cand_dict,
    channel_name,
):

    retrieved_true_links = 0

    all_match_count = 0
    at_least_one_count = 0

    candidate_sizes = []

    for s1_id in s1_dev["entity_id"].values:

        candidates = cand_dict.get(
            s1_id,
            set()
        )

        candidate_sizes.append(
            len(candidates)
        )

        true_targets = dev_gt_s2[s1_id]

        if not true_targets:
            continue

        hits = (
            candidates &
            true_targets
        )

        retrieved_true_links += len(hits)

        if len(hits) == len(true_targets):
            all_match_count += 1

        if len(hits) >= 1:
            at_least_one_count += 1


    candidate_sizes = np.asarray(
        candidate_sizes,
        dtype=np.int32
    )

    num_matched_s1 = len(
        s1_with_s2_matches
    )

    link_recall = (
        retrieved_true_links /
        total_s2_true_links
    ) * 100.0

    all_match_entity_recall = (
        all_match_count /
        num_matched_s1
    ) * 100.0

    at_least_one_recall = (
        at_least_one_count /
        num_matched_s1
    ) * 100.0

    mean_candidates = np.mean(
        candidate_sizes
    )

    reduction_ratio = (
        1.0 -
        (
            mean_candidates /
            TOTAL_S2_RECORDS
        )
    ) * 100.0

    return {
        "Channel": channel_name,
        "Link Recall (%)":
            f"{link_recall:.2f}%",
        "All-Match Entity Recall (%)":
            f"{all_match_entity_recall:.2f}%",
        "At-Least-1 Recall (%)":
            f"{at_least_one_recall:.2f}%",
        "Mean Cands / S1":
            f"{mean_candidates:.2f}",
        "Median Cands":
            f"{np.median(candidate_sizes):.0f}",
        "P90 Cands":
            f"{np.percentile(candidate_sizes, 90):.0f}",
        "P99 Cands":
            f"{np.percentile(candidate_sizes, 99):.0f}",
        "Max Cands":
            f"{np.max(candidate_sizes):,}",
        "Reduction Ratio (%)":
            f"{reduction_ratio:.4f}%",
    }


# ==============================================================================
# 10. EVALUATE BOTH CHANNELS + UNION
# ==============================================================================

metrics_exact = evaluate_retriever(
    cand_exact,
    "Key A: (Country, Exact Norm Name)"
)

metrics_token1 = evaluate_retriever(
    cand_token1,
    "Key B: (Country, First Token)"
)


# We don't create a third giant candidate dictionary.
# Instead evaluate the union directly.

cand_union = defaultdict(set)

for s1_id in s1_dev["entity_id"].values:

    exact_candidates = cand_exact.get(
        s1_id,
        set()
    )

    token_candidates = cand_token1.get(
        s1_id,
        set()
    )

    if exact_candidates:
        cand_union[s1_id].update(
            exact_candidates
        )

    if token_candidates:
        cand_union[s1_id].update(
            token_candidates
        )


metrics_union = evaluate_retriever(
    cand_union,
    "Key A ∪ Key B (Cheap Blocking)"
)


# ==============================================================================
# 11. RESULTS
# ==============================================================================

df_results = pd.DataFrame(
    [
        metrics_exact,
        metrics_token1,
        metrics_union,
    ]
)

print("\n")
print("=" * 80)
print("BENCHMARK REPORT")
print("=" * 80)

print(
    df_results.to_string(
        index=False
    )
)

print(
    f"\nTotal execution time: "
    f"{(time.time() - t0_start) / 60:.2f} minutes"
)

print("=" * 80)


# ==============================================================================
# 12. CLEANUP
# ==============================================================================

del cand_exact
del cand_token1
del cand_union
del dev_gt_s2
del dev_gt_dict
del s1_dev

gc.collect()

print("\nPhase 10 Step 2 completed.")
