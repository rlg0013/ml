# ==============================================================================
# PHASE 10 — STEP 3
# CHARACTER N-GRAM TF-IDF RETRIEVER — STANDALONE LOCAL VERSION
#
# Reconstructed from the original Phase 10 Step 3 implementation.
#
# Validation slice:
#   50,000 fast-dev S1 entities
#
# Retrieval:
#   char_wb TF-IDF
#   ngram_range=(3,4)
#   max_features=50,000
#   top-K = 50
#   minimum cosine similarity = 0.40
#
# Evaluates:
#   Recall@10
#   Recall@25
#   Recall@50
#
# Saves:
#   output/phase10_step3_char_tfidf_results.tsv
#   output/phase10_step3_char_tfidf_top50.tsv
#   output/phase10_step3_char_tfidf_config.json
#   models/phase10_step3_char_tfidf_vectorizer.joblib
#
# IMPORTANT:
#   This script does not depend on variables left in a notebook kernel.
#   It reconstructs the required S1 slice and ground truth from disk.
# ==============================================================================

import gc
import heapq
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sparse_dot_topn import sp_matmul_topn
    USE_SPARSE_DOT_TOPN = True
except ImportError:
    USE_SPARSE_DOT_TOPN = False


# ==============================================================================
# 0. PROJECT PATHS
# ==============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"
MODELS_DIR = PROJECT_DIR / "models"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SPLIT_PATH = OUTPUT_DIR / "s1_splits.tsv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_DIR))

from src.normalization import normalize_text
from src.metrics import load_ground_truth


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

FAST_SLICE_SIZE = 50_000

CHUNK_SIZE = 250_000
TOTAL_S2_RECORDS = 5_034_616

TOP_K = 50
SIM_LOWER_BOUND = 0.40

RANDOM_SEED = 42

TFIDF_CONFIG = {
    "analyzer": "char_wb",
    "ngram_range": (3, 4),
    "min_df": 2,
    "max_features": 50_000,
    "sublinear_tf": True,
    "dtype": np.float32,
}

N_THREADS = max(1, (os.cpu_count() or 2) - 1)

t0_start = time.time()


# ==============================================================================
# 2. HELPERS
# ==============================================================================

def load_fast_dev_s1():
    """Reconstruct the exact 50k fast-dev S1 slice from split metadata."""

    splits = pd.read_csv(
        SPLIT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id", "fast_slice"],
    )

    mask = (
        splits["fast_slice"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    slice_ids = set(
        splits.loc[mask, "source1_entity_id"]
    )

    del splits
    gc.collect()

    if len(slice_ids) != FAST_SLICE_SIZE:
        raise RuntimeError(
            f"Expected {FAST_SLICE_SIZE:,} fast-dev IDs, "
            f"found {len(slice_ids):,}."
        )

    rows = []

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        usecols=["entity_id", "business_name", "country"],
    )

    for chunk in reader:
        matched = chunk[
            chunk["entity_id"].isin(slice_ids)
        ]

        if not matched.empty:
            rows.append(matched.copy())

        del chunk
        gc.collect()

    s1 = pd.concat(rows, ignore_index=True)

    if len(s1) != FAST_SLICE_SIZE:
        raise RuntimeError(
            f"Expected {FAST_SLICE_SIZE:,} S1 records, "
            f"found {len(s1):,}."
        )

    s1["norm_name"] = (
        s1["business_name"]
        .map(normalize_text)
    )

    return s1, slice_ids


def prepare_ground_truth(s1_ids):
    """Load and restrict ground truth to the fast-dev S1 IDs."""

    gt = load_ground_truth(GT_PATH)

    dev_gt = {
        s1_id: gt.get(s1_id, set())
        for s1_id in s1_ids
    }

    del gt
    gc.collect()

    dev_gt_s2 = {
        s1_id: {
            target
            for target in targets
            if target.startswith("S2-")
        }
        for s1_id, targets in dev_gt.items()
    }

    return dev_gt_s2


def update_topk_heap(heap, similarity, s2_id):
    """Keep the best TOP_K candidates in a bounded min-heap."""

    item = (float(similarity), s2_id)

    if len(heap) < TOP_K:
        heapq.heappush(heap, item)

    elif similarity > heap[0][0]:
        heapq.heapreplace(heap, item)


def search_sparse_topn(
    query_matrix,
    candidate_matrix,
    query_ids,
    candidate_ids,
    heaps,
):
    """
    Sparse top-N retrieval.

    query_matrix:
        S1 TF-IDF rows for one country.

    candidate_matrix:
        S2 TF-IDF rows for one country/chunk.

    Only sparse similarities above SIM_LOWER_BOUND are retained.
    """

    if (
        query_matrix.shape[0] == 0
        or candidate_matrix.shape[0] == 0
    ):
        return

    if USE_SPARSE_DOT_TOPN:

        similarities = sp_matmul_topn(
            query_matrix,
            candidate_matrix.T,
            top_n=TOP_K,
            threshold=SIM_LOWER_BOUND,
            sort=True,
            n_threads=N_THREADS,
        )

        similarities = similarities.tocoo()

        for row_idx, col_idx, sim in zip(
            similarities.row,
            similarities.col,
            similarities.data,
        ):
            s1_id = query_ids[row_idx]
            s2_id = candidate_ids[col_idx]

            update_topk_heap(
                heaps[s1_id],
                float(sim),
                s2_id,
            )

        del similarities
        return

    # ------------------------------------------------------------------
    # Scipy fallback.
    #
    # Process query rows in small batches so no dense matrix is created.
    # ------------------------------------------------------------------

    candidate_transpose = candidate_matrix.T.tocsc()

    batch_size = 200

    for start in range(
        0,
        query_matrix.shape[0],
        batch_size,
    ):

        end = min(
            start + batch_size,
            query_matrix.shape[0],
        )

        block = (
            query_matrix[start:end]
            .dot(candidate_transpose)
            .tocoo()
        )

        for q_rel, s2_idx, sim in zip(
            block.row,
            block.col,
            block.data,
        ):

            sim = float(sim)

            if sim < SIM_LOWER_BOUND:
                continue

            s1_id = query_ids[
                start + q_rel
            ]

            s2_id = candidate_ids[
                s2_idx
            ]

            update_topk_heap(
                heaps[s1_id],
                sim,
                s2_id,
            )

        del block

    del candidate_transpose
    gc.collect()


def evaluate_cutoff(
    candidates,
    s1_ids,
    dev_gt_s2,
    cutoff,
):
    """Evaluate link recall and entity recall at one candidate cutoff."""

    total_true_links = sum(
        len(targets)
        for targets in dev_gt_s2.values()
    )

    matched_s1 = [
        s1_id
        for s1_id, targets in dev_gt_s2.items()
        if targets
    ]

    retrieved_true_links = 0
    all_match_count = 0
    at_least_one_count = 0

    candidate_sizes = []

    for s1_id in s1_ids:

        candidate_list = candidates.get(
            s1_id,
            [],
        )

        selected = set(
            candidate_list[:cutoff]
        )

        candidate_sizes.append(
            len(selected)
        )

        true_targets = dev_gt_s2[s1_id]

        if not true_targets:
            continue

        hits = (
            selected &
            true_targets
        )

        retrieved_true_links += len(hits)

        if len(hits) == len(true_targets):
            all_match_count += 1

        if len(hits) >= 1:
            at_least_one_count += 1

    candidate_sizes = np.asarray(
        candidate_sizes,
        dtype=np.int32,
    )

    link_recall = (
        retrieved_true_links /
        total_true_links
    ) * 100.0

    all_match_recall = (
        all_match_count /
        len(matched_s1)
    ) * 100.0

    at_least_one_recall = (
        at_least_one_count /
        len(matched_s1)
    ) * 100.0

    return {
        "Channel": f"Channel 2: Char TF-IDF (Top-{cutoff})",
        "Link Recall (%)": round(
            link_recall,
            4,
        ),
        "All-Match Entity Recall (%)": round(
            all_match_recall,
            4,
        ),
        "At-Least-1 Recall (%)": round(
            at_least_one_recall,
            4,
        ),
        "Mean Cands / S1": round(
            float(np.mean(candidate_sizes)),
            2,
        ),
        "Median Cands": int(
            np.median(candidate_sizes)
        ),
        "P90 Cands": int(
            np.percentile(
                candidate_sizes,
                90,
            )
        ),
        "P99 Cands": int(
            np.percentile(
                candidate_sizes,
                99,
            )
        ),
        "Max Cands": int(
            np.max(candidate_sizes)
        ),
    }


# ==============================================================================
# 3. PRECONDITION CHECK
# ==============================================================================

print("=" * 80)
print("PHASE 10 (STEP 3): CHARACTER N-GRAM TF-IDF RETRIEVER")
print("=" * 80)

print(f"\nProject: {PROJECT_DIR}")
print(f"S1: {S1_PATH}")
print(f"S2: {S2_PATH}")
print(f"GT: {GT_PATH}")
print(f"Split: {SPLIT_PATH}")

print(
    f"\nsparse_dot_topn available: "
    f"{USE_SPARSE_DOT_TOPN}"
)

required_files = [
    S1_PATH,
    S2_PATH,
    GT_PATH,
    SPLIT_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ==============================================================================
# 4. LOAD FAST-DEV S1 + GROUND TRUTH
# ==============================================================================

print(
    "\n[1/4] Reconstructing fast-dev S1 "
    "and ground truth..."
)

s1_dev, slice_s1_ids = load_fast_dev_s1()

dev_gt_s2 = prepare_ground_truth(
    slice_s1_ids
)

s1_ids = s1_dev[
    "entity_id"
].values

s1_countries = s1_dev[
    "country"
].values

total_s2_true_links = sum(
    len(targets)
    for targets in dev_gt_s2.values()
)

s1_with_s2_matches = [
    s1_id
    for s1_id, targets in dev_gt_s2.items()
    if targets
]

print(
    f"S1 queries: "
    f"{len(s1_dev):,}"
)

print(
    f"S1 with S2 matches: "
    f"{len(s1_with_s2_matches):,}"
)

print(
    f"True S2 links: "
    f"{total_s2_true_links:,}"
)


# ==============================================================================
# 5. FIT TF-IDF ON S1 DEV
# ==============================================================================

print(
    "\n[2/4] Fitting character TF-IDF..."
)

t0_vec = time.time()

vectorizer = TfidfVectorizer(
    **TFIDF_CONFIG
)

s1_names = (
    s1_dev["norm_name"]
    .fillna("")
    .tolist()
)

s1_tfidf_all = (
    vectorizer
    .fit_transform(s1_names)
)

print(
    f"Vocabulary size: "
    f"{len(vectorizer.vocabulary_):,}"
)

print(
    f"S1 matrix: "
    f"{s1_tfidf_all.shape}"
)

print(
    f"S1 nonzeros: "
    f"{s1_tfidf_all.nnz:,}"
)

print(
    f"Vectorizer fit time: "
    f"{time.time() - t0_vec:.2f}s"
)


# ==============================================================================
# 6. GENERIC COUNTRY PARTITIONING
# ==============================================================================

# The original benchmark partitioned by US / India.
# Here we generalize it to arbitrary country strings so this code does
# not assume the test set will contain only the training countries.

country_query_indices = defaultdict(list)

for idx, country in enumerate(s1_countries):
    country_query_indices[country].append(idx)

country_query_matrices = {}

for country, indices in country_query_indices.items():

    country_query_matrices[country] = {
        "matrix": s1_tfidf_all[indices],
        "ids": s1_ids[indices],
    }

print(
    f"Countries represented in fast-dev S1: "
    f"{list(country_query_indices.keys())}"
)


# ==============================================================================
# 7. INITIALIZE TOP-K HEAPS
# ==============================================================================

topk_heaps = {
    s1_id: []
    for s1_id in s1_ids
}


# ==============================================================================
# 8. STREAM S2
# ==============================================================================

print(
    f"\n[3/4] Streaming S2 in "
    f"{CHUNK_SIZE:,}-row chunks..."
)

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

chunk_num = 0
rows_processed = 0
t0_stream = time.time()

for chunk in s2_reader:

    chunk_num += 1

    chunk_len = len(chunk)
    chunk_start_row = rows_processed

    rows_processed += chunk_len

    # --------------------------------------------------------------
    # Normalize names using the canonical Phase 7 function.
    # --------------------------------------------------------------

    norm_s2_names = [
        normalize_text(name)
        for name in chunk[
            "business_name"
        ].values
    ]

    chunk_ids = (
        chunk["entity_id"]
        .values
    )

    chunk_countries = (
        chunk["country"]
        .values
    )

    # --------------------------------------------------------------
    # Transform this S2 chunk into TF-IDF space.
    # --------------------------------------------------------------

    s2_tfidf_chunk = (
        vectorizer
        .transform(norm_s2_names)
    )

    # --------------------------------------------------------------
    # Search each country represented in this chunk.
    # --------------------------------------------------------------

    unique_countries = np.unique(
        chunk_countries
    )

    for country in unique_countries:

        query_info = (
            country_query_matrices
            .get(country)
        )

        if query_info is None:
            # No S1 query from this country.
            continue

        country_mask = (
            chunk_countries == country
        )

        if not np.any(country_mask):
            continue

        s2_country_matrix = (
            s2_tfidf_chunk[
                country_mask
            ]
        )

        s2_country_ids = (
            chunk_ids[
                country_mask
            ]
        )

        search_sparse_topn(
            query_info["matrix"],
            s2_country_matrix,
            query_info["ids"],
            s2_country_ids,
            topk_heaps,
        )

        del s2_country_matrix
        del s2_country_ids

    elapsed = (
        time.time() -
        t0_stream
    )

    percent = (
        rows_processed /
        TOTAL_S2_RECORDS
    ) * 100.0

    print(
        f"Chunk {chunk_num:02d} | "
        f"{rows_processed:,}/"
        f"{TOTAL_S2_RECORDS:,} "
        f"({percent:.1f}%) | "
        f"Elapsed: {elapsed:.1f}s"
    )

    del chunk
    del norm_s2_names
    del chunk_ids
    del chunk_countries
    del s2_tfidf_chunk

    gc.collect()


# ==============================================================================
# 9. CONVERT HEAPS TO SORTED CANDIDATE LISTS
# ==============================================================================

print(
    "\nConverting bounded heaps "
    "to sorted candidate lists..."
)

tfidf_candidates = {}

for s1_id, heap in topk_heaps.items():

    ordered = sorted(
        heap,
        key=lambda x: x[0],
        reverse=True,
    )

    tfidf_candidates[s1_id] = [
        s2_id
        for _, s2_id in ordered
    ]


del topk_heaps
gc.collect()


# ==============================================================================
# 10. EVALUATE
# ==============================================================================

print(
    "\n[4/4] Evaluating Recall@10, "
    "Recall@25 and Recall@50..."
)

results = []

for cutoff in [10, 25, 50]:

    result = evaluate_cutoff(
        tfidf_candidates,
        s1_ids,
        dev_gt_s2,
        cutoff,
    )

    results.append(result)


results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("CHARACTER TF-IDF RETRIEVAL RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ==============================================================================
# 11. SAVE RESULTS
# ==============================================================================

results_path = (
    OUTPUT_DIR /
    "phase10_step3_char_tfidf_results.tsv"
)

results_df.to_csv(
    results_path,
    sep="\t",
    index=False,
)


# ==============================================================================
# 12. SAVE TOP-50 CANDIDATES
# ==============================================================================

candidate_rows = []

for s1_id in s1_ids:

    candidate_ids = (
        tfidf_candidates[
            s1_id
        ][:TOP_K]
    )

    candidate_rows.append(
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids":
                ",".join(candidate_ids),
        }
    )


candidate_df = pd.DataFrame(
    candidate_rows
)

candidate_path = (
    OUTPUT_DIR /
    "phase10_step3_char_tfidf_top50.tsv"
)

candidate_df.to_csv(
    candidate_path,
    sep="\t",
    index=False,
)


# ==============================================================================
# 13. SAVE VECTORIZER
# ==============================================================================

import joblib

vectorizer_path = (
    MODELS_DIR /
    "phase10_step3_char_tfidf_vectorizer.joblib"
)

joblib.dump(
    vectorizer,
    vectorizer_path,
)


# ==============================================================================
# 14. SAVE CONFIG
# ==============================================================================

config = {
    "phase": "10_step3",
    "method": "character_n_gram_tfidf",
    "analyzer": "char_wb",
    "ngram_range": [3, 4],
    "min_df": 2,
    "max_features": 50_000,
    "sublinear_tf": True,
    "dtype": "float32",
    "top_k": TOP_K,
    "similarity_lower_bound": SIM_LOWER_BOUND,
    "chunk_size": CHUNK_SIZE,
    "fast_dev_size": FAST_SLICE_SIZE,
    "random_seed": RANDOM_SEED,
    "fit_corpus": "50k fast-dev S1 names",
    "country_partitioned": True,
    "country_partition_is_generic": True,
    "s2_total_rows": TOTAL_S2_RECORDS,
}

config_path = (
    OUTPUT_DIR /
    "phase10_step3_char_tfidf_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
    )


# ==============================================================================
# 15. CLEAN UP
# ==============================================================================

del tfidf_candidates
del results_df
del candidate_df
del vectorizer
del s1_tfidf_all
del country_query_matrices
del country_query_indices
del s1_dev
del dev_gt_s2

gc.collect()


# ==============================================================================
# 16. FINISH
# ==============================================================================

elapsed_total = (
    time.time() -
    t0_start
)

print("\n")
print("=" * 80)
print("PHASE 10 STEP 3 COMPLETED")
print("=" * 80)

print(
    f"Results:    {results_path}"
)

print(
    f"Candidates: {candidate_path}"
)

print(
    f"Vectorizer: {vectorizer_path}"
)

print(
    f"Config:     {config_path}"
)

print(
    f"Total time: "
    f"{elapsed_total / 60:.2f} minutes"
)

print("=" * 80)
