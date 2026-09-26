# ==============================================================================
# PHASE 10 — STEP 4
# WORD / TOKEN TF-IDF RETRIEVER — STANDALONE LOCAL VERSION
#
# Reconstructed/corrected from the original Phase 10 Step 4 implementation.
#
# Validation slice:
#   50,000 canonical fast-dev S1 entities
#
# Retrieval:
#   Word TF-IDF
#   analyzer="word"
#   ngram_range=(1,2)
#   min_df=2
#   max_features=50,000
#   Top-K = 50
#   minimum cosine similarity = 0.35
#
# Evaluates:
#   Recall@10
#   Recall@25
#   Recall@50
#   Char Top-50 ∪ Word Top-50
#
# Important:
#   - Standalone: no notebook globals required
#   - Uses the current canonical s1_splits.tsv
#   - Uses src/normalization.py
#   - Uses src/metrics.py
#   - Uses generic country partitioning
#   - Saves all important state/results to disk
# ==============================================================================

import gc
import heapq
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

STEP3_TOP50_PATH = (
    OUTPUT_DIR /
    "phase10_step3_char_tfidf_top50.tsv"
)

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
SIM_LOWER_BOUND = 0.35

WORD_TFIDF_CONFIG = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 50_000,
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": "l2",
    "lowercase": False,
}

N_THREADS = max(1, (os.cpu_count() or 2) - 1)

t0_start = time.time()


# ==============================================================================
# 2. FILE CHECKS
# ==============================================================================

required_files = [
    S1_PATH,
    S2_PATH,
    GT_PATH,
    SPLIT_PATH,
    STEP3_TOP50_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ==============================================================================
# 3. RECONSTRUCT THE CANONICAL FAST-DEV S1 SLICE
# ==============================================================================

def reconstruct_fast_dev():
    """
    Reconstruct the canonical fast-dev slice from output/s1_splits.tsv.

    Current split schema:
      source1_entity_id
      country
      card_bucket
      split
      fast_slice

    This avoids depending on notebook globals.
    """

    print("[1/5] Loading canonical fast-dev split...")

    split_df = pd.read_csv(
        SPLIT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_split_cols = {
        "source1_entity_id",
        "country",
        "split",
        "fast_slice",
    }

    missing = required_split_cols - set(split_df.columns)

    if missing:
        raise ValueError(
            f"s1_splits.tsv is missing columns: {sorted(missing)}"
        )

    fast_mask = (
        split_df["fast_slice"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    fast_ids = set(
        split_df.loc[
            fast_mask,
            "source1_entity_id",
        ]
    )

    del split_df
    gc.collect()

    if len(fast_ids) != FAST_SLICE_SIZE:
        raise RuntimeError(
            f"Expected {FAST_SLICE_SIZE:,} fast-dev S1 IDs, "
            f"found {len(fast_ids):,}."
        )

    # --------------------------------------------------------------
    # Stream S1 and retain only the fast-dev IDs.
    # --------------------------------------------------------------

    rows = []

    reader = pd.read_csv(
        S1_PATH,
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

    for chunk in reader:

        matched = chunk[
            chunk["entity_id"].isin(fast_ids)
        ]

        if not matched.empty:
            rows.append(
                matched.copy()
            )

        del chunk
        gc.collect()

    s1 = pd.concat(
        rows,
        ignore_index=True,
    )

    del rows
    gc.collect()

    if len(s1) != FAST_SLICE_SIZE:
        raise RuntimeError(
            f"Expected {FAST_SLICE_SIZE:,} S1 rows, "
            f"found {len(s1):,}."
        )

    # Canonical normalization.
    s1["norm_name"] = (
        s1["business_name"]
        .map(normalize_text)
    )

    return s1, fast_ids


# ==============================================================================
# 4. LOAD GROUND TRUTH FOR FAST DEV
# ==============================================================================

def build_dev_gt(
    s1_ids,
):
    """
    Return S2-only ground-truth targets for the fast-dev S1 entities.
    """

    print("[2/5] Loading ground truth...")

    gt = load_ground_truth(
        GT_PATH
    )

    dev_gt_s2 = {}

    for s1_id in s1_ids:

        targets = gt.get(
            s1_id,
            set(),
        )

        dev_gt_s2[s1_id] = {
            target
            for target in targets
            if target.startswith("S2-")
        }

    del gt
    gc.collect()

    total_true_links = sum(
        len(targets)
        for targets in dev_gt_s2.values()
    )

    s1_with_s2 = [
        s1_id
        for s1_id, targets in dev_gt_s2.items()
        if targets
    ]

    print(
        f"  * S1 queries: {len(s1_ids):,}"
    )

    print(
        f"  * S1 queries with S2 matches: "
        f"{len(s1_with_s2):,}"
    )

    print(
        f"  * Total true S2 links: "
        f"{total_true_links:,}"
    )

    return (
        dev_gt_s2,
        total_true_links,
        s1_with_s2,
    )


# ==============================================================================
# 5. FIT WORD TF-IDF
# ==============================================================================

def fit_word_vectorizer(s1):
    """
    Fit the word-level TF-IDF representation on the 50k S1 query names.

    This follows the original Step 4 experiment.
    """

    print(
        "\n[3/5] Fitting Word TF-IDF vectorizer..."
    )

    t0 = time.time()

    vectorizer = TfidfVectorizer(
        **WORD_TFIDF_CONFIG
    )

    s1_names = (
        s1["norm_name"]
        .fillna("")
        .tolist()
    )

    s1_matrix = (
        vectorizer
        .fit_transform(s1_names)
    )

    print(
        f"  * Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"  * S1 matrix shape: "
        f"{s1_matrix.shape}"
    )

    print(
        f"  * S1 nonzeros: "
        f"{s1_matrix.nnz:,}"
    )

    print(
        f"  * Fit time: "
        f"{time.time() - t0:.2f}s"
    )

    # --------------------------------------------------------------
    # Partition S1 query matrix by the country labels actually
    # present in the fast-dev slice.
    # --------------------------------------------------------------

    s1_entities = s1["entity_id"].values
    s1_countries = s1["country"].values

    country_partitions = {}

    for country in np.unique(
        s1_countries
    ):

        indices = np.flatnonzero(
            s1_countries == country
        )

        country_partitions[country] = {
            "matrix": s1_matrix[indices],
            "ids": s1_entities[indices],
        }

    print(
        "  * Query countries: "
        f"{list(country_partitions.keys())}"
    )

    return (
        vectorizer,
        s1_matrix,
        country_partitions,
        s1_entities,
    )


# ==============================================================================
# 6. RETRIEVAL HELPERS
# ==============================================================================

def update_heap(
    heap,
    similarity,
    candidate_id,
):
    """
    Keep only the best TOP_K candidates.
    """

    item = (
        float(similarity),
        candidate_id,
    )

    if len(heap) < TOP_K:
        heapq.heappush(
            heap,
            item,
        )

    elif similarity > heap[0][0]:
        heapq.heapreplace(
            heap,
            item,
        )


def search_country_chunk(
    query_matrix,
    query_ids,
    s2_matrix,
    s2_ids,
    heaps,
):
    """
    Search one country-specific S2 chunk against one country-specific S1
    query matrix.

    Uses sparse_dot_topn when available, otherwise a batched sparse
    SciPy fallback that avoids dense matrix materialization.
    """

    if (
        query_matrix.shape[0] == 0
        or s2_matrix.shape[0] == 0
    ):
        return

    if USE_SPARSE_DOT_TOPN:

        similarities = sp_matmul_topn(
            query_matrix,
            s2_matrix.T,
            top_n=TOP_K,
            threshold=SIM_LOWER_BOUND,
            sort=True,
            n_threads=N_THREADS,
        )

        similarities = similarities.tocoo()

        for row_idx, col_idx, similarity in zip(
            similarities.row,
            similarities.col,
            similarities.data,
        ):

            update_heap(
                heaps[query_ids[row_idx]],
                float(similarity),
                s2_ids[col_idx],
            )

        del similarities

        return

    # --------------------------------------------------------------
    # Safe sparse fallback.
    # --------------------------------------------------------------

    s2_t = s2_matrix.T.tocsc()

    batch_size = 250

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
            .dot(s2_t)
            .tocoo()
        )

        for row_idx, col_idx, similarity in zip(
            block.row,
            block.col,
            block.data,
        ):

            similarity = float(similarity)

            if similarity < SIM_LOWER_BOUND:
                continue

            update_heap(
                heaps[
                    query_ids[
                        start + row_idx
                    ]
                ],
                similarity,
                s2_ids[col_idx],
            )

        del block

    del s2_t


# ==============================================================================
# 7. STREAM S2 AND RETRIEVE TOP-50
# ==============================================================================

def run_retrieval(
    vectorizer,
    country_partitions,
    s1_entities,
):
    """
    Stream the entire S2 file and retain Top-50 candidates for each S1.
    """

    print(
        f"\n[4/5] Streaming S2 in "
        f"{CHUNK_SIZE:,}-row chunks..."
    )

    heaps = {
        s1_id: []
        for s1_id in s1_entities
    }

    reader = pd.read_csv(
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

    rows_done = 0
    chunk_num = 0
    t0_stream = time.time()

    for chunk in reader:

        chunk_num += 1

        chunk_len = len(chunk)

        norm_names = [
            normalize_text(name)
            for name in chunk[
                "business_name"
            ].values
        ]

        countries = (
            chunk["country"].values
        )

        ids = (
            chunk["entity_id"].values
        )

        # Transform this chunk into the same feature space
        # as the S1 matrix.
        s2_matrix = (
            vectorizer
            .transform(norm_names)
        )

        # ----------------------------------------------------------
        # Generic country partitioning.
        # If a country has no S1 queries, it is skipped.
        # ----------------------------------------------------------

        for country in np.unique(
            countries
        ):

            query_info = (
                country_partitions
                .get(country)
            )

            if query_info is None:
                continue

            mask = (
                countries == country
            )

            if not np.any(mask):
                continue

            s2_country_matrix = (
                s2_matrix[mask]
            )

            s2_country_ids = (
                ids[mask]
            )

            search_country_chunk(
                query_info["matrix"],
                query_info["ids"],
                s2_country_matrix,
                s2_country_ids,
                heaps,
            )

            del s2_country_matrix
            del s2_country_ids

        rows_done += chunk_len

        elapsed = (
            time.time() -
            t0_stream
        )

        pct = (
            rows_done /
            TOTAL_S2_RECORDS
        ) * 100.0

        print(
            f"  Chunk {chunk_num:02d} | "
            f"{rows_done:,}/"
            f"{TOTAL_S2_RECORDS:,} "
            f"({pct:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s"
        )

        del chunk
        del norm_names
        del countries
        del ids
        del s2_matrix

        gc.collect()

    # --------------------------------------------------------------
    # Convert heaps to descending candidate lists.
    # --------------------------------------------------------------

    sorted_candidates = {}

    for s1_id, heap in heaps.items():

        ordered = sorted(
            heap,
            key=lambda x: x[0],
            reverse=True,
        )

        sorted_candidates[s1_id] = [
            candidate_id
            for _, candidate_id in ordered
        ]

    del heaps
    gc.collect()

    return sorted_candidates


# ==============================================================================
# 8. EVALUATION
# ==============================================================================

def evaluate_retriever(
    candidates,
    s1_entities,
    dev_gt_s2,
    total_true_links,
    s1_with_s2,
    cutoff,
    label,
):
    """
    Evaluate candidate recall at one cutoff.
    """

    retrieved_true = 0
    all_match_count = 0
    at_least_one_count = 0

    candidate_sizes = []

    for s1_id in s1_entities:

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

        true_targets = dev_gt_s2.get(
            s1_id,
            set(),
        )

        if not true_targets:
            continue

        hits = (
            selected &
            true_targets
        )

        retrieved_true += len(hits)

        if len(hits) == len(true_targets):
            all_match_count += 1

        if len(hits) >= 1:
            at_least_one_count += 1

    sizes = np.asarray(
        candidate_sizes,
        dtype=np.int32,
    )

    link_recall = (
        retrieved_true /
        total_true_links
    ) * 100.0

    all_match_recall = (
        all_match_count /
        len(s1_with_s2)
    ) * 100.0

    at_least_one_recall = (
        at_least_one_count /
        len(s1_with_s2)
    ) * 100.0

    mean_candidates = float(
        np.mean(sizes)
    )

    reduction_ratio = (
        1.0 -
        (
            mean_candidates /
            TOTAL_S2_RECORDS
        )
    ) * 100.0

    return {
        "Channel": label,
        "Link Recall (%)": f"{link_recall:.4f}",
        "All-Match Entity Recall (%)": f"{all_match_recall:.4f}",
        "At-Least-1 Recall (%)": f"{at_least_one_recall:.4f}",
        "Mean Cands / S1": f"{mean_candidates:.2f}",
        "Median Cands": f"{np.median(sizes):.0f}",
        "P90 Cands": f"{np.percentile(sizes, 90):.0f}",
        "P99 Cands": f"{np.percentile(sizes, 99):.0f}",
        "Max Cands": f"{np.max(sizes):,}",
        "Reduction Ratio (%)": f"{reduction_ratio:.4f}",
    }


# ==============================================================================
# 9. LOAD CHARACTER TF-IDF TOP-50 FROM STEP 3
# ==============================================================================

def load_step3_candidates(
    s1_entities,
):
    """
    Load the actual Step 3 file produced on this machine.

    Expected schema:
        source1_entity_id
        candidate_entity_ids

    This replaces the incompatible:
        entity_id
        candidates
    schema from the original Step 4 script.
    """

    df = pd.read_csv(
        STEP3_TOP50_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "candidate_entity_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Step 3 candidate file is missing columns: "
            f"{sorted(missing)}"
        )

    step3_dict = {}

    for row in df.itertuples(index=False):

        value = (
            row.candidate_entity_ids
            if isinstance(
                row.candidate_entity_ids,
                str,
            )
            else ""
        )

        if value.strip():

            step3_dict[
                row.source1_entity_id
            ] = [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

        else:

            step3_dict[
                row.source1_entity_id
            ] = []

    del df
    gc.collect()

    # Every canonical S1 should be represented.
    missing_s1 = [
        s1_id
        for s1_id in s1_entities
        if s1_id not in step3_dict
    ]

    if missing_s1:
        raise RuntimeError(
            f"Step 3 candidate file is missing "
            f"{len(missing_s1):,} S1 entities."
        )

    return step3_dict


# ==============================================================================
# 10. MAIN
# ==============================================================================

def main():

    print("=" * 80)
    print("PHASE 10 (STEP 4): WORD / TOKEN TF-IDF RETRIEVER")
    print("=" * 80)

    print(
        f"\nProject directory: "
        f"{PROJECT_DIR}"
    )

    print(
        f"Using sparse_dot_topn: "
        f"{USE_SPARSE_DOT_TOPN}"
    )

    # --------------------------------------------------------------
    # A. Fast-dev S1 + GT
    # --------------------------------------------------------------

    s1, fast_ids = (
        reconstruct_fast_dev()
    )

    dev_gt_s2, total_true_links, s1_with_s2 = (
        build_dev_gt(
            fast_ids
        )
    )

    s1_entities = (
        s1["entity_id"].values
    )

    # --------------------------------------------------------------
    # B. Word TF-IDF
    # --------------------------------------------------------------

    (
        vectorizer,
        s1_matrix,
        country_partitions,
        s1_entities,
    ) = fit_word_vectorizer(
        s1
    )

    # --------------------------------------------------------------
    # C. S2 retrieval
    # --------------------------------------------------------------

    sorted_candidates = run_retrieval(
        vectorizer,
        country_partitions,
        s1_entities,
    )

    # --------------------------------------------------------------
    # D. Evaluate Word TF-IDF @10/@25/@50
    # --------------------------------------------------------------

    print(
        "\n[5/5] Evaluating Word TF-IDF "
        "at Top-10 / Top-25 / Top-50..."
    )

    cands_10 = {
        s1_id: candidates[:10]
        for s1_id, candidates
        in sorted_candidates.items()
    }

    cands_25 = {
        s1_id: candidates[:25]
        for s1_id, candidates
        in sorted_candidates.items()
    }

    cands_50 = {
        s1_id: candidates[:50]
        for s1_id, candidates
        in sorted_candidates.items()
    }

    results = []

    results.append(
        evaluate_retriever(
            cands_10,
            s1_entities,
            dev_gt_s2,
            total_true_links,
            s1_with_s2,
            10,
            "Channel 3: Word TF-IDF (Top-10)",
        )
    )

    results.append(
        evaluate_retriever(
            cands_25,
            s1_entities,
            dev_gt_s2,
            total_true_links,
            s1_with_s2,
            25,
            "Channel 3: Word TF-IDF (Top-25)",
        )
    )

    results.append(
        evaluate_retriever(
            cands_50,
            s1_entities,
            dev_gt_s2,
            total_true_links,
            s1_with_s2,
            50,
            "Channel 3: Word TF-IDF (Top-50)",
        )
    )

    results_df = pd.DataFrame(results)

    results_path = (
        OUTPUT_DIR /
        "phase10_step4_word_tfidf_results.tsv"
    )

    results_df.to_csv(
        results_path,
        sep="\t",
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("WORD TF-IDF RETRIEVAL RESULTS")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # E. Save Word Top-50
    # --------------------------------------------------------------

    candidate_rows = []

    for s1_id in s1_entities:

        candidate_rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(
                    cands_50.get(
                        s1_id,
                        [],
                    )
                ),
            }
        )

    word_candidate_df = pd.DataFrame(
        candidate_rows
    )

    word_candidate_path = (
        OUTPUT_DIR /
        "phase10_step4_word_tfidf_top50.tsv"
    )

    word_candidate_df.to_csv(
        word_candidate_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------------
    # F. Cumulative union with Step 3 Char Top-50
    # --------------------------------------------------------------

    print(
        "\nEvaluating cumulative union:"
        "\nChar Top-50 ∪ Word Top-50..."
    )

    step3_dict = (
        load_step3_candidates(
            s1_entities
        )
    )

    union_candidates = {}

    for s1_id in s1_entities:

        char_candidates = set(
            step3_dict.get(
                s1_id,
                [],
            )
        )

        word_candidates = set(
            cands_50.get(
                s1_id,
                [],
            )
        )

        union_candidates[s1_id] = list(
            char_candidates |
            word_candidates
        )

    # IMPORTANT:
    # The union can contain up to 100 records, unlike a single
    # channel's Top-50. Evaluate the full union so recall and
    # candidate-size statistics reflect the actual combined pool.
    union_candidate_sizes = []

    retrieved_true = 0
    all_match_count = 0
    at_least_one_count = 0

    for s1_id in s1_entities:

        candidates = set(
            union_candidates.get(
                s1_id,
                [],
            )
        )

        union_candidate_sizes.append(
            len(candidates)
        )

        true_targets = dev_gt_s2.get(
            s1_id,
            set(),
        )

        if not true_targets:
            continue

        hits = (
            candidates &
            true_targets
        )

        retrieved_true += len(hits)

        if len(hits) == len(true_targets):
            all_match_count += 1

        if len(hits) >= 1:
            at_least_one_count += 1

    union_sizes = np.asarray(
        union_candidate_sizes,
        dtype=np.int32,
    )

    union_result = {
        "Channel":
            "Cumulative Union: Char Top-50 ∪ Word Top-50",
        "Link Recall (%)":
            f"{(retrieved_true / total_true_links) * 100.0:.4f}",
        "All-Match Entity Recall (%)":
            f"{(all_match_count / len(s1_with_s2)) * 100.0:.4f}",
        "At-Least-1 Recall (%)":
            f"{(at_least_one_count / len(s1_with_s2)) * 100.0:.4f}",
        "Mean Cands / S1":
            f"{np.mean(union_sizes):.2f}",
        "Median Cands":
            f"{np.median(union_sizes):.0f}",
        "P90 Cands":
            f"{np.percentile(union_sizes, 90):.0f}",
        "P99 Cands":
            f"{np.percentile(union_sizes, 99):.0f}",
        "Max Cands":
            f"{np.max(union_sizes):,}",
        "Reduction Ratio (%)":
            f"{(1.0 - (np.mean(union_sizes) / TOTAL_S2_RECORDS)) * 100.0:.4f}",
    }

    union_df = pd.DataFrame(
        [
            results[-1],
            union_result,
        ]
    )

    union_path = (
        OUTPUT_DIR /
        "phase10_step4_cumulative_union_results.tsv"
    )

    union_df.to_csv(
        union_path,
        sep="\t",
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("CUMULATIVE MULTI-CHANNEL UNION RESULTS")
    print("=" * 80)

    print(
        union_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # G. Save config
    # --------------------------------------------------------------

    config = {
        "phase": "10_step4",
        "method": "word_tfidf",
        "analyzer": "word",
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_features": 50_000,
        "sublinear_tf": True,
        "dtype": "float32",
        "norm": "l2",
        "lowercase": False,
        "top_k": TOP_K,
        "similarity_lower_bound": SIM_LOWER_BOUND,
        "chunk_size": CHUNK_SIZE,
        "fast_dev_size": FAST_SLICE_SIZE,
        "country_partitioned": True,
        "step3_union_file": str(
            STEP3_TOP50_PATH
        ),
    }

    config_path = (
        OUTPUT_DIR /
        "phase10_step4_word_tfidf_config.json"
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

    # --------------------------------------------------------------
    # H. Save vectorizer
    # --------------------------------------------------------------

    vectorizer_path = (
        MODELS_DIR /
        "phase10_step4_word_tfidf_vectorizer.joblib"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    # --------------------------------------------------------------
    # I. Clean up
    # --------------------------------------------------------------

    del sorted_candidates
    del cands_10
    del cands_25
    del cands_50
    del union_candidates
    del step3_dict
    del union_df
    del word_candidate_df
    del results_df
    del vectorizer
    del s1_matrix
    del country_partitions
    del dev_gt_s2
    del s1

    gc.collect()

    total_minutes = (
        time.time() -
        t0_start
    ) / 60.0

    print("\n")
    print("=" * 80)
    print(
        f"PHASE 10 STEP 4 COMPLETED "
        f"in {total_minutes:.2f} minutes"
    )
    print("=" * 80)

    print(
        f"Results:    {results_path}"
    )

    print(
        f"Candidates: {word_candidate_path}"
    )

    print(
        f"Union:      {union_path}"
    )

    print(
        f"Vectorizer: {vectorizer_path}"
    )

    print(
        f"Config:     {config_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
