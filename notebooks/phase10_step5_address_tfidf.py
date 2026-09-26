# ==============================================================================
# PHASE 10 — STEP 5
# ADDRESS TF-IDF RETRIEVER — STANDALONE LOCAL VERSION
#
# Corrected to match the current local project state:
#   - uses output/s1_splits.tsv schema:
#       source1_entity_id, country, card_bucket, split, fast_slice
#   - uses Step 3/4 candidate-file schema:
#       source1_entity_id, candidate_entity_ids
#   - uses canonical src/normalization.py
#   - uses canonical src/metrics.py
#   - no notebook globals required
#
# Validation slice:
#   50,000 fast-dev S1 entities
#
# Retrieval:
#   Address word-level TF-IDF
#   analyzer="word"
#   ngram_range=(1,2)
#   min_df=2
#   max_features=50,000
#   Top-K = 30
#   minimum cosine similarity = 0.40
#
# Evaluates:
#   Address TF-IDF @10 / @20 / @30
#   Char Top-50 ∪ Word Top-50 ∪ Address Top-30
#
# NOTE:
#   Address-number overlap/conflict is an important later pair feature.
#   This retrieval experiment itself uses address TF-IDF; it does not replace
#   number-aware pair scoring.
# ==============================================================================

import gc
import heapq
import json
import os
import sys
import time
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

STEP4_TOP50_PATH = (
    OUTPUT_DIR /
    "phase10_step4_word_tfidf_top50.tsv"
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

TOP_K = 30
SIM_LOWER_BOUND = 0.40

ADDRESS_TFIDF_CONFIG = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 50_000,
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": "l2",
    "lowercase": False,
}

N_THREADS = max(
    1,
    (os.cpu_count() or 2) - 1
)

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
    STEP4_TOP50_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ==============================================================================
# 3. RECONSTRUCT FAST-DEV S1
# ==============================================================================

def reconstruct_fast_dev():
    print(
        "[1/6] Loading canonical fast-dev "
        "S1 slice and addresses..."
    )

    split_df = pd.read_csv(
        SPLIT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_split_columns = {
        "source1_entity_id",
        "fast_slice",
    }

    missing = (
        required_split_columns -
        set(split_df.columns)
    )

    if missing:
        raise ValueError(
            "s1_splits.tsv missing columns: "
            f"{sorted(missing)}"
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
            f"Expected {FAST_SLICE_SIZE:,} fast-dev IDs, "
            f"found {len(fast_ids):,}"
        )

    rows = []

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        usecols=[
            "entity_id",
            "business_address",
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
            f"found {len(s1):,}"
        )

    # Canonical Phase 7 normalization.
    s1["norm_address"] = (
        s1["business_address"]
        .map(normalize_text)
    )

    return s1, fast_ids


# ==============================================================================
# 4. GROUND TRUTH
# ==============================================================================

def build_dev_gt(
    fast_ids,
):
    print(
        "[2/6] Loading ground truth..."
    )

    gt = load_ground_truth(
        GT_PATH
    )

    dev_gt_s2 = {}

    for s1_id in fast_ids:

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
        len(v)
        for v in dev_gt_s2.values()
    )

    s1_with_s2 = [
        s1_id
        for s1_id, targets
        in dev_gt_s2.items()
        if targets
    ]

    print(
        f"  * S1 queries: "
        f"{len(fast_ids):,}"
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
# 5. FIT ADDRESS TF-IDF
# ==============================================================================

def fit_address_vectorizer(
    s1,
):
    print(
        "\n[3/6] Fitting Address TF-IDF..."
    )

    t0 = time.time()

    vectorizer = TfidfVectorizer(
        **ADDRESS_TFIDF_CONFIG
    )

    addresses = (
        s1["norm_address"]
        .fillna("")
        .tolist()
    )

    s1_tfidf = (
        vectorizer
        .fit_transform(addresses)
    )

    print(
        f"  * Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"  * S1 matrix shape: "
        f"{s1_tfidf.shape}"
    )

    print(
        f"  * S1 nonzeros: "
        f"{s1_tfidf.nnz:,}"
    )

    print(
        f"  * Fit time: "
        f"{time.time() - t0:.2f}s"
    )

    # Generic country partitioning.
    s1_ids = s1["entity_id"].values
    countries = s1["country"].values

    country_matrices = {}

    for country in np.unique(
        countries
    ):

        indices = np.flatnonzero(
            countries == country
        )

        country_matrices[country] = {
            "matrix": s1_tfidf[
                indices
            ],
            "ids": s1_ids[
                indices
            ],
        }

        print(
            f"    - {country}: "
            f"{len(indices):,} queries"
        )

    # Save vectorizer immediately so the experiment's
    # learned representation survives the run.
    vectorizer_path = (
        MODELS_DIR /
        "phase10_step5_address_tfidf_vectorizer.joblib"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    return (
        vectorizer,
        s1_tfidf,
        country_matrices,
        s1_ids,
    )


# ==============================================================================
# 6. SPARSE TOP-K SEARCH
# ==============================================================================

def update_topk(
    heap,
    similarity,
    s2_id,
):
    item = (
        float(similarity),
        s2_id,
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
    if (
        query_matrix.shape[0] == 0
        or s2_matrix.shape[0] == 0
    ):
        return

    if USE_SPARSE_DOT_TOPN:

        topn = sp_matmul_topn(
            query_matrix,
            s2_matrix.T,
            top_n=TOP_K,
            threshold=SIM_LOWER_BOUND,
            sort=True,
            n_threads=N_THREADS,
        )

        coo = topn.tocoo()

        for row_idx, col_idx, value in zip(
            coo.row,
            coo.col,
            coo.data,
        ):

            update_topk(
                heaps[
                    query_ids[row_idx]
                ],
                float(value),
                s2_ids[col_idx],
            )

        del topn
        return

    # Safe sparse SciPy fallback.
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

        for row_idx, col_idx, value in zip(
            block.row,
            block.col,
            block.data,
        ):

            value = float(value)

            if value < SIM_LOWER_BOUND:
                continue

            update_topk(
                heaps[
                    query_ids[
                        start + row_idx
                    ]
                ],
                value,
                s2_ids[col_idx],
            )

        del block

    del s2_t


# ==============================================================================
# 7. STREAM S2
# ==============================================================================

def run_retrieval(
    vectorizer,
    country_matrices,
    s1_ids,
):
    print(
        f"\n[4/6] Streaming S2 in "
        f"{CHUNK_SIZE:,}-row chunks..."
    )

    heaps = {
        s1_id: []
        for s1_id in s1_ids
    }

    reader = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        usecols=[
            "entity_id",
            "business_address",
            "country",
        ],
    )

    rows_done = 0
    chunk_idx = 0
    t0_stream = time.time()

    for chunk in reader:

        chunk_idx += 1

        norm_addresses = [
            normalize_text(address)
            for address in chunk[
                "business_address"
            ].values
        ]

        countries = (
            chunk["country"].values
        )

        ids = (
            chunk["entity_id"].values
        )

        s2_tfidf = (
            vectorizer
            .transform(norm_addresses)
        )

        # Search only within matching country.
        # This is a soft retrieval restriction based on the
        # observed country field; the implementation remains
        # generic for unseen country strings.
        for country, query_info in (
            country_matrices.items()
        ):

            mask = (
                countries == country
            )

            if not np.any(mask):
                continue

            s2_sub = (
                s2_tfidf[mask]
            )

            s2_ids = (
                ids[mask]
            )

            search_country_chunk(
                query_info["matrix"],
                query_info["ids"],
                s2_sub,
                s2_ids,
                heaps,
            )

            del s2_sub
            del s2_ids

        rows_done += len(chunk)

        elapsed = (
            time.time() -
            t0_stream
        )

        pct = (
            rows_done /
            TOTAL_S2_RECORDS
        ) * 100.0

        print(
            f"  Chunk {chunk_idx:02d} | "
            f"{rows_done:,}/"
            f"{TOTAL_S2_RECORDS:,} "
            f"({pct:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s"
        )

        del chunk
        del norm_addresses
        del countries
        del ids
        del s2_tfidf

        gc.collect()

    print(
        "\nSorting address candidate lists..."
    )

    sorted_candidates = {}

    for s1_id, heap in heaps.items():

        ordered = sorted(
            heap,
            key=lambda x: x[0],
            reverse=True,
        )

        sorted_candidates[s1_id] = [
            s2_id
            for _, s2_id
            in ordered
        ]

    del heaps
    gc.collect()

    return sorted_candidates


# ==============================================================================
# 8. EVALUATION
# ==============================================================================

def evaluate_retriever(
    candidates,
    s1_ids,
    dev_gt_s2,
    total_true_links,
    s1_with_s2,
    cutoff,
    label,
):
    retrieved_true = 0
    all_match_count = 0
    at_least_one_count = 0

    candidate_sizes = []

    for s1_id in s1_ids:

        selected = set(
            candidates.get(
                s1_id,
                []
            )[:cutoff]
        )

        candidate_sizes.append(
            len(selected)
        )

        true_targets = dev_gt_s2.get(
            s1_id,
            set()
        )

        if not true_targets:
            continue

        hits = (
            selected &
            true_targets
        )

        retrieved_true += len(hits)

        if len(hits) == len(
            true_targets
        ):
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
        "Link Recall (%)":
            f"{link_recall:.4f}",
        "All-Match Entity Recall (%)":
            f"{all_match_recall:.4f}",
        "At-Least-1 Recall (%)":
            f"{at_least_one_recall:.4f}",
        "Mean Cands / S1":
            f"{mean_candidates:.2f}",
        "Median Cands":
            f"{np.median(sizes):.0f}",
        "P90 Cands":
            f"{np.percentile(sizes, 90):.0f}",
        "P99 Cands":
            f"{np.percentile(sizes, 99):.0f}",
        "Max Cands":
            f"{np.max(sizes):,}",
        "Reduction Ratio (%)":
            f"{reduction_ratio:.4f}",
    }


def evaluate_full_union(
    union_candidates,
    s1_ids,
    dev_gt_s2,
    total_true_links,
    s1_with_s2,
):
    """
    Evaluate the actual union without truncating it.
    """

    retrieved_true = 0
    all_match_count = 0
    at_least_one_count = 0

    sizes = []

    for s1_id in s1_ids:

        candidates = set(
            union_candidates.get(
                s1_id,
                [],
            )
        )

        sizes.append(
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

        if len(hits) == len(
            true_targets
        ):
            all_match_count += 1

        if len(hits) >= 1:
            at_least_one_count += 1

    sizes = np.asarray(
        sizes,
        dtype=np.int32,
    )

    return {
        "Channel":
            "3-Channel Union: Char50 ∪ Word50 ∪ Addr30",
        "Link Recall (%)":
            f"{(retrieved_true / total_true_links) * 100.0:.4f}",
        "All-Match Entity Recall (%)":
            f"{(all_match_count / len(s1_with_s2)) * 100.0:.4f}",
        "At-Least-1 Recall (%)":
            f"{(at_least_one_count / len(s1_with_s2)) * 100.0:.4f}",
        "Mean Cands / S1":
            f"{np.mean(sizes):.2f}",
        "Median Cands":
            f"{np.median(sizes):.0f}",
        "P90 Cands":
            f"{np.percentile(sizes, 90):.0f}",
        "P99 Cands":
            f"{np.percentile(sizes, 99):.0f}",
        "Max Cands":
            f"{np.max(sizes):,}",
        "Reduction Ratio (%)":
            f"{(1.0 - (np.mean(sizes) / TOTAL_S2_RECORDS)) * 100.0:.4f}",
    }


# ==============================================================================
# 9. LOAD EXISTING CHAR / WORD CANDIDATE FILES
# ==============================================================================

def load_candidates_file(
    path,
    s1_ids,
):
    """
    Current project schema:
        source1_entity_id
        candidate_entity_ids

    Supports no old/incompatible schema so that malformed files
    fail early instead of silently producing bad union results.
    """

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "candidate_entity_ids",
    }

    missing = (
        required -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} missing columns: "
            f"{sorted(missing)}"
        )

    candidates = {}

    for row in df.itertuples(
        index=False
    ):

        value = (
            row.candidate_entity_ids
        )

        if value.strip():

            candidates[
                row.source1_entity_id
            ] = [
                x.strip()
                for x in value.split(",")
                if x.strip()
            ]

        else:

            candidates[
                row.source1_entity_id
            ] = []

    # Ensure all S1 queries are present.
    missing_ids = [
        s1_id
        for s1_id in s1_ids
        if s1_id not in candidates
    ]

    if missing_ids:
        raise RuntimeError(
            f"{path.name} is missing "
            f"{len(missing_ids):,} S1 rows."
        )

    del df
    gc.collect()

    return candidates


# ==============================================================================
# 10. MAIN
# ==============================================================================

def main():

    print("=" * 80)
    print(
        "PHASE 10 (STEP 5): "
        "ADDRESS TF-IDF RETRIEVER"
    )
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
    # A. Fast-dev data + GT
    # --------------------------------------------------------------

    s1, fast_ids = (
        reconstruct_fast_dev()
    )

    (
        dev_gt_s2,
        total_true_links,
        s1_with_s2,
    ) = build_dev_gt(
        fast_ids
    )

    s1_entities = (
        s1["entity_id"].values
    )

    # --------------------------------------------------------------
    # B. Address TF-IDF
    # --------------------------------------------------------------

    (
        vectorizer,
        s1_tfidf,
        country_matrices,
        s1_entities,
    ) = fit_address_vectorizer(
        s1
    )

    # --------------------------------------------------------------
    # C. Retrieval
    # --------------------------------------------------------------

    sorted_candidates = (
        run_retrieval(
            vectorizer,
            country_matrices,
            s1_entities,
        )
    )

    # --------------------------------------------------------------
    # D. Evaluate address @10/@20/@30
    # --------------------------------------------------------------

    print(
        "\n[5/6] Evaluating Address TF-IDF "
        "Top-10 / Top-20 / Top-30..."
    )

    cands_10 = {
        s1_id: candidates[:10]
        for s1_id, candidates
        in sorted_candidates.items()
    }

    cands_20 = {
        s1_id: candidates[:20]
        for s1_id, candidates
        in sorted_candidates.items()
    }

    cands_30 = {
        s1_id: candidates[:30]
        for s1_id, candidates
        in sorted_candidates.items()
    }

    address_results = [
        evaluate_retriever(
            cands_10,
            s1_entities,
            dev_gt_s2,
            total_true_links,
            s1_with_s2,
            10,
            "Channel 4: Address TF-IDF (Top-10)",
        ),
        evaluate_retriever(
            cands_20,
            s1_entities,
            dev_gt_s2,
            total_true_links,
            s1_with_s2,
            20,
            "Channel 4: Address TF-IDF (Top-20)",
        ),
        evaluate_retriever(
            cands_30,
            s1_entities,
            dev_gt_s2,
            total_true_links,
            s1_with_s2,
            30,
            "Channel 4: Address TF-IDF (Top-30)",
        ),
    ]

    address_df = pd.DataFrame(
        address_results
    )

    address_results_path = (
        OUTPUT_DIR /
        "phase10_step5_address_tfidf_results.tsv"
    )

    address_df.to_csv(
        address_results_path,
        sep="\t",
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("ADDRESS TF-IDF RETRIEVAL RESULTS")
    print("=" * 80)

    print(
        address_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # E. Save address Top-30
    # --------------------------------------------------------------

    address_candidate_rows = []

    for s1_id in s1_entities:

        address_candidate_rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(
                    cands_30.get(
                        s1_id,
                        [],
                    )
                ),
            }
        )

    address_candidates_df = pd.DataFrame(
        address_candidate_rows
    )

    address_candidates_path = (
        OUTPUT_DIR /
        "phase10_step5_address_tfidf_top30.tsv"
    )

    address_candidates_df.to_csv(
        address_candidates_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------------
    # F. Complete 3-channel union
    # --------------------------------------------------------------

    print(
        "\n[6/6] Evaluating complete "
        "3-channel union..."
    )

    char_candidates = load_candidates_file(
        STEP3_TOP50_PATH,
        s1_entities,
    )

    word_candidates = load_candidates_file(
        STEP4_TOP50_PATH,
        s1_entities,
    )

    union_candidates = {}

    for s1_id in s1_entities:

        char_set = set(
            char_candidates.get(
                s1_id,
                [],
            )
        )

        word_set = set(
            word_candidates.get(
                s1_id,
                [],
            )
        )

        address_set = set(
            cands_30.get(
                s1_id,
                [],
            )
        )

        union_candidates[s1_id] = list(
            char_set |
            word_set |
            address_set
        )

    union_result = evaluate_full_union(
        union_candidates,
        s1_entities,
        dev_gt_s2,
        total_true_links,
        s1_with_s2,
    )

    union_df = pd.DataFrame(
        [
            address_results[-1],
            union_result,
        ]
    )

    union_path = (
        OUTPUT_DIR /
        "phase10_step5_cumulative_union_results.tsv"
    )

    union_df.to_csv(
        union_path,
        sep="\t",
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("3-CHANNEL CUMULATIVE UNION RESULTS")
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
        "phase": "10_step5",
        "method": "address_tfidf",
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
        "step3_candidate_file": str(
            STEP3_TOP50_PATH
        ),
        "step4_candidate_file": str(
            STEP4_TOP50_PATH
        ),
    }

    config_path = (
        OUTPUT_DIR /
        "phase10_step5_address_tfidf_config.json"
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
    # H. Cleanup
    # --------------------------------------------------------------

    del sorted_candidates
    del cands_10
    del cands_20
    del cands_30
    del char_candidates
    del word_candidates
    del union_candidates
    del address_results
    del address_df
    del address_candidates_df
    del union_df
    del vectorizer
    del s1_tfidf
    del country_matrices
    del dev_gt_s2
    del s1

    gc.collect()

    elapsed_minutes = (
        time.time() -
        t0_start
    ) / 60.0

    print("\n")
    print("=" * 80)
    print(
        f"PHASE 10 STEP 5 COMPLETED "
        f"in {elapsed_minutes:.2f} minutes"
    )
    print("=" * 80)

    print(
        f"Results:    "
        f"{address_results_path}"
    )

    print(
        f"Candidates: "
        f"{address_candidates_path}"
    )

    print(
        f"Union:      "
        f"{union_path}"
    )

    print(
        f"Vectorizer: "
        f"{MODELS_DIR / 'phase10_step5_address_tfidf_vectorizer.joblib'}"
    )

    print(
        f"Config:     "
        f"{config_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
