from pathlib import Path
import pandas as pd


EXPECTED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def load_source(path, usecols=None):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
    )


def load_s1_fast_slice(s1_path, split_path, chunksize=250_000):
    """
    Load only S1 records marked fast_slice=True in s1_splits.tsv.
    """
    splits = pd.read_csv(
        split_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id", "fast_slice"],
    )

    mask = splits["fast_slice"].astype(str).str.lower().eq("true")
    slice_ids = set(splits.loc[mask, "source1_entity_id"])

    rows = []

    reader = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        usecols=EXPECTED_SOURCE_COLUMNS,
    )

    for chunk in reader:
        matched = chunk[chunk["entity_id"].isin(slice_ids)]
        if not matched.empty:
            rows.append(matched.copy())

    if not rows:
        raise RuntimeError("Fast-dev S1 slice was not found.")

    result = pd.concat(rows, ignore_index=True)

    if len(result) != len(slice_ids):
        raise RuntimeError(
            f"Fast-dev S1 size mismatch: {len(result)} rows for {len(slice_ids)} IDs."
        )

    return result, slice_ids
