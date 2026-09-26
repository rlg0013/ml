import pandas as pd


def get_cardinality_bucket(value):
    value = "" if value is None else str(value).strip()

    if not value:
        return "0"

    count = len(value.split(","))

    if count == 1:
        return "1"
    if count in (2, 3):
        return "2-3"
    return "4+"


def build_grouped_split(
    gt_path,
    s1_path,
    output_path,
    holdout_frac=0.20,
    fast_slice_size=50_000,
    random_state=42,
):
    """
    Rebuild the project's S1-grouped validation metadata.

    Splitting is done at S1 level, before pair generation.
    The fast slice is sampled only from development S1s.
    """
    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    s1 = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "country"],
    ).rename(columns={"entity_id": "source1_entity_id"})

    df = gt.merge(
        s1,
        on="source1_entity_id",
        how="inner",
        validate="one_to_one",
    )

    df["card_bucket"] = df["matched_entity_ids"].map(get_cardinality_bucket)
    df["strat_key"] = df["country"] + "_" + df["card_bucket"]

    holdout = (
        df.groupby("strat_key", group_keys=False)
        .sample(frac=holdout_frac, random_state=random_state)
    )

    df["split"] = "dev"
    df.loc[holdout.index, "split"] = "holdout"

    dev = df[df["split"] == "dev"].copy()

    # Proportional exact-size fast slice.
    group_sizes = dev["strat_key"].value_counts().sort_index()
    raw = group_sizes / len(dev) * fast_slice_size
    allocation = raw.astype(int)

    remainder = fast_slice_size - int(allocation.sum())
    fractional = raw - allocation

    if remainder > 0:
        for key in fractional.sort_values(ascending=False).index[:remainder]:
            allocation.loc[key] += 1

    parts = []
    for key, n in allocation.items():
        group = dev[dev["strat_key"] == key]
        parts.append(
            group.sample(n=int(n), random_state=random_state)
        )

    fast_slice = pd.concat(parts, axis=0)

    if len(fast_slice) != fast_slice_size:
        raise RuntimeError("Fast slice size mismatch.")

    df["fast_slice"] = False
    df.loc[fast_slice.index, "fast_slice"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df[
        [
            "source1_entity_id",
            "country",
            "card_bucket",
            "split",
            "fast_slice",
        ]
    ].to_csv(output_path, sep="\t", index=False)

    return df
