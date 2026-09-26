import numpy as np
import pandas as pd


def compute_entity_f05(actual_set, pred_set) -> float:
    if not isinstance(actual_set, set):
        actual_set = set(actual_set)

    if not isinstance(pred_set, set):
        pred_set = set(pred_set)

    len_act = len(actual_set)
    len_pred = len(pred_set)

    if len_act == 0:
        return 1.0 if len_pred == 0 else 0.0

    if len_pred == 0:
        return 0.0

    tp = len(actual_set & pred_set)

    if tp == 0:
        return 0.0

    precision = tp / len_pred
    recall = tp / len_act

    beta_sq = 0.25

    return float(
        (1.0 + beta_sq)
        * (precision * recall)
        / (beta_sq * precision + recall)
    )


def compute_macro_f05(actual_dict, pred_dict) -> float:
    all_s1 = set(actual_dict.keys()) | set(pred_dict.keys())

    if not all_s1:
        return 0.0

    total = 0.0

    for s1_id in all_s1:
        actual = actual_dict.get(s1_id, set())
        predicted = pred_dict.get(s1_id, set())

        total += compute_entity_f05(
            actual,
            predicted
        )

    return total / len(all_s1)


def parse_match_ids(value):
    value = "" if value is None else str(value).strip()

    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def load_ground_truth(path):
    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt["target_set"] = (
        gt["matched_entity_ids"]
        .map(parse_match_ids)
    )

    return dict(
        zip(
            gt["source1_entity_id"],
            gt["target_set"],
        )
    )
