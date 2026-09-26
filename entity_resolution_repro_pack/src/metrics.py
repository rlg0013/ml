import numpy as np


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
        (1.0 + beta_sq) * (precision * recall)
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
        total += compute_entity_f05(actual, predicted)

    return total / len(all_s1)


def parse_match_ids(value):
    """
    Parse the challenge's comma-separated matched_entity_ids field.
    Empty string means no matches.
    """
    value = "" if value is None else str(value).strip()

    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def load_ground_truth(path):
    import pandas as pd

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    gt["target_set"] = gt["matched_entity_ids"].map(parse_match_ids)

    return dict(zip(gt["source1_entity_id"], gt["target_set"]))


def run_metric_tests():
    tests = [
        ("perfect", compute_entity_f05({"S2-1", "S2-2"}, {"S2-1", "S2-2"}), 1.0),
        ("true_singleton", compute_entity_f05(set(), set()), 1.0),
        ("false_singleton", compute_entity_f05(set(), {"S2-1"}), 0.0),
        ("false_negative", compute_entity_f05({"S2-1"}, set()), 0.0),
        ("precision_penalty", compute_entity_f05({"S2-1"}, {"S2-1", "S2-2"}), 0.5555555556),
        ("recall_penalty", compute_entity_f05({"S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S2-6"}, {"S2-1", "S2-2", "S2-3", "S2-4", "S2-5"}), 0.8333333333),
    ]

    for name, actual, expected in tests:
        assert abs(actual - expected) < 1e-9, (name, actual, expected)

    return True


if __name__ == "__main__":
    assert run_metric_tests()
    print("All metric tests passed.")
