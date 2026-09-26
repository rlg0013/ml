from src.metrics import compute_entity_f05


def test_perfect():
    assert compute_entity_f05({"S2-1"}, {"S2-1"}) == 1.0


def test_true_singleton():
    assert compute_entity_f05(set(), set()) == 1.0


def test_false_singleton():
    assert compute_entity_f05(set(), {"S2-1"}) == 0.0


def test_false_negative():
    assert compute_entity_f05({"S2-1"}, set()) == 0.0


def test_precision_penalty():
    value = compute_entity_f05({"S2-1"}, {"S2-1", "S2-2"})
    assert abs(value - 0.5555555555555556) < 1e-9
