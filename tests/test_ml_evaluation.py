import json
import math

import numpy as np
import pytest

from backend.ml.evaluation import (
    EvaluationError,
    compute_confusion_matrix,
    evaluate_hard_predictions,
    evaluate_probabilistic_predictions,
    evaluate_probabilities,
)


def test_confusion_matrix_and_hard_metrics_match_known_values():
    # TN=3, FP=1, FN=2, TP=4 (total=10; normal=4, attack=6)
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    categories = [[], [], [], [], ["escalate"], ["dirb"], ["dirb"], ["dirb"], ["foothold"], ["escalate", "foothold"]]

    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm == {"tn": 3, "fp": 1, "fn": 2, "tp": 4}

    res = evaluate_hard_predictions(y_true, y_pred, attack_categories=categories)
    assert res["confusion_matrix"] == cm
    assert res["single_class_ground_truth"] is False
    assert math.isclose(res["accuracy"], 0.7)
    assert math.isclose(res["attack_precision"], 4 / 5)
    assert math.isclose(res["attack_recall"], 4 / 6)
    expected_attack_f1 = 2 * (4 / 5) * (4 / 6) / ((4 / 5) + (4 / 6))
    assert math.isclose(res["attack_f1"], expected_attack_f1)
    assert math.isclose(res["normal_precision"], 3 / 5)
    assert math.isclose(res["normal_recall"], 3 / 4)
    expected_normal_f1 = 2 * (3 / 5) * (3 / 4) / ((3 / 5) + (3 / 4))
    assert math.isclose(res["normal_f1"], expected_normal_f1)
    assert math.isclose(res["macro_f1"], (expected_attack_f1 + expected_normal_f1) / 2)
    assert math.isclose(res["balanced_accuracy"], ((4 / 6) + (3 / 4)) / 2)
    assert math.isclose(res["normal_false_positive_rate"], 1 / 4)
    assert res["per_attack_category_recall"] == {
        "dirb": 2 / 3,
        "escalate": 1 / 2,
        "foothold": 1.0,
    }
    # Must be JSON-serializable and deterministic
    assert json.dumps(res, sort_keys=True) == json.dumps(
        evaluate_hard_predictions(y_true, y_pred, attack_categories=categories),
        sort_keys=True,
    )


def test_probabilistic_evaluation_thresholds_and_normal_score_inversion():
    # Perfectly separated probabilities: normal -> low p_attack, attack -> high p_attack
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.9]

    res = evaluate_probabilistic_predictions(
        y_true, y_prob, selected_threshold=0.7
    )
    prob = res["probabilistic_metrics"]
    assert math.isclose(prob["pr_auc_attack"], 1.0)
    assert math.isclose(prob["pr_auc_normal"], 1.0)
    expected_brier = ((0.1 - 0) ** 2 + (0.4 - 0) ** 2 + (0.6 - 1) ** 2 + (0.9 - 1) ** 2) / 4
    assert math.isclose(prob["brier_score"], expected_brier)

    # At fixed 0.5: [0, 0, 1, 1] -> perfect
    fixed = res["fixed_threshold_metrics"]
    assert fixed["threshold"] == 0.5
    assert fixed["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert math.isclose(fixed["macro_f1"], 1.0)
    assert math.isclose(fixed["normal_false_positive_rate"], 0.0)

    # At selected 0.7: [0, 0, 0, 1] -> 1 FN
    sel = res["selected_threshold_metrics"]
    assert sel["threshold"] == 0.7
    assert sel["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert math.isclose(sel["attack_recall"], 0.5)

    # Inverted probabilities: normal gets high p_attack -> low 1 - p_attack -> worse pr_auc_normal
    inverted = evaluate_probabilities(y_true, [0.9, 0.6, 0.4, 0.1])
    assert inverted["pr_auc_attack"] < 0.5
    assert inverted["pr_auc_normal"] < 0.5


def test_single_class_and_zero_division_behaviour():
    # Constant-prediction baseline (e.g. DummyClassifier most_frequent -> all 1s)
    const_res = evaluate_hard_predictions([0, 0, 1, 1], [1, 1, 1, 1])
    assert const_res["normal_precision"] == 0.0
    assert const_res["normal_recall"] == 0.0
    assert const_res["normal_f1"] == 0.0
    assert const_res["normal_false_positive_rate"] == 1.0

    # Single-class ground truth: undefined metrics stay None (not invented)
    only_attack = evaluate_probabilistic_predictions([1, 1, 1], [0.2, 0.8, 0.9])
    assert only_attack["probabilistic_metrics"]["single_class_ground_truth"] is True
    assert only_attack["probabilistic_metrics"]["pr_auc_attack"] is None
    assert only_attack["probabilistic_metrics"]["pr_auc_normal"] is None
    assert only_attack["probabilistic_metrics"]["brier_score"] is not None
    fixed = only_attack["fixed_threshold_metrics"]
    assert fixed["macro_f1"] is None
    assert fixed["balanced_accuracy"] is None
    assert fixed["normal_false_positive_rate"] is None


def test_invalid_inputs_fail_closed():
    with pytest.raises(EvaluationError):
        compute_confusion_matrix([], [])
    with pytest.raises(EvaluationError):
        compute_confusion_matrix([0, 1], [0])
    with pytest.raises(EvaluationError):
        compute_confusion_matrix([True, False], [1, 0])
    with pytest.raises(EvaluationError):
        compute_confusion_matrix([0, 2], [0, 1])
    with pytest.raises(EvaluationError):
        evaluate_probabilities([0, 1], [0.5, float("nan")])
    with pytest.raises(EvaluationError):
        evaluate_probabilities([0, 1], [0.5, float("inf")])
    with pytest.raises(EvaluationError):
        evaluate_probabilities([0, 1], [-0.1, 0.5])
    with pytest.raises(EvaluationError):
        evaluate_probabilities([0, 1], [0.5, 1.1])
    with pytest.raises(EvaluationError):
        evaluate_probabilities([0, 1], np.array([[0.2], [0.8]]))
    with pytest.raises(EvaluationError):
        evaluate_probabilistic_predictions([0, 1], [0.2, 0.8], selected_threshold=1.5)
