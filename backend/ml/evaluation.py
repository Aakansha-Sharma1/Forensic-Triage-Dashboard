from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


FIXED_THRESHOLD = 0.5
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0
ZERO_DIVISION = 0


class EvaluationError(ValueError):
    """Raised when evaluation inputs (labels, predictions, probabilities, thresholds) are invalid."""


def _validate_binary_vector(values: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or len(values) == 0:
            raise EvaluationError(f"{name} must be a non-empty 1D array.")
        if values.dtype == np.bool_:
            raise EvaluationError(f"{name} must contain integer 0 or 1, not booleans.")
        raw = values.tolist()
    else:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) == 0:
            raise EvaluationError(f"{name} must be a non-empty 1D sequence.")
        raw = list(values)

    parsed: list[int] = []
    for idx, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in (0, 1):
            raise EvaluationError(
                f"{name}[{idx}] must be integer 0 or 1, got {value!r}."
            )
        parsed.append(int(value))
    return np.asarray(parsed, dtype=np.int64)


def _validate_probabilities(
    probabilities: Sequence[Any] | np.ndarray,
    expected_length: int,
) -> np.ndarray:
    if isinstance(probabilities, np.ndarray):
        if probabilities.ndim != 1 or len(probabilities) != expected_length:
            raise EvaluationError(
                f"y_prob_attack must be a 1D array of length {expected_length}."
            )
        if probabilities.dtype == np.bool_:
            raise EvaluationError("y_prob_attack must contain numeric probabilities, not booleans.")
        if probabilities.dtype == object:
            return _validate_probabilities(probabilities.tolist(), expected_length)
        try:
            array = np.asarray(probabilities, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise EvaluationError("y_prob_attack must contain finite floats in [0, 1].") from exc
    else:
        if (
            not isinstance(probabilities, Sequence)
            or isinstance(probabilities, (str, bytes))
            or len(probabilities) != expected_length
        ):
            raise EvaluationError(
                f"y_prob_attack must be a 1D sequence of length {expected_length}."
            )
        parsed: list[float] = []
        for idx, value in enumerate(probabilities):
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise EvaluationError(
                    f"y_prob_attack[{idx}] must be a numeric probability, got {value!r}."
                )
            parsed.append(float(value))
        array = np.asarray(parsed, dtype=np.float64)

    if not np.isfinite(array).all():
        raise EvaluationError("y_prob_attack must not contain NaN or infinity.")
    if ((array < 0.0) | (array > 1.0)).any():
        raise EvaluationError("y_prob_attack values must lie within [0.0, 1.0].")
    return array


def _validate_threshold(threshold: object, name: str = "threshold") -> float:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not math.isfinite(float(threshold))
        or not (0.0 <= float(threshold) <= 1.0)
    ):
        raise EvaluationError(f"{name} must be a finite number in [0.0, 1.0], got {threshold!r}.")
    return float(threshold)


def compute_confusion_matrix(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
) -> dict[str, int]:
    """Compute binary confusion matrix with attack_associated=1 (positive) and normal=0 (negative)."""
    yt = _validate_binary_vector(y_true, "y_true")
    yp = _validate_binary_vector(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise EvaluationError("y_true and y_pred must have identical lengths.")
    return {
        "tn": int(((yt == NEGATIVE_CLASS) & (yp == NEGATIVE_CLASS)).sum()),
        "fp": int(((yt == NEGATIVE_CLASS) & (yp == POSITIVE_CLASS)).sum()),
        "fn": int(((yt == POSITIVE_CLASS) & (yp == NEGATIVE_CLASS)).sum()),
        "tp": int(((yt == POSITIVE_CLASS) & (yp == POSITIVE_CLASS)).sum()),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator > 0 else float(ZERO_DIVISION)


def _safe_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return float((2.0 * precision * recall) / denom) if denom > 0.0 else float(ZERO_DIVISION)


def _compute_per_category_recall(
    yt: np.ndarray,
    yp: np.ndarray,
    attack_categories: Sequence[Sequence[str]],
) -> dict[str, float]:
    if (
        not isinstance(attack_categories, Sequence)
        or isinstance(attack_categories, (str, bytes))
        or len(attack_categories) != len(yt)
    ):
        raise EvaluationError(
            f"attack_categories must be a sequence of length {len(yt)}."
        )
    totals: Counter[str] = Counter()
    recalled: Counter[str] = Counter()
    for idx, categories in enumerate(attack_categories):
        if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
            raise EvaluationError(f"attack_categories[{idx}] must be a sequence of strings.")
        if int(yt[idx]) != POSITIVE_CLASS:
            continue
        seen_in_event: set[str] = set()
        for cat in categories:
            if not isinstance(cat, str) or not cat.strip():
                raise EvaluationError(f"Invalid attack category at index {idx}: {cat!r}")
            if cat in seen_in_event:
                continue
            seen_in_event.add(cat)
            totals[cat] += 1
            if int(yp[idx]) == POSITIVE_CLASS:
                recalled[cat] += 1
    return {cat: float(recalled[cat] / totals[cat]) for cat in sorted(totals)}


def evaluate_hard_predictions(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    *,
    attack_categories: Sequence[Sequence[str]] | None = None,
) -> dict[str, object]:
    """Compute protocol metrics for hard binary {0, 1} predictions."""
    yt = _validate_binary_vector(y_true, "y_true")
    yp = _validate_binary_vector(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise EvaluationError("y_true and y_pred must have identical lengths.")

    cm = compute_confusion_matrix(yt, yp)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = len(yt)
    normal_support = tn + fp
    attack_support = tp + fn
    both_classes_present = normal_support > 0 and attack_support > 0

    accuracy = float((tp + tn) / total)
    attack_precision = _safe_ratio(tp, tp + fp)
    attack_recall = _safe_ratio(tp, attack_support)
    attack_f1 = _safe_f1(attack_precision, attack_recall)

    normal_precision = _safe_ratio(tn, tn + fn)
    normal_recall = _safe_ratio(tn, normal_support)
    normal_f1 = _safe_f1(normal_precision, normal_recall)

    macro_f1: float | None = (
        float((attack_f1 + normal_f1) / 2.0) if both_classes_present else None
    )
    balanced_accuracy: float | None = (
        float((attack_recall + normal_recall) / 2.0) if both_classes_present else None
    )
    normal_fpr: float | None = (
        float(fp / normal_support) if normal_support > 0 else None
    )

    result: dict[str, object] = {
        "sample_count": total,
        "class_support": {
            "normal": normal_support,
            "attack_associated": attack_support,
        },
        "single_class_ground_truth": not both_classes_present,
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "normal_precision": normal_precision,
        "normal_recall": normal_recall,
        "normal_f1": normal_f1,
        "normal_false_positive_rate": normal_fpr,
    }
    if attack_categories is not None:
        result["per_attack_category_recall"] = _compute_per_category_recall(
            yt, yp, attack_categories
        )
    return result


def evaluate_probabilities(
    y_true: Sequence[Any] | np.ndarray,
    y_prob_attack: Sequence[Any] | np.ndarray,
) -> dict[str, object]:
    """Compute threshold-independent probability metrics (class-specific PR-AUC and Brier score)."""
    yt = _validate_binary_vector(y_true, "y_true")
    probs = _validate_probabilities(y_prob_attack, len(yt))
    normal_support = int((yt == NEGATIVE_CLASS).sum())
    attack_support = int((yt == POSITIVE_CLASS).sum())
    both_classes_present = normal_support > 0 and attack_support > 0

    brier_score = float(np.mean((probs - yt.astype(np.float64)) ** 2))
    if both_classes_present:
        pr_auc_attack: float | None = float(
            average_precision_score((yt == POSITIVE_CLASS).astype(np.int64), probs)
        )
        pr_auc_normal: float | None = float(
            average_precision_score(
                (yt == NEGATIVE_CLASS).astype(np.int64), 1.0 - probs
            )
        )
    else:
        pr_auc_attack = None
        pr_auc_normal = None

    return {
        "single_class_ground_truth": not both_classes_present,
        "pr_auc_attack": pr_auc_attack,
        "pr_auc_normal": pr_auc_normal,
        "brier_score": brier_score,
    }


def evaluate_probabilistic_predictions(
    y_true: Sequence[Any] | np.ndarray,
    y_prob_attack: Sequence[Any] | np.ndarray,
    *,
    selected_threshold: float | None = None,
    fixed_threshold: float = FIXED_THRESHOLD,
    attack_categories: Sequence[Sequence[str]] | None = None,
) -> dict[str, object]:
    """Evaluate probabilistic outputs at fixed threshold 0.5 and an optional validation-selected threshold."""
    yt = _validate_binary_vector(y_true, "y_true")
    probs = _validate_probabilities(y_prob_attack, len(yt))
    fixed_t = _validate_threshold(fixed_threshold, "fixed_threshold")
    selected_t = (
        None
        if selected_threshold is None
        else _validate_threshold(selected_threshold, "selected_threshold")
    )

    fixed_preds = (probs >= fixed_t).astype(np.int64)
    fixed_metrics = {
        "threshold": fixed_t,
        **evaluate_hard_predictions(yt, fixed_preds, attack_categories=attack_categories),
    }

    selected_metrics: dict[str, object] | None = None
    if selected_t is not None:
        selected_preds = (probs >= selected_t).astype(np.int64)
        selected_metrics = {
            "threshold": selected_t,
            **evaluate_hard_predictions(
                yt, selected_preds, attack_categories=attack_categories
            ),
        }

    return {
        "label_convention": {
            "positive_class": POSITIVE_CLASS,
            "positive_name": "attack_associated",
            "negative_class": NEGATIVE_CLASS,
            "negative_name": "normal",
            "zero_division": ZERO_DIVISION,
            "normal_false_positive_rate_formula": "FP / (FP + TN)",
        },
        "probabilistic_metrics": evaluate_probabilities(yt, probs),
        "fixed_threshold_metrics": fixed_metrics,
        "selected_threshold_metrics": selected_metrics,
    }
