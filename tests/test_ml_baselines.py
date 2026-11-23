import math

import numpy as np
import pytest

from backend.ml.baselines import (
    BINARY_FEATURE_NAMES,
    BaselineModelError,
    FIXED_CLASSIFICATION_THRESHOLD,
    LOG1P_FEATURE_NAMES,
    compute_sample_weights,
    records_to_arrays,
    train_logistic_regression_baseline,
)
from backend.ml.training_data import EXPECTED_FEATURE_NAMES, SupervisedRecord


def _synthetic_dataset():
    """Return synthetic 13-feature rows across two imbalanced scenarios."""
    rows = []
    labels = []
    scenarios = []
    # Scenario A: 10 normal, 2 attack (low volume, normal-heavy)
    for i in range(10):
        rows.append(
            [
                200 + i * 10,          # transfer_size_bytes
                1 + (i % 2),           # request_frequency_5m
                0,                     # failed_request_frequency_5m
                0,                     # post_request_frequency_5m
                None if i == 0 else 60.0 + i,  # time_since_previous_event_seconds
                1 + (i % 3),           # unique_resources_accessed
                0,                     # sensitive_resource_indicator
                0, 0, 0, 0, 0,
                1 if i == 0 else 0,    # time_since_previous_event_missing
            ]
        )
        labels.append(0)
        scenarios.append("santos")
    for i in range(2):
        rows.append(
            [
                5000 + i * 500,
                40 + i * 5,
                20 + i * 2,
                10 + i,
                1.5 + i * 0.5,
                30 + i * 5,
                1,
                0, 0, 0, 0, 0, 0,
            ]
        )
        labels.append(1)
        scenarios.append("santos")

    # Scenario B: 2 normal, 20 attack (higher volume, attack-heavy)
    for i in range(2):
        rows.append(
            [
                180 + i * 20,
                2,
                0,
                0,
                None if i == 0 else 45.0,
                2,
                0,
                0, 0, 0, 0, 0,
                1 if i == 0 else 0,
            ]
        )
        labels.append(0)
        scenarios.append("fox")
    for i in range(20):
        rows.append(
            [
                8000 + i * 200,
                60 + i * 3,
                35 + i * 2,
                15 + (i % 4),
                0.5 + (i % 3) * 0.2,
                50 + i,
                1,
                0, 0, 0, 0, 0, 0,
            ]
        )
        labels.append(1)
        scenarios.append("fox")
    return rows, labels, scenarios


def test_weighting_strategies_normalize_to_mean_one_without_double_weighting():
    _, labels, scenarios = _synthetic_dataset()
    n = len(labels)

    w_class = compute_sample_weights(labels, "class_weighted")
    assert math.isclose(float(w_class.mean()), 1.0, rel_tol=1e-9)
    assert math.isclose(float(w_class[np.array(labels) == 0].sum()), n / 2.0, rel_tol=1e-9)
    assert math.isclose(float(w_class[np.array(labels) == 1].sum()), n / 2.0, rel_tol=1e-9)

    w_scen = compute_sample_weights(
        labels, "scenario_equalized_class_weighted", scenario_ids=scenarios
    )
    assert math.isclose(float(w_scen.mean()), 1.0, rel_tol=1e-9)
    for scen in ("fox", "santos"):
        for cls in (0, 1):
            mask = (np.array(scenarios) == scen) & (np.array(labels) == cls)
            assert math.isclose(float(w_scen[mask].sum()), n / 4.0, rel_tol=1e-9)


def test_logistic_regression_preprocessing_preserves_binary_flags_and_is_reproducible():
    rows, labels, scenarios = _synthetic_dataset()

    first = train_logistic_regression_baseline(
        rows,
        labels,
        weighting_regime="scenario_equalized_class_weighted",
        scenario_ids=scenarios,
        C=1.0,
        random_state=42,
    )
    second = train_logistic_regression_baseline(
        rows,
        labels,
        weighting_regime="scenario_equalized_class_weighted",
        scenario_ids=scenarios,
        C=1.0,
        random_state=42,
    )

    assert first.pipeline.named_steps["classifier"].class_weight is None
    assert first.hyperparameters["class_weight"] is None

    # Verify binary flags pass through untransformed (still exact 0.0 / 1.0)
    transformed = first.pipeline.named_steps["preprocessor"].transform(
        np.asarray([[math.nan if v is None else float(v) for v in r] for r in rows])
    )
    assert transformed.shape == (len(rows), len(EXPECTED_FEATURE_NAMES))
    binary_transformed = transformed[:, len(LOG1P_FEATURE_NAMES) :]
    assert binary_transformed.shape[1] == len(BINARY_FEATURE_NAMES)
    assert np.isin(binary_transformed, (0.0, 1.0)).all()

    probs_1 = first.predict_proba(rows)
    probs_2 = second.predict_proba(rows)
    preds_1 = first.predict(rows, threshold=FIXED_CLASSIFICATION_THRESHOLD)
    preds_2 = second.predict(rows, threshold=FIXED_CLASSIFICATION_THRESHOLD)

    assert np.array_equal(probs_1, probs_2)
    assert np.array_equal(preds_1, preds_2)
    assert ((probs_1 >= 0.0) & (probs_1 <= 1.0)).all()
    assert set(preds_1.tolist()).issubset({0, 1})
    # Clearly separable synthetic data should achieve high accuracy
    assert (preds_1 == np.asarray(labels)).mean() >= 0.95


def test_records_to_arrays_and_fail_closed_validation():
    rows, labels, scenarios = _synthetic_dataset()
    model = train_logistic_regression_baseline(
        rows, labels, weighting_regime="class_weighted", C=0.1, random_state=42
    )

    rec = SupervisedRecord(
        scenario_id="fox",
        event_id="intranet/access.log:1",
        source_filename="intranet/access.log",
        source_line_number=1,
        source_sha256="a" * 64,
        original_evidence_reference="raw/fox/gather/intranet/access.log",
        label=0,
        label_name="normal",
        label_basis="absence_from_verified_ground_truth_file",
        label_provenance={"policy_id": "test"},
        attack_categories=(),
        features=tuple(rows[0]),
    )
    x_rec, y_rec, s_rec = records_to_arrays([rec])
    assert x_rec.shape == (1, 13)
    assert y_rec.tolist() == [0]
    assert s_rec == ("fox",)
    assert model.predict(x_rec).shape == (1,)

    # Wrong feature width
    with pytest.raises(BaselineModelError, match="expected 13"):
        model.predict_proba([[1, 2, 3]])

    # Negative count/size feature
    bad_neg = list(rows[0])
    bad_neg[0] = -5
    with pytest.raises(BaselineModelError, match="Negative values"):
        model.predict_proba([bad_neg])

    # Non-finite inf
    bad_inf = list(rows[0])
    bad_inf[4] = float("inf")
    with pytest.raises(BaselineModelError, match="Non-finite"):
        model.predict_proba([bad_inf])

    # Invalid binary indicator
    bad_bin = list(rows[0])
    bad_bin[6] = 2
    with pytest.raises(BaselineModelError, match="Binary indicator"):
        model.predict_proba([bad_bin])

    # Invalid label (boolean / non-binary / single-class)
    with pytest.raises(BaselineModelError, match="integer 0 or 1"):
        train_logistic_regression_baseline(rows[:2], [True, False])
    with pytest.raises(BaselineModelError, match="both normal"):
        train_logistic_regression_baseline(rows[:2], [0, 0])

    # Missing scenario_ids for scenario_equalized_class_weighted
    with pytest.raises(BaselineModelError, match="scenario_ids"):
        train_logistic_regression_baseline(
            rows, labels, weighting_regime="scenario_equalized_class_weighted"
        )

    # Unconfigured C or random_state
    with pytest.raises(BaselineModelError, match="hyperparameter_grid"):
        train_logistic_regression_baseline(rows, labels, C=99.0)
    with pytest.raises(BaselineModelError, match="random_seeds"):
        train_logistic_regression_baseline(rows, labels, random_state=7)
