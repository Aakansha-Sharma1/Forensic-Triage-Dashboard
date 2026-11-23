from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .training_data import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    EXPECTED_FEATURE_NAMES,
    SupervisedRecord,
)


FIXED_CLASSIFICATION_THRESHOLD = 0.5
WEIGHTING_REGIMES: tuple[str, ...] = (
    "class_weighted",
    "scenario_equalized_class_weighted",
)
LOG1P_FEATURE_NAMES: tuple[str, ...] = (
    "transfer_size_bytes",
    "request_frequency_5m",
    "failed_request_frequency_5m",
    "post_request_frequency_5m",
    "time_since_previous_event_seconds",
    "unique_resources_accessed",
)
BINARY_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name in EXPECTED_FEATURE_NAMES if name not in LOG1P_FEATURE_NAMES
)
_LOG1P_INDICES: tuple[int, ...] = tuple(
    EXPECTED_FEATURE_NAMES.index(name) for name in LOG1P_FEATURE_NAMES
)
_BINARY_INDICES: tuple[int, ...] = tuple(
    EXPECTED_FEATURE_NAMES.index(name) for name in BINARY_FEATURE_NAMES
)


class BaselineModelError(ValueError):
    """Raised when baseline model inputs, weights, or config settings are invalid."""


@dataclass(frozen=True)
class FittedLogisticRegressionBaseline:
    """Reproducible fitted Logistic Regression pipeline with fixed-threshold inference."""

    pipeline: Pipeline
    weighting_regime: str
    hyperparameters: Mapping[str, object]
    random_state: int
    feature_names: tuple[str, ...] = EXPECTED_FEATURE_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hyperparameters", MappingProxyType(dict(self.hyperparameters))
        )

    def predict_proba(self, features: Sequence[Sequence[Any]] | np.ndarray) -> np.ndarray:
        """Return positive-class (1 = attack_associated) probabilities."""
        matrix = _validate_feature_matrix(features)
        proba = self.pipeline.predict_proba(matrix)
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        if classes != [0, 1]:
            raise BaselineModelError(f"Unexpected classifier class ordering: {classes!r}")
        return np.asarray(proba[:, 1], dtype=np.float64)

    def predict(
        self,
        features: Sequence[Sequence[Any]] | np.ndarray,
        *,
        threshold: float = FIXED_CLASSIFICATION_THRESHOLD,
    ) -> np.ndarray:
        """Classify events using the specified probability threshold (default 0.5)."""
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not (0.0 <= float(threshold) <= 1.0)
        ):
            raise BaselineModelError(f"Invalid classification threshold: {threshold!r}")
        probs = self.predict_proba(features)
        return (probs >= float(threshold)).astype(np.int64)


def load_logistic_regression_config(
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read and validate LogisticRegression settings from experiment_config.json."""
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineModelError(f"Unable to read experiment config: {config_path}") from exc
    if not isinstance(config, dict):
        raise BaselineModelError("Experiment config must be a JSON object.")

    features_cfg = config.get("features", {})
    if features_cfg.get("ordered_names") != list(EXPECTED_FEATURE_NAMES):
        raise BaselineModelError("Experiment config feature names do not match FEATURE_SCHEMA v1.0.0.")
    if features_cfg.get("log1p_transform_features") != list(LOG1P_FEATURE_NAMES):
        raise BaselineModelError("Experiment config log1p_transform_features mismatch.")
    if features_cfg.get("binary_indicator_features") != list(BINARY_FEATURE_NAMES):
        raise BaselineModelError("Experiment config binary_indicator_features mismatch.")

    weighting_policy = config.get("weighting_policy", {})
    if (
        weighting_policy.get("mechanism") != "sample_weight_only"
        or weighting_policy.get("estimator_class_weight") is not None
    ):
        raise BaselineModelError("Experiment config must enforce sample_weight_only with class_weight=null.")

    models = config.get("models")
    if not isinstance(models, list):
        raise BaselineModelError("Experiment config is missing models list.")
    lr_cfg = next(
        (item for item in models if isinstance(item, dict) and item.get("name") == "LogisticRegression"),
        None,
    )
    if lr_cfg is None:
        raise BaselineModelError("LogisticRegression is not defined in experiment config.")
    if lr_cfg.get("fixed_parameters", {}).get("class_weight") is not None:
        raise BaselineModelError("LogisticRegression fixed_parameters.class_weight must be null.")

    grid = lr_cfg.get("hyperparameter_grid", {})
    penalties = grid.get("penalty")
    solvers = grid.get("solver")
    max_iters = grid.get("max_iter")
    c_values = grid.get("C")
    seeds = config.get("random_seeds")
    if (
        penalties != ["l2"]
        or not isinstance(solvers, list)
        or not solvers
        or not isinstance(max_iters, list)
        or not max_iters
        or not isinstance(c_values, list)
        or not c_values
        or not isinstance(seeds, list)
        or not seeds
    ):
        raise BaselineModelError("Incomplete LogisticRegression hyperparameter grid or random_seeds.")

    return {
        "C_values": [float(v) for v in c_values],
        "penalty": penalties[0],
        "solver": str(solvers[0]),
        "max_iter": int(max_iters[0]),
        "random_seeds": [int(s) for s in seeds],
        "weighting_regimes": tuple(
            r["name"] for r in weighting_policy.get("regimes", []) if isinstance(r, dict)
        ),
    }


def _validate_feature_matrix(
    features: Sequence[Sequence[Any]] | np.ndarray,
) -> np.ndarray:
    expected_cols = len(EXPECTED_FEATURE_NAMES)
    if isinstance(features, np.ndarray):
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != expected_cols:
            raise BaselineModelError(
                f"Expected non-empty 2D feature array with {expected_cols} columns."
            )
        if features.dtype == np.bool_:
            raise BaselineModelError("Boolean feature arrays are not permitted.")
        if features.dtype == object:
            rows = features.tolist()
            return _validate_feature_matrix(rows)
        try:
            matrix = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise BaselineModelError("Features must contain numeric values or None/NaN.") from exc
    else:
        if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or len(features) == 0:
            raise BaselineModelError("Features must be a non-empty 2D sequence.")
        parsed_rows: list[list[float]] = []
        for row_idx, row in enumerate(features):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != expected_cols:
                raise BaselineModelError(
                    f"Row {row_idx}: expected {expected_cols} ordered schema features."
                )
            parsed_row: list[float] = []
            for col_idx, value in enumerate(row):
                if value is None:
                    parsed_row.append(math.nan)
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                    raise BaselineModelError(
                        f"Row {row_idx}, col {col_idx}: non-numeric feature value {value!r}."
                    )
                parsed_row.append(float(value))
            parsed_rows.append(parsed_row)
        matrix = np.asarray(parsed_rows, dtype=np.float64)

    if np.isinf(matrix).any():
        raise BaselineModelError("Non-finite (+/-inf) feature values are not permitted.")

    continuous_block = matrix[:, _LOG1P_INDICES]
    finite_cont = continuous_block[~np.isnan(continuous_block)]
    if (finite_cont < 0.0).any():
        raise BaselineModelError("Negative values are disallowed for nonnegative count/size/time features.")

    binary_block = matrix[:, _BINARY_INDICES]
    finite_bin = binary_block[~np.isnan(binary_block)]
    if not np.isin(finite_bin, (0.0, 1.0)).all():
        raise BaselineModelError("Binary indicator features must contain only 0, 1, or missing values.")

    return matrix


def _validate_labels(
    labels: Sequence[Any] | np.ndarray,
    expected_length: int,
    *,
    require_both_classes: bool = True,
) -> np.ndarray:
    if isinstance(labels, np.ndarray):
        if labels.ndim != 1 or len(labels) != expected_length:
            raise BaselineModelError(f"Expected 1D labels of length {expected_length}.")
        if labels.dtype == np.bool_:
            raise BaselineModelError("Boolean labels are not permitted; use integer 0 or 1.")
        raw_labels = labels.tolist()
    else:
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or len(labels) != expected_length:
            raise BaselineModelError(f"Expected 1D label sequence of length {expected_length}.")
        raw_labels = list(labels)

    validated: list[int] = []
    for idx, value in enumerate(raw_labels):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in (0, 1):
            raise BaselineModelError(f"Label at index {idx} must be integer 0 or 1, got {value!r}.")
        validated.append(int(value))

    array = np.asarray(validated, dtype=np.int64)
    if require_both_classes and set(array.tolist()) != {0, 1}:
        raise BaselineModelError("Training labels must contain both normal (0) and attack_associated (1).")
    return array


def compute_sample_weights(
    labels: Sequence[Any] | np.ndarray,
    weighting_regime: str,
    *,
    scenario_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute train-only sample weights normalized to sum to n_train (mean 1.0)."""
    y = _validate_labels(labels, len(labels), require_both_classes=True)
    n_train = len(y)

    if weighting_regime == "class_weighted":
        counts = Counter(y.tolist())
        weights = np.array(
            [n_train / (2.0 * counts[int(label)]) for label in y],
            dtype=np.float64,
        )
        return weights

    if weighting_regime == "scenario_equalized_class_weighted":
        if (
            scenario_ids is None
            or isinstance(scenario_ids, (str, bytes))
            or len(scenario_ids) != n_train
        ):
            raise BaselineModelError(
                "scenario_ids of length n_train is required for scenario_equalized_class_weighted."
            )
        normalized_scenarios: list[str] = []
        for idx, scenario in enumerate(scenario_ids):
            if not isinstance(scenario, str) or not scenario.strip():
                raise BaselineModelError(f"Invalid scenario_id at index {idx}: {scenario!r}")
            normalized_scenarios.append(scenario.strip())

        unique_scenarios = sorted(set(normalized_scenarios))
        n_scenarios = len(unique_scenarios)
        pair_counts = Counter(zip(normalized_scenarios, y.tolist()))
        for scenario in unique_scenarios:
            if pair_counts[(scenario, 0)] == 0 or pair_counts[(scenario, 1)] == 0:
                raise BaselineModelError(
                    f"Scenario {scenario!r} must contain both classes (0 and 1) for scenario-equalized weighting."
                )
        weights = np.array(
            [
                n_train / (2.0 * n_scenarios * pair_counts[(scenario, int(label))])
                for scenario, label in zip(normalized_scenarios, y)
            ],
            dtype=np.float64,
        )
        return weights

    raise BaselineModelError(f"Unsupported weighting_regime: {weighting_regime!r}")


def build_logistic_regression_pipeline(
    *,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Construct the train-only preprocessing + L2 LogisticRegression pipeline."""
    continuous_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipe, list(_LOG1P_INDICES)),
            ("binary", binary_pipe, list(_BINARY_INDICES)),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=float(C),
        solver=solver,
        max_iter=int(max_iter),
        class_weight=None,
        random_state=int(random_state),
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def records_to_arrays(
    records: Sequence[SupervisedRecord],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Convert Verified SupervisedRecords into (X, y, scenario_ids) without leaking metadata."""
    if not records:
        raise BaselineModelError("At least one SupervisedRecord is required.")
    x = _validate_feature_matrix([record.features for record in records])
    y = _validate_labels([record.label for record in records], len(records), require_both_classes=False)
    scenarios = tuple(record.scenario_id for record in records)
    return x, y, scenarios


def train_logistic_regression_baseline(
    features: Sequence[Sequence[Any]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    *,
    weighting_regime: str = "class_weighted",
    scenario_ids: Sequence[str] | None = None,
    C: float | None = None,
    random_state: int | None = None,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> FittedLogisticRegressionBaseline:
    """Fit a reproducible L2 Logistic Regression baseline strictly on training data."""
    lr_cfg = load_logistic_regression_config(config_path)
    if weighting_regime not in lr_cfg["weighting_regimes"]:
        raise BaselineModelError(
            f"weighting_regime {weighting_regime!r} is not configured in experiment_config.json."
        )

    selected_c = 1.0 if C is None else float(C)
    if selected_c not in lr_cfg["C_values"]:
        raise BaselineModelError(
            f"C={selected_c!r} is not in configured hyperparameter_grid {lr_cfg['C_values']!r}."
        )
    selected_seed = lr_cfg["random_seeds"][0] if random_state is None else int(random_state)
    if selected_seed not in lr_cfg["random_seeds"]:
        raise BaselineModelError(
            f"random_state={selected_seed!r} is not in configured random_seeds {lr_cfg['random_seeds']!r}."
        )

    x_train = _validate_feature_matrix(features)
    y_train = _validate_labels(labels, len(x_train), require_both_classes=True)
    sample_weight = compute_sample_weights(
        y_train,
        weighting_regime,
        scenario_ids=scenario_ids,
    )

    pipeline = build_logistic_regression_pipeline(
        C=selected_c,
        solver=lr_cfg["solver"],
        max_iter=lr_cfg["max_iter"],
        random_state=selected_seed,
    )
    if pipeline.named_steps["classifier"].class_weight is not None:
        raise BaselineModelError("Estimator class_weight must remain None to prevent double-weighting.")

    pipeline.fit(x_train, y_train, classifier__sample_weight=sample_weight)
    return FittedLogisticRegressionBaseline(
        pipeline=pipeline,
        weighting_regime=weighting_regime,
        hyperparameters={
            "C": selected_c,
            "penalty": lr_cfg["penalty"],
            "solver": lr_cfg["solver"],
            "max_iter": lr_cfg["max_iter"],
            "class_weight": None,
        },
        random_state=selected_seed,
    )


@dataclass(frozen=True)
class FittedDecisionTreeBaseline:
    """Reproducible fitted Decision Tree pipeline with thresholded inference."""

    pipeline: Pipeline
    weighting_regime: str
    hyperparameters: Mapping[str, object]
    random_state: int
    feature_names: tuple[str, ...] = EXPECTED_FEATURE_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hyperparameters", MappingProxyType(dict(self.hyperparameters))
        )

    def predict_proba(self, features: Sequence[Sequence[Any]] | np.ndarray) -> np.ndarray:
        """Return positive-class (1 = attack_associated) probabilities."""
        matrix = _validate_feature_matrix(features)
        proba = self.pipeline.predict_proba(matrix)
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        if classes != [0, 1]:
            raise BaselineModelError(f"Unexpected classifier class ordering: {classes!r}")
        return np.asarray(proba[:, 1], dtype=np.float64)

    def predict(
        self,
        features: Sequence[Sequence[Any]] | np.ndarray,
        *,
        threshold: float = FIXED_CLASSIFICATION_THRESHOLD,
    ) -> np.ndarray:
        """Classify events using the specified probability threshold (default 0.5)."""
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not (0.0 <= float(threshold) <= 1.0)
        ):
            raise BaselineModelError(f"Invalid classification threshold: {threshold!r}")
        probs = self.predict_proba(features)
        return (probs >= float(threshold)).astype(np.int64)


def load_decision_tree_config(
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read and validate DecisionTreeClassifier settings from experiment_config.json."""
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineModelError(f"Unable to read experiment config: {config_path}") from exc
    if not isinstance(config, dict):
        raise BaselineModelError("Experiment config must be a JSON object.")

    features_cfg = config.get("features", {})
    if features_cfg.get("ordered_names") != list(EXPECTED_FEATURE_NAMES):
        raise BaselineModelError("Experiment config feature names do not match FEATURE_SCHEMA v1.0.0.")

    tree_preproc = config.get("preprocessing", {}).get("tree_models", {})
    if tree_preproc.get("apply_log1p") is not False or tree_preproc.get("scaler") is not None:
        raise BaselineModelError("Tree preprocessing policy must disable log1p and scaling.")

    weighting_policy = config.get("weighting_policy", {})
    if (
        weighting_policy.get("mechanism") != "sample_weight_only"
        or weighting_policy.get("estimator_class_weight") is not None
    ):
        raise BaselineModelError("Experiment config must enforce sample_weight_only with class_weight=null.")

    models = config.get("models")
    if not isinstance(models, list):
        raise BaselineModelError("Experiment config is missing models list.")
    dt_cfg = next(
        (
            item
            for item in models
            if isinstance(item, dict) and item.get("name") == "DecisionTreeClassifier"
        ),
        None,
    )
    if dt_cfg is None:
        raise BaselineModelError("DecisionTreeClassifier is not defined in experiment config.")
    fixed = dt_cfg.get("fixed_parameters", {})
    if fixed.get("class_weight") is not None or fixed.get("criterion") != "gini":
        raise BaselineModelError(
            "DecisionTreeClassifier must pin class_weight=null and criterion='gini'."
        )

    grid = dt_cfg.get("hyperparameter_grid", {})
    max_depths = grid.get("max_depth")
    min_leaves = grid.get("min_samples_leaf")
    seeds = config.get("random_seeds")
    if (
        not isinstance(max_depths, list)
        or not max_depths
        or not isinstance(min_leaves, list)
        or not min_leaves
        or not isinstance(seeds, list)
        or not seeds
    ):
        raise BaselineModelError("Incomplete DecisionTreeClassifier hyperparameter grid or random_seeds.")

    return {
        "criterion": "gini",
        "max_depth_values": [int(v) for v in max_depths],
        "min_samples_leaf_values": [int(v) for v in min_leaves],
        "random_seeds": [int(s) for s in seeds],
        "weighting_regimes": tuple(
            r["name"] for r in weighting_policy.get("regimes", []) if isinstance(r, dict)
        ),
    }


def build_decision_tree_pipeline(
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 10,
    criterion: str = "gini",
    random_state: int = 42,
) -> Pipeline:
    """Construct train-only median imputation + DecisionTreeClassifier without scaling or log1p."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=int(max_depth),
                    min_samples_leaf=int(min_samples_leaf),
                    class_weight=None,
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def train_decision_tree_baseline(
    features: Sequence[Sequence[Any]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    *,
    weighting_regime: str = "class_weighted",
    scenario_ids: Sequence[str] | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    random_state: int | None = None,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> FittedDecisionTreeBaseline:
    """Fit a reproducible DecisionTreeClassifier baseline strictly on training data."""
    dt_cfg = load_decision_tree_config(config_path)
    if weighting_regime not in dt_cfg["weighting_regimes"]:
        raise BaselineModelError(
            f"weighting_regime {weighting_regime!r} is not configured in experiment_config.json."
        )

    selected_depth = dt_cfg["max_depth_values"][0] if max_depth is None else int(max_depth)
    if isinstance(max_depth, bool) or selected_depth not in dt_cfg["max_depth_values"]:
        raise BaselineModelError(
            f"max_depth={max_depth!r} is not in configured hyperparameter_grid {dt_cfg['max_depth_values']!r}."
        )
    selected_leaf = (
        dt_cfg["min_samples_leaf_values"][0]
        if min_samples_leaf is None
        else int(min_samples_leaf)
    )
    if isinstance(min_samples_leaf, bool) or selected_leaf not in dt_cfg["min_samples_leaf_values"]:
        raise BaselineModelError(
            f"min_samples_leaf={min_samples_leaf!r} is not in configured hyperparameter_grid {dt_cfg['min_samples_leaf_values']!r}."
        )
    selected_seed = dt_cfg["random_seeds"][0] if random_state is None else int(random_state)
    if isinstance(random_state, bool) or selected_seed not in dt_cfg["random_seeds"]:
        raise BaselineModelError(
            f"random_state={random_state!r} is not in configured random_seeds {dt_cfg['random_seeds']!r}."
        )

    x_train = _validate_feature_matrix(features)
    y_train = _validate_labels(labels, len(x_train), require_both_classes=True)
    sample_weight = compute_sample_weights(
        y_train,
        weighting_regime,
        scenario_ids=scenario_ids,
    )

    pipeline = build_decision_tree_pipeline(
        max_depth=selected_depth,
        min_samples_leaf=selected_leaf,
        criterion=dt_cfg["criterion"],
        random_state=selected_seed,
    )
    if pipeline.named_steps["classifier"].class_weight is not None:
        raise BaselineModelError("Estimator class_weight must remain None to prevent double-weighting.")

    pipeline.fit(x_train, y_train, classifier__sample_weight=sample_weight)
    return FittedDecisionTreeBaseline(
        pipeline=pipeline,
        weighting_regime=weighting_regime,
        hyperparameters={
            "criterion": dt_cfg["criterion"],
            "max_depth": selected_depth,
            "min_samples_leaf": selected_leaf,
            "class_weight": None,
        },
        random_state=selected_seed,
    )


@dataclass(frozen=True)
class FittedRandomForestBaseline:
    """Reproducible fitted Random Forest pipeline with thresholded inference."""

    pipeline: Pipeline
    weighting_regime: str
    hyperparameters: Mapping[str, object]
    random_state: int
    feature_names: tuple[str, ...] = EXPECTED_FEATURE_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hyperparameters", MappingProxyType(dict(self.hyperparameters))
        )

    def predict_proba(self, features: Sequence[Sequence[Any]] | np.ndarray) -> np.ndarray:
        """Return positive-class (1 = attack_associated) probabilities."""
        matrix = _validate_feature_matrix(features)
        proba = self.pipeline.predict_proba(matrix)
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        if classes != [0, 1]:
            raise BaselineModelError(f"Unexpected classifier class ordering: {classes!r}")
        return np.asarray(proba[:, 1], dtype=np.float64)

    def predict(
        self,
        features: Sequence[Sequence[Any]] | np.ndarray,
        *,
        threshold: float = FIXED_CLASSIFICATION_THRESHOLD,
    ) -> np.ndarray:
        """Classify events using the specified probability threshold (default 0.5)."""
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not (0.0 <= float(threshold) <= 1.0)
        ):
            raise BaselineModelError(f"Invalid classification threshold: {threshold!r}")
        probs = self.predict_proba(features)
        return (probs >= float(threshold)).astype(np.int64)


def load_random_forest_config(
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read and validate RandomForestClassifier settings from experiment_config.json."""
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineModelError(f"Unable to read experiment config: {config_path}") from exc
    if not isinstance(config, dict):
        raise BaselineModelError("Experiment config must be a JSON object.")

    features_cfg = config.get("features", {})
    if features_cfg.get("ordered_names") != list(EXPECTED_FEATURE_NAMES):
        raise BaselineModelError("Experiment config feature names do not match FEATURE_SCHEMA v1.0.0.")

    tree_preproc = config.get("preprocessing", {}).get("tree_models", {})
    if tree_preproc.get("apply_log1p") is not False or tree_preproc.get("scaler") is not None:
        raise BaselineModelError("Tree preprocessing policy must disable log1p and scaling.")

    weighting_policy = config.get("weighting_policy", {})
    if (
        weighting_policy.get("mechanism") != "sample_weight_only"
        or weighting_policy.get("estimator_class_weight") is not None
    ):
        raise BaselineModelError("Experiment config must enforce sample_weight_only with class_weight=null.")

    models = config.get("models")
    if not isinstance(models, list):
        raise BaselineModelError("Experiment config is missing models list.")
    rf_cfg = next(
        (
            item
            for item in models
            if isinstance(item, dict) and item.get("name") == "RandomForestClassifier"
        ),
        None,
    )
    if rf_cfg is None:
        raise BaselineModelError("RandomForestClassifier is not defined in experiment config.")
    fixed = rf_cfg.get("fixed_parameters", {})
    if (
        fixed.get("class_weight") is not None
        or fixed.get("criterion") != "gini"
        or fixed.get("bootstrap") is not True
    ):
        raise BaselineModelError(
            "RandomForestClassifier must pin class_weight=null, criterion='gini', and bootstrap=true."
        )

    grid = rf_cfg.get("hyperparameter_grid", {})
    n_estimators = grid.get("n_estimators")
    max_depths = grid.get("max_depth")
    min_leaves = grid.get("min_samples_leaf")
    max_features = grid.get("max_features")
    seeds = config.get("random_seeds")
    if (
        not isinstance(n_estimators, list)
        or not n_estimators
        or not isinstance(max_depths, list)
        or not max_depths
        or not isinstance(min_leaves, list)
        or not min_leaves
        or not isinstance(max_features, list)
        or not max_features
        or not isinstance(seeds, list)
        or not seeds
    ):
        raise BaselineModelError("Incomplete RandomForestClassifier hyperparameter grid or random_seeds.")

    return {
        "criterion": "gini",
        "bootstrap": True,
        "n_estimators_values": [int(v) for v in n_estimators],
        "max_depth_values": [int(v) for v in max_depths],
        "min_samples_leaf_values": [int(v) for v in min_leaves],
        "max_features_values": [str(v) for v in max_features],
        "random_seeds": [int(s) for s in seeds],
        "weighting_regimes": tuple(
            r["name"] for r in weighting_policy.get("regimes", []) if isinstance(r, dict)
        ),
    }


def build_random_forest_pipeline(
    *,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    max_features: str = "sqrt",
    criterion: str = "gini",
    bootstrap: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """Construct train-only median imputation + RandomForestClassifier without scaling or log1p."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    criterion=criterion,
                    max_depth=int(max_depth),
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=max_features,
                    bootstrap=bool(bootstrap),
                    class_weight=None,
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def train_random_forest_baseline(
    features: Sequence[Sequence[Any]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    *,
    weighting_regime: str = "class_weighted",
    scenario_ids: Sequence[str] | None = None,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    max_features: str | None = None,
    random_state: int | None = None,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> FittedRandomForestBaseline:
    """Fit a reproducible RandomForestClassifier baseline strictly on training data."""
    rf_cfg = load_random_forest_config(config_path)
    if weighting_regime not in rf_cfg["weighting_regimes"]:
        raise BaselineModelError(
            f"weighting_regime {weighting_regime!r} is not configured in experiment_config.json."
        )

    selected_n_est = (
        rf_cfg["n_estimators_values"][0] if n_estimators is None else int(n_estimators)
    )
    if isinstance(n_estimators, bool) or selected_n_est not in rf_cfg["n_estimators_values"]:
        raise BaselineModelError(
            f"n_estimators={n_estimators!r} is not in configured hyperparameter_grid {rf_cfg['n_estimators_values']!r}."
        )
    selected_depth = rf_cfg["max_depth_values"][0] if max_depth is None else int(max_depth)
    if isinstance(max_depth, bool) or selected_depth not in rf_cfg["max_depth_values"]:
        raise BaselineModelError(
            f"max_depth={max_depth!r} is not in configured hyperparameter_grid {rf_cfg['max_depth_values']!r}."
        )
    selected_leaf = (
        rf_cfg["min_samples_leaf_values"][0]
        if min_samples_leaf is None
        else int(min_samples_leaf)
    )
    if isinstance(min_samples_leaf, bool) or selected_leaf not in rf_cfg["min_samples_leaf_values"]:
        raise BaselineModelError(
            f"min_samples_leaf={min_samples_leaf!r} is not in configured hyperparameter_grid {rf_cfg['min_samples_leaf_values']!r}."
        )
    selected_max_feat = (
        rf_cfg["max_features_values"][0] if max_features is None else str(max_features)
    )
    if selected_max_feat not in rf_cfg["max_features_values"]:
        raise BaselineModelError(
            f"max_features={max_features!r} is not in configured hyperparameter_grid {rf_cfg['max_features_values']!r}."
        )
    selected_seed = rf_cfg["random_seeds"][0] if random_state is None else int(random_state)
    if isinstance(random_state, bool) or selected_seed not in rf_cfg["random_seeds"]:
        raise BaselineModelError(
            f"random_state={random_state!r} is not in configured random_seeds {rf_cfg['random_seeds']!r}."
        )

    x_train = _validate_feature_matrix(features)
    y_train = _validate_labels(labels, len(x_train), require_both_classes=True)
    sample_weight = compute_sample_weights(
        y_train,
        weighting_regime,
        scenario_ids=scenario_ids,
    )

    pipeline = build_random_forest_pipeline(
        n_estimators=selected_n_est,
        max_depth=selected_depth,
        min_samples_leaf=selected_leaf,
        max_features=selected_max_feat,
        criterion=rf_cfg["criterion"],
        bootstrap=rf_cfg["bootstrap"],
        random_state=selected_seed,
    )
    if pipeline.named_steps["classifier"].class_weight is not None:
        raise BaselineModelError("Estimator class_weight must remain None to prevent double-weighting.")

    pipeline.fit(x_train, y_train, classifier__sample_weight=sample_weight)
    return FittedRandomForestBaseline(
        pipeline=pipeline,
        weighting_regime=weighting_regime,
        hyperparameters={
            "n_estimators": selected_n_est,
            "criterion": rf_cfg["criterion"],
            "bootstrap": rf_cfg["bootstrap"],
            "max_depth": selected_depth,
            "min_samples_leaf": selected_leaf,
            "max_features": selected_max_feat,
            "class_weight": None,
        },
        random_state=selected_seed,
    )
