from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .baselines import (
    BaselineModelError,
    FittedRandomForestBaseline,
    train_random_forest_baseline,
)
from .benchmark import (
    CHECKPOINT_SCHEMA_VERSION,
    CompactScenarioDataset,
    DEFAULT_CHECKPOINT_DIR,
    BenchmarkError,
    load_verified_scenario_dataset,
)
from .experiment_runner import (
    ExperimentRunnerError,
    build_nested_loso_plan,
)
from .model_artifact import (
    DEFAULT_TRUSTED_MODELS_DIR,
    InferencePrediction,
    LoadedModelArtifact,
    ModelArtifactError,
    load_model_artifact,
    save_model_artifact,
)
from .training_data import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    PROJECT_ROOT,
    TrainingDataError,
    load_and_verify_experiment_config,
)


DEPLOYMENT_POLICY_ID = "inner_validation_checkpoint_aggregation_v1"
DEFAULT_DEPLOYMENT_MODEL_ID = "forensicinsight-rf-ait-lds-v2.1"
DEFAULT_DEPLOYMENT_MODEL_VERSION = "1.0.0"
DEFAULT_DEPLOYMENT_WEIGHTING_REGIME = "class_weighted"
DEFAULT_DEPLOYMENT_MODEL_FAMILY = "RandomForestClassifier"


class DeploymentTrainingError(ValueError):
    """Raised when deployment policy selection, training, or artifact verification fails."""


@dataclass(frozen=True)
class DeploymentPolicySelection:
    """Deterministic deployment hyperparameters and threshold chosen strictly from inner validation."""

    policy_id: str
    model_family: str
    weighting_regime: str
    selected_hyperparameters: Mapping[str, Any]
    selected_threshold: float
    deployment_random_state: int
    inner_validation_mean_macro_f1: float
    inner_validation_observation_count: int
    inner_checkpoint_count: int
    inner_validation_folds_evaluated: tuple[str, ...]
    tie_breaking_policy: Mapping[str, Any]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "model_family": self.model_family,
            "weighting_regime": self.weighting_regime,
            "selected_hyperparameters": dict(self.selected_hyperparameters),
            "selected_threshold": self.selected_threshold,
            "deployment_random_state": self.deployment_random_state,
            "inner_validation_mean_macro_f1": self.inner_validation_mean_macro_f1,
            "inner_validation_observation_count": self.inner_validation_observation_count,
            "inner_checkpoint_count": self.inner_checkpoint_count,
            "inner_validation_folds_evaluated": list(self.inner_validation_folds_evaluated),
            "tie_breaking_policy": dict(self.tie_breaking_policy),
            "rationale": self.rationale,
            "outer_test_used_for_selection": False,
        }


@dataclass(frozen=True)
class DeploymentTrainingResult:
    """Verified output of one reproducible deployment-training execution."""

    model_id: str
    model_version: str
    artifact_dir: Path
    manifest_path: Path
    model_file_path: Path
    model_file_sha256: str
    model_file_size_bytes: int
    policy_selection: DeploymentPolicySelection
    total_training_records: int
    normal_records: int
    attack_associated_records: int
    excluded_unknown_records: int
    scenario_record_counts: Mapping[str, Mapping[str, int]]
    verification_predictions_match: bool
    verification_predictions: tuple[InferencePrediction, ...]
    manifest: Mapping[str, object]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "artifact_dir": str(self.artifact_dir),
            "manifest_path": str(self.manifest_path),
            "model_file_path": str(self.model_file_path),
            "model_file_sha256": self.model_file_sha256,
            "model_file_size_bytes": self.model_file_size_bytes,
            "policy_selection": self.policy_selection.to_dict(),
            "training_record_counts": {
                "total_training_records": self.total_training_records,
                "normal_records": self.normal_records,
                "attack_associated_records": self.attack_associated_records,
                "excluded_unknown_records": self.excluded_unknown_records,
                "per_scenario": {k: dict(v) for k, v in self.scenario_record_counts.items()},
            },
            "verification_predictions_match": self.verification_predictions_match,
            "verification_predictions": [p.to_dict() for p in self.verification_predictions],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _expected_checkpoint_provenance(
    config: Mapping[str, object],
    config_sha256: str,
) -> dict[str, object]:
    dataset_prov = config.get("dataset_provenance")
    if not isinstance(dataset_prov, Mapping):
        raise DeploymentTrainingError("Experiment config is missing dataset_provenance.")
    scenarios = dataset_prov.get("scenarios")
    split_manifest = dataset_prov.get("split_manifest")
    quality_report = dataset_prov.get("quality_report")
    if (
        not isinstance(scenarios, Mapping)
        or not isinstance(split_manifest, Mapping)
        or not isinstance(quality_report, Mapping)
    ):
        raise DeploymentTrainingError("Incomplete dataset_provenance in experiment config.")
    return {
        "experiment_id": str(config["experiment_id"]),
        "experiment_config_sha256": config_sha256.lower(),
        "split_manifest_sha256": str(split_manifest["sha256"]),
        "quality_report_sha256": str(quality_report["sha256"]),
        "scenario_labelled_sha256": {
            str(name): str(ref["labelled_records_sha256"])
            for name, ref in sorted(scenarios.items())
        },
    }


def select_deployment_configuration_from_inner_checkpoints(
    config: Mapping[str, object],
    *,
    config_sha256: str,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    base_dir: str | Path | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    model_name: str = DEFAULT_DEPLOYMENT_MODEL_FAMILY,
    weighting_regime: str = DEFAULT_DEPLOYMENT_WEIGHTING_REGIME,
) -> DeploymentPolicySelection:
    """Select deployment hyperparameters and threshold strictly from verified inner-validation checkpoints.

    Never reads outer-test checkpoints, outer-test metrics, or outer-test predictions.
    """
    if model_name != DEFAULT_DEPLOYMENT_MODEL_FAMILY:
        raise DeploymentTrainingError(
            f"Deployment policy currently supports {DEFAULT_DEPLOYMENT_MODEL_FAMILY!r}, got {model_name!r}."
        )
    if weighting_regime != DEFAULT_DEPLOYMENT_WEIGHTING_REGIME:
        raise DeploymentTrainingError(
            f"Deployment policy currently supports {DEFAULT_DEPLOYMENT_WEIGHTING_REGIME!r}, got {weighting_regime!r}."
        )

    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.is_dir():
        raise DeploymentTrainingError(f"Inner-validation checkpoint directory not found: {ckpt_root}")

    try:
        plan = build_nested_loso_plan(config_path, base_dir=base_dir)
    except ExperimentRunnerError as exc:
        raise DeploymentTrainingError(f"Failed to build nested LOSO plan: {exc}") from exc

    rf_spec = next((spec for spec in plan.model_specs if spec.name == model_name), None)
    if rf_spec is None:
        raise DeploymentTrainingError(f"Model {model_name!r} not found in experiment config.")

    expected_provenance = _expected_checkpoint_provenance(config, config_sha256)
    candidate_thresholds = tuple(float(t) for t in plan.candidate_thresholds)
    seeds = tuple(int(s) for s in plan.random_seeds)
    param_grid = tuple(dict(hp) for hp in rf_spec.parameter_combinations)
    grid_index_by_key = {
        json.dumps(hp, sort_keys=True): idx for idx, hp in enumerate(param_grid)
    }

    grouped_scores: dict[tuple[str, float], list[float]] = defaultdict(list)
    checkpoint_count = 0
    folds_evaluated: list[str] = []

    for fold in plan.outer_folds:
        folds_evaluated.append(fold.fold_id)
        for inner in fold.inner_candidates:
            for param_idx, expected_hp in enumerate(param_grid):
                for seed in seeds:
                    unit_key = (
                        f"inner__{model_name}__{weighting_regime}__{fold.fold_id}__"
                        f"{inner.candidate_id}__p{param_idx}__s{seed}"
                    )
                    ckpt_path = ckpt_root / f"{unit_key}.json"
                    if not ckpt_path.is_file():
                        raise DeploymentTrainingError(
                            f"Missing required inner-validation checkpoint: {ckpt_path.name}"
                        )
                    try:
                        record = json.loads(ckpt_path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise DeploymentTrainingError(
                            f"Corrupted inner-validation checkpoint: {ckpt_path.name}"
                        ) from exc

                    if not isinstance(record, dict):
                        raise DeploymentTrainingError(f"Invalid checkpoint object: {ckpt_path.name}")
                    if record.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
                        raise DeploymentTrainingError(
                            f"Unsupported checkpoint_schema_version in {ckpt_path.name}."
                        )
                    expected_unit_key = {
                        "candidate_id": inner.candidate_id,
                        "candidate_thresholds": list(candidate_thresholds),
                        "fold_id": fold.fold_id,
                        "hyperparameters": dict(expected_hp),
                        "model_name": model_name,
                        "param_index": param_idx,
                        "seed": seed,
                        "validation_scenario": inner.validation_scenarios[0],
                        "weighting_regime": weighting_regime,
                    }
                    if record.get("unit_type") != "inner_fit" or record.get("unit_key") != expected_unit_key:
                        raise DeploymentTrainingError(
                            f"Non-inner or mismatched unit checkpoint rejected: {ckpt_path.name}"
                        )
                    if record.get("provenance") != expected_provenance:
                        raise DeploymentTrainingError(
                            f"Checkpoint provenance mismatch in {ckpt_path.name}."
                        )
                    payload = record.get("payload")
                    if (
                        not isinstance(payload, dict)
                        or _canonical_json_sha256(payload) != record.get("payload_sha256")
                    ):
                        raise DeploymentTrainingError(
                            f"Checkpoint payload SHA-256 mismatch in {ckpt_path.name}."
                        )
                    observations = payload.get("observations")
                    if not isinstance(observations, list) or len(observations) != len(candidate_thresholds):
                        raise DeploymentTrainingError(
                            f"Unexpected observation count in {ckpt_path.name}."
                        )

                    seen_thresholds: set[float] = set()
                    for obs in observations:
                        if not isinstance(obs, dict):
                            raise DeploymentTrainingError(f"Invalid observation entry in {ckpt_path.name}.")
                        if obs.get("partition_role") != "validation":
                            raise DeploymentTrainingError(
                                f"Rejected non-validation observation ({obs.get('partition_role')!r}) in {ckpt_path.name}."
                            )
                        if (
                            obs.get("model_name") != model_name
                            or obs.get("weighting_regime") != weighting_regime
                            or obs.get("fold_id") != fold.fold_id
                            or obs.get("candidate_id") != inner.candidate_id
                            or obs.get("seed") != seed
                            or obs.get("hyperparameters") != expected_hp
                        ):
                            raise DeploymentTrainingError(
                                f"Observation metadata mismatch in {ckpt_path.name}."
                            )
                        thr = float(obs["threshold"])
                        if thr not in candidate_thresholds or thr in seen_thresholds:
                            raise DeploymentTrainingError(
                                f"Invalid or duplicate threshold {thr!r} in {ckpt_path.name}."
                            )
                        score = float(obs["macro_f1"])
                        if not math.isfinite(score) or not (0.0 <= score <= 1.0):
                            raise DeploymentTrainingError(
                                f"Non-finite or out-of-range macro_f1 {score!r} in {ckpt_path.name}."
                            )
                        seen_thresholds.add(thr)
                        hp_key = json.dumps(expected_hp, sort_keys=True)
                        grouped_scores[(hp_key, thr)].append(score)

                    checkpoint_count += 1

    expected_obs_per_pair = (
        len(plan.outer_folds) * len(plan.outer_folds[0].inner_candidates) * len(seeds)
    )
    if not grouped_scores:
        raise DeploymentTrainingError("No inner-validation observations were collected.")

    best_rank_key: tuple[float, float, float, int] | None = None
    best_hp: dict[str, Any] | None = None
    best_threshold: float | None = None
    best_mean_f1: float | None = None
    best_obs_count = 0

    for (hp_key, thr), scores in grouped_scores.items():
        if len(scores) != expected_obs_per_pair:
            raise DeploymentTrainingError(
                f"Incomplete inner-validation support for {(hp_key, thr)}: "
                f"expected {expected_obs_per_pair}, got {len(scores)}."
            )
        mean_f1 = statistics.fmean(scores)
        grid_idx = grid_index_by_key[hp_key]
        # Exact tie-breaking from experiment_config.json:
        # 1. highest mean inner-validation macro_f1
        # 2. threshold closest to 0.5
        # 3. smaller threshold
        # 4. configured hyperparameter grid order (smaller index preferred)
        rank_key = (mean_f1, -abs(thr - 0.5), -thr, -grid_idx)
        if best_rank_key is None or rank_key > best_rank_key:
            best_rank_key = rank_key
            best_hp = dict(param_grid[grid_idx])
            best_threshold = thr
            best_mean_f1 = mean_f1
            best_obs_count = len(scores)

    assert best_hp is not None and best_threshold is not None and best_mean_f1 is not None

    deployment_seed = int(seeds[0])
    rationale = (
        "Selected the (hyperparameters, decision_threshold) pair maximizing mean "
        "inner-validation Macro-F1 across all 12 inner LOSO validation splits and all 5 "
        "configured seeds (60 inner-validation observations per candidate), using the "
        "experiment_config.json tie-breaking order (closest_to_0.5 -> smaller_threshold -> "
        "configured_grid_order) without reading any outer-test metrics or predictions. "
        f"Pinned deployment training random_state to the first configured seed ({deployment_seed}) "
        "without seed cherry-picking."
    )

    return DeploymentPolicySelection(
        policy_id=DEPLOYMENT_POLICY_ID,
        model_family=model_name,
        weighting_regime=weighting_regime,
        selected_hyperparameters=best_hp,
        selected_threshold=best_threshold,
        deployment_random_state=deployment_seed,
        inner_validation_mean_macro_f1=best_mean_f1,
        inner_validation_observation_count=best_obs_count,
        inner_checkpoint_count=checkpoint_count,
        inner_validation_folds_evaluated=tuple(folds_evaluated),
        tie_breaking_policy={
            "threshold_priority": ["closest_to_0.5", "smaller_threshold"],
            "hyperparameter_priority": "configured_grid_order",
        },
        rationale=rationale,
    )


def build_synthetic_verification_fixture() -> np.ndarray:
    """Deterministic 4-row synthetic 13-feature matrix covering normal-like, scan-like, and missing patterns."""
    return np.asarray(
        [
            # transfer_size, req_5m, fail_5m, post_5m, dt_prev, uniq_res, sens, 6x missing flags
            [512.0, 2.0, 0.0, 0.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [240.0, 450.0, 440.0, 0.0, 0.01, 420.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8192.0, 18.0, 4.0, 12.0, 1.2, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [math.nan, 3.0, 1.0, 0.0, math.nan, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def _predict_from_fitted(
    fitted: FittedRandomForestBaseline,
    features: np.ndarray,
    *,
    model_id: str,
    model_version: str,
    decision_threshold: float,
) -> tuple[InferencePrediction, ...]:
    scores = fitted.predict_proba(features)
    preds: list[InferencePrediction] = []
    for score_val in scores.tolist():
        label = 1 if score_val >= decision_threshold else 0
        preds.append(
            InferencePrediction(
                model_id=model_id,
                model_version=model_version,
                model_family=DEFAULT_DEPLOYMENT_MODEL_FAMILY,
                feature_schema_version="1.0.0",
                decision_threshold=decision_threshold,
                predicted_label=label,
                predicted_label_name="attack_associated" if label == 1 else "normal",
                model_score=float(score_val),
                is_calibrated_probability=False,
            )
        )
    return tuple(preds)


def train_and_save_deployment_random_forest(
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    artifact_dir: str | Path | None = None,
    trusted_root: str | Path = DEFAULT_TRUSTED_MODELS_DIR,
    model_id: str = DEFAULT_DEPLOYMENT_MODEL_ID,
    model_version: str = DEFAULT_DEPLOYMENT_MODEL_VERSION,
    n_jobs: int | None = -1,
    base_dir: str | Path | None = None,
    scenario_datasets: Mapping[str, CompactScenarioDataset] | None = None,
    verified_config: Mapping[str, object] | None = None,
    config_sha256: str | None = None,
    created_at_utc: str | None = None,
) -> DeploymentTrainingResult:
    """Train, save, reload, and deterministically verify the deployment RandomForestClassifier artifact."""
    resolved_config_path = Path(config_path)
    resolved_base = PROJECT_ROOT if base_dir is None else Path(base_dir).resolve()

    if verified_config is None:
        try:
            config = load_and_verify_experiment_config(
                resolved_config_path, base_dir=resolved_base
            )
        except TrainingDataError as exc:
            raise DeploymentTrainingError(str(exc)) from exc
    else:
        config = verified_config

    resolved_config_sha256 = (
        _sha256_file(resolved_config_path) if config_sha256 is None else config_sha256.lower()
    )

    policy = select_deployment_configuration_from_inner_checkpoints(
        config,
        config_sha256=resolved_config_sha256,
        config_path=resolved_config_path,
        base_dir=resolved_base,
        checkpoint_dir=checkpoint_dir,
        model_name=DEFAULT_DEPLOYMENT_MODEL_FAMILY,
        weighting_regime=DEFAULT_DEPLOYMENT_WEIGHTING_REGIME,
    )

    dataset_prov = config.get("dataset_provenance")
    if not isinstance(dataset_prov, Mapping) or not isinstance(dataset_prov.get("scenarios"), Mapping):
        raise DeploymentTrainingError("Experiment config is missing dataset_provenance.scenarios.")
    configured_scenarios = sorted(str(name) for name in dataset_prov["scenarios"].keys())

    datasets: dict[str, CompactScenarioDataset] = {}
    if scenario_datasets is not None:
        if sorted(scenario_datasets.keys()) != configured_scenarios:
            raise DeploymentTrainingError(
                f"Injected scenario_datasets keys {sorted(scenario_datasets.keys())} "
                f"do not match configured scenarios {configured_scenarios}."
            )
        datasets = dict(scenario_datasets)
    else:
        for scenario_id in configured_scenarios:
            try:
                datasets[scenario_id] = load_verified_scenario_dataset(
                    config, scenario_id, base_dir=resolved_base
                )
            except BenchmarkError as exc:
                raise DeploymentTrainingError(str(exc)) from exc

    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    scenario_record_counts: dict[str, dict[str, int]] = {}
    scenario_event_id_hashes: dict[str, str] = {}
    total_excluded_unknown = 0

    for scenario_id in configured_scenarios:
        ds = datasets[scenario_id]
        if ds.scenario_id != scenario_id:
            raise DeploymentTrainingError(
                f"Scenario dataset mismatch: expected {scenario_id!r}, got {ds.scenario_id!r}."
            )
        if ds.features.ndim != 2 or ds.features.shape[1] != 13:
            raise DeploymentTrainingError(
                f"{scenario_id}: expected 13-feature matrix, got shape {ds.features.shape}."
            )
        if len(ds.labels) != ds.features.shape[0]:
            raise DeploymentTrainingError(f"{scenario_id}: feature/label length mismatch.")

        n_normal = int((ds.labels == 0).sum())
        n_attack = int((ds.labels == 1).sum())
        if n_normal + n_attack != len(ds.labels):
            raise DeploymentTrainingError(
                f"{scenario_id}: non-binary or unknown labels detected in supervised dataset."
            )
        if (
            n_normal != ds.load_summary.included_normal_count
            or n_attack != ds.load_summary.included_attack_count
        ):
            raise DeploymentTrainingError(
                f"{scenario_id}: label counts do not match verified ScenarioLoadSummary."
            )

        feature_blocks.append(ds.features)
        label_blocks.append(ds.labels)
        total_excluded_unknown += int(ds.load_summary.excluded_unknown_count)
        scenario_event_id_hashes[scenario_id] = ds.event_ids_sha256
        scenario_record_counts[scenario_id] = {
            "total_supervised": int(len(ds.labels)),
            "normal": n_normal,
            "attack_associated": n_attack,
            "excluded_unknown": int(ds.load_summary.excluded_unknown_count),
        }

    x_train = np.vstack(feature_blocks)
    y_train = np.concatenate(label_blocks)
    total_records = int(len(y_train))
    total_normal = int((y_train == 0).sum())
    total_attack = int((y_train == 1).sum())

    try:
        fitted = train_random_forest_baseline(
            x_train,
            y_train,
            weighting_regime=policy.weighting_regime,
            n_estimators=int(policy.selected_hyperparameters["n_estimators"]),
            max_depth=int(policy.selected_hyperparameters["max_depth"]),
            min_samples_leaf=int(policy.selected_hyperparameters["min_samples_leaf"]),
            max_features=str(policy.selected_hyperparameters["max_features"]),
            random_state=policy.deployment_random_state,
            n_jobs=n_jobs,
            config_path=resolved_config_path,
        )
    except BaselineModelError as exc:
        raise DeploymentTrainingError(f"Deployment RandomForest training failed: {exc}") from exc

    # Pin inference to single-threaded execution so predict_proba accumulation is
    # bit-for-bit deterministic across repeated calls and reloads.
    fitted.pipeline.named_steps["classifier"].n_jobs = 1

    verification_fixture = build_synthetic_verification_fixture()
    pre_save_predictions = _predict_from_fitted(
        fitted,
        verification_fixture,
        model_id=model_id,
        model_version=model_version,
        decision_threshold=policy.selected_threshold,
    )

    resolved_trusted_root = Path(trusted_root)
    target_artifact_dir = (
        resolved_trusted_root / model_id / model_version
        if artifact_dir is None
        else Path(artifact_dir)
    )

    training_dataset_provenance: dict[str, object] = {
        "dataset_name": str(dataset_prov["dataset_name"]),
        "dataset_version": str(dataset_prov["version"]),
        "experiment_id": str(config["experiment_id"]),
        "scenarios": configured_scenarios,
        "scenario_labelled_sha256": {
            name: str(dataset_prov["scenarios"][name]["labelled_records_sha256"])
            for name in configured_scenarios
        },
        "scenario_event_ids_sha256": scenario_event_id_hashes,
        "unknown_records_excluded": True,
        "benchmark_outer_test_predictions_excluded": True,
        "record_counts": {
            "total_training_records": total_records,
            "normal_records": total_normal,
            "attack_associated_records": total_attack,
            "excluded_unknown_records": total_excluded_unknown,
            "per_scenario": scenario_record_counts,
        },
        "deployment_policy": policy.to_dict(),
    }

    intended_use = (
        "Exploratory deployment candidate for ForensicInsight Apache HTTP event scoring "
        "using the frozen 13-feature FEATURE_SCHEMA v1.0.0. Intended for supervised triage "
        "and research integration behind explicit operator approval, never as a standalone "
        "forensic verdict or calibrated probability estimator."
    )
    limitations = [
        "Trained on all four eligible verified AIT-LDS v2.1 scenarios (fox, harrison, russellmitchell, santos) and has NOT undergone independent external evaluation on unseen production or external testbed networks.",
        "Predicted scores from predict_proba are uncalibrated model scores (is_calibrated_probability=false), not calibrated probabilities.",
        "All 617,876 unknown-labelled AIT-LDS v2.1 records (42.0% of total events) were excluded from supervised training and are never treated as normal.",
        "Source dataset manifests retain scientifically_adequate=false and selected_assignment=null; severe class imbalance (83.84:1 attack:normal) and shared synthetic simulation patterns limit external generalization.",
        "Requires exact 13-feature FEATURE_SCHEMA v1.0.0 inputs; scenario identifiers and forensic provenance fields are excluded from the feature matrix.",
    ]

    try:
        manifest_path, manifest = save_model_artifact(
            fitted,
            target_artifact_dir,
            trusted_root=resolved_trusted_root,
            model_id=model_id,
            model_version=model_version,
            decision_threshold=policy.selected_threshold,
            training_dataset_provenance=training_dataset_provenance,
            training_config_sha256=resolved_config_sha256,
            intended_use=intended_use,
            limitations=limitations,
            created_at_utc=created_at_utc,
        )
        model_file_meta = manifest["model_file"]
        assert isinstance(model_file_meta, dict)
        expected_model_sha = str(model_file_meta["sha256"])

        loaded: LoadedModelArtifact = load_model_artifact(
            target_artifact_dir,
            trusted_root=resolved_trusted_root,
            operator_approved=True,
            expected_model_id=model_id,
            expected_model_version=model_version,
            expected_model_sha256=expected_model_sha,
        )
        post_reload_predictions = loaded.predict_batch(verification_fixture)
    except ModelArtifactError as exc:
        raise DeploymentTrainingError(f"Model artifact save/reload verification failed: {exc}") from exc

    if post_reload_predictions != pre_save_predictions:
        raise DeploymentTrainingError(
            "Post-reload inference predictions do not match pre-save fitted model predictions."
        )

    model_file_path = loaded.artifact_dir / str(model_file_meta["filename"])
    return DeploymentTrainingResult(
        model_id=model_id,
        model_version=model_version,
        artifact_dir=loaded.artifact_dir,
        manifest_path=manifest_path,
        model_file_path=model_file_path,
        model_file_sha256=loaded.model_file_sha256,
        model_file_size_bytes=int(model_file_meta["size_bytes"]),
        policy_selection=policy,
        total_training_records=total_records,
        normal_records=total_normal,
        attack_associated_records=total_attack,
        excluded_unknown_records=total_excluded_unknown,
        scenario_record_counts=scenario_record_counts,
        verification_predictions_match=True,
        verification_predictions=post_reload_predictions,
        manifest=manifest,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and verify the versioned ForensicInsight deployment RandomForestClassifier artifact."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG_PATH,
        help="Path to verified experiment_config.json.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing verified inner-validation checkpoints.",
    )
    parser.add_argument(
        "--trusted-root",
        type=Path,
        default=DEFAULT_TRUSTED_MODELS_DIR,
        help="Trusted root directory for saved model artifacts.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional explicit artifact directory inside --trusted-root.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_DEPLOYMENT_MODEL_ID,
        help="Versioned model identifier.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=DEFAULT_DEPLOYMENT_MODEL_VERSION,
        help="Semantic version for the deployment model artifact.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel worker threads for RandomForest training (default: -1).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = train_and_save_deployment_random_forest(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        artifact_dir=args.artifact_dir,
        trusted_root=args.trusted_root,
        model_id=args.model_id,
        model_version=args.model_version,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(result.to_summary_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
