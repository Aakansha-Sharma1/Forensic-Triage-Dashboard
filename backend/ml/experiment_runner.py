from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .baselines import (
    BaselineModelError,
    records_to_arrays,
    train_decision_tree_baseline,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from .evaluation import (
    EvaluationError,
    evaluate_hard_predictions,
    evaluate_probabilistic_predictions,
)
from .schema import FEATURE_SCHEMA_VERSION
from .training_data import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    PROJECT_ROOT,
    SupervisedRecord,
    TrainingDataError,
    load_and_verify_experiment_config,
)


SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "RuleBaseline_R_A_to_D",
    "DummyClassifier",
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
)
SUPPORTED_WEIGHTING_REGIMES: tuple[str, ...] = (
    "class_weighted",
    "scenario_equalized_class_weighted",
)


class ExperimentRunnerError(ValueError):
    """Raised when experiment config, split manifest, or nested LOSO folds are invalid."""


@dataclass(frozen=True)
class InnerCandidatePlan:
    """Validated inner train/validation split isolated from its outer test scenario."""

    candidate_id: str
    train_scenarios: tuple[str, ...]
    validation_scenarios: tuple[str, ...]
    outer_test_scenarios: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "train_scenarios": list(self.train_scenarios),
            "validation_scenarios": list(self.validation_scenarios),
            "outer_test_scenarios": list(self.outer_test_scenarios),
        }


@dataclass(frozen=True)
class OuterFoldPlan:
    """Validated outer LOSO fold with strictly isolated inner train/validation candidates."""

    fold_id: str
    test_scenarios: tuple[str, ...]
    outer_train_scenarios: tuple[str, ...]
    inner_candidates: tuple[InnerCandidatePlan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "test_scenarios": list(self.test_scenarios),
            "outer_train_scenarios": list(self.outer_train_scenarios),
            "inner_candidates": [c.to_dict() for c in self.inner_candidates],
        }


@dataclass(frozen=True)
class ModelGridSpec:
    """Validated, deterministically ordered model specification from experiment_config.json."""

    name: str
    family: str
    probabilistic: bool
    preprocessing_profile: str | None
    uses_weighting_regimes: bool
    uses_threshold_tuning: bool
    stochastic: bool
    fixed_parameters: Mapping[str, object]
    parameter_combinations: tuple[Mapping[str, object], ...]
    weighting_regimes: tuple[str, ...]
    random_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixed_parameters", MappingProxyType(dict(self.fixed_parameters))
        )
        object.__setattr__(
            self,
            "parameter_combinations",
            tuple(MappingProxyType(dict(combo)) for combo in self.parameter_combinations),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "probabilistic": self.probabilistic,
            "preprocessing_profile": self.preprocessing_profile,
            "uses_weighting_regimes": self.uses_weighting_regimes,
            "uses_threshold_tuning": self.uses_threshold_tuning,
            "stochastic": self.stochastic,
            "fixed_parameters": dict(self.fixed_parameters),
            "parameter_combinations": [dict(c) for c in self.parameter_combinations],
            "weighting_regimes": list(self.weighting_regimes),
            "random_seeds": list(self.random_seeds),
        }


@dataclass(frozen=True)
class NestedLOSOPlan:
    """Complete provenance-verified nested 4-fold LOSO execution plan."""

    experiment_id: str
    config_schema_version: str
    study_status: str
    feature_schema_version: str
    split_manifest_path: str
    split_manifest_sha256: str
    quality_report_path: str
    quality_report_sha256: str
    scenario_ids: tuple[str, ...]
    scenario_provenance: Mapping[str, Mapping[str, str]]
    weighting_regimes: tuple[str, ...]
    random_seeds: tuple[int, ...]
    fixed_threshold: float
    candidate_thresholds: tuple[float, ...]
    primary_selection_metric: str
    outer_folds: tuple[OuterFoldPlan, ...]
    model_specs: tuple[ModelGridSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_provenance",
            MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in self.scenario_provenance.items()}
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "config_schema_version": self.config_schema_version,
            "study_status": self.study_status,
            "feature_schema_version": self.feature_schema_version,
            "split_manifest_path": self.split_manifest_path,
            "split_manifest_sha256": self.split_manifest_sha256,
            "quality_report_path": self.quality_report_path,
            "quality_report_sha256": self.quality_report_sha256,
            "scenario_ids": list(self.scenario_ids),
            "scenario_provenance": {
                k: dict(v) for k, v in self.scenario_provenance.items()
            },
            "weighting_regimes": list(self.weighting_regimes),
            "random_seeds": list(self.random_seeds),
            "fixed_threshold": self.fixed_threshold,
            "candidate_thresholds": list(self.candidate_thresholds),
            "primary_selection_metric": self.primary_selection_metric,
            "outer_folds": [fold.to_dict() for fold in self.outer_folds],
            "model_specs": [spec.to_dict() for spec in self.model_specs],
        }


class FoldEvaluator(Protocol):
    """Orchestration hook invoked once per (model_spec, outer_fold) in deterministic order."""

    def evaluate_outer_fold(
        self,
        plan: NestedLOSOPlan,
        fold: OuterFoldPlan,
        model_spec: ModelGridSpec,
    ) -> Mapping[str, object]: ...


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _validate_scenario_list(value: object, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ExperimentRunnerError(f"{description} must be a non-empty list of scenario IDs.")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ExperimentRunnerError(f"Invalid scenario ID in {description}: {item!r}")
        cleaned.append(item.strip())
    if len(set(cleaned)) != len(cleaned):
        raise ExperimentRunnerError(f"Duplicate scenario IDs in {description}: {cleaned!r}")
    return tuple(cleaned)


def _validate_seeds(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ExperimentRunnerError("random_seeds must be a non-empty list of integers.")
    seeds: list[int] = []
    for seed in value:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ExperimentRunnerError(f"Invalid random seed: {seed!r}")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ExperimentRunnerError("random_seeds must not contain duplicates.")
    return tuple(seeds)


def _validate_thresholds(threshold_cfg: object) -> tuple[float, tuple[float, ...]]:
    if not isinstance(threshold_cfg, dict):
        raise ExperimentRunnerError("evaluation_protocol.threshold_selection is required.")
    if threshold_cfg.get("fit_scope") != "inner_validation_only":
        raise ExperimentRunnerError("threshold_selection.fit_scope must be 'inner_validation_only'.")
    fixed = threshold_cfg.get("fixed_threshold")
    if (
        isinstance(fixed, bool)
        or not isinstance(fixed, (int, float))
        or not math.isfinite(float(fixed))
        or not (0.0 <= float(fixed) <= 1.0)
    ):
        raise ExperimentRunnerError(f"Invalid fixed_threshold: {fixed!r}")
    candidates_raw = threshold_cfg.get("candidate_thresholds")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ExperimentRunnerError("candidate_thresholds must be a non-empty list.")
    candidates: list[float] = []
    for item in candidates_raw:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not (0.0 <= float(item) <= 1.0)
        ):
            raise ExperimentRunnerError(f"Invalid candidate threshold: {item!r}")
        candidates.append(float(item))
    if len(set(candidates)) != len(candidates):
        raise ExperimentRunnerError("candidate_thresholds must not contain duplicates.")
    return float(fixed), tuple(candidates)


def _expand_hyperparameter_grid(grid: object, model_name: str) -> tuple[dict[str, object], ...]:
    if not isinstance(grid, dict) or not grid:
        raise ExperimentRunnerError(f"Model {model_name!r} requires a non-empty hyperparameter_grid.")
    keys = list(grid.keys())
    value_lists: list[list[object]] = []
    for key in keys:
        vals = grid[key]
        if not isinstance(vals, list) or not vals:
            raise ExperimentRunnerError(
                f"Model {model_name!r} hyperparameter {key!r} must be a non-empty list."
            )
        value_lists.append(vals)
    return tuple(dict(zip(keys, combo)) for combo in product(*value_lists))


def _validate_model_specs(
    config: Mapping[str, Any],
    weighting_regimes: tuple[str, ...],
    random_seeds: tuple[int, ...],
) -> tuple[ModelGridSpec, ...]:
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ExperimentRunnerError("Experiment config must define a non-empty models list.")

    seen_names: set[str] = set()
    specs: list[ModelGridSpec] = []
    for entry in models:
        if not isinstance(entry, dict):
            raise ExperimentRunnerError("Each model entry must be a JSON object.")
        name = entry.get("name")
        if not isinstance(name, str) or name not in SUPPORTED_MODEL_NAMES:
            raise ExperimentRunnerError(f"Unsupported model name in config: {name!r}")
        if name in seen_names:
            raise ExperimentRunnerError(f"Duplicate model definition: {name!r}")
        seen_names.add(name)

        family = entry.get("family")
        probabilistic = entry.get("probabilistic")
        preproc = entry.get("preprocessing_profile")
        uses_weights = entry.get("uses_weighting_regimes")
        uses_thresh = entry.get("uses_threshold_tuning")
        stochastic = entry.get("stochastic")
        if (
            not isinstance(family, str)
            or not family.strip()
            or not isinstance(probabilistic, bool)
            or not isinstance(uses_weights, bool)
            or not isinstance(uses_thresh, bool)
            or not isinstance(stochastic, bool)
        ):
            raise ExperimentRunnerError(f"Model {name!r} has incomplete or invalid metadata.")

        fixed_params = entry.get("fixed_parameters", {})
        if not isinstance(fixed_params, dict):
            raise ExperimentRunnerError(f"Model {name!r} fixed_parameters must be an object.")
        if uses_weights and fixed_params.get("class_weight") is not None:
            raise ExperimentRunnerError(
                f"Model {name!r} must set fixed_parameters.class_weight=null to prevent double-weighting."
            )

        if name == "RuleBaseline_R_A_to_D":
            if probabilistic or uses_weights or uses_thresh or stochastic:
                raise ExperimentRunnerError("RuleBaseline_R_A_to_D must be non-probabilistic and unweighted.")
            combos: tuple[dict[str, object], ...] = ({},)
            model_seeds: tuple[int, ...] = ()
        else:
            combos = _expand_hyperparameter_grid(entry.get("hyperparameter_grid"), name)
            model_seeds = random_seeds if stochastic else (random_seeds[0],)

        specs.append(
            ModelGridSpec(
                name=name,
                family=family,
                probabilistic=probabilistic,
                preprocessing_profile=preproc,
                uses_weighting_regimes=uses_weights,
                uses_threshold_tuning=uses_thresh,
                stochastic=stochastic,
                fixed_parameters=fixed_params,
                parameter_combinations=combos,
                weighting_regimes=weighting_regimes if uses_weights else (),
                random_seeds=model_seeds,
            )
        )
    return tuple(specs)


def build_nested_loso_plan(
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    *,
    base_dir: str | Path | None = None,
) -> NestedLOSOPlan:
    """Validate experiment config & split manifest and build deterministic nested LOSO folds."""
    resolved_base = PROJECT_ROOT if base_dir is None else Path(base_dir).resolve()
    try:
        config = load_and_verify_experiment_config(config_path, base_dir=resolved_base)
    except TrainingDataError as exc:
        raise ExperimentRunnerError(str(exc)) from exc

    provenance = config["dataset_provenance"]
    split_ref = provenance["split_manifest"]
    quality_ref = provenance["quality_report"]
    split_path = _resolve_path(split_ref["path"], resolved_base)
    try:
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentRunnerError(f"Unable to read split manifest: {split_path}") from exc

    scenarios_cfg = provenance["scenarios"]
    scenario_ids = tuple(sorted(scenarios_cfg))
    if len(scenario_ids) != 4:
        raise ExperimentRunnerError(
            f"nested_4fold_loso requires exactly 4 configured scenarios, found {len(scenario_ids)}."
        )
    if split_manifest.get("scenario_ids") != list(scenario_ids):
        raise ExperimentRunnerError("Split manifest scenario_ids do not match experiment config.")

    cand_eval = split_manifest.get("candidate_evaluation", {})
    if (
        cand_eval.get("scientifically_adequate_candidate_count") != 0
        or cand_eval.get("selected_assignment") is not None
    ):
        raise ExperimentRunnerError(
            "Split manifest adequacy/selection state does not match exploratory protocol invariants."
        )

    split_summaries = split_manifest.get("scenario_summaries", {})
    scenario_provenance: dict[str, dict[str, str]] = {}
    for scenario in scenario_ids:
        cfg_ref = scenarios_cfg[scenario]
        sm = split_summaries.get(scenario)
        if not isinstance(sm, dict):
            raise ExperimentRunnerError(f"Split manifest is missing scenario summary for {scenario!r}.")
        for field in (
            "label_manifest_sha256",
            "labelled_records_sha256",
            "prepared_manifest_sha256",
            "prepared_records_sha256",
        ):
            if cfg_ref.get(field) != sm.get(field):
                raise ExperimentRunnerError(
                    f"Scenario {scenario!r} {field} mismatch between experiment config and split manifest."
                )
        scenario_provenance[scenario] = {
            "label_manifest_path": str(cfg_ref["label_manifest_path"]),
            "label_manifest_sha256": str(cfg_ref["label_manifest_sha256"]),
            "labelled_records_sha256": str(cfg_ref["labelled_records_sha256"]),
            "prepared_manifest_sha256": str(cfg_ref["prepared_manifest_sha256"]),
            "prepared_records_sha256": str(cfg_ref["prepared_records_sha256"]),
        }

    weighting_policy = config.get("weighting_policy", {})
    if (
        weighting_policy.get("mechanism") != "sample_weight_only"
        or weighting_policy.get("estimator_class_weight") is not None
        or weighting_policy.get("double_weighting_prohibited") is not True
    ):
        raise ExperimentRunnerError("Experiment config must enforce sample_weight_only weighting policy.")
    regimes_raw = weighting_policy.get("regimes")
    if not isinstance(regimes_raw, list) or not regimes_raw:
        raise ExperimentRunnerError("weighting_policy.regimes must be a non-empty list.")
    weighting_regimes = tuple(
        r["name"] for r in regimes_raw if isinstance(r, dict) and isinstance(r.get("name"), str)
    )
    if len(weighting_regimes) != len(regimes_raw) or any(
        r not in SUPPORTED_WEIGHTING_REGIMES for r in weighting_regimes
    ):
        raise ExperimentRunnerError(f"Unsupported weighting regime in config: {weighting_regimes!r}")

    random_seeds = _validate_seeds(config.get("random_seeds"))

    eval_proto = config.get("evaluation_protocol")
    if not isinstance(eval_proto, dict):
        raise ExperimentRunnerError("Experiment config is missing evaluation_protocol.")
    if eval_proto.get("strategy") != "nested_4fold_loso":
        raise ExperimentRunnerError(
            f"Unsupported evaluation strategy: {eval_proto.get('strategy')!r}"
        )
    if eval_proto.get("outer_test_isolation") is not True:
        raise ExperimentRunnerError("evaluation_protocol.outer_test_isolation must be true.")
    if eval_proto.get("selection_scope") != "inner_validation_only":
        raise ExperimentRunnerError("evaluation_protocol.selection_scope must be 'inner_validation_only'.")

    fixed_threshold, candidate_thresholds = _validate_thresholds(
        eval_proto.get("threshold_selection")
    )

    manifest_candidates = split_manifest.get("candidates")
    if not isinstance(manifest_candidates, list):
        raise ExperimentRunnerError("Split manifest is missing candidates list.")
    manifest_cand_by_id: dict[str, dict] = {}
    for cand in manifest_candidates:
        if not isinstance(cand, dict) or not isinstance(cand.get("candidate_id"), str):
            raise ExperimentRunnerError("Invalid candidate entry in split manifest.")
        cid = cand["candidate_id"]
        if cid in manifest_cand_by_id:
            raise ExperimentRunnerError(f"Duplicate candidate_id in split manifest: {cid!r}")
        if cand.get("scenario_partition_isolation") is not True:
            raise ExperimentRunnerError(f"Split manifest candidate {cid!r} failed partition isolation.")
        manifest_cand_by_id[cid] = cand

    raw_outer_folds = eval_proto.get("outer_folds")
    if not isinstance(raw_outer_folds, list) or len(raw_outer_folds) != len(scenario_ids):
        raise ExperimentRunnerError(
            f"Expected {len(scenario_ids)} outer_folds, found {len(raw_outer_folds) if isinstance(raw_outer_folds, list) else 0}."
        )

    outer_folds: list[OuterFoldPlan] = []
    seen_fold_ids: set[str] = set()
    seen_outer_tests: set[str] = set()
    seen_candidate_ids: set[str] = set()

    for raw_fold in raw_outer_folds:
        if not isinstance(raw_fold, dict):
            raise ExperimentRunnerError("Each outer_fold entry must be a JSON object.")
        fold_id = raw_fold.get("fold_id")
        if not isinstance(fold_id, str) or not fold_id.strip() or fold_id in seen_fold_ids:
            raise ExperimentRunnerError(f"Invalid or duplicate fold_id: {fold_id!r}")
        seen_fold_ids.add(fold_id)

        test_scenarios = _validate_scenario_list(raw_fold.get("test_scenarios"), f"{fold_id}.test_scenarios")
        outer_train = _validate_scenario_list(
            raw_fold.get("outer_train_scenarios"), f"{fold_id}.outer_train_scenarios"
        )
        if len(test_scenarios) != 1:
            raise ExperimentRunnerError(f"{fold_id}: LOSO outer fold must hold out exactly 1 test scenario.")
        if not set(test_scenarios).isdisjoint(outer_train):
            raise ExperimentRunnerError(f"{fold_id}: outer test scenario leaks into outer_train_scenarios.")
        if tuple(sorted(test_scenarios + outer_train)) != scenario_ids:
            raise ExperimentRunnerError(f"{fold_id}: test and outer_train do not cover all configured scenarios.")
        if test_scenarios[0] in seen_outer_tests:
            raise ExperimentRunnerError(f"Duplicate outer test scenario: {test_scenarios[0]!r}")
        seen_outer_tests.add(test_scenarios[0])

        raw_inners = raw_fold.get("inner_candidates")
        if not isinstance(raw_inners, list) or len(raw_inners) != len(outer_train):
            raise ExperimentRunnerError(
                f"{fold_id}: expected {len(outer_train)} inner_candidates."
            )

        inner_plans: list[InnerCandidatePlan] = []
        seen_inner_vals: set[str] = set()
        for raw_inner in raw_inners:
            if not isinstance(raw_inner, dict):
                raise ExperimentRunnerError(f"{fold_id}: inner candidate must be an object.")
            cid = raw_inner.get("candidate_id")
            if not isinstance(cid, str) or cid not in manifest_cand_by_id:
                raise ExperimentRunnerError(f"{fold_id}: unknown split manifest candidate_id {cid!r}.")
            if cid in seen_candidate_ids:
                raise ExperimentRunnerError(f"Duplicate inner candidate_id across plan: {cid!r}")
            seen_candidate_ids.add(cid)

            in_train = _validate_scenario_list(
                raw_inner.get("train_scenarios"), f"{fold_id}.{cid}.train_scenarios"
            )
            in_val = _validate_scenario_list(
                raw_inner.get("validation_scenarios"), f"{fold_id}.{cid}.validation_scenarios"
            )
            if not set(in_train).isdisjoint(in_val):
                raise ExperimentRunnerError(f"{fold_id}.{cid}: inner train and validation overlap.")
            if not set(in_train).isdisjoint(test_scenarios) or not set(in_val).isdisjoint(test_scenarios):
                raise ExperimentRunnerError(
                    f"{fold_id}.{cid}: outer test scenario leaked into inner candidate."
                )
            if tuple(sorted(in_train + in_val)) != tuple(sorted(outer_train)):
                raise ExperimentRunnerError(
                    f"{fold_id}.{cid}: inner train/validation do not partition outer_train_scenarios."
                )
            if len(in_val) != 1 or in_val[0] in seen_inner_vals:
                raise ExperimentRunnerError(
                    f"{fold_id}.{cid}: each outer-train scenario must be inner validation exactly once."
                )
            seen_inner_vals.add(in_val[0])

            manifest_assignment = manifest_cand_by_id[cid].get("assignment", {})
            if (
                manifest_assignment.get("train") != list(in_train)
                or manifest_assignment.get("validation") != list(in_val)
                or manifest_assignment.get("test") != list(test_scenarios)
            ):
                raise ExperimentRunnerError(
                    f"{fold_id}.{cid}: inner assignment does not match verified split manifest."
                )

            inner_plans.append(
                InnerCandidatePlan(
                    candidate_id=cid,
                    train_scenarios=in_train,
                    validation_scenarios=in_val,
                    outer_test_scenarios=test_scenarios,
                )
            )

        outer_folds.append(
            OuterFoldPlan(
                fold_id=fold_id,
                test_scenarios=test_scenarios,
                outer_train_scenarios=outer_train,
                inner_candidates=tuple(inner_plans),
            )
        )

    expected_total_candidates = len(scenario_ids) * (len(scenario_ids) - 1)
    if seen_outer_tests != set(scenario_ids) or len(seen_candidate_ids) != expected_total_candidates:
        raise ExperimentRunnerError(
            "Nested LOSO plan does not cover all scenarios and unique split candidates."
        )

    primary_metric = config.get("metrics", {}).get("primary_selection_metric")
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        raise ExperimentRunnerError("metrics.primary_selection_metric is required.")

    model_specs = _validate_model_specs(config, weighting_regimes, random_seeds)

    return NestedLOSOPlan(
        experiment_id=str(config.get("experiment_id", "")),
        config_schema_version=str(config.get("config_schema_version", "")),
        study_status=str(config.get("study_status", "")),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        split_manifest_path=str(split_ref["path"]),
        split_manifest_sha256=str(split_ref["sha256"]),
        quality_report_path=str(quality_ref["path"]),
        quality_report_sha256=str(quality_ref["sha256"]),
        scenario_ids=scenario_ids,
        scenario_provenance=scenario_provenance,
        weighting_regimes=weighting_regimes,
        random_seeds=random_seeds,
        fixed_threshold=fixed_threshold,
        candidate_thresholds=candidate_thresholds,
        primary_selection_metric=primary_metric,
        outer_folds=tuple(outer_folds),
        model_specs=model_specs,
    )


def run_nested_loso_plan(
    plan: NestedLOSOPlan,
    evaluator: FoldEvaluator,
    *,
    model_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Deterministically orchestrate outer-fold evaluation via an injected FoldEvaluator."""
    specs_by_name = {spec.name: spec for spec in plan.model_specs}
    if model_names is None:
        selected_specs = plan.model_specs
    else:
        if not model_names or len(set(model_names)) != len(model_names):
            raise ExperimentRunnerError("model_names must be a non-empty sequence of unique names.")
        missing = [name for name in model_names if name not in specs_by_name]
        if missing:
            raise ExperimentRunnerError(f"Requested models not found in plan: {missing!r}")
        selected_specs = tuple(specs_by_name[name] for name in model_names)

    model_results: dict[str, list[dict[str, object]]] = {}
    for spec in selected_specs:
        fold_outputs: list[dict[str, object]] = []
        for fold in plan.outer_folds:
            outcome = evaluator.evaluate_outer_fold(plan, fold, spec)
            if not isinstance(outcome, Mapping):
                raise ExperimentRunnerError(
                    f"FoldEvaluator returned non-mapping result for {spec.name}/{fold.fold_id}."
                )
            fold_outputs.append(
                {
                    "fold_id": fold.fold_id,
                    "test_scenarios": list(fold.test_scenarios),
                    "result": dict(outcome),
                }
            )
        model_results[spec.name] = fold_outputs

    return {
        "experiment_id": plan.experiment_id,
        "study_status": plan.study_status,
        "split_manifest_sha256": plan.split_manifest_sha256,
        "quality_report_sha256": plan.quality_report_sha256,
        "models_evaluated": [spec.name for spec in selected_specs],
        "results": model_results,
    }


@dataclass(frozen=True)
class InnerValidationObservation:
    """One precomputed inner-validation Macro-F1 observation for a joint configuration."""

    fold_id: str
    candidate_id: str
    validation_scenario: str
    model_name: str
    weighting_regime: str
    seed: int
    hyperparameters: Mapping[str, object]
    threshold: float
    macro_f1: float | None
    partition_role: str = "validation"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hyperparameters", MappingProxyType(dict(self.hyperparameters))
        )


@dataclass(frozen=True)
class InnerSelectionResult:
    """Deterministic inner-validation selection result for one (fold, model, weighting_regime)."""

    experiment_id: str
    split_manifest_sha256: str
    quality_report_sha256: str
    fold_id: str
    excluded_outer_test_scenarios: tuple[str, ...]
    inner_candidate_ids: tuple[str, ...]
    model_name: str
    weighting_regime: str
    seeds_used: tuple[int, ...]
    selected_hyperparameters: Mapping[str, object]
    selected_threshold: float
    mean_validation_macro_f1: float
    std_validation_macro_f1: float
    evaluated_joint_configurations: int
    observations_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_hyperparameters",
            MappingProxyType(dict(self.selected_hyperparameters)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "split_manifest_sha256": self.split_manifest_sha256,
            "quality_report_sha256": self.quality_report_sha256,
            "fold_id": self.fold_id,
            "excluded_outer_test_scenarios": list(self.excluded_outer_test_scenarios),
            "inner_candidate_ids": list(self.inner_candidate_ids),
            "model_name": self.model_name,
            "weighting_regime": self.weighting_regime,
            "seeds_used": list(self.seeds_used),
            "selected_hyperparameters": dict(self.selected_hyperparameters),
            "selected_threshold": self.selected_threshold,
            "mean_validation_macro_f1": self.mean_validation_macro_f1,
            "std_validation_macro_f1": self.std_validation_macro_f1,
            "evaluated_joint_configurations": self.evaluated_joint_configurations,
            "observations_count": self.observations_count,
        }


def _canonical_param_key(params: Mapping[str, object]) -> str:
    return json.dumps(dict(params), sort_keys=True, separators=(",", ":"))


def select_inner_validation_configuration(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    model_spec: ModelGridSpec,
    weighting_regime: str,
    observations: Sequence[InnerValidationObservation],
) -> InnerSelectionResult:
    """Select the best joint (hyperparameters, threshold) strictly from inner validation."""
    if fold not in plan.outer_folds:
        raise ExperimentRunnerError(f"Fold {fold.fold_id!r} is not part of the validated plan.")
    if model_spec not in plan.model_specs:
        raise ExperimentRunnerError(f"Model {model_spec.name!r} is not part of the validated plan.")
    if not (model_spec.probabilistic and model_spec.uses_threshold_tuning and model_spec.uses_weighting_regimes):
        raise ExperimentRunnerError(
            f"Model {model_spec.name!r} does not use weighted joint hyperparameter/threshold selection."
        )
    if weighting_regime not in model_spec.weighting_regimes:
        raise ExperimentRunnerError(
            f"Weighting regime {weighting_regime!r} is not configured for {model_spec.name!r}."
        )
    if not observations:
        raise ExperimentRunnerError("Inner validation observations must be non-empty.")

    expected_candidates = {
        ic.candidate_id: ic.validation_scenarios[0] for ic in fold.inner_candidates
    }
    outer_test_set = set(fold.test_scenarios)
    expected_seeds = plan.random_seeds
    expected_seed_set = set(expected_seeds)
    expected_thresholds = plan.candidate_thresholds
    expected_threshold_set = set(expected_thresholds)

    param_index_by_key: dict[str, int] = {}
    params_by_index: list[Mapping[str, object]] = []
    for idx, combo in enumerate(model_spec.parameter_combinations):
        key = _canonical_param_key(combo)
        if key in param_index_by_key:
            raise ExperimentRunnerError(f"Duplicate hyperparameter combination in {model_spec.name!r}.")
        param_index_by_key[key] = idx
        params_by_index.append(combo)

    seen_keys: set[tuple[str, int, int, float]] = set()
    scores_by_joint: dict[tuple[int, float], dict[tuple[str, int], float]] = {
        (param_idx, threshold): {}
        for param_idx in range(len(params_by_index))
        for threshold in expected_thresholds
    }

    for idx, obs in enumerate(observations):
        if not isinstance(obs, InnerValidationObservation):
            raise ExperimentRunnerError(f"Observation {idx} must be an InnerValidationObservation.")
        if obs.partition_role != "validation":
            raise ExperimentRunnerError(
                f"Observation {idx}: partition_role={obs.partition_role!r} is prohibited in inner selection."
            )
        if obs.fold_id != fold.fold_id:
            raise ExperimentRunnerError(
                f"Observation {idx}: fold_id={obs.fold_id!r} does not match {fold.fold_id!r}."
            )
        if obs.model_name != model_spec.name:
            raise ExperimentRunnerError(
                f"Observation {idx}: model_name={obs.model_name!r} does not match {model_spec.name!r}."
            )
        if obs.weighting_regime != weighting_regime:
            raise ExperimentRunnerError(
                f"Observation {idx}: mixed weighting regime {obs.weighting_regime!r} != {weighting_regime!r}."
            )
        if obs.validation_scenario in outer_test_set:
            raise ExperimentRunnerError(
                f"Observation {idx}: outer test scenario {obs.validation_scenario!r} leaked into inner selection."
            )
        if obs.candidate_id not in expected_candidates:
            raise ExperimentRunnerError(
                f"Observation {idx}: unexpected candidate_id {obs.candidate_id!r} for fold {fold.fold_id!r}."
            )
        if obs.validation_scenario != expected_candidates[obs.candidate_id]:
            raise ExperimentRunnerError(
                f"Observation {idx}: validation_scenario {obs.validation_scenario!r} does not match candidate {obs.candidate_id!r}."
            )
        if isinstance(obs.seed, bool) or obs.seed not in expected_seed_set:
            raise ExperimentRunnerError(f"Observation {idx}: unconfigured seed {obs.seed!r}.")
        if (
            isinstance(obs.threshold, bool)
            or not isinstance(obs.threshold, (int, float))
            or not math.isfinite(float(obs.threshold))
            or float(obs.threshold) not in expected_threshold_set
        ):
            raise ExperimentRunnerError(f"Observation {idx}: invalid threshold {obs.threshold!r}.")
        threshold_val = float(obs.threshold)

        param_key = _canonical_param_key(obs.hyperparameters)
        if param_key not in param_index_by_key:
            raise ExperimentRunnerError(
                f"Observation {idx}: unconfigured hyperparameters {dict(obs.hyperparameters)!r}."
            )
        param_idx = param_index_by_key[param_key]

        if (
            obs.macro_f1 is None
            or isinstance(obs.macro_f1, bool)
            or not isinstance(obs.macro_f1, (int, float))
            or not math.isfinite(float(obs.macro_f1))
            or not (0.0 <= float(obs.macro_f1) <= 1.0)
        ):
            raise ExperimentRunnerError(
                f"Observation {idx}: undefined or invalid Macro-F1 score {obs.macro_f1!r}."
            )

        unique_key = (obs.candidate_id, obs.seed, param_idx, threshold_val)
        if unique_key in seen_keys:
            raise ExperimentRunnerError(f"Duplicate inner validation observation: {unique_key!r}")
        seen_keys.add(unique_key)
        scores_by_joint[(param_idx, threshold_val)][(obs.candidate_id, obs.seed)] = float(
            obs.macro_f1
        )

    expected_per_joint = len(expected_candidates) * len(expected_seeds)
    expected_total = len(params_by_index) * len(expected_thresholds) * expected_per_joint
    if len(seen_keys) != expected_total:
        raise ExperimentRunnerError(
            f"Incomplete inner validation observations: got {len(seen_keys)}, expected {expected_total}."
        )

    ordered_candidate_ids = tuple(ic.candidate_id for ic in fold.inner_candidates)
    ranked: list[tuple[tuple[float, float, float, int], int, float, float, float]] = []
    for param_idx in range(len(params_by_index)):
        for threshold_val in expected_thresholds:
            cell_map = scores_by_joint[(param_idx, threshold_val)]
            ordered_scores = [
                cell_map[(cid, seed)]
                for cid in ordered_candidate_ids
                for seed in expected_seeds
            ]
            mean_f1 = sum(ordered_scores) / len(ordered_scores)
            variance = sum((score - mean_f1) ** 2 for score in ordered_scores) / len(ordered_scores)
            std_f1 = math.sqrt(variance)
            rank_key = (
                -round(mean_f1, 12),
                round(abs(threshold_val - 0.5), 12),
                threshold_val,
                param_idx,
            )
            ranked.append((rank_key, param_idx, threshold_val, mean_f1, std_f1))

    ranked.sort(key=lambda item: item[0])
    _, best_param_idx, best_threshold, best_mean, best_std = ranked[0]

    return InnerSelectionResult(
        experiment_id=plan.experiment_id,
        split_manifest_sha256=plan.split_manifest_sha256,
        quality_report_sha256=plan.quality_report_sha256,
        fold_id=fold.fold_id,
        excluded_outer_test_scenarios=fold.test_scenarios,
        inner_candidate_ids=ordered_candidate_ids,
        model_name=model_spec.name,
        weighting_regime=weighting_regime,
        seeds_used=expected_seeds,
        selected_hyperparameters=params_by_index[best_param_idx],
        selected_threshold=best_threshold,
        mean_validation_macro_f1=best_mean,
        std_validation_macro_f1=best_std,
        evaluated_joint_configurations=len(ranked),
        observations_count=len(seen_keys),
    )


def select_inner_configurations_by_regime(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    model_spec: ModelGridSpec,
    observations_by_regime: Mapping[str, Sequence[InnerValidationObservation]],
) -> dict[str, InnerSelectionResult]:
    """Select best joint inner configuration separately for each configured weighting regime."""
    expected_regimes = model_spec.weighting_regimes
    if set(observations_by_regime) != set(expected_regimes):
        raise ExperimentRunnerError(
            f"observations_by_regime keys {sorted(observations_by_regime)!r} must match {list(expected_regimes)!r}."
        )
    return {
        regime: select_inner_validation_configuration(
            plan, fold, model_spec, regime, observations_by_regime[regime]
        )
        for regime in expected_regimes
    }


_TUNABLE_MODEL_NAMES: frozenset[str] = frozenset(
    ("LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier")
)


def _validate_partition_records(
    records: Sequence[SupervisedRecord],
    *,
    expected_scenarios: tuple[str, ...],
    forbidden_scenarios: frozenset[str],
    partition_name: str,
) -> set[tuple[str, str, str]]:
    if not records:
        raise ExperimentRunnerError(f"{partition_name} records must be non-empty.")
    expected_set = set(expected_scenarios)
    seen_scenarios: set[str] = set()
    identities: set[tuple[str, str, str]] = set()
    for idx, record in enumerate(records):
        if not isinstance(record, SupervisedRecord):
            raise ExperimentRunnerError(
                f"{partition_name}[{idx}] must be a SupervisedRecord."
            )
        if record.scenario_id in forbidden_scenarios:
            raise ExperimentRunnerError(
                f"{partition_name}[{idx}] contains forbidden scenario {record.scenario_id!r}."
            )
        if record.scenario_id not in expected_set:
            raise ExperimentRunnerError(
                f"{partition_name}[{idx}] scenario {record.scenario_id!r} not in expected {expected_scenarios!r}."
            )
        if type(record.label) is not int or record.label not in (0, 1):
            raise ExperimentRunnerError(
                f"{partition_name}[{idx}] has invalid supervised label {record.label!r}."
            )
        if not record.label_provenance:
            raise ExperimentRunnerError(
                f"{partition_name}[{idx}] is missing label_provenance."
            )
        identity = (record.scenario_id, record.source_filename, record.event_id)
        if identity in identities:
            raise ExperimentRunnerError(
                f"{partition_name} contains duplicate record identity {identity!r}."
            )
        identities.add(identity)
        seen_scenarios.add(record.scenario_id)

    missing_scenarios = sorted(expected_set - seen_scenarios)
    if missing_scenarios:
        raise ExperimentRunnerError(
            f"{partition_name} is missing expected scenarios: {missing_scenarios!r}."
        )
    return identities


def _fit_configured_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    weighting_regime: str,
    train_scenarios: tuple[str, ...],
    params: Mapping[str, object],
    seed: int,
    config_path: str | Path,
):
    if model_name == "LogisticRegression":
        return train_logistic_regression_baseline(
            x_train,
            y_train,
            weighting_regime=weighting_regime,
            scenario_ids=train_scenarios,
            C=float(params["C"]),
            random_state=seed,
            config_path=config_path,
        )
    if model_name == "DecisionTreeClassifier":
        return train_decision_tree_baseline(
            x_train,
            y_train,
            weighting_regime=weighting_regime,
            scenario_ids=train_scenarios,
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed,
            config_path=config_path,
        )
    if model_name == "RandomForestClassifier":
        return train_random_forest_baseline(
            x_train,
            y_train,
            weighting_regime=weighting_regime,
            scenario_ids=train_scenarios,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=str(params["max_features"]),
            random_state=seed,
            config_path=config_path,
        )
    raise ExperimentRunnerError(f"Unsupported tunable model for inner execution: {model_name!r}")


def execute_inner_candidate_validation(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    candidate: InnerCandidatePlan,
    model_spec: ModelGridSpec,
    weighting_regime: str,
    train_records: Sequence[SupervisedRecord],
    validation_records: Sequence[SupervisedRecord],
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> tuple[InnerValidationObservation, ...]:
    """Fit on inner training scenarios only and evaluate all grids/thresholds on inner validation."""
    if fold not in plan.outer_folds:
        raise ExperimentRunnerError(f"Fold {fold.fold_id!r} is not part of the validated plan.")
    if candidate not in fold.inner_candidates:
        raise ExperimentRunnerError(
            f"Candidate {candidate.candidate_id!r} is not part of fold {fold.fold_id!r}."
        )
    if model_spec not in plan.model_specs:
        raise ExperimentRunnerError(f"Model {model_spec.name!r} is not part of the validated plan.")
    if model_spec.name not in _TUNABLE_MODEL_NAMES:
        raise ExperimentRunnerError(
            f"Model {model_spec.name!r} is not supported by inner candidate validation executor."
        )
    if weighting_regime not in model_spec.weighting_regimes:
        raise ExperimentRunnerError(
            f"Weighting regime {weighting_regime!r} is not configured for {model_spec.name!r}."
        )

    outer_test_set = frozenset(fold.test_scenarios)
    train_ids = _validate_partition_records(
        train_records,
        expected_scenarios=candidate.train_scenarios,
        forbidden_scenarios=outer_test_set | frozenset(candidate.validation_scenarios),
        partition_name="train_records",
    )
    val_ids = _validate_partition_records(
        validation_records,
        expected_scenarios=candidate.validation_scenarios,
        forbidden_scenarios=outer_test_set | frozenset(candidate.train_scenarios),
        partition_name="validation_records",
    )
    if not train_ids.isdisjoint(val_ids):
        raise ExperimentRunnerError("Record identity overlap detected between train and validation.")

    try:
        x_train, y_train, train_scenarios = records_to_arrays(train_records)
        x_val, y_val, _ = records_to_arrays(validation_records)
    except BaselineModelError as exc:
        raise ExperimentRunnerError(str(exc)) from exc

    if set(y_val.tolist()) != {0, 1}:
        raise ExperimentRunnerError(
            f"Validation scenario {candidate.validation_scenarios[0]!r} must contain both classes (0 and 1)."
        )

    validation_scenario = candidate.validation_scenarios[0]
    observations: list[InnerValidationObservation] = []

    for seed in plan.random_seeds:
        for params in model_spec.parameter_combinations:
            try:
                fitted = _fit_configured_model(
                    model_spec.name,
                    x_train,
                    y_train,
                    weighting_regime=weighting_regime,
                    train_scenarios=train_scenarios,
                    params=params,
                    seed=seed,
                    config_path=config_path,
                )
                val_probs = fitted.predict_proba(x_val)
            except BaselineModelError as exc:
                raise ExperimentRunnerError(str(exc)) from exc

            for threshold in plan.candidate_thresholds:
                val_preds = (val_probs >= threshold).astype(np.int64)
                try:
                    metrics = evaluate_hard_predictions(y_val, val_preds)
                except EvaluationError as exc:
                    raise ExperimentRunnerError(str(exc)) from exc
                macro_f1 = metrics["macro_f1"]
                if macro_f1 is None:
                    raise ExperimentRunnerError(
                        f"Undefined validation Macro-F1 for {model_spec.name}/{candidate.candidate_id}."
                    )
                observations.append(
                    InnerValidationObservation(
                        fold_id=fold.fold_id,
                        candidate_id=candidate.candidate_id,
                        validation_scenario=validation_scenario,
                        model_name=model_spec.name,
                        weighting_regime=weighting_regime,
                        seed=seed,
                        hyperparameters=params,
                        threshold=threshold,
                        macro_f1=float(macro_f1),
                        partition_role="validation",
                    )
                )

    return tuple(observations)


def execute_outer_fold_inner_validation(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    model_spec: ModelGridSpec,
    weighting_regime: str,
    records_by_scenario: Mapping[str, Sequence[SupervisedRecord]],
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> tuple[InnerValidationObservation, ...]:
    """Execute all 3 inner candidates for one outer fold while strictly excluding outer test data."""
    if fold not in plan.outer_folds:
        raise ExperimentRunnerError(f"Fold {fold.fold_id!r} is not part of the validated plan.")
    if any(test_s in records_by_scenario for test_s in fold.test_scenarios):
        raise ExperimentRunnerError(
            f"Outer test scenario {fold.test_scenarios!r} must not be passed to inner validation."
        )
    if set(records_by_scenario) != set(fold.outer_train_scenarios):
        raise ExperimentRunnerError(
            f"records_by_scenario keys {sorted(records_by_scenario)!r} must match outer_train_scenarios {list(fold.outer_train_scenarios)!r}."
        )

    all_obs: list[InnerValidationObservation] = []
    for candidate in fold.inner_candidates:
        train_recs: list[SupervisedRecord] = []
        for scenario in candidate.train_scenarios:
            train_recs.extend(records_by_scenario[scenario])
        val_recs = list(records_by_scenario[candidate.validation_scenarios[0]])
        all_obs.extend(
            execute_inner_candidate_validation(
                plan,
                fold,
                candidate,
                model_spec,
                weighting_regime,
                train_recs,
                val_recs,
                config_path=config_path,
            )
        )
    return tuple(all_obs)


@dataclass(frozen=True)
class SeedOuterEvaluation:
    """Outer-test evaluation output for one configured random seed."""

    seed: int
    model_hyperparameters: Mapping[str, object]
    evaluation: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_hyperparameters",
            MappingProxyType(dict(self.model_hyperparameters)),
        )
        object.__setattr__(
            self,
            "evaluation",
            MappingProxyType(dict(self.evaluation)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "model_hyperparameters": dict(self.model_hyperparameters),
            "evaluation": dict(self.evaluation),
        }


@dataclass(frozen=True)
class OuterFoldEvaluationResult:
    """Complete JSON-serializable outer-fold refit and untouched test evaluation result."""

    experiment_id: str
    study_status: str
    split_manifest_sha256: str
    quality_report_sha256: str
    fold_id: str
    outer_train_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    scenario_provenance: Mapping[str, Mapping[str, str]]
    model_name: str
    weighting_regime: str
    selected_hyperparameters: Mapping[str, object]
    fixed_threshold: float
    selected_threshold: float
    inner_validation_mean_macro_f1: float
    outer_train_record_count: int
    outer_test_record_count: int
    outer_test_event_ids_sha256: str
    seeds_evaluated: tuple[int, ...]
    per_seed_results: tuple[SeedOuterEvaluation, ...]
    seed_summary: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_provenance",
            MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in self.scenario_provenance.items()}
            ),
        )
        object.__setattr__(
            self,
            "selected_hyperparameters",
            MappingProxyType(dict(self.selected_hyperparameters)),
        )
        object.__setattr__(
            self,
            "seed_summary",
            MappingProxyType(dict(self.seed_summary)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "study_status": self.study_status,
            "split_manifest_sha256": self.split_manifest_sha256,
            "quality_report_sha256": self.quality_report_sha256,
            "fold_id": self.fold_id,
            "outer_train_scenarios": list(self.outer_train_scenarios),
            "test_scenarios": list(self.test_scenarios),
            "scenario_provenance": {
                k: dict(v) for k, v in self.scenario_provenance.items()
            },
            "model_name": self.model_name,
            "weighting_regime": self.weighting_regime,
            "selected_hyperparameters": dict(self.selected_hyperparameters),
            "fixed_threshold": self.fixed_threshold,
            "selected_threshold": self.selected_threshold,
            "inner_validation_mean_macro_f1": self.inner_validation_mean_macro_f1,
            "outer_train_record_count": self.outer_train_record_count,
            "outer_test_record_count": self.outer_test_record_count,
            "outer_test_event_ids_sha256": self.outer_test_event_ids_sha256,
            "seeds_evaluated": list(self.seeds_evaluated),
            "per_seed_results": [r.to_dict() for r in self.per_seed_results],
            "seed_summary": dict(self.seed_summary),
        }


def _aggregate_seed_metric(values: Sequence[float | None]) -> dict[str, float] | None:
    if not values or any(v is None for v in values):
        return None
    nums = [float(v) for v in values if v is not None]
    mean_val = sum(nums) / len(nums)
    variance = sum((x - mean_val) ** 2 for x in nums) / len(nums)
    return {
        "mean": mean_val,
        "std": math.sqrt(variance),
        "min": min(nums),
        "max": max(nums),
    }


def _summarize_outer_seed_results(
    per_seed_results: Sequence[SeedOuterEvaluation],
) -> dict[str, object]:
    prob_keys = ("pr_auc_attack", "pr_auc_normal", "brier_score")
    hard_keys = (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "normal_precision",
        "normal_recall",
        "normal_f1",
        "normal_false_positive_rate",
    )
    return {
        "probabilistic_metrics": {
            key: _aggregate_seed_metric(
                [r.evaluation["probabilistic_metrics"][key] for r in per_seed_results]
            )
            for key in prob_keys
        },
        "fixed_threshold_metrics": {
            key: _aggregate_seed_metric(
                [r.evaluation["fixed_threshold_metrics"][key] for r in per_seed_results]
            )
            for key in hard_keys
        },
        "selected_threshold_metrics": {
            key: _aggregate_seed_metric(
                [r.evaluation["selected_threshold_metrics"][key] for r in per_seed_results]
            )
            for key in hard_keys
        },
    }


def _hash_event_identities(records: Sequence[SupervisedRecord]) -> str:
    payload = [
        [r.scenario_id, r.source_filename, r.event_id] for r in records
    ]
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def _verify_selection_compatibility(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    model_spec: ModelGridSpec,
    selection: InnerSelectionResult,
) -> None:
    if not isinstance(selection, InnerSelectionResult):
        raise ExperimentRunnerError("selection must be an InnerSelectionResult.")
    if (
        selection.experiment_id != plan.experiment_id
        or selection.split_manifest_sha256 != plan.split_manifest_sha256
        or selection.quality_report_sha256 != plan.quality_report_sha256
    ):
        raise ExperimentRunnerError(
            "InnerSelectionResult provenance does not match the verified NestedLOSOPlan."
        )
    if selection.fold_id != fold.fold_id:
        raise ExperimentRunnerError(
            f"InnerSelectionResult fold_id {selection.fold_id!r} does not match {fold.fold_id!r}."
        )
    if selection.excluded_outer_test_scenarios != fold.test_scenarios:
        raise ExperimentRunnerError(
            "InnerSelectionResult excluded_outer_test_scenarios does not match fold test_scenarios."
        )
    expected_cids = tuple(ic.candidate_id for ic in fold.inner_candidates)
    if selection.inner_candidate_ids != expected_cids:
        raise ExperimentRunnerError(
            "InnerSelectionResult inner_candidate_ids do not match fold inner_candidates."
        )
    if selection.model_name != model_spec.name:
        raise ExperimentRunnerError(
            f"InnerSelectionResult model_name {selection.model_name!r} does not match {model_spec.name!r}."
        )
    if selection.weighting_regime not in model_spec.weighting_regimes:
        raise ExperimentRunnerError(
            f"InnerSelectionResult weighting_regime {selection.weighting_regime!r} is not configured for {model_spec.name!r}."
        )
    if selection.seeds_used != plan.random_seeds:
        raise ExperimentRunnerError(
            f"InnerSelectionResult seeds_used {selection.seeds_used!r} do not match configured random_seeds {plan.random_seeds!r}."
        )
    if selection.selected_threshold not in plan.candidate_thresholds:
        raise ExperimentRunnerError(
            f"InnerSelectionResult selected_threshold {selection.selected_threshold!r} is not in configured candidate_thresholds."
        )
    valid_param_keys = {
        _canonical_param_key(combo) for combo in model_spec.parameter_combinations
    }
    if _canonical_param_key(selection.selected_hyperparameters) not in valid_param_keys:
        raise ExperimentRunnerError(
            "InnerSelectionResult selected_hyperparameters are not in configured hyperparameter_grid."
        )


def execute_outer_fold_refit_and_evaluation(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    model_spec: ModelGridSpec,
    selection: InnerSelectionResult,
    outer_train_records: Sequence[SupervisedRecord],
    outer_test_records: Sequence[SupervisedRecord],
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> OuterFoldEvaluationResult:
    """Refit an inner-selected configuration on the 3 outer training scenarios and evaluate once on outer test."""
    if fold not in plan.outer_folds:
        raise ExperimentRunnerError(f"Fold {fold.fold_id!r} is not part of the validated plan.")
    if model_spec not in plan.model_specs:
        raise ExperimentRunnerError(f"Model {model_spec.name!r} is not part of the validated plan.")
    if model_spec.name not in _TUNABLE_MODEL_NAMES:
        raise ExperimentRunnerError(
            f"Model {model_spec.name!r} is not supported by outer-fold refit executor."
        )
    _verify_selection_compatibility(plan, fold, model_spec, selection)

    train_ids = _validate_partition_records(
        outer_train_records,
        expected_scenarios=fold.outer_train_scenarios,
        forbidden_scenarios=frozenset(fold.test_scenarios),
        partition_name="outer_train_records",
    )
    test_ids = _validate_partition_records(
        outer_test_records,
        expected_scenarios=fold.test_scenarios,
        forbidden_scenarios=frozenset(fold.outer_train_scenarios),
        partition_name="outer_test_records",
    )
    if not train_ids.isdisjoint(test_ids):
        raise ExperimentRunnerError(
            "Record identity overlap detected between outer_train_records and outer_test_records."
        )

    try:
        x_train, y_train, train_scenarios = records_to_arrays(outer_train_records)
        x_test, y_test, _ = records_to_arrays(outer_test_records)
    except BaselineModelError as exc:
        raise ExperimentRunnerError(str(exc)) from exc

    has_attack_categories = any(
        r.label == 1 and bool(r.attack_categories) for r in outer_test_records
    )
    test_categories = (
        [r.attack_categories for r in outer_test_records] if has_attack_categories else None
    )

    per_seed_results: list[SeedOuterEvaluation] = []
    for seed in plan.random_seeds:
        try:
            fitted = _fit_configured_model(
                model_spec.name,
                x_train,
                y_train,
                weighting_regime=selection.weighting_regime,
                train_scenarios=train_scenarios,
                params=selection.selected_hyperparameters,
                seed=seed,
                config_path=config_path,
            )
            test_probs = fitted.predict_proba(x_test)
        except BaselineModelError as exc:
            raise ExperimentRunnerError(str(exc)) from exc

        try:
            evaluation = evaluate_probabilistic_predictions(
                y_test,
                test_probs,
                selected_threshold=selection.selected_threshold,
                fixed_threshold=plan.fixed_threshold,
                attack_categories=test_categories,
            )
        except EvaluationError as exc:
            raise ExperimentRunnerError(str(exc)) from exc

        per_seed_results.append(
            SeedOuterEvaluation(
                seed=seed,
                model_hyperparameters=fitted.hyperparameters,
                evaluation=evaluation,
            )
        )

    fold_scenarios = fold.outer_train_scenarios + fold.test_scenarios
    fold_provenance = {
        s: dict(plan.scenario_provenance[s]) for s in sorted(fold_scenarios)
    }

    return OuterFoldEvaluationResult(
        experiment_id=plan.experiment_id,
        study_status=plan.study_status,
        split_manifest_sha256=plan.split_manifest_sha256,
        quality_report_sha256=plan.quality_report_sha256,
        fold_id=fold.fold_id,
        outer_train_scenarios=fold.outer_train_scenarios,
        test_scenarios=fold.test_scenarios,
        scenario_provenance=fold_provenance,
        model_name=model_spec.name,
        weighting_regime=selection.weighting_regime,
        selected_hyperparameters=selection.selected_hyperparameters,
        fixed_threshold=plan.fixed_threshold,
        selected_threshold=selection.selected_threshold,
        inner_validation_mean_macro_f1=selection.mean_validation_macro_f1,
        outer_train_record_count=len(outer_train_records),
        outer_test_record_count=len(outer_test_records),
        outer_test_event_ids_sha256=_hash_event_identities(outer_test_records),
        seeds_evaluated=plan.random_seeds,
        per_seed_results=tuple(per_seed_results),
        seed_summary=_summarize_outer_seed_results(per_seed_results),
    )


class ScenarioRecordProvider(Protocol):
    """Supplies verified SupervisedRecords for one requested scenario ID."""

    def load_scenario_records(self, scenario_id: str) -> Sequence[SupervisedRecord]: ...


@dataclass(frozen=True)
class NestedLOSOFoldRunResult:
    """JSON-serializable end-to-end result for one outer LOSO fold, model, and weighting regime."""

    experiment_id: str
    study_status: str
    split_manifest_sha256: str
    quality_report_sha256: str
    fold_id: str
    model_name: str
    weighting_regime: str
    inner_selection: InnerSelectionResult
    outer_evaluation: OuterFoldEvaluationResult

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "study_status": self.study_status,
            "split_manifest_sha256": self.split_manifest_sha256,
            "quality_report_sha256": self.quality_report_sha256,
            "fold_id": self.fold_id,
            "model_name": self.model_name,
            "weighting_regime": self.weighting_regime,
            "selected_hyperparameters": dict(self.inner_selection.selected_hyperparameters),
            "fixed_threshold": self.outer_evaluation.fixed_threshold,
            "selected_threshold": self.inner_selection.selected_threshold,
            "inner_selection": self.inner_selection.to_dict(),
            "outer_evaluation": self.outer_evaluation.to_dict(),
        }


def _load_validated_provider_scenario(
    record_provider: ScenarioRecordProvider,
    scenario_id: str,
    *,
    forbidden_scenarios: frozenset[str],
) -> tuple[SupervisedRecord, ...]:
    try:
        raw_records = record_provider.load_scenario_records(scenario_id)
    except Exception as exc:
        raise ExperimentRunnerError(
            f"Record provider failed while loading scenario {scenario_id!r}: {exc}"
        ) from exc
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ExperimentRunnerError(
            f"Record provider for {scenario_id!r} must return a sequence of SupervisedRecord."
        )
    records = tuple(raw_records)
    _validate_partition_records(
        records,
        expected_scenarios=(scenario_id,),
        forbidden_scenarios=forbidden_scenarios,
        partition_name=f"provider[{scenario_id}]",
    )
    return records


def run_single_outer_fold_experiment(
    *,
    fold_id: str,
    model_name: str,
    weighting_regime: str,
    record_provider: ScenarioRecordProvider,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    base_dir: str | Path | None = None,
    plan: NestedLOSOPlan | None = None,
) -> NestedLOSOFoldRunResult:
    """Orchestrate inner validation, selection, outer refit, and test evaluation for one outer fold."""
    active_plan = (
        build_nested_loso_plan(config_path, base_dir=base_dir) if plan is None else plan
    )
    fold = next((f for f in active_plan.outer_folds if f.fold_id == fold_id), None)
    if fold is None:
        raise ExperimentRunnerError(f"Unknown fold_id {fold_id!r} in validated plan.")
    model_spec = next((m for m in active_plan.model_specs if m.name == model_name), None)
    if model_spec is None or model_spec.name not in _TUNABLE_MODEL_NAMES:
        raise ExperimentRunnerError(
            f"Model {model_name!r} is not a supported tunable model in validated plan."
        )
    if weighting_regime not in model_spec.weighting_regimes:
        raise ExperimentRunnerError(
            f"Weighting regime {weighting_regime!r} is not configured for {model_name!r}."
        )

    # Phase 1: Load ONLY outer-train scenarios for inner validation and selection.
    all_scenarios_set = frozenset(active_plan.scenario_ids)
    outer_train_by_scenario: dict[str, tuple[SupervisedRecord, ...]] = {}
    for scenario_id in fold.outer_train_scenarios:
        outer_train_by_scenario[scenario_id] = _load_validated_provider_scenario(
            record_provider,
            scenario_id,
            forbidden_scenarios=all_scenarios_set - {scenario_id},
        )

    inner_observations = execute_outer_fold_inner_validation(
        active_plan,
        fold,
        model_spec,
        weighting_regime,
        outer_train_by_scenario,
        config_path=config_path,
    )
    selection = select_inner_validation_configuration(
        active_plan,
        fold,
        model_spec,
        weighting_regime,
        inner_observations,
    )

    # Phase 2: Only after inner selection is frozen, load the held-out outer test scenario.
    outer_test_records: list[SupervisedRecord] = []
    for test_scenario in fold.test_scenarios:
        outer_test_records.extend(
            _load_validated_provider_scenario(
                record_provider,
                test_scenario,
                forbidden_scenarios=all_scenarios_set - {test_scenario},
            )
        )

    outer_train_records = [
        rec
        for scenario_id in fold.outer_train_scenarios
        for rec in outer_train_by_scenario[scenario_id]
    ]
    outer_evaluation = execute_outer_fold_refit_and_evaluation(
        active_plan,
        fold,
        model_spec,
        selection,
        outer_train_records,
        outer_test_records,
        config_path=config_path,
    )

    return NestedLOSOFoldRunResult(
        experiment_id=active_plan.experiment_id,
        study_status=active_plan.study_status,
        split_manifest_sha256=active_plan.split_manifest_sha256,
        quality_report_sha256=active_plan.quality_report_sha256,
        fold_id=fold.fold_id,
        model_name=model_spec.name,
        weighting_regime=weighting_regime,
        inner_selection=selection,
        outer_evaluation=outer_evaluation,
    )
