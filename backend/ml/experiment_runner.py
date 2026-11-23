from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .schema import FEATURE_SCHEMA_VERSION
from .training_data import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    PROJECT_ROOT,
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
