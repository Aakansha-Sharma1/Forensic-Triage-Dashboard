from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .baselines import (
    BaselineModelError,
    train_decision_tree_baseline,
    train_dummy_baseline,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from .dataset import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESEARCH_DATA_ROOT
from .evaluation import (
    EvaluationError,
    evaluate_hard_predictions,
    evaluate_probabilistic_predictions,
    evaluate_probabilities,
)
from .experiment_runner import (
    ExperimentRunnerError,
    InnerValidationObservation,
    ModelGridSpec,
    NestedLOSOPlan,
    OuterFoldPlan,
    build_nested_loso_plan,
    select_inner_validation_configuration,
)
from .rule_baseline import (
    RuleBaselineError,
    RuleBaselineInputEvent,
    run_rule_baseline,
)
from .schema import CanonicalEvent
from .training_data import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    PROJECT_ROOT,
    ScenarioLoadSummary,
    SupervisedRecord,
    TrainingDataError,
    iter_scenario_supervised_records,
    load_and_verify_experiment_config,
)


EXPERIMENTS_OUTPUT_DIR = RESEARCH_DATA_ROOT / "experiments"
DEFAULT_CHECKPOINT_DIR = (
    EXPERIMENTS_OUTPUT_DIR / "checkpoints" / "ait-lds-v2.1-4scenario-loso-exploratory-v1"
)
DEFAULT_BENCHMARK_OUTPUT_PATH = (
    EXPERIMENTS_OUTPUT_DIR / "ait_lds_v2_1_4scenario_loso_exploratory_v1.json"
)
BENCHMARK_REPORT_SCHEMA_VERSION = "1.0.0"
CHECKPOINT_SCHEMA_VERSION = "1.0.0"

_HARD_METRIC_KEYS: tuple[str, ...] = (
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
_PROB_METRIC_KEYS: tuple[str, ...] = (
    "pr_auc_attack",
    "pr_auc_normal",
    "brier_score",
)


class BenchmarkError(ValueError):
    """Raised when benchmark loading, isolation, execution, or artifact storage fails."""


class BenchmarkIncompleteError(BenchmarkError):
    """Raised when a resumable run stops before all configured experiment units complete."""


@dataclass(frozen=True)
class CompactScenarioDataset:
    """Bounded-memory NumPy representation of one verified scenario's known-labelled records."""

    scenario_id: str
    features: np.ndarray
    labels: np.ndarray
    attack_categories: tuple[tuple[str, ...], ...]
    event_ids_sha256: str
    load_summary: ScenarioLoadSummary


class VerifiedDatasetScenarioProvider:
    """ScenarioRecordProvider backed by provenance-verified training_data.py streaming."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
        *,
        base_dir: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.base_dir = PROJECT_ROOT if base_dir is None else Path(base_dir).resolve()
        try:
            self.config = load_and_verify_experiment_config(
                self.config_path, base_dir=self.base_dir
            )
        except TrainingDataError as exc:
            raise BenchmarkError(str(exc)) from exc
        self.summaries: dict[str, ScenarioLoadSummary] = {}

    def load_scenario_records(self, scenario_id: str) -> tuple[SupervisedRecord, ...]:
        try:
            records = tuple(
                iter_scenario_supervised_records(
                    self.config,
                    scenario_id,
                    base_dir=self.base_dir,
                    summary_out=self.summaries,
                )
            )
        except TrainingDataError as exc:
            raise BenchmarkError(str(exc)) from exc
        return records


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_scenario_dataset(
    config: Mapping[str, object],
    scenario_id: str,
    *,
    base_dir: str | Path | None = None,
) -> CompactScenarioDataset:
    """Stream one verified scenario into compact NumPy arrays without retaining unknown records."""
    resolved_base = PROJECT_ROOT if base_dir is None else Path(base_dir).resolve()
    summaries: dict[str, ScenarioLoadSummary] = {}
    feature_rows: list[list[float]] = []
    label_rows: list[int] = []
    categories: list[tuple[str, ...]] = []
    seen_identities: set[tuple[str, str]] = set()
    id_digest = hashlib.sha256()

    try:
        for rec in iter_scenario_supervised_records(
            config, scenario_id, base_dir=resolved_base, summary_out=summaries
        ):
            ident = (rec.source_filename, rec.event_id)
            if ident in seen_identities:
                raise BenchmarkError(
                    f"{scenario_id}: duplicate supervised event identity {ident!r}."
                )
            seen_identities.add(ident)
            id_digest.update(
                f"{rec.scenario_id}\x1f{rec.source_filename}\x1f{rec.event_id}\n".encode("utf-8")
            )
            feature_rows.append(
                [math.nan if v is None else float(v) for v in rec.features]
            )
            label_rows.append(rec.label)
            categories.append(rec.attack_categories)
    except TrainingDataError as exc:
        raise BenchmarkError(str(exc)) from exc

    if not feature_rows:
        raise BenchmarkError(f"{scenario_id}: no verified known-labelled records found.")

    return CompactScenarioDataset(
        scenario_id=scenario_id,
        features=np.asarray(feature_rows, dtype=np.float64),
        labels=np.asarray(label_rows, dtype=np.int64),
        attack_categories=tuple(categories),
        event_ids_sha256=id_digest.hexdigest(),
        load_summary=summaries[scenario_id],
    )


def load_verified_rule_events(
    config: Mapping[str, object],
    scenario_id: str,
    *,
    base_dir: str | Path | None = None,
) -> tuple[RuleBaselineInputEvent, ...]:
    """Stream verified known-labelled CanonicalEvents for one scenario's rule-baseline evaluation."""
    resolved_base = PROJECT_ROOT if base_dir is None else Path(base_dir).resolve()
    provenance = config.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise BenchmarkError("Experiment config is missing dataset_provenance.")
    scenarios_cfg = provenance.get("scenarios")
    if not isinstance(scenarios_cfg, Mapping) or scenario_id not in scenarios_cfg:
        raise BenchmarkError(f"Scenario {scenario_id!r} not in experiment config.")
    scenario_ref = scenarios_cfg[scenario_id]
    raw_manifest_path = Path(str(scenario_ref["label_manifest_path"]))
    manifest_path = (
        raw_manifest_path
        if raw_manifest_path.is_absolute()
        else (resolved_base / raw_manifest_path).resolve()
    )
    if _sha256_file(manifest_path) != scenario_ref["label_manifest_sha256"]:
        raise BenchmarkError(f"{scenario_id}: label manifest SHA-256 mismatch.")
    label_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labelled_path = manifest_path.parent / label_manifest["labelled_output"]["filename"]
    if _sha256_file(labelled_path) != scenario_ref["labelled_records_sha256"]:
        raise BenchmarkError(f"{scenario_id}: labelled_events.jsonl SHA-256 mismatch.")

    coverage_by_source = {
        entry["source_filename"]: entry["ground_truth_coverage"]
        for entry in label_manifest.get("per_source_distributions", [])
        if isinstance(entry, dict) and "source_filename" in entry
    }

    rule_events: list[RuleBaselineInputEvent] = []
    with labelled_path.open("rb") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            row = json.loads(raw_line.decode("utf-8"))
            label = row.get("research_label")
            if label is None:
                continue
            if type(label) is not int or label not in (0, 1):
                raise BenchmarkError(f"{scenario_id} line {line_num}: invalid research_label.")
            prov = row["provenance"]
            ev_dict = row["event"]
            source_filename = prov["source_filename"]
            coverage = coverage_by_source.get(source_filename)
            if coverage != "verified":
                raise BenchmarkError(
                    f"{scenario_id} line {line_num}: known label from non-verified source {source_filename!r}."
                )
            ts_text = str(ev_dict["timestamp"]).strip()
            if ts_text.endswith("Z"):
                ts_text = ts_text[:-1] + "+00:00"
            ts = datetime.fromisoformat(ts_text).astimezone(timezone.utc)
            canonical = CanonicalEvent(
                case_id=ev_dict["case_id"],
                evidence_id=ev_dict["evidence_id"],
                event_id=ev_dict["event_id"],
                timestamp=ts,
                source_ip=ev_dict.get("source_ip"),
                event_type=ev_dict.get("event_type"),
                resource=ev_dict.get("resource"),
                response_status=ev_dict.get("response_status"),
                transfer_size=ev_dict.get("transfer_size"),
                original_evidence_reference=ev_dict["original_evidence_reference"],
            )
            gt = row.get("ground_truth")
            cats = (
                tuple(gt["labels"])
                if label == 1 and isinstance(gt, dict) and isinstance(gt.get("labels"), list)
                else ()
            )
            rule_events.append(
                RuleBaselineInputEvent(
                    event=canonical,
                    scenario_id=scenario_id,
                    source_filename=source_filename,
                    source_line_number=int(prov["source_line_number"]),
                    label=label,
                    label_name=str(row["research_label_name"]),
                    ground_truth_coverage=coverage,
                    attack_categories=cats,
                )
            )
    if not rule_events:
        raise BenchmarkError(f"{scenario_id}: no verified rule-baseline events found.")
    return tuple(rule_events)


@contextmanager
def _isolated_forensic_env() -> Iterator[Path]:
    """Isolate environment variables so lazy backend.main imports never touch operational forensic.db."""
    env_keys = (
        "FORENSIC_INSIGHT_DB_PATH",
        "FORENSIC_INSIGHT_UPLOAD_DIR",
        "FORENSIC_INSIGHT_WORKING_DIR",
        "FORENSIC_INSIGHT_REPORTS_DIR",
    )
    previous = {key: os.environ.get(key) for key in env_keys}
    with tempfile.TemporaryDirectory(prefix="forensic-rule-research-db-") as tmp_dir:
        root = Path(tmp_dir)
        os.environ["FORENSIC_INSIGHT_DB_PATH"] = str(root / "research_isolated.db")
        os.environ["FORENSIC_INSIGHT_UPLOAD_DIR"] = str(root / "uploads" / "original")
        os.environ["FORENSIC_INSIGHT_WORKING_DIR"] = str(root / "uploads" / "working")
        os.environ["FORENSIC_INSIGHT_REPORTS_DIR"] = str(root / "reports")
        try:
            yield root
        finally:
            for key, old_val in previous.items():
                if old_val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_val


def _aggregate_stats(values: Sequence[float | None]) -> dict[str, float] | None:
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


def _summarize_seed_blocks(per_seed: Sequence[Mapping[str, Any]], *, has_selected: bool) -> dict[str, object]:
    summary: dict[str, object] = {
        "probabilistic_metrics": {
            key: _aggregate_stats([r["evaluation"]["probabilistic_metrics"][key] for r in per_seed])
            for key in _PROB_METRIC_KEYS
        },
        "fixed_threshold_metrics": {
            key: _aggregate_stats([r["evaluation"]["fixed_threshold_metrics"][key] for r in per_seed])
            for key in _HARD_METRIC_KEYS
        },
    }
    if has_selected:
        summary["selected_threshold_metrics"] = {
            key: _aggregate_stats([r["evaluation"]["selected_threshold_metrics"][key] for r in per_seed])
            for key in _HARD_METRIC_KEYS
        }
    return summary


def _summarize_across_folds_probabilistic(
    fold_results: Sequence[Mapping[str, Any]],
    *,
    has_selected: bool,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "probabilistic_metrics": {
            key: _aggregate_stats(
                [
                    (f["seed_summary"]["probabilistic_metrics"][key] or {}).get("mean")
                    if f["seed_summary"]["probabilistic_metrics"][key] is not None
                    else None
                    for f in fold_results
                ]
            )
            for key in _PROB_METRIC_KEYS
        },
        "fixed_threshold_metrics": {
            key: _aggregate_stats(
                [
                    (f["seed_summary"]["fixed_threshold_metrics"][key] or {}).get("mean")
                    if f["seed_summary"]["fixed_threshold_metrics"][key] is not None
                    else None
                    for f in fold_results
                ]
            )
            for key in _HARD_METRIC_KEYS
        },
    }
    if has_selected:
        summary["selected_threshold_metrics"] = {
            key: _aggregate_stats(
                [
                    (f["seed_summary"]["selected_threshold_metrics"][key] or {}).get("mean")
                    if f["seed_summary"]["selected_threshold_metrics"][key] is not None
                    else None
                    for f in fold_results
                ]
            )
            for key in _HARD_METRIC_KEYS
        }
    return summary


def run_rule_baseline_outer_fold(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    test_rule_events: Sequence[RuleBaselineInputEvent],
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    detector=None,
) -> dict[str, object]:
    """Evaluate RuleBaseline_R_A_to_D on one outer test scenario in an isolated DB environment."""
    expected_test = fold.test_scenarios[0]
    if not test_rule_events or any(e.scenario_id != expected_test for e in test_rule_events):
        raise BenchmarkError(
            f"{fold.fold_id}: rule events must be non-empty and belong strictly to {expected_test!r}."
        )
    try:
        with _isolated_forensic_env():
            res = run_rule_baseline(
                test_rule_events, config_path=config_path, detector=detector
            )
    except RuleBaselineError as exc:
        raise BenchmarkError(str(exc)) from exc

    rule_trigger_counts = Counter(
        rid for pred in res.predictions for rid in pred.matched_rule_ids
    )
    return {
        "fold_id": fold.fold_id,
        "test_scenarios": list(fold.test_scenarios),
        "outer_test_record_count": len(test_rule_events),
        "rule_registry_fingerprint": res.rule_registry_fingerprint,
        "rule_versions": dict(res.rule_versions),
        "rule_trigger_counts": dict(sorted(rule_trigger_counts.items())),
        "evaluation": dict(res.evaluation),
        "not_applicable_metrics": dict(res.not_applicable_metrics),
        "methodological_caveats": list(res.methodological_caveats),
    }


def run_dummy_outer_fold(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    model_spec: ModelGridSpec,
    outer_train_datasets: Mapping[str, CompactScenarioDataset],
    outer_test_dataset: CompactScenarioDataset,
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> dict[str, object]:
    """Select DummyClassifier strategy via inner validation and evaluate across all seeds on outer test."""
    if model_spec.name != "DummyClassifier":
        raise BenchmarkError(f"Expected DummyClassifier model_spec, got {model_spec.name!r}.")
    if set(outer_train_datasets) != set(fold.outer_train_scenarios):
        raise BenchmarkError("outer_train_datasets must match fold.outer_train_scenarios exactly.")
    if outer_test_dataset.scenario_id != fold.test_scenarios[0]:
        raise BenchmarkError("outer_test_dataset must match fold.test_scenarios[0].")

    strategies = [str(combo["strategy"]) for combo in model_spec.parameter_combinations]
    scores_by_strategy: dict[str, list[float]] = {s: [] for s in strategies}

    for candidate in fold.inner_candidates:
        x_tr = np.vstack([outer_train_datasets[s].features for s in candidate.train_scenarios])
        y_tr = np.concatenate([outer_train_datasets[s].labels for s in candidate.train_scenarios])
        val_ds = outer_train_datasets[candidate.validation_scenarios[0]]
        for seed in plan.random_seeds:
            for strategy in strategies:
                try:
                    fitted = train_dummy_baseline(
                        x_tr, y_tr, strategy=strategy, random_state=seed, config_path=config_path
                    )
                    preds = fitted.predict(val_ds.features)
                    metrics = evaluate_hard_predictions(val_ds.labels, preds)
                except (BaselineModelError, EvaluationError) as exc:
                    raise BenchmarkError(str(exc)) from exc
                if metrics["macro_f1"] is None:
                    raise BenchmarkError("Undefined validation Macro-F1 for DummyClassifier.")
                scores_by_strategy[strategy].append(float(metrics["macro_f1"]))

    ranked = sorted(
        [
            (
                -round(sum(scores_by_strategy[s]) / len(scores_by_strategy[s]), 12),
                idx,
                s,
                sum(scores_by_strategy[s]) / len(scores_by_strategy[s]),
            )
            for idx, s in enumerate(strategies)
        ],
        key=lambda item: (item[0], item[1]),
    )
    _, _, selected_strategy, mean_val_f1 = ranked[0]

    # Refit on 3 outer train scenarios and evaluate once on untouched outer test
    x_outer_tr = np.vstack([outer_train_datasets[s].features for s in fold.outer_train_scenarios])
    y_outer_tr = np.concatenate([outer_train_datasets[s].labels for s in fold.outer_train_scenarios])
    has_cats = any(
        outer_test_dataset.labels[i] == 1 and bool(outer_test_dataset.attack_categories[i])
        for i in range(len(outer_test_dataset.labels))
    )
    test_cats = outer_test_dataset.attack_categories if has_cats else None

    per_seed_results: list[dict[str, object]] = []
    for seed in plan.random_seeds:
        try:
            fitted = train_dummy_baseline(
                x_outer_tr,
                y_outer_tr,
                strategy=selected_strategy,
                random_state=seed,
                config_path=config_path,
            )
            hard_preds = fitted.predict(outer_test_dataset.features)
            probs = fitted.predict_proba(outer_test_dataset.features)
            hard_eval = evaluate_hard_predictions(
                outer_test_dataset.labels, hard_preds, attack_categories=test_cats
            )
            prob_eval = evaluate_probabilities(outer_test_dataset.labels, probs)
        except (BaselineModelError, EvaluationError) as exc:
            raise BenchmarkError(str(exc)) from exc
        per_seed_results.append(
            {
                "seed": seed,
                "model_hyperparameters": {"strategy": selected_strategy},
                "evaluation": {
                    "probabilistic_metrics": prob_eval,
                    "fixed_threshold_metrics": {"threshold": None, **hard_eval},
                    "selected_threshold_metrics": None,
                },
            }
        )

    return {
        "fold_id": fold.fold_id,
        "outer_train_scenarios": list(fold.outer_train_scenarios),
        "test_scenarios": list(fold.test_scenarios),
        "model_name": "DummyClassifier",
        "weighting_regime": None,
        "inner_selection": {
            "selected_strategy": selected_strategy,
            "mean_validation_macro_f1": mean_val_f1,
            "inner_candidate_ids": [ic.candidate_id for ic in fold.inner_candidates],
            "seeds_used": list(plan.random_seeds),
        },
        "outer_train_record_count": int(len(y_outer_tr)),
        "outer_test_record_count": int(len(outer_test_dataset.labels)),
        "outer_test_event_ids_sha256": outer_test_dataset.event_ids_sha256,
        "per_seed_results": per_seed_results,
        "seed_summary": _summarize_seed_blocks(per_seed_results, has_selected=False),
    }


def _fit_compact_tunable(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    weighting_regime: str,
    train_scenarios: tuple[str, ...],
    params: Mapping[str, object],
    seed: int,
    config_path: str | Path,
    rf_n_jobs: int | None = None,
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
            n_jobs=rf_n_jobs,
            config_path=config_path,
        )
    raise BenchmarkError(f"Unsupported tunable model: {model_name!r}")


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ).encode("utf-8")


class _CheckpointStore:
    """Atomic, provenance-verified per-unit checkpoint store inside the experiments directory."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        allowed_root: str | Path,
        provenance: Mapping[str, object],
        max_new_units: int | None = None,
    ) -> None:
        resolved_root = Path(allowed_root).resolve()
        resolved_dir = Path(checkpoint_dir).resolve()
        if not resolved_dir.is_relative_to(resolved_root) or resolved_dir == resolved_root:
            raise BenchmarkError(
                f"Checkpoint directory {resolved_dir} must lie inside {resolved_root}."
            )
        for forbidden in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
            resolved_forbidden = forbidden.resolve()
            if resolved_dir.is_relative_to(resolved_forbidden):
                raise BenchmarkError(
                    f"Refusing to store checkpoints inside {resolved_forbidden}."
                )
        if max_new_units is not None and (
            isinstance(max_new_units, bool) or not isinstance(max_new_units, int) or max_new_units < 0
        ):
            raise BenchmarkError(f"max_new_units must be a non-negative int, got {max_new_units!r}.")

        self.checkpoint_dir = resolved_dir
        self.provenance = dict(provenance)
        self.max_new_units = max_new_units
        self.resumed_count = 0
        self.computed_count = 0

    def _unit_path(self, filename: str) -> Path:
        path = (self.checkpoint_dir / filename).resolve()
        if path.parent != self.checkpoint_dir:
            raise BenchmarkError(f"Invalid checkpoint filename: {filename!r}")
        return path

    def load_if_exists(
        self,
        filename: str,
        *,
        unit_type: str,
        unit_key: Mapping[str, object],
    ) -> dict[str, Any] | None:
        path = self._unit_path(filename)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"Corrupted checkpoint file {path.name}: {exc}") from exc
        if not isinstance(envelope, dict):
            raise BenchmarkError(f"Corrupted checkpoint file {path.name}: expected JSON object.")
        if envelope.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise BenchmarkError(
                f"Checkpoint {path.name}: schema version mismatch {envelope.get('checkpoint_schema_version')!r}."
            )
        if envelope.get("unit_type") != unit_type:
            raise BenchmarkError(
                f"Checkpoint {path.name}: unit_type mismatch {envelope.get('unit_type')!r} != {unit_type!r}."
            )
        if envelope.get("provenance") != self.provenance:
            raise BenchmarkError(
                f"Checkpoint {path.name}: config or dataset provenance mismatch."
            )
        if envelope.get("unit_key") != dict(unit_key):
            raise BenchmarkError(
                f"Checkpoint {path.name}: unit_key mismatch."
            )
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise BenchmarkError(f"Checkpoint {path.name}: missing or invalid payload.")
        expected_sha = _sha256_bytes(_canonical_json_bytes(payload))
        if envelope.get("payload_sha256") != expected_sha:
            raise BenchmarkError(f"Checkpoint {path.name}: payload SHA-256 integrity mismatch.")
        self.resumed_count += 1
        return payload

    def check_budget_before_compute(self, label: str) -> None:
        if self.max_new_units is not None and self.computed_count >= self.max_new_units:
            raise BenchmarkIncompleteError(
                f"Stopped before {label}: reached max_new_units={self.max_new_units} "
                f"(resumed={self.resumed_count}, computed={self.computed_count})."
            )

    def save(
        self,
        filename: str,
        *,
        unit_type: str,
        unit_key: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> Path:
        path = self._unit_path(filename)
        payload_dict = dict(payload)
        envelope = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "unit_type": unit_type,
            "provenance": self.provenance,
            "unit_key": dict(unit_key),
            "payload_sha256": _sha256_bytes(_canonical_json_bytes(payload_dict)),
            "payload": payload_dict,
        }
        content = (
            json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".ckpt.", suffix=".tmp", dir=self.checkpoint_dir
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        self.computed_count += 1
        return path


def _count_total_units(plan: NestedLOSOPlan) -> int:
    total = 0
    for spec in plan.model_specs:
        if spec.name in ("RuleBaseline_R_A_to_D", "DummyClassifier"):
            total += len(plan.outer_folds)
        else:
            for _ in spec.weighting_regimes:
                for fold in plan.outer_folds:
                    total += (
                        len(fold.inner_candidates)
                        * len(spec.parameter_combinations)
                        * len(plan.random_seeds)
                    )
                    total += len(plan.random_seeds)
    return total


def run_tunable_model_outer_fold_compact(
    plan: NestedLOSOPlan,
    fold: OuterFoldPlan,
    model_spec: ModelGridSpec,
    weighting_regime: str,
    outer_train_datasets: Mapping[str, CompactScenarioDataset],
    outer_test_dataset: CompactScenarioDataset,
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    rf_n_jobs: int | None = None,
    checkpoint_store: _CheckpointStore | None = None,
    progress_state: dict[str, int] | None = None,
    progress: bool = False,
) -> dict[str, object]:
    """Run inner validation, joint selection, outer refit, and outer test evaluation on compact arrays."""
    if set(outer_train_datasets) != set(fold.outer_train_scenarios):
        raise BenchmarkError("outer_train_datasets must match fold.outer_train_scenarios exactly.")
    if outer_test_dataset.scenario_id != fold.test_scenarios[0]:
        raise BenchmarkError("outer_test_dataset must match fold.test_scenarios[0].")

    def _emit_progress(label: str, status: str, elapsed: float | None = None) -> None:
        if progress_state is not None:
            progress_state["done"] += 1
            if progress:
                suffix = f" ({status}, {elapsed:.1f}s)" if elapsed is not None else f" ({status})"
                print(
                    f"[{progress_state['done']}/{progress_state['total']}] {label}{suffix}",
                    flush=True,
                )

    inner_observations: list[InnerValidationObservation] = []
    for candidate in fold.inner_candidates:
        x_tr: np.ndarray | None = None
        y_tr: np.ndarray | None = None
        scen_tr: tuple[str, ...] | None = None
        val_ds = outer_train_datasets[candidate.validation_scenarios[0]]
        for param_idx, params in enumerate(model_spec.parameter_combinations):
            cached_val_probs: np.ndarray | None = None
            for seed in plan.random_seeds:
                ckpt_name = (
                    f"inner__{model_spec.name}__{weighting_regime}__{fold.fold_id}"
                    f"__{candidate.candidate_id}__p{param_idx}__s{seed}.json"
                )
                unit_key = {
                    "model_name": model_spec.name,
                    "weighting_regime": weighting_regime,
                    "fold_id": fold.fold_id,
                    "candidate_id": candidate.candidate_id,
                    "validation_scenario": candidate.validation_scenarios[0],
                    "param_index": param_idx,
                    "hyperparameters": dict(params),
                    "seed": seed,
                    "candidate_thresholds": list(plan.candidate_thresholds),
                }
                label = (
                    f"{model_spec.name} | {weighting_regime} | {fold.fold_id} | "
                    f"inner {candidate.candidate_id} p{param_idx} s{seed}"
                )
                loaded = (
                    checkpoint_store.load_if_exists(
                        ckpt_name, unit_type="inner_fit", unit_key=unit_key
                    )
                    if checkpoint_store is not None
                    else None
                )
                if loaded is not None:
                    raw_obs = loaded.get("observations")
                    if not isinstance(raw_obs, list) or len(raw_obs) != len(plan.candidate_thresholds):
                        raise BenchmarkError(
                            f"Checkpoint {ckpt_name}: incomplete inner observations."
                        )
                    for item in raw_obs:
                        inner_observations.append(
                            InnerValidationObservation(
                                fold_id=str(item["fold_id"]),
                                candidate_id=str(item["candidate_id"]),
                                validation_scenario=str(item["validation_scenario"]),
                                model_name=str(item["model_name"]),
                                weighting_regime=str(item["weighting_regime"]),
                                seed=int(item["seed"]),
                                hyperparameters=dict(item["hyperparameters"]),
                                threshold=float(item["threshold"]),
                                macro_f1=float(item["macro_f1"]),
                                partition_role=str(item.get("partition_role", "validation")),
                            )
                        )
                    _emit_progress(label, "resumed")
                    continue

                if checkpoint_store is not None:
                    checkpoint_store.check_budget_before_compute(label)

                t0 = time.time()
                if x_tr is None or y_tr is None or scen_tr is None:
                    x_tr = np.vstack(
                        [outer_train_datasets[s].features for s in candidate.train_scenarios]
                    )
                    y_tr = np.concatenate(
                        [outer_train_datasets[s].labels for s in candidate.train_scenarios]
                    )
                    scen_tr = tuple(
                        s
                        for s in candidate.train_scenarios
                        for _ in range(len(outer_train_datasets[s].labels))
                    )
                if not model_spec.stochastic and cached_val_probs is not None:
                    val_probs = cached_val_probs
                else:
                    fitted = _fit_compact_tunable(
                        model_spec.name,
                        x_tr,
                        y_tr,
                        weighting_regime=weighting_regime,
                        train_scenarios=scen_tr,
                        params=params,
                        seed=seed,
                        config_path=config_path,
                        rf_n_jobs=rf_n_jobs,
                    )
                    val_probs = fitted.predict_proba(val_ds.features)
                    if not model_spec.stochastic:
                        cached_val_probs = val_probs

                unit_obs_dicts: list[dict[str, object]] = []
                for threshold in plan.candidate_thresholds:
                    val_preds = (val_probs >= threshold).astype(np.int64)
                    metrics = evaluate_hard_predictions(val_ds.labels, val_preds)
                    if metrics["macro_f1"] is None:
                        raise BenchmarkError("Undefined validation Macro-F1.")
                    obs = InnerValidationObservation(
                        fold_id=fold.fold_id,
                        candidate_id=candidate.candidate_id,
                        validation_scenario=candidate.validation_scenarios[0],
                        model_name=model_spec.name,
                        weighting_regime=weighting_regime,
                        seed=seed,
                        hyperparameters=params,
                        threshold=threshold,
                        macro_f1=float(metrics["macro_f1"]),
                        partition_role="validation",
                    )
                    inner_observations.append(obs)
                    unit_obs_dicts.append(
                        {
                            "fold_id": obs.fold_id,
                            "candidate_id": obs.candidate_id,
                            "validation_scenario": obs.validation_scenario,
                            "model_name": obs.model_name,
                            "weighting_regime": obs.weighting_regime,
                            "seed": obs.seed,
                            "hyperparameters": dict(obs.hyperparameters),
                            "threshold": obs.threshold,
                            "macro_f1": obs.macro_f1,
                            "partition_role": obs.partition_role,
                        }
                    )
                if checkpoint_store is not None:
                    checkpoint_store.save(
                        ckpt_name,
                        unit_type="inner_fit",
                        unit_key=unit_key,
                        payload={"observations": unit_obs_dicts},
                    )
                _emit_progress(label, "computed", time.time() - t0)

    selection = select_inner_validation_configuration(
        plan, fold, model_spec, weighting_regime, inner_observations
    )
    inner_selection_sha = _sha256_bytes(_canonical_json_bytes(selection.to_dict()))

    x_outer_tr: np.ndarray | None = None
    y_outer_tr: np.ndarray | None = None
    scen_outer_tr: tuple[str, ...] | None = None
    outer_train_count = sum(len(outer_train_datasets[s].labels) for s in fold.outer_train_scenarios)
    has_cats = any(
        outer_test_dataset.labels[i] == 1 and bool(outer_test_dataset.attack_categories[i])
        for i in range(len(outer_test_dataset.labels))
    )
    test_cats = outer_test_dataset.attack_categories if has_cats else None

    per_seed_results: list[dict[str, object]] = []
    cached_outer: tuple[Mapping[str, object], Mapping[str, object]] | None = None
    for seed in plan.random_seeds:
        ckpt_name = (
            f"outer__{model_spec.name}__{weighting_regime}__{fold.fold_id}__s{seed}.json"
        )
        unit_key = {
            "model_name": model_spec.name,
            "weighting_regime": weighting_regime,
            "fold_id": fold.fold_id,
            "test_scenarios": list(fold.test_scenarios),
            "seed": seed,
            "selected_hyperparameters": dict(selection.selected_hyperparameters),
            "fixed_threshold": plan.fixed_threshold,
            "selected_threshold": selection.selected_threshold,
            "inner_selection_sha256": inner_selection_sha,
        }
        label = f"{model_spec.name} | {weighting_regime} | {fold.fold_id} | outer s{seed}"
        loaded = (
            checkpoint_store.load_if_exists(
                ckpt_name, unit_type="outer_seed_fit", unit_key=unit_key
            )
            if checkpoint_store is not None
            else None
        )
        if loaded is not None:
            per_seed_results.append(dict(loaded))
            _emit_progress(label, "resumed")
            continue

        if checkpoint_store is not None:
            checkpoint_store.check_budget_before_compute(label)

        t0 = time.time()
        if not model_spec.stochastic and cached_outer is not None:
            model_hp, evaluation = cached_outer
        else:
            if x_outer_tr is None or y_outer_tr is None or scen_outer_tr is None:
                x_outer_tr = np.vstack(
                    [outer_train_datasets[s].features for s in fold.outer_train_scenarios]
                )
                y_outer_tr = np.concatenate(
                    [outer_train_datasets[s].labels for s in fold.outer_train_scenarios]
                )
                scen_outer_tr = tuple(
                    s
                    for s in fold.outer_train_scenarios
                    for _ in range(len(outer_train_datasets[s].labels))
                )
            fitted = _fit_compact_tunable(
                model_spec.name,
                x_outer_tr,
                y_outer_tr,
                weighting_regime=weighting_regime,
                train_scenarios=scen_outer_tr,
                params=selection.selected_hyperparameters,
                seed=seed,
                config_path=config_path,
                rf_n_jobs=rf_n_jobs,
            )
            test_probs = fitted.predict_proba(outer_test_dataset.features)
            evaluation = evaluate_probabilistic_predictions(
                outer_test_dataset.labels,
                test_probs,
                selected_threshold=selection.selected_threshold,
                fixed_threshold=plan.fixed_threshold,
                attack_categories=test_cats,
            )
            model_hp = dict(fitted.hyperparameters)
            if not model_spec.stochastic:
                cached_outer = (model_hp, evaluation)

        seed_payload = {
            "seed": seed,
            "model_hyperparameters": dict(model_hp),
            "evaluation": dict(evaluation),
        }
        if checkpoint_store is not None:
            checkpoint_store.save(
                ckpt_name,
                unit_type="outer_seed_fit",
                unit_key=unit_key,
                payload=seed_payload,
            )
        per_seed_results.append(seed_payload)
        _emit_progress(label, "computed", time.time() - t0)

    return {
        "fold_id": fold.fold_id,
        "outer_train_scenarios": list(fold.outer_train_scenarios),
        "test_scenarios": list(fold.test_scenarios),
        "model_name": model_spec.name,
        "weighting_regime": weighting_regime,
        "selected_hyperparameters": dict(selection.selected_hyperparameters),
        "fixed_threshold": plan.fixed_threshold,
        "selected_threshold": selection.selected_threshold,
        "inner_selection": selection.to_dict(),
        "outer_train_record_count": int(outer_train_count),
        "outer_test_record_count": int(len(outer_test_dataset.labels)),
        "outer_test_event_ids_sha256": outer_test_dataset.event_ids_sha256,
        "per_seed_results": per_seed_results,
        "seed_summary": _summarize_seed_blocks(per_seed_results, has_selected=True),
    }


def write_benchmark_artifact(
    payload: Mapping[str, object],
    output_path: str | Path,
    *,
    allowed_root: str | Path = EXPERIMENTS_OUTPUT_DIR,
) -> tuple[Path, str]:
    """Atomically write a deterministic JSON artifact inside the Git-ignored experiments directory."""
    resolved_root = Path(allowed_root).resolve()
    resolved_out = Path(output_path).resolve()
    if not resolved_out.is_relative_to(resolved_root) or resolved_out == resolved_root:
        raise BenchmarkError(
            f"Artifact output path {resolved_out} must lie inside {resolved_root}."
        )
    for forbidden in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
        resolved_forbidden = forbidden.resolve()
        if resolved_out.is_relative_to(resolved_forbidden):
            raise BenchmarkError(f"Refusing to write artifact inside {resolved_forbidden}.")

    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = _sha256_bytes(content)

    if resolved_out.exists():
        if resolved_out.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to overwrite existing different benchmark artifact: {resolved_out}"
            )
        return resolved_out, digest

    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".benchmark.", suffix=".tmp", dir=resolved_out.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(tmp_path, resolved_out)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return resolved_out, digest


def run_full_nested_loso_benchmark(
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    base_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    allowed_output_root: str | Path = EXPERIMENTS_OUTPUT_DIR,
    checkpoint_dir: str | Path | None = None,
    max_new_units: int | None = None,
    progress: bool = False,
    rf_n_jobs: int | None = -1,
    rule_detector=None,
) -> tuple[dict[str, object], Path | None, str | None]:
    """Execute or resume the complete 4-scenario nested LOSO exploratory benchmark."""
    resolved_base = PROJECT_ROOT if base_dir is None else Path(base_dir).resolve()
    cfg_path = Path(config_path)
    config_sha256 = _sha256_file(cfg_path)
    config = load_and_verify_experiment_config(cfg_path, base_dir=resolved_base)
    plan = build_nested_loso_plan(cfg_path, base_dir=resolved_base)

    # Load all 4 verified scenarios once into bounded-memory compact NumPy datasets
    datasets: dict[str, CompactScenarioDataset] = {
        s: load_verified_scenario_dataset(config, s, base_dir=resolved_base)
        for s in plan.scenario_ids
    }

    ckpt_provenance = {
        "experiment_id": plan.experiment_id,
        "experiment_config_sha256": config_sha256,
        "split_manifest_sha256": plan.split_manifest_sha256,
        "quality_report_sha256": plan.quality_report_sha256,
        "scenario_labelled_sha256": {
            s: plan.scenario_provenance[s]["labelled_records_sha256"]
            for s in plan.scenario_ids
        },
    }
    checkpoint_store = (
        _CheckpointStore(
            checkpoint_dir,
            allowed_root=allowed_output_root,
            provenance=ckpt_provenance,
            max_new_units=max_new_units,
        )
        if checkpoint_dir is not None
        else None
    )
    progress_state = {"done": 0, "total": _count_total_units(plan)}

    models_output: dict[str, object] = {}
    for spec in plan.model_specs:
        if spec.name == "RuleBaseline_R_A_to_D":
            rule_folds: list[dict[str, object]] = []
            for fold in plan.outer_folds:
                ckpt_name = f"rule__{fold.fold_id}.json"
                unit_key = {
                    "model_name": "RuleBaseline_R_A_to_D",
                    "fold_id": fold.fold_id,
                    "test_scenarios": list(fold.test_scenarios),
                }
                label = f"RuleBaseline_R_A_to_D | {fold.fold_id}"
                loaded = (
                    checkpoint_store.load_if_exists(
                        ckpt_name, unit_type="rule_fold", unit_key=unit_key
                    )
                    if checkpoint_store is not None
                    else None
                )
                if loaded is not None:
                    rule_folds.append(dict(loaded))
                    progress_state["done"] += 1
                    if progress:
                        print(
                            f"[{progress_state['done']}/{progress_state['total']}] {label} (resumed)",
                            flush=True,
                        )
                    continue

                if checkpoint_store is not None:
                    checkpoint_store.check_budget_before_compute(label)

                t0 = time.time()
                test_s = fold.test_scenarios[0]
                rule_events = load_verified_rule_events(
                    config, test_s, base_dir=resolved_base
                )
                fold_res = run_rule_baseline_outer_fold(
                    plan,
                    fold,
                    rule_events,
                    config_path=cfg_path,
                    detector=rule_detector,
                )
                if checkpoint_store is not None:
                    checkpoint_store.save(
                        ckpt_name,
                        unit_type="rule_fold",
                        unit_key=unit_key,
                        payload=fold_res,
                    )
                rule_folds.append(fold_res)
                progress_state["done"] += 1
                if progress:
                    print(
                        f"[{progress_state['done']}/{progress_state['total']}] {label} (computed, {time.time() - t0:.1f}s)",
                        flush=True,
                    )
            models_output[spec.name] = {
                "probabilistic": False,
                "outer_folds": rule_folds,
                "cross_scenario_summary": {
                    key: _aggregate_stats([f["evaluation"][key] for f in rule_folds])
                    for key in _HARD_METRIC_KEYS
                },
            }
        elif spec.name == "DummyClassifier":
            dummy_folds: list[dict[str, object]] = []
            for fold in plan.outer_folds:
                ckpt_name = f"dummy__{fold.fold_id}.json"
                unit_key = {
                    "model_name": "DummyClassifier",
                    "fold_id": fold.fold_id,
                    "test_scenarios": list(fold.test_scenarios),
                    "seeds": list(plan.random_seeds),
                    "strategies": [c["strategy"] for c in spec.parameter_combinations],
                }
                label = f"DummyClassifier | {fold.fold_id}"
                loaded = (
                    checkpoint_store.load_if_exists(
                        ckpt_name, unit_type="dummy_fold", unit_key=unit_key
                    )
                    if checkpoint_store is not None
                    else None
                )
                if loaded is not None:
                    dummy_folds.append(dict(loaded))
                    progress_state["done"] += 1
                    if progress:
                        print(
                            f"[{progress_state['done']}/{progress_state['total']}] {label} (resumed)",
                            flush=True,
                        )
                    continue

                if checkpoint_store is not None:
                    checkpoint_store.check_budget_before_compute(label)

                t0 = time.time()
                fold_res = run_dummy_outer_fold(
                    plan,
                    fold,
                    spec,
                    {s: datasets[s] for s in fold.outer_train_scenarios},
                    datasets[fold.test_scenarios[0]],
                    config_path=cfg_path,
                )
                if checkpoint_store is not None:
                    checkpoint_store.save(
                        ckpt_name,
                        unit_type="dummy_fold",
                        unit_key=unit_key,
                        payload=fold_res,
                    )
                dummy_folds.append(fold_res)
                progress_state["done"] += 1
                if progress:
                    print(
                        f"[{progress_state['done']}/{progress_state['total']}] {label} (computed, {time.time() - t0:.1f}s)",
                        flush=True,
                    )
            models_output[spec.name] = {
                "probabilistic": True,
                "outer_folds": dummy_folds,
                "cross_scenario_summary": _summarize_across_folds_probabilistic(
                    dummy_folds, has_selected=False
                ),
            }
        else:
            by_regime: dict[str, object] = {}
            for regime in spec.weighting_regimes:
                regime_folds = [
                    run_tunable_model_outer_fold_compact(
                        plan,
                        fold,
                        spec,
                        regime,
                        {s: datasets[s] for s in fold.outer_train_scenarios},
                        datasets[fold.test_scenarios[0]],
                        config_path=cfg_path,
                        rf_n_jobs=rf_n_jobs,
                        checkpoint_store=checkpoint_store,
                        progress_state=progress_state,
                        progress=progress,
                    )
                    for fold in plan.outer_folds
                ]
                by_regime[regime] = {
                    "outer_folds": regime_folds,
                    "cross_scenario_summary": _summarize_across_folds_probabilistic(
                        regime_folds, has_selected=True
                    ),
                }
            models_output[spec.name] = {
                "probabilistic": True,
                "weighting_regimes": by_regime,
            }

    if progress_state["done"] != progress_state["total"]:
        raise BenchmarkIncompleteError(
            f"Incomplete benchmark execution: {progress_state['done']}/{progress_state['total']} units finished."
        )

    report: dict[str, object] = {
        "benchmark_report_schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "experiment_id": plan.experiment_id,
        "experiment_config_sha256": config_sha256,
        "study_status": plan.study_status,
        "evaluation_framing": (
            "Exploratory internal evaluation on four AIT-LDS v2.1 scenarios; "
            "not independently validated production performance."
        ),
        "source_manifest_status": {
            "scientifically_adequate": False,
            "selected_assignment": None,
        },
        "dataset_provenance": {
            "feature_schema_version": plan.feature_schema_version,
            "split_manifest_path": plan.split_manifest_path,
            "split_manifest_sha256": plan.split_manifest_sha256,
            "quality_report_path": plan.quality_report_path,
            "quality_report_sha256": plan.quality_report_sha256,
            "scenarios": {
                s: {
                    **dict(plan.scenario_provenance[s]),
                    "included_normal_count": datasets[s].load_summary.included_normal_count,
                    "included_attack_count": datasets[s].load_summary.included_attack_count,
                    "excluded_unknown_count": datasets[s].load_summary.excluded_unknown_count,
                    "event_ids_sha256": datasets[s].event_ids_sha256,
                }
                for s in plan.scenario_ids
            },
        },
        "random_seeds": list(plan.random_seeds),
        "fixed_threshold": plan.fixed_threshold,
        "candidate_thresholds": list(plan.candidate_thresholds),
        "models": models_output,
        "scientific_limitations": list(config.get("scientific_limitations", [])),
    }

    written_path: Path | None = None
    artifact_sha: str | None = None
    if output_path is not None:
        written_path, artifact_sha = write_benchmark_artifact(
            report, output_path, allowed_root=allowed_output_root
        )
    return report, written_path, artifact_sha


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the ForensicInsight 4-scenario nested LOSO exploratory benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG_PATH,
        help="Path to experiment_config.json.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory inside research_datasets/ait-lds-v2.1/experiments/ for per-unit checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT_PATH,
        help="Final JSON artifact path inside research_datasets/ait-lds-v2.1/experiments/.",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=-1,
        help="CPU threads for RandomForestClassifier (default: -1 for all cores).",
    )
    parser.add_argument(
        "--max-new-units",
        type=int,
        default=None,
        help="Optional cap on newly computed units in this invocation (stops cleanly without writing final report).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    t0 = time.time()
    try:
        report, written_path, artifact_sha = run_full_nested_loso_benchmark(
            config_path=args.config,
            output_path=args.output,
            checkpoint_dir=args.checkpoint_dir,
            max_new_units=args.max_new_units,
            progress=True,
            rf_n_jobs=args.rf_n_jobs,
        )
    except BenchmarkIncompleteError as exc:
        print(f"PAUSED (INCOMPLETE): {exc}", flush=True)
        return 2
    elapsed = time.time() - t0
    print(
        f"COMPLETE ({elapsed:.1f}s) | experiment_id={report['experiment_id']} | "
        f"artifact={written_path} | sha256={artifact_sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
