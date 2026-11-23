from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from backend.rules import RULE_REGISTRY, calculate_rule_fingerprint, get_rule_metadata

from .evaluation import evaluate_hard_predictions
from .schema import CanonicalEvent
from .training_data import DEFAULT_EXPERIMENT_CONFIG_PATH


NOT_APPLICABLE_METRICS: tuple[str, ...] = (
    "pr_auc_attack",
    "pr_auc_normal",
    "brier_score",
)
RULE_BASELINE_CAVEATS: tuple[str, ...] = (
    "Rule R-C clears post_requests[source_ip] after firing and therefore emits at most one finding per source IP per file.",
    "Rules R-C and R-D compute counts and baseline statistics over the full source file (non-causal batch heuristics).",
)
_CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "source_ip",
    "event_type",
    "resource",
    "status",
    "bytes",
)


class RuleBaselineError(ValueError):
    """Raised when rule-baseline inputs, config provenance, or finding alignment fail."""


@dataclass(frozen=True)
class RuleBaselineInputEvent:
    """Verified known-labelled canonical Apache event eligible for rule-baseline evaluation."""

    event: CanonicalEvent
    scenario_id: str
    source_filename: str
    source_line_number: int
    label: int
    label_name: str
    ground_truth_coverage: str = "verified"
    attack_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.event, CanonicalEvent):
            raise RuleBaselineError("event must be a CanonicalEvent.")
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise RuleBaselineError("scenario_id must be a non-empty string.")
        if not isinstance(self.source_filename, str) or not self.source_filename.strip():
            raise RuleBaselineError("source_filename must be a non-empty string.")
        if isinstance(self.source_line_number, bool) or not isinstance(self.source_line_number, int) or self.source_line_number < 1:
            raise RuleBaselineError("source_line_number must be a positive integer.")
        if isinstance(self.label, bool) or not isinstance(self.label, int) or self.label not in (0, 1):
            raise RuleBaselineError(
                f"Only verified known labels 0 or 1 are eligible; got {self.label!r}."
            )
        expected_name = "normal" if self.label == 0 else "attack_associated"
        if self.label_name != expected_name:
            raise RuleBaselineError(
                f"label_name {self.label_name!r} does not match label {self.label}."
            )
        if self.ground_truth_coverage != "verified":
            raise RuleBaselineError(
                f"Only verified ground-truth sources are eligible; got {self.ground_truth_coverage!r}."
            )


@dataclass(frozen=True)
class RuleBaselineEventPrediction:
    """Per-event hard prediction and exact rule provenance aligned to a source event."""

    scenario_id: str
    event_id: str
    source_filename: str
    source_line_number: int
    timestamp: str
    original_evidence_reference: str
    label: int
    predicted: int
    matched_rule_ids: tuple[str, ...]
    matched_rule_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matched_rule_versions",
            MappingProxyType(dict(self.matched_rule_versions)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "event_id": self.event_id,
            "source_filename": self.source_filename,
            "source_line_number": self.source_line_number,
            "timestamp": self.timestamp,
            "original_evidence_reference": self.original_evidence_reference,
            "label": self.label,
            "predicted": self.predicted,
            "matched_rule_ids": list(self.matched_rule_ids),
            "matched_rule_versions": dict(self.matched_rule_versions),
        }


@dataclass(frozen=True)
class RuleBaselineResult:
    """Deterministic hard-prediction evaluation result for RULE_REGISTRY R-A through R-D."""

    rule_registry_fingerprint: str
    rule_versions: Mapping[str, str]
    predictions: tuple[RuleBaselineEventPrediction, ...]
    hard_predictions: tuple[int, ...]
    evaluation: Mapping[str, object]
    not_applicable_metrics: Mapping[str, None]
    methodological_caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_versions", MappingProxyType(dict(self.rule_versions)))
        object.__setattr__(self, "evaluation", MappingProxyType(dict(self.evaluation)))
        object.__setattr__(
            self, "not_applicable_metrics", MappingProxyType(dict(self.not_applicable_metrics))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": "RuleBaseline_R_A_to_D",
            "probabilistic": False,
            "rule_registry_fingerprint": self.rule_registry_fingerprint,
            "rule_versions": dict(self.rule_versions),
            "hard_predictions": list(self.hard_predictions),
            "evaluation": dict(self.evaluation),
            "not_applicable_metrics": dict(self.not_applicable_metrics),
            "methodological_caveats": list(self.methodological_caveats),
            "predictions": [pred.to_dict() for pred in self.predictions],
        }


def load_rule_baseline_config(
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> dict[str, Any]:
    """Verify RuleBaseline_R_A_to_D provenance and fingerprint against RULE_REGISTRY."""
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleBaselineError(f"Unable to read experiment config: {config_path}") from exc
    if not isinstance(config, dict):
        raise RuleBaselineError("Experiment config must be a JSON object.")

    models = config.get("models")
    if not isinstance(models, list):
        raise RuleBaselineError("Experiment config is missing models list.")
    rule_cfg = next(
        (
            item
            for item in models
            if isinstance(item, dict) and item.get("name") == "RuleBaseline_R_A_to_D"
        ),
        None,
    )
    if rule_cfg is None:
        raise RuleBaselineError("RuleBaseline_R_A_to_D is not registered in experiment config.")
    if rule_cfg.get("probabilistic") is not False:
        raise RuleBaselineError("RuleBaseline_R_A_to_D must be non-probabilistic.")

    active_fingerprint = calculate_rule_fingerprint()
    if rule_cfg.get("rule_registry_fingerprint") != active_fingerprint:
        raise RuleBaselineError(
            "Active RULE_REGISTRY fingerprint does not match experiment_config.json."
        )
    if rule_cfg.get("rules") != sorted(RULE_REGISTRY):
        raise RuleBaselineError("Configured rule IDs do not match active RULE_REGISTRY.")
    return rule_cfg


def canonical_event_to_detector_row(event: CanonicalEvent) -> dict[str, str]:
    """Map a CanonicalEvent to the exact CSV columns consumed by backend.main.detect_threats."""
    return {
        "timestamp": event.timestamp.isoformat(),
        "source_ip": "" if event.source_ip is None else event.source_ip,
        "event_type": "" if event.event_type is None else event.event_type,
        "resource": "" if event.resource is None else event.resource,
        "status": "" if event.response_status is None else event.response_status,
        "bytes": "" if event.transfer_size is None else str(event.transfer_size),
    }


def _default_detect_threats(csv_path: Path, evidence_id: int) -> list[dict[str, Any]]:
    from backend.main import detect_threats

    return detect_threats(csv_path, evidence_id)


def run_rule_baseline(
    events: Sequence[RuleBaselineInputEvent],
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    detector: Callable[[Path, int], list[dict[str, Any]]] | None = None,
) -> RuleBaselineResult:
    """Run existing R-A–R-D detector via a read-only adapter and produce aligned hard predictions."""
    if not events:
        raise RuleBaselineError("At least one verified known-labelled event is required.")
    load_rule_baseline_config(config_path)

    identities: set[tuple[str, str, str]] = set()
    rows: list[dict[str, str]] = []
    for idx, item in enumerate(events):
        if not isinstance(item, RuleBaselineInputEvent):
            raise RuleBaselineError(f"events[{idx}] must be a RuleBaselineInputEvent.")
        identity = (item.scenario_id, item.source_filename, item.event.event_id)
        if identity in identities:
            raise RuleBaselineError(f"Duplicate event identity in rule baseline input: {identity!r}")
        identities.add(identity)
        rows.append(canonical_event_to_detector_row(item.event))

    fingerprint_before = calculate_rule_fingerprint()
    metadata_before = get_rule_metadata()
    rule_versions = {
        rule_id: str(RULE_REGISTRY[rule_id]["version"]) for rule_id in sorted(RULE_REGISTRY)
    }

    active_detector = _default_detect_threats if detector is None else detector
    with tempfile.TemporaryDirectory(prefix="forensic-rule-baseline-") as tmp_dir:
        temp_csv = Path(tmp_dir) / "events.csv"
        with temp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_CSV_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        findings = active_detector(temp_csv, 0)

    if calculate_rule_fingerprint() != fingerprint_before or get_rule_metadata() != metadata_before:
        raise RuleBaselineError("RULE_REGISTRY was unexpectedly mutated during rule execution.")

    if not isinstance(findings, list):
        raise RuleBaselineError("Rule detector must return a list of finding dicts.")

    matched_by_index: dict[int, set[str]] = {idx: set() for idx in range(len(events))}
    for f_idx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise RuleBaselineError(f"Finding {f_idx} must be a dict.")
        rule_id = finding.get("rule_id")
        if not isinstance(rule_id, str) or rule_id not in RULE_REGISTRY:
            raise RuleBaselineError(f"Finding {f_idx} has unknown or missing rule_id: {rule_id!r}.")
        raw_event_id = finding.get("event_id")
        if not isinstance(raw_event_id, str) or not raw_event_id.isascii() or not raw_event_id.isdigit():
            raise RuleBaselineError(
                f"Finding {f_idx} lacks an unambiguous numeric row event_id: {raw_event_id!r}."
            )
        row_idx = int(raw_event_id)
        if row_idx < 0 or row_idx >= len(events):
            raise RuleBaselineError(
                f"Finding {f_idx} event_id {row_idx} is out of range for {len(events)} events."
            )
        expected_row = rows[row_idx]
        if (
            finding.get("timestamp") != expected_row["timestamp"]
            or finding.get("source") != expected_row["source_ip"]
            or finding.get("event_type") != expected_row["event_type"]
            or finding.get("resource") != expected_row["resource"]
        ):
            raise RuleBaselineError(
                f"Finding {f_idx} fields do not unambiguously align with source event row {row_idx}."
            )
        matched_by_index[row_idx].add(rule_id)

    predictions: list[RuleBaselineEventPrediction] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    categories: list[tuple[str, ...]] = []
    for idx, item in enumerate(events):
        matched_ids = tuple(sorted(matched_by_index[idx]))
        pred_val = 1 if matched_ids else 0
        predictions.append(
            RuleBaselineEventPrediction(
                scenario_id=item.scenario_id,
                event_id=item.event.event_id,
                source_filename=item.source_filename,
                source_line_number=item.source_line_number,
                timestamp=item.event.timestamp.isoformat(),
                original_evidence_reference=item.event.original_evidence_reference,
                label=item.label,
                predicted=pred_val,
                matched_rule_ids=matched_ids,
                matched_rule_versions={rid: rule_versions[rid] for rid in matched_ids},
            )
        )
        y_true.append(item.label)
        y_pred.append(pred_val)
        categories.append(item.attack_categories)

    evaluation = evaluate_hard_predictions(y_true, y_pred, attack_categories=categories)
    na_metrics = {name: None for name in NOT_APPLICABLE_METRICS}

    return RuleBaselineResult(
        rule_registry_fingerprint=fingerprint_before,
        rule_versions=rule_versions,
        predictions=tuple(predictions),
        hard_predictions=tuple(y_pred),
        evaluation=evaluation,
        not_applicable_metrics=na_metrics,
        methodological_caveats=RULE_BASELINE_CAVEATS,
    )
