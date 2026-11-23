from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping

from .schema import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.json"

_FEATURE_DEFINITIONS: tuple[dict, ...] = tuple(FEATURE_SCHEMA["features"])
EXPECTED_FEATURE_NAMES: tuple[str, ...] = tuple(
    definition["name"] for definition in _FEATURE_DEFINITIONS
)
_FEATURE_TYPES: dict[str, str] = {
    definition["name"]: definition["type"] for definition in _FEATURE_DEFINITIONS
}
_EXPECTED_FEATURE_SET: frozenset[str] = frozenset(EXPECTED_FEATURE_NAMES)
_INDICATOR_FEATURES: frozenset[str] = frozenset(
    name
    for name in EXPECTED_FEATURE_NAMES
    if name.endswith("_missing") or name == "sensitive_resource_indicator"
)
_LABEL_NAMES: dict[int | None, str] = {
    0: "normal",
    1: "attack_associated",
    None: "unknown",
}


class TrainingDataError(ValueError):
    """Raised when experiment config, manifests, hashes, or records fail validation."""


@dataclass(frozen=True)
class SupervisedRecord:
    """Traceable supervised record with ordered schema-1.0.0 numerical features."""

    scenario_id: str
    event_id: str
    source_filename: str
    source_line_number: int
    source_sha256: str
    original_evidence_reference: str
    label: int
    label_name: str
    label_basis: str
    label_provenance: Mapping[str, object]
    attack_categories: tuple[str, ...]
    features: tuple[int | float | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "label_provenance", MappingProxyType(dict(self.label_provenance))
        )


@dataclass(frozen=True)
class ScenarioLoadSummary:
    """Verified provenance and record counts produced while streaming one scenario."""

    scenario_id: str
    label_manifest_path: str
    label_manifest_sha256: str
    labelled_records_sha256: str
    prepared_manifest_sha256: str
    prepared_records_sha256: str
    total_records_read: int
    included_normal_count: int
    included_attack_count: int
    excluded_unknown_count: int


def _resolve_path(raw_path: object, base_dir: Path, description: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise TrainingDataError(f"Missing or invalid {description} path.")
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _safe_filename(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or "\\" in value
        or value in (".", "..")
    ):
        raise TrainingDataError(f"Invalid {description} filename.")
    return value


def _sha256_file(path: Path, description: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrainingDataError(f"Unable to read {description}: {path}") from exc
    return digest.hexdigest()


def _read_json_and_sha256(path: Path, description: str) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Unable to read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise TrainingDataError(f"{description} must be a JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def load_and_verify_experiment_config(
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    *,
    base_dir: str | Path | None = None,
) -> dict:
    """Load experiment_config.json and fail closed on schema or manifest hash mismatches."""
    resolved_base = PROJECT_ROOT if base_dir is None else Path(base_dir).resolve()
    config, _ = _read_json_and_sha256(Path(config_path), "experiment config")

    features_cfg = config.get("features")
    if not isinstance(features_cfg, dict):
        raise TrainingDataError("Experiment config is missing features metadata.")
    if features_cfg.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise TrainingDataError(
            f"Unsupported feature schema version: {features_cfg.get('schema_version')!r}"
        )
    if features_cfg.get("ordered_names") != list(EXPECTED_FEATURE_NAMES):
        raise TrainingDataError(
            "Experiment config feature names do not match FEATURE_SCHEMA v1.0.0 order."
        )

    label_def = config.get("label_definition")
    if not isinstance(label_def, dict):
        raise TrainingDataError("Experiment config is missing label_definition.")
    if (
        label_def.get("positive_class", {}).get("value") != 1
        or label_def.get("negative_class", {}).get("value") != 0
        or label_def.get("excluded_class", {}).get("value") is not None
        or label_def.get("excluded_class", {}).get("exclude_from_supervised") is not True
    ):
        raise TrainingDataError("Experiment config label_definition violates supervised policy.")

    provenance = config.get("dataset_provenance")
    if not isinstance(provenance, dict):
        raise TrainingDataError("Experiment config is missing dataset_provenance.")

    split_ref = provenance.get("split_manifest")
    quality_ref = provenance.get("quality_report")
    if not isinstance(split_ref, dict) or not isinstance(quality_ref, dict):
        raise TrainingDataError("Experiment config is missing split or quality manifest refs.")

    split_path = _resolve_path(split_ref.get("path"), resolved_base, "split manifest")
    _, split_sha = _read_json_and_sha256(split_path, "split manifest")
    if split_sha != split_ref.get("sha256"):
        raise TrainingDataError("Split manifest SHA-256 mismatch against experiment config.")

    quality_path = _resolve_path(quality_ref.get("path"), resolved_base, "quality report")
    quality_payload, quality_sha = _read_json_and_sha256(quality_path, "quality report")
    if quality_sha != quality_ref.get("sha256"):
        raise TrainingDataError("Quality report SHA-256 mismatch against experiment config.")
    if quality_payload.get("quality_status") == "failed":
        raise TrainingDataError("Refusing to load data when quality report status is failed.")
    if quality_payload.get("quality_status") != quality_ref.get("expected_quality_status"):
        raise TrainingDataError("Quality report status does not match experiment config.")

    scenarios = provenance.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise TrainingDataError("Experiment config defines no scenarios.")

    return config


def _extract_ordered_features(
    features: object,
    record_number: int,
    scenario_id: str,
) -> tuple[int | float | None, ...]:
    if not isinstance(features, dict):
        raise TrainingDataError(
            f"{scenario_id} record {record_number}: features must be a JSON object."
        )
    actual_keys = set(features)
    if actual_keys != _EXPECTED_FEATURE_SET:
        missing = sorted(_EXPECTED_FEATURE_SET - actual_keys)
        unexpected = sorted(actual_keys - _EXPECTED_FEATURE_SET)
        raise TrainingDataError(
            f"{scenario_id} record {record_number}: feature schema mismatch "
            f"(missing={missing}, unexpected={unexpected})."
        )

    values: list[int | float | None] = []
    for name in EXPECTED_FEATURE_NAMES:
        value = features[name]
        if value is None:
            values.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingDataError(
                f"{scenario_id} record {record_number}: non-numeric value for {name!r}."
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise TrainingDataError(
                f"{scenario_id} record {record_number}: non-finite value for {name!r}."
            )
        if _FEATURE_TYPES[name] == "integer" and type(value) is not int:
            raise TrainingDataError(
                f"{scenario_id} record {record_number}: non-integer value for {name!r}."
            )
        if name in _INDICATOR_FEATURES:
            if value not in (0, 1):
                raise TrainingDataError(
                    f"{scenario_id} record {record_number}: invalid indicator {name!r}={value!r}."
                )
        elif value < 0:
            raise TrainingDataError(
                f"{scenario_id} record {record_number}: negative value for {name!r}."
            )
        values.append(value)
    return tuple(values)


def iter_scenario_supervised_records(
    config: Mapping[str, object],
    scenario_id: str,
    *,
    base_dir: str | Path | None = None,
    summary_out: dict[str, ScenarioLoadSummary] | None = None,
) -> Iterator[SupervisedRecord]:
    """Stream verified known-labelled (0/1) records for one scenario, excluding unknowns."""
    resolved_base = PROJECT_ROOT if base_dir is None else Path(base_dir).resolve()
    provenance_cfg = config.get("dataset_provenance")
    if not isinstance(provenance_cfg, Mapping):
        raise TrainingDataError("Experiment config is missing dataset_provenance.")
    scenarios_cfg = provenance_cfg.get("scenarios")
    if not isinstance(scenarios_cfg, Mapping) or scenario_id not in scenarios_cfg:
        raise TrainingDataError(f"Scenario {scenario_id!r} is not registered in experiment config.")
    scenario_ref = scenarios_cfg[scenario_id]
    if not isinstance(scenario_ref, Mapping):
        raise TrainingDataError(f"Invalid scenario provenance entry for {scenario_id!r}.")

    label_manifest_path = _resolve_path(
        scenario_ref.get("label_manifest_path"), resolved_base, f"{scenario_id} label manifest"
    )
    label_manifest, label_manifest_sha = _read_json_and_sha256(
        label_manifest_path, f"{scenario_id} label manifest"
    )
    if label_manifest_sha != scenario_ref.get("label_manifest_sha256"):
        raise TrainingDataError(f"{scenario_id}: label manifest SHA-256 mismatch.")
    if label_manifest.get("scenario_id") != scenario_id:
        raise TrainingDataError(f"{scenario_id}: label manifest scenario_id mismatch.")
    if label_manifest.get("label_schema_version") != provenance_cfg.get("label_schema_version"):
        raise TrainingDataError(f"{scenario_id}: label schema version mismatch.")

    scenario_dir = label_manifest_path.parent
    prepared_manifest_path = scenario_dir / "manifest.json"
    prepared_manifest, prepared_manifest_sha = _read_json_and_sha256(
        prepared_manifest_path, f"{scenario_id} prepared manifest"
    )
    if prepared_manifest_sha != scenario_ref.get("prepared_manifest_sha256"):
        raise TrainingDataError(f"{scenario_id}: prepared manifest SHA-256 mismatch.")
    if prepared_manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise TrainingDataError(f"{scenario_id}: prepared manifest feature schema mismatch.")

    label_input = label_manifest.get("input")
    processed_output = prepared_manifest.get("processed_output")
    labelled_output = label_manifest.get("labelled_output")
    if (
        not isinstance(label_input, dict)
        or not isinstance(processed_output, dict)
        or not isinstance(labelled_output, dict)
    ):
        raise TrainingDataError(f"{scenario_id}: incomplete manifest input/output metadata.")

    expected_prepared_sha = scenario_ref.get("prepared_records_sha256")
    expected_labelled_sha = scenario_ref.get("labelled_records_sha256")
    if (
        label_input.get("prepared_manifest_sha256") != prepared_manifest_sha
        or label_input.get("prepared_records_sha256") != expected_prepared_sha
        or processed_output.get("sha256") != expected_prepared_sha
        or labelled_output.get("sha256") != expected_labelled_sha
    ):
        raise TrainingDataError(f"{scenario_id}: manifest hash chain mismatch.")

    labelled_filename = _safe_filename(
        labelled_output.get("filename"), f"{scenario_id} labelled dataset"
    )
    labelled_path = scenario_dir / labelled_filename
    actual_labelled_sha = _sha256_file(labelled_path, f"{scenario_id} labelled dataset")
    if actual_labelled_sha != expected_labelled_sha:
        raise TrainingDataError(f"{scenario_id}: labelled_events.jsonl SHA-256 mismatch.")

    source_entries = prepared_manifest.get("source_files")
    per_source_stats = label_manifest.get("per_source_distributions")
    if not isinstance(source_entries, list) or not isinstance(per_source_stats, list):
        raise TrainingDataError(f"{scenario_id}: missing source provenance tables.")
    sources: dict[str, dict] = {}
    for entry in source_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("source_filename"), str):
            raise TrainingDataError(f"{scenario_id}: invalid source entry in prepared manifest.")
        sources[entry["source_filename"]] = entry
    source_stats: dict[str, dict] = {}
    for entry in per_source_stats:
        if not isinstance(entry, dict) or not isinstance(entry.get("source_filename"), str):
            raise TrainingDataError(f"{scenario_id}: invalid source stats in label manifest.")
        source_stats[entry["source_filename"]] = entry
    if set(sources) != set(source_stats):
        raise TrainingDataError(f"{scenario_id}: source coverage mismatch across manifests.")
    for name, source in sources.items():
        if (
            source_stats[name].get("source_sha256") != source.get("source_sha256")
            or source_stats[name].get("ground_truth_sha256") != source.get("ground_truth_sha256")
        ):
            raise TrainingDataError(f"{scenario_id}: source hash mismatch for {name!r}.")

    policy_id = label_manifest.get("label_policy", {}).get("policy_id")
    label_schema_version = label_manifest.get("label_schema_version")
    total_read = 0
    normal_count = 0
    attack_count = 0
    unknown_count = 0

    try:
        with labelled_path.open("rb") as handle:
            for record_number, raw_line in enumerate(handle, start=1):
                total_read += 1
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: invalid UTF-8 JSON."
                    ) from exc
                if not isinstance(record, dict):
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: record must be a JSON object."
                    )

                event = record.get("event")
                provenance = record.get("provenance")
                if not isinstance(event, dict) or not isinstance(provenance, dict):
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: missing event or provenance."
                    )
                if provenance.get("scenario_id") != scenario_id:
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: provenance scenario_id mismatch."
                    )
                source_filename = provenance.get("source_filename")
                source = sources.get(source_filename)
                if (
                    source is None
                    or provenance.get("source_sha256") != source.get("source_sha256")
                ):
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: unverified source provenance."
                    )
                source_line_number = provenance.get("source_line_number")
                if type(source_line_number) is not int or source_line_number < 1:
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: invalid source_line_number."
                    )

                event_id = event.get("event_id")
                original_ref = event.get("original_evidence_reference")
                if (
                    not isinstance(event_id, str)
                    or not event_id.strip()
                    or not isinstance(original_ref, str)
                    or not original_ref.strip()
                ):
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: missing event identity or reference."
                    )

                label = record.get("research_label")
                if (label is not None and type(label) is not int) or label not in (0, 1, None):
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: invalid research_label {label!r}."
                    )
                expected_label_name = _LABEL_NAMES[label]
                if record.get("research_label_name") != expected_label_name:
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: research_label_name mismatch."
                    )

                label_provenance = record.get("research_label_provenance")
                if (
                    not isinstance(label_provenance, dict)
                    or label_provenance.get("policy_id") != policy_id
                    or label_provenance.get("label_schema_version") != label_schema_version
                ):
                    raise TrainingDataError(
                        f"{scenario_id} record {record_number}: invalid research_label_provenance."
                    )

                ground_truth = record.get("ground_truth")
                label_basis = record.get("research_label_basis")
                if label is None:
                    gt_labels = (
                        ground_truth.get("labels") if isinstance(ground_truth, dict) else None
                    )
                    if gt_labels:
                        raise TrainingDataError(
                            f"{scenario_id} record {record_number}: unknown label has explicit categories."
                        )
                    if label_basis not in (
                        "insufficient_verified_label_coverage",
                        "explicit_ground_truth_without_categories",
                    ):
                        raise TrainingDataError(
                            f"{scenario_id} record {record_number}: invalid unknown label basis."
                        )
                    unknown_count += 1
                    continue

                attack_categories: tuple[str, ...] = ()
                if label == 0:
                    if (
                        source_stats[source_filename].get("ground_truth_coverage") != "verified"
                        or ground_truth is not None
                        or label_basis != "absence_from_verified_ground_truth_file"
                        or label_provenance.get("ground_truth_reference")
                        != source.get("ground_truth_reference")
                        or label_provenance.get("ground_truth_sha256")
                        != source.get("ground_truth_sha256")
                    ):
                        raise TrainingDataError(
                            f"{scenario_id} record {record_number}: unverified normal label."
                        )
                    normal_count += 1
                else:
                    gt_prov = record.get("ground_truth_provenance")
                    gt_labels = (
                        ground_truth.get("labels") if isinstance(ground_truth, dict) else None
                    )
                    if (
                        not isinstance(gt_labels, list)
                        or not gt_labels
                        or any(not isinstance(item, str) or not item for item in gt_labels)
                        or not isinstance(gt_prov, dict)
                        or gt_prov.get("filename") != source.get("ground_truth_filename")
                        or gt_prov.get("reference") != source.get("ground_truth_reference")
                        or gt_prov.get("sha256") != source.get("ground_truth_sha256")
                        or type(gt_prov.get("label_file_line_number")) is not int
                        or gt_prov.get("label_file_line_number") < 1
                        or label_provenance.get("ground_truth_reference")
                        != source.get("ground_truth_reference")
                        or label_provenance.get("ground_truth_sha256")
                        != source.get("ground_truth_sha256")
                        or label_basis != "explicit_ground_truth"
                    ):
                        raise TrainingDataError(
                            f"{scenario_id} record {record_number}: unverified attack label."
                        )
                    attack_categories = tuple(gt_labels)
                    attack_count += 1

                feature_values = _extract_ordered_features(
                    record.get("features"), record_number, scenario_id
                )
                yield SupervisedRecord(
                    scenario_id=scenario_id,
                    event_id=event_id,
                    source_filename=source_filename,
                    source_line_number=source_line_number,
                    source_sha256=source["source_sha256"],
                    original_evidence_reference=original_ref,
                    label=label,
                    label_name=expected_label_name,
                    label_basis=label_basis,
                    label_provenance=label_provenance,
                    attack_categories=attack_categories,
                    features=feature_values,
                )
    except OSError as exc:
        raise TrainingDataError(f"Unable to stream {scenario_id} labelled dataset.") from exc

    expected_eval_count = label_manifest.get("record_counts", {}).get("evaluation_records")
    expected_dist = label_manifest.get("evaluation_class_distribution")
    actual_dist = {
        "normal": normal_count,
        "attack_associated": attack_count,
        "unknown": unknown_count,
    }
    if total_read != expected_eval_count or actual_dist != expected_dist:
        raise TrainingDataError(
            f"{scenario_id}: streamed record counts do not match label manifest."
        )

    if summary_out is not None:
        summary_out[scenario_id] = ScenarioLoadSummary(
            scenario_id=scenario_id,
            label_manifest_path=str(label_manifest_path),
            label_manifest_sha256=label_manifest_sha,
            labelled_records_sha256=actual_labelled_sha,
            prepared_manifest_sha256=prepared_manifest_sha,
            prepared_records_sha256=expected_prepared_sha,
            total_records_read=total_read,
            included_normal_count=normal_count,
            included_attack_count=attack_count,
            excluded_unknown_count=unknown_count,
        )


def iter_supervised_records(
    scenario_ids: Iterable[str],
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    base_dir: str | Path | None = None,
    summaries_out: dict[str, ScenarioLoadSummary] | None = None,
) -> Iterator[SupervisedRecord]:
    """Verify experiment config and stream supervised records for requested scenarios."""
    requested = list(scenario_ids)
    if not requested or len(set(requested)) != len(requested):
        raise TrainingDataError("scenario_ids must be a non-empty sequence of distinct IDs.")
    config = load_and_verify_experiment_config(config_path, base_dir=base_dir)
    for scenario_id in requested:
        yield from iter_scenario_supervised_records(
            config,
            scenario_id,
            base_dir=base_dir,
            summary_out=summaries_out,
        )


def load_supervised_scenarios(
    scenario_ids: Iterable[str],
    *,
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    base_dir: str | Path | None = None,
) -> tuple[list[SupervisedRecord], dict[str, ScenarioLoadSummary]]:
    """Materialize verified supervised records and per-scenario summaries."""
    summaries: dict[str, ScenarioLoadSummary] = {}
    records = list(
        iter_supervised_records(
            scenario_ids,
            config_path=config_path,
            base_dir=base_dir,
            summaries_out=summaries,
        )
    )
    return records, summaries
