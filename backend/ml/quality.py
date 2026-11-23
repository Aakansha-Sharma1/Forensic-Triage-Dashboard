from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
from typing import Mapping

from .features import extract_features
from .schema import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION
from .splits import DEFAULT_LABEL_MANIFEST_PATHS, SPLIT_OUTPUT_PATH


QUALITY_REPORT_SCHEMA_VERSION = "1.0.0"
QUALITY_REPORT_PATH = (
    SPLIT_OUTPUT_PATH.parent.parent / "quality" / "dataset_quality_report.json"
)
_DERIVED_LABEL_FIELDS = {
    "research_label",
    "research_label_name",
    "research_label_basis",
    "research_label_provenance",
}
_DIRECT_TARGET_TOKENS = (
    "label",
    "ground_truth",
    "attack",
    "category",
    "rule",
    "malicious",
    "anomaly",
)
_SENSITIVE_RESOURCE_TERMS = ("secret", "admin", "password", "confidential", ".zip")


class QualityAuditError(ValueError):
    """Raised when quality audit inputs cannot be read or interpreted."""


def _read_json(path: Path, description: str) -> tuple[bytes, dict]:
    try:
        content = path.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityAuditError(f"Unable to read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise QualityAuditError(f"{description} must be a JSON object: {path}")
    return content, payload


def _safe_filename(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or "\\" in value
        or value in (".", "..")
    ):
        raise QualityAuditError(f"Invalid {description} filename.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise QualityAuditError(f"Unable to hash dataset file: {path}") from exc
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _temporal_prefix_check() -> dict:
    rows = [
        {
            "timestamp": "2022-01-01T00:00:00Z",
            "source_ip": "192.0.2.1",
            "event_type": "GET",
            "resource": "/first",
            "status": "200",
            "bytes": "10",
        },
        {
            "timestamp": "2022-01-01T00:01:00Z",
            "source_ip": "192.0.2.1",
            "event_type": "POST",
            "resource": "/second",
            "status": "403",
            "bytes": "20",
        },
        {
            "timestamp": "2022-01-01T00:02:00Z",
            "source_ip": "192.0.2.1",
            "event_type": "POST",
            "resource": "/secret",
            "status": "500",
            "bytes": "30",
        },
    ]
    prefix = extract_features(rows[:2], "quality-check", "synthetic", "synthetic")
    with_future = extract_features(
        rows, "quality-check", "synthetic", "synthetic"
    )
    extended_by_id = {record.event.event_id: record for record in with_future}
    passed = all(
        extended_by_id[record.event.event_id].features == record.features
        for record in prefix
    )
    return {
        "passed": passed,
        "method": (
            "Compare as-of-event feature vectors for a fixed event prefix with and "
            "without later events, including later failures, POSTs, and resources."
        ),
    }


def _make_schema_definition() -> tuple[list[dict], set[str]]:
    definitions = FEATURE_SCHEMA.get("features")
    if not isinstance(definitions, list):
        raise QualityAuditError("Feature schema does not define a feature list.")
    names = {
        definition.get("name")
        for definition in definitions
        if isinstance(definition, dict)
    }
    if len(names) != len(definitions) or None in names:
        raise QualityAuditError("Feature schema contains invalid or duplicate names.")
    return definitions, names


def _validate_feature_record(
    record: dict,
    definitions: list[dict],
    expected_names: set[str],
    feature_counts: Counter,
    violations: Counter,
    leakage_keys: set[str],
) -> None:
    event = record.get("event")
    features = record.get("features")
    if not isinstance(event, dict) or not isinstance(features, dict):
        violations["missing_event_or_feature_mapping"] += 1
        return

    actual_names = set(features)
    for name in actual_names - expected_names:
        violations["unexpected_feature_keys"] += 1
        if any(token in str(name).casefold() for token in _DIRECT_TARGET_TOKENS):
            leakage_keys.add(str(name))
    for name in expected_names - actual_names:
        violations["missing_feature_keys"] += 1

    for definition in definitions:
        name = definition["name"]
        if name not in features:
            continue
        value = features[name]
        feature_counts[f"records:{name}"] += 1
        if value is None:
            feature_counts[f"missing:{name}"] += 1
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            violations[f"non_numeric:{name}"] += 1
            continue
        if isinstance(value, float) and not math.isfinite(value):
            violations[f"non_finite:{name}"] += 1
            continue
        if definition.get("type") == "integer" and type(value) is not int:
            violations[f"non_integer:{name}"] += 1
        if name.endswith("_missing") or name == "sensitive_resource_indicator":
            if value not in (0, 1):
                violations[f"invalid_indicator:{name}"] += 1
        elif value < 0:
            violations[f"negative_value:{name}"] += 1

    indicators = (
        ("source_ip_missing", event.get("source_ip") is None),
        ("event_type_missing", event.get("event_type") is None),
        ("resource_missing", event.get("resource") is None),
        (
            "response_status_missing",
            not (
                isinstance(event.get("response_status"), str)
                and event["response_status"].isascii()
                and event["response_status"].isdigit()
            ),
        ),
        ("transfer_size_missing", event.get("transfer_size") is None),
        ("time_since_previous_event_missing", features.get("time_since_previous_event_seconds") is None),
    )
    for name, should_be_missing in indicators:
        if features.get(name) != int(should_be_missing):
            violations[f"missingness_indicator_mismatch:{name}"] += 1

    for name in (
        "request_frequency_5m",
        "failed_request_frequency_5m",
        "post_request_frequency_5m",
        "unique_resources_accessed",
    ):
        if (features.get(name) is None) != (event.get("source_ip") is None):
            violations[f"source_dependent_nullability_mismatch:{name}"] += 1
    request_frequency = features.get("request_frequency_5m")
    if isinstance(request_frequency, (int, float)) and not isinstance(
        request_frequency, bool
    ):
        for name in ("failed_request_frequency_5m", "post_request_frequency_5m"):
            count = features.get(name)
            if isinstance(count, (int, float)) and count > request_frequency:
                violations[f"frequency_exceeds_request_frequency:{name}"] += 1

    transfer_size = event.get("transfer_size")
    if transfer_size is not None and (
        type(transfer_size) is not int or transfer_size < 0
    ):
        violations["invalid_event_transfer_size"] += 1
    if features.get("transfer_size_bytes") != transfer_size:
        violations["transfer_size_feature_mismatch"] += 1

    resource = event.get("resource")
    expected_sensitive = int(
        isinstance(resource, str)
        and any(term in resource.casefold() for term in _SENSITIVE_RESOURCE_TERMS)
    )
    if features.get("sensitive_resource_indicator") != expected_sensitive:
        violations["sensitive_resource_indicator_mismatch"] += 1


def _decode_record(raw_line: bytes) -> dict | None:
    try:
        record = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    return record


def _source_map(prepared_manifest: dict, scenario_id: str) -> dict[str, dict]:
    entries = prepared_manifest.get("source_files")
    if not isinstance(entries, list):
        raise QualityAuditError(f"{scenario_id} prepared manifest lacks source_files.")
    sources = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise QualityAuditError(f"{scenario_id} has an invalid source entry.")
        name = entry.get("source_filename")
        if not isinstance(name, str) or not name or name in sources:
            raise QualityAuditError(f"{scenario_id} has invalid source filenames.")
        sources[name] = entry
    return sources


def _audit_scenario(
    scenario_id: str,
    label_manifest_path: Path,
    connection: sqlite3.Connection,
    definitions: list[dict],
    feature_names: set[str],
) -> dict:
    label_manifest_bytes, label_manifest = _read_json(
        label_manifest_path, "label manifest"
    )
    if label_manifest.get("scenario_id") != scenario_id:
        raise QualityAuditError(
            f"Label manifest scenario_id does not match {scenario_id}."
        )
    directory = label_manifest_path.parent
    prepared_manifest_path = directory / "manifest.json"
    prepared_manifest_bytes, prepared_manifest = _read_json(
        prepared_manifest_path, "prepared manifest"
    )
    processed = prepared_manifest.get("processed_output")
    labelled_output = label_manifest.get("labelled_output")
    label_input = label_manifest.get("input")
    if not isinstance(processed, dict) or not isinstance(labelled_output, dict):
        raise QualityAuditError(f"{scenario_id} output metadata is missing.")
    if not isinstance(label_input, dict):
        raise QualityAuditError(f"{scenario_id} label input metadata is missing.")

    prepared_path = directory / _safe_filename(
        processed.get("filename"), "prepared dataset"
    )
    labelled_path = directory / _safe_filename(
        labelled_output.get("filename"), "labelled dataset"
    )
    expected_prepared_hash = processed.get("sha256")
    expected_labelled_hash = labelled_output.get("sha256")
    actual_prepared_hash = _sha256_file(prepared_path)
    actual_labelled_hash = _sha256_file(labelled_path)
    prepared_manifest_hash = hashlib.sha256(prepared_manifest_bytes).hexdigest()
    integrity_failures = []
    if actual_prepared_hash != expected_prepared_hash:
        integrity_failures.append("prepared_dataset_sha256_mismatch")
    if actual_labelled_hash != expected_labelled_hash:
        integrity_failures.append("labelled_dataset_sha256_mismatch")
    if label_input.get("prepared_records_sha256") != expected_prepared_hash:
        integrity_failures.append("label_manifest_prepared_hash_mismatch")
    if label_input.get("prepared_manifest_sha256") != prepared_manifest_hash:
        integrity_failures.append("label_manifest_prepared_manifest_hash_mismatch")
    if prepared_manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        integrity_failures.append("feature_schema_version_mismatch")
    if label_manifest.get("label_schema_version") is None:
        integrity_failures.append("label_schema_version_missing")

    sources = _source_map(prepared_manifest, scenario_id)
    declared_source_stats = label_manifest.get("per_source_distributions")
    if not isinstance(declared_source_stats, list):
        raise QualityAuditError(f"{scenario_id} label manifest lacks source statistics.")
    source_stats = {
        item.get("source_filename"): item
        for item in declared_source_stats
        if isinstance(item, dict)
    }
    if set(source_stats) != set(sources):
        integrity_failures.append("per_source_manifest_coverage_mismatch")

    classes: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    source_classes: dict[str, Counter[str]] = defaultdict(Counter)
    prepared_source_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    feature_violations: Counter[str] = Counter()
    target_leakage_keys = {
        name
        for name in feature_names
        if any(token in name.casefold() for token in _DIRECT_TARGET_TOKENS)
    }
    provenance_violations = 0
    label_provenance_violations: Counter[str] = Counter()
    malformed_prepared_rows = 0
    malformed_labelled_rows = 0
    label_correspondence_mismatches = 0
    prepared_count = 0
    labelled_count = 0
    duplicate_counts = Counter()
    identity_buffer = []
    content_buffer = []
    record_buffer = []
    records_digest = hashlib.sha256()
    labelled_digest = hashlib.sha256()
    period = label_manifest.get("simulation_period", {})
    start = _timestamp(period.get("start_inclusive")) if isinstance(period, dict) else None
    end = _timestamp(period.get("end_exclusive")) if isinstance(period, dict) else None
    period_violations = 0
    label_policy = label_manifest.get("label_policy", {})
    policy_id = label_policy.get("policy_id") if isinstance(label_policy, dict) else None

    def flush_duplicates() -> None:
        for kind, buffer in (
            ("identity", identity_buffer),
            ("event_content", content_buffer),
            ("record", record_buffer),
        ):
            if buffer:
                cursor = connection.executemany(
                    "INSERT OR IGNORE INTO duplicate_keys(scenario, kind, value) "
                    "VALUES (?, ?, ?)",
                    buffer,
                )
                duplicate_counts[kind] += len(buffer) - cursor.rowcount
                buffer.clear()

    try:
        with prepared_path.open("rb") as prepared_file, labelled_path.open(
            "rb"
        ) as labelled_file:
            prepared_iter = iter(prepared_file)
            labelled_iter = iter(labelled_file)
            while True:
                prepared_line = next(prepared_iter, None)
                labelled_line = next(labelled_iter, None)
                if prepared_line is None and labelled_line is None:
                    break
                prepared_record = None
                labelled_record = None
                if prepared_line is not None:
                    prepared_count += 1
                    records_digest.update(prepared_line)
                    prepared_record = _decode_record(
                        prepared_line
                    )
                    if prepared_record is None:
                        malformed_prepared_rows += 1
                    else:
                        provenance = prepared_record.get("provenance")
                        event = prepared_record.get("event")
                        if not isinstance(provenance, dict) or not isinstance(
                            event, dict
                        ):
                            provenance_violations += 1
                        else:
                            source_name = provenance.get("source_filename")
                            source = sources.get(source_name)
                            if source is not None:
                                prepared_source_counts[source_name] += 1
                            if (
                                source is None
                                or provenance.get("scenario_id") != scenario_id
                                or provenance.get("source_sha256")
                                != source.get("source_sha256")
                            ):
                                provenance_violations += 1
                            if (
                                not isinstance(event.get("event_id"), str)
                                or not event.get("event_id")
                                or not isinstance(event.get("case_id"), (str, int))
                                or not str(event.get("case_id")).strip()
                                or not isinstance(event.get("evidence_id"), str)
                                or not event.get("evidence_id")
                                or not isinstance(
                                    event.get("original_evidence_reference"), str
                                )
                                or not event.get("original_evidence_reference")
                                or _timestamp(event.get("timestamp")) is None
                            ):
                                provenance_violations += 1
                            source_line_number = provenance.get("source_line_number")
                            if (
                                type(source_line_number) is not int
                                or source_line_number < 1
                            ):
                                provenance_violations += 1
                            _validate_feature_record(
                                prepared_record,
                                definitions,
                                feature_names,
                                feature_counts,
                                feature_violations,
                                target_leakage_keys,
                            )
                            if isinstance(source_name, str):
                                event_id = event.get("event_id")
                                identity_buffer.append(
                                    (
                                        scenario_id,
                                        "identity",
                                        _fingerprint([source_name, event_id]),
                                    )
                                )
                                event_payload = {
                                    key: value
                                    for key, value in event.items()
                                    if key
                                    not in (
                                        "case_id",
                                        "evidence_id",
                                        "event_id",
                                        "original_evidence_reference",
                                    )
                                }
                                content_buffer.append(
                                    (
                                        scenario_id,
                                        "event_content",
                                        _fingerprint([source_name, event_payload]),
                                    )
                                )
                                record_buffer.append(
                                    (
                                        scenario_id,
                                        "record",
                                        _fingerprint(prepared_record),
                                    )
                                )
                                if len(identity_buffer) >= 4096:
                                    flush_duplicates()

                if labelled_line is not None:
                    labelled_count += 1
                    labelled_digest.update(labelled_line)
                    labelled_record = _decode_record(
                        labelled_line
                    )
                    if labelled_record is None:
                        malformed_labelled_rows += 1
                    elif prepared_record is None:
                        label_correspondence_mismatches += 1
                        _audit_label_record(
                            labelled_record,
                            scenario_id,
                            sources,
                            source_stats,
                            source_classes,
                            classes,
                            categories,
                            label_manifest,
                            label_provenance_violations,
                            start,
                            end,
                        )
                    else:
                        base_record = {
                            key: value
                            for key, value in labelled_record.items()
                            if key not in _DERIVED_LABEL_FIELDS
                        }
                        if base_record != prepared_record:
                            label_correspondence_mismatches += 1
                        _audit_label_record(
                            labelled_record,
                            scenario_id,
                            sources,
                            source_stats,
                            source_classes,
                            classes,
                            categories,
                            label_manifest,
                            label_provenance_violations,
                            start,
                            end,
                        )
        flush_duplicates()
    except OSError as exc:
        raise QualityAuditError(f"Unable to scan {scenario_id} datasets.") from exc

    if records_digest.hexdigest() != expected_prepared_hash:
        integrity_failures.append("prepared_dataset_stream_hash_mismatch")
    if labelled_digest.hexdigest() != expected_labelled_hash:
        integrity_failures.append("labelled_dataset_stream_hash_mismatch")
    period_violations = label_provenance_violations[
        "labelled_event_outside_simulation_period"
    ]
    input_counts = label_manifest.get("record_counts", {})
    if prepared_count != processed.get("record_count"):
        integrity_failures.append("prepared_record_count_mismatch")
    if prepared_count != input_counts.get("input_records"):
        integrity_failures.append("label_input_record_count_mismatch")
    if labelled_count != input_counts.get("evaluation_records"):
        integrity_failures.append("labelled_record_count_mismatch")
    if labelled_count != sum(classes.values()):
        integrity_failures.append("labelled_class_count_mismatch")
    if classes != Counter(label_manifest.get("evaluation_class_distribution", {})):
        integrity_failures.append("label_class_distribution_mismatch")
    if categories != Counter(label_manifest.get("per_category_counts", {})):
        integrity_failures.append("attack_category_count_mismatch")
    for source_name, source in sources.items():
        declared = source_stats.get(source_name, {}).get("classes", {})
        actual = {
            name: source_classes[source_name][name]
            for name in ("normal", "attack_associated", "unknown")
        }
        if declared != actual:
            integrity_failures.append(f"source_class_count_mismatch:{source_name}")
        if source.get("parsed_record_count") != prepared_source_counts[source_name]:
            integrity_failures.append(f"source_record_count_mismatch:{source_name}")

    if malformed_prepared_rows:
        integrity_failures.append("malformed_prepared_rows")
    if malformed_labelled_rows:
        integrity_failures.append("malformed_labelled_rows")
    if provenance_violations:
        integrity_failures.append("event_or_source_provenance_violations")
    if label_provenance_violations:
        integrity_failures.append("label_provenance_violations")
    if label_correspondence_mismatches:
        integrity_failures.append("prepared_labelled_record_correspondence_mismatch")
    if period_violations:
        integrity_failures.append("labelled_event_outside_simulation_period")
    if feature_violations:
        integrity_failures.append("feature_schema_or_value_violations")
    if target_leakage_keys:
        integrity_failures.append("direct_target_leakage_feature_keys")

    known_labels = classes["normal"] + classes["attack_associated"]
    known_label_sources = [
        source_name
        for source_name in sorted(sources)
        if source_classes[source_name]["normal"]
        or source_classes[source_name]["attack_associated"]
    ]
    known_source_counts = {
        source_name: {
            name: source_classes[source_name][name]
            for name in ("normal", "attack_associated", "unknown")
        }
        for source_name in sorted(sources)
    }
    return {
        "scenario_id": scenario_id,
        "integrity": {
            "passed": not integrity_failures,
            "failures": sorted(set(integrity_failures)),
            "prepared_dataset": {
                "expected_sha256": expected_prepared_hash,
                "actual_sha256": actual_prepared_hash,
            },
            "labelled_dataset": {
                "expected_sha256": expected_labelled_hash,
                "actual_sha256": actual_labelled_hash,
            },
            "prepared_manifest_sha256": prepared_manifest_hash,
            "label_manifest_sha256": hashlib.sha256(label_manifest_bytes).hexdigest(),
            "computed_prepared_records_sha256": records_digest.hexdigest(),
            "computed_labelled_records_sha256": labelled_digest.hexdigest(),
        },
        "record_counts": {
            "prepared": prepared_count,
            "labelled": labelled_count,
            "excluded_from_evaluation": max(0, prepared_count - labelled_count),
            "outside_period_violations": period_violations,
            "malformed_prepared_rows": malformed_prepared_rows,
            "malformed_labelled_rows": malformed_labelled_rows,
            "prepared_labelled_correspondence_mismatches": (
                label_correspondence_mismatches
            ),
        },
        "label_quality": {
            "class_distribution": {
                name: classes[name]
                for name in ("normal", "attack_associated", "unknown")
            },
            "known_label_coverage": (
                round(known_labels / labelled_count, 6) if labelled_count else 0.0
            ),
            "attack_to_normal_ratio": (
                round(classes["attack_associated"] / classes["normal"], 6)
                if classes["normal"]
                else None
            ),
            "attack_category_counts": dict(sorted(categories.items())),
            "sources_with_known_labels": known_label_sources,
            "known_label_source_count": len(known_label_sources),
            "per_source_class_distribution": known_source_counts,
            "label_provenance_violations": sum(
                label_provenance_violations.values()
            ),
            "label_provenance_violation_counts": dict(
                sorted(label_provenance_violations.items())
            ),
            "policy_id": policy_id,
        },
        "feature_quality": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "expected_feature_count": len(feature_names),
            "missing_value_counts": {
                definition["name"]: feature_counts[f"missing:{definition['name']}"]
                for definition in definitions
            },
            "feature_record_counts": {
                definition["name"]: feature_counts[f"records:{definition['name']}"]
                for definition in definitions
            },
            "violations": dict(sorted(feature_violations.items())),
            "target_leakage_feature_keys": sorted(target_leakage_keys),
            "potential_proxy_features": [
                "request_frequency_5m",
                "failed_request_frequency_5m",
                "post_request_frequency_5m",
                "sensitive_resource_indicator",
            ],
            "proxy_feature_note": (
                "These are event-derived behavioral indicators, not labels, but may "
                "correlate with attack activity and require held-out validation."
            ),
        },
        "duplicates": {
            "duplicate_source_event_identities": duplicate_counts["identity"],
            "duplicate_event_content_records": duplicate_counts["event_content"],
            "duplicate_prepared_records": duplicate_counts["record"],
            "identity_definition": "(scenario_id, source_filename, event_id)",
            "event_content_definition": (
                "Same source filename and canonical event fields excluding case, "
                "evidence, event ID, and original evidence reference."
            ),
            "event_content_interpretation": (
                "Repeated canonical content on distinct source rows may be "
                "legitimate repeated requests; it is not an exact prepared-record "
                "duplicate or a repeated source event identity."
            ),
        },
        "source_sha256": {
            name: entry.get("source_sha256")
            for name, entry in sorted(sources.items())
        },
    }


def _audit_label_record(
    record: dict,
    scenario_id: str,
    sources: dict[str, dict],
    source_stats: dict[str, dict],
    source_classes: dict[str, Counter[str]],
    classes: Counter[str],
    categories: Counter[str],
    label_manifest: dict,
    violations: Counter[str],
    start: datetime | None,
    end: datetime | None,
) -> None:
    provenance = record.get("provenance")
    label = record.get("research_label")
    label_name = record.get("research_label_name")
    expected_names = {0: "normal", 1: "attack_associated", None: "unknown"}
    if (
        not isinstance(provenance, dict)
        or provenance.get("scenario_id") != scenario_id
        or not isinstance(provenance.get("source_filename"), str)
    ):
        violations["invalid_event_provenance"] += 1
        return
    source_name = provenance["source_filename"]
    if source_name not in sources:
        violations["unknown_source"] += 1
        return
    if (label is not None and type(label) is not int) or label not in (0, 1, None):
        violations["invalid_research_label"] += 1
        return

    class_name = expected_names[label]
    classes[class_name] += 1
    source_classes[source_name][class_name] += 1
    ground_truth = record.get("ground_truth")
    source = sources[source_name]
    label_provenance = record.get("research_label_provenance")
    expected_policy = label_manifest.get("label_policy", {})
    expected_policy_id = (
        expected_policy.get("policy_id") if isinstance(expected_policy, dict) else None
    )
    if not isinstance(label_provenance, dict):
        violations["missing_label_provenance"] += 1
    else:
        if label_provenance.get("policy_id") != expected_policy_id:
            violations["label_policy_id_mismatch"] += 1
        if label_provenance.get("label_schema_version") != label_manifest.get(
            "label_schema_version"
        ):
            violations["label_schema_version_mismatch"] += 1
    if label == 1:
        ground_truth_provenance = record.get("ground_truth_provenance")
        ground_truth_labels = (
            ground_truth.get("labels") if isinstance(ground_truth, dict) else None
        )
        if (
            not isinstance(ground_truth_labels, list)
            or not ground_truth_labels
            or any(not isinstance(item, str) or not item for item in ground_truth_labels)
        ):
            violations["attack_without_explicit_categories"] += 1
        else:
            categories.update(ground_truth_labels)
        if (
            not isinstance(ground_truth_provenance, dict)
            or ground_truth_provenance.get("filename")
            != source.get("ground_truth_filename")
            or ground_truth_provenance.get("reference")
            != source.get("ground_truth_reference")
            or ground_truth_provenance.get("sha256")
            != source.get("ground_truth_sha256")
            or type(ground_truth_provenance.get("label_file_line_number")) is not int
            or ground_truth_provenance.get("label_file_line_number") < 1
            or not isinstance(label_provenance, dict)
            or label_provenance.get("ground_truth_reference")
            != source.get("ground_truth_reference")
            or label_provenance.get("ground_truth_sha256")
            != source.get("ground_truth_sha256")
            or record.get("research_label_basis") != "explicit_ground_truth"
        ):
            violations["attack_label_source_mismatch"] += 1
    if label == 0:
        if (
            source_stats.get(source_name, {}).get("ground_truth_coverage")
            != "verified"
            or ground_truth is not None
            or not isinstance(label_provenance, dict)
            or label_provenance.get("ground_truth_reference")
            != source.get("ground_truth_reference")
            or label_provenance.get("ground_truth_sha256")
            != source.get("ground_truth_sha256")
            or record.get("research_label_basis")
            != "absence_from_verified_ground_truth_file"
        ):
            violations["normal_label_without_verified_coverage"] += 1
    if label is None:
        categories_in_unknown = (
            ground_truth.get("labels")
            if isinstance(ground_truth, dict)
            else None
        )
        if categories_in_unknown:
            violations["unknown_with_explicit_attack_categories"] += 1
        if record.get("research_label_basis") not in (
            "insufficient_verified_label_coverage",
            "explicit_ground_truth_without_categories",
        ):
            violations["invalid_unknown_label_basis"] += 1
        if isinstance(label_provenance, dict) and (
            label_provenance.get("ground_truth_reference") is not None
            or label_provenance.get("ground_truth_sha256") is not None
        ):
            violations["unknown_has_label_reference"] += 1
    if label_name != class_name:
        violations["research_label_name_mismatch"] += 1

    event = record.get("event")
    event_time = _timestamp(event.get("timestamp")) if isinstance(event, dict) else None
    if (
        event_time is None
        or start is None
        or end is None
        or not start <= event_time < end
    ):
        violations["labelled_event_outside_simulation_period"] += 1


def _cross_scenario_similarity(
    path: Path,
    scenario_hashes: Mapping[str, str],
) -> dict:
    if not path.is_file():
        return {
            "available": False,
            "reason": "No prior scenario similarity analysis manifest was found.",
        }
    _, manifest = _read_json(path, "prior split analysis manifest")
    summaries = manifest.get("scenario_summaries")
    pairwise = manifest.get("cross_scenario_leakage_analysis")
    if not isinstance(summaries, dict) or not isinstance(pairwise, dict):
        return {
            "available": False,
            "reason": "Prior scenario similarity manifest has an incomplete schema.",
        }
    verified = all(
        scenario in summaries
        and summaries[scenario].get("labelled_records_sha256") == expected_hash
        for scenario, expected_hash in scenario_hashes.items()
    )
    return {
        "available": True,
        "input_hashes_verified": verified,
        "analysis": pairwise,
        "interpretation": (
            "Shared feature vectors and category vocabularies indicate repeated "
            "simulation patterns; they are not proof of duplicate events."
        ),
    }


def _write_if_unchanged(path: Path, content: bytes) -> None:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise QualityAuditError(f"Unable to read existing quality report: {path}") from exc
        if existing != content:
            raise FileExistsError(
                f"Refusing to overwrite a different quality report: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(content)
    except OSError as exc:
        raise QualityAuditError(f"Unable to write quality report: {path}") from exc


def create_quality_report(
    label_manifest_paths: Mapping[str, str | Path] | None = None,
    output_path: str | Path = QUALITY_REPORT_PATH,
    *,
    similarity_manifest_path: str | Path = SPLIT_OUTPUT_PATH,
) -> dict:
    """Verify prepared and labelled datasets and write a deterministic QA report."""
    manifest_paths = (
        DEFAULT_LABEL_MANIFEST_PATHS
        if label_manifest_paths is None
        else label_manifest_paths
    )
    if len(manifest_paths) != 3 or len(set(manifest_paths)) != 3:
        raise QualityAuditError("Exactly three distinct scenario manifests are required.")
    definitions, feature_names = _make_schema_definition()
    temporal_check = _temporal_prefix_check()

    with tempfile.TemporaryDirectory(prefix="forensic-quality-") as directory:
        database_path = Path(directory) / "duplicates.sqlite"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                "CREATE TABLE duplicate_keys (scenario TEXT NOT NULL, kind TEXT NOT NULL, "
                "value TEXT NOT NULL, PRIMARY KEY (scenario, kind, value)) WITHOUT ROWID"
            )
            results = {
                scenario: _audit_scenario(
                    scenario,
                    Path(manifest_paths[scenario]).resolve(),
                    connection,
                    definitions,
                    feature_names,
                )
                for scenario in sorted(manifest_paths)
            }
            connection.commit()
        finally:
            connection.close()

    failures = [
        f"{scenario}:{failure}"
        for scenario, result in results.items()
        for failure in result["integrity"]["failures"]
    ]
    scenario_hashes = {
        scenario: result["integrity"]["computed_labelled_records_sha256"]
        for scenario, result in results.items()
    }
    similarity = _cross_scenario_similarity(
        Path(similarity_manifest_path).resolve(), scenario_hashes
    )
    if not temporal_check["passed"]:
        failures.append("temporal_prefix_invariance_failed")
    if similarity.get("available") and not similarity.get("input_hashes_verified"):
        failures.append("cross_scenario_similarity_input_hash_mismatch")

    total_classes = {
        name: sum(
            result["label_quality"]["class_distribution"][name]
            for result in results.values()
        )
        for name in ("normal", "attack_associated", "unknown")
    }
    limitations = [
        "Class imbalance and unknown-label volume vary substantially by scenario.",
        "Verified binary labels are concentrated in one source per scenario.",
        "Three scenarios do not support replicated scenario-level estimates.",
        "Repeated feature patterns and shared attack-category vocabularies limit claims of cross-scenario independence.",
        "Behavioral proxy features may correlate with attack labels and require held-out evaluation.",
    ]
    report = {
        "report_schema_version": QUALITY_REPORT_SCHEMA_VERSION,
        "quality_status": (
            "failed"
            if failures
            else "passed_with_research_limitations"
            if limitations
            else "passed"
        ),
        "quality_failures": sorted(set(failures)),
        "methodology": (
            "Stream prepared and labelled JSONL records; verify manifest and file "
            "SHA-256 values, source/label provenance, class/category counts, feature "
            "schema and value invariants, duplicate identities/content, and "
            "prepared-to-labelled record correspondence. Do not modify inputs."
        ),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "scenario_results": results,
        "overall_class_distribution": total_classes,
        "temporal_feature_leakage_check": temporal_check,
        "cross_scenario_similarity": similarity,
        "research_limitations": limitations,
        "recommendations": [
            "Resolve any reported integrity, schema, provenance, or target-leakage failures before model work.",
            "Retain unknown labels; do not interpret them as normal.",
            "Evaluate generalization with additional independently sourced scenarios before making broad claims.",
            "Fit all future preprocessing only on training data and evaluate proxy features on held-out scenarios.",
        ],
        "reproducibility": {
            "deterministic": True,
            "random_seed": None,
            "input_scenario_ids": sorted(results),
            "dataset_records_modified": False,
            "raw_evidence_modified": False,
        },
    }
    content = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_if_unchanged(Path(output_path).resolve(), content)
    return report
