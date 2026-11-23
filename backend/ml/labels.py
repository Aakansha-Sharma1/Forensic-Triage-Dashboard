from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from .dataset import AIT_LDS_LICENSE, AIT_LDS_SOURCE_URL, AIT_LDS_VERSION, RAW_DATA_DIR


LABEL_SCHEMA_VERSION = "1.0.0"
LABEL_POLICY_ID = "ait-lds-v2.1-explicit-ground-truth-v1"
RUSSELLMITCHELL_SIMULATION_START = datetime(2022, 1, 21, tzinfo=timezone.utc)
RUSSELLMITCHELL_SIMULATION_END = datetime(2022, 1, 25, tzinfo=timezone.utc)
LABELLED_RECORDS_FILENAME = "labelled_events.jsonl"
LABEL_MANIFEST_FILENAME = "labels_manifest.json"

_CLASS_NAMES = {
    0: "normal",
    1: "attack_associated",
    None: "unknown",
}


class LabelPreparationError(ValueError):
    """Raised when prepared events or label provenance cannot be verified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc_timestamp(value: object, record_number: int) -> datetime:
    if not isinstance(value, str):
        raise LabelPreparationError(
            f"Record {record_number}: event timestamp must be an ISO-8601 string."
        )
    timestamp_text = value.strip()
    if timestamp_text.endswith("Z"):
        timestamp_text = timestamp_text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp_text)
    except ValueError as exc:
        raise LabelPreparationError(
            f"Record {record_number}: event timestamp is invalid."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized_relative_path(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return path.parts


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _source_entries(source_manifest: Mapping[str, object]) -> dict[str, dict]:
    entries = source_manifest.get("source_files")
    if not isinstance(entries, list):
        raise LabelPreparationError("Source manifest must contain a source_files list.")

    sources: dict[str, dict] = {}
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise LabelPreparationError(
                f"Source manifest entry {index} must be an object."
            )
        source_filename = entry.get("source_filename")
        source_parts = _normalized_relative_path(source_filename)
        if source_parts is None:
            raise LabelPreparationError(
                f"Source manifest entry {index} has an invalid source filename."
            )
        normalized_filename = "/".join(source_parts)
        if normalized_filename in sources:
            raise LabelPreparationError(
                f"Source manifest contains duplicate source filename: {normalized_filename}"
            )
        if not _is_sha256(entry.get("source_sha256")):
            raise LabelPreparationError(
                f"Source manifest has an invalid source hash: {normalized_filename}"
            )

        label_reference = entry.get("ground_truth_reference")
        label_hash = entry.get("ground_truth_sha256")
        label_filename = entry.get("ground_truth_filename")
        label_parts = _normalized_relative_path(label_reference)
        mirrored_path = None
        if label_parts is not None and "labels" in label_parts:
            labels_index = len(label_parts) - 1 - tuple(reversed(label_parts)).index("labels")
            mirrored_path = label_parts[labels_index + 1 :]
        mirrored = (
            mirrored_path == source_parts
            and label_filename == source_parts[-1]
            and _is_sha256(label_hash)
        )
        counts_are_complete = (
            isinstance(entry.get("ground_truth_physical_line_count"), int)
            and not isinstance(entry.get("ground_truth_physical_line_count"), bool)
            and entry.get("ground_truth_physical_line_count")
            == entry.get("ground_truth_record_count")
            == entry.get("attached_ground_truth_count")
            and isinstance(entry.get("rejected_row_count"), int)
            and entry.get("rejected_row_count") == 0
            and entry.get("rejected_log_row_count") == 0
            and entry.get("parsed_record_count") == entry.get("physical_line_count")
        )
        entry["_normalized_source_filename"] = normalized_filename
        entry["_label_coverage_verified"] = bool(mirrored and counts_are_complete)
        entry["_label_file_present"] = bool(
            label_reference and label_filename and label_hash
        )
        sources[normalized_filename] = entry
    return sources


class _LabelPreparation:
    def __init__(
        self,
        source_manifest: Mapping[str, object],
        *,
        scenario_id: str,
        simulation_start: datetime,
        simulation_end: datetime,
    ) -> None:
        self.source_manifest = source_manifest
        self.sources = _source_entries(source_manifest)
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise LabelPreparationError("scenario_id must be a non-empty string.")
        if not isinstance(simulation_start, datetime) or not isinstance(
            simulation_end, datetime
        ):
            raise LabelPreparationError("Simulation boundaries must be datetimes.")
        if simulation_start.tzinfo is None:
            simulation_start = simulation_start.replace(tzinfo=timezone.utc)
        if simulation_end.tzinfo is None:
            simulation_end = simulation_end.replace(tzinfo=timezone.utc)
        self.simulation_start = simulation_start.astimezone(timezone.utc)
        self.simulation_end = simulation_end.astimezone(timezone.utc)
        if self.simulation_start >= self.simulation_end:
            raise LabelPreparationError(
                "Simulation start must be earlier than simulation end."
            )
        self.scenario_id = scenario_id.strip()
        source_scenarios = {
            entry.get("scenario_id")
            for entry in self.sources.values()
            if entry.get("scenario_id") is not None
        }
        if source_scenarios != {self.scenario_id}:
            raise LabelPreparationError(
                "Requested scenario_id does not match the prepared source manifest."
            )
        self.seen_event_ids: set[str] = set()
        self.repeated_event_ids: Counter[str] = Counter()
        self.duplicate_examples: list[str] = []
        self.input_record_count = 0
        self.output_record_count = 0
        self.excluded_before_start = 0
        self.excluded_at_or_after_end = 0
        self.explicit_attack_excluded = 0
        self.class_counts: Counter[str] = Counter()
        self.category_counts: Counter[str] = Counter()
        self.source_counts: dict[str, dict] = {}
        self._initialize_source_counts()

    def _initialize_source_counts(self) -> None:
        for source_filename, entry in self.sources.items():
            if entry["_label_coverage_verified"]:
                coverage = "verified"
            elif entry["_label_file_present"]:
                coverage = "incomplete"
            else:
                coverage = "unavailable"
            self.source_counts[source_filename] = {
                "source_sha256": entry["source_sha256"],
                "ground_truth_coverage": coverage,
                "ground_truth_sha256": entry.get("ground_truth_sha256"),
                "evaluation_records": 0,
                "classes": {"normal": 0, "attack_associated": 0, "unknown": 0},
                "explicit_attack_records": 0,
                "excluded_outside_simulation_period": 0,
            }

    def add(self, source_record: object, record_number: int) -> dict | None:
        self.input_record_count += 1
        if not isinstance(source_record, dict):
            raise LabelPreparationError(
                f"Record {record_number}: prepared event must be a JSON object."
            )
        event = source_record.get("event")
        provenance = source_record.get("provenance")
        if not isinstance(event, dict) or not isinstance(provenance, dict):
            raise LabelPreparationError(
                f"Record {record_number}: event and provenance objects are required."
            )
        features = source_record.get("features")
        if not isinstance(features, dict):
            raise LabelPreparationError(
                f"Record {record_number}: numerical features must be an object."
            )
        if any(
            name in features
            for name in ("research_label", "research_label_name", "ground_truth")
        ):
            raise LabelPreparationError(
                f"Record {record_number}: labels must remain outside numerical features."
            )

        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise LabelPreparationError(
                f"Record {record_number}: event_id must be a non-empty string."
            )
        if event_id in self.seen_event_ids:
            self.repeated_event_ids[event_id] += 1
            if len(self.duplicate_examples) < 10 and event_id not in self.duplicate_examples:
                self.duplicate_examples.append(event_id)
        else:
            self.seen_event_ids.add(event_id)

        source_parts = _normalized_relative_path(provenance.get("source_filename"))
        if source_parts is None:
            raise LabelPreparationError(
                f"Record {record_number}: provenance source_filename is invalid."
            )
        source_filename = "/".join(source_parts)
        source = self.sources.get(source_filename)
        if source is None:
            raise LabelPreparationError(
                f"Record {record_number}: source is missing from source manifest: {source_filename}"
            )
        if provenance.get("source_sha256") != source["source_sha256"]:
            raise LabelPreparationError(
                f"Record {record_number}: source hash does not match the source manifest."
            )
        source_line_number = provenance.get("source_line_number")
        if (
            isinstance(source_line_number, bool)
            or not isinstance(source_line_number, int)
            or source_line_number < 1
            or source_line_number > source.get("physical_line_count", 0)
        ):
            raise LabelPreparationError(
                f"Record {record_number}: source line number is invalid."
            )

        ground_truth = source_record.get("ground_truth")
        ground_truth_provenance = source_record.get("ground_truth_provenance")
        labels: list[str] = []
        if ground_truth is not None:
            if not isinstance(ground_truth, dict):
                raise LabelPreparationError(
                    f"Record {record_number}: ground_truth must be an object or null."
                )
            raw_labels = ground_truth.get("labels")
            if not isinstance(raw_labels, list) or any(
                not isinstance(label, str) or not label.strip()
                for label in raw_labels
            ):
                raise LabelPreparationError(
                    f"Record {record_number}: ground-truth labels must be non-empty strings."
                )
            if not isinstance(ground_truth_provenance, dict):
                raise LabelPreparationError(
                    f"Record {record_number}: explicit labels require provenance."
                )
            if (
                ground_truth_provenance.get("reference")
                != source.get("ground_truth_reference")
                or ground_truth_provenance.get("sha256")
                != source.get("ground_truth_sha256")
                or ground_truth_provenance.get("filename")
                != source.get("ground_truth_filename")
            ):
                raise LabelPreparationError(
                    f"Record {record_number}: ground-truth provenance does not match the source manifest."
                )
            label_file_line_number = ground_truth_provenance.get(
                "label_file_line_number"
            )
            if (
                isinstance(label_file_line_number, bool)
                or not isinstance(label_file_line_number, int)
                or label_file_line_number < 1
                or label_file_line_number
                > source.get("ground_truth_physical_line_count", 0)
            ):
                raise LabelPreparationError(
                    f"Record {record_number}: ground-truth label line provenance is invalid."
                )
            labels = list(raw_labels)
        elif ground_truth_provenance is not None:
            raise LabelPreparationError(
                f"Record {record_number}: ground-truth provenance exists without labels."
            )

        timestamp = _parse_utc_timestamp(event.get("timestamp"), record_number)
        if timestamp < self.simulation_start:
            self.excluded_before_start += 1
            self.source_counts[source_filename][
                "excluded_outside_simulation_period"
            ] += 1
            self.explicit_attack_excluded += int(bool(labels))
            return None
        if timestamp >= self.simulation_end:
            self.excluded_at_or_after_end += 1
            self.source_counts[source_filename][
                "excluded_outside_simulation_period"
            ] += 1
            self.explicit_attack_excluded += int(bool(labels))
            return None

        if labels:
            label_value = 1
            basis = "explicit_ground_truth"
        elif ground_truth is not None:
            label_value = None
            basis = "explicit_ground_truth_without_categories"
        elif source["_label_coverage_verified"]:
            label_value = 0
            basis = "absence_from_verified_ground_truth_file"
        else:
            label_value = None
            basis = "insufficient_verified_label_coverage"

        class_name = _CLASS_NAMES[label_value]
        self.output_record_count += 1
        self.class_counts[class_name] += 1
        source_counts = self.source_counts[source_filename]
        source_counts["evaluation_records"] += 1
        source_counts["classes"][class_name] += 1
        source_counts["explicit_attack_records"] += int(label_value == 1)
        for category in labels if label_value == 1 else ():
            self.category_counts[category] += 1

        output_record = dict(source_record)
        output_record["research_label"] = label_value
        output_record["research_label_name"] = class_name
        output_record["research_label_basis"] = basis
        output_record["research_label_provenance"] = {
            "policy_id": LABEL_POLICY_ID,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "ground_truth_reference": (
                ground_truth_provenance["reference"]
                if label_value == 1
                else source.get("ground_truth_reference")
                if label_value == 0
                else None
            ),
            "ground_truth_sha256": (
                ground_truth_provenance["sha256"]
                if label_value == 1
                else source.get("ground_truth_sha256")
                if label_value == 0
                else None
            ),
        }
        return output_record

    def create_manifest(
        self,
        *,
        input_records_sha256: str,
        input_manifest_sha256: str,
        labelled_records_sha256: str,
    ) -> dict:
        class_counts = {
            name: self.class_counts[name]
            for name in ("normal", "attack_associated", "unknown")
        }
        known_count = class_counts["normal"] + class_counts["attack_associated"]
        event_sources = [
            source
            for source in self.source_counts.values()
            if source["evaluation_records"] or source["excluded_outside_simulation_period"]
        ]
        known_label_sources = sum(
            bool(
                source["classes"]["normal"]
                or source["classes"]["attack_associated"]
            )
            for source in event_sources
        )
        verified_sources = sum(
            source["ground_truth_coverage"] == "verified"
            for source in event_sources
        )
        source_bias_warning = (
            "Known labels come from one or fewer Apache source files; results are "
            "source-specific and should not be treated as representative of all hosts."
            if known_label_sources <= 1 and known_count
            else None
        )
        source_distributions = [
            {"source_filename": filename, **self.source_counts[filename]}
            for filename in sorted(self.source_counts)
        ]
        input_counts = {
            "input_records": self.input_record_count,
            "evaluation_records": self.output_record_count,
            "excluded_outside_simulation_period": (
                self.excluded_before_start + self.excluded_at_or_after_end
            ),
            "excluded_before_start": self.excluded_before_start,
            "excluded_at_or_after_end": self.excluded_at_or_after_end,
            "explicit_attack_records_excluded": self.explicit_attack_excluded,
        }
        unique_repeated = len(self.repeated_event_ids)
        duplicate_rows = sum(self.repeated_event_ids.values())
        return {
            "manifest_schema_version": LABEL_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "dataset_name": self.source_manifest.get("dataset_name", "AIT Log Data Set"),
            "source_url": self.source_manifest.get("source_url", AIT_LDS_SOURCE_URL),
            "version": self.source_manifest.get("version", AIT_LDS_VERSION),
            "license": self.source_manifest.get("license", AIT_LDS_LICENSE),
            "scenario_id": self.scenario_id,
            "label_policy": {
                "policy_id": LABEL_POLICY_ID,
                "source": AIT_LDS_SOURCE_URL,
                "attack": "Assign 1 only when the corresponding source event has explicit non-empty ground-truth labels.",
                "normal": (
                    "Assign 0 only when the event is absent from its corresponding, "
                    "verified and fully aligned ground-truth file. The official AIT "
                    "policy states that omitted events are normal."
                ),
                "unknown": (
                    "Use null when there is insufficient verified ground-truth "
                    "coverage or when an explicit label has no categories."
                ),
                "multi_label_categories": (
                    "Original ground_truth labels, labeling rules and provenance are "
                    "preserved without collapsing or modifying the source payload."
                ),
            },
            "simulation_period": {
                "start_inclusive": self.simulation_start.isoformat(),
                "end_exclusive": self.simulation_end.isoformat(),
                "timezone": "UTC",
                "source": (
                    f"Official Zenodo record 19483937 {self.scenario_id} "
                    "simulation-time table and scenario metadata; timezone is "
                    "normalized to UTC to match event timestamps."
                ),
                "exclusion_rule": (
                    "Events before the start or at/after the end are excluded from "
                    "labelled_events.jsonl."
                ),
            },
            "input": {
                "prepared_records_sha256": input_records_sha256,
                "prepared_manifest_sha256": input_manifest_sha256,
                "source_archive": self.source_manifest.get("source_archive"),
            },
            "labelled_output": {
                "filename": LABELLED_RECORDS_FILENAME,
                "sha256": labelled_records_sha256,
            },
            "record_counts": input_counts,
            "evaluation_class_distribution": class_counts,
            "known_class_percentages": {
                "normal": round(class_counts["normal"] * 100 / known_count, 6)
                if known_count
                else 0.0,
                "attack_associated": round(
                    class_counts["attack_associated"] * 100 / known_count, 6
                )
                if known_count
                else 0.0,
            },
            "per_category_counts": dict(sorted(self.category_counts.items())),
            "per_source_distributions": source_distributions,
            "duplicate_event_identifiers": {
                "unique_identifiers_repeated": unique_repeated,
                "duplicate_rows_after_first": duplicate_rows,
                "examples": sorted(self.duplicate_examples),
            },
            "source_bias_assessment": {
                "event_bearing_source_count": len(event_sources),
                "sources_with_verified_label_coverage": verified_sources,
                "sources_contributing_known_labels": known_label_sources,
                "known_label_coverage_fraction": round(
                    known_label_sources / len(event_sources), 6
                )
                if event_sources
                else 0.0,
                "potential_bias": bool(source_bias_warning),
                "warning": source_bias_warning,
            },
            "methodology": (
                "Read prepared AIT events without changing their features. Verify each "
                "source filename and SHA-256 against the import manifest, preserve "
                "explicit labels and provenance, and derive normal labels only for "
                "events absent from a complete, mirrored label file with zero import "
                "rejections. Retain unknowns for uncovered sources. Filter evaluation "
                "records to the configured scenario's official UTC simulation interval. "
                "No detection-rule findings or model outputs are used."
            ),
        }


def _write_if_unchanged(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to overwrite existing research output: {path}"
            )
        return
    created = False
    try:
        with path.open("xb") as output:
            created = True
            output.write(content)
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


def _copy_if_unchanged(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise FileExistsError(
                f"Refusing to overwrite existing research output: {destination}"
            )
        return
    created = False
    try:
        with source.open("rb") as input_file, destination.open("xb") as output:
            created = True
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                output.write(chunk)
    except Exception:
        if created and destination.exists():
            destination.unlink()
        raise


def prepare_labelled_dataset(
    prepared_records_path: str | Path,
    prepared_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    scenario_id: str = "russellmitchell",
    simulation_start: datetime = RUSSELLMITCHELL_SIMULATION_START,
    simulation_end: datetime = RUSSELLMITCHELL_SIMULATION_END,
) -> dict:
    """Write a separately labelled, simulation-window-only evaluation dataset."""
    records_path = Path(prepared_records_path).resolve()
    manifest_path = Path(prepared_manifest_path).resolve()
    destination_dir = Path(output_dir).resolve()
    raw_root = RAW_DATA_DIR.resolve()
    if destination_dir == raw_root or destination_dir.is_relative_to(raw_root):
        raise LabelPreparationError("Labelled output must not be written under raw data.")

    output_records_path = destination_dir / LABELLED_RECORDS_FILENAME
    output_manifest_path = destination_dir / LABEL_MANIFEST_FILENAME
    if output_records_path == records_path or output_manifest_path == manifest_path:
        raise LabelPreparationError(
            "Labelled output must be separate from the prepared dataset and manifest."
        )

    try:
        manifest_bytes = manifest_path.read_bytes()
        source_manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelPreparationError("Unable to read the prepared dataset manifest.") from exc
    if not isinstance(source_manifest, dict):
        raise LabelPreparationError("Prepared dataset manifest must be a JSON object.")
    processed_output = source_manifest.get("processed_output")
    expected_input_hash = (
        processed_output.get("sha256") if isinstance(processed_output, dict) else None
    )
    if not _is_sha256(expected_input_hash):
        raise LabelPreparationError(
            "Prepared dataset manifest must contain a valid processed-output SHA-256."
        )

    run = _LabelPreparation(
        source_manifest,
        scenario_id=scenario_id,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".labelled_events.", suffix=".tmp", dir=destination_dir
    )
    temporary_path = Path(temporary_name)
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as output, records_path.open("rb") as source:
            for record_number, raw_line in enumerate(source, start=1):
                input_digest.update(raw_line)
                try:
                    source_record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise LabelPreparationError(
                        f"Prepared event line {record_number} is not valid UTF-8 JSON."
                    ) from exc
                labelled_record = run.add(source_record, record_number)
                if labelled_record is None:
                    continue
                serialized = (
                    json.dumps(
                        labelled_record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                output.write(serialized)
                output_digest.update(serialized)

        input_sha256 = input_digest.hexdigest()
        if input_sha256 != expected_input_hash:
            raise LabelPreparationError(
                "Prepared event file SHA-256 does not match its manifest."
            )
        output_sha256 = output_digest.hexdigest()
        result_manifest = run.create_manifest(
            input_records_sha256=input_sha256,
            input_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            labelled_records_sha256=output_sha256,
        )
        manifest_content = (
            json.dumps(result_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        _copy_if_unchanged(temporary_path, output_records_path, output_sha256)
        _write_if_unchanged(output_manifest_path, manifest_content)
        return result_manifest
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
