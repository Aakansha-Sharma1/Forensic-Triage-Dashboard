from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .schema import CanonicalEvent


AIT_LDS_SOURCE_URL = "https://zenodo.org/records/19483937"
AIT_LDS_VERSION = "v2_1"
AIT_LDS_LICENSE = "CC-BY-NC-SA-4.0"
DATASET_METHOD = (
    "Parse Apache access-log records in source-file line order; retain one-based "
    "source line and SHA-256; attach only explicit line-referenced labels from "
    "the corresponding ground-truth JSONL file; report all malformed and "
    "unmatched rows without inferring labels."
)

RESEARCH_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "research_datasets"
    / "ait-lds-v2.1"
)
RAW_DATA_DIR = RESEARCH_DATA_ROOT / "raw"
PROCESSED_DATA_DIR = RESEARCH_DATA_ROOT / "processed"

_APACHE_LINE = re.compile(
    r'^(?P<remote_host>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'(?P<request>"(?:\\.|[^"\\])*")\s+'
    r'(?P<status>\d{3}|-)\s+(?P<bytes>\d+|-)(?:\s+.*)?$'
)
_APACHE_TIMESTAMP = re.compile(
    r"^(?P<day>\d{2})/(?P<month>[A-Za-z]{3})/(?P<year>\d{4}):"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}) "
    r"(?P<offset>[+-]\d{4})$"
)
_MONTH_NUMBERS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


class DatasetImportError(ValueError):
    """Raised when a source file or import configuration cannot be processed."""


@dataclass(frozen=True)
class RejectedRow:
    source_filename: str
    source_line_number: int
    reason: str


@dataclass(frozen=True)
class ApacheLogRecord:
    event: CanonicalEvent
    source_filename: str
    source_line_number: int
    scenario_id: str | None
    source_sha256: str


@dataclass(frozen=True)
class ApacheLogImport:
    source_filename: str
    source_reference: str
    scenario_id: str | None
    source_sha256: str
    physical_line_count: int
    records: tuple[ApacheLogRecord, ...]
    rejected_rows: tuple[RejectedRow, ...]


@dataclass(frozen=True)
class GroundTruthLabel:
    event_line_number: int
    label_file_line_number: int
    payload: dict


@dataclass(frozen=True)
class GroundTruthImport:
    source_filename: str
    label_filename: str
    label_reference: str
    label_sha256: str
    physical_line_count: int
    labels: tuple[GroundTruthLabel, ...]
    rejected_rows: tuple[RejectedRow, ...]


@dataclass(frozen=True)
class PreparedDatasetRecord:
    event: CanonicalEvent
    source_filename: str
    source_line_number: int
    scenario_id: str | None
    source_sha256: str
    ground_truth: dict | None
    ground_truth_filename: str | None
    ground_truth_reference: str | None
    ground_truth_sha256: str | None
    ground_truth_label_file_line_number: int | None

    def to_dict(self) -> dict:
        return {
            "event": {
                "case_id": self.event.case_id,
                "evidence_id": self.event.evidence_id,
                "event_id": self.event.event_id,
                "timestamp": self.event.timestamp.isoformat(),
                "source_ip": self.event.source_ip,
                "event_type": self.event.event_type,
                "resource": self.event.resource,
                "response_status": self.event.response_status,
                "transfer_size": self.event.transfer_size,
                "original_evidence_reference": self.event.original_evidence_reference,
            },
            "provenance": {
                "source_filename": self.source_filename,
                "source_line_number": self.source_line_number,
                "scenario_id": self.scenario_id,
                "source_sha256": self.source_sha256,
            },
            "ground_truth": deepcopy(self.ground_truth),
            "ground_truth_provenance": (
                {
                    "filename": self.ground_truth_filename,
                    "reference": self.ground_truth_reference,
                    "sha256": self.ground_truth_sha256,
                    "label_file_line_number": self.ground_truth_label_file_line_number,
                }
                if self.ground_truth_filename is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PreparedApacheSource:
    apache_import: ApacheLogImport
    ground_truth_import: GroundTruthImport | None
    records: tuple[PreparedDatasetRecord, ...]
    rejected_rows: tuple[RejectedRow, ...]


def _read_utf8_file(path: Path) -> tuple[bytes, str]:
    content = path.read_bytes()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetImportError(f"{path.name}: input is not valid UTF-8.") from exc
    return content, text


def _parse_apache_timestamp(value: str) -> datetime:
    match = _APACHE_TIMESTAMP.fullmatch(value)
    if not match:
        raise ValueError("invalid Apache timestamp")

    month = _MONTH_NUMBERS.get(match.group("month"))
    if month is None:
        raise ValueError("invalid Apache timestamp month")

    offset_text = match.group("offset")
    offset_sign = 1 if offset_text[0] == "+" else -1
    offset_hours = int(offset_text[1:3])
    offset_minutes = int(offset_text[3:5])
    if offset_hours > 23 or offset_minutes > 59:
        raise ValueError("invalid Apache timestamp offset")
    offset = timezone(
        offset_sign * timedelta(hours=offset_hours, minutes=offset_minutes)
    )
    parsed = datetime(
        int(match.group("year")),
        month,
        int(match.group("day")),
        int(match.group("hour")),
        int(match.group("minute")),
        int(match.group("second")),
        tzinfo=offset,
    )
    return parsed.astimezone(timezone.utc)


def _parse_request(value: str) -> tuple[str | None, str | None]:
    request = value[1:-1].replace(r"\"", '"').replace(r"\\", "\\")
    if request == "-":
        return None, None
    parts = request.split()
    if len(parts) < 2:
        raise ValueError("invalid request field")
    return parts[0], parts[1]


def _normalize_source_filename(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        raise DatasetImportError("source_filename is required.")
    return normalized


def parse_apache_access_log(
    source_path: str | Path,
    *,
    case_id: int | str,
    evidence_id: str,
    scenario_id: str | None,
    source_filename: str | None = None,
) -> ApacheLogImport:
    """Parse a common/combined Apache access log without altering the source."""
    path = Path(source_path)
    content, text = _read_utf8_file(path)
    source_sha256 = hashlib.sha256(content).hexdigest()
    normalized_source_filename = _normalize_source_filename(
        source_filename if source_filename is not None else path.name
    )
    source_reference = str(path)
    normalized_scenario = str(scenario_id).strip() if scenario_id is not None else None
    if normalized_scenario == "":
        normalized_scenario = None

    records = []
    rejected_rows = []
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        match = _APACHE_LINE.fullmatch(line)
        if match is None:
            rejected_rows.append(
                RejectedRow(
                    normalized_source_filename, line_number, "malformed_apache_record"
                )
            )
            continue

        try:
            timestamp = _parse_apache_timestamp(match.group("timestamp"))
            event_type, resource = _parse_request(match.group("request"))
        except ValueError as exc:
            rejected_rows.append(
                RejectedRow(normalized_source_filename, line_number, str(exc))
            )
            continue

        remote_host = match.group("remote_host")
        status = match.group("status")
        transfer_size = match.group("bytes")
        if transfer_size != "-" and not transfer_size.isascii():
            rejected_rows.append(
                RejectedRow(
                    normalized_source_filename, line_number, "invalid_transfer_size"
                )
            )
            continue

        event = CanonicalEvent(
            case_id=case_id,
            evidence_id=evidence_id,
            event_id=f"{normalized_source_filename}:{line_number}",
            timestamp=timestamp,
            source_ip=None if remote_host == "-" else remote_host,
            event_type=event_type,
            resource=resource,
            response_status=None if status == "-" else status,
            transfer_size=None if transfer_size == "-" else int(transfer_size),
            original_evidence_reference=source_reference,
        )
        records.append(
            ApacheLogRecord(
                event=event,
                source_filename=normalized_source_filename,
                source_line_number=line_number,
                scenario_id=normalized_scenario,
                source_sha256=source_sha256,
            )
        )

    return ApacheLogImport(
        source_filename=normalized_source_filename,
        source_reference=source_reference,
        scenario_id=normalized_scenario,
        source_sha256=source_sha256,
        physical_line_count=len(lines),
        records=tuple(records),
        rejected_rows=tuple(rejected_rows),
    )


def import_ground_truth_labels(
    label_path: str | Path,
    *,
    source_filename: str,
) -> GroundTruthImport:
    """Load line-referenced JSONL labels verbatim; never infer missing labels."""
    path = Path(label_path)
    content, text = _read_utf8_file(path)
    label_sha256 = hashlib.sha256(content).hexdigest()
    labels = []
    rejected_rows = []
    seen_event_lines = set()
    lines = text.splitlines()

    for label_file_line, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            rejected_rows.append(
                RejectedRow(path.name, label_file_line, "malformed_label_json")
            )
            continue
        if not isinstance(payload, dict):
            rejected_rows.append(
                RejectedRow(path.name, label_file_line, "label_record_must_be_object")
            )
            continue

        event_line = payload.get("line")
        if (
            isinstance(event_line, bool)
            or not isinstance(event_line, int)
            or event_line < 1
        ):
            rejected_rows.append(
                RejectedRow(path.name, label_file_line, "invalid_event_line_reference")
            )
            continue
        if not isinstance(payload.get("labels"), list):
            rejected_rows.append(
                RejectedRow(path.name, label_file_line, "labels_must_be_list")
            )
            continue
        if event_line in seen_event_lines:
            rejected_rows.append(
                RejectedRow(path.name, label_file_line, "duplicate_event_line_label")
            )
            continue

        seen_event_lines.add(event_line)
        labels.append(
            GroundTruthLabel(
                event_line_number=event_line,
                label_file_line_number=label_file_line,
                payload=deepcopy(payload),
            )
        )

    return GroundTruthImport(
        source_filename=_normalize_source_filename(source_filename),
        label_filename=path.name,
        label_reference=str(path),
        label_sha256=label_sha256,
        physical_line_count=len(lines),
        labels=tuple(labels),
        rejected_rows=tuple(rejected_rows),
    )


def prepare_apache_source(
    apache_import: ApacheLogImport,
    ground_truth_import: GroundTruthImport | None = None,
) -> PreparedApacheSource:
    """Align explicit label lines to parsed events and report unmatched labels."""
    if (
        ground_truth_import is not None
        and ground_truth_import.source_filename != apache_import.source_filename
    ):
        raise DatasetImportError(
            "Ground-truth source filename does not match the Apache log filename."
        )

    records_by_line = {
        record.source_line_number: record for record in apache_import.records
    }
    labels_by_event_line = {}
    rejected_rows = list(apache_import.rejected_rows)
    if ground_truth_import is not None:
        rejected_rows.extend(ground_truth_import.rejected_rows)
        for label in ground_truth_import.labels:
            if label.event_line_number > apache_import.physical_line_count:
                rejected_rows.append(
                    RejectedRow(
                        ground_truth_import.label_filename,
                        label.label_file_line_number,
                        "label_line_reference_out_of_range",
                    )
                )
            elif label.event_line_number not in records_by_line:
                rejected_rows.append(
                    RejectedRow(
                        ground_truth_import.label_filename,
                        label.label_file_line_number,
                        "label_references_rejected_log_row",
                    )
                )
            else:
                labels_by_event_line[label.event_line_number] = label

    prepared_records = []
    for source_record in apache_import.records:
        label = labels_by_event_line.get(source_record.source_line_number)
        prepared_records.append(
            PreparedDatasetRecord(
                event=source_record.event,
                source_filename=source_record.source_filename,
                source_line_number=source_record.source_line_number,
                scenario_id=source_record.scenario_id,
                source_sha256=source_record.source_sha256,
                ground_truth=deepcopy(label.payload) if label else None,
                ground_truth_filename=(
                    ground_truth_import.label_filename if label and ground_truth_import else None
                ),
                ground_truth_reference=(
                    ground_truth_import.label_reference if label and ground_truth_import else None
                ),
                ground_truth_sha256=(
                    ground_truth_import.label_sha256 if label and ground_truth_import else None
                ),
                ground_truth_label_file_line_number=(
                    label.label_file_line_number if label else None
                ),
            )
        )

    return PreparedApacheSource(
        apache_import=apache_import,
        ground_truth_import=ground_truth_import,
        records=tuple(prepared_records),
        rejected_rows=tuple(rejected_rows),
    )


def create_dataset_manifest(
    sources: Iterable[PreparedApacheSource],
    *,
    source_url: str = AIT_LDS_SOURCE_URL,
    dataset_name: str = "AIT Log Data Set",
    version: str = AIT_LDS_VERSION,
    license_id: str = AIT_LDS_LICENSE,
    methodology: str = DATASET_METHOD,
    processed_records_filename: str | None = None,
    processed_records_sha256: str | None = None,
) -> dict:
    """Create deterministic provenance and record-count metadata."""
    ordered_sources = sorted(
        sources,
        key=lambda source: (
            source.apache_import.scenario_id or "",
            source.apache_import.source_filename,
            source.apache_import.source_reference,
        ),
    )
    file_entries = []
    parsed_count = rejected_count = label_count = 0
    for source in ordered_sources:
        log_import = source.apache_import
        label_import = source.ground_truth_import
        attached_labels = sum(record.ground_truth is not None for record in source.records)
        file_entries.append(
            {
                "source_filename": log_import.source_filename,
                "source_reference": log_import.source_reference,
                "scenario_id": log_import.scenario_id,
                "source_sha256": log_import.source_sha256,
                "physical_line_count": log_import.physical_line_count,
                "parsed_record_count": len(log_import.records),
                "rejected_log_row_count": len(log_import.rejected_rows),
                "ground_truth_filename": (
                    label_import.label_filename if label_import else None
                ),
                "ground_truth_reference": (
                    label_import.label_reference if label_import else None
                ),
                "ground_truth_sha256": label_import.label_sha256 if label_import else None,
                "ground_truth_physical_line_count": (
                    label_import.physical_line_count if label_import else 0
                ),
                "ground_truth_record_count": len(label_import.labels) if label_import else 0,
                "attached_ground_truth_count": attached_labels,
                "rejected_row_count": len(source.rejected_rows),
            }
        )
        parsed_count += len(source.records)
        rejected_count += len(source.rejected_rows)
        label_count += attached_labels

    manifest = {
        "manifest_schema_version": "1.0.0",
        "dataset_name": dataset_name,
        "version": version,
        "source_url": source_url,
        "license": license_id,
        "methodology": methodology,
        "source_files": file_entries,
        "record_counts": {
            "parsed_records": parsed_count,
            "attached_ground_truth_records": label_count,
            "rejected_rows": rejected_count,
        },
    }
    if processed_records_filename is not None:
        manifest["processed_output"] = {
            "filename": processed_records_filename,
            "sha256": processed_records_sha256,
            "record_count": parsed_count,
        }
    return manifest


def _write_if_unchanged(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to overwrite different processed dataset output: {path}"
            )
        return
    path.write_bytes(content)


def write_processed_dataset(
    sources: Iterable[PreparedApacheSource],
    output_dir: str | Path = PROCESSED_DATA_DIR,
    *,
    raw_data_dir: str | Path = RAW_DATA_DIR,
    source_url: str = AIT_LDS_SOURCE_URL,
    dataset_name: str = "AIT Log Data Set",
    version: str = AIT_LDS_VERSION,
    license_id: str = AIT_LDS_LICENSE,
) -> dict:
    """Write deterministic JSONL and a manifest outside the raw-data directory."""
    ordered_sources = sorted(
        sources,
        key=lambda source: (
            source.apache_import.scenario_id or "",
            source.apache_import.source_filename,
            source.apache_import.source_reference,
        ),
    )
    target_dir = Path(output_dir).resolve()
    raw_dir = Path(raw_data_dir).resolve()
    if (
        target_dir == raw_dir
        or target_dir.is_relative_to(raw_dir)
        or raw_dir.is_relative_to(target_dir)
    ):
        raise DatasetImportError("Processed output and raw dataset directories must be separate.")

    records = [
        record.to_dict()
        for source in ordered_sources
        for record in sorted(source.records, key=lambda item: item.source_line_number)
    ]
    records_content = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    records_filename = "prepared_events.jsonl"
    records_sha256 = hashlib.sha256(records_content).hexdigest()
    manifest = create_dataset_manifest(
        ordered_sources,
        source_url=source_url,
        dataset_name=dataset_name,
        version=version,
        license_id=license_id,
        processed_records_filename=records_filename,
        processed_records_sha256=records_sha256,
    )
    manifest_content = (
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    target_dir.mkdir(parents=True, exist_ok=True)
    _write_if_unchanged(target_dir / records_filename, records_content)
    _write_if_unchanged(target_dir / "manifest.json", manifest_content)
    return manifest
