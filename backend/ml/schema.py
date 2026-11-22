from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike, fspath
from types import MappingProxyType
from typing import Iterable, Mapping


FEATURE_SCHEMA_VERSION = "1.0.0"
FEATURE_WINDOW_SECONDS = 300

FEATURE_SCHEMA = {
    "version": FEATURE_SCHEMA_VERSION,
    "frequency_window_seconds": FEATURE_WINDOW_SECONDS,
    "features": [
        {
            "name": "transfer_size_bytes",
            "type": "integer",
            "meaning": "Transfer size in bytes for this event.",
            "calculation": "Parsed from bytes; missing values remain null.",
        },
        {
            "name": "request_frequency_5m",
            "type": "integer",
            "meaning": "Requests from this source in the inclusive trailing five-minute window.",
            "calculation": "Count of source events at or before this event, including this event.",
        },
        {
            "name": "failed_request_frequency_5m",
            "type": "integer",
            "meaning": "HTTP failures from this source in the inclusive trailing five-minute window.",
            "calculation": "Count of numeric response codes from 400 through 599 in the window.",
        },
        {
            "name": "post_request_frequency_5m",
            "type": "integer",
            "meaning": "POST events from this source in the inclusive trailing five-minute window.",
            "calculation": "Count of event types containing POST, case-insensitively.",
        },
        {
            "name": "time_since_previous_event_seconds",
            "type": "number",
            "meaning": "Elapsed time since the preceding event from this source.",
            "calculation": "Current UTC timestamp minus the preceding timestamp; first event is null.",
        },
        {
            "name": "unique_resources_accessed",
            "type": "integer",
            "meaning": "Distinct non-empty resources accessed by this source through this event.",
            "calculation": "Count of stripped resource strings, compared case-insensitively.",
        },
        {
            "name": "sensitive_resource_indicator",
            "type": "integer",
            "meaning": "Whether a resource contains a configured sensitive term.",
            "calculation": "1 if any of secret, admin, password, confidential, or .zip occurs case-insensitively; otherwise 0.",
        },
        {
            "name": "source_ip_missing",
            "type": "integer",
            "meaning": "Whether the source IP is missing.",
            "calculation": "1 when source_ip/source is blank; otherwise 0.",
        },
        {
            "name": "event_type_missing",
            "type": "integer",
            "meaning": "Whether the event type is missing.",
            "calculation": "1 when event_type/event is blank; otherwise 0.",
        },
        {
            "name": "resource_missing",
            "type": "integer",
            "meaning": "Whether the resource is missing.",
            "calculation": "1 when resource/description is blank; otherwise 0.",
        },
        {
            "name": "response_status_missing",
            "type": "integer",
            "meaning": "Whether the response status is absent or not numeric.",
            "calculation": "1 for blank or non-integer status; otherwise 0.",
        },
        {
            "name": "transfer_size_missing",
            "type": "integer",
            "meaning": "Whether transfer size is absent.",
            "calculation": "1 when bytes is blank; otherwise 0.",
        },
        {
            "name": "time_since_previous_event_missing",
            "type": "integer",
            "meaning": "Whether no preceding event from this source exists.",
            "calculation": "1 when elapsed time is null; otherwise 0.",
        },
    ],
}

_COLUMN_ALIASES = {
    "timestamp": ("timestamp",),
    "source_ip": ("source_ip", "source"),
    "event_type": ("event_type", "event"),
    "resource": ("resource", "description"),
    "response_status": ("status",),
    "transfer_size": ("bytes",),
}


class EventDataError(ValueError):
    """Raised when an input event cannot be represented in the canonical schema."""


@dataclass(frozen=True)
class CanonicalEvent:
    case_id: int | str
    evidence_id: str
    event_id: str
    timestamp: datetime
    source_ip: str | None
    event_type: str | None
    resource: str | None
    response_status: str | None
    transfer_size: int | None
    original_evidence_reference: str

    def __post_init__(self) -> None:
        if isinstance(self.case_id, bool) or not str(self.case_id).strip():
            raise EventDataError("case_id is required.")
        if isinstance(self.case_id, str):
            object.__setattr__(self, "case_id", self.case_id.strip())
        normalized_evidence_id = str(self.evidence_id).strip()
        if not normalized_evidence_id:
            raise EventDataError("evidence_id is required.")
        normalized_event_id = str(self.event_id).strip()
        if not normalized_event_id:
            raise EventDataError("event_id is required.")
        object.__setattr__(self, "evidence_id", normalized_evidence_id)
        object.__setattr__(self, "event_id", normalized_event_id)
        if not isinstance(self.timestamp, datetime):
            raise EventDataError("timestamp must be a datetime.")
        normalized_timestamp = self.timestamp
        if normalized_timestamp.tzinfo is None:
            normalized_timestamp = normalized_timestamp.replace(tzinfo=timezone.utc)
        else:
            normalized_timestamp = normalized_timestamp.astimezone(timezone.utc)
        object.__setattr__(self, "timestamp", normalized_timestamp)
        if self.transfer_size is not None and (
            isinstance(self.transfer_size, bool)
            or not isinstance(self.transfer_size, int)
            or self.transfer_size < 0
        ):
            raise EventDataError("transfer_size must be a non-negative integer or missing.")
        normalized_reference = str(self.original_evidence_reference).strip()
        if not normalized_reference:
            raise EventDataError("original_evidence_reference is required.")
        object.__setattr__(self, "original_evidence_reference", normalized_reference)
        for field_name in ("source_ip", "event_type", "resource", "response_status"):
            object.__setattr__(
                self, field_name, _normalized_optional(getattr(self, field_name))
            )


@dataclass(frozen=True)
class FeatureRecord:
    event: CanonicalEvent
    features: Mapping[str, int | float | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    @property
    def model_inputs(self) -> dict[str, int | float | None]:
        """Return numerical features without case/evidence identifiers or file references."""
        return dict(self.features)


def get_feature_schema() -> dict:
    return deepcopy(FEATURE_SCHEMA)


def _normalized_optional(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _row_value(row: Mapping[str, object], logical_name: str) -> object:
    for column in _COLUMN_ALIASES[logical_name]:
        if column in row:
            return row[column]
    return None


def _parse_timestamp(value: object, row_number: int) -> datetime:
    try:
        if isinstance(value, datetime):
            parsed = value
        else:
            timestamp_text = str(value).strip()
            if timestamp_text.endswith("Z"):
                timestamp_text = timestamp_text[:-1] + "+00:00"
            if "T" not in timestamp_text and " " not in timestamp_text:
                raise ValueError
            parsed = datetime.fromisoformat(timestamp_text)
    except (TypeError, ValueError):
        raise EventDataError(f"Row {row_number}: timestamp is missing or invalid.")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_transfer_size(value: object, row_number: int) -> int | None:
    normalized = _normalized_optional(value)
    if normalized is None:
        return None
    if not normalized.isascii() or not normalized.isdigit():
        raise EventDataError(
            f"Row {row_number}: transfer size must be a non-negative integer or missing."
        )
    return int(normalized)


def canonicalize_events(
    rows: Iterable[Mapping[str, object]],
    case_id: int | str,
    evidence_id: str,
    original_evidence_reference: str | PathLike[str],
) -> list[CanonicalEvent]:
    """Convert validated or partial event rows into canonical, chronological events."""
    if isinstance(case_id, bool) or not str(case_id).strip():
        raise EventDataError("case_id is required.")
    normalized_evidence_id = str(evidence_id).strip()
    evidence_reference = fspath(original_evidence_reference)
    if not normalized_evidence_id:
        raise EventDataError("evidence_id is required.")
    if not isinstance(evidence_reference, str) or not evidence_reference.strip():
        raise EventDataError("original_evidence_reference is required.")

    events = []
    for row_index, row in enumerate(rows):
        row_number = row_index + 1
        if not isinstance(row, Mapping):
            raise EventDataError(f"Row {row_number}: event must be a mapping.")

        timestamp = _parse_timestamp(_row_value(row, "timestamp"), row_number)
        transfer_size = _parse_transfer_size(_row_value(row, "transfer_size"), row_number)
        events.append(
            (
                row_index,
                CanonicalEvent(
                    case_id=case_id,
                    evidence_id=normalized_evidence_id,
                    event_id=str(row_index),
                    timestamp=timestamp,
                    source_ip=_normalized_optional(_row_value(row, "source_ip")),
                    event_type=_normalized_optional(_row_value(row, "event_type")),
                    resource=_normalized_optional(_row_value(row, "resource")),
                    response_status=_normalized_optional(_row_value(row, "response_status")),
                    transfer_size=transfer_size,
                    original_evidence_reference=evidence_reference,
                ),
            )
        )

    events.sort(key=lambda item: (item[1].timestamp, item[0]))
    return [event for _, event in events]
