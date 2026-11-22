from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta
from os import PathLike
from typing import Iterable, Mapping

from .schema import (
    FEATURE_WINDOW_SECONDS,
    CanonicalEvent,
    EventDataError,
    FeatureRecord,
    canonicalize_events,
)


SENSITIVE_RESOURCE_TERMS = ("secret", "admin", "password", "confidential", ".zip")


def _status_code(value: str | None) -> int | None:
    if value is None or not value.isascii() or not value.isdigit():
        return None
    return int(value)


def engineer_features(events: Iterable[CanonicalEvent]) -> list[FeatureRecord]:
    """Compute deterministic as-of-event features from canonical events."""
    indexed_events = list(enumerate(events))
    for _, event in indexed_events:
        if not isinstance(event, CanonicalEvent):
            raise EventDataError("engineer_features accepts only CanonicalEvent values.")
    indexed_events.sort(key=lambda item: (item[1].timestamp, item[0]))

    histories = defaultdict(deque)
    previous_timestamps = {}
    seen_resources = defaultdict(set)
    records = []
    window = timedelta(seconds=FEATURE_WINDOW_SECONDS)

    for _, event in indexed_events:
        source_ip = event.source_ip
        event_type = event.event_type or ""
        resource = event.resource
        status = _status_code(event.response_status)
        is_post = int("POST" in event_type.upper())
        is_failed = int(status is not None and 400 <= status <= 599)
        resource_key = resource.casefold() if resource else None
        resource_history = seen_resources[source_ip] if source_ip else None
        if resource_key and resource_history is not None:
            resource_history.add(resource_key)

        if source_ip:
            history = histories[source_ip]
            cutoff = event.timestamp - window
            while history and history[0][0] < cutoff:
                history.popleft()
            history.append((event.timestamp, is_failed, is_post))
            request_frequency = len(history)
            failed_request_frequency = sum(item[1] for item in history)
            post_request_frequency = sum(item[2] for item in history)
            previous_timestamp = previous_timestamps.get(source_ip)
            time_since_previous = (
                (event.timestamp - previous_timestamp).total_seconds()
                if previous_timestamp is not None
                else None
            )
            previous_timestamps[source_ip] = event.timestamp
            unique_resource_count = len(resource_history)
        else:
            request_frequency = None
            failed_request_frequency = None
            post_request_frequency = None
            time_since_previous = None
            unique_resource_count = None

        features = {
            "transfer_size_bytes": event.transfer_size,
            "request_frequency_5m": request_frequency,
            "failed_request_frequency_5m": failed_request_frequency,
            "post_request_frequency_5m": post_request_frequency,
            "time_since_previous_event_seconds": time_since_previous,
            "unique_resources_accessed": unique_resource_count,
            "sensitive_resource_indicator": int(
                bool(resource)
                and any(term in resource.casefold() for term in SENSITIVE_RESOURCE_TERMS)
            ),
            "source_ip_missing": int(source_ip is None),
            "event_type_missing": int(event.event_type is None),
            "resource_missing": int(resource is None),
            "response_status_missing": int(status is None),
            "transfer_size_missing": int(event.transfer_size is None),
            "time_since_previous_event_missing": int(time_since_previous is None),
        }
        records.append(FeatureRecord(event=event, features=features))

    return records


def extract_features(
    rows: Iterable[Mapping[str, object]],
    case_id: int | str,
    evidence_id: str,
    original_evidence_reference: str | PathLike[str],
) -> list[FeatureRecord]:
    """Canonicalize event rows and produce traceable feature records."""
    events = canonicalize_events(
        rows,
        case_id=case_id,
        evidence_id=evidence_id,
        original_evidence_reference=original_evidence_reference,
    )
    return engineer_features(events)
