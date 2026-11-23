import pytest

from backend.ml import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_VERSION,
    EventDataError,
    extract_features,
)


CASE_ID = 42
EVIDENCE_ID = "EV-42-FEATURES"
ORIGINAL_REFERENCE = "uploads/original/EV-42-FEATURES.csv"


def event(timestamp, source="10.0.0.1", event_type="GET", resource="/home", status="200", size="100"):
    return {
        "timestamp": timestamp,
        "source_ip": source,
        "event_type": event_type,
        "resource": resource,
        "status": status,
        "bytes": size,
    }


def test_canonical_features_are_correct_and_deterministic():
    rows = [
        event("2026-01-01T00:02:00Z", event_type="POST", resource="/admin", status="500", size="200"),
        event("2026-01-01T00:00:00Z", resource="/home", size="100"),
        event("2026-01-01T00:07:01Z", resource="/home", size="300"),
    ]

    records = extract_features(rows, CASE_ID, EVIDENCE_ID, ORIGINAL_REFERENCE)
    repeated = extract_features(rows, CASE_ID, EVIDENCE_ID, ORIGINAL_REFERENCE)
    by_id = {record.event.event_id: record for record in records}

    assert records == repeated
    assert [record.event.event_id for record in records] == ["1", "0", "2"]
    assert by_id["1"].features["request_frequency_5m"] == 1
    assert by_id["1"].features["time_since_previous_event_seconds"] is None
    assert by_id["0"].features["request_frequency_5m"] == 2
    assert by_id["0"].features["failed_request_frequency_5m"] == 1
    assert by_id["0"].features["post_request_frequency_5m"] == 1
    assert by_id["0"].features["time_since_previous_event_seconds"] == 120
    assert by_id["0"].features["unique_resources_accessed"] == 2
    assert by_id["0"].features["sensitive_resource_indicator"] == 1
    assert by_id["2"].features["request_frequency_5m"] == 1
    assert by_id["2"].features["failed_request_frequency_5m"] == 0
    assert by_id["2"].features["post_request_frequency_5m"] == 0
    assert by_id["2"].features["time_since_previous_event_seconds"] == 301
    assert by_id["2"].features["unique_resources_accessed"] == 2
    assert by_id["2"].features["transfer_size_bytes"] == 300


def test_future_events_do_not_change_prior_event_features():
    prior_rows = [
        event("2026-01-01T00:00:00Z", resource="/first"),
        event("2026-01-01T00:01:00Z", event_type="POST", resource="/second", status="403"),
    ]
    with_future = prior_rows + [
        event("2026-01-01T00:02:00Z", event_type="POST", resource="/secret", status="500")
    ]

    before = extract_features(prior_rows, CASE_ID, EVIDENCE_ID, ORIGINAL_REFERENCE)
    after = extract_features(with_future, CASE_ID, EVIDENCE_ID, ORIGINAL_REFERENCE)
    future_extended = {record.event.event_id: record for record in after}

    for record in before:
        assert future_extended[record.event.event_id].features == record.features


def test_rolling_counts_preserve_inclusive_window_expiration_boundaries():
    rows = [
        event("2026-01-01T00:00:00Z", event_type="POST", status="500"),
        event("2026-01-01T00:05:00Z"),
        event("2026-01-01T00:05:00.000001Z"),
    ]

    records = extract_features(rows, CASE_ID, EVIDENCE_ID, ORIGINAL_REFERENCE)

    assert records[1].features["request_frequency_5m"] == 2
    assert records[1].features["failed_request_frequency_5m"] == 1
    assert records[1].features["post_request_frequency_5m"] == 1
    assert records[2].features["request_frequency_5m"] == 2
    assert records[2].features["failed_request_frequency_5m"] == 0
    assert records[2].features["post_request_frequency_5m"] == 0


def test_traceability_is_separate_from_numerical_model_inputs():
    record = extract_features(
        [event("2026-01-01T00:00:00Z")],
        CASE_ID,
        EVIDENCE_ID,
        ORIGINAL_REFERENCE,
    )[0]

    assert record.event.case_id == CASE_ID
    assert record.event.evidence_id == EVIDENCE_ID
    assert record.event.event_id == "0"
    assert record.event.original_evidence_reference == ORIGINAL_REFERENCE
    assert record.event.timestamp.isoformat() == "2026-01-01T00:00:00+00:00"
    assert set(record.model_inputs) == {feature["name"] for feature in FEATURE_SCHEMA["features"]}
    assert "case_id" not in record.model_inputs
    assert "evidence_id" not in record.model_inputs
    assert "event_id" not in record.model_inputs
    assert "original_evidence_reference" not in record.model_inputs


def test_missing_values_are_explicit_and_invalid_inputs_raise():
    record = extract_features(
        [{
            "timestamp": "2026-01-01 00:00:00",
            "source": "",
            "event": "",
            "description": "",
            "status": "unknown",
            "bytes": "",
        }],
        CASE_ID,
        EVIDENCE_ID,
        ORIGINAL_REFERENCE,
    )[0]

    assert record.event.source_ip is None
    assert record.event.event_type is None
    assert record.event.resource is None
    assert record.event.response_status == "unknown"
    assert record.features["transfer_size_bytes"] is None
    assert record.features["request_frequency_5m"] is None
    assert record.features["time_since_previous_event_seconds"] is None
    assert record.features["source_ip_missing"] == 1
    assert record.features["event_type_missing"] == 1
    assert record.features["resource_missing"] == 1
    assert record.features["response_status_missing"] == 1
    assert record.features["transfer_size_missing"] == 1
    assert record.features["time_since_previous_event_missing"] == 1

    with pytest.raises(EventDataError, match="timestamp"):
        extract_features(
            [event("not-a-timestamp")], CASE_ID, EVIDENCE_ID, ORIGINAL_REFERENCE
        )
    with pytest.raises(EventDataError, match="transfer size"):
        extract_features(
            [event("2026-01-01T00:00:00Z", size="-1")],
            CASE_ID,
            EVIDENCE_ID,
            ORIGINAL_REFERENCE,
        )


def test_feature_schema_is_versioned_and_documents_feature_meanings():
    assert FEATURE_SCHEMA_VERSION == "1.0.0"
    assert FEATURE_SCHEMA["version"] == FEATURE_SCHEMA_VERSION
    assert FEATURE_SCHEMA["frequency_window_seconds"] == 300
    assert all(feature["meaning"] and feature["calculation"] for feature in FEATURE_SCHEMA["features"])
