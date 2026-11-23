from datetime import datetime, timezone
import json

import pytest

from backend.ml.rule_baseline import (
    NOT_APPLICABLE_METRICS,
    RuleBaselineError,
    RuleBaselineInputEvent,
    canonical_event_to_detector_row,
    run_rule_baseline,
)
from backend.ml.schema import CanonicalEvent
from backend.rules import calculate_rule_fingerprint, get_rule_metadata


def _make_event(
    idx: int,
    *,
    source_ip: str = "192.0.2.10",
    event_type: str = "GET",
    resource: str = "/home",
    status: str = "200",
    transfer_size: int | None = 100,
    label: int = 0,
    categories: tuple[str, ...] = (),
) -> RuleBaselineInputEvent:
    ev = CanonicalEvent(
        case_id="case-synthetic",
        evidence_id="ev-synthetic",
        event_id=f"intranet/access.log:{idx}",
        timestamp=datetime(2022, 1, 15, 0, idx, 0, tzinfo=timezone.utc),
        source_ip=source_ip,
        event_type=event_type,
        resource=resource,
        response_status=status,
        transfer_size=transfer_size,
        original_evidence_reference="raw/fox/gather/intranet/access.log",
    )
    return RuleBaselineInputEvent(
        event=ev,
        scenario_id="fox",
        source_filename="intranet/access.log",
        source_line_number=idx,
        label=label,
        label_name="normal" if label == 0 else "attack_associated",
        ground_truth_coverage="verified",
        attack_categories=categories,
    )


def test_rule_baseline_maps_aligns_triggers_all_rules_deterministically(tmp_path, monkeypatch):
    # Isolate SQLite DB path before backend.main is lazily imported
    monkeypatch.setenv("FORENSIC_INSIGHT_DB_PATH", str(tmp_path / "isolated_rule_test.db"))
    monkeypatch.setenv("FORENSIC_INSIGHT_UPLOAD_DIR", str(tmp_path / "orig"))
    monkeypatch.setenv("FORENSIC_INSIGHT_WORKING_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("FORENSIC_INSIGHT_REPORTS_DIR", str(tmp_path / "reports"))

    # 8 synthetic events covering non-triggering, R-A (403), R-B (POST /admin),
    # R-C (3 POSTs from 192.0.2.50 -> fires only once on first POST), and R-D (large transfer)
    events = [
        _make_event(1, source_ip="192.0.2.1", event_type="GET", resource="/index", status="200", transfer_size=100, label=0),
        _make_event(2, source_ip="192.0.2.2", event_type="GET", resource="/docs", status="200", transfer_size=100, label=0),
        _make_event(3, source_ip="192.0.2.3", event_type="GET", resource="/about", status="200", transfer_size=100, label=0),
        _make_event(4, source_ip="192.0.2.4", event_type="GET", resource="/denied", status="403", transfer_size=100, label=1, categories=("dirb",)),
        _make_event(5, source_ip="192.0.2.50", event_type="POST", resource="/admin/login", status="200", transfer_size=100, label=1, categories=("foothold",)),
        _make_event(6, source_ip="192.0.2.50", event_type="POST", resource="/api/submit", status="200", transfer_size=100, label=1, categories=("foothold",)),
        _make_event(7, source_ip="192.0.2.50", event_type="POST", resource="/api/submit", status="200", transfer_size=100, label=1, categories=("foothold",)),
        _make_event(8, source_ip="192.0.2.99", event_type="GET", resource="/dump", status="200", transfer_size=500000, label=1, categories=("escalate",)),
    ]

    row0 = canonical_event_to_detector_row(events[0].event)
    assert row0 == {
        "timestamp": "2022-01-15T00:01:00+00:00",
        "source_ip": "192.0.2.1",
        "event_type": "GET",
        "resource": "/index",
        "status": "200",
        "bytes": "100",
    }

    fp_before = calculate_rule_fingerprint()
    meta_before = get_rule_metadata()

    res_1 = run_rule_baseline(events)
    res_2 = run_rule_baseline(events)

    assert calculate_rule_fingerprint() == fp_before
    assert get_rule_metadata() == meta_before
    assert res_1 == res_2
    assert json.dumps(res_1.to_dict(), sort_keys=True) == json.dumps(
        res_2.to_dict(), sort_keys=True
    )
    assert res_1.rule_registry_fingerprint == fp_before
    assert dict(res_1.not_applicable_metrics) == {m: None for m in NOT_APPLICABLE_METRICS}

    # Check exact per-event rule matches:
    # 0,1,2 -> no trigger (0)
    # 3 -> R-A (403)
    # 4 -> R-B (POST /admin) and R-C (first of 3 POSTs from 192.0.2.50)
    # 5,6 -> R-C already cleared post_requests[source_ip], so 0
    # 7 -> R-D (500000 > mean + 2*stdev)
    assert res_1.hard_predictions == (0, 0, 0, 1, 1, 0, 0, 1)
    assert res_1.predictions[3].matched_rule_ids == ("R-A",)
    assert res_1.predictions[4].matched_rule_ids == ("R-B", "R-C")
    assert res_1.predictions[5].matched_rule_ids == ()
    assert res_1.predictions[7].matched_rule_ids == ("R-D",)
    assert res_1.evaluation["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 2, "tp": 3}


def test_rule_baseline_rejects_ambiguous_findings_unknown_labels_and_duplicates():
    e1 = _make_event(1, label=0)
    e2 = _make_event(2, label=1, categories=("foothold",))

    # Reject unknown / non-binary label or unverified coverage at input construction
    with pytest.raises(RuleBaselineError, match="known labels 0 or 1"):
        RuleBaselineInputEvent(
            event=e1.event,
            scenario_id="fox",
            source_filename="intranet/access.log",
            source_line_number=1,
            label=None,  # type: ignore[arg-type]
            label_name="unknown",
        )
    with pytest.raises(RuleBaselineError, match="verified ground-truth"):
        RuleBaselineInputEvent(
            event=e1.event,
            scenario_id="fox",
            source_filename="intranet/access.log",
            source_line_number=1,
            label=0,
            label_name="normal",
            ground_truth_coverage="unavailable",
        )

    # Reject duplicate event identities
    with pytest.raises(RuleBaselineError, match="Duplicate event identity"):
        run_rule_baseline([e1, e1], detector=lambda _p, _id: [])

    # Reject finding with missing/non-numeric event_id
    with pytest.raises(RuleBaselineError, match="unambiguous numeric row event_id"):
        run_rule_baseline(
            [e1, e2],
            detector=lambda _p, _id: [{"rule_id": "R-A", "event_id": "ambiguous"}],
        )

    # Reject finding whose fields contradict the indexed event
    with pytest.raises(RuleBaselineError, match="do not unambiguously align"):
        run_rule_baseline(
            [e1, e2],
            detector=lambda _p, _id: [
                {
                    "rule_id": "R-A",
                    "event_id": "0",
                    "timestamp": e1.event.timestamp.isoformat(),
                    "source": "198.51.100.99",  # mismatched source IP
                    "event_type": "GET",
                    "resource": "/home",
                }
            ],
        )

    # Reject finding with unknown rule_id
    with pytest.raises(RuleBaselineError, match="unknown or missing rule_id"):
        run_rule_baseline(
            [e1, e2],
            detector=lambda _p, _id: [{"rule_id": "R-Z", "event_id": "0"}],
        )
