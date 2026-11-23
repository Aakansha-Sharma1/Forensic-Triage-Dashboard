import copy
import hashlib
import json
from datetime import datetime, timezone

from backend.ml.features import engineer_features
from backend.ml.quality import create_quality_report
from backend.ml.schema import CanonicalEvent


SCENARIO_STARTS = {
    "russellmitchell": "2022-01-21",
    "santos": "2022-01-14",
    "harrison": "2022-02-04",
}
SCENARIOS = tuple(SCENARIO_STARTS)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _make_source(scenario, filename, line_count, *, covered):
    entry = {
        "source_filename": filename,
        "source_reference": f"raw/{scenario}/gather/{filename}",
        "scenario_id": scenario,
        "source_sha256": _sha(f"{scenario}:{filename}".encode()),
        "physical_line_count": line_count,
        "parsed_record_count": line_count,
        "rejected_log_row_count": 0,
        "rejected_row_count": 0,
        "ground_truth_filename": None,
        "ground_truth_reference": None,
        "ground_truth_sha256": None,
        "ground_truth_physical_line_count": 0,
        "ground_truth_record_count": 0,
        "attached_ground_truth_count": 0,
    }
    if covered:
        entry.update(
            {
                "ground_truth_filename": filename.rsplit("/", 1)[-1],
                "ground_truth_reference": f"raw/{scenario}/labels/{filename}",
                "ground_truth_sha256": _sha(
                    f"{scenario}:labels:{filename}".encode()
                ),
                "ground_truth_physical_line_count": 1,
                "ground_truth_record_count": 1,
                "attached_ground_truth_count": 1,
            }
        )
    return entry


def _prepared_record(scenario, source, source_entry, line_number, timestamp, method, status, resource, size):
    event = CanonicalEvent(
        case_id=f"case-{scenario}",
        evidence_id=f"evidence-{scenario}",
        event_id=f"{source}:{line_number}",
        timestamp=datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(
            timezone.utc
        ),
        source_ip="192.0.2.10",
        event_type=method,
        resource=resource,
        response_status=status,
        transfer_size=size,
        original_evidence_reference=source_entry["source_reference"],
    )
    features = dict(engineer_features([event])[0].features)
    record = {
        "event": {
            "case_id": event.case_id,
            "evidence_id": event.evidence_id,
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "source_ip": event.source_ip,
            "event_type": event.event_type,
            "resource": event.resource,
            "response_status": event.response_status,
            "transfer_size": event.transfer_size,
            "original_evidence_reference": event.original_evidence_reference,
        },
        "provenance": {
            "source_filename": source,
            "source_line_number": line_number,
            "scenario_id": scenario,
            "source_sha256": source_entry["source_sha256"],
            "virtual_host": None,
        },
        "features": features,
        "ground_truth": None,
        "ground_truth_provenance": None,
    }
    return record


def _write_scenario(tmp_path, scenario, *, corrupt_feature=False):
    scenario_dir = tmp_path / scenario
    scenario_dir.mkdir()
    covered_name = "intranet/access.log"
    unknown_name = "web/access.log"
    covered = _make_source(scenario, covered_name, 2, covered=True)
    unknown = _make_source(scenario, unknown_name, 1, covered=False)
    start = SCENARIO_STARTS[scenario]
    prepared = [
        _prepared_record(
            scenario, covered_name, covered, 1, f"{start}T00:00:00Z", "GET", "200", "/home", 10
        ),
        _prepared_record(
            scenario, covered_name, covered, 2, f"{start}T00:01:00Z", "POST", "500", "/admin", 20
        ),
        _prepared_record(
            scenario, unknown_name, unknown, 1, f"{start}T00:02:00Z", "GET", "200", "/home", 5
        ),
    ]
    if corrupt_feature:
        prepared[0]["features"]["research_label"] = 0
        prepared[0]["features"]["transfer_size_bytes"] = float("inf")

    attack_gt = {
        "labels": ["foothold"],
        "rules": {"foothold": ["synthetic.rule"]},
    }
    attack_gt_provenance = {
        "filename": covered["ground_truth_filename"],
        "reference": covered["ground_truth_reference"],
        "sha256": covered["ground_truth_sha256"],
        "label_file_line_number": 1,
    }
    prepared[1]["ground_truth"] = attack_gt
    prepared[1]["ground_truth_provenance"] = attack_gt_provenance
    labelled = copy.deepcopy(prepared)
    labels = [
        (0, "normal", "absence_from_verified_ground_truth_file"),
        (1, "attack_associated", "explicit_ground_truth"),
        (None, "unknown", "insufficient_verified_label_coverage"),
    ]
    per_source = [
        {
            "source_filename": covered_name,
            "source_sha256": covered["source_sha256"],
            "ground_truth_sha256": covered["ground_truth_sha256"],
            "ground_truth_coverage": "verified",
            "evaluation_records": 2,
            "classes": {"normal": 1, "attack_associated": 1, "unknown": 0},
        },
        {
            "source_filename": unknown_name,
            "source_sha256": unknown["source_sha256"],
            "ground_truth_sha256": None,
            "ground_truth_coverage": "unavailable",
            "evaluation_records": 1,
            "classes": {"normal": 0, "attack_associated": 0, "unknown": 1},
        },
    ]
    for row, (label, name, basis) in zip(labelled, labels):
        row["research_label"] = label
        row["research_label_name"] = name
        row["research_label_basis"] = basis
        row["research_label_provenance"] = {
            "policy_id": "ait-lds-v2.1-explicit-ground-truth-v1",
            "label_schema_version": "1.0.0",
            "ground_truth_reference": (
                covered["ground_truth_reference"] if label in (0, 1) else None
            ),
            "ground_truth_sha256": (
                covered["ground_truth_sha256"] if label in (0, 1) else None
            ),
        }
    prepared_bytes = _jsonl(prepared)
    labelled_bytes = _jsonl(labelled)
    (scenario_dir / "prepared_events.jsonl").write_bytes(prepared_bytes)
    (scenario_dir / "labelled_events.jsonl").write_bytes(labelled_bytes)
    prepared_manifest = {
        "manifest_schema_version": "1.0.0",
        "feature_schema_version": "1.0.0",
        "source_files": [covered, unknown],
        "record_counts": {
            "parsed_records": len(prepared),
            "attached_ground_truth_records": 1,
            "rejected_rows": 0,
        },
        "processed_output": {
            "filename": "prepared_events.jsonl",
            "sha256": _sha(prepared_bytes),
            "record_count": len(prepared),
        },
    }
    prepared_manifest_bytes = (
        json.dumps(prepared_manifest, sort_keys=True, indent=2) + "\n"
    ).encode()
    (scenario_dir / "manifest.json").write_bytes(prepared_manifest_bytes)
    label_manifest = {
        "manifest_schema_version": "1.0.0",
        "label_schema_version": "1.0.0",
        "scenario_id": scenario,
        "label_policy": {
            "policy_id": "ait-lds-v2.1-explicit-ground-truth-v1",
        },
        "simulation_period": {
            "start_inclusive": f"{start}T00:00:00+00:00",
            "end_exclusive": (
                "2022-01-25T00:00:00+00:00"
                if scenario == "russellmitchell"
                else "2022-01-18T00:00:00+00:00"
                if scenario == "santos"
                else "2022-02-09T00:00:00+00:00"
            ),
        },
        "input": {
            "prepared_records_sha256": _sha(prepared_bytes),
            "prepared_manifest_sha256": _sha(prepared_manifest_bytes),
        },
        "labelled_output": {
            "filename": "labelled_events.jsonl",
            "sha256": _sha(labelled_bytes),
        },
        "record_counts": {
            "input_records": len(prepared),
            "evaluation_records": len(labelled),
            "excluded_outside_simulation_period": 0,
        },
        "evaluation_class_distribution": {
            "normal": 1,
            "attack_associated": 1,
            "unknown": 1,
        },
        "per_category_counts": {"foothold": 1},
        "per_source_distributions": per_source,
    }
    label_manifest_path = scenario_dir / "labels_manifest.json"
    label_manifest_path.write_text(
        json.dumps(label_manifest, sort_keys=True), encoding="utf-8"
    )
    return label_manifest_path


def _jsonl(records):
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode()


def _write_all(tmp_path, *, corrupt_feature=False):
    return {
        scenario: _write_scenario(
            tmp_path, scenario, corrupt_feature=corrupt_feature
        )
        for scenario in SCENARIOS
    }


def test_quality_report_verifies_data_and_is_reproducible(tmp_path):
    manifests = _write_all(tmp_path)
    output = tmp_path / "quality_report.json"

    report = create_quality_report(
        manifests, output, similarity_manifest_path=tmp_path / "no-prior-analysis.json"
    )
    first_bytes = output.read_bytes()
    repeated = create_quality_report(
        manifests, output, similarity_manifest_path=tmp_path / "no-prior-analysis.json"
    )

    assert report["quality_status"] == "passed_with_research_limitations"
    assert report["overall_class_distribution"] == {
        "normal": 3,
        "attack_associated": 3,
        "unknown": 3,
    }
    assert report["temporal_feature_leakage_check"]["passed"] is True
    assert repeated == report
    assert output.read_bytes() == first_bytes
    for scenario in SCENARIOS:
        result = report["scenario_results"][scenario]
        assert result["integrity"]["passed"] is True
        assert result["feature_quality"]["violations"] == {}
        assert result["feature_quality"]["target_leakage_feature_keys"] == []
        assert result["duplicates"]["duplicate_source_event_identities"] == 0


def test_quality_report_detects_hash_tampering(tmp_path):
    manifests = _write_all(tmp_path)
    dataset = manifests["santos"].parent / "labelled_events.jsonl"
    with dataset.open("ab") as output:
        output.write(b"tampered\n")

    report = create_quality_report(
        manifests,
        tmp_path / "quality_report.json",
        similarity_manifest_path=tmp_path / "no-prior-analysis.json",
    )

    assert report["quality_status"] == "failed"
    assert (
        "labelled_dataset_sha256_mismatch"
        in report["scenario_results"]["santos"]["integrity"]["failures"]
    )


def test_quality_report_flags_nonfinite_values_and_direct_target_leakage(tmp_path):
    manifests = _write_all(tmp_path, corrupt_feature=True)

    report = create_quality_report(
        manifests,
        tmp_path / "quality_report.json",
        similarity_manifest_path=tmp_path / "no-prior-analysis.json",
    )
    quality = report["scenario_results"]["russellmitchell"]["feature_quality"]

    assert report["quality_status"] == "failed"
    assert quality["target_leakage_feature_keys"] == ["research_label"]
    assert quality["violations"]["non_finite:transfer_size_bytes"] == 1
