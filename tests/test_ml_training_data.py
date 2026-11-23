from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from backend.ml.features import engineer_features
from backend.ml.schema import FEATURE_SCHEMA, CanonicalEvent
from backend.ml.training_data import (
    EXPECTED_FEATURE_NAMES,
    TrainingDataError,
    load_and_verify_experiment_config,
    load_supervised_scenarios,
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: dict) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return path, _sha(raw)


def _build_synthetic_fixture(
    tmp_path: Path,
    *,
    corrupt_hash: str | None = None,
    override_record: dict | None = None,
    quality_status: str = "passed_with_research_limitations",
):
    scenario = "fox"
    scenario_dir = tmp_path / "processed" / scenario
    scenario_dir.mkdir(parents=True)

    covered_name = "intranet/access.log"
    unknown_name = "web/access.log"
    covered_source_sha = _sha(b"fox:covered")
    unknown_source_sha = _sha(b"fox:unknown")
    gt_sha = _sha(b"fox:labels:covered")
    gt_ref = f"raw/{scenario}/labels/{covered_name}"

    events = [
        CanonicalEvent(
            case_id="case-fox",
            evidence_id="ev-fox-1",
            event_id=f"{covered_name}:1",
            timestamp=datetime(2022, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            source_ip="192.0.2.10",
            event_type="GET",
            resource="/home",
            response_status="200",
            transfer_size=100,
            original_evidence_reference=f"raw/{scenario}/gather/{covered_name}",
        ),
        CanonicalEvent(
            case_id="case-fox",
            evidence_id="ev-fox-1",
            event_id=f"{covered_name}:2",
            timestamp=datetime(2022, 1, 15, 0, 1, 0, tzinfo=timezone.utc),
            source_ip="192.0.2.10",
            event_type="POST",
            resource="/admin",
            response_status="403",
            transfer_size=250,
            original_evidence_reference=f"raw/{scenario}/gather/{covered_name}",
        ),
        CanonicalEvent(
            case_id="case-fox",
            evidence_id="ev-fox-2",
            event_id=f"{unknown_name}:1",
            timestamp=datetime(2022, 1, 15, 0, 2, 0, tzinfo=timezone.utc),
            source_ip="192.0.2.20",
            event_type="GET",
            resource="/public",
            response_status="200",
            transfer_size=50,
            original_evidence_reference=f"raw/{scenario}/gather/{unknown_name}",
        ),
    ]
    covered_feats = engineer_features(events[:2])
    unknown_feats = engineer_features(events[2:])

    def _to_row(ev, feat_map, source_name, line_no, source_sha, label, label_name, basis, gt=None, gt_prov=None):
        return {
            "event": {
                "case_id": ev.case_id,
                "evidence_id": ev.evidence_id,
                "event_id": ev.event_id,
                "timestamp": ev.timestamp.isoformat(),
                "source_ip": ev.source_ip,
                "event_type": ev.event_type,
                "resource": ev.resource,
                "response_status": ev.response_status,
                "transfer_size": ev.transfer_size,
                "original_evidence_reference": ev.original_evidence_reference,
            },
            "provenance": {
                "scenario_id": scenario,
                "source_filename": source_name,
                "source_line_number": line_no,
                "source_sha256": source_sha,
            },
            "features": dict(feat_map),
            "ground_truth": gt,
            "ground_truth_provenance": gt_prov,
            "research_label": label,
            "research_label_name": label_name,
            "research_label_basis": basis,
            "research_label_provenance": {
                "policy_id": "ait-lds-v2.1-explicit-ground-truth-v1",
                "label_schema_version": "1.0.0",
                "ground_truth_reference": gt_ref if label in (0, 1) else None,
                "ground_truth_sha256": gt_sha if label in (0, 1) else None,
            },
        }

    rows = [
        _to_row(
            events[0],
            covered_feats[0].features,
            covered_name,
            1,
            covered_source_sha,
            0,
            "normal",
            "absence_from_verified_ground_truth_file",
        ),
        _to_row(
            events[1],
            covered_feats[1].features,
            covered_name,
            2,
            covered_source_sha,
            1,
            "attack_associated",
            "explicit_ground_truth",
            gt={"labels": ["foothold", "dirb"], "rules": {"foothold": ["r1"]}},
            gt_prov={
                "filename": "access.log",
                "reference": gt_ref,
                "sha256": gt_sha,
                "label_file_line_number": 1,
            },
        ),
        _to_row(
            events[2],
            unknown_feats[0].features,
            unknown_name,
            1,
            unknown_source_sha,
            None,
            "unknown",
            "insufficient_verified_label_coverage",
        ),
    ]
    if override_record is not None:
        rows[0].update(override_record)

    prepared_bytes = b"prepared-placeholder\n"
    (scenario_dir / "prepared_events.jsonl").write_bytes(prepared_bytes)
    prepared_sha = _sha(prepared_bytes)

    labelled_bytes = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")
    labelled_path = scenario_dir / "labelled_events.jsonl"
    labelled_path.write_bytes(labelled_bytes)
    labelled_sha = _sha(labelled_bytes)

    prepared_manifest = {
        "manifest_schema_version": "1.0.0",
        "feature_schema_version": "1.0.0",
        "source_files": [
            {
                "source_filename": covered_name,
                "source_sha256": covered_source_sha,
                "ground_truth_filename": "access.log",
                "ground_truth_reference": gt_ref,
                "ground_truth_sha256": gt_sha,
            },
            {
                "source_filename": unknown_name,
                "source_sha256": unknown_source_sha,
                "ground_truth_filename": None,
                "ground_truth_reference": None,
                "ground_truth_sha256": None,
            },
        ],
        "processed_output": {
            "filename": "prepared_events.jsonl",
            "sha256": prepared_sha,
            "record_count": len(rows),
        },
    }
    _, prepared_manifest_sha = _write_json(scenario_dir / "manifest.json", prepared_manifest)

    label_manifest = {
        "manifest_schema_version": "1.0.0",
        "label_schema_version": "1.0.0",
        "scenario_id": scenario,
        "label_policy": {"policy_id": "ait-lds-v2.1-explicit-ground-truth-v1"},
        "input": {
            "prepared_manifest_sha256": prepared_manifest_sha,
            "prepared_records_sha256": prepared_sha,
        },
        "labelled_output": {
            "filename": "labelled_events.jsonl",
            "sha256": labelled_sha,
        },
        "record_counts": {"evaluation_records": len(rows)},
        "evaluation_class_distribution": {
            "normal": 1,
            "attack_associated": 1,
            "unknown": 1,
        },
        "per_source_distributions": [
            {
                "source_filename": covered_name,
                "source_sha256": covered_source_sha,
                "ground_truth_sha256": gt_sha,
                "ground_truth_coverage": "verified",
            },
            {
                "source_filename": unknown_name,
                "source_sha256": unknown_source_sha,
                "ground_truth_sha256": None,
                "ground_truth_coverage": "unavailable",
            },
        ],
    }
    label_manifest_path, label_manifest_sha = _write_json(
        scenario_dir / "labels_manifest.json", label_manifest
    )

    split_path, split_sha = _write_json(
        tmp_path / "splits" / "four_scenario_split_manifest.json",
        {"scenario_ids": [scenario]},
    )
    quality_path, quality_sha = _write_json(
        tmp_path / "quality" / "four_scenario_quality_report.json",
        {"quality_status": quality_status},
    )

    if corrupt_hash == "labelled_file":
        labelled_path.write_bytes(labelled_bytes + b"{}\n")
    elif corrupt_hash == "split_manifest":
        split_sha = "0" * 64

    config = {
        "config_schema_version": "1.0.0",
        "features": {
            "schema_version": "1.0.0",
            "ordered_names": [f["name"] for f in FEATURE_SCHEMA["features"]],
        },
        "label_definition": {
            "positive_class": {"value": 1, "name": "attack_associated"},
            "negative_class": {"value": 0, "name": "normal"},
            "excluded_class": {
                "value": None,
                "name": "unknown",
                "exclude_from_supervised": True,
            },
        },
        "dataset_provenance": {
            "feature_schema_version": "1.0.0",
            "label_schema_version": "1.0.0",
            "split_manifest": {
                "path": str(split_path.relative_to(tmp_path)),
                "sha256": split_sha,
            },
            "quality_report": {
                "path": str(quality_path.relative_to(tmp_path)),
                "sha256": quality_sha,
                "expected_quality_status": quality_status,
            },
            "scenarios": {
                scenario: {
                    "label_manifest_path": str(label_manifest_path.relative_to(tmp_path)),
                    "label_manifest_sha256": label_manifest_sha,
                    "labelled_records_sha256": labelled_sha,
                    "prepared_manifest_sha256": prepared_manifest_sha,
                    "prepared_records_sha256": prepared_sha,
                }
            },
        },
    }
    config_path, _ = _write_json(tmp_path / "experiment_config.json", config)
    return config_path


def test_loader_extracts_ordered_features_excludes_unknowns_and_preserves_nulls(tmp_path):
    config_path = _build_synthetic_fixture(tmp_path)

    records, summaries = load_supervised_scenarios(
        ["fox"], config_path=config_path, base_dir=tmp_path
    )

    assert len(records) == 2
    assert [r.label for r in records] == [0, 1]
    assert [r.label_name for r in records] == ["normal", "attack_associated"]
    assert records[0].scenario_id == "fox"
    assert records[0].event_id == "intranet/access.log:1"
    assert records[1].attack_categories == ("foothold", "dirb")
    assert len(records[0].features) == len(EXPECTED_FEATURE_NAMES) == 13
    delta_idx = EXPECTED_FEATURE_NAMES.index("time_since_previous_event_seconds")
    delta_missing_idx = EXPECTED_FEATURE_NAMES.index("time_since_previous_event_missing")
    assert records[0].features[delta_idx] is None
    assert records[0].features[delta_missing_idx] == 1
    assert records[1].features[delta_idx] == 60.0
    assert records[1].features[delta_missing_idx] == 0

    summary = summaries["fox"]
    assert summary.total_records_read == 3
    assert summary.included_normal_count == 1
    assert summary.included_attack_count == 1
    assert summary.excluded_unknown_count == 1


def test_loader_rejects_hash_tampering_on_labelled_dataset_and_split_manifest(tmp_path):
    tampered_data_cfg = _build_synthetic_fixture(tmp_path / "a", corrupt_hash="labelled_file")
    with pytest.raises(TrainingDataError, match="labelled_events.jsonl SHA-256 mismatch"):
        load_supervised_scenarios(["fox"], config_path=tampered_data_cfg, base_dir=tmp_path / "a")

    tampered_split_cfg = _build_synthetic_fixture(tmp_path / "b", corrupt_hash="split_manifest")
    with pytest.raises(TrainingDataError, match="Split manifest SHA-256 mismatch"):
        load_and_verify_experiment_config(tampered_split_cfg, base_dir=tmp_path / "b")


def test_loader_rejects_invalid_label_and_feature_schema_leakage(tmp_path):
    bad_label_cfg = _build_synthetic_fixture(
        tmp_path / "bad_label", override_record={"research_label": True}
    )
    with pytest.raises(TrainingDataError, match="invalid research_label"):
        load_supervised_scenarios(["fox"], config_path=bad_label_cfg, base_dir=tmp_path / "bad_label")

    leaked_features = {name: 0 for name in EXPECTED_FEATURE_NAMES}
    leaked_features["scenario_id"] = "fox"
    bad_feat_cfg = _build_synthetic_fixture(
        tmp_path / "bad_feat", override_record={"features": leaked_features}
    )
    with pytest.raises(TrainingDataError, match="feature schema mismatch"):
        load_supervised_scenarios(["fox"], config_path=bad_feat_cfg, base_dir=tmp_path / "bad_feat")


def test_loader_rejects_provenance_mismatch_and_failed_quality_report(tmp_path):
    bad_prov_cfg = _build_synthetic_fixture(
        tmp_path / "bad_prov",
        override_record={
            "provenance": {
                "scenario_id": "fox",
                "source_filename": "intranet/access.log",
                "source_line_number": 1,
                "source_sha256": "f" * 64,
            }
        },
    )
    with pytest.raises(TrainingDataError, match="unverified source provenance"):
        load_supervised_scenarios(["fox"], config_path=bad_prov_cfg, base_dir=tmp_path / "bad_prov")

    failed_q_cfg = _build_synthetic_fixture(
        tmp_path / "failed_q", quality_status="failed"
    )
    with pytest.raises(TrainingDataError, match="quality report status is failed"):
        load_and_verify_experiment_config(failed_q_cfg, base_dir=tmp_path / "failed_q")
