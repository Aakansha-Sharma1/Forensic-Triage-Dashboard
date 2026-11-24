from __future__ import annotations

import hashlib
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.ml.advisory_service import ml_advisory_service
from backend.ml.baselines import train_random_forest_baseline
from backend.ml.features import extract_features
from backend.ml.model_artifact import load_model_artifact, save_model_artifact
from backend.ml.training_data import DEFAULT_EXPERIMENT_CONFIG_PATH, EXPECTED_FEATURE_NAMES


@pytest.fixture
def client(isolated_app):
    return isolated_app[0]


@pytest.fixture
def paths(isolated_app):
    return isolated_app[1]


@pytest.fixture(autouse=True)
def _reset_ml_service_cache():
    ml_advisory_service.clear_cache()
    yield
    ml_advisory_service.clear_cache()


def _create_synthetic_model_artifact(tmp_path: Path) -> tuple[Path, str, str, str]:
    rng = np.random.default_rng(77)
    n_normal = 40
    n_attack = 40
    normal = np.column_stack(
        [
            rng.uniform(200.0, 900.0, size=n_normal),
            rng.uniform(1.0, 4.0, size=n_normal),
            np.zeros(n_normal),
            np.zeros(n_normal),
            rng.uniform(10.0, 60.0, size=n_normal),
            rng.uniform(1.0, 3.0, size=n_normal),
            np.zeros(n_normal),
            np.zeros((n_normal, 6)),
        ]
    )
    attack = np.column_stack(
        [
            rng.uniform(100.0, 400.0, size=n_attack),
            rng.uniform(8.0, 50.0, size=n_attack),
            rng.uniform(4.0, 40.0, size=n_attack),
            rng.uniform(2.0, 20.0, size=n_attack),
            rng.uniform(0.0, 1.0, size=n_attack),
            rng.uniform(4.0, 20.0, size=n_attack),
            np.ones(n_attack),
            np.zeros((n_attack, 6)),
        ]
    )
    x_train = np.vstack([normal, attack]).astype(np.float64)
    y_train = np.array([0] * n_normal + [1] * n_attack, dtype=np.int64)

    fitted = train_random_forest_baseline(
        x_train,
        y_train,
        weighting_regime="class_weighted",
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
        config_path=DEFAULT_EXPERIMENT_CONFIG_PATH,
    )
    fitted.pipeline.named_steps["classifier"].n_jobs = 1

    trusted_root = tmp_path / "models"
    model_id = "synthetic-rf-test"
    model_version = "1.0.0"
    artifact_dir = trusted_root / model_id / model_version
    config_sha = hashlib.sha256(DEFAULT_EXPERIMENT_CONFIG_PATH.read_bytes()).hexdigest()
    _, manifest = save_model_artifact(
        fitted,
        artifact_dir,
        trusted_root=trusted_root,
        model_id=model_id,
        model_version=model_version,
        decision_threshold=0.5,
        training_dataset_provenance={
            "dataset_name": "Synthetic Test Dataset",
            "dataset_version": "test_v1",
            "scenarios": ["s1"],
            "scenario_labelled_sha256": {"s1": "a" * 64},
            "unknown_records_excluded": True,
            "benchmark_outer_test_predictions_excluded": True,
        },
        training_config_sha256=config_sha,
        intended_use="Synthetic integration test artifact.",
        limitations=["Synthetic test artifact; not externally validated."],
        created_at_utc="2026-11-24T00:00:00+00:00",
    )
    return trusted_root, model_id, model_version, str(manifest["model_file"]["sha256"])


def _create_case(client, suffix: str) -> dict:
    resp = client.post(
        "/cases",
        json={
            "case_number": f"CASE-ML-{suffix}",
            "title": "ML Advisory Integration Case",
            "investigator": "Forensic Tester",
            "description": "Synthetic test case",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["case"]


def _upload_evidence(client, case_id: int, filename: str, content: str, content_type: str = "text/csv") -> dict:
    resp = client.post(
        f"/cases/{case_id}/evidence",
        params={"source_description": "Synthetic test evidence"},
        files={"file": (filename, content.encode("utf-8"), content_type)},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["evidence"]


def _sample_apache_csv() -> str:
    return (
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01T10:00:00Z,10.0.0.1,GET,/index.html,200,512\n"
        "2026-01-01T10:00:30Z,10.0.0.1,GET,/about.html,200,480\n"
        "2026-01-01T10:01:00Z,192.168.1.99,POST,/admin/secret,403,200\n"
        "2026-01-01T10:01:01Z,192.168.1.99,POST,/admin/password,403,210\n"
        "2026-01-01T10:01:02Z,192.168.1.99,POST,/admin/confidential,403,220\n"
    )


def _configure_ml_env(
    monkeypatch: pytest.MonkeyPatch,
    trusted_root: Path,
    model_id: str,
    model_version: str,
    model_sha256: str,
    *,
    enabled: bool = True,
    approved: bool = True,
    max_events: int | None = None,
) -> None:
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_OPERATOR_APPROVED", "true" if approved else "false")
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_MODELS_DIR", str(trusted_root))
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_MODEL_ID", model_id)
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_MODEL_VERSION", model_version)
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_MODEL_SHA256", model_sha256)
    if max_events is not None:
        monkeypatch.setenv("FORENSIC_INSIGHT_ML_MAX_EVENTS", str(max_events))
    else:
        monkeypatch.delenv("FORENSIC_INSIGHT_ML_MAX_EVENTS", raising=False)


def _snapshot_db_tables(db_path: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(db_path)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        assert "cases" in tables and "evidence" in tables and "findings" in tables
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid ASC").fetchall()
            for table in tables
        }
    finally:
        conn.close()


def test_startup_does_not_deserialize_model_automatically() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import backend.main; "
                "from backend.ml.advisory_service import ml_advisory_service; "
                "assert ml_advisory_service._cached_artifact is None"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_ml_authorization_and_rule_analysis_independence(client, monkeypatch, tmp_path: Path) -> None:
    trusted_root, model_id, model_version, model_sha256 = _create_synthetic_model_artifact(tmp_path)
    case = _create_case(client, "auth-gates")
    _upload_evidence(client, case["id"], "access.csv", _sample_apache_csv())

    # 1. Disabled by default -> 503, while deterministic /analyze still succeeds normally.
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_ENABLED", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_OPERATOR_APPROVED", raising=False)
    resp_disabled = client.post(f"/cases/{case['id']}/ml-advisory")
    assert resp_disabled.status_code == 503
    assert resp_disabled.json()["detail"]["ml_advisory_error"] == "ml_not_enabled"
    assert resp_disabled.json()["detail"]["deterministic_analysis_available"] is True

    analyze_resp = client.post(f"/cases/{case['id']}/analyze")
    assert analyze_resp.status_code == 200
    assert analyze_resp.json()["findings_count"] > 0

    # 2. Enabled without operator approval -> 503.
    _configure_ml_env(
        monkeypatch, trusted_root, model_id, model_version, model_sha256,
        enabled=True, approved=False,
    )
    resp_unapproved = client.post(f"/cases/{case['id']}/ml-advisory")
    assert resp_unapproved.status_code == 503
    assert resp_unapproved.json()["detail"]["ml_advisory_error"] == "ml_not_approved"

    # 3. Wrong pinned SHA-256 -> 503.
    _configure_ml_env(
        monkeypatch, trusted_root, model_id, model_version, "0" * 64,
        enabled=True, approved=True,
    )
    resp_wrong_hash = client.post(f"/cases/{case['id']}/ml-advisory")
    assert resp_wrong_hash.status_code == 503
    assert resp_wrong_hash.json()["detail"]["ml_advisory_error"] == "model_unavailable_or_invalid"

    # 4. Missing model directory -> 503.
    _configure_ml_env(
        monkeypatch, tmp_path / "nonexistent_models", model_id, model_version, model_sha256,
        enabled=True, approved=True,
    )
    resp_missing = client.post(f"/cases/{case['id']}/ml-advisory")
    assert resp_missing.status_code == 503
    assert resp_missing.json()["detail"]["ml_advisory_error"] == "model_unavailable_or_invalid"


def test_valid_advisory_inference_is_deterministic_read_only_and_schema_exact(
    client, paths, monkeypatch, tmp_path: Path
) -> None:
    trusted_root, model_id, model_version, model_sha256 = _create_synthetic_model_artifact(tmp_path)
    _configure_ml_env(monkeypatch, trusted_root, model_id, model_version, model_sha256)

    case = _create_case(client, "valid-advisory")
    csv_text = _sample_apache_csv()
    evidence = _upload_evidence(client, case["id"], "access.csv", csv_text)

    # Establish deterministic rule findings and chain-of-custody before ML advisory runs.
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200
    assert client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={"person": "Analyst", "action": "Verified", "remarks": "Baseline custody"},
    ).status_code == 200

    original_file = paths["upload"] / f"{evidence['evidence_id']}.csv"
    working_file = paths["working"] / f"{evidence['evidence_id']}_working.csv"
    orig_sha_before = hashlib.sha256(original_file.read_bytes()).hexdigest()
    work_sha_before = hashlib.sha256(working_file.read_bytes()).hexdigest()
    db_before = _snapshot_db_tables(paths["db"])
    findings_before = client.get(f"/cases/{case['id']}/findings").json()

    # Run advisory inference twice and verify bit-for-bit determinism.
    r1 = client.post(f"/cases/{case['id']}/ml-advisory")
    r2 = client.post(f"/cases/{case['id']}/ml-advisory")
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    body1 = r1.json()
    body2 = r2.json()
    assert body1 == body2

    assert body1["advisory_only"] is True
    assert body1["deterministic_rules_unchanged"] is True
    assert body1["model"]["model_id"] == model_id
    assert body1["model"]["model_version"] == model_version
    assert body1["model"]["model_sha256"] == model_sha256
    assert body1["model"]["feature_schema_version"] == "1.0.0"
    assert body1["model"]["score_semantics"]["calibrated_probability"] is False
    assert body1["summary"]["total_events_scored"] == 5
    assert body1["summary"]["returned_predictions_count"] == 5
    assert len(body1["predictions"]) == 5
    assert len(body1["limitations"]) >= 2

    # Verify exact feature-extraction & prediction parity with backend.ml.features + model_artifact.
    import csv as _csv
    rows = list(_csv.DictReader(csv_text.splitlines()))
    expected_records = extract_features(
        rows,
        case_id=case["id"],
        evidence_id=evidence["evidence_id"],
        original_evidence_reference=str(original_file),
    )
    matrix = np.asarray(
        [
            [math.nan if rec.features[col] is None else float(rec.features[col]) for col in EXPECTED_FEATURE_NAMES]
            for rec in expected_records
        ],
        dtype=np.float64,
    )
    loaded = load_model_artifact(
        trusted_root / model_id / model_version,
        trusted_root=trusted_root,
        operator_approved=True,
        expected_model_id=model_id,
        expected_model_version=model_version,
        expected_model_sha256=model_sha256,
    )
    direct_preds = loaded.predict_batch(matrix)
    for api_pred, direct_pred in zip(body1["predictions"], direct_preds):
        assert api_pred["predicted_label"] == direct_pred.predicted_label
        assert api_pred["model_score"] == direct_pred.model_score
        assert api_pred["score_type"] == "uncalibrated_model_score"
        assert api_pred["is_calibrated_probability"] is False
        assert api_pred["source_event_reference"]["evidence_id"] == evidence["evidence_id"]

    # Verify only_attack_associated filter preserves summary totals.
    r_filtered = client.post(
        f"/cases/{case['id']}/ml-advisory", params={"only_attack_associated": "true"}
    )
    assert r_filtered.status_code == 200
    filtered = r_filtered.json()
    assert filtered["summary"]["total_events_scored"] == 5
    assert all(p["predicted_label"] == 1 for p in filtered["predictions"])

    # Verify zero mutation of original file, working file, findings, custody, and audit_log.
    assert hashlib.sha256(original_file.read_bytes()).hexdigest() == orig_sha_before
    assert hashlib.sha256(working_file.read_bytes()).hexdigest() == work_sha_before
    assert _snapshot_db_tables(paths["db"]) == db_before
    assert client.get(f"/cases/{case['id']}/findings").json() == findings_before


def test_tampered_model_unsupported_input_and_event_cap_fail_safely(
    client, paths, monkeypatch, tmp_path: Path
) -> None:
    trusted_root, model_id, model_version, model_sha256 = _create_synthetic_model_artifact(tmp_path)
    _configure_ml_env(monkeypatch, trusted_root, model_id, model_version, model_sha256)

    # 1. Unsupported non-CSV format -> 422.
    case_txt = _create_case(client, "unsupported-txt")
    _upload_evidence(client, case_txt["id"], "notes.txt", "raw log line\n", content_type="text/plain")
    resp_txt = client.post(f"/cases/{case_txt['id']}/ml-advisory")
    assert resp_txt.status_code == 422
    assert resp_txt.json()["detail"]["validation_error"] == "unsupported_format"

    # 2. Event limit exceeded -> 413.
    case_cap = _create_case(client, "event-cap")
    _upload_evidence(client, case_cap["id"], "access.csv", _sample_apache_csv())
    _configure_ml_env(
        monkeypatch, trusted_root, model_id, model_version, model_sha256, max_events=2
    )
    resp_cap = client.post(f"/cases/{case_cap['id']}/ml-advisory")
    assert resp_cap.status_code == 413
    assert resp_cap.json()["detail"]["ml_advisory_error"] == "event_limit_exceeded"

    # 3. Post-load on-disk model tampering is caught even when previously cached -> 503.
    _configure_ml_env(monkeypatch, trusted_root, model_id, model_version, model_sha256)
    assert client.post(f"/cases/{case_cap['id']}/ml-advisory").status_code == 200
    model_file = trusted_root / model_id / model_version / "model.joblib"
    model_file.write_bytes(model_file.read_bytes() + b"TAMPERED")
    resp_tampered = client.post(f"/cases/{case_cap['id']}/ml-advisory")
    assert resp_tampered.status_code == 503
    assert resp_tampered.json()["detail"]["ml_advisory_error"] == "model_unavailable_or_invalid"
