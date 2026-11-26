from __future__ import annotations

import hashlib
import io
import sqlite3
from pathlib import Path

import pytest
from pypdf import PdfReader

from backend.ml.advisory_service import (
    DEFAULT_DEPLOYED_MODEL_SHA256,
    ml_advisory_service,
)
from backend.ml.deployment_training import (
    DEFAULT_DEPLOYMENT_MODEL_ID,
    DEFAULT_DEPLOYMENT_MODEL_VERSION,
)
from backend.ml.model_artifact import DEFAULT_TRUSTED_MODELS_DIR


DEPLOYED_MODEL_FILE = (
    DEFAULT_TRUSTED_MODELS_DIR
    / DEFAULT_DEPLOYMENT_MODEL_ID
    / DEFAULT_DEPLOYMENT_MODEL_VERSION
    / "model.joblib"
)


@pytest.fixture
def client(isolated_app):
    return isolated_app[0]


@pytest.fixture
def paths(isolated_app):
    return isolated_app[1]


@pytest.fixture(autouse=True)
def _reset_ml_cache():
    ml_advisory_service.clear_cache()
    yield
    ml_advisory_service.clear_cache()


def _build_synthetic_30_event_csv() -> str:
    """Synthetic Apache-like CSV with normal traffic plus deterministic triggers for R-A, R-B, R-C, R-D."""
    lines = ["timestamp,source_ip,event_type,resource,status,bytes"]
    # 24 normal baseline GET events (bytes ~ 450-550)
    for idx in range(24):
        sec = f"{idx:02d}"
        lines.append(
            f"2026-02-01T12:00:{sec}Z,10.10.0.{(idx % 4) + 1},GET,/pages/item_{idx}.html,200,{480 + (idx % 5) * 10}"
        )
    # 3 rapid POST events to sensitive resources with 403 status (triggers R-A, R-B, R-C)
    lines.append("2026-02-01T12:01:00Z,192.168.50.99,POST,/admin/secret,403,240")
    lines.append("2026-02-01T12:01:01Z,192.168.50.99,POST,/admin/password,403,250")
    lines.append("2026-02-01T12:01:02Z,192.168.50.99,POST,/admin/confidential,403,260")
    # 2 more rapid 404 probes from same attacker IP
    lines.append("2026-02-01T12:01:03Z,192.168.50.99,GET,/backup.zip,404,180")
    lines.append("2026-02-01T12:01:04Z,192.168.50.99,GET,/secret.bak,404,190")
    # 1 unusually large transfer (triggers R-D: > mean + 2*stdev)
    lines.append("2026-02-01T12:01:10Z,192.168.50.99,GET,/export/archive.zip,200,500000")
    return "\n".join(lines) + "\n"


def _snapshot_all_tables(db_path: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(db_path)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid ASC").fetchall()
            for table in tables
        }
    finally:
        conn.close()


@pytest.mark.skipif(
    not DEPLOYED_MODEL_FILE.is_file(),
    reason="Deployed RandomForest artifact not present on disk.",
)
def test_complete_e2e_forensic_workflow_with_real_deployed_model_and_pdf(
    client, paths, monkeypatch
) -> None:
    # 1-3. Create synthetic case & upload 30-event CSV evidence.
    case_resp = client.post(
        "/cases",
        json={
            "case_number": "CASE-E2E-001",
            "title": "Phase 5 End-to-End Validation",
            "investigator": "Lead Examiner",
            "description": "Isolated synthetic workflow validation",
        },
    )
    assert case_resp.status_code == 200, case_resp.text
    case = case_resp.json()["case"]

    csv_content = _build_synthetic_30_event_csv()
    expected_sha256 = hashlib.sha256(csv_content.encode("utf-8")).hexdigest()
    upload_resp = client.post(
        f"/cases/{case['id']}/evidence",
        params={"source_description": "Synthetic Apache CSV export"},
        files={"file": ("synthetic_apache.csv", csv_content.encode("utf-8"), "text/csv")},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    evidence = upload_resp.json()["evidence"]
    assert evidence["sha256"] == expected_sha256

    # 4. Verify original and working file SHA-256 hashes on disk.
    orig_file = paths["upload"] / f"{evidence['evidence_id']}.csv"
    work_file = paths["working"] / f"{evidence['evidence_id']}_working.csv"
    assert hashlib.sha256(orig_file.read_bytes()).hexdigest() == expected_sha256
    assert hashlib.sha256(work_file.read_bytes()).hexdigest() == expected_sha256

    # 5. Record chain-of-custody and verify audit chain integrity.
    coc_resp = client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={
            "person": "Lead Examiner",
            "action": "Verified",
            "remarks": "SHA-256 confirmed at intake",
        },
    )
    assert coc_resp.status_code == 200, coc_resp.text
    coc_history = client.get(f"/evidence/{evidence['id']}/chain-of-custody")
    assert coc_history.status_code == 200
    assert len(coc_history.json()["chain_of_custody"]) == 1

    audit_verify = client.get("/audit/verify")
    assert audit_verify.status_code == 200
    assert audit_verify.json()["valid"] is True

    # 6. Execute deterministic R-A through R-D analysis.
    analyze_resp = client.post(f"/cases/{case['id']}/analyze")
    assert analyze_resp.status_code == 200, analyze_resp.text
    findings_resp = client.get(f"/cases/{case['id']}/findings")
    assert findings_resp.status_code == 200
    findings_data = findings_resp.json()
    rule_ids_triggered = {item["rule_id"] for item in findings_data["findings"]}
    assert {"R-A", "R-B", "R-C", "R-D"}.issubset(rule_ids_triggered)

    # Snapshot all files and DB tables before running optional ML advisory.
    db_before_ml = _snapshot_all_tables(paths["db"])
    findings_before_ml = findings_data

    # 7-9. Run optional ML advisory using the real SHA-256-pinned deployment model.
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_ENABLED", "true")
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_OPERATOR_APPROVED", "true")
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MODELS_DIR", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MODEL_ID", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MODEL_VERSION", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MODEL_SHA256", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MAX_EVENTS", raising=False)

    adv_resp_1 = client.post(f"/cases/{case['id']}/ml-advisory")
    adv_resp_2 = client.post(f"/cases/{case['id']}/ml-advisory")
    assert adv_resp_1.status_code == 200, adv_resp_1.text
    assert adv_resp_2.status_code == 200, adv_resp_2.text
    adv1 = adv_resp_1.json()
    adv2 = adv_resp_2.json()
    assert adv1 == adv2  # Deterministic across calls

    assert adv1["advisory_only"] is True
    assert adv1["deterministic_rules_unchanged"] is True
    assert adv1["model"]["model_id"] == DEFAULT_DEPLOYMENT_MODEL_ID
    assert adv1["model"]["model_version"] == DEFAULT_DEPLOYMENT_MODEL_VERSION
    assert adv1["model"]["model_sha256"] == DEFAULT_DEPLOYED_MODEL_SHA256
    assert adv1["model"]["decision_threshold"] == 0.5
    assert adv1["model"]["feature_schema_version"] == "1.0.0"
    assert adv1["model"]["score_semantics"]["calibrated_probability"] is False
    assert adv1["summary"]["total_events_scored"] == 30
    assert (
        adv1["summary"]["attack_associated_count"] + adv1["summary"]["normal_count"]
        == 30
    )
    assert len(adv1["predictions"]) == 30
    for pred in adv1["predictions"]:
        assert pred["score_type"] == "uncalibrated_model_score"
        assert pred["is_calibrated_probability"] is False
        assert pred["source_event_reference"]["evidence_id"] == evidence["evidence_id"]

    # Zero mutation of original file, working file, DB tables, findings, custody, or audit chain.
    assert hashlib.sha256(orig_file.read_bytes()).hexdigest() == expected_sha256
    assert hashlib.sha256(work_file.read_bytes()).hexdigest() == expected_sha256
    assert _snapshot_all_tables(paths["db"]) == db_before_ml
    assert client.get(f"/cases/{case['id']}/findings").json() == findings_before_ml
    assert client.get("/audit/verify").json()["valid"] is True

    # 10. Disable ML and verify ordinary forensic analysis & reporting still work.
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_ENABLED", "false")
    disabled_resp = client.post(f"/cases/{case['id']}/ml-advisory")
    assert disabled_resp.status_code == 503
    assert disabled_resp.json()["detail"]["ml_advisory_error"] == "ml_not_enabled"
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200

    # 12. Generate and validate PDF report; confirm advisory predictions are NOT silently included.
    report_resp = client.post(f"/cases/{case['id']}/report")
    assert report_resp.status_code == 200, report_resp.text
    report_info = report_resp.json()["report"]
    download_resp = client.get(
        f"/cases/{case['id']}/download", params={"path": report_info["path"]}
    )
    assert download_resp.status_code == 200, download_resp.text
    assert hashlib.sha256(download_resp.content).hexdigest() == report_info["sha256"]
    assert download_resp.content.startswith(b"%PDF")

    reader = PdfReader(io.BytesIO(download_resp.content))
    pdf_text = "\n".join((page.extract_text() or "") for page in reader.pages)
    assert "CASE-E2E-001" in pdf_text
    assert evidence["evidence_id"] in pdf_text
    assert DEFAULT_DEPLOYMENT_MODEL_ID not in pdf_text
    assert "uncalibrated_model_score" not in pdf_text
    assert "ML Advisory" not in pdf_text


def test_e2e_tampering_unsupported_format_oversized_and_client_control_rejection(
    client, paths, monkeypatch
) -> None:
    case = client.post(
        "/cases",
        json={
            "case_number": "CASE-E2E-SEC",
            "title": "Security & Fail-Closed Validation",
            "investigator": "Security Auditor",
        },
    ).json()["case"]

    csv_content = _build_synthetic_30_event_csv()
    evidence = client.post(
        f"/cases/{case['id']}/evidence",
        files={"file": ("sec.csv", csv_content.encode("utf-8"), "text/csv")},
    ).json()["evidence"]

    # 15. External client cannot activate ML, pass arbitrary paths, or override operator approval.
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_ENABLED", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_OPERATOR_APPROVED", raising=False)
    rogue_resp = client.post(
        f"/cases/{case['id']}/ml-advisory",
        params={
            "operator_approved": "true",
            "enabled": "true",
            "artifact_dir": "/tmp/evil",
            "model_path": "/tmp/evil/model.joblib",
        },
        json={"operator_approved": True, "artifact_dir": "/tmp/evil"},
    )
    assert rogue_resp.status_code == 503
    assert rogue_resp.json()["detail"]["ml_advisory_error"] == "ml_not_enabled"

    # Enable real deployed model for remaining security checks (if present).
    if DEPLOYED_MODEL_FILE.is_file():
        monkeypatch.setenv("FORENSIC_INSIGHT_ML_ENABLED", "true")
        monkeypatch.setenv("FORENSIC_INSIGHT_ML_OPERATOR_APPROVED", "true")

        # 14a. Oversized ML advisory request -> 413.
        monkeypatch.setenv("FORENSIC_INSIGHT_ML_MAX_EVENTS", "5")
        cap_resp = client.post(f"/cases/{case['id']}/ml-advisory")
        assert cap_resp.status_code == 413
        assert cap_resp.json()["detail"]["ml_advisory_error"] == "event_limit_exceeded"
        monkeypatch.delenv("FORENSIC_INSIGHT_ML_MAX_EVENTS", raising=False)

        # 13. Deliberate working-file tampering -> fail closed (409) for both ML advisory and /analyze.
        work_file = paths["working"] / f"{evidence['evidence_id']}_working.csv"
        work_file.write_text(csv_content + "2026-02-01T12:09:00Z,1.2.3.4,GET,/tampered,200,1\n", encoding="utf-8")
        assert client.post(f"/cases/{case['id']}/ml-advisory").status_code == 409
        assert client.post(f"/cases/{case['id']}/analyze").status_code == 409

        # 14b. Unsupported file format (.log) in a clean case -> 422.
        case_log = client.post(
            "/cases",
            json={
                "case_number": "CASE-E2E-LOG",
                "title": "Unsupported Format Case",
                "investigator": "Security Auditor",
            },
        ).json()["case"]
        client.post(
            f"/cases/{case_log['id']}/evidence",
            files={"file": ("raw_apache.log", b"10.0.0.1 - - [01/Feb/2026:12:00:00 +0000] \"GET / HTTP/1.1\" 200 10\n", "text/plain")},
        )
        log_resp = client.post(f"/cases/{case_log['id']}/ml-advisory")
        assert log_resp.status_code == 422
        assert log_resp.json()["detail"]["validation_error"] == "unsupported_format"
