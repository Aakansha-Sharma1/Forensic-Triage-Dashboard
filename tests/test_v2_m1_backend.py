from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from backend.database import get_connection, initialize_database
from backend.ml.advisory_service import ml_advisory_service


@pytest.fixture
def client(isolated_app):
    return isolated_app[0]


@pytest.fixture
def paths(isolated_app):
    return isolated_app[1]


@pytest.fixture(autouse=True)
def _clear_ml_cache():
    ml_advisory_service.clear_cache()
    yield
    ml_advisory_service.clear_cache()


def _sample_csv() -> str:
    return (
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-03-01T10:00:00Z,10.0.0.1,GET,/index.html,200,500\n"
        "2026-03-01T10:00:05Z,10.0.0.1,GET,/about.html,200,450\n"
        "2026-03-01T10:00:10Z,10.0.0.2,POST,/admin/secret,403,200\n"
        "2026-03-01T10:00:12Z,10.0.0.2,POST,/admin/password,403,210\n"
        "2026-03-01T10:00:14Z,10.0.0.2,POST,/admin/confidential,403,220\n"
        "2026-03-01T10:00:20Z,10.0.0.3,GET,/export/data.zip,200,250000\n"
    )


def _create_case(client, suffix: str) -> dict:
    resp = client.post(
        "/cases",
        json={
            "case_number": f"CASE-M1-{suffix}",
            "title": f"M1 Test Case {suffix}",
            "investigator": "M1 Examiner",
            "description": "Initial description",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["case"]


def _upload_csv(client, case_id: int, name: str, content: str) -> dict:
    resp = client.post(
        f"/cases/{case_id}/evidence",
        params={"source_description": "M1 test evidence"},
        files={"file": (name, content.encode("utf-8"), "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["evidence"]


def test_database_foreign_keys_indexes_and_idempotent_migrations(client, paths) -> None:
    # Foreign keys enabled per connection
    conn = get_connection()
    try:
        fk_enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk_enabled == 1
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    finally:
        conn.close()

    expected_indexes = {
        "idx_evidence_case_id",
        "idx_findings_evidence_id",
        "idx_findings_severity",
        "idx_chain_of_custody_evidence_id",
        "idx_audit_events_case_id",
        "idx_audit_events_evidence_id",
        "idx_reports_case_id",
    }
    assert expected_indexes.issubset(indexes)

    # Create data, then run initialize_database() repeatedly and verify zero data loss
    case = _create_case(client, "mig")
    _upload_csv(client, case["id"], "mig.csv", _sample_csv())
    initialize_database()
    initialize_database()

    detail = client.get(f"/cases/{case['id']}")
    assert detail.status_code == 200
    assert detail.json()["case"]["case_number"] == case["case_number"]
    assert detail.json()["summary"]["evidence_count"] == 1
    assert client.get("/audit/verify").json()["valid"] is True


def test_cors_hardening_defaults_to_localhost_and_forbids_wildcard_with_credentials(client) -> None:
    from backend.main import CORS_ALLOW_CREDENTIALS, CORS_ALLOWED_ORIGINS

    assert "*" not in CORS_ALLOWED_ORIGINS
    assert any("127.0.0.1" in origin for origin in CORS_ALLOWED_ORIGINS)
    assert not ("*" in CORS_ALLOWED_ORIGINS and CORS_ALLOW_CREDENTIALS)


def test_events_endpoint_integrity_pagination_filtering_and_sorting(client, paths) -> None:
    case = _create_case(client, "events")
    csv_text = _sample_csv()
    evidence = _upload_csv(client, case["id"], "events.csv", csv_text)
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200

    # Backwards-compatible unpaginated call
    all_resp = client.get(f"/cases/{case['id']}/events")
    assert all_resp.status_code == 200
    all_body = all_resp.json()
    assert all_body["total_events"] == 6
    assert all_body["filtered_events"] == 6
    assert len(all_body["events"]) == 6
    assert all_body["event_type_counts"] == {"GET": 3, "POST": 3}
    assert len(all_body["source_summary"]) == 3

    # Paginated + sorted by bytes desc
    page1 = client.get(
        f"/cases/{case['id']}/events",
        params={"page": 1, "page_size": 2, "sort_by": "bytes", "sort_order": "desc"},
    )
    assert page1.status_code == 200
    p1 = page1.json()
    assert p1["total_pages"] == 3
    assert len(p1["events"]) == 2
    assert p1["events"][0]["bytes"] == "250000"

    # Filter by source_ip and only_flagged
    flagged = client.get(
        f"/cases/{case['id']}/events",
        params={"source_ip": "10.0.0.2", "only_flagged": "true"},
    )
    assert flagged.status_code == 200
    assert flagged.json()["filtered_events"] == 3
    assert all(ev["source"] == "10.0.0.2" and len(ev["findings"]) > 0 for ev in flagged.json()["events"])

    # Invalid pagination bounds -> 422
    assert client.get(f"/cases/{case['id']}/events", params={"page": 0}).status_code == 422
    assert client.get(f"/cases/{case['id']}/events", params={"page_size": 501}).status_code == 422

    # Tampered working file -> fail closed with HTTP 409 (never silently skipped)
    work_file = paths["working"] / f"{evidence['evidence_id']}_working.csv"
    work_file.write_text(csv_text + "2026-03-01T10:01:00Z,9.9.9.9,GET,/tampered,200,1\n", encoding="utf-8")
    tampered_resp = client.get(f"/cases/{case['id']}/events")
    assert tampered_resp.status_code == 409
    assert tampered_resp.json()["detail"]["integrity_error"] == "working_hash_mismatch"


def test_workspace_endpoints_dashboard_case_preview_verify_triage_reports_and_ml_status(
    client, paths
) -> None:
    case = _create_case(client, "ws")
    csv_text = _sample_csv()
    expected_sha = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    evidence = _upload_csv(client, case["id"], "ws.csv", csv_text)
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200

    # 1. GET /ml/status never deserializes model.joblib
    ml_status = client.get("/ml/status")
    assert ml_status.status_code == 200
    assert ml_status.json()["deserialized"] is False
    assert ml_status.json()["advisory_only"] is True
    assert ml_advisory_service._cached_artifact is None

    # 2. GET /cases/{id} returns real verified CSV event count (6, not fabricated 5x findings)
    detail = client.get(f"/cases/{case['id']}")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["case"]["status"] == "Open"
    assert detail_body["summary"]["verified_csv_event_count"] == 6
    assert detail_body["summary"]["integrity_verified_evidence_count"] == 1
    assert detail_body["summary"]["findings_count"] >= 4

    # 3. PATCH /cases/{id} updates lifecycle status & audits CASE_UPDATED
    patch_case = client.patch(
        f"/cases/{case['id']}",
        json={"status": "In Analysis", "description": "Updated scope"},
    )
    assert patch_case.status_code == 200
    assert patch_case.json()["case"]["status"] == "In Analysis"
    assert client.patch(f"/cases/{case['id']}", json={"status": "InvalidStatus"}).status_code == 422

    # 4. GET /evidence/{ref}/preview is read-only, paginated, and does not mutate files or DB
    orig_file = paths["upload"] / f"{evidence['evidence_id']}.csv"
    work_file = paths["working"] / f"{evidence['evidence_id']}_working.csv"
    preview = client.get(
        f"/evidence/{evidence['evidence_id']}/preview",
        params={"page": 1, "page_size": 2},
    )
    assert preview.status_code == 200
    prev_body = preview.json()
    assert prev_body["total_rows"] == 6
    assert prev_body["total_pages"] == 3
    assert len(prev_body["rows"]) == 2
    assert hashlib.sha256(orig_file.read_bytes()).hexdigest() == expected_sha
    assert hashlib.sha256(work_file.read_bytes()).hexdigest() == expected_sha

    # 5. POST /evidence/{ref}/verify confirms integrity and audits
    verify_resp = client.post(f"/evidence/{evidence['evidence_id']}/verify")
    assert verify_resp.status_code == 200
    assert verify_resp.json()["verified"] is True
    assert verify_resp.json()["sha256"] == expected_sha

    # 6. PATCH /findings/{finding_id}/status updates triage status without changing rule metadata
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    target_finding = findings[0]
    triage = client.patch(
        f"/findings/{target_finding['finding_id']}/status",
        json={"status": "Confirmed", "triage_notes": "Confirmed unauthorized POST"},
    )
    assert triage.status_code == 200
    updated_f = triage.json()["finding"]
    assert updated_f["status"] == "Confirmed"
    assert updated_f["triage_notes"] == "Confirmed unauthorized POST"
    assert updated_f["rule_id"] == target_finding["rule_id"]
    assert updated_f["rule_fingerprint"] == target_finding["rule_fingerprint"]

    # 7. POST /cases/{id}/report persists into GET /cases/{id}/reports
    rep_resp = client.post(f"/cases/{case['id']}/report")
    assert rep_resp.status_code == 200
    rep_list = client.get(f"/cases/{case['id']}/reports")
    assert rep_list.status_code == 200
    assert rep_list.json()["total_reports"] == 1
    assert rep_list.json()["reports"][0]["sha256"] == rep_resp.json()["report"]["sha256"]
    assert rep_list.json()["reports"][0]["file_exists"] is True

    # 8. GET /dashboard/summary aggregates real metrics & valid audit chain
    dash = client.get("/dashboard/summary")
    assert dash.status_code == 200
    dash_body = dash.json()
    assert dash_body["metrics"]["total_cases"] == 1
    assert dash_body["metrics"]["cases_by_status"]["In Analysis"] == 1
    assert dash_body["metrics"]["total_evidence"] == 1
    assert dash_body["metrics"]["total_reports"] == 1
    assert dash_body["audit_chain"]["valid"] is True
    assert dash_body["ml_status"]["deserialized"] is False
    assert "authentication" in dash_body["security_notice"].lower()

    # 9. On-demand verify & preview fail closed on tampering
    orig_file.write_bytes(orig_file.read_bytes() + b"x")
    assert client.post(f"/evidence/{evidence['evidence_id']}/verify").status_code == 409
    assert client.get(f"/evidence/{evidence['evidence_id']}/preview").status_code == 409
    assert client.get("/audit/verify").json()["valid"] is True
