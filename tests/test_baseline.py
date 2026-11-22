import hashlib
import sqlite3
import statistics
import threading
from pathlib import Path

import pytest
from pypdf import PdfReader


@pytest.fixture
def client(isolated_app):
    return isolated_app[0]


@pytest.fixture
def paths(isolated_app):
    return isolated_app[1]


def create_case(client, suffix="baseline"):
    payload = {
        "case_number": f"CASE-{suffix}",
        "title": "Automated Baseline Case",
        "investigator": "Baseline Investigator",
        "description": "Temporary isolated test case",
    }
    response = client.post("/cases", json=payload)
    assert response.status_code == 200
    return response.json()["case"]


def upload_csv(client, case_id, name, content):
    response = client.post(
        f"/cases/{case_id}/evidence",
        params={"source_description": "Automated baseline evidence"},
        files={"file": (name, content.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()["evidence"]


def test_case_creation_and_retrieval(client):
    case = create_case(client, "case-retrieval")

    response = client.get("/cases")
    assert response.status_code == 200
    cases = response.json()["cases"]
    assert any(item["id"] == case["id"] for item in cases)

    duplicate = client.post("/cases", json={
        "case_number": case["case_number"],
        "title": "Duplicate",
        "investigator": "Investigator",
    })
    assert duplicate.status_code == 409


def test_invalid_case_input_is_rejected(client):
    response = client.post("/cases", json={
        "case_number": "INVALID",
        "investigator": "Baseline Investigator",
    })
    assert response.status_code == 422


def test_evidence_upload_hash_and_inventory(client, paths):
    case = create_case(client, "evidence-hash")
    content = "timestamp,source_ip,event_type,resource,status,bytes\n2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100\n"

    evidence = upload_csv(client, case["id"], "evidence.csv", content)
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    assert evidence["evidence_id"].startswith(f"EV-{case['id']}-")
    assert evidence["sha256"] == expected_hash
    assert evidence["file_size"] == len(content.encode("utf-8"))
    assert evidence["status"] == "Registered"

    stored = Path(evidence["id"] and evidence["evidence_id"])
    original_files = list(paths["upload"].glob("*.csv"))
    assert len(original_files) == 1
    stored_file = original_files[0]
    assert stored_file.resolve().is_relative_to(paths["upload"].resolve())
    assert stored_file.read_bytes() == content.encode("utf-8")
    assert hashlib.sha256(stored_file.read_bytes()).hexdigest() == expected_hash

    inventory = client.get(f"/cases/{case['id']}/evidence")
    assert inventory.status_code == 200
    inventory_evidence = inventory.json()["evidence"][0]
    assert inventory_evidence["sha256"] == expected_hash

    import sqlite3
    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT stored_filename, sha256 FROM evidence WHERE evidence_id = ?",
        (evidence["evidence_id"],),
    ).fetchone()
    connection.close()
    assert row is not None
    assert Path(row["stored_filename"]).resolve().is_relative_to(paths["upload"].resolve())
    assert Path(row["stored_filename"]).resolve() == stored_file.resolve()
    assert row["sha256"] == expected_hash


def test_unsupported_evidence_type(client):
    case = create_case(client, "unsupported-type")
    response = client.post(
        f"/cases/{case['id']}/evidence",
        files={"file": ("evidence.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "File type not allowed" in response.json()["detail"]


def test_chain_of_custody_creation_and_retrieval(client):
    case = create_case(client, "custody")
    evidence = upload_csv(client, case["id"], "custody.csv", "timestamp,source_ip,event_type,resource,status,bytes\n")

    response = client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={"person": "Forensic Analyst", "action": "Received", "remarks": "Initial intake"},
    )
    assert response.status_code == 200
    entry = response.json()["entry"]
    assert entry["evidence_id"] == evidence["id"]
    assert entry["person"] == "Forensic Analyst"

    history = client.get(f"/evidence/{evidence['id']}/chain-of-custody")
    assert history.status_code == 200
    assert history.json()["chain_of_custody"][0]["remarks"] == "Initial intake"


def test_chain_of_custody_requires_all_fields(client):
    case = create_case(client, "custody-invalid")
    evidence = upload_csv(client, case["id"], "custody-invalid.csv", "timestamp,source_ip,event_type,resource,status,bytes\n")

    response = client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={"person": "", "action": "", "remarks": ""},
    )
    assert response.status_code == 422


def test_rule_a_http_403(client):
    case = create_case(client, "rule-a")
    upload_csv(client, case["id"], "rule-a.csv",
               "timestamp,source_ip,event_type,resource,status,bytes\n"
               "2026-01-01 00:00:00,10.0.0.1,HTTP,/private,403,100\n")

    analysis = client.post(f"/cases/{case['id']}/analyze")
    assert analysis.status_code == 200
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    matching = [item for item in findings if "HTTP 403" in item["reason"]]
    assert len(matching) == 1
    assert matching[0]["severity"] == "MEDIUM"
    assert matching[0]["source"] == "10.0.0.1"


def test_rule_a_negative_no_403(client):
    case = create_case(client, "rule-a-negative")
    upload_csv(client, case["id"], "rule-a-negative.csv",
               "timestamp,source_ip,event_type,resource,status,bytes\n"
               "2026-01-01 00:00:00,10.0.0.1,HTTP,/private,200,100\n")
    client.post(f"/cases/{case['id']}/analyze")
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    assert not [item for item in findings if "HTTP 403" in item["reason"]]


def test_rule_b_post_to_sensitive_resource(client):
    case = create_case(client, "rule-b")
    upload_csv(client, case["id"], "rule-b.csv",
               "timestamp,source_ip,event_type,resource,status,bytes\n"
               "2026-01-01 00:00:00,10.0.0.2,POST,/api/admin/users,200,100\n")

    analysis = client.post(f"/cases/{case['id']}/analyze")
    assert analysis.status_code == 200
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    matching = [item for item in findings if "restricted resource" in item["reason"]]
    assert len(matching) == 1
    assert matching[0]["severity"] == "HIGH"
    assert matching[0]["source"] == "10.0.0.2"


def test_rule_b_negative_regular_post(client):
    case = create_case(client, "rule-b-negative")
    upload_csv(client, case["id"], "rule-b-negative.csv",
               "timestamp,source_ip,event_type,resource,status,bytes\n"
               "2026-01-01 00:00:00,10.0.0.2,POST,/api/public/page,200,100\n")
    client.post(f"/cases/{case['id']}/analyze")
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    assert not [item for item in findings if "restricted resource" in item["reason"]]


def test_rule_c_repeated_posts_from_same_source(client):
    case = create_case(client, "rule-c")
    rows = "timestamp,source_ip,event_type,resource,status,bytes\n" + "".join(
        f"2026-01-01 00:00:{index:02d},10.0.0.3,POST,/upload/{index},200,100\n"
        for index in range(3)
    )
    upload_csv(client, case["id"], "rule-c.csv", rows)

    analysis = client.post(f"/cases/{case['id']}/analyze")
    assert analysis.status_code == 200
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    matching = [item for item in findings if "Multiple POST requests" in item["reason"]]

    # Current implementation clears post_requests[source_ip] after the first row,
    # so exactly one Rule C finding is produced for this three-row source.
    assert len(matching) == 1
    assert "(3)" in matching[0]["reason"]
    assert matching[0]["source"] == "10.0.0.3"
    assert matching[0]["severity"] == "MEDIUM"


def test_rule_c_negative_two_posts_do_not_trigger(client):
    case = create_case(client, "rule-c-negative")
    rows = "timestamp,source_ip,event_type,resource,status,bytes\n" + "".join(
        f"2026-01-01 00:00:{index:02d},10.0.0.8,POST,/upload/{index},200,100\n"
        for index in range(2)
    )
    upload_csv(client, case["id"], "rule-c-negative.csv", rows)
    client.post(f"/cases/{case['id']}/analyze")
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    assert not [item for item in findings if "Multiple POST requests" in item["reason"]]


def test_rule_d_large_transfer_preserves_mean_plus_two_stdev_threshold(client):
    case = create_case(client, "rule-d")
    values = [100] * 9 + [1_000_000]
    rows = "timestamp,source_ip,event_type,resource,status,bytes\n" + "".join(
        f"2026-01-01 00:00:{index:02d},10.0.0.4,GET,/normal-{index},200,{values[index]}\n"
        for index in range(9)
    ) + "2026-01-01 00:00:09,10.0.0.4,GET,/large,200,1000000\n"
    upload_csv(client, case["id"], "rule-d.csv", rows)

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    threshold = mean + (2 * stdev)
    assert 1_000_000 > threshold

    analysis = client.post(f"/cases/{case['id']}/analyze")
    assert analysis.status_code == 200
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    matching = [item for item in findings if "2+ standard deviations" in item["reason"]]
    assert len(matching) == 1
    assert matching[0]["resource"] == "/large"
    assert matching[0]["severity"] == "MEDIUM"


def test_rule_d_negative_small_uniform_transfers(client):
    case = create_case(client, "rule-d-negative")
    rows = "timestamp,source_ip,event_type,resource,status,bytes\n" + "".join(
        f"2026-01-01 00:00:{index:02d},10.0.0.9,GET,/normal-{index},200,100\n"
        for index in range(5)
    )
    upload_csv(client, case["id"], "rule-d-negative.csv", rows)
    client.post(f"/cases/{case['id']}/analyze")
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    assert not [item for item in findings if "2+ standard deviations" in item["reason"]]


def test_rule_registry_metadata_is_complete_and_deterministic(client):
    from backend.rules import calculate_rule_fingerprint, get_rule_metadata

    response = client.get("/rules")
    assert response.status_code == 200
    metadata = response.json()

    assert [rule["rule_id"] for rule in metadata["rules"]] == ["R-A", "R-B", "R-C", "R-D"]
    assert {rule["version"] for rule in metadata["rules"]} == {"1.0.0"}
    assert {rule["severity"] for rule in metadata["rules"]} == {"MEDIUM", "HIGH"}
    assert metadata["fingerprint"] == calculate_rule_fingerprint()
    assert len(metadata["fingerprint"]) == 64
    assert metadata["rules"][0]["detection_conditions"]["status_contains"] == "403"
    assert metadata["rules"][1]["detection_conditions"]["resource_keywords"] == [
        "secret", "admin", "password", "confidential", ".zip"
    ]
    assert metadata["rules"][2]["detection_conditions"]["minimum_request_count"] == 3
    assert metadata["rules"][3]["detection_conditions"]["threshold_multiplier"] == 2
    assert get_rule_metadata()["fingerprint"] == metadata["fingerprint"]


def test_new_findings_and_analysis_audit_store_rule_provenance(client):
    from backend.rules import calculate_rule_fingerprint

    case = create_case(client, "rule-provenance")
    rows = ["timestamp,source_ip,event_type,resource,status,bytes"]
    rows.extend(
        f"2026-01-01 00:00:{index:02d},10.0.0.11,"
        f"{'POST' if index < 3 else 'GET'},"
        f"{'/admin' if index == 0 else '/upload' if index < 3 else '/normal'},"
        f"{'403' if index == 0 else '200'},{100 if index < 9 else 1000000}"
        for index in range(10)
    )
    upload_csv(client, case["id"], "rule-provenance.csv", "\n".join(rows) + "\n")

    analysis = client.post(f"/cases/{case['id']}/analyze")
    assert analysis.status_code == 200
    expected_fingerprint = calculate_rule_fingerprint()
    findings = client.get(f"/cases/{case['id']}/findings").json()["findings"]

    assert {finding["rule_id"] for finding in findings} == {"R-A", "R-B", "R-C", "R-D"}
    assert all(finding["rule_version"] == "1.0.0" for finding in findings)
    assert {finding["rule_fingerprint"] for finding in findings} == {expected_fingerprint}

    audit = client.get("/audit", params={"case_id": case["id"]}).json()["history"]
    completed = [event for event in audit if event["action_type"] == "ANALYSIS_COMPLETED"]
    assert len(completed) == 1
    assert completed[0]["metadata"]["rule_fingerprint"] == expected_fingerprint


def test_finding_persistence_and_event_reference(client):
    case = create_case(client, "finding-reference")
    evidence = upload_csv(client, case["id"], "finding-reference.csv",
                         "timestamp,source_ip,event_type,resource,status,bytes\n"
                         "2026-01-01 00:00:00,10.0.0.5,HTTP,/private,403,100\n")

    analysis = client.post(f"/cases/{case['id']}/analyze")
    assert analysis.status_code == 200
    assert analysis.json()["findings_count"] == 1

    findings_response = client.get(f"/cases/{case['id']}/findings")
    assert findings_response.status_code == 200
    findings = findings_response.json()["findings"]
    assert len(findings) == 1
    finding = findings[0]

    assert finding["evidence_code"] == evidence["evidence_id"]
    assert finding["event_id"] == "0"
    assert finding["source"] == "10.0.0.5"
    assert finding["resource"] == "/private"
    assert finding["severity"] == "MEDIUM"
    assert "HTTP 403" in finding["reason"]

    detail = client.get(f"/findings/{finding['finding_id']}")
    assert detail.status_code == 200
    detail_data = detail.json()
    assert detail_data["evidence_code"] == evidence["evidence_id"]
    assert detail_data["event_id"] == "0"
    assert detail_data["status"] == "Requires Review"
    assert detail_data["technical_observation"]
    assert detail_data["plain_language_explanation"]

    events = client.get(f"/cases/{case['id']}/events")
    assert events.status_code == 200
    event = events.json()["events"][0]
    assert event["evidence_id"] == evidence["id"]
    assert event["event_id"] == "0"
    assert event["resource"] == "/private"
    assert any(link["finding_id"] == finding["finding_id"] for link in event["findings"])


def test_missing_evidence_and_finding_return_404(client):
    missing_coc = client.get("/evidence/999999/chain-of-custody")
    assert missing_coc.status_code == 404

    missing_coc_post = client.post(
        "/evidence/999999/chain-of-custody",
        json={"person": "Nobody", "action": "Received", "remarks": "Missing evidence"},
    )
    assert missing_coc_post.status_code == 404

    missing_finding = client.get("/findings/FD-999999-NOTFOUND")
    assert missing_finding.status_code == 404


def test_pdf_report_generation_and_download(client, paths):
    case = create_case(client, "report")
    evidence_content = (
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.6,HTTP,/private,403,100\n"
    )
    evidence = upload_csv(client, case["id"], "report.csv", evidence_content)

    custody = client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={"person": "Report Analyst", "action": "Verified", "remarks": "Report test custody"},
    )
    assert custody.status_code == 200

    analysis = client.post(f"/cases/{case['id']}/analyze")
    assert analysis.status_code == 200

    response = client.post(f"/cases/{case['id']}/report")
    assert response.status_code == 200
    report = response.json()["report"]

    report_path = Path(report["path"])
    assert report_path.is_file()
    assert report_path.resolve().is_relative_to(paths["reports"].resolve())
    assert report_path.read_bytes().startswith(b"%PDF")

    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert report["sha256"] == digest

    reader = PdfReader(str(report_path))
    report_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert evidence["evidence_id"] in report_text
    assert evidence["sha256"] in report_text
    assert "HTTP 403 Forbidden status detected" in report_text
    assert "Rule ID: R-A" in report_text
    assert "Rule Version: 1.0.0" in report_text
    from backend.rules import calculate_rule_fingerprint
    assert calculate_rule_fingerprint() in report_text
    assert "Report Analyst" in report_text
    assert "Verified" in report_text
    assert "Report test custody" in report_text

    download = client.get(f"/cases/{case['id']}/download", params={"path": str(report_path)})
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"
    assert download.content.startswith(b"%PDF")

    working_files = list(paths["working"].glob("*_working.csv"))
    assert len(working_files) == 1
    assert working_files[0].resolve().is_relative_to(paths["working"].resolve())

def get_isolated_stored_paths(paths, evidence_id):
    import sqlite3
    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT stored_filename, sha256 FROM evidence WHERE evidence_id = ?",
        (evidence_id,),
    ).fetchone()
    connection.close()
    assert row is not None
    return Path(row["stored_filename"]), paths["working"] / f"{evidence_id}_working.csv", row["sha256"]


def test_integrity_valid_original_and_working_pass(client, paths):
    case = create_case(client, "integrity-valid")
    evidence = upload_csv(
        client,
        case["id"],
        "integrity-valid.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.7,HTTP,/private,403,100\n",
    )
    original, working, recorded_hash = get_isolated_stored_paths(paths, evidence["evidence_id"])

    assert original.is_file()
    assert working.is_file()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == recorded_hash
    assert hashlib.sha256(working.read_bytes()).hexdigest() == recorded_hash

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 200
    assert response.json()["findings_count"] == 1


def test_modified_original_evidence_is_rejected_without_new_findings(client, paths):
    case = create_case(client, "integrity-original-modified")
    evidence = upload_csv(
        client,
        case["id"],
        "integrity-original-modified.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.7,HTTP,/private,403,100\n",
    )
    original, working, recorded_hash = get_isolated_stored_paths(paths, evidence["evidence_id"])

    assert original.read_bytes() != b"modified evidence"
    original.write_bytes(b"modified evidence")

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 409
    assert response.json()["detail"]["integrity_error"] == "original_hash_mismatch"
    assert response.json()["detail"]["evidence_id"] == evidence["evidence_id"]
    assert response.json()["detail"]["recorded_sha256"] == recorded_hash

    assert client.get(f"/cases/{case['id']}/findings").json()["findings"] == []


def test_modified_working_evidence_is_rejected(client, paths):
    case = create_case(client, "integrity-working-modified")
    evidence = upload_csv(
        client,
        case["id"],
        "integrity-working-modified.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.7,HTTP,/private,403,100\n",
    )
    original, working, recorded_hash = get_isolated_stored_paths(paths, evidence["evidence_id"])

    working.write_text("modified working evidence", encoding="utf-8")
    response = client.post(f"/cases/{case['id']}/analyze")

    assert response.status_code == 409
    assert response.json()["detail"]["integrity_error"] == "working_hash_mismatch"
    assert response.json()["detail"]["recorded_sha256"] == recorded_hash
    assert original.is_file()


def test_missing_original_evidence_is_rejected(client, paths):
    case = create_case(client, "integrity-original-missing")
    evidence = upload_csv(
        client,
        case["id"],
        "integrity-original-missing.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.7,HTTP,/private,403,100\n",
    )
    original, working, _ = get_isolated_stored_paths(paths, evidence["evidence_id"])

    original.unlink()
    response = client.post(f"/cases/{case['id']}/analyze")

    assert response.status_code == 409
    assert response.json()["detail"]["integrity_error"] == "missing_original"
    assert working.is_file()


def test_missing_working_evidence_is_rejected(client, paths):
    case = create_case(client, "integrity-working-missing")
    evidence = upload_csv(
        client,
        case["id"],
        "integrity-working-missing.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.7,HTTP,/private,403,100\n",
    )
    original, working, _ = get_isolated_stored_paths(paths, evidence["evidence_id"])

    working.unlink()
    response = client.post(f"/cases/{case['id']}/analyze")

    assert response.status_code == 409
    assert response.json()["detail"]["integrity_error"] == "missing_working"
    assert original.is_file()


def test_integrity_failure_preserves_previously_stored_findings(client, paths):
    case = create_case(client, "integrity-preserve-findings")
    evidence = upload_csv(
        client,
        case["id"],
        "integrity-preserve-findings.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.7,HTTP,/private,403,100\n",
    )

    first = client.post(f"/cases/{case['id']}/analyze")
    assert first.status_code == 200
    before = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    assert len(before) == 1

    original, _, _ = get_isolated_stored_paths(paths, evidence["evidence_id"])
    original.write_bytes(original.read_bytes() + b"tampered")

    blocked = client.post(f"/cases/{case['id']}/analyze")
    assert blocked.status_code == 409

    after = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    assert after == before


def test_report_valid_evidence_generation(client):
    case = create_case(client, "report-integrity-valid")
    evidence = upload_csv(
        client,
        case["id"],
        "report-integrity-valid.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.6,HTTP,/private,403,100\n",
    )
    custody = client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={"person": "Report Analyst", "action": "Verified", "remarks": "Integrity approved"},
    )
    assert custody.status_code == 200
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200

    report = client.post(f"/cases/{case['id']}/report")
    assert report.status_code == 200
    assert report.json()["report"]["sha256"]


def test_report_modified_original_blocks_without_overwrite(client, paths):
    case = create_case(client, "report-original-modified")
    evidence = upload_csv(
        client,
        case["id"],
        "report-original-modified.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.6,HTTP,/private,403,100\n",
    )
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200

    custody_before = client.get(f"/evidence/{evidence['id']}/chain-of-custody").json()["chain_of_custody"]
    findings_before = client.get(f"/cases/{case['id']}/findings").json()["findings"]

    first = client.post(f"/cases/{case['id']}/report")
    assert first.status_code == 200
    report_path = Path(first.json()["report"]["path"])
    original_bytes = report_path.read_bytes()

    original, _, recorded_hash = get_isolated_stored_paths(paths, evidence["evidence_id"])
    original.write_bytes(original.read_bytes() + b"tampered")

    blocked = client.post(f"/cases/{case['id']}/report")
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["integrity_error"] == "original_hash_mismatch"
    assert blocked.json()["detail"]["evidence_id"] == evidence["evidence_id"]
    assert blocked.json()["detail"]["recorded_sha256"] == recorded_hash

    assert report_path.read_bytes() == original_bytes
    assert client.get(f"/cases/{case['id']}/findings").json()["findings"] == findings_before
    assert client.get(f"/evidence/{evidence['id']}/chain-of-custody").json()["chain_of_custody"] == custody_before


def test_report_modified_working_blocks_generation(client, paths):
    case = create_case(client, "report-working-modified")
    evidence = upload_csv(
        client,
        case["id"],
        "report-working-modified.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.6,HTTP,/private,403,100\n",
    )
    _, working, recorded_hash = get_isolated_stored_paths(paths, evidence["evidence_id"])
    working.write_text("tampered working evidence", encoding="utf-8")

    blocked = client.post(f"/cases/{case['id']}/report")
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["integrity_error"] == "working_hash_mismatch"
    assert blocked.json()["detail"]["recorded_sha256"] == recorded_hash


def test_report_missing_original_blocks_generation(client, paths):
    case = create_case(client, "report-original-missing")
    evidence = upload_csv(
        client,
        case["id"],
        "report-original-missing.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.6,HTTP,/private,403,100\n",
    )
    original, _, _ = get_isolated_stored_paths(paths, evidence["evidence_id"])
    original.unlink()

    blocked = client.post(f"/cases/{case['id']}/report")
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["integrity_error"] == "missing_original"


def test_report_missing_working_blocks_generation(client, paths):
    case = create_case(client, "report-working-missing")
    evidence = upload_csv(
        client,
        case["id"],
        "report-working-missing.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.6,HTTP,/private,403,100\n",
    )
    _, working, _ = get_isolated_stored_paths(paths, evidence["evidence_id"])
    working.unlink()

    blocked = client.post(f"/cases/{case['id']}/report")
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["integrity_error"] == "missing_working"


def test_evidence_upload_exactly_at_maximum_size(client, paths):
    case = create_case(client, "upload-exact-limit")
    maximum_size = 100 * 1024 * 1024

    response = client.post(
        f"/cases/{case['id']}/evidence",
        params={"source_description": "Exact limit evidence"},
        files={"file": ("exact-limit.csv", b"A" * maximum_size, "text/csv")},
    )

    assert response.status_code == 200, response.text
    evidence = response.json()["evidence"]
    assert evidence["file_size"] == maximum_size

    original = next(paths["upload"].glob(f"{evidence['evidence_id']}.*"))
    working = paths["working"] / f"{evidence['evidence_id']}_working.csv"
    assert original.stat().st_size == maximum_size
    assert working.stat().st_size == maximum_size


def test_oversized_evidence_is_rejected_without_partial_artifacts(client, paths):
    case = create_case(client, "upload-over-limit")
    maximum_size = 100 * 1024 * 1024

    response = client.post(
        f"/cases/{case['id']}/evidence",
        params={"source_description": "Oversized evidence"},
        files={"file": ("over-limit.csv", b"B" * (maximum_size + 1), "text/csv")},
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["upload_error"] == "file_too_large"
    assert detail["maximum_size_bytes"] == maximum_size
    assert detail["received_size_bytes"] == maximum_size + 1

    evidence_response = client.get(f"/cases/{case['id']}/evidence")
    assert evidence_response.status_code == 200
    assert evidence_response.json()["evidence"] == []
    assert list(paths["upload"].iterdir()) == []
    assert list(paths["working"].iterdir()) == []


def test_oversized_upload_does_not_create_custody_entries(client, paths):
    case = create_case(client, "upload-over-limit-custody")
    maximum_size = 100 * 1024 * 1024

    response = client.post(
        f"/cases/{case['id']}/evidence",
        files={"file": ("over-limit.txt", b"C" * (maximum_size + 1024), "text/plain")},
    )

    assert response.status_code == 413
    evidence_response = client.get(f"/cases/{case['id']}/evidence")
    assert evidence_response.json()["evidence"] == []

    db_rows = client.get(f"/cases/{case['id']}/evidence").json()["evidence"]
    assert db_rows == []
    assert list(paths["upload"].iterdir()) == []
    assert list(paths["working"].iterdir()) == []


def test_valid_csv_analysis_succeeds_with_schema(client):
    case = create_case(client, "validation-valid")
    upload_csv(
        client,
        case["id"],
        "valid-schema.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100\n",
    )

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 200
    assert response.json()["findings_count"] == 0


def test_missing_columns_rejected(client):
    case = create_case(client, "validation-missing-columns")
    evidence = upload_csv(client, case["id"], "missing-columns.csv", "timestamp,status\n2026-01-01 00:00:00,200\n")

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["validation_error"] == "missing_columns"
    assert detail["evidence_id"] == evidence["evidence_id"]


def test_invalid_timestamp_and_bytes_rejected(client):
    case = create_case(client, "validation-invalid-values")
    upload_csv(
        client,
        case["id"],
        "invalid-values.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "not-a-timestamp,10.0.0.1,GET,/ok,200,not-a-number\n",
    )

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 422
    assert response.json()["detail"]["validation_error"] in {"invalid_timestamp", "invalid_numeric_bytes"}


def test_empty_malformed_duplicate_and_missing_row_values_rejected(client):
    case = create_case(client, "validation-empty")
    upload_csv(client, case["id"], "empty.csv", "")
    empty = client.post(f"/cases/{case['id']}/analyze")
    assert empty.status_code == 422
    assert empty.json()["detail"]["validation_error"] == "empty_file"

    case = create_case(client, "validation-malformed")
    upload_csv(
        client,
        case["id"],
        "malformed.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200\n",
    )
    malformed = client.post(f"/cases/{case['id']}/analyze")
    assert malformed.status_code == 422
    assert malformed.json()["detail"]["validation_error"] == "malformed_row"

    case = create_case(client, "validation-duplicates")
    upload_csv(
        client,
        case["id"],
        "duplicates.csv",
        "timestamp,source_ip,source,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,,GET,/ok,200,100\n",
    )
    duplicates = client.post(f"/cases/{case['id']}/analyze")
    assert duplicates.status_code == 422
    assert duplicates.json()["detail"]["validation_error"] == "duplicate_headers"


def test_duplicate_headers_rejected(client):
    case = create_case(client, "validation-duplicate-headers")
    upload_csv(
        client,
        case["id"],
        "duplicate-headers.csv",
        "timestamp,source_ip,event_type,resource,status,bytes,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100,100\n",
    )

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 422
    assert response.json()["detail"]["validation_error"] == "duplicate_headers"


def test_unsupported_json_is_stored_but_analysis_blocked(client):
    case = create_case(client, "validation-json-format")
    evidence = upload_csv(client, case["id"], "events.json", '{"timestamp":"2026-01-01"}')

    inventory = client.get(f"/cases/{case['id']}/evidence")
    assert inventory.status_code == 200
    assert inventory.json()["evidence"][0]["evidence_id"] == evidence["evidence_id"]

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["validation_error"] == "unsupported_format"
    assert detail["file_type"] == ".json"


def test_validation_failure_preserves_existing_findings(client, paths):
    case = create_case(client, "validation-preserve-findings")
    valid = upload_csv(
        client,
        case["id"],
        "valid.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,HTTP,/private,403,100\n",
    )
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200
    before = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    assert len(before) == 1

    invalid = upload_csv(
        client,
        case["id"],
        "invalid.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "not-a-timestamp,10.0.0.1,GET,/ok,200,not-a-number\n",
    )
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 422

    after = client.get(f"/cases/{case['id']}/findings").json()["findings"]
    assert after == before

    original, _, recorded_hash = get_isolated_stored_paths(paths, valid["evidence_id"])
    assert original.is_file()
    assert original.read_bytes() != b"modified evidence"
    original.write_bytes(b"modified evidence")

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 409
    assert response.json()["detail"]["integrity_error"] == "original_hash_mismatch"
    assert response.json()["detail"]["evidence_id"] == valid["evidence_id"]
    assert response.json()["detail"]["recorded_sha256"] == recorded_hash

    assert client.get(f"/cases/{case['id']}/findings").json()["findings"] == after


def test_missing_values_rejected(client):
    case = create_case(client, "validation-missing-values")
    upload_csv(
        client,
        case["id"],
        "missing-values.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,,GET,/ok,200,100\n",
    )
    missing_values = client.post(f"/cases/{case['id']}/analyze")
    assert missing_values.status_code == 422
    assert missing_values.json()["detail"]["validation_error"] == "missing_required_values"


def test_audit_chain_covers_case_evidence_custody_and_integrity_failure(client, paths):
    case = create_case(client, "audit-complete")
    evidence = upload_csv(
        client,
        case["id"],
        "audit.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100\n",
    )

    custody = client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={"person": "Examiner", "action": "Reviewed", "remarks": "Audit verification"},
    )
    assert custody.status_code == 200

    analysis = client.post(f"/cases/{case['id']}/analyze")
    assert analysis.status_code == 200

    original, _, _ = get_isolated_stored_paths(paths, evidence["evidence_id"])
    original.write_bytes(b"tampered")
    integrity = client.post(f"/cases/{case['id']}/analyze")
    assert integrity.status_code == 409

    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT action_type, outcome FROM audit_events ORDER BY id ASC"
    ).fetchall()
    connection.close()

    action_types = {row["action_type"] for row in rows}
    assert "CASE_CREATED" in action_types
    assert "EVIDENCE_INGESTED" in action_types
    assert "CHAIN_OF_CUSTODY_UPDATED" in action_types
    assert "ANALYSIS_COMPLETED" in action_types
    assert "EVIDENCE_INTEGRITY_FAILED" in action_types

    history = client.get("/audit/verify")
    assert history.status_code == 200
    assert history.json()["valid"] is True


def test_audit_chain_detects_modification_delete_and_reorder(client, paths):
    case = create_case(client, "audit-tamper")
    upload_csv(
        client,
        case["id"],
        "audit-tamper.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100\n",
    )

    connection = sqlite3.connect(paths["db"])
    connection.execute("DROP TRIGGER IF EXISTS prevent_audit_update")
    connection.execute("DROP TRIGGER IF EXISTS prevent_audit_delete")
    connection.execute("UPDATE audit_events SET outcome = 'MODIFIED' WHERE id = 1")
    connection.commit()

    from backend.audit import verify_audit_chain
    result = verify_audit_chain(connection)
    connection.close()
    assert result["valid"] is False

    protected = sqlite3.connect(paths["db"])
    protected.execute(
        "CREATE TRIGGER prevent_audit_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'read-only'); END"
    )
    protected.execute(
        "CREATE TRIGGER prevent_audit_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'read-only'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        protected.execute("DELETE FROM audit_events")
        protected.commit()
    protected.close()


def test_audit_append_is_concurrent_and_transaction_consistent(paths):
    from backend.audit import append_audit_event, verify_audit_chain

    def worker(index):
        connection = sqlite3.connect(paths["db"])
        connection.row_factory = sqlite3.Row
        append_audit_event(
            connection,
            action_type="CONCURRENT_TEST",
            case_id=index,
            evidence_id=None,
            outcome="SUCCESS",
            metadata={"index": index},
        )
        connection.close()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    result = verify_audit_chain(connection)
    count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    connection.close()

    assert result["valid"] is True
    assert count == 20


def test_audit_history_case_and_evidence_filters(client):
    case = create_case(client, "audit-filters")
    evidence = upload_csv(
        client,
        case["id"],
        "audit-filter.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100\n",
    )
    custody = client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={"person": "Filter Examiner", "action": "Reviewed", "remarks": "Filter test"},
    )
    assert custody.status_code == 200

    case_history = client.get("/audit", params={"case_id": case["id"]})
    assert case_history.status_code == 200
    case_records = case_history.json()["history"]
    assert case_records
    assert all(record["case_id"] == case["id"] for record in case_records)

    evidence_history = client.get("/audit", params={"evidence_id": evidence["evidence_id"]})
    assert evidence_history.status_code == 200
    evidence_records = evidence_history.json()["history"]
    assert evidence_records
    assert all(record["evidence_id"] == evidence["evidence_id"] for record in evidence_records)


def test_evidence_validation_failure_audit_event(client, paths):
    case = create_case(client, "audit-validation")
    evidence = upload_csv(
        client,
        case["id"],
        "audit-validation.txt",
        "timestamp,source_ip,event_type,resource,status,bytes\n",
    )

    response = client.post(f"/cases/{case['id']}/analyze")
    assert response.status_code == 422

    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT action_type, outcome, case_id, evidence_id FROM audit_events "
        "WHERE action_type = 'EVIDENCE_VALIDATION_FAILED' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    connection.close()

    assert row is not None
    assert row["outcome"] == "FAILURE"
    assert row["case_id"] == case["id"]
    assert row["evidence_id"] == evidence["evidence_id"]


def test_successful_and_failed_report_audit_events(client, paths):
    case = create_case(client, "audit-report")
    upload_csv(
        client,
        case["id"],
        "audit-report.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100\n",
    )

    success = client.post(f"/cases/{case['id']}/report")
    assert success.status_code == 200

    failed = client.post("/cases/999999/report")
    assert failed.status_code == 500

    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    actions = {
        row["action_type"]: row["outcome"]
        for row in connection.execute(
            "SELECT action_type, outcome FROM audit_events "
            "WHERE action_type IN ('REPORT_GENERATED', 'REPORT_GENERATION_FAILED') "
            "ORDER BY id ASC"
        ).fetchall()
    }
    connection.close()

    assert actions["REPORT_GENERATED"] == "SUCCESS"
    assert actions["REPORT_GENERATION_FAILED"] == "FAILURE"


def test_audit_chain_detects_internal_deletion(client, paths):
    case = create_case(client, "audit-delete")
    upload_csv(
        client,
        case["id"],
        "audit-delete.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100\n",
    )

    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    rows = connection.execute("SELECT id FROM audit_events ORDER BY id ASC").fetchall()
    assert len(rows) >= 2

    connection.execute("DROP TRIGGER IF EXISTS prevent_audit_delete")
    connection.execute("DELETE FROM audit_events WHERE id = ?", (rows[0]["id"],))
    connection.commit()

    from backend.audit import verify_audit_chain
    result = verify_audit_chain(connection)
    connection.close()

    assert result["valid"] is False


def test_audit_chain_detects_reordering(client, paths):
    case = create_case(client, "audit-reorder")
    evidence = upload_csv(
        client,
        case["id"],
        "audit-reorder.csv",
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-01-01 00:00:00,10.0.0.1,GET,/ok,200,100\n",
    )
    assert evidence["evidence_id"]

    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT id, event_id FROM audit_events ORDER BY id ASC"
    ).fetchall()
    assert len(rows) >= 2

    first_id = rows[0]["id"]
    first_event = rows[0]["event_id"]
    second_id = rows[1]["id"]
    second_event = rows[1]["event_id"]

    connection.execute("DROP TRIGGER IF EXISTS prevent_audit_update")
    temporary_event = "AUD-TEMPORARY-REORDER"
    connection.execute("UPDATE audit_events SET event_id = ? WHERE id = ?", (temporary_event, first_id))
    connection.execute("UPDATE audit_events SET event_id = ? WHERE id = ?", (first_event, second_id))
    connection.execute("UPDATE audit_events SET event_id = ? WHERE id = ?", (second_event, first_id))
    connection.commit()

    from backend.audit import verify_audit_chain
    result = verify_audit_chain(connection)
    connection.close()

    assert result["valid"] is False


def test_caller_transaction_rollback_removes_successful_audit_entry(paths):
    from backend.audit import append_audit_event, verify_audit_chain

    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")
    append_audit_event(
        connection,
        action_type="ROLLBACK_TEST",
        case_id=123,
        evidence_id="EV-ROLLBACK",
        outcome="SUCCESS",
        metadata={"test": True},
    )
    connection.rollback()

    count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    result = verify_audit_chain(connection)
    connection.close()

    assert count == 0
    assert result["valid"] is True


def test_mixed_concurrent_owned_and_caller_transactions(paths):
    from backend.audit import append_audit_event, verify_audit_chain

    def owned_worker(index):
        connection = sqlite3.connect(paths["db"], timeout=10)
        append_audit_event(
            connection,
            action_type="MIXED_OWNED",
            case_id=index,
            evidence_id=None,
            outcome="SUCCESS",
            metadata={"index": index},
        )
        connection.close()

    def caller_worker(index):
        connection = sqlite3.connect(paths["db"], timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        for _ in range(20):
            try:
                connection.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
        else:
            connection.close()
            return

        append_audit_event(
            connection,
            action_type="MIXED_CALLER",
            case_id=index,
            evidence_id=None,
            outcome="SUCCESS",
            metadata={"index": index},
        )
        if index % 2:
            connection.rollback()
        else:
            connection.commit()
        connection.close()

    threads = []
    for index in range(10):
        threads.append(threading.Thread(target=owned_worker, args=(index,)))
        threads.append(threading.Thread(target=caller_worker, args=(index,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    connection = sqlite3.connect(paths["db"])
    connection.row_factory = sqlite3.Row
    result = verify_audit_chain(connection)
    count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    connection.close()

    assert result["valid"] is True
    assert count == 15


def test_audit_schema_migration_for_preexisting_database():
    from backend.audit import initialize_audit_table, append_audit_event, verify_audit_chain

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            action_type TEXT NOT NULL,
            outcome TEXT NOT NULL,
            metadata TEXT NOT NULL,
            previous_hash TEXT NOT NULL,
            current_hash TEXT NOT NULL UNIQUE
        )
        """
    )

    initialize_audit_table(connection)
    append_audit_event(
        connection,
        action_type="SCHEMA_COMPATIBILITY",
        case_id=7,
        evidence_id="EV-SCHEMA",
        outcome="SUCCESS",
        metadata={"migrated": True},
    )
    result = verify_audit_chain(connection)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(audit_events)").fetchall()}
    connection.close()

    assert {"case_id", "evidence_id"}.issubset(columns)
    assert result["valid"] is True


def test_findings_schema_migration_preserves_historical_rows(monkeypatch, tmp_path):
    from backend.database import initialize_database

    database_path = tmp_path / "legacy-findings.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            finding_id TEXT NOT NULL UNIQUE,
            evidence_id INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT,
            event_type TEXT NOT NULL,
            resource TEXT,
            reason TEXT NOT NULL,
            severity TEXT NOT NULL,
            technical_observation TEXT NOT NULL,
            plain_language_explanation TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO findings (
            finding_id, evidence_id, event_id, timestamp, source, event_type,
            resource, reason, severity, technical_observation,
            plain_language_explanation, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "FD-HISTORICAL", 1, "0", "2026-01-01T00:00:00Z", "10.0.0.1",
            "HTTP", "/private", "HTTP 403 Forbidden status detected", "MEDIUM",
            "Historical observation", "Historical explanation", "Requires Review",
            "2026-01-01T00:00:00Z",
        ),
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("FORENSIC_INSIGHT_DB_PATH", str(database_path))
    initialize_database()

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    finding = connection.execute(
        "SELECT finding_id, reason, rule_id, rule_version, rule_fingerprint FROM findings"
    ).fetchone()
    connection.close()

    assert finding["finding_id"] == "FD-HISTORICAL"
    assert finding["reason"] == "HTTP 403 Forbidden status detected"
    assert finding["rule_id"] is None
    assert finding["rule_version"] is None
    assert finding["rule_fingerprint"] is None
