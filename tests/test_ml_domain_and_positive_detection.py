from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import subprocess
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


FRONTEND_INDEX = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
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


def _enable_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_ENABLED", "true")
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_OPERATOR_APPROVED", "true")
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MODELS_DIR", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MODEL_ID", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MODEL_VERSION", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MODEL_SHA256", raising=False)
    monkeypatch.delenv("FORENSIC_INSIGHT_ML_MAX_EVENTS", raising=False)


def _create_case_and_upload(client, suffix: str, csv_text: str) -> tuple[dict, dict]:
    case = client.post(
        "/cases",
        json={
            "case_number": f"CASE-DOM-{suffix}",
            "title": f"Domain Validation {suffix}",
            "investigator": "Forensic Examiner",
        },
    ).json()["case"]
    evidence = client.post(
        f"/cases/{case['id']}/evidence",
        params={"source_description": "Synthetic validation CSV"},
        files={"file": (f"{suffix}.csv", csv_text.encode("utf-8"), "text/csv")},
    ).json()["evidence"]
    return case, evidence


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


def _build_synthetic_positive_apache_csv() -> str:
    """Disposable synthetic Apache HTTP CSV with normal requests and a high-rate directory scan."""
    lines = ["timestamp,source_ip,event_type,resource,status,bytes"]
    # 15 normal spaced-out GET requests
    for idx in range(15):
        lines.append(
            f"2026-02-01T12:00:{idx:02d}Z,10.0.0.10,GET,/page/{idx},200,512"
        )
    # 60 rapid 404 directory-scan probes (matches AIT-LDS dirb-like high-frequency failure pattern)
    for idx in range(60):
        lines.append(
            f"2026-02-01T12:01:{idx // 10:02d}.{idx % 10}00000Z,192.168.50.99,GET,/admin/secret_{idx},404,280"
        )
    # 3 rapid POST 403 requests to sensitive paths (also triggers deterministic R-A, R-B, R-C)
    lines.append("2026-02-01T12:02:00Z,192.168.50.99,POST,/admin/password,403,240")
    lines.append("2026-02-01T12:02:01Z,192.168.50.99,POST,/admin/secret,403,245")
    lines.append("2026-02-01T12:02:02Z,192.168.50.99,POST,/admin/confidential,403,250")
    return "\n".join(lines) + "\n"


@pytest.mark.skipif(
    not DEPLOYED_MODEL_FILE.is_file(),
    reason="Deployed RandomForest artifact not present on disk.",
)
def test_unsupported_mixed_and_ambiguous_domains_fail_closed_without_affecting_rules(
    client, monkeypatch
) -> None:
    _enable_real_model(monkeypatch)

    # 1. Clearly unsupported host/OS/USB/firewall telemetry (observed real-world failure case).
    unsupported_csv = (
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-02-01T10:00:00Z,10.0.0.5,PROCESS_START,C:\\Windows\\System32\\cmd.exe,0,0\n"
        "2026-02-01T10:00:01Z,10.0.0.5,FILE_READ,C:\\Users\\Alice\\secret.docx,OK,4096\n"
        "2026-02-01T10:00:02Z,10.0.0.5,FILE_WRITE,C:\\Temp\\archive.zip,OK,8192\n"
        "2026-02-01T10:00:03Z,10.0.0.5,USB_CONNECTED,USB\\VID_0781,CONNECTED,0\n"
        "2026-02-01T10:00:04Z,10.0.0.5,NETWORK_BLOCKED,198.51.100.2:443,BLOCKED,64\n"
        "2026-02-01T10:00:05Z,10.0.0.5,NETWORK_ALLOWED,198.51.100.3:443,ALLOWED,512\n"
    )
    case_unsup, _ = _create_case_and_upload(client, "unsup", unsupported_csv)
    resp_unsup = client.post(f"/cases/{case_unsup['id']}/ml-advisory")
    assert resp_unsup.status_code == 422
    detail_unsup = resp_unsup.json()["detail"]
    assert detail_unsup["ml_advisory_error"] == "unsupported_event_domain"
    assert detail_unsup["deterministic_analysis_available"] is True
    assert detail_unsup["supported_http_events"] == 0
    assert detail_unsup["unsupported_domain_events"] == 6
    assert "USB_CONNECTED" in detail_unsup["observed_unsupported_event_types"]
    # Deterministic rule analysis still works on the same case.
    assert client.post(f"/cases/{case_unsup['id']}/analyze").status_code == 200

    # 2. Mixed-domain evidence (legitimate Apache HTTP + host/USB telemetry).
    mixed_csv = (
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-02-01T10:00:00Z,10.0.0.5,GET,/index.html,200,512\n"
        "2026-02-01T10:00:01Z,10.0.0.5,POST,/admin/secret,403,256\n"
        "2026-02-01T10:00:02Z,10.0.0.5,USB_CONNECTED,USB\\VID_0781,0,0\n"
        "2026-02-01T10:00:03Z,10.0.0.5,PROCESS_START,C:\\Windows\\notepad.exe,0,0\n"
    )
    case_mixed, _ = _create_case_and_upload(client, "mixed", mixed_csv)
    resp_mixed = client.post(f"/cases/{case_mixed['id']}/ml-advisory")
    assert resp_mixed.status_code == 422
    detail_mixed = resp_mixed.json()["detail"]
    assert detail_mixed["ml_advisory_error"] == "mixed_event_domain"
    assert detail_mixed["supported_http_events"] == 2
    assert detail_mixed["unsupported_domain_events"] == 2
    assert detail_mixed["deterministic_analysis_available"] is True
    # Official R-A / R-B findings are still generated normally.
    analyze_mixed = client.post(f"/cases/{case_mixed['id']}/analyze")
    assert analyze_mixed.status_code == 200
    assert analyze_mixed.json()["findings_count"] >= 2

    # 3. Ambiguous evidence (non-IP source, non-URI resource, non-3-digit HTTP status, non-standard event_type).
    ambiguous_csv = (
        "timestamp,source_ip,event_type,resource,status,bytes\n"
        "2026-02-01T10:00:00Z,WORKSTATION-01,HTTP GET,payroll.xlsx,OK,1024\n"
        "2026-02-01T10:00:05Z,WORKSTATION-01,CUSTOM_ACCESS,report.pdf,SUCCESS,2048\n"
    )
    case_ambig, _ = _create_case_and_upload(client, "ambig", ambiguous_csv)
    resp_ambig = client.post(f"/cases/{case_ambig['id']}/ml-advisory")
    assert resp_ambig.status_code == 422
    detail_ambig = resp_ambig.json()["detail"]
    assert detail_ambig["ml_advisory_error"] == "ambiguous_event_domain"
    assert detail_ambig["ambiguous_events"] == 2
    assert set(detail_ambig["incompatibility_reasons"]) == {
        "non_http_event_type",
        "non_http_status_code",
        "non_http_resource_path",
        "non_ip_source_identifier",
    }


@pytest.mark.skipif(
    not DEPLOYED_MODEL_FILE.is_file(),
    reason="Deployed RandomForest artifact not present on disk.",
)
def test_positive_detection_end_to_end_with_real_model_frontend_and_non_mutation(
    client, paths, monkeypatch
) -> None:
    _enable_real_model(monkeypatch)

    csv_text = _build_synthetic_positive_apache_csv()
    expected_sha256 = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    case, evidence = _create_case_and_upload(client, "pos-rf", csv_text)

    # Record custody and run deterministic R-A through R-D analysis first.
    assert client.post(
        f"/evidence/{evidence['id']}/chain-of-custody",
        json={"person": "Examiner", "action": "Verified", "remarks": "Intake verified"},
    ).status_code == 200
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200

    orig_file = paths["upload"] / f"{evidence['evidence_id']}.csv"
    work_file = paths["working"] / f"{evidence['evidence_id']}_working.csv"
    db_before = _snapshot_all_tables(paths["db"])
    findings_before = client.get(f"/cases/{case['id']}/findings").json()

    # Run real pinned Random Forest v1.0.0 advisory inference (no retraining, threshold=0.5).
    resp = client.post(f"/cases/{case['id']}/ml-advisory")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["model"]["model_id"] == DEFAULT_DEPLOYMENT_MODEL_ID
    assert body["model"]["model_version"] == DEFAULT_DEPLOYMENT_MODEL_VERSION
    assert body["model"]["model_sha256"] == DEFAULT_DEPLOYED_MODEL_SHA256
    assert body["model"]["decision_threshold"] == 0.5
    assert body["summary"]["total_events_scored"] == 78
    # Verify real positive predictions (predicted_label == 1) and normal predictions both occur.
    assert body["summary"]["attack_associated_count"] >= 1
    assert body["summary"]["normal_count"] >= 1
    assert (
        body["summary"]["attack_associated_count"] + body["summary"]["normal_count"]
        == 78
    )
    attack_preds = [p for p in body["predictions"] if p["predicted_label"] == 1]
    assert len(attack_preds) == body["summary"]["attack_associated_count"]
    assert all(p["model_score"] >= 0.5 for p in attack_preds)
    assert all(p["is_calibrated_probability"] is False for p in body["predictions"])

    # Verify server-side only_attack_associated=true matches.
    resp_filtered = client.post(
        f"/cases/{case['id']}/ml-advisory", params={"only_attack_associated": "true"}
    )
    assert resp_filtered.status_code == 200
    filtered_body = resp_filtered.json()
    assert filtered_body["summary"]["total_events_scored"] == 78
    assert len(filtered_body["predictions"]) == body["summary"]["attack_associated_count"]
    assert all(p["predicted_label"] == 1 for p in filtered_body["predictions"])

    # Verify zero mutation of evidence files, DB tables, findings, custody, and audit chain.
    assert hashlib.sha256(orig_file.read_bytes()).hexdigest() == expected_sha256
    assert hashlib.sha256(work_file.read_bytes()).hexdigest() == expected_sha256
    assert _snapshot_all_tables(paths["db"]) == db_before
    assert client.get(f"/cases/{case['id']}/findings").json() == findings_before
    assert client.get("/audit/verify").json()["valid"] is True

    # Disable ML and verify ordinary analysis + PDF report still succeed without ML content.
    monkeypatch.setenv("FORENSIC_INSIGHT_ML_ENABLED", "false")
    assert client.post(f"/cases/{case['id']}/ml-advisory").status_code == 503
    assert client.post(f"/cases/{case['id']}/analyze").status_code == 200
    rep = client.post(f"/cases/{case['id']}/report")
    assert rep.status_code == 200
    pdf_dl = client.get(
        f"/cases/{case['id']}/download", params={"path": rep.json()["report"]["path"]}
    )
    assert pdf_dl.status_code == 200
    pdf_text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(pdf_dl.content)).pages)
    assert DEFAULT_DEPLOYMENT_MODEL_ID not in pdf_text
    assert "ML Advisory" not in pdf_text

    # Exercise the real frontend rendering, positive badges, pagination, attack-only filter,
    # and unsupported-domain error messaging in Node.js using the actual FastAPI response.
    if shutil.which("node") is not None:
        html = FRONTEND_INDEX.read_text(encoding="utf-8")
        start = html.index("const ML_ADVISORY_PAGE_SIZE = 25;")
        end = html.index("async function analyzeCase()", start)
        ml_script = html[start:end]

        harness = (
            r"""
const assert = require("assert");
class MockElement {
    constructor(tag, id = "") {
        this.tagName = tag.toUpperCase(); this.id = id; this.className = "";
        this.style = {}; this.disabled = false; this.checked = false;
        this.children = []; this.attributes = {}; this._textContent = "";
    }
    set textContent(v) { this._textContent = String(v); this.children = []; }
    get textContent() { return this.children.length ? this.children.map(c => c.textContent).join("") : this._textContent; }
    appendChild(c) { if (c && c._isFragment) this.children.push(...c.children); else this.children.push(c); return c; }
    replaceChildren(...n) { this.children = []; this._textContent = ""; n.forEach(x => this.appendChild(x)); }
    setAttribute(k, v) { this.attributes[k] = String(v); }
    removeAttribute(k) { delete this.attributes[k]; }
}
const ids = [
    "runMlAdvisoryBtn","mlAttackOnlyFilter","mlAdvisoryStatus","mlAdvisoryLoading",
    "mlAdvisoryResults","mlMetaModelId","mlMetaModelVersion","mlMetaFeatureSchema",
    "mlMetaThreshold","mlMetaAdvisoryStatus","mlTotalEventsNum","mlAttackCountNum",
    "mlNormalCountNum","mlShownCountNum","mlAdvisoryEmpty","mlAdvisoryEmptyDetail",
    "mlAdvisoryTableContainer","mlAdvisoryTableBody","mlPageInfo","mlPrevPageBtn",
    "mlNextPageBtn","mlLimitationsBox","mlLimitationsList"
];
const els = {}; ids.forEach(id => { els[id] = new MockElement("div", id); });
global.document = {
    getElementById: id => els[id],
    createElement: tag => new MockElement(tag),
    createDocumentFragment: () => ({ _isFragment: true, children: [], appendChild(c) { this.children.push(c); } }),
};
let phase2CaseId = 1;
__ML_SCRIPT__

// Verify unsupported, mixed, and ambiguous domain messages in UI
const unsupUi = classifyMlAdvisoryError(422, { detail: { ml_advisory_error: "unsupported_event_domain" } });
assert.strictEqual(unsupUi.variant, "warning");
assert.ok(unsupUi.message.includes("outside the ML model's validated Apache HTTP domain"));
assert.ok(unsupUi.message.includes("Rule-based forensic analysis remains available"));

const mixedUi = classifyMlAdvisoryError(422, { detail: { ml_advisory_error: "mixed_event_domain" } });
assert.strictEqual(mixedUi.variant, "warning");
assert.ok(mixedUi.message.includes("mixes Apache HTTP events"));

const ambigUi = classifyMlAdvisoryError(422, { detail: { ml_advisory_error: "ambiguous_event_domain" } });
assert.strictEqual(ambigUi.variant, "warning");
assert.ok(ambigUi.message.includes("could not be verified as compatible Apache HTTP"));

// Render the REAL FastAPI positive-detection response (78 events -> 4 pages of 25/25/25/3)
const liveResponse = __LIVE_RESPONSE__;
renderMlAdvisoryResponse(liveResponse);
assert.strictEqual(els.mlTotalEventsNum.textContent, "78");
assert.strictEqual(Number(els.mlAttackCountNum.textContent), liveResponse.summary.attack_associated_count);
assert.strictEqual(els.mlAdvisoryTableBody.children.length, 25);
assert.ok(els.mlPageInfo.textContent.includes("Page 1 of 4"));

// Enable attack-only filter -> strictly positive predictions with ml-pred-attack badges
els.mlAttackOnlyFilter.checked = true;
mlAdvisoryState.currentPage = 1;
renderMlAdvisoryTable();
assert.strictEqual(Number(els.mlShownCountNum.textContent), liveResponse.summary.attack_associated_count);
assert.strictEqual(els.mlTotalEventsNum.textContent, "78");
els.mlAdvisoryTableBody.children.forEach(tr => {
    const badge = tr.children[3].children[0];
    assert.strictEqual(badge.textContent, "attack_associated");
    assert.ok(badge.className.includes("ml-pred-attack"));
});
console.log("POSITIVE_FRONTEND_OK");
"""
            .replace("__ML_SCRIPT__", ml_script)
            .replace("__LIVE_RESPONSE__", json.dumps(body))
        )
        proc = subprocess.run(
            ["node", "-"], input=harness, capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        assert "POSITIVE_FRONTEND_OK" in proc.stdout
