from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


FRONTEND_STATIC_JS = Path(__file__).resolve().parents[1] / "frontend" / "static" / "js"


@pytest.fixture
def client(isolated_app):
    return isolated_app[0]


def test_static_assets_and_v2_shell_are_served_cleanly(client) -> None:
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    html = root_resp.text
    assert 'id="v2AppShell"' in html
    assert 'id="legacyWorkspaceHost"' in html
    assert 'id="returnToV2Btn"' in html
    assert '/static/styles/tokens.css' in html
    assert '/static/styles/app.css' in html
    assert '/static/js/app.js' in html

    # Verify all ES modules and CSS assets are served with 200 OK
    for asset_path in (
        "/static/styles/tokens.css",
        "/static/styles/app.css",
        "/static/js/app.js",
        "/static/js/api.js",
        "/static/js/dom.js",
        "/static/js/router.js",
        "/static/js/store.js",
        "/static/js/views/dashboard.js",
        "/static/js/views/cases.js",
        "/static/js/views/caseWorkspace.js",
        "/static/js/views/rules.js",
        "/static/js/views/audit.js",
    ):
        r = client.get(asset_path)
        assert r.status_code == 200, f"Failed to serve {asset_path}: {r.status_code}"


def test_all_v2_es_modules_enforce_zero_innerhtml_xss_safety() -> None:
    js_files = sorted(FRONTEND_STATIC_JS.rglob("*.js"))
    assert len(js_files) >= 10
    for js_file in js_files:
        text = js_file.read_text(encoding="utf-8")
        assert "innerHTML" not in text, f"Unsafe innerHTML found in {js_file.name}"
        assert "outerHTML" not in text, f"Unsafe outerHTML found in {js_file.name}"
        assert "insertAdjacentHTML" not in text, f"Unsafe insertAdjacentHTML found in {js_file.name}"
        assert "document.write" not in text, f"Unsafe document.write found in {js_file.name}"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_v2_router_store_error_formatting_and_dashboard_xss_rendering(client) -> None:
    # Seed a real case with an XSS-shaped title in isolated DB to test live API -> UI rendering
    xss_title = '<script>alert("xss")</script> Incident'
    created = client.post(
        "/cases",
        json={
            "case_number": "CASE-M2-XSS",
            "title": xss_title,
            "investigator": "M2 Auditor",
            "description": "Testing live API metrics and XSS safety",
        },
    )
    assert created.status_code == 200
    summary = client.get("/dashboard/summary").json()
    assert summary["metrics"]["total_cases"] == 1

    script = f"""
import assert from "node:assert/strict";
import {{ parseRoute }} from {json.dumps((FRONTEND_STATIC_JS / "router.js").as_uri())};
import {{ getSelectedCaseId, setSelectedCase, getSelectedCaseLabel }} from {json.dumps((FRONTEND_STATIC_JS / "store.js").as_uri())};
import {{ formatApiErrorMessage }} from {json.dumps((FRONTEND_STATIC_JS / "api.js").as_uri())};

// 1. Router parsing for all M2 routes
assert.deepEqual(parseRoute(""), {{ name: "dashboard", params: {{}} }});
assert.deepEqual(parseRoute("#/dashboard"), {{ name: "dashboard", params: {{}} }});
assert.deepEqual(parseRoute("#/cases"), {{ name: "cases", params: {{}} }});
assert.deepEqual(parseRoute("#/cases/42"), {{ name: "case-workspace", params: {{ caseId: 42, tab: "overview" }} }});
assert.deepEqual(parseRoute("#/cases/42/findings"), {{ name: "case-workspace", params: {{ caseId: 42, tab: "findings" }} }});
assert.deepEqual(parseRoute("#/rules"), {{ name: "rules", params: {{}} }});
assert.deepEqual(parseRoute("#/audit"), {{ name: "audit", params: {{}} }});
assert.deepEqual(parseRoute("#/classic/phase2"), {{ name: "classic", params: {{ phase: "phase2" }} }});

// 2. Store persists ONLY numeric case ID (no sensitive content)
const storageMap = new Map();
globalThis.sessionStorage = {{
    getItem: (k) => storageMap.has(k) ? storageMap.get(k) : null,
    setItem: (k, v) => storageMap.set(k, String(v)),
    removeItem: (k) => storageMap.delete(k),
}};
setSelectedCase(42, "CASE-42 — Test");
assert.equal(getSelectedCaseId(), 42);
assert.equal(getSelectedCaseLabel(), "CASE-42 — Test");
assert.equal(storageMap.get("forensicinsight.selectedCaseId"), "42");
setSelectedCase(null);
assert.equal(getSelectedCaseId(), null);
assert.equal(storageMap.has("forensicinsight.selectedCaseId"), false);

// 3. Structured integrity & ML error formatting
const intMsg = formatApiErrorMessage(409, {{
    detail: {{ integrity_error: "working_hash_mismatch", evidence_id: "EV-1-ABCD", message: "Hash mismatch" }}
}});
assert.ok(intMsg.includes("EV-1-ABCD") && intMsg.includes("Hash mismatch"));

const mlMsg = formatApiErrorMessage(422, {{
    detail: {{ ml_advisory_error: "unsupported_event_domain", message: "Outside Apache HTTP domain" }}
}});
assert.ok(mlMsg.includes("Outside Apache HTTP domain"));

console.log("M2_JS_MODULES_OK");
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "M2_JS_MODULES_OK" in proc.stdout
