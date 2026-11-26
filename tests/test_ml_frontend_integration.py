from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


FRONTEND_INDEX = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


@pytest.fixture
def client(isolated_app):
    return isolated_app[0]


def _extract_ml_advisory_script(html: str) -> str:
    start = html.index("const ML_ADVISORY_PAGE_SIZE = 25;")
    end = html.index("async function analyzeCase()", start)
    return html[start:end]


def test_dashboard_serves_ml_advisory_section_with_required_labels_and_safeguards(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Existing deterministic R-A through R-D elements remain present and unchanged.
    assert 'id="analyzeBtn"' in html
    assert 'id="analysisResults"' in html
    assert 'id="findingsList"' in html
    assert 'id="noFindingsState"' in html

    # Separate ML Advisory Analysis section and deliberate action button.
    assert 'id="mlAdvisorySection"' in html
    assert 'id="runMlAdvisoryBtn"' in html
    assert "Run ML Advisory" in html
    assert "ADVISORY ONLY" in html
    assert "Uncalibrated model score — not a calibrated probability or confirmed threat" in html

    # Filter, metadata, summary counts, pagination, and accessible live status regions.
    for element_id in (
        "mlAttackOnlyFilter",
        "mlAdvisoryStatus",
        "mlAdvisoryLoading",
        "mlAdvisoryResults",
        "mlMetaModelId",
        "mlMetaModelVersion",
        "mlMetaFeatureSchema",
        "mlMetaThreshold",
        "mlMetaAdvisoryStatus",
        "mlTotalEventsNum",
        "mlAttackCountNum",
        "mlNormalCountNum",
        "mlShownCountNum",
        "mlAdvisoryEmpty",
        "mlAdvisoryTableBody",
        "mlPrevPageBtn",
        "mlNextPageBtn",
        "mlPageInfo",
        "mlLimitationsList",
    ):
        assert f'id="{element_id}"' in html

    # Calls the exact existing advisory endpoint via POST.
    ml_script = _extract_ml_advisory_script(html)
    assert "fetch(`/cases/${phase2CaseId}/ml-advisory`" in ml_script
    assert 'method: "POST"' in ml_script

    # Safe DOM rendering only: no innerHTML, outerHTML, insertAdjacentHTML, or document.write in ML code.
    assert "innerHTML" not in ml_script
    assert "outerHTML" not in ml_script
    assert "insertAdjacentHTML" not in ml_script
    assert "document.write" not in ml_script

    # No arbitrary model-path, env-var, approval, or deserialization controls exposed in UI.
    section_match = re.search(
        r'<section[^>]+id="mlAdvisorySection".*?</section>', html, flags=re.DOTALL
    )
    assert section_match is not None
    section_html = section_match.group(0)
    assert 'type="file"' not in section_html
    assert "FORENSIC_INSIGHT_ML" not in section_html
    assert "operator_approved" not in section_html
    assert "model.joblib" not in section_html
    assert "artifact_dir" not in section_html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available in test environment")
def test_ml_advisory_js_rendering_filtering_pagination_errors_and_xss_safety() -> None:
    html = FRONTEND_INDEX.read_text(encoding="utf-8")
    ml_script = _extract_ml_advisory_script(html)

    harness = r"""
const assert = require("assert");

class MockElement {
    constructor(tagName, id = "") {
        this.tagName = tagName.toUpperCase();
        this.id = id;
        this.className = "";
        this.style = {};
        this.disabled = false;
        this.checked = false;
        this.children = [];
        this.attributes = {};
        this._textContent = "";
    }
    set textContent(val) {
        this._textContent = String(val);
        this.children = [];
    }
    get textContent() {
        if (this.children.length === 0) return this._textContent;
        return this.children.map((c) => c.textContent).join("");
    }
    appendChild(child) {
        if (child && child._isFragment) {
            this.children.push(...child.children);
        } else {
            this.children.push(child);
        }
        return child;
    }
    replaceChildren(...nodes) {
        this.children = [];
        this._textContent = "";
        nodes.forEach((n) => this.appendChild(n));
    }
    setAttribute(k, v) { this.attributes[k] = String(v); }
    removeAttribute(k) { delete this.attributes[k]; }
}

const elementIds = [
    "runMlAdvisoryBtn", "mlAttackOnlyFilter", "mlAdvisoryStatus", "mlAdvisoryLoading",
    "mlAdvisoryResults", "mlMetaModelId", "mlMetaModelVersion", "mlMetaFeatureSchema",
    "mlMetaThreshold", "mlMetaAdvisoryStatus", "mlTotalEventsNum", "mlAttackCountNum",
    "mlNormalCountNum", "mlShownCountNum", "mlAdvisoryEmpty", "mlAdvisoryEmptyDetail",
    "mlAdvisoryTableContainer", "mlAdvisoryTableBody", "mlPageInfo", "mlPrevPageBtn",
    "mlNextPageBtn", "mlLimitationsBox", "mlLimitationsList"
];
const elements = {};
elementIds.forEach((id) => { elements[id] = new MockElement("div", id); });

global.document = {
    getElementById: (id) => {
        if (!elements[id]) throw new Error("Missing mock element: " + id);
        return elements[id];
    },
    createElement: (tag) => new MockElement(tag),
    createDocumentFragment: () => ({ _isFragment: true, children: [], appendChild(c) { this.children.push(c); } }),
};

let phase2CaseId = 7;
let fetchCalls = [];
let nextFetchResponse = null;
global.fetch = async (url, opts) => {
    fetchCalls.push({ url, opts });
    return nextFetchResponse;
};

__ML_SCRIPT__

// 1. Test error classification states
assert.strictEqual(
    classifyMlAdvisoryError(503, { detail: { ml_advisory_error: "ml_not_enabled" } }).variant,
    "info"
);
assert.strictEqual(
    classifyMlAdvisoryError(503, { detail: { ml_advisory_error: "ml_not_approved" } }).variant,
    "info"
);
assert.strictEqual(
    classifyMlAdvisoryError(503, { detail: { ml_advisory_error: "model_unavailable_or_invalid", message: "Missing artifact" } }).variant,
    "warning"
);
assert.strictEqual(
    classifyMlAdvisoryError(503, { detail: { ml_advisory_error: "model_unavailable_or_invalid", message: "Model file SHA-256 mismatch" } }).variant,
    "error"
);
assert.strictEqual(
    classifyMlAdvisoryError(422, { detail: { validation_error: "unsupported_format" } }).variant,
    "warning"
);
assert.strictEqual(
    classifyMlAdvisoryError(413, { detail: { ml_advisory_error: "event_limit_exceeded", message: "Too many events" } }).variant,
    "warning"
);

// 2. Test rendering, pagination (30 rows -> 25 on page 1, 5 on page 2), filtering, and XSS-safe textContent
const xssPayload = '<img src=x onerror="alert(1)">';
const predictions = [];
for (let i = 0; i < 30; i++) {
    const isAttack = i < 4;
    predictions.push({
        model_id: "forensicinsight-rf-ait-lds-v2.1",
        model_version: "1.0.0",
        feature_schema_version: "1.0.0",
        decision_threshold: 0.5,
        predicted_label: isAttack ? 1 : 0,
        predicted_label_name: isAttack ? "attack_associated" : "normal",
        model_score: isAttack ? 0.9123 : 0.0456,
        is_calibrated_probability: false,
        source_event_reference: {
            evidence_id: "EV-7-TEST",
            event_id: String(i),
            timestamp: "2026-01-01T10:00:00+00:00",
            source_ip: "10.0.0.5",
            event_type: "GET",
            resource: i === 0 ? xssPayload : "/page/" + i,
            response_status: "200",
            transfer_size: 512,
        },
    });
}

(async () => {
    nextFetchResponse = {
        ok: true,
        status: 200,
        json: async () => ({
            case_id: 7,
            advisory_only: true,
            deterministic_rules_unchanged: true,
            model: {
                model_id: "forensicinsight-rf-ait-lds-v2.1",
                model_version: "1.0.0",
                feature_schema_version: "1.0.0",
                decision_threshold: 0.5,
            },
            summary: {
                total_events_scored: 30,
                attack_associated_count: 4,
                normal_count: 26,
            },
            predictions,
            limitations: ["Not externally validated.", xssPayload],
        }),
    };

    await runMlAdvisoryAnalysis();

    assert.strictEqual(fetchCalls.length, 1);
    assert.strictEqual(fetchCalls[0].url, "/cases/7/ml-advisory");
    assert.strictEqual(fetchCalls[0].opts.method, "POST");
    assert.strictEqual(elements.mlMetaModelId.textContent, "forensicinsight-rf-ait-lds-v2.1");
    assert.strictEqual(elements.mlTotalEventsNum.textContent, "30");
    assert.strictEqual(elements.mlAttackCountNum.textContent, "4");
    assert.strictEqual(elements.mlNormalCountNum.textContent, "26");
    assert.strictEqual(elements.mlAdvisoryTableBody.children.length, 25);
    assert.strictEqual(elements.mlNextPageBtn.disabled, false);

    // Server-provided XSS payload is stored as literal textContent inside child elements.
    assert.ok(elements.mlAdvisoryTableBody.children[0].textContent.includes(xssPayload));
    assert.strictEqual(elements.mlLimitationsList.children[1]._textContent, xssPayload);

    // Page 2 shows remaining 5 rows.
    mlAdvisoryState.currentPage = 2;
    renderMlAdvisoryTable();
    assert.strictEqual(elements.mlAdvisoryTableBody.children.length, 5);
    assert.strictEqual(elements.mlNextPageBtn.disabled, true);

    // Filtering to attack-associated only shows 4 rows without changing summary counts.
    elements.mlAttackOnlyFilter.checked = true;
    mlAdvisoryState.currentPage = 1;
    renderMlAdvisoryTable();
    assert.strictEqual(elements.mlAdvisoryTableBody.children.length, 4);
    assert.strictEqual(elements.mlShownCountNum.textContent, "4");
    assert.strictEqual(elements.mlTotalEventsNum.textContent, "30");

    // Empty state when no predictions exist.
    renderMlAdvisoryResponse({
        advisory_only: true,
        model: { model_id: "m", model_version: "1.0.0", feature_schema_version: "1.0.0", decision_threshold: 0.5 },
        summary: { total_events_scored: 0, attack_associated_count: 0, normal_count: 0 },
        predictions: [],
        limitations: [],
    });
    assert.strictEqual(elements.mlAdvisoryEmpty.style.display, "block");
    assert.strictEqual(elements.mlAdvisoryTableContainer.style.display, "none");

    console.log("JS_HARNESS_OK");
})().catch((err) => {
    console.error(err);
    process.exit(1);
});
""".replace("__ML_SCRIPT__", ml_script)

    proc = subprocess.run(
        ["node", "-e", harness],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "JS_HARNESS_OK" in proc.stdout
