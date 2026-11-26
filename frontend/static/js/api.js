/**
 * Centralized FastAPI client with structured forensic error formatting.
 */

export class ApiError extends Error {
    constructor(status, message, detail = null) {
        super(message);
        this.name = "ApiError";
        this.status = status;
        this.detail = detail;
    }
}

export function formatApiErrorMessage(status, payload) {
    const detail = payload && typeof payload === "object" ? payload.detail : null;
    if (detail && typeof detail === "object") {
        if (detail.integrity_error) {
            const ev = detail.evidence_id ? ` (${detail.evidence_id})` : "";
            return `Evidence integrity verification failed${ev}: ${detail.message || detail.integrity_error}`;
        }
        if (detail.ml_advisory_error) {
            return detail.message || `ML advisory unavailable (${detail.ml_advisory_error}).`;
        }
        if (detail.validation_error) {
            const row = detail.row_number ? ` at row ${detail.row_number}` : "";
            return `Validation error${row}: ${detail.message || detail.validation_error}`;
        }
        if (typeof detail.message === "string") {
            return detail.message;
        }
    }
    if (typeof detail === "string" && detail.trim()) {
        return detail;
    }
    if (Array.isArray(detail) && detail.length > 0 && detail[0].msg) {
        return `Invalid input: ${detail[0].msg}`;
    }
    return `Request failed with HTTP ${status}.`;
}

export async function apiRequest(path, options = {}) {
    const headers = Object.assign({ Accept: "application/json" }, options.headers || {});
    const fetchOpts = Object.assign({}, options, { headers });
    if (fetchOpts.body && typeof fetchOpts.body === "object" && !(fetchOpts.body instanceof FormData)) {
        headers["Content-Type"] = "application/json";
        fetchOpts.body = JSON.stringify(fetchOpts.body);
    }

    const response = await fetch(path, fetchOpts);
    let payload = null;
    try {
        payload = await response.json();
    } catch (_) {
        payload = null;
    }

    if (!response.ok) {
        throw new ApiError(
            response.status,
            formatApiErrorMessage(response.status, payload),
            payload ? payload.detail : null
        );
    }
    return payload;
}

export const api = {
    getDashboardSummary: () => apiRequest("/dashboard/summary"),
    getCases: () => apiRequest("/cases"),
    createCase: (body) => apiRequest("/cases", { method: "POST", body }),
    getCaseDetail: (caseId) => apiRequest(`/cases/${caseId}`),
    updateCase: (caseId, body) => apiRequest(`/cases/${caseId}`, { method: "PATCH", body }),
    getRules: () => apiRequest("/rules"),
    getAudit: (params = {}) => {
        const qs = new URLSearchParams();
        if (params.case_id) qs.set("case_id", String(params.case_id));
        if (params.evidence_id) qs.set("evidence_id", String(params.evidence_id));
        const suffix = qs.toString() ? `?${qs.toString()}` : "";
        return apiRequest(`/audit${suffix}`);
    },
    verifyAuditChain: () => apiRequest("/audit/verify"),
    getMlStatus: () => apiRequest("/ml/status"),
};
