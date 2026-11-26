/**
 * Minimal reactive application store.
 * Persists ONLY the numeric selectedCaseId in sessionStorage — never sensitive evidence content.
 */

const STORAGE_KEY = "forensicinsight.selectedCaseId";

const listeners = new Set();
const state = {
    selectedCaseId: loadInitialCaseId(),
    selectedCaseLabel: null,
};

function loadInitialCaseId() {
    try {
        if (typeof sessionStorage === "undefined") return null;
        const raw = sessionStorage.getItem(STORAGE_KEY);
        if (!raw || !/^\d+$/.test(raw)) return null;
        return parseInt(raw, 10);
    } catch (_) {
        return null;
    }
}

export function getSelectedCaseId() {
    return state.selectedCaseId;
}

export function getSelectedCaseLabel() {
    return state.selectedCaseLabel;
}

export function setSelectedCase(caseId, label = null) {
    const normalized = caseId === null || caseId === undefined || caseId === "" ? null : parseInt(String(caseId), 10);
    state.selectedCaseId = Number.isInteger(normalized) && normalized > 0 ? normalized : null;
    state.selectedCaseLabel = state.selectedCaseId ? label : null;
    try {
        if (typeof sessionStorage !== "undefined") {
            if (state.selectedCaseId !== null) {
                sessionStorage.setItem(STORAGE_KEY, String(state.selectedCaseId));
            } else {
                sessionStorage.removeItem(STORAGE_KEY);
            }
        }
    } catch (_) {
        // Ignore storage restrictions
    }
    listeners.forEach((fn) => fn(state));
}

export function subscribeStore(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}
