/**
 * Hash-based client router for ForensicInsight 2.0.
 * Routes:
 *   #/dashboard
 *   #/cases
 *   #/cases/:id
 *   #/cases/:id/:tab
 *   #/rules
 *   #/audit
 *   #/classic/:phase
 */

export function parseRoute(hashValue) {
    const raw = String(hashValue || "").replace(/^#/, "").trim();
    const path = raw.startsWith("/") ? raw : raw ? `/${raw}` : "/dashboard";
    const segments = path.split("/").filter(Boolean);

    if (segments.length === 0 || segments[0] === "dashboard") {
        return { name: "dashboard", params: {} };
    }
    if (segments[0] === "cases") {
        if (segments.length === 1) {
            return { name: "cases", params: {} };
        }
        const caseId = /^\d+$/.test(segments[1]) ? parseInt(segments[1], 10) : null;
        if (caseId !== null) {
            return {
                name: "case-workspace",
                params: {
                    caseId,
                    tab: segments[2] || "overview",
                },
            };
        }
        return { name: "cases", params: {} };
    }
    if (segments[0] === "rules") {
        return { name: "rules", params: {} };
    }
    if (segments[0] === "audit") {
        return { name: "audit", params: {} };
    }
    if (segments[0] === "classic") {
        const phase = segments[1] || "phase1";
        return { name: "classic", params: { phase } };
    }
    return { name: "dashboard", params: {} };
}

export function navigateTo(hashPath) {
    const target = hashPath.startsWith("#") ? hashPath : `#${hashPath}`;
    if (window.location.hash === target) {
        window.dispatchEvent(new HashChangeEvent("hashchange"));
    } else {
        window.location.hash = target;
    }
}
