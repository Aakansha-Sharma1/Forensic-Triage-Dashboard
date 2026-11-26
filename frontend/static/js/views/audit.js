import { api } from "../api.js";
import {
    badge,
    clear,
    el,
    emptyState,
    errorBanner,
    formatUtcShort,
    loadingBlock,
    statusBadge,
} from "../dom.js";

export async function renderAuditView(container) {
    clear(container);
    container.appendChild(loadingBlock("Verifying cryptographic audit chain..."));

    try {
        const [auditData, verifyData] = await Promise.all([
            api.getAudit(),
            api.verifyAuditChain(),
        ]);
        const events = Array.isArray(auditData.events) ? auditData.events.slice().reverse() : [];
        clear(container);

        container.appendChild(
            el("div", { className: "fi-page-header" }, [
                el("div", {}, [
                    el("h1", { className: "fi-page-title", text: "Tamper-Evident Audit Trail" }),
                    el("p", {
                        className: "fi-page-subtitle",
                        text: "Append-only SHA-256 hash-chained operational ledger protected by SQLite immutability triggers.",
                    }),
                ]),
                el("div", { className: "fi-actions" }, [
                    verifyData.valid
                        ? badge(`CHAIN VALID (${verifyData.event_count} events)`, "success")
                        : badge(`CHAIN BROKEN (${(verifyData.issues || []).length} issue(s))`, "danger"),
                    el("button", {
                        type: "button",
                        className: "fi-btn fi-btn-secondary",
                        text: "Re-verify Chain",
                        onClick: () => renderAuditView(container),
                    }),
                ]),
            ])
        );

        if (events.length === 0) {
            container.appendChild(
                emptyState("No audit events recorded", "System actions such as case creation, evidence upload, analysis, and reporting appear here.")
            );
            return;
        }

        const tbody = el("tbody");
        events.slice(0, 100).forEach((ev) => {
            tbody.appendChild(
                el("tr", {}, [
                    el("td", { className: "fi-mono", text: formatUtcShort(ev.timestamp) }),
                    el("td", { style: "color: var(--fi-text-primary); font-weight: 600;", text: ev.action_type }),
                    el("td", {}, [statusBadge(ev.outcome)]),
                    el("td", { className: "fi-mono", text: ev.case_id != null ? `#${ev.case_id}` : "—" }),
                    el("td", { className: "fi-mono", text: ev.evidence_id || "—" }),
                    el("td", { className: "fi-mono", text: String(ev.current_hash || "").slice(0, 16) + "…" }),
                ])
            );
        });

        container.appendChild(
            el("section", { className: "fi-panel" }, [
                el("div", { className: "fi-table-wrap" }, [
                    el("table", { className: "fi-table", "aria-label": "Audit trail events" }, [
                        el("thead", {}, [
                            el("tr", {}, [
                                el("th", { text: "Timestamp (UTC)" }),
                                el("th", { text: "Action" }),
                                el("th", { text: "Outcome" }),
                                el("th", { text: "Case" }),
                                el("th", { text: "Evidence" }),
                                el("th", { text: "Record Hash" }),
                            ]),
                        ]),
                        tbody,
                    ]),
                ]),
            ])
        );
    } catch (err) {
        clear(container);
        container.appendChild(errorBanner(err.message || "Unable to load audit trail."));
    }
}
