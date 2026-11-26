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
    svgIcon,
} from "../dom.js";
import { navigateTo } from "../router.js";
import { setSelectedCase } from "../store.js";
import { openCreateCaseModal } from "./cases.js";

function renderMlAvailabilityBadge(mlStatus) {
    if (!mlStatus) return badge("Unavailable", "neutral");
    if (mlStatus.ready_for_advisory) {
        return badge(`Ready (${mlStatus.model_id} v${mlStatus.model_version})`, "advisory");
    }
    if (mlStatus.enabled && !mlStatus.operator_approved) {
        return badge("Awaiting Operator Approval", "warning");
    }
    if (!mlStatus.enabled) {
        return badge("Disabled by Server Config", "neutral");
    }
    return badge("Model Artifact Missing", "warning");
}

export async function renderDashboardView(container) {
    clear(container);
    container.appendChild(loadingBlock("Loading Command Center metrics..."));

    try {
        const data = await api.getDashboardSummary();
        clear(container);

        const metrics = data.metrics || {};
        const casesByStatus = metrics.cases_by_status || {};
        const sev = metrics.findings_by_severity || {};
        const audit = data.audit_chain || {};
        const ml = data.ml_status || {};
        const recentCases = Array.isArray(data.recent_cases) ? data.recent_cases : [];
        const recentAudit = Array.isArray(data.recent_audit_events) ? data.recent_audit_events : [];

        // Header
        container.appendChild(
            el("div", { className: "fi-page-header" }, [
                el("div", {}, [
                    el("h1", { className: "fi-page-title", text: "Investigation Command Center" }),
                    el("p", {
                        className: "fi-page-subtitle",
                        text: "Live operational overview of cases, verified evidence, deterministic findings, and cryptographic audit health.",
                    }),
                ]),
                el("div", { className: "fi-actions" }, [
                    el(
                        "button",
                        {
                            type: "button",
                            className: "fi-btn fi-btn-primary",
                            onClick: () =>
                                openCreateCaseModal((created) => {
                                    setSelectedCase(created.id, `${created.case_number} — ${created.title}`);
                                    navigateTo(`/cases/${created.id}`);
                                }),
                        },
                        [svgIcon("plus", 15), "New Case"]
                    ),
                    el("a", { href: "#/cases", className: "fi-btn fi-btn-secondary", text: "All Cases" }),
                    el("a", { href: "#/audit", className: "fi-btn fi-btn-secondary", text: "Verify Audit Chain" }),
                ]),
            ])
        );

        // KPI Grid — 100% live API metrics, zero fabricated statistics
        const activeCount =
            (casesByStatus["Open"] || 0) +
            (casesByStatus["In Analysis"] || 0) +
            (casesByStatus["Under Review"] || 0);

        container.appendChild(
            el("div", { className: "fi-kpi-grid", "aria-label": "Operational summary metrics" }, [
                el("div", { className: "fi-kpi-card" }, [
                    el("div", { className: "fi-kpi-label", text: "Total Cases" }),
                    el("div", { className: "fi-kpi-value", id: "kpiTotalCases", text: String(metrics.total_cases ?? 0) }),
                    el("div", {
                        className: "fi-kpi-meta",
                        text: `${activeCount} active • ${casesByStatus["Closed"] || 0} closed`,
                    }),
                ]),
                el("div", { className: "fi-kpi-card" }, [
                    el("div", { className: "fi-kpi-label", text: "Verified Evidence Items" }),
                    el("div", { className: "fi-kpi-value", id: "kpiTotalEvidence", text: String(metrics.total_evidence ?? 0) }),
                    el("div", {
                        className: "fi-kpi-meta",
                        text: `${metrics.total_reports ?? 0} forensic PDF report(s) stored`,
                    }),
                ]),
                el("div", { className: "fi-kpi-card" }, [
                    el("div", { className: "fi-kpi-label", text: "Deterministic Findings" }),
                    el("div", { className: "fi-kpi-value", id: "kpiTotalFindings", text: String(metrics.total_findings ?? 0) }),
                    el("div", {
                        className: "fi-kpi-meta",
                        text: `${sev.HIGH || 0} High • ${sev.MEDIUM || 0} Medium • ${sev.LOW || 0} Low`,
                    }),
                ]),
                el("div", { className: "fi-kpi-card" }, [
                    el("div", { className: "fi-kpi-label", text: "Audit Hash Chain" }),
                    el("div", { style: "margin: 6px 0 8px;" }, [
                        audit.valid
                            ? badge("CRYPTOGRAPHICALLY VALID", "success")
                            : badge(`INTEGRITY ISSUE (${audit.issues_count || 0})`, "danger"),
                    ]),
                    el("div", {
                        className: "fi-kpi-meta",
                        text: `${audit.event_count ?? 0} immutable audit record(s)`,
                    }),
                ]),
                el("div", { className: "fi-kpi-card" }, [
                    el("div", { className: "fi-kpi-label", text: "ML Advisory Status" }),
                    el("div", { style: "margin: 6px 0 8px;" }, [renderMlAvailabilityBadge(ml)]),
                    el("div", {
                        className: "fi-kpi-meta",
                        text: "Read-only advisory • Never loaded on startup",
                    }),
                ]),
            ])
        );

        // Two-column main area: Recent Investigations + Activity & Severity
        const leftPanel = el("section", { className: "fi-panel", "aria-label": "Recent investigations" }, [
            el("div", { className: "fi-panel-header" }, [
                el("h2", { className: "fi-panel-title", text: "Recent Investigations" }),
                el("a", { href: "#/cases", className: "fi-btn fi-btn-ghost fi-btn-sm", text: "View All" }),
            ]),
        ]);

        if (recentCases.length === 0) {
            leftPanel.appendChild(
                emptyState(
                    "No investigations created yet",
                    "Create a case to register evidence, verify SHA-256 hashes, and run deterministic R-A through R-D analysis.",
                    el("button", {
                        type: "button",
                        className: "fi-btn fi-btn-primary",
                        text: "Create First Case",
                        onClick: () =>
                            openCreateCaseModal((created) => {
                                setSelectedCase(created.id, `${created.case_number} — ${created.title}`);
                                navigateTo(`/cases/${created.id}`);
                            }),
                    })
                )
            );
        } else {
            const tbody = el("tbody");
            recentCases.forEach((c) => {
                tbody.appendChild(
                    el("tr", {}, [
                        el("td", {}, [
                            el("div", { className: "fi-mono", style: "color: var(--fi-text-primary); font-weight: 600;", text: c.case_number }),
                            el("div", { style: "font-size: 12px; color: var(--fi-text-muted);", text: c.title }),
                        ]),
                        el("td", { text: c.investigator }),
                        el("td", {}, [statusBadge(c.status || "Open")]),
                        el("td", { text: String(c.evidence_count ?? 0) }),
                        el("td", {}, [
                            el("span", { text: String(c.findings_count ?? 0) }),
                            c.high_severity_count > 0
                                ? el("span", { style: "margin-left: 6px;" }, [badge(`${c.high_severity_count} HIGH`, "danger")])
                                : null,
                        ]),
                        el("td", {}, [
                            el("button", {
                                type: "button",
                                className: "fi-btn fi-btn-secondary fi-btn-sm",
                                text: "Open",
                                onClick: () => {
                                    setSelectedCase(c.id, `${c.case_number} — ${c.title}`);
                                    navigateTo(`/cases/${c.id}`);
                                },
                            }),
                        ]),
                    ])
                );
            });
            leftPanel.appendChild(
                el("div", { className: "fi-table-wrap" }, [
                    el("table", { className: "fi-table" }, [
                        el("thead", {}, [
                            el("tr", {}, [
                                el("th", { text: "Case" }),
                                el("th", { text: "Investigator" }),
                                el("th", { text: "Status" }),
                                el("th", { text: "Evidence" }),
                                el("th", { text: "Findings" }),
                                el("th", { text: "Action" }),
                            ]),
                        ]),
                        tbody,
                    ]),
                ])
            );
        }

        // Right column: Rule Breakdown + Recent Audit Activity
        const rightCol = el("div");
        const ruleCounts = metrics.findings_by_rule || {};
        const rulePanel = el("section", { className: "fi-panel", "aria-label": "Deterministic rule triggers" }, [
            el("div", { className: "fi-panel-header" }, [
                el("h2", { className: "fi-panel-title", text: "Rule Distribution (R-A–R-D)" }),
                el("a", { href: "#/rules", className: "fi-btn fi-btn-ghost fi-btn-sm", text: "Rule Registry" }),
            ]),
        ]);
        const ruleList = el("div", { style: "display: grid; gap: 10px;" });
        ["R-A", "R-B", "R-C", "R-D"].forEach((rid) => {
            ruleList.appendChild(
                el(
                    "div",
                    {
                        style: "display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; background: var(--fi-bg-surface-elevated); border-radius: 6px;",
                    },
                    [
                        el("span", { className: "fi-mono", text: rid }),
                        badge(`${ruleCounts[rid] || 0} finding(s)`, ruleCounts[rid] ? "info" : "neutral"),
                    ]
                )
            );
        });
        rulePanel.appendChild(ruleList);
        rightCol.appendChild(rulePanel);

        const auditPanel = el("section", { className: "fi-panel", "aria-label": "Recent audit events" }, [
            el("div", { className: "fi-panel-header" }, [
                el("h2", { className: "fi-panel-title", text: "Recent Audit Activity" }),
                el("a", { href: "#/audit", className: "fi-btn fi-btn-ghost fi-btn-sm", text: "Full Trail" }),
            ]),
        ]);
        if (recentAudit.length === 0) {
            auditPanel.appendChild(el("p", { style: "font-size: 13px; color: var(--fi-text-muted);", text: "No audit events recorded yet." }));
        } else {
            const list = el("div", { style: "display: grid; gap: 8px;" });
            recentAudit.slice(0, 6).forEach((ev) => {
                list.appendChild(
                    el(
                        "div",
                        {
                            style: "padding: 8px 10px; border: 1px solid var(--fi-border-subtle); border-radius: 6px; font-size: 12px;",
                        },
                        [
                            el("div", { style: "display: flex; justify-content: space-between; gap: 8px; margin-bottom: 4px;" }, [
                                el("strong", { style: "color: var(--fi-text-primary);", text: ev.action_type }),
                                statusBadge(ev.outcome),
                            ]),
                            el("div", { className: "fi-mono", style: "color: var(--fi-text-muted);", text: formatUtcShort(ev.timestamp) }),
                        ]
                    )
                );
            });
            auditPanel.appendChild(list);
        }
        rightCol.appendChild(auditPanel);

        container.appendChild(el("div", { className: "fi-grid-2" }, [leftPanel, rightCol]));

        if (data.security_notice) {
            container.appendChild(
                el("div", { className: "fi-banner", text: data.security_notice })
            );
        }
    } catch (err) {
        clear(container);
        container.appendChild(errorBanner(err.message || "Failed to load Command Center metrics."));
    }
}
