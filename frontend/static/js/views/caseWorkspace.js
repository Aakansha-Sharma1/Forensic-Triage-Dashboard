import { api } from "../api.js";
import {
    badge,
    clear,
    el,
    errorBanner,
    formatUtcShort,
    loadingBlock,
    statusBadge,
} from "../dom.js";
import { navigateTo } from "../router.js";
import { setSelectedCase } from "../store.js";

const WORKSPACE_TABS = [
    { id: "overview", label: "Overview", milestone: "M2", legacyPhase: null },
    { id: "evidence", label: "Evidence", milestone: "M3", legacyPhase: "phase1" },
    { id: "timeline", label: "Events & Timeline", milestone: "M4", legacyPhase: "phase3" },
    { id: "findings", label: "Findings (R-A–R-D)", milestone: "M5", legacyPhase: "phase2" },
    { id: "ml-advisory", label: "ML Advisory", milestone: "M5", legacyPhase: "phase2" },
    { id: "custody", label: "Chain of Custody", milestone: "M3", legacyPhase: "phase1" },
    { id: "reports", label: "Reports", milestone: "M6", legacyPhase: "phase5" },
];

export async function renderCaseWorkspaceView(container, caseId, activeTab = "overview", openLegacyFn = null) {
    clear(container);
    container.appendChild(loadingBlock(`Loading Case #${caseId}...`));

    try {
        const data = await api.getCaseDetail(caseId);
        const c = data.case;
        const summary = data.summary || {};
        const sev = summary.severity_counts || {};

        setSelectedCase(c.id, `${c.case_number} — ${c.title}`);
        clear(container);

        const statusFeedback = el("div");
        const statusSelect = el(
            "select",
            {
                className: "fi-select",
                "aria-label": "Case lifecycle status",
                onChange: async (e) => {
                    clear(statusFeedback);
                    const nextStatus = e.target.value;
                    statusSelect.disabled = true;
                    try {
                        await api.updateCase(c.id, { status: nextStatus });
                        statusFeedback.appendChild(
                            el("div", {
                                className: "fi-banner fi-banner-success",
                                role: "status",
                                text: `Case status updated to "${nextStatus}" (recorded in hash-chained audit log).`,
                            })
                        );
                    } catch (err) {
                        statusFeedback.appendChild(errorBanner(err.message || "Failed to update status."));
                    } finally {
                        statusSelect.disabled = false;
                    }
                },
            },
            ["Open", "In Analysis", "Under Review", "Closed"].map((st) =>
                el("option", { value: st, text: st, selected: st === (c.status || "Open") })
            )
        );

        // Case Header
        container.appendChild(
            el("div", { className: "fi-page-header" }, [
                el("div", {}, [
                    el("div", { style: "display: flex; align-items: center; gap: 10px; margin-bottom: 6px;" }, [
                        el("span", { className: "fi-mono", style: "color: var(--fi-accent-primary); font-weight: 700;", text: c.case_number }),
                        statusBadge(c.status || "Open"),
                    ]),
                    el("h1", { className: "fi-page-title", text: c.title }),
                    el("p", {
                        className: "fi-page-subtitle",
                        text: `Investigator: ${c.investigator} • Created: ${formatUtcShort(c.created_at)}`,
                    }),
                ]),
                el("div", { className: "fi-actions" }, [statusSelect]),
            ])
        );

        container.appendChild(statusFeedback);

        if (summary.integrity_failed_evidence && summary.integrity_failed_evidence.length > 0) {
            container.appendChild(
                errorBanner(
                    `Integrity alert: SHA-256 verification failed for evidence ${summary.integrity_failed_evidence.join(", ")}. Forensic processing is blocked until resolved.`
                )
            );
        }

        // Workspace Sub-Navigation
        const subnav = el("nav", { className: "fi-subnav", "aria-label": "Case workspace sections" });
        WORKSPACE_TABS.forEach((tab) => {
            subnav.appendChild(
                el("button", {
                    type: "button",
                    className: `fi-tab-btn${tab.id === activeTab ? " active" : ""}`,
                    text: tab.label,
                    onClick: () => navigateTo(`/cases/${c.id}/${tab.id}`),
                })
            );
        });
        container.appendChild(subnav);

        // Tab Body
        const tabSpec = WORKSPACE_TABS.find((t) => t.id === activeTab) || WORKSPACE_TABS[0];

        if (tabSpec.id === "overview") {
            container.appendChild(
                el("div", { className: "fi-kpi-grid" }, [
                    el("div", { className: "fi-kpi-card" }, [
                        el("div", { className: "fi-kpi-label", text: "Verified CSV Events" }),
                        el("div", { className: "fi-kpi-value", text: String(summary.verified_csv_event_count ?? 0) }),
                        el("div", { className: "fi-kpi-meta", text: "Counted from integrity-verified working copies" }),
                    ]),
                    el("div", { className: "fi-kpi-card" }, [
                        el("div", { className: "fi-kpi-label", text: "Evidence Items" }),
                        el("div", { className: "fi-kpi-value", text: String(summary.evidence_count ?? 0) }),
                        el("div", {
                            className: "fi-kpi-meta",
                            text: `${summary.integrity_verified_evidence_count ?? 0} passed SHA-256 verification`,
                        }),
                    ]),
                    el("div", { className: "fi-kpi-card" }, [
                        el("div", { className: "fi-kpi-label", text: "Deterministic Findings" }),
                        el("div", { className: "fi-kpi-value", text: String(summary.findings_count ?? 0) }),
                        el("div", {
                            className: "fi-kpi-meta",
                            text: `${sev.HIGH || 0} High • ${sev.MEDIUM || 0} Medium • ${sev.LOW || 0} Low`,
                        }),
                    ]),
                    el("div", { className: "fi-kpi-card" }, [
                        el("div", { className: "fi-kpi-label", text: "Custody & Reports" }),
                        el("div", { className: "fi-kpi-value", text: String(summary.custody_count ?? 0) }),
                        el("div", {
                            className: "fi-kpi-meta",
                            text: `Custody entries • ${summary.reports_count ?? 0} PDF report(s)`,
                        }),
                    ]),
                ])
            );

            container.appendChild(
                el("section", { className: "fi-panel" }, [
                    el("div", { className: "fi-panel-header" }, [
                        el("h2", { className: "fi-panel-title", text: "Case Scope & Actions" }),
                        summary.last_analysis_at
                            ? badge(`Last Analyzed: ${formatUtcShort(summary.last_analysis_at)}`, "success")
                            : badge("Not Yet Analyzed", "neutral"),
                    ]),
                    el("p", {
                        style: "font-size: 13.5px; color: var(--fi-text-secondary); margin: 0 0 18px; line-height: 1.6;",
                        text: c.description || "No case description recorded.",
                    }),
                    el("div", { className: "fi-actions" }, [
                        el("button", {
                            type: "button",
                            className: "fi-btn fi-btn-primary",
                            text: "Upload Evidence & Custody (Classic View)",
                            onClick: () => openLegacyFn && openLegacyFn("phase1", c.id),
                        }),
                        el("button", {
                            type: "button",
                            className: "fi-btn fi-btn-secondary",
                            text: "Run Deterministic & ML Advisory (Classic View)",
                            onClick: () => openLegacyFn && openLegacyFn("phase2", c.id),
                        }),
                        el("button", {
                            type: "button",
                            className: "fi-btn fi-btn-secondary",
                            text: "Generate PDF Report (Classic View)",
                            onClick: () => openLegacyFn && openLegacyFn("phase5", c.id),
                        }),
                    ]),
                ])
            );
        } else {
            // Honest, transparent placeholder for M3-M6 tabs with direct bridge to working classic workflow
            container.appendChild(
                el("section", { className: "fi-panel" }, [
                    el("div", { className: "fi-panel-header" }, [
                        el("h2", { className: "fi-panel-title", text: `${tabSpec.label} — Scheduled for Milestone ${tabSpec.milestone}` }),
                        badge(`Planned: ${tabSpec.milestone}`, "info"),
                    ]),
                    el("p", {
                        style: "font-size: 13.5px; color: var(--fi-text-secondary); line-height: 1.6; margin: 0 0 16px;",
                        text: `The redesigned 2.0 ${tabSpec.label} view is scheduled for Milestone ${tabSpec.milestone}. All underlying backend APIs are active, and the full working workflow remains immediately accessible in the Classic View with Case ${c.case_number} pre-selected.`,
                    }),
                    tabSpec.legacyPhase
                        ? el("button", {
                              type: "button",
                              className: "fi-btn fi-btn-primary",
                              text: `Open ${tabSpec.label} in Classic Workflow`,
                              onClick: () => openLegacyFn && openLegacyFn(tabSpec.legacyPhase, c.id),
                          })
                        : null,
                ])
            );
        }
    } catch (err) {
        clear(container);
        container.appendChild(errorBanner(err.message || "Unable to load case workspace."));
    }
}
