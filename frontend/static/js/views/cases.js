import { api } from "../api.js";
import {
    clear,
    el,
    emptyState,
    errorBanner,
    formatUtcShort,
    loadingBlock,
    openModal,
    statusBadge,
    svgIcon,
} from "../dom.js";
import { navigateTo } from "../router.js";
import { setSelectedCase } from "../store.js";

export function openCreateCaseModal(onCreated) {
    const numberInput = el("input", {
        type: "text",
        id: "v2CaseNumber",
        className: "fi-input",
        placeholder: "e.g., FI-2026-001",
        required: true,
    });
    const titleInput = el("input", {
        type: "text",
        id: "v2CaseTitle",
        className: "fi-input",
        placeholder: "e.g., Intranet Apache Compromise Triage",
        required: true,
    });
    const invInput = el("input", {
        type: "text",
        id: "v2CaseInvestigator",
        className: "fi-input",
        placeholder: "Lead investigator name",
        required: true,
    });
    const descInput = el("textarea", {
        id: "v2CaseDescription",
        className: "fi-textarea",
        rows: "3",
        placeholder: "Investigation scope and context...",
    });
    const errSlot = el("div");

    const formBody = el("div", {}, [
        errSlot,
        el("div", { className: "fi-field" }, [
            el("label", { for: "v2CaseNumber", text: "Case Number *" }),
            numberInput,
        ]),
        el("div", { className: "fi-field" }, [
            el("label", { for: "v2CaseTitle", text: "Case Title *" }),
            titleInput,
        ]),
        el("div", { className: "fi-field" }, [
            el("label", { for: "v2CaseInvestigator", text: "Investigator *" }),
            invInput,
        ]),
        el("div", { className: "fi-field" }, [
            el("label", { for: "v2CaseDescription", text: "Description" }),
            descInput,
        ]),
    ]);

    let modalCtrl = null;
    const submitBtn = el("button", {
        type: "button",
        className: "fi-btn fi-btn-primary",
        text: "Create Case",
        onClick: async () => {
            clear(errSlot);
            const case_number = numberInput.value.trim();
            const title = titleInput.value.trim();
            const investigator = invInput.value.trim();
            const description = descInput.value.trim();
            if (!case_number || !title || !investigator) {
                errSlot.appendChild(errorBanner("Case Number, Title, and Investigator are required.", "warning"));
                return;
            }
            submitBtn.disabled = true;
            submitBtn.textContent = "Creating...";
            try {
                const res = await api.createCase({ case_number, title, investigator, description });
                if (modalCtrl) modalCtrl.close();
                if (typeof onCreated === "function" && res && res.case) {
                    onCreated(res.case);
                }
            } catch (err) {
                errSlot.appendChild(errorBanner(err.message || "Failed to create case."));
                submitBtn.disabled = false;
                submitBtn.textContent = "Create Case";
            }
        },
    });
    const cancelBtn = el("button", {
        type: "button",
        className: "fi-btn fi-btn-secondary",
        text: "Cancel",
        onClick: () => modalCtrl && modalCtrl.close(),
    });

    modalCtrl = openModal({
        title: "Create New Investigation Case",
        bodyNode: formBody,
        footerNodes: [cancelBtn, submitBtn],
    });
    numberInput.focus();
}

export async function renderCasesView(container) {
    clear(container);
    container.appendChild(loadingBlock("Loading cases directory..."));

    try {
        const [casesData, summaryData] = await Promise.all([
            api.getCases(),
            api.getDashboardSummary().catch(() => null),
        ]);
        const recentById = new Map();
        if (summaryData && Array.isArray(summaryData.recent_cases)) {
            summaryData.recent_cases.forEach((rc) => recentById.set(rc.id, rc));
        }

        const allCases = (casesData.cases || []).map((c) => {
            const enriched = recentById.get(c.id);
            return Object.assign({}, c, {
                status: (enriched && enriched.status) || c.status || "Open",
                evidence_count: enriched ? enriched.evidence_count : null,
                findings_count: enriched ? enriched.findings_count : null,
            });
        });

        clear(container);

        let searchQuery = "";
        let statusFilter = "ALL";

        const searchInput = el("input", {
            type: "search",
            className: "fi-input fi-search-input",
            placeholder: "Search by case number, title, or investigator...",
            "aria-label": "Search cases",
            onInput: (e) => {
                searchQuery = e.target.value.trim().toLowerCase();
                renderTable();
            },
        });

        const statusSelect = el(
            "select",
            {
                className: "fi-select",
                "aria-label": "Filter cases by status",
                onChange: (e) => {
                    statusFilter = e.target.value;
                    renderTable();
                },
            },
            ["ALL", "Open", "In Analysis", "Under Review", "Closed"].map((st) =>
                el("option", { value: st, text: st === "ALL" ? "All Statuses" : st })
            )
        );

        const listHost = el("div");

        const renderTable = () => {
            clear(listHost);
            const filtered = allCases.filter((c) => {
                if (statusFilter !== "ALL" && c.status !== statusFilter) return false;
                if (searchQuery) {
                    const hay = `${c.case_number} ${c.title} ${c.investigator} ${c.description || ""}`.toLowerCase();
                    if (!hay.includes(searchQuery)) return false;
                }
                return true;
            });

            if (filtered.length === 0) {
                listHost.appendChild(
                    emptyState(
                        allCases.length === 0 ? "No cases registered" : "No matching cases",
                        allCases.length === 0
                            ? "Create your first forensic case to begin evidence ingestion and analysis."
                            : "Adjust your search or status filter."
                    )
                );
                return;
            }

            const tbody = el("tbody");
            filtered.forEach((c) => {
                tbody.appendChild(
                    el("tr", {}, [
                        el("td", { className: "fi-mono", style: "color: var(--fi-text-primary); font-weight: 600;", text: c.case_number }),
                        el("td", {}, [
                            el("div", { style: "color: var(--fi-text-primary); font-weight: 500;", text: c.title }),
                            c.description
                                ? el("div", { style: "font-size: 12px; color: var(--fi-text-muted);", text: c.description.slice(0, 100) })
                                : null,
                        ]),
                        el("td", { text: c.investigator }),
                        el("td", {}, [statusBadge(c.status)]),
                        el("td", { className: "fi-mono", text: formatUtcShort(c.created_at) }),
                        el("td", {}, [
                            el("button", {
                                type: "button",
                                className: "fi-btn fi-btn-primary fi-btn-sm",
                                text: "Open Workspace",
                                onClick: () => {
                                    setSelectedCase(c.id, `${c.case_number} — ${c.title}`);
                                    navigateTo(`/cases/${c.id}`);
                                },
                            }),
                        ]),
                    ])
                );
            });

            listHost.appendChild(
                el("div", { className: "fi-table-wrap" }, [
                    el("table", { className: "fi-table", "aria-label": "Cases directory" }, [
                        el("thead", {}, [
                            el("tr", {}, [
                                el("th", { text: "Case Number" }),
                                el("th", { text: "Title & Scope" }),
                                el("th", { text: "Investigator" }),
                                el("th", { text: "Status" }),
                                el("th", { text: "Created" }),
                                el("th", { text: "Workspace" }),
                            ]),
                        ]),
                        tbody,
                    ]),
                ])
            );
        };

        container.appendChild(
            el("div", { className: "fi-page-header" }, [
                el("div", {}, [
                    el("h1", { className: "fi-page-title", text: "Cases Directory" }),
                    el("p", {
                        className: "fi-page-subtitle",
                        text: `${allCases.length} investigation case(s) registered in ForensicInsight.`,
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
                        [svgIcon("plus", 15), "Create Case"]
                    ),
                ]),
            ])
        );

        container.appendChild(
            el("section", { className: "fi-panel" }, [
                el("div", { className: "fi-toolbar" }, [searchInput, statusSelect]),
                listHost,
            ])
        );

        renderTable();
    } catch (err) {
        clear(container);
        container.appendChild(errorBanner(err.message || "Unable to load cases directory."));
    }
}
