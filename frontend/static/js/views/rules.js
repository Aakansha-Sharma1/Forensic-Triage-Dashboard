import { api } from "../api.js";
import { badge, clear, el, errorBanner, loadingBlock, statusBadge } from "../dom.js";

export async function renderRulesView(container) {
    clear(container);
    container.appendChild(loadingBlock("Loading deterministic rule registry..."));

    try {
        const data = await api.getRules();
        const rules = Array.isArray(data.rules) ? data.rules : [];
        clear(container);

        container.appendChild(
            el("div", { className: "fi-page-header" }, [
                el("div", {}, [
                    el("h1", { className: "fi-page-title", text: "Deterministic Detection Rules (R-A–R-D)" }),
                    el("p", {
                        className: "fi-page-subtitle",
                        text: "Immutable, versioned forensic rule definitions audited by SHA-256 fingerprint.",
                    }),
                ]),
                badge(`${rules.length} Active Rule(s)`, "info"),
            ])
        );

        container.appendChild(
            el("section", { className: "fi-panel" }, [
                el("div", { className: "fi-kpi-label", text: "Canonical Rule Registry SHA-256 Fingerprint" }),
                el("div", { className: "fi-mono", style: "color: var(--fi-status-success);", text: data.fingerprint || "N/A" }),
            ])
        );

        const tbody = el("tbody");
        rules.forEach((r) => {
            tbody.appendChild(
                el("tr", {}, [
                    el("td", { className: "fi-mono", style: "color: var(--fi-text-primary); font-weight: 700;", text: r.rule_id }),
                    el("td", { style: "color: var(--fi-text-primary); font-weight: 500;", text: r.name }),
                    el("td", { className: "fi-mono", text: `v${r.version}` }),
                    el("td", {}, [statusBadge(r.severity)]),
                    el("td", { text: r.description }),
                ])
            );
        });

        container.appendChild(
            el("section", { className: "fi-panel" }, [
                el("div", { className: "fi-table-wrap" }, [
                    el("table", { className: "fi-table", "aria-label": "Deterministic rule registry" }, [
                        el("thead", {}, [
                            el("tr", {}, [
                                el("th", { text: "Rule ID" }),
                                el("th", { text: "Name" }),
                                el("th", { text: "Version" }),
                                el("th", { text: "Severity" }),
                                el("th", { text: "Description" }),
                            ]),
                        ]),
                        tbody,
                    ]),
                ]),
            ])
        );
    } catch (err) {
        clear(container);
        container.appendChild(errorBanner(err.message || "Unable to load rule registry."));
    }
}
