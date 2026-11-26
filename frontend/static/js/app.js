import { clear, el, svgIcon } from "./dom.js";
import { navigateTo, parseRoute } from "./router.js";
import { getSelectedCaseId, getSelectedCaseLabel, subscribeStore } from "./store.js";
import { renderAuditView } from "./views/audit.js";
import { renderCasesView } from "./views/cases.js";
import { renderCaseWorkspaceView } from "./views/caseWorkspace.js";
import { renderDashboardView } from "./views/dashboard.js";
import { renderRulesView } from "./views/rules.js";

const PRIMARY_NAV = [
    { id: "dashboard", label: "Command Center", href: "#/dashboard", icon: "dashboard" },
    { id: "cases", label: "Cases", href: "#/cases", icon: "folder" },
    { id: "rules", label: "Detection Rules", href: "#/rules", icon: "rules" },
    { id: "audit", label: "Audit Trail", href: "#/audit", icon: "audit" },
];

function openLegacyPhase(phaseId, caseId = null) {
    const shell = document.getElementById("v2AppShell");
    const legacy = document.getElementById("legacyWorkspaceHost");
    if (!shell || !legacy) return;

    shell.hidden = true;
    legacy.hidden = false;
    document.body.classList.remove("fi-v2-active");

    // Switch legacy tab
    const tabBtn = document.querySelector(`#legacyWorkspaceHost .nav-tab[data-page="${phaseId}"]`);
    if (tabBtn) tabBtn.click();

    // Pre-select case across legacy dropdowns if a case is active
    const targetCaseId = caseId || getSelectedCaseId();
    if (targetCaseId) {
        const selectMap = {
            phase1: "caseSelect",
            phase2: "phase2CaseSelect",
            phase3: "phase3CaseSelect",
            phase4: "phase4CaseSelect",
            phase5: "phase5CaseSelect",
        };
        const selectId = selectMap[phaseId];
        setTimeout(() => {
            const sel = selectId ? document.getElementById(selectId) : null;
            if (sel && Array.from(sel.options).some((opt) => String(opt.value) === String(targetCaseId))) {
                sel.value = String(targetCaseId);
                sel.dispatchEvent(new Event("change"));
            }
        }, 180);
    }
}

function showV2Shell() {
    const shell = document.getElementById("v2AppShell");
    const legacy = document.getElementById("legacyWorkspaceHost");
    if (!shell || !legacy) return;
    legacy.hidden = true;
    shell.hidden = false;
    document.body.classList.add("fi-v2-active");
}

function buildShell(shellRoot) {
    clear(shellRoot);

    const sidebar = el("aside", { className: "fi-sidebar", id: "v2Sidebar", "aria-label": "Primary navigation" }, [
        el("div", { className: "fi-brand" }, [
            el("div", { className: "fi-brand-mark" }, [svgIcon("shield", 18)]),
            el("div", {}, [
                el("div", { className: "fi-brand-title", text: "ForensicInsight" }),
                el("div", { className: "fi-brand-sub", text: "Digital Forensics 2.0" }),
            ]),
        ]),
    ]);

    const nav = el("nav", { className: "fi-nav" });
    nav.appendChild(el("div", { className: "fi-nav-section-label", text: "Investigation" }));
    const navLinks = {};
    PRIMARY_NAV.forEach((item) => {
        const link = el("a", {
            href: item.href,
            className: "fi-nav-link",
            dataset: { navId: item.id },
            onClick: () => sidebar.classList.remove("open"),
        }, [svgIcon(item.icon, 16), el("span", { text: item.label })]);
        navLinks[item.id] = link;
        nav.appendChild(link);
    });

    nav.appendChild(el("div", { className: "fi-nav-section-label", text: "Classic Workflows (M3–M6 Bridge)" }));
    nav.appendChild(
        el("button", {
            type: "button",
            className: "fi-nav-link",
            id: "v2OpenClassicBtn",
            onClick: () => {
                sidebar.classList.remove("open");
                navigateTo("/classic/phase1");
            },
        }, [svgIcon("classic", 16), el("span", { text: "Evidence, Analysis & Reports" })])
    );

    sidebar.appendChild(nav);
    sidebar.appendChild(
        el("div", { className: "fi-sidebar-footer" }, [
            el("div", { text: "Read-only ML Advisory • R-A–R-D Rules" }),
            el("div", { text: "SHA-256 Evidence & Audit Integrity" }),
        ])
    );

    const breadcrumb = el("div", { className: "fi-breadcrumb", id: "v2Breadcrumb", text: "Command Center" });
    const casePillSlot = el("div", { id: "v2CasePillSlot" });

    const menuBtn = el("button", {
        type: "button",
        className: "fi-menu-btn",
        "aria-label": "Toggle navigation menu",
        onClick: () => sidebar.classList.toggle("open"),
    }, [svgIcon("menu", 18)]);

    const topbar = el("header", { className: "fi-topbar" }, [
        el("div", { className: "fi-topbar-left" }, [menuBtn, breadcrumb]),
        el("div", { className: "fi-topbar-right" }, [casePillSlot]),
    ]);

    const mainContent = el("main", { className: "fi-content", id: "v2Main", tabindex: "-1" });
    const mainCol = el("div", { className: "fi-main-col" }, [topbar, mainContent]);

    shellRoot.appendChild(el("a", { href: "#v2Main", className: "fi-skip-link", text: "Skip to main content" }));
    shellRoot.appendChild(sidebar);
    shellRoot.appendChild(mainCol);

    return { navLinks, breadcrumb, casePillSlot, mainContent };
}

function updateCasePill(slot) {
    clear(slot);
    const id = getSelectedCaseId();
    if (!id) return;
    const label = getSelectedCaseLabel() || `Case #${id}`;
    slot.appendChild(
        el("a", {
            href: `#/cases/${id}`,
            className: "fi-case-pill",
            title: "Return to active case workspace",
            text: `Active Case: ${label}`,
        })
    );
}

export function initForensicInsightApp() {
    const shellRoot = document.getElementById("v2AppShell");
    if (!shellRoot) return;

    const { navLinks, breadcrumb, casePillSlot, mainContent } = buildShell(shellRoot);

    const returnBtn = document.getElementById("returnToV2Btn");
    if (returnBtn) {
        returnBtn.addEventListener("click", () => {
            const activeId = getSelectedCaseId();
            navigateTo(activeId ? `/cases/${activeId}` : "/dashboard");
        });
    }

    subscribeStore(() => updateCasePill(casePillSlot));
    updateCasePill(casePillSlot);

    const renderCurrentRoute = async () => {
        const route = parseRoute(window.location.hash);

        if (route.name === "classic") {
            openLegacyPhase(route.params.phase || "phase1");
            return;
        }

        showV2Shell();
        Object.entries(navLinks).forEach(([id, link]) => {
            const isActive =
                id === route.name || (id === "cases" && route.name === "case-workspace");
            link.classList.toggle("active", isActive);
            if (isActive) {
                link.setAttribute("aria-current", "page");
            } else {
                link.removeAttribute("aria-current");
            }
        });

        if (route.name === "dashboard") {
            breadcrumb.textContent = "Command Center";
            document.title = "ForensicInsight 2.0 — Command Center";
            await renderDashboardView(mainContent);
        } else if (route.name === "cases") {
            breadcrumb.textContent = "Cases Directory";
            document.title = "ForensicInsight 2.0 — Cases";
            await renderCasesView(mainContent);
        } else if (route.name === "case-workspace") {
            breadcrumb.textContent = `Case Workspace #${route.params.caseId}`;
            document.title = `ForensicInsight 2.0 — Case #${route.params.caseId}`;
            await renderCaseWorkspaceView(
                mainContent,
                route.params.caseId,
                route.params.tab,
                (phase, cid) => openLegacyPhase(phase, cid)
            );
        } else if (route.name === "rules") {
            breadcrumb.textContent = "Detection Rules";
            document.title = "ForensicInsight 2.0 — Detection Rules";
            await renderRulesView(mainContent);
        } else if (route.name === "audit") {
            breadcrumb.textContent = "Audit Trail";
            document.title = "ForensicInsight 2.0 — Audit Trail";
            await renderAuditView(mainContent);
        }
    };

    window.addEventListener("hashchange", renderCurrentRoute);
    if (!window.location.hash) {
        window.location.hash = "#/dashboard";
    } else {
        renderCurrentRoute();
    }
}

document.addEventListener("DOMContentLoaded", initForensicInsightApp);
