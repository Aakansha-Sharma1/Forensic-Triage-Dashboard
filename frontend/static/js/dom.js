/**
 * Safe DOM construction helpers — strictly createElement and textContent only.
 * Every server-provided string is rendered exclusively via textContent.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    if (attrs && typeof attrs === "object") {
        for (const [key, value] of Object.entries(attrs)) {
            if (value === null || value === undefined || value === false) continue;
            if (key === "className") {
                node.className = String(value);
            } else if (key === "text") {
                node.textContent = String(value);
            } else if (key === "disabled" || key === "checked" || key === "selected" || key === "hidden") {
                node[key] = Boolean(value);
            } else if (key.startsWith("on") && typeof value === "function") {
                node.addEventListener(key.slice(2).toLowerCase(), value);
            } else if (key === "dataset" && typeof value === "object") {
                for (const [dk, dv] of Object.entries(value)) {
                    node.dataset[dk] = String(dv);
                }
            } else {
                node.setAttribute(key, String(value));
            }
        }
    }
    appendChildren(node, children);
    return node;
}

export function appendChildren(parent, children) {
    const list = Array.isArray(children) ? children : [children];
    for (const child of list) {
        if (child === null || child === undefined || child === false) continue;
        if (typeof child === "string" || typeof child === "number") {
            parent.appendChild(document.createTextNode(String(child)));
        } else {
            parent.appendChild(child);
        }
    }
}

export function clear(node) {
    if (typeof node.replaceChildren === "function") {
        node.replaceChildren();
    } else {
        while (node.firstChild) {
            node.removeChild(node.firstChild);
        }
    }
}

const ICON_PATHS = {
    shield: ["M12 2L4 5v6c0 5 3.4 9.7 8 11 4.6-1.3 8-6 8-11V5l-8-3z"],
    dashboard: ["M3 3h8v8H3z", "M13 3h8v5h-8z", "M13 10h8v11h-8z", "M3 13h8v8H3z"],
    folder: ["M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"],
    rules: ["M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2", "M9 3h6v4H9z", "M9 12h6", "M9 16h4"],
    audit: ["M12 8v4l3 2", "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"],
    classic: ["M4 6h16", "M4 12h16", "M4 18h10"],
    menu: ["M4 6h16", "M4 12h16", "M4 18h16"],
    plus: ["M12 5v14", "M5 12h14"],
};

export function svgIcon(name, size = 16) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    const paths = ICON_PATHS[name] || ICON_PATHS.shield;
    for (const d of paths) {
        const p = document.createElementNS(SVG_NS, "path");
        p.setAttribute("d", d);
        svg.appendChild(p);
    }
    return svg;
}

export function badge(text, variant = "neutral") {
    return el("span", { className: `fi-badge fi-badge-${variant}`, text });
}

export function statusBadge(status) {
    const normalized = String(status || "Open");
    const map = {
        Open: "info",
        "In Analysis": "warning",
        "Under Review": "advisory",
        Closed: "success",
        HIGH: "danger",
        MEDIUM: "warning",
        LOW: "info",
        SUCCESS: "success",
        FAILURE: "danger",
    };
    return badge(normalized, map[normalized] || "neutral");
}

export function loadingBlock(message = "Loading...") {
    return el("div", { className: "fi-loading", role: "status", "aria-live": "polite" }, [
        el("span", { className: "fi-spinner", "aria-hidden": "true" }),
        el("span", { text: message }),
    ]);
}

export function emptyState(title, detail, actionNode = null) {
    return el("div", { className: "fi-empty" }, [
        el("p", { className: "fi-empty-title", text: title }),
        el("p", { text: detail }),
        actionNode ? el("div", { style: "margin-top: 14px;" }, [actionNode]) : null,
    ]);
}

export function errorBanner(message, variant = "danger") {
    return el("div", { className: `fi-banner fi-banner-${variant}`, role: "alert", text: message });
}

export function formatUtcShort(isoString) {
    if (!isoString) return "—";
    const text = String(isoString);
    return text.replace("T", " ").slice(0, 19) + " UTC";
}

export function openModal({ title, bodyNode, footerNodes = [], onClose = null }) {
    const prevFocus = document.activeElement;
    const close = () => {
        backdrop.remove();
        document.removeEventListener("keydown", onKey);
        if (prevFocus && typeof prevFocus.focus === "function") {
            prevFocus.focus();
        }
        if (typeof onClose === "function") onClose();
    };
    const onKey = (ev) => {
        if (ev.key === "Escape") close();
    };
    const closeBtn = el("button", {
        type: "button",
        className: "fi-btn fi-btn-ghost fi-btn-sm",
        "aria-label": "Close dialog",
        text: "Close",
        onClick: close,
    });
    const modal = el(
        "div",
        {
            className: "fi-modal",
            role: "dialog",
            "aria-modal": "true",
            "aria-label": title,
        },
        [
            el("div", { className: "fi-modal-header" }, [
                el("h2", { className: "fi-modal-title", text: title }),
                closeBtn,
            ]),
            el("div", { className: "fi-modal-body" }, [bodyNode]),
            el("div", { className: "fi-modal-footer" }, footerNodes),
        ]
    );
    const backdrop = el(
        "div",
        {
            className: "fi-modal-backdrop",
            onClick: (ev) => {
                if (ev.target === backdrop) close();
            },
        },
        [modal]
    );
    document.addEventListener("keydown", onKey);
    document.body.appendChild(backdrop);
    closeBtn.focus();
    return { close, modal, backdrop };
}
