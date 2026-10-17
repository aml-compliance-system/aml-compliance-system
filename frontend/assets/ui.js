import { caseHref, apiToken, apiBase } from "./api.js";

const ICONS = {
  cases: '<path d="M4 5h16v14H4z"/><path d="M8 9h8M8 13h8M8 17h5"/>',
  list: '<path d="M8 6h12M8 12h12M8 18h12"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/>',
  graph: '<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="7" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M8 7l8 0M7 8l4 8M17 9l-3 7"/>',
  chat: '<path d="M4 5h16v11H9l-5 4z"/><path d="M8 9h8M8 12h5"/>',
  sar: '<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 12h7M9 16h7"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.4-2.4 1a8 8 0 0 0-1.7-1L14.5 3h-4l-.4 3.1a8 8 0 0 0-1.7 1l-2.4-1-2 3.4L6 11a7 7 0 0 0 0 2l-2 1.5 2 3.4 2.4-1a8 8 0 0 0 1.7 1l.4 3.1h4l.4-3.1a8 8 0 0 0 1.7-1l2.4 1 2-3.4-2-1.5c.1-.3.1-.7.1-1z"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  send: '<path d="M4 12l16-8-6 16-2.5-6.5z"/><path d="M11.5 13.5L20 4"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5H5V6h5"/>',
  check: '<path d="M20 6L9 17l-5-5"/>',
  alert: '<path d="M12 3l9 17H3z"/><path d="M12 9v5M12 17h.01"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
  bank: '<path d="M3 10l9-6 9 6M5 10v9M19 10v9M9 19v-5h6v5M3 21h18"/>',
  filter: '<path d="M4 5h16l-6 8v6l-4 2v-8z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  save: '<path d="M5 4h12l2 2v14H5z"/><path d="M8 4v6h8V4M8 20v-6h8v6"/>',
  play: '<path d="M7 4l12 8-12 8z"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
};

export function icon(name, className = "") {
  const path = ICONS[name] || ICONS.alert;
  return `<svg class="${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${path}</svg>`;
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function fmtMoney(value, currency = "") {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const formatted = number.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return currency ? `${formatted} ${currency}` : formatted;
}

export function fmtScore(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "—";
}

export function fmtDate(value) {
  if (!value) return "—";
  const text = String(value);
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return text;
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return text;
  return date.toISOString().slice(0, 10);
}

export function fmtDateTime(value) {
  if (!value) return "—";
  const date = new Date(String(value).endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(String(value)) ? value : `${value}Z`);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

export function fmtNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("en-US") : "—";
}

export function statusBadge(status) {
  const value = String(status || "NEW").toUpperCase();
  const map = {
    NEW: "badge-new",
    IN_REVIEW: "badge-progress",
    WAITING_INFO: "badge-waiting",
    ESCALATED: "badge-escalated",
    CLOSED: "badge-closed",
  };
  return `<span class="badge ${map[value] || "badge-neutral"}">${escapeHtml(value.replace(/_/g, " "))}</span>`;
}

export function riskBadge(label, tier) {
  const text = String(label || tier || "—");
  const cls = /high/i.test(text) ? "badge-high" : /medium/i.test(text) ? "badge-medium" : /low/i.test(text) ? "badge-low" : "badge-neutral";
  return `<span class="badge ${cls}">${escapeHtml(text)}</span>`;
}

export function sarBadge(status) {
  const value = String(status || "DRAFT").toUpperCase();
  const cls = value === "APPROVED" ? "badge-high" : value === "IN_REVIEW" ? "badge-progress" : value === "SUPERSEDED" ? "badge-closed" : "badge-neutral";
  return `<span class="badge ${cls}">${escapeHtml(value.replace(/_/g, " "))}</span>`;
}

export function renderMarkdown(text) {
  const source = escapeHtml(text || "");
  const lines = source.split(/\r?\n/);
  const out = [];
  let inList = false;
  const inline = (value) =>
    value
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|\s)\*([^*]+)\*(?=\s|$)/g, "$1<em>$2</em>");
  const closeList = () => {
    if (inList) { out.push("</ul>"); inList = false; }
  };
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) { closeList(); return; }
    if (/^###\s+/.test(trimmed)) { closeList(); out.push(`<h3>${inline(trimmed.replace(/^###\s+/, ""))}</h3>`); return; }
    if (/^##\s+/.test(trimmed)) { closeList(); out.push(`<h2>${inline(trimmed.replace(/^##\s+/, ""))}</h2>`); return; }
    if (/^#\s+/.test(trimmed)) { closeList(); out.push(`<h1>${inline(trimmed.replace(/^#\s+/, ""))}</h1>`); return; }
    if (/^[-*]\s+/.test(trimmed)) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inline(trimmed.replace(/^[-*]\s+/, ""))}</li>`);
      return;
    }
    closeList();
    out.push(`<p>${inline(trimmed)}</p>`);
  });
  closeList();
  return out.join("");
}

export function renderChrome({ active, caseId = "", runId = "", title = "", subtitle = "" }) {
  const sidebar = document.getElementById("sidebar");
  const topbar = document.getElementById("topbar");
  if (!sidebar || !topbar) return;

  const links = [
    { id: "cases", label: "Cases", icon: "cases", href: "index.html" },
    { id: "graph", label: "Graph Explorer", icon: "graph", href: caseHref("graph.html", caseId, runId), requiresCase: true },
    { id: "assistant", label: "AI Assistant", icon: "chat", href: caseHref("assistant.html", caseId, runId), requiresCase: true },
    { id: "sar", label: "SAR Reports", icon: "sar", href: caseHref("sar.html", caseId, runId), requiresCase: true },
  ];
  const nav = links.map((link) => {
    const disabled = link.requiresCase && !caseId;
    return `<a class="nav-link ${active === link.id ? "active" : ""} ${disabled ? "disabled" : ""}" href="${disabled ? "#" : link.href}" ${disabled ? 'aria-disabled="true" title="Select a case first"' : ""}>${icon(link.icon)}<span>${link.label}</span></a>`;
  }).join("");

  sidebar.innerHTML = `
    <div class="brand">
      <div class="brand-mark">FD</div>
      <div><div class="brand-title">FINDETECT</div><div class="brand-sub">AML investigation</div></div>
    </div>
    <div class="nav">
      <div class="nav-label">Workspace</div>
      ${nav}
      <div class="nav-label">Configuration</div>
      <a class="nav-link ${active === "settings" ? "active" : ""}" href="settings.html">${icon("settings")}<span>Settings</span></a>
    </div>
    <div class="sidebar-footer">
      <div>${icon("shield", "inline")} Local decision support</div>
      <div style="margin-top:4px">No automated filing. Human review required.</div>
      <div style="margin-top:8px"><a href="settings.html">API: ${escapeHtml(apiBase())}</a></div>
    </div>`;

  const caseText = caseId ? escapeHtml(caseId) : "No case selected";
  topbar.innerHTML = `
    <div>
      <div class="topbar-title">${escapeHtml(title || "AML Investigator")}</div>
      <div class="topbar-sub">${escapeHtml(subtitle || "Cases · evidence · graph · AI · SAR")}</div>
    </div>
    <div class="topbar-right">
      ${caseId ? `<span class="badge badge-teal mono">${caseText}</span>` : `<span class="badge badge-neutral">No case</span>`}
      ${runId ? `<span class="badge badge-neutral" title="Pipeline run">${escapeHtml(String(runId).slice(-18))}</span>` : ""}
      <span class="badge badge-neutral" title="Authentication mode">${apiToken() ? "Token" : "Local"}</span>
      ${caseId ? `<a class="btn btn-secondary btn-sm" href="${caseHref("case.html", caseId, runId)}">Case workspace</a>` : ""}
    </div>`;
  document.title = `${title || "FINDECT AML"} · FINDETECT`;
}

export function setContent(html) {
  const content = document.getElementById("content");
  if (content) content.innerHTML = html;
  return content;
}

export function loadingBlock(message = "Loading…") {
  return `<div class="card"><div class="stack">${Array.from({ length: 5 }).map(() => '<div class="skeleton" style="width:' + (55 + Math.random() * 40) + '%"></div>').join("")}<div class="muted small">${escapeHtml(message)}</div></div></div>`;
}

export function emptyState(title, message, action = "") {
  return `<div class="empty"><h3>${escapeHtml(title)}</h3><p>${escapeHtml(message)}</p>${action ? `<div style="margin-top:14px">${action}</div>` : ""}</div>`;
}

export function errorState(title, message, retryId = "") {
  return `<div class="error-state"><h3>${escapeHtml(title)}</h3><p>${escapeHtml(message)}</p>${retryId ? `<div style="margin-top:14px"><button class="btn btn-secondary" id="${retryId}">${icon("refresh")} Retry</button></div>` : ""}</div>`;
}

export function toast(message, kind = "") {
  let region = document.querySelector(".toast-region");
  if (!region) {
    region = document.createElement("div");
    region.className = "toast-region";
    document.body.appendChild(region);
  }
  const item = document.createElement("div");
  item.className = `toast ${kind}`;
  item.textContent = message;
  region.appendChild(item);
  setTimeout(() => item.remove(), 5000);
}

export function setBusy(button, busy, label = "Working…") {
  if (!button) return;
  if (busy) {
    button.dataset.original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = `${icon("clock")} ${escapeHtml(label)}`;
  } else {
    button.disabled = false;
    if (button.dataset.original) button.innerHTML = button.dataset.original;
  }
}

export function requireCase(caseId) {
  if (caseId && caseId !== "UNKNOWN-CASE") return true;
  setContent(emptyState(
    "No case selected",
    "Open the case queue and select an account/day case. The workspace never invents a case identifier.",
    `<a class="btn" href="index.html">${icon("cases")} Open case queue</a>`
  ));
  return false;
}

export function hrefCase(caseId, runId, page = "case.html") {
  return caseHref(page, caseId, runId);
}

export function formatStatus(value) {
  return String(value || "").replace(/_/g, " ");
}
