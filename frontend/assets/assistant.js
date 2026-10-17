import { api, qs, param, caseHref } from "./api.js";
import {
  renderChrome, setContent, requireCase, escapeHtml, fmtScore, fmtNumber,
  fmtDateTime, riskBadge, icon, toast, renderMarkdown,
} from "./ui.js";

const caseId = param("case_id");
const runId = param("run");
const history = [];
let caseInfo = null;
let controller = null;
let busy = false;

if (!requireCase(caseId)) {
  // requireCase rendered the explanatory state.
} else {
  renderChrome({ active: "assistant", caseId, runId, title: "AI Assistant", subtitle: "Case-aware investigator" });
  boot();
}

function boot() {
  setContent(`
    <div class="chat-shell">
      <div class="between" style="margin-bottom:12px">
        <div>
          <h1 style="font-size:20px">AI investigation assistant</h1>
          <p class="muted small" style="margin-top:3px">Answers are grounded in the selected case and the three read-only evidence tools.</p>
        </div>
        <div class="flex">
          <a class="btn btn-secondary btn-sm" href="${caseHref("case.html", caseId, runId)}">${icon("arrow")} Case workspace</a>
          <a class="btn btn-secondary btn-sm" href="${caseHref("sar.html", caseId, runId)}">${icon("sar")} SAR</a>
        </div>
      </div>
      <div class="card" id="context-card" style="margin-bottom:12px"><div class="skeleton" style="width:55%"></div><div class="skeleton" style="width:35%;margin-top:8px"></div></div>
      <div id="chat-log" class="chat-log" aria-live="polite"></div>
      <div class="chat-input">
        <form class="chat-form" id="chat-form">
          <input class="input" id="chat-input" maxlength="4000" placeholder="Ask about the score, counterparties, network exposure or missing evidence…" autocomplete="off">
          <button class="btn" id="send-button" type="submit">${icon("send")} Ask</button>
          <button class="btn btn-secondary" id="stop-button" type="button" hidden>Stop</button>
        </form>
        <div class="chat-status"><span id="chat-status">Local model · 7B on CPU can take 2–4 minutes per answer.</span><span id="chat-timer"></span></div>
      </div>
    </div>
  `);

  document.getElementById("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });
  document.getElementById("stop-button").addEventListener("click", () => {
    if (controller) controller.abort();
  });
  loadContext();
  addMessage("assistant", `Case **${caseId}** is loaded. Ask about the risk score, transaction evidence, network exposure, or what evidence is missing. Every factual claim should be traceable to the evidence tools.`);
}

async function loadContext() {
  try {
    caseInfo = await api(`/api/cases/${encodeURIComponent(caseId)}${qs({ run_id: runId, track: false })}`);
    const tx = caseInfo.transaction_summary || {};
    document.getElementById("context-card").innerHTML = `
      <div class="between">
        <div>
          <div class="flex" style="gap:8px;flex-wrap:wrap">
            <span class="badge badge-teal mono">${escapeHtml(caseInfo.case_id)}</span>
            ${riskBadge(caseInfo.risk_label, caseInfo.tier)}
          </div>
          <div class="small" style="margin-top:8px"><strong>${escapeHtml(caseInfo.account_id)}</strong> · ${escapeHtml(caseInfo.holder_name || "holder unavailable")} · ${escapeHtml(caseInfo.graph_date)}</div>
          <div class="tiny muted" style="margin-top:3px">${fmtNumber(tx.transaction_count)} transactions · ${escapeHtml((tx.currencies || []).join(", ") || "no currency data")}</div>
        </div>
        <div class="right"><div class="kpi-label">Risk score</div><div class="kpi-value sm">${fmtScore(caseInfo.risk_score)}</div></div>
      </div>`;
  } catch (error) {
    document.getElementById("context-card").innerHTML = `<div class="notice danger">${escapeHtml(error.message)}</div>`;
  }
}

function toolChips(toolCalls) {
  if (!Array.isArray(toolCalls) || !toolCalls.length) return "";
  const groups = new Map();
  toolCalls.forEach((call) => {
    const name = call.name || "tool";
    if (!groups.has(name)) groups.set(name, { count: 0, ok: 0, summary: "" });
    const group = groups.get(name);
    group.count += 1;
    if (call.ok) group.ok += 1;
    group.summary = call.summary || group.summary;
  });
  return `<div class="chip-row" style="margin-bottom:7px">
    <span class="tiny muted" style="text-transform:uppercase;letter-spacing:.08em">Evidence</span>
    ${[...groups.entries()].map(([name, group]) => {
      const target = name === "get_transactions"
        ? `${caseHref("case.html", caseId, runId)}#transactions`
        : name === "get_graph_exposure"
          ? caseHref("graph.html", caseId, runId)
          : `${caseHref("case.html", caseId, runId)}#overview`;
      return `<a class="chip ${group.ok === group.count ? "" : "err"}" href="${target}" title="${escapeHtml(group.summary)}">${escapeHtml(name)}${group.count > 1 ? ` ×${group.count}` : ""}</a>`;
    }).join("")}
  </div>`;
}

function addMessage(role, text, meta = null) {
  const log = document.getElementById("chat-log");
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  if (role === "user") {
    wrapper.innerHTML = `<div class="bubble"><div class="msg-head">Investigator</div><div style="white-space:pre-wrap">${escapeHtml(text)}</div></div>`;
  } else {
    const warning = meta?.degraded
      ? `<div class="notice warn" style="margin-bottom:8px"><strong>Offline evidence mode.</strong> ${escapeHtml((meta.warnings || []).join(" ") || "The language model was unavailable; a deterministic evidence summary was used.")}</div>`
      : "";
    const footer = meta
      ? `<div class="tiny muted" style="margin-top:8px">${escapeHtml(meta.model || "")}${meta.elapsed_ms ? ` · ${(meta.elapsed_ms / 1000).toFixed(1)}s` : ""}${meta.steps ? ` · ${meta.steps} step(s)` : ""}</div>`
      : "";
    wrapper.innerHTML = `<div class="bubble"><div class="msg-head">AML Intelligence AI</div>${toolChips(meta?.tool_calls)}${warning}<div class="copy">${renderMarkdown(text)}</div>${footer}</div>`;
  }
  log.appendChild(wrapper);
  wrapper.scrollIntoView({ behavior: "smooth", block: "end" });
  return wrapper;
}

function addLoading() {
  const log = document.getElementById("chat-log");
  const wrapper = document.createElement("div");
  wrapper.className = "msg assistant";
  wrapper.id = "chat-loading";
  wrapper.innerHTML = `<div class="bubble"><div class="msg-head">AML Intelligence AI</div><div class="copy"><span class="muted">Gathering evidence… 0s</span></div></div>`;
  log.appendChild(wrapper);
  wrapper.scrollIntoView({ behavior: "smooth", block: "end" });
  return wrapper;
}

async function sendMessage() {
  if (busy) return;
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return;
  busy = true;
  const sendButton = document.getElementById("send-button");
  const stopButton = document.getElementById("stop-button");
  sendButton.disabled = true;
  stopButton.hidden = false;
  input.value = "";
  addMessage("user", text);
  history.push({ role: "user", content: text });
  const loading = addLoading();
  const started = Date.now();
  const timer = setInterval(() => {
    const target = loading.querySelector(".copy");
    if (target) target.innerHTML = `<span class="muted">Gathering evidence… ${Math.round((Date.now() - started) / 1000)}s</span>`;
    document.getElementById("chat-timer").textContent = `${Math.round((Date.now() - started) / 1000)}s`;
  }, 1000);
  controller = new AbortController();
  try {
    const response = await api(`/api/cases/${encodeURIComponent(caseId)}/chat${qs({ run_id: runId })}`, {
      method: "POST",
      body: JSON.stringify({ message: text, history: history.slice(-10) }),
      signal: controller.signal,
    });
    loading.remove();
    const answer = response.response || response.answer || "No answer returned.";
    history.push({ role: "assistant", content: answer });
    addMessage("assistant", answer, response);
  } catch (error) {
    loading.remove();
    if (error.name === "AbortError") {
      addMessage("assistant", "Request cancelled by the investigator. No answer was recorded.", { degraded: true, warnings: ["The client cancelled the request."] });
    } else {
      addMessage("assistant", `⚠️ ${error.message || "The investigation request failed."}`, { degraded: true });
      toast(error.message || "Request failed", "error");
      if (!input.value) input.value = text;
    }
  } finally {
    clearInterval(timer);
    busy = false;
    sendButton.disabled = false;
    stopButton.hidden = true;
    document.getElementById("chat-timer").textContent = "";
    controller = null;
    input.focus();
  }
}
