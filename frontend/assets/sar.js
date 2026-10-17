import { api, qs, param, caseHref, download } from "./api.js";
import {
  renderChrome, setContent, requireCase, emptyState, errorState, escapeHtml,
  fmtScore, fmtMoney, fmtNumber, fmtDateTime, riskBadge, sarBadge, statusBadge,
  icon, toast, setBusy, renderMarkdown,
} from "./ui.js";

const caseId = param("case_id");
const runId = param("run");
let sarData = null;
let caseInfo = null;
let selectedId = null;
let selectedDraft = null;

document.getElementById("legacy-close").addEventListener("click", () => document.getElementById("legacy-dialog").close());
document.getElementById("legacy-ok").addEventListener("click", () => document.getElementById("legacy-dialog").close());

if (!requireCase(caseId)) {
  // requireCase rendered the explanatory state.
} else {
  renderChrome({ active: "sar", caseId, runId, title: "SAR workspace", subtitle: "Evidence-grounded draft, review and export" });
  boot();
}

function boot() {
  setContent(`
    <div class="between" style="margin-bottom:12px">
      <div>
        <h1 style="font-size:20px">SAR draft & review</h1>
        <p class="muted small" style="margin-top:3px">Case-specific, evidence-grounded. AI generation is not approval; no filing occurs.</p>
      </div>
      <div class="flex">
        <a class="btn btn-secondary" href="${caseHref("case.html", caseId, runId)}">${icon("arrow")} Case workspace</a>
        <button class="btn" id="generate-draft">${icon("plus")} Generate draft</button>
      </div>
    </div>
    <div id="sar-body"><div class="card"><div class="skeleton" style="width:45%"></div><div class="skeleton" style="width:70%;margin-top:10px"></div></div></div>
    <dialog id="legacy-inline-dialog" class="dialog"><div class="dialog-head"><h3>Pipeline template draft</h3><button class="btn btn-ghost btn-icon" id="legacy-inline-close">×</button></div><div class="dialog-body" id="legacy-inline-body"></div></dialog>
  `);
  document.getElementById("generate-draft").addEventListener("click", generateDraft);
  document.getElementById("legacy-inline-close")?.addEventListener("click", () => document.getElementById("legacy-inline-dialog").close());
  loadAll();
}

async function loadAll() {
  const container = document.getElementById("sar-body");
  try {
    const [info, data] = await Promise.all([
      api(`/api/cases/${encodeURIComponent(caseId)}${qs({ run_id: runId, track: false })}`),
      api(`/api/cases/${encodeURIComponent(caseId)}/sar${qs({ run_id: runId })}`),
    ]);
    caseInfo = info;
    sarData = data;
    if (!selectedId && data.drafts.length) selectedId = data.drafts[0].id;
    renderLayout();
    if (selectedId) await loadDraft(selectedId);
  } catch (error) {
    container.innerHTML = errorState("Could not load SAR workspace", error.message, "retry");
    document.getElementById("retry")?.addEventListener("click", loadAll);
  }
}

function renderLayout() {
  const drafts = sarData?.drafts || [];
  const legacy = sarData?.legacy_report;
  const policy = sarData?.policy_validation || {};
  const container = document.getElementById("sar-body");
  container.innerHTML = `
    <div class="sar-layout">
      <div class="stack">
        <div class="card">
          <div class="between"><div class="card-title">Drafts</div><span class="badge badge-neutral">${drafts.length}</span></div>
          <div class="draft-list" id="draft-list" style="margin-top:10px">
            ${drafts.length ? drafts.map((draft) => `
              <div class="draft-item ${draft.id === selectedId ? "active" : ""}" data-draft="${draft.id}">
                <div class="between"><strong>v${draft.version}</strong>${sarBadge(draft.status)}</div>
                <div class="tiny muted" style="margin-top:4px">${escapeHtml(fmtDateTime(draft.updated_at))} · ${escapeHtml(draft.created_by)}</div>
              </div>`).join("") : `<p class="muted small">No draft yet. Generate one from the case evidence.</p>`}
          </div>
        </div>
        <div class="card">
          <div class="card-title">Case context</div>
          ${caseInfo ? `<div class="small" style="margin-top:8px"><strong>${escapeHtml(caseInfo.account_id)}</strong> · ${escapeHtml(caseInfo.graph_date)}</div>
          <div style="margin-top:6px">${riskBadge(caseInfo.risk_label, caseInfo.tier)} <span class="badge badge-neutral">${fmtScore(caseInfo.risk_score)}</span></div>
          <div class="tiny muted" style="margin-top:6px">${escapeHtml(caseInfo.holder_name || "holder unavailable")}</div>` : ""}
          <div class="notice warn" style="margin-top:10px">Policy validation is unavailable in this release. No policy citation or regulatory validation is claimed.</div>
          ${legacy ? `<button class="btn btn-secondary btn-sm" id="show-legacy" style="margin-top:10px">${icon("external")} Pipeline draft</button>` : ""}
        </div>
      </div>
      <div id="editor"><div class="card"><p class="muted small">Select or generate a draft.</p></div></div>
    </div>
  `;
  container.querySelectorAll("[data-draft]").forEach((element) => {
    element.addEventListener("click", () => { selectedId = Number(element.dataset.draft); renderLayout(); loadDraft(selectedId); });
  });
  document.getElementById("show-legacy")?.addEventListener("click", () => {
    document.getElementById("legacy-inline-body").innerHTML = `<div class="copy">${renderMarkdown(legacy || "")}</div><div class="notice warn" style="margin-top:10px">Legacy template output. It may contain no transaction evidence and is not a completed SAR.</div>`;
    document.getElementById("legacy-inline-dialog").showModal();
  });
}

async function loadDraft(id) {
  const editor = document.getElementById("editor");
  editor.innerHTML = `<div class="card"><div class="skeleton" style="width:50%"></div><div class="skeleton" style="width:80%;margin-top:10px"></div></div>`;
  try {
    selectedDraft = await api(`/api/sar/drafts/${id}`);
    renderEditor();
  } catch (error) {
    editor.innerHTML = errorState("Could not load draft", error.message, "retry");
    document.getElementById("retry")?.addEventListener("click", () => loadDraft(id));
  }
}

function renderEditor() {
  const d = selectedDraft;
  if (!d) return;
  const content = d.content || {};
  const narrative = content.narrative || {};
  const risk = content.risk || {};
  const activity = content.activity || {};
  const evidence = d.evidence || content.evidence || [];
  const missing = d.missing_fields || [];
  const approved = d.status === "APPROVED";
  const editor = document.getElementById("editor");
  editor.innerHTML = `
    <div class="card">
      <div class="between">
        <div>
          <div class="flex">${sarBadge(d.status)}<span class="badge badge-neutral">v${d.version}</span><span class="tiny muted">${escapeHtml(fmtDateTime(d.updated_at))}</span></div>
          <h2 style="font-size:18px;margin-top:8px">Suspicious Activity Report</h2>
          <p class="tiny muted">${escapeHtml(d.case_id)} · ${escapeHtml(d.created_by)} · ${d.approved_by ? `approved by ${escapeHtml(d.approved_by)} at ${escapeHtml(fmtDateTime(d.approved_at))}` : "not approved"}</p>
        </div>
        <div class="flex flex-wrap">
          ${!approved ? `<button class="btn" id="save-draft">${icon("save")} Save changes</button>` : ""}
          ${d.status === "DRAFT" ? `<button class="btn btn-secondary" id="submit-review">${icon("check")} Submit for review</button>` : ""}
          ${d.status === "IN_REVIEW" ? `<button class="btn btn-danger" id="approve-draft">${icon("check")} Approve</button>` : ""}
          <button class="btn btn-secondary" id="export-pdf">${icon("download")} PDF</button>
          <button class="btn btn-secondary" id="export-docx">${icon("download")} DOCX</button>
        </div>
      </div>

      <div class="sar-grid" style="margin-top:14px">
        <div><div class="kpi-label">Final risk score</div><div class="kpi-value sm">${fmtScore(risk.final_risk_score)}</div></div>
        <div><div class="kpi-label">Tier / label</div><div class="small">${escapeHtml(risk.tier || "—")} · ${escapeHtml(risk.risk_label || "—")}</div></div>
        <div><div class="kpi-label">Transactions</div><div class="small">${fmtNumber(activity.transaction_count)}</div></div>
        <div><div class="kpi-label">Currencies</div><div class="small">${escapeHtml((activity.currencies || []).join(", ") || "—")}</div></div>
      </div>

      <div class="notice warn" style="margin-top:12px">
        <strong>Missing information (${missing.length}).</strong> These fields are not in the pipeline source data. Complete them externally or explicitly acknowledge them at approval.
        <div class="missing-list" style="margin-top:6px">${missing.map((field) => `<span class="missing">${escapeHtml(field.replaceAll("_", " "))}</span>`).join("") || "None"}</div>
      </div>

      <div class="sar-section">
        <div class="card-title">Narrative</div>
        <div class="stack" style="margin-top:8px">
          ${narrativeField("summary", "Summary", narrative.summary)}
          ${narrativeField("suspicious_activity", "Suspicious activity (facts and indicators)", narrative.suspicious_activity)}
          ${narrativeField("evidence_summary", "Evidence summary", narrative.evidence_summary)}
          ${narrativeField("network_summary", "Network exposure", narrative.network_summary)}
          ${narrativeField("conclusion", "Conclusion and limitations", narrative.conclusion)}
        </div>
      </div>

      <div class="sar-section">
        <div class="card-title">Activity by currency</div>
        <div class="table-wrap" style="margin-top:8px"><table class="table"><thead><tr><th>Direction</th><th>Currency</th><th class="num">Count</th><th class="num">Total</th></tr></thead><tbody>
          ${(activity.aggregates || []).map((row) => `<tr><td>${escapeHtml(row.direction)}</td><td>${escapeHtml(row.currency || "—")}</td><td class="num">${fmtNumber(row.count)}</td><td class="num">${fmtMoney(row.total)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty-row">No activity aggregates.</td></tr>`}
        </tbody></table></div>
      </div>

      <div class="sar-section">
        <div class="card-title">Transaction evidence (${evidence.length})</div>
        <div class="table-wrap" style="margin-top:8px"><table class="table"><thead><tr><th>Reference</th><th>Time</th><th>Direction</th><th>Counterparty</th><th class="num">Amount</th><th>Currency</th><th>Format</th></tr></thead><tbody>
          ${evidence.map((item) => `<tr><td class="mono tiny">${escapeHtml(item.transaction_ref)}</td><td class="nowrap">${escapeHtml(item.timestamp)}</td><td>${escapeHtml(item.direction)}</td><td class="mono">${escapeHtml(item.counterparty)}</td><td class="num">${fmtMoney(item.amount)}</td><td>${escapeHtml(item.currency || "—")}</td><td>${escapeHtml(item.payment_format || "—")}</td></tr>`).join("") || `<tr><td colspan="7" class="empty-row">No evidence rows.</td></tr>`}
        </tbody></table></div>
      </div>

      <div class="sar-section">
        <div class="card-title">Investigator findings</div>
        ${(content.investigator_findings || []).length ? `<div class="list" style="margin-top:8px">${content.investigator_findings.map((finding) => `<div class="list-item"><div class="meta">Note ${finding.note_id} · ${escapeHtml(finding.author)} · ${escapeHtml(fmtDateTime(finding.created_at))}</div><p style="white-space:pre-wrap">${escapeHtml(finding.body)}</p></div>`).join("")}</div>` : `<p class="muted small" style="margin-top:8px">No investigator findings linked. Add a FINDING note in the case workspace and regenerate the draft.</p>`}
      </div>

      <div class="sar-section">
        <div class="between">
          <div>
            <div class="card-title">Review & approval</div>
            <p class="tiny muted" style="margin-top:4px">AI generation is not approval. Approval records your identity server-side. No external filing is performed.</p>
          </div>
          ${d.status === "IN_REVIEW" ? `<label class="small flex" style="text-transform:none;letter-spacing:0"><input type="checkbox" id="ack-missing"> I acknowledge the missing information</label>` : ""}
        </div>
      </div>
    </div>
  `;

  document.getElementById("save-draft")?.addEventListener("click", saveDraft);
  document.getElementById("submit-review")?.addEventListener("click", submitReview);
  document.getElementById("approve-draft")?.addEventListener("click", approveDraft);
  document.getElementById("export-pdf")?.addEventListener("click", () => exportDraft("pdf"));
  document.getElementById("export-docx")?.addEventListener("click", () => exportDraft("docx"));
}

function narrativeField(key, label, value) {
  return `<div class="field"><label for="narrative-${key}">${escapeHtml(label)}</label><textarea class="textarea" id="narrative-${key}" data-narrative="${key}">${escapeHtml(value || "")}</textarea></div>`;
}

function collectNarrative() {
  const content = structuredClone(selectedDraft.content || {});
  content.narrative = content.narrative || {};
  document.querySelectorAll("[data-narrative]").forEach((field) => {
    content.narrative[field.dataset.narrative] = field.value.trim();
  });
  return content;
}

async function saveDraft() {
  if (!window.confirm("Save changes to this SAR draft? This creates a new version and resets approval.")) return;
  const button = document.getElementById("save-draft");
  setBusy(button, true, "Saving…");
  try {
    selectedDraft = await api(`/api/sar/drafts/${selectedDraft.id}`, { method: "PATCH", body: JSON.stringify({ content: collectNarrative(), confirm: true }) });
    toast("Draft saved as a new version.", "ok");
    await loadAll();
  } catch (error) { toast(error.message, "error"); } finally { setBusy(button, false); }
}

async function submitReview() {
  if (!window.confirm("Submit this draft for review? The draft remains editable and approval is still required.")) return;
  const button = document.getElementById("submit-review");
  setBusy(button, true, "Submitting…");
  try {
    selectedDraft = await api(`/api/sar/drafts/${selectedDraft.id}/submit-review`, { method: "POST", body: JSON.stringify({ confirm: true }) });
    toast("Draft submitted for review.", "ok");
    await loadAll();
  } catch (error) { toast(error.message, "error"); } finally { setBusy(button, false); }
}

async function approveDraft() {
  const ack = document.getElementById("ack-missing");
  if (!ack?.checked) { toast("Acknowledge the missing information before approval.", "error"); return; }
  if (!window.confirm("Approve this SAR draft as the investigator? This records your identity and timestamp. It does not file the report.")) return;
  const button = document.getElementById("approve-draft");
  setBusy(button, true, "Approving…");
  try {
    selectedDraft = await api(`/api/sar/drafts/${selectedDraft.id}/approve`, { method: "POST", body: JSON.stringify({ confirm: true, acknowledge_missing: true }) });
    toast("Draft approved.", "ok");
    await loadAll();
  } catch (error) { toast(error.message, "error"); } finally { setBusy(button, false); }
}

async function exportDraft(format) {
  try {
    await download(
      `/api/sar/drafts/${selectedDraft.id}/export?format=${format}`,
      `sar-draft-${selectedDraft.case_id}-v${selectedDraft.version}.${format}`,
      { method: "POST" },
    );
    toast(`${format.toUpperCase()} export started.`, "ok");
  } catch (error) { toast(error.message, "error"); }
}

async function generateDraft() {
  if (!window.confirm("Generate a new SAR draft from the current case evidence and investigator findings?")) return;
  const button = document.getElementById("generate-draft");
  setBusy(button, true, "Generating…");
  try {
    const draft = await api(`/api/cases/${encodeURIComponent(caseId)}/sar/drafts${qs({ run_id: runId })}`, { method: "POST", body: JSON.stringify({ confirm: true, include_legacy_report: true }) });
    selectedId = draft.id;
    toast("Draft generated.", "ok");
    await loadAll();
  } catch (error) { toast(error.message, "error"); } finally { setBusy(button, false); }
}
