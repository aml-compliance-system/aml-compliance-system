import { api, apiBase, apiToken, setApiBase, setApiToken } from "./api.js";
import {
  renderChrome, setContent, escapeHtml, fmtNumber, fmtDateTime, icon, toast, setBusy,
} from "./ui.js";

renderChrome({ active: "settings", title: "Settings", subtitle: "Connection, run and capability status" });
boot();

function boot() {
  setContent(`
    <div class="between" style="margin-bottom:14px">
      <div>
        <h1 style="font-size:20px">Settings</h1>
        <p class="muted small" style="margin-top:3px">Local-first configuration. Secrets stay in this browser session; no token is written into project files.</p>
      </div>
      <button class="btn btn-secondary" id="refresh-meta">${icon("refresh")} Refresh status</button>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <div class="card-header"><div><div class="card-title">API connection</div><div class="card-sub">The dashboard and API can run on different local ports.</div></div></div>
        <div class="stack">
          <div class="field"><label for="api-base">API base URL</label><input class="input" id="api-base" value="${escapeHtml(apiBase())}" placeholder="http://127.0.0.1:8000"></div>
          <div class="field"><label for="api-token">Bearer token (optional)</label><input class="input" id="api-token" type="password" value="${escapeHtml(apiToken())}" placeholder="Only if FINDETECT_API_TOKEN is set on the server"></div>
          <div class="flex"><button class="btn" id="save-settings">${icon("save")} Save connection</button><button class="btn btn-secondary" id="test-connection">Test</button><button class="btn btn-ghost" id="clear-token">Clear token</button></div>
          <div class="notice warn">The default deployment is local-only. Binding the API to a network interface without a token and an explicit CORS allowlist is not supported for sensitive data.</div>
          <div class="small" id="connection-result"></div>
        </div>
      </div>

      <div class="card" id="status-card">
        <div class="card-header"><div><div class="card-title">Service status</div><div class="card-sub">Loaded from <span class="mono">/api/meta</span></div></div></div>
        <p class="muted small">Loading status…</p>
      </div>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="card-header"><div><div class="card-title">Run the pipeline</div><div class="card-sub">Used only to create a new case batch. The request is synchronous and can take several minutes.</div></div></div>
      <div class="filters">
        <div class="field wide"><label for="pipeline-input">Input parquet (inside Data/processed)</label><input class="input" id="pipeline-input" value="Data/processed/prod_sample_oct23_25.parquet"></div>
        <div class="field"><label for="pipeline-date">Date (optional)</label><input class="input" id="pipeline-date" type="date"></div>
        <div class="field" style="flex:0 0 auto"><label>&nbsp;</label><button class="btn" id="run-pipeline">${icon("play")} Run pipeline</button></div>
      </div>
      <p class="tiny muted" style="margin-top:10px">The pipeline writes a new run under <span class="mono">outputs/FINDETECT_Pipeline_Output/</span>. The case queue reads the newest saved run. The API only accepts inputs under <span class="mono">FINDETECT_DATA_ROOT</span>.</p>
      <div id="pipeline-result" style="margin-top:10px"></div>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="card-title">Capabilities and limitations</div>
      <div class="grid grid-2" style="margin-top:10px">
        <div class="small">
          <p><strong>Available:</strong> real case queue, transactions with source references, bounded graph analysis, case-aware AI assistant, investigator notes, audit timeline, SAR draft/review/approval, PDF/DOCX export.</p>
          <p style="margin-top:8px"><strong>Not available in this release:</strong> policy retrieval/RAG, regulatory validation, external SAR filing, multi-user role-based access, per-feature model explanations.</p>
        </div>
        <div class="small">
          <p><strong>Data:</strong> scores and transactions come from Parquet outputs. Workflow state is local SQLite. The application never fabricates missing cases, transactions or model outputs.</p>
          <p style="margin-top:8px"><strong>AI:</strong> the local model is decision support only. A timeout returns an explicit offline evidence memo; it never silently substitutes mock data.</p>
        </div>
      </div>
    </div>
  `);
  document.getElementById("save-settings").addEventListener("click", saveSettings);
  document.getElementById("test-connection").addEventListener("click", testConnection);
  document.getElementById("clear-token").addEventListener("click", () => { setApiToken(""); document.getElementById("api-token").value = ""; toast("Token cleared for this session.", "ok"); });
  document.getElementById("refresh-meta").addEventListener("click", loadMeta);
  document.getElementById("run-pipeline").addEventListener("click", runPipeline);
  loadMeta();
}

async function saveSettings() {
  setApiBase(document.getElementById("api-base").value);
  setApiToken(document.getElementById("api-token").value);
  toast("Connection settings saved.", "ok");
  document.getElementById("api-base").value = apiBase();
  await testConnection();
}

async function testConnection() {
  const target = document.getElementById("connection-result");
  target.innerHTML = `<span class="muted">Testing…</span>`;
  try {
    const meta = await api("/api/meta");
    target.innerHTML = `<span class="badge badge-low">Connected</span> <span class="small muted">run ${escapeHtml(meta.run_id || "none")}</span>`;
    renderMeta(meta);
  } catch (error) {
    target.innerHTML = `<span class="badge badge-high">Failed</span> <span class="small">${escapeHtml(error.message)}</span>`;
  }
}

async function loadMeta() {
  const card = document.getElementById("status-card");
  try {
    const meta = await api("/api/meta");
    renderMeta(meta);
  } catch (error) {
    card.innerHTML = `<div class="card-header"><div><div class="card-title">Service status</div></div></div><div class="notice danger">${escapeHtml(error.message)}</div>`;
  }
}

function renderMeta(meta) {
  const card = document.getElementById("status-card");
  if (!card) return;
  const caps = Object.entries(meta.capabilities || {}).map(([key, value]) => `<span class="badge ${value ? "badge-low" : "badge-neutral"}">${escapeHtml(key.replaceAll("_", " "))}</span>`).join(" ");
  card.innerHTML = `
    <div class="card-header"><div><div class="card-title">Service status</div><div class="card-sub">Run, auth and model configuration</div></div></div>
    <table class="table" style="min-width:0"><tbody>
      <tr><td>Pipeline run</td><td class="mono">${escapeHtml(meta.run_id || "none")}</td></tr>
      <tr><td>Auth mode</td><td>${escapeHtml(meta.auth_mode)}</td></tr>
      <tr><td>Investigator</td><td>${escapeHtml(meta.investigator)}</td></tr>
      <tr><td>Agent model</td><td class="mono">${escapeHtml(meta.model)}</td></tr>
      <tr><td>Native tools</td><td>${meta.native_tools ? "enabled" : "disabled (JSON protocol)"}</td></tr>
      <tr><td>Policy validation</td><td><span class="badge badge-medium">unavailable</span></td></tr>
      <tr><td>Case limits</td><td>${escapeHtml(JSON.stringify(meta.limits))}</td></tr>
    </tbody></table>
    <div class="chip-row" style="margin-top:10px">${caps}</div>
    <p class="tiny muted" style="margin-top:10px">${escapeHtml((meta.run_summary && JSON.stringify(meta.run_summary)) || "")}</p>
  `;
}

async function runPipeline() {
  const input = document.getElementById("pipeline-input").value.trim();
  const date = document.getElementById("pipeline-date").value;
  if (!window.confirm("Run the pipeline? This can take several minutes and will create a new saved run.")) return;
  const button = document.getElementById("run-pipeline");
  const result = document.getElementById("pipeline-result");
  setBusy(button, true, "Running pipeline…");
  const started = Date.now();
  const timer = setInterval(() => {
    result.innerHTML = `<div class="notice">Pipeline running… ${Math.round((Date.now() - started) / 1000)}s. Keep this page open.</div>`;
  }, 1000);
  try {
    const data = await api("/pipeline/analyze", { method: "POST", body: JSON.stringify({ final_parquet_path: input, date: date || null }) });
    result.innerHTML = `<div class="notice ok">Pipeline finished: ${fmtNumber(data.high_risk_alerts_found || 0)} top alerts returned. <a href="index.html">Open the case queue</a>.</div><p class="tiny muted mono">${escapeHtml(data.output_folder || "")}</p>`;
    toast("Pipeline completed.", "ok");
  } catch (error) {
    result.innerHTML = `<div class="notice danger">${escapeHtml(error.message)}</div>`;
    toast(error.message, "error");
  } finally {
    clearInterval(timer);
    setBusy(button, false);
  }
}
