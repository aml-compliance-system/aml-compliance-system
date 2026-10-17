import { api, qs, param, caseHref, ApiError } from "./api.js";
import {
  renderChrome, setContent, errorState, requireCase, escapeHtml, fmtScore,
  fmtMoney, fmtNumber, fmtDateTime, riskBadge, icon, toast,
} from "./ui.js";
import { renderGraph } from "./graph-render.js";

const caseId = param("case_id");
const runId = param("run");
const initialNode = param("node");
const filters = { depth: 1, from: "", to: "", include_holders: false };
let graphData = null;
let renderer = null;
let caseInfo = null;
let selected = null;
let lastFindings = null;

if (!requireCase(caseId)) {
  // requireCase has rendered the explanatory empty state.
} else {
  renderChrome({ active: "graph", caseId, runId, title: "Graph explorer", subtitle: "Transaction relationships for the selected case" });
  boot();
}

function boot() {
  setContent(`
    <div class="between" style="margin-bottom:10px">
      <div>
        <h1 style="font-size:20px;color:#f1f5f9">Relationship graph</h1>
        <p class="small" id="graph-subtitle" style="color:#8ca3b3;margin-top:3px">Loading case…</p>
      </div>
      <div class="flex">
        <a class="btn btn-secondary btn-sm" href="${caseHref("case.html", caseId, runId)}">${icon("arrow")} Case workspace</a>
        <button class="btn btn-secondary btn-sm" id="graph-refresh">${icon("refresh")} Reload</button>
      </div>
    </div>
    <div class="card" style="margin-bottom:10px;padding:12px">
      <div class="graph-toolbar">
        <div class="field"><label for="depth">Depth</label><select class="select" id="depth"><option value="1">1 hop</option><option value="2">2 hops</option></select></div>
        <div class="field"><label for="from-date">From date</label><input class="input" id="from-date" type="date"></div>
        <div class="field"><label for="to-date">To date</label><input class="input" id="to-date" type="date"></div>
        <div class="field" style="flex:0 0 auto"><label>&nbsp;</label><label class="small flex" style="text-transform:none;letter-spacing:0;color:#c8d7e1"><input type="checkbox" id="include-holders"> show holder nodes</label></div>
        <div class="field" style="flex:0 0 auto"><label>&nbsp;</label><button class="btn" id="apply-filters">${icon("filter")} Apply</button></div>
        <div class="spacer"></div>
        <div class="flex">
          <button class="btn btn-secondary btn-sm" id="zoom-in" title="Zoom in">+</button>
          <button class="btn btn-secondary btn-sm" id="zoom-out" title="Zoom out">−</button>
          <button class="btn btn-secondary btn-sm" id="zoom-reset" title="Reset view">Reset</button>
        </div>
      </div>
      <div class="between" style="margin-top:8px">
        <div class="tiny" id="graph-status" style="color:#8ca3b3">Loading graph…</div>
        <div class="tiny" id="graph-window" style="color:#8ca3b3"></div>
      </div>
    </div>
    <div class="graph-layout">
      <div class="graph-canvas" id="graph-canvas">
        <div class="empty" style="margin:40px">Loading graph…</div>
        <div class="graph-legend"><span><i style="background:#14b8a6"></i>focus account</span><span><i style="background:#38bdf8"></i>account</span><span><i style="background:#b45309;border-radius:2px"></i>holder (source record)</span><span>arrows = money direction</span></div>
      </div>
      <div class="graph-panel">
        <div class="card" id="selection-panel">
          <div class="card-title">Selection</div>
          <p class="small muted" style="margin-top:6px">Select a node or transaction edge to inspect evidence.</p>
        </div>
        <div class="card">
          <div class="card-title">Analysis</div>
          <div class="stack" style="margin-top:8px">
            <div class="notice">Algorithms run on the selected time window and are bounded. A finding is a lead, not proof of laundering.</div>
            <div class="field"><label for="path-source">Path source</label><input class="input" id="path-source" placeholder="account id"></div>
            <div class="field"><label for="path-target">Path target</label><input class="input" id="path-target" placeholder="account id"></div>
            <div class="flex"><button class="btn btn-secondary btn-sm" id="run-paths">${icon("arrow")} Trace paths</button><button class="btn btn-secondary btn-sm" id="set-source">Use selected as source</button></div>
            <div class="field"><label for="shared-accounts">Shared counterparties (comma-separated accounts)</label><input class="input" id="shared-accounts" placeholder="account1, account2"></div>
            <button class="btn btn-secondary btn-sm" id="run-shared">Find shared counterparties</button>
            <div class="flex">
              <button class="btn btn-secondary btn-sm" id="run-cycles">${icon("refresh")} Find cycles</button>
              <button class="btn btn-secondary btn-sm" id="run-components">${icon("graph")} Component</button>
            </div>
          </div>
        </div>
        <div class="card" id="analysis-panel">
          <div class="card-title">Findings</div>
          <p class="small muted" style="margin-top:6px">Run an analysis to see evidence-backed findings.</p>
        </div>
      </div>
    </div>
  `);

  document.getElementById("graph-refresh").addEventListener("click", loadGraph);
  document.getElementById("apply-filters").addEventListener("click", () => {
    filters.depth = Number(document.getElementById("depth").value);
    filters.from = document.getElementById("from-date").value;
    filters.to = document.getElementById("to-date").value;
    filters.include_holders = document.getElementById("include-holders").checked;
    loadGraph();
  });
  document.getElementById("zoom-in").addEventListener("click", () => renderer?.zoomIn());
  document.getElementById("zoom-out").addEventListener("click", () => renderer?.zoomOut());
  document.getElementById("zoom-reset").addEventListener("click", () => renderer?.reset());
  document.getElementById("run-paths").addEventListener("click", runPaths);
  document.getElementById("run-cycles").addEventListener("click", runCycles);
  document.getElementById("run-shared").addEventListener("click", runShared);
  document.getElementById("run-components").addEventListener("click", runComponents);
  document.getElementById("set-source").addEventListener("click", () => {
    if (!selected || selected.type !== "account") return toast("Select an account node first.", "error");
    document.getElementById("path-source").value = selected.account_id;
  });

  loadCaseInfo();
  loadGraph();
}

async function loadCaseInfo() {
  try {
    caseInfo = await api(`/api/cases/${encodeURIComponent(caseId)}${qs({ run_id: runId, track: false })}`);
    const subtitle = document.getElementById("graph-subtitle");
    if (subtitle) subtitle.innerHTML = `${escapeHtml(caseInfo.account_id)} · ${escapeHtml(caseInfo.holder_name || "holder unavailable")} · ${riskBadge(caseInfo.risk_label, caseInfo.tier)} · score ${fmtScore(caseInfo.risk_score)}`;
  } catch (error) {
    toast(error.message, "error");
  }
}

function windowParams() {
  return {
    run_id: runId,
    depth: filters.depth,
    limit_nodes: 100,
    limit_edges: 300,
    include_holders: filters.include_holders,
    from: filters.from,
    to: filters.to,
  };
}

async function loadGraph() {
  const canvas = document.getElementById("graph-canvas");
  const status = document.getElementById("graph-status");
  canvas.innerHTML = `<div class="empty" style="margin:40px">Loading graph…</div>`;
  try {
    graphData = await api(`/api/cases/${encodeURIComponent(caseId)}/graph${qs(windowParams())}`);
    if (!graphData.nodes.length) {
      canvas.innerHTML = `<div class="empty" style="margin:40px">No graph nodes in this window.</div>`;
      status.textContent = "No nodes";
      return;
    }
    canvas.innerHTML = "";
    renderer = renderGraph(canvas, graphData, {
      onNode: selectNode,
      onEdge: selectEdge,
      onExpand: expandNode,
    });
    const accounts = graphData.nodes.filter((node) => node.type === "account").length;
    const holders = graphData.nodes.length - accounts;
    status.innerHTML = `${fmtNumber(accounts)} accounts${holders ? ` · ${fmtNumber(holders)} holders` : ""} · ${fmtNumber(graphData.edges.length)} transaction edges ${graphData.truncated ? '<span class="badge badge-medium" style="margin-left:6px">truncated</span>' : ""}`;
    document.getElementById("graph-window").textContent = `${graphData.window.from} → ${graphData.window.to}`;
    if (graphData.truncated) {
      status.innerHTML += ` <span title="${escapeHtml(graphData.truncation_reason || "")}">— ${escapeHtml(graphData.truncation_reason || "bounded result")}</span>`;
    }
    const focal = graphData.nodes.find((node) => node.is_focal) || graphData.nodes[0];
    const target = initialNode ? graphData.nodes.find((node) => node.id === initialNode) : focal;
    if (target) selectNode(target);
  } catch (error) {
    canvas.innerHTML = `<div class="empty" style="margin:40px">${escapeHtml(error.message)}</div>`;
    status.textContent = "Graph unavailable";
  }
}

function selectNode(node) {
  selected = node;
  const panel = document.getElementById("selection-panel");
  const isAccount = node.type === "account";
  panel.innerHTML = `
    <div class="between"><div class="card-title">${isAccount ? "Account node" : "Holder node"}</div><span class="badge badge-neutral">${node.is_focal ? "focus" : `hop ${node.hop}`}</span></div>
    <dl class="graph-detail">
      <dt>Identifier</dt><dd class="mono">${escapeHtml(node.account_id || node.entity_id || node.id)}</dd>
      ${isAccount ? `<dt>Holder (source record)</dt><dd>${escapeHtml(node.holder_name || "unavailable")}</dd>` : `<dt>Linked account</dt><dd class="mono">${escapeHtml(node.id)}</dd>`}
      ${node.bank ? `<dt>Bank</dt><dd>${escapeHtml(node.bank)}</dd>` : ""}
    </dl>
    <div class="flex flex-wrap" style="margin-top:10px">
      ${isAccount ? `<button class="btn btn-secondary btn-sm" id="expand-selected">${icon("plus")} Expand one hop</button><a class="btn btn-secondary btn-sm" href="${caseHref("case.html", caseId, runId)}#transactions">${icon("list")} Transactions</a>` : ""}
    </div>
    <p class="tiny muted" style="margin-top:8px">Holder nodes use source-record entity names, not verified KYC ownership.</p>
  `;
  document.getElementById("expand-selected")?.addEventListener("click", () => expandNode(node));
  if (isAccount) {
    document.getElementById("path-source").value = node.account_id;
    if (!document.getElementById("path-target").value) {
      const focal = graphData.nodes.find((item) => item.is_focal);
      if (focal && focal.account_id !== node.account_id) document.getElementById("path-target").value = focal.account_id;
    }
    const shared = document.getElementById("shared-accounts");
    if (!shared.value) {
      const other = graphData.nodes.find((item) => item.type === "account" && item.account_id !== node.account_id)?.account_id;
      shared.value = [node.account_id, other].filter(Boolean).join(", ");
    }
  }
}

function selectEdge(edge) {
  if (!edge || edge.type === "holder") return;
  const panel = document.getElementById("selection-panel");
  panel.innerHTML = `
    <div class="between"><div class="card-title">Transaction edge</div><span class="badge badge-neutral mono">${escapeHtml(edge.transaction_ref || edge.id)}</span></div>
    <dl class="graph-detail">
      <dt>From → To</dt><dd class="mono">${escapeHtml(edge.source)} → ${escapeHtml(edge.target)}</dd>
      <dt>Timestamp</dt><dd>${escapeHtml(edge.timestamp || "—")}</dd>
      <dt>Amount</dt><dd>${fmtMoney(edge.amount, edge.currency || "")}</dd>
      <dt>Payment format</dt><dd>${escapeHtml(edge.payment_format || "—")}</dd>
      <dt>Source file</dt><dd class="mono">${escapeHtml(edge.source_file || "—")}</dd>
    </dl>
  `;
}

async function expandNode(node) {
  if (!node || node.type !== "account") return;
  try {
    const extra = await api(`/api/cases/${encodeURIComponent(caseId)}/graph/expand${qs({ run_id: runId, from: filters.from, to: filters.to })}`, {
      method: "POST",
      body: JSON.stringify({ account: node.account_id, direction: "both" }),
    });
    const nodes = new Map(graphData.nodes.map((item) => [item.id, item]));
    extra.nodes.forEach((item) => nodes.set(item.id, item));
    const edges = new Map(graphData.edges.map((item) => [item.id, item]));
    extra.edges.forEach((item) => edges.set(item.id, item));
    graphData = { ...graphData, nodes: [...nodes.values()], edges: [...edges.values()], truncated: graphData.truncated || extra.truncated };
    const canvas = document.getElementById("graph-canvas");
    canvas.innerHTML = "";
    renderer = renderGraph(canvas, graphData, { onNode: selectNode, onEdge: selectEdge, onExpand: expandNode });
    document.getElementById("graph-status").innerHTML = `${fmtNumber(graphData.nodes.length)} nodes · ${fmtNumber(graphData.edges.length)} edges ${extra.truncated ? '<span class="badge badge-medium">truncated</span>' : ""}`;
    toast("Node expanded.", "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

function findingHtml(finding, index) {
  const refs = (finding.edges || []).map((edge) => edge.transaction_ref).filter(Boolean);
  const nodes = finding.nodes || [];
  const title = finding.nodes?.length ? finding.nodes.join(" → ") : (finding.counterparty ? `Counterparty ${finding.counterparty}` : "Finding");
  return `<div class="finding" data-finding="${index}" data-refs="${escapeHtml(refs.join(","))}" data-nodes="${escapeHtml(nodes.join(","))}">
    <div class="small">${escapeHtml(title)}</div>
    ${finding.size ? `<div class="tiny muted">${fmtNumber(finding.size)} nodes in component</div>` : ""}
    ${finding.account_count ? `<div class="tiny muted">shared by ${fmtNumber(finding.account_count)} accounts: ${escapeHtml((finding.accounts || []).join(", "))}</div>` : ""}
    <div class="refs">${escapeHtml(refs.slice(0, 6).join(" · "))}${refs.length > 6 ? " …" : ""}</div>
  </div>`;
}

function showFindings(title, rule, findings, truncated) {
  lastFindings = findings;
  const panel = document.getElementById("analysis-panel");
  panel.innerHTML = `
    <div class="card-title">${escapeHtml(title)}</div>
    <p class="tiny" style="color:#8ca3b3;margin-top:4px">${escapeHtml(rule || "")}</p>
    ${truncated ? '<div class="notice warn" style="margin-top:8px">Result was truncated by the graph bounds.</div>' : ""}
    <div style="margin-top:8px">
      ${findings.length ? findings.map((finding, index) => findingHtml(finding, index)).join("") : '<p class="small muted">No matching findings in this window. That is a valid result, not an error.</p>'}
    </div>
    <div class="flex" style="margin-top:8px"><button class="btn btn-secondary btn-sm" id="clear-highlight">Clear highlight</button></div>
  `;
  panel.querySelectorAll("[data-finding]").forEach((element) => {
    element.addEventListener("click", () => {
      const refs = element.dataset.refs ? element.dataset.refs.split(",").filter(Boolean) : [];
      const nodes = element.dataset.nodes ? element.dataset.nodes.split(",").filter(Boolean) : [];
      renderer?.highlight({ refs, nodeIds: nodes, finding: true });
    });
  });
  document.getElementById("clear-highlight")?.addEventListener("click", () => renderer?.highlight());
}

async function runPaths() {
  const source = document.getElementById("path-source").value.trim();
  const target = document.getElementById("path-target").value.trim();
  if (!source || !target) return toast("Enter source and target accounts.", "error");
  try {
    const result = await api(`/api/cases/${encodeURIComponent(caseId)}/graph/paths${qs({ run_id: runId, from: filters.from, to: filters.to })}`, {
      method: "POST",
      body: JSON.stringify({ source, target, max_depth: 4 }),
    });
    showFindings("Money-flow paths", result.rule, result.findings, result.truncated);
  } catch (error) { toast(error.message, "error"); }
}

async function runCycles() {
  try {
    const result = await api(`/api/cases/${encodeURIComponent(caseId)}/graph/cycles${qs({ run_id: runId, from: filters.from, to: filters.to })}`, {
      method: "POST",
      body: JSON.stringify({ max_length: 4 }),
    });
    showFindings("Time-ordered cycles", result.rule, result.findings, result.truncated);
  } catch (error) { toast(error.message, "error"); }
}

async function runShared() {
  const raw = document.getElementById("shared-accounts").value.trim();
  const accounts = raw.split(",").map((item) => item.trim()).filter(Boolean);
  if (accounts.length < 2) return toast("Enter at least two accounts.", "error");
  try {
    const result = await api(`/api/cases/${encodeURIComponent(caseId)}/graph/shared-counterparties${qs({ run_id: runId, from: filters.from, to: filters.to })}`, {
      method: "POST",
      body: JSON.stringify({ accounts, min_shared: 2 }),
    });
    showFindings("Shared counterparties", result.rule, result.findings, result.truncated);
  } catch (error) { toast(error.message, "error"); }
}

async function runComponents() {
  const account = selected?.account_id || graphData?.nodes.find((node) => node.is_focal)?.account_id;
  try {
    const result = await api(`/api/cases/${encodeURIComponent(caseId)}/graph/components${qs({ run_id: runId, from: filters.from, to: filters.to })}`, {
      method: "POST",
      body: JSON.stringify({ account }),
    });
    showFindings("Connected component", result.rule, result.findings, result.truncated);
  } catch (error) { toast(error.message, "error"); }
}
