import { api, qs, param, caseHref, ApiError } from "./api.js";
import {
  renderChrome, setContent, loadingBlock, emptyState, errorState,
  escapeHtml, fmtNumber, fmtScore, fmtMoney, fmtDate, riskBadge, statusBadge,
  icon, toast,
} from "./ui.js";

const runId = param("run");
const state = {
  page: Number(param("page", "1")) || 1,
  page_size: 25,
  status: param("status"),
  tier: param("tier"),
  q: param("q"),
  date_from: param("date_from"),
  date_to: param("date_to"),
  min_score: param("min_score"),
  sort: param("sort", "score_desc"),
};

renderChrome({
  active: "cases",
  runId,
  title: "Case queue",
  subtitle: "Account/day cases from the latest completed pipeline run",
});

const content = setContent(`
  <div class="between" style="margin-bottom:14px">
    <div>
      <h1 style="font-size:22px">Risk cases</h1>
      <p class="muted small" style="margin-top:4px">Every account/day flagged by the risk agent. Open a case to inspect evidence, graph, notes and the SAR draft.</p>
    </div>
    <div class="flex">
      <button class="btn btn-secondary" id="refresh">${icon("refresh")} Refresh</button>
      <a class="btn btn-secondary" href="settings.html">${icon("settings")} Settings</a>
    </div>
  </div>

  <div class="card" style="margin-bottom:14px">
    <form id="filters" class="filters">
      <div class="field wide">
        <label for="q">Account or case</label>
        <input class="input" id="q" name="q" placeholder="e.g. 8201A5710 or CASE-8201A5710-20221024" value="${escapeHtml(state.q)}">
      </div>
      <div class="field">
        <label for="status">Status</label>
        <select class="select" id="status" name="status">
          <option value="">All statuses</option>
          ${["NEW", "IN_REVIEW", "WAITING_INFO", "ESCALATED", "CLOSED"].map((value) => `<option value="${value}" ${state.status === value ? "selected" : ""}>${value.replace("_", " ")}</option>`).join("")}
        </select>
      </div>
      <div class="field">
        <label for="tier">Tier</label>
        <select class="select" id="tier" name="tier">
          <option value="">All tiers</option>
          ${["TIER_1_HIGH", "TIER_2_MEDIUM", "TIER_3_LOW"].map((value) => `<option value="${value}" ${state.tier === value ? "selected" : ""}>${value.replaceAll("_", " ")}</option>`).join("")}
        </select>
      </div>
      <div class="field">
        <label for="date_from">From date</label>
        <input class="input" id="date_from" name="date_from" type="date" value="${escapeHtml(state.date_from)}">
      </div>
      <div class="field">
        <label for="date_to">To date</label>
        <input class="input" id="date_to" name="date_to" type="date" value="${escapeHtml(state.date_to)}">
      </div>
      <div class="field">
        <label for="min_score">Min score</label>
        <input class="input" id="min_score" name="min_score" type="number" min="0" max="1" step="0.01" value="${escapeHtml(state.min_score)}">
      </div>
      <div class="field">
        <label for="sort">Sort</label>
        <select class="select" id="sort" name="sort">
          <option value="score_desc" ${state.sort === "score_desc" ? "selected" : ""}>Highest score</option>
          <option value="score_asc" ${state.sort === "score_asc" ? "selected" : ""}>Lowest score</option>
          <option value="date_desc" ${state.sort === "date_desc" ? "selected" : ""}>Newest date</option>
          <option value="date_asc" ${state.sort === "date_asc" ? "selected" : ""}>Oldest date</option>
          <option value="account" ${state.sort === "account" ? "selected" : ""}>Account id</option>
        </select>
      </div>
      <div class="field" style="flex:0 0 auto">
        <label>&nbsp;</label>
        <button class="btn" type="submit">${icon("search")} Apply</button>
      </div>
      <div class="field" style="flex:0 0 auto">
        <label>&nbsp;</label>
        <button class="btn btn-ghost" type="button" id="clear">Clear</button>
      </div>
    </form>
  </div>

  <div id="date-hints" class="tiny muted" style="margin-bottom:10px"></div>
  <div id="status-tabs" class="flex flex-wrap" style="gap:6px;margin-bottom:10px"></div>
  <div class="card card-flush">
    <div class="between" style="padding:14px 16px;border-bottom:1px solid var(--line)">
      <div class="small muted" id="total-label">Loading cases…</div>
      <div class="small muted" id="run-label">${runId ? `Run ${escapeHtml(runId)}` : ""}</div>
    </div>
    <div class="table-wrap" style="border:0;border-radius:0">
      <table class="table">
        <thead>
          <tr>
            <th>Case</th><th>Holder</th><th>Date</th><th>Risk</th><th class="num">Score</th><th>Status</th><th>Signals</th><th></th>
          </tr>
        </thead>
        <tbody id="case-body">
          <tr><td colspan="8" class="empty-row">Loading cases…</td></tr>
        </tbody>
      </table>
    </div>
    <div class="pagination" style="padding:12px 16px">
      <div class="page-info" id="page-info"></div>
      <div class="page-buttons">
        <button class="btn btn-secondary btn-sm" id="prev">Previous</button>
        <button class="btn btn-secondary btn-sm" id="next">Next</button>
      </div>
    </div>
  </div>
`);

const body = document.getElementById("case-body");
const totalLabel = document.getElementById("total-label");
const pageInfo = document.getElementById("page-info");
const prevButton = document.getElementById("prev");
const nextButton = document.getElementById("next");
const statusTabs = document.getElementById("status-tabs");
const dateHints = document.getElementById("date-hints");
let lastTotal = 0;
let lastPages = 0;

function collect() {
  const form = document.getElementById("filters");
  const data = new FormData(form);
  state.q = String(data.get("q") || "").trim();
  state.status = String(data.get("status") || "");
  state.tier = String(data.get("tier") || "");
  state.date_from = String(data.get("date_from") || "");
  state.date_to = String(data.get("date_to") || "");
  state.min_score = String(data.get("min_score") || "");
  state.sort = String(data.get("sort") || "score_desc");
  state.page = 1;
}

function syncUrl() {
  const query = qs({
    run: runId,
    page: state.page > 1 ? state.page : "",
    status: state.status,
    tier: state.tier,
    q: state.q,
    date_from: state.date_from,
    date_to: state.date_to,
    min_score: state.min_score,
    sort: state.sort !== "score_desc" ? state.sort : "",
  });
  window.history.replaceState(null, "", `index.html${query}`);
}

function signals(item) {
  const chips = [];
  if (item.is_high_risk_seed) chips.push(`<span class="chip">seed</span>`);
  if (item.hop1_count) chips.push(`<span class="chip neutral">1-hop ${item.hop1_count}</span>`);
  if (item.bidir_count) chips.push(`<span class="chip neutral">bidir ${item.bidir_count}</span>`);
  if (item.fan_score) chips.push(`<span class="chip neutral">fan ${Number(item.fan_score).toFixed(2)}</span>`);
  return chips.length ? chips.join("") : `<span class="muted tiny">—</span>`;
}

function renderStatusTabs(counts, total) {
  const entries = [["", "All", total], ["NEW", "New", counts.NEW || 0], ["IN_REVIEW", "In review", counts.IN_REVIEW || 0], ["WAITING_INFO", "Waiting", counts.WAITING_INFO || 0], ["ESCALATED", "Escalated", counts.ESCALATED || 0], ["CLOSED", "Closed", counts.CLOSED || 0]];
  statusTabs.innerHTML = entries.map(([value, label, count]) => `<button class="btn ${state.status === value ? "" : "btn-secondary"} btn-sm" data-status="${value}">${label} <span class="tiny">${fmtNumber(count)}</span></button>`).join("");
  statusTabs.querySelectorAll("[data-status]").forEach((button) => {
    button.addEventListener("click", () => {
      state.status = button.dataset.status;
      state.page = 1;
      document.getElementById("status").value = state.status;
      load();
    });
  });
}

function renderRows(items, data) {
  if (!items.length) {
    const dates = data.available_dates || [];
    const range = dates.length ? `${dates[0]} → ${dates[dates.length - 1]}` : "no dates";
    const selected = state.date_from || state.date_to
      ? ` Your date filter is ${state.date_from || "open"} → ${state.date_to || "open"}.`
      : "";
    body.innerHTML = `<tr><td colspan="8" class="empty-row">No cases match these filters.${selected} This run covers <strong>${escapeHtml(range)}</strong>. Use a run date chip above, clear the filters, or select the matching run.<div style="margin-top:10px"><button class="btn btn-secondary btn-sm" id="reset-empty">Clear filters</button></div></td></tr>`;
    document.getElementById("reset-empty")?.addEventListener("click", clearFilters);
    return;
  }
  body.innerHTML = items.map((item) => `
    <tr class="clickable" data-href="${caseHref("case.html", item.case_id, item.run_id)}">
      <td>
        <a class="mono" href="${caseHref("case.html", item.case_id, item.run_id)}">${escapeHtml(item.case_id)}</a>
        <div class="tiny muted">${escapeHtml(item.account_id)}</div>
      </td>
      <td>${escapeHtml(item.holder_name || "—")}</td>
      <td class="nowrap">${escapeHtml(fmtDate(item.graph_date))}</td>
      <td>${riskBadge(item.risk_label, item.tier)}</td>
      <td class="num"><strong>${fmtScore(item.risk_score)}</strong><div class="tiny muted">${escapeHtml(item.tier.replaceAll("_", " "))}</div></td>
      <td>${statusBadge(item.status)}</td>
      <td><div class="chip-row">${signals(item)}</div></td>
      <td class="right"><a class="btn btn-secondary btn-sm" href="${caseHref("case.html", item.case_id, item.run_id)}">Open ${icon("arrow")}</a></td>
    </tr>
  `).join("");
  body.querySelectorAll("tr[data-href]").forEach((row) => {
    row.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      window.location.href = row.dataset.href;
    });
  });
}

function renderDateHints(dates) {
  if (!dates || !dates.length) {
    dateHints.textContent = "";
    return;
  }
  const fromInput = document.getElementById("date_from");
  const toInput = document.getElementById("date_to");
  const first = dates[0];
  const last = dates[dates.length - 1];
  fromInput.min = first;
  fromInput.max = last;
  toInput.min = first;
  toInput.max = last;
  dateHints.innerHTML = `<span style="margin-right:6px">Available run dates:</span>${dates.map((day) => `<button class="btn btn-ghost btn-sm" data-run-date="${day}" style="padding:2px 7px;font-family:var(--mono)">${day}</button>`).join(" ")}<span style="margin-left:6px">(${first} → ${last})</span>`;
  dateHints.querySelectorAll("[data-run-date]").forEach((button) => {
    button.addEventListener("click", () => {
      state.date_from = button.dataset.runDate;
      state.date_to = button.dataset.runDate;
      state.page = 1;
      document.getElementById("date_from").value = state.date_from;
      document.getElementById("date_to").value = state.date_to;
      load();
    });
  });
}

function clearFilters() {
  document.getElementById("filters").reset();
  Object.assign(state, { page: 1, status: "", tier: "", q: "", date_from: "", date_to: "", min_score: "", sort: "score_desc" });
  load();
}

async function load() {
  syncUrl();
  body.innerHTML = `<tr><td colspan="8" class="empty-row">Loading cases…</td></tr>`;
  totalLabel.textContent = "Loading cases…";
  prevButton.disabled = true;
  nextButton.disabled = true;
  try {
    const data = await api(`/api/cases${qs({
      run_id: runId,
      page: state.page,
      page_size: state.page_size,
      status: state.status,
      tier: state.tier,
      risk_label: "",
      q: state.q,
      date_from: state.date_from,
      date_to: state.date_to,
      min_score: state.min_score,
      sort: state.sort,
    })}`);
    lastTotal = data.total;
    lastPages = data.page_count;
    renderDateHints(data.available_dates || []);
    renderRows(data.items, data);
    renderStatusTabs(data.status_counts || {}, data.total);
    totalLabel.textContent = `${fmtNumber(data.total)} flagged cases · page ${data.page} of ${data.page_count || 1}`;
    pageInfo.textContent = data.total ? `Showing ${(data.page - 1) * data.page_size + 1}–${Math.min(data.page * data.page_size, data.total)} of ${fmtNumber(data.total)}` : "No cases";
    prevButton.disabled = data.page <= 1;
    nextButton.disabled = data.page >= data.page_count;
    document.getElementById("run-label").textContent = `Run ${data.run_id}`;
  } catch (error) {
    body.innerHTML = `<tr><td colspan="8" class="empty-row">${escapeHtml(error.message)}</td></tr>`;
    totalLabel.textContent = "Unable to load cases";
    if (error instanceof ApiError && error.status === 404) {
      setContent(errorState(
        "No completed pipeline run",
        "No saved run with risk scores was found. Run the pipeline first, then refresh this queue.",
        "retry"
      ) + `<div class="card" style="margin-top:12px"><div class="stack"><p class="small muted">The pipeline is available from Settings or the command line:</p><code class="mono">python run_full_pipeline.py --input Data/processed/prod_sample_oct23_25.parquet</code><a class="btn" href="settings.html">${icon("settings")} Open Settings</a></div></div>`);
      document.getElementById("retry")?.addEventListener("click", () => window.location.reload());
    } else if (error instanceof ApiError && error.status === 401) {
      setContent(errorState("Authentication required", "Set the API token in Settings, then reload the case queue.", "retry") + `<div style="margin-top:12px"><a class="btn" href="settings.html">Open Settings</a></div>`);
      document.getElementById("retry")?.addEventListener("click", () => window.location.reload());
    } else {
      toast(error.message, "error");
    }
  }
}

document.getElementById("filters").addEventListener("submit", (event) => {
  event.preventDefault();
  collect();
  load();
});
document.getElementById("clear").addEventListener("click", clearFilters);
document.getElementById("refresh").addEventListener("click", load);
prevButton.addEventListener("click", () => { if (state.page > 1) { state.page -= 1; load(); } });
nextButton.addEventListener("click", () => { if (state.page < lastPages) { state.page += 1; load(); } });

load();
