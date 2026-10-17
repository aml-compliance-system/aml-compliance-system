export const API_BASE_KEY = "findetect.apiBase";
export const TOKEN_KEY = "findetect.apiToken";

export function apiBase() {
  const saved = localStorage.getItem(API_BASE_KEY);
  if (saved) return saved.replace(/\/+$/, "");
  // When the UI is served by FastAPI at /ui/, use the same origin. The
  // separate local static server (port 5600) still targets the API on 8000.
  if (window.location.port === "8000" || window.location.pathname.startsWith("/ui")) {
    return window.location.origin;
  }
  return "http://127.0.0.1:8000";
}

export function apiToken() {
  return sessionStorage.getItem(TOKEN_KEY) || "";
}

export function setApiBase(value) {
  const cleaned = String(value || "").trim().replace(/\/+$/, "");
  if (cleaned) localStorage.setItem(API_BASE_KEY, cleaned);
  else localStorage.removeItem(API_BASE_KEY);
}

export function setApiToken(value) {
  const cleaned = String(value || "").trim();
  if (cleaned) sessionStorage.setItem(TOKEN_KEY, cleaned);
  else sessionStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload || {};
  }
}


export async function api(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (options.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const token = apiToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetch(apiBase() + path, { ...options, headers });
  } catch (error) {
    if (error && error.name === "AbortError") throw error;
    throw new ApiError(
      `Cannot reach the FINDETECT API at ${apiBase()}. Start uvicorn main:app and check Settings.`,
      0,
      {}
    );
  }
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => ({}))
    : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" && payload ? payload.detail : payload;
    throw new ApiError(detail || `Request failed (HTTP ${response.status})`, response.status, payload);
  }
  return payload;
}

export function qs(params) {
  const search = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") search.set(key, value);
  });
  const text = search.toString();
  return text ? `?${text}` : "";
}

export function param(name, fallback = "") {
  return new URLSearchParams(window.location.search).get(name) || fallback;
}

export function caseHref(page, caseId, runId) {
  const query = new URLSearchParams();
  if (caseId) query.set("case_id", caseId);
  if (runId) query.set("run", runId);
  const suffix = query.toString();
  return `${page}${suffix ? `?${suffix}` : ""}`;
}

export async function download(path, filename, options = {}) {
  // Same-origin when the UI is served by FastAPI at /ui/, otherwise the API on port 8000.
  const base = window.location.pathname.startsWith("/ui/") ? "" : "http://127.0.0.1:8000";

  // Optional bearer token (only needed if FINDETECT_API_TOKEN is set).
  const headers = {};
  try {
    for (let i = 0; i < sessionStorage.length; i += 1) {
      const key = sessionStorage.key(i);
      if (/token/i.test(key)) {
        const value = sessionStorage.getItem(key);
        if (value) { headers["Authorization"] = `Bearer ${value}`; break; }
      }
    }
  } catch {}

  if (options.body) headers["Content-Type"] = "application/json";

  const response = await fetch(`${base}${path}`, {
    method: options.method || "GET",
    headers,
    body: options.body,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    } catch {}
    throw new Error(detail);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}