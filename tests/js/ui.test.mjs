import test from "node:test";
import assert from "node:assert/strict";

import {
  escapeHtml, fmtScore, fmtMoney, renderMarkdown, riskBadge, statusBadge,
} from "../../frontend/assets/ui.js";
import { qs } from "../../frontend/assets/api.js";

test("escapeHtml neutralizes markup", () => {
  assert.equal(escapeHtml('<script>"x"</script>'), "&lt;script&gt;&quot;x&quot;&lt;/script&gt;");
});

test("markdown renderer escapes raw HTML but keeps safe formatting", () => {
  const html = renderMarkdown("## Heading\n\n**bold** and `<script>`\n\n- item");
  assert.ok(html.includes("<h2>Heading</h2>"));
  assert.ok(html.includes("<strong>bold</strong>"));
  assert.ok(html.includes("&lt;script&gt;"));
  assert.ok(!html.includes("<script>"));
  assert.ok(html.includes("<li>item</li>"));
});

test("numeric formatters are stable", () => {
  assert.equal(fmtScore(0.9), "0.900");
  assert.equal(fmtScore(null), "—");
  assert.equal(fmtMoney(1234.5), "1,234.50");
  assert.equal(fmtMoney(1234.5, "Yuan"), "1,234.50 Yuan");
});

test("risk and status badges distinguish high risk", () => {
  assert.ok(riskBadge("High Risk", "TIER_1_HIGH").includes("badge-high"));
  assert.ok(riskBadge("Medium Risk", "TIER_3_LOW").includes("badge-medium"));
  assert.ok(statusBadge("IN_REVIEW").includes("badge-progress"));
  assert.ok(statusBadge("CLOSED").includes("badge-closed"));
});

test("query string builder skips empty values", () => {
  assert.equal(qs({ page: 1, status: "", run_id: null, q: "A1" }), "?page=1&q=A1");
});
