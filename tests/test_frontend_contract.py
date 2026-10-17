"""Static contracts for the investigator frontend.

Browser interaction tests require a browser automation dependency that is not
available in this environment; these checks still catch missing assets, broken
page wiring and the two known placeholder regressions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@pytest.mark.parametrize(
    "page",
    ["index.html", "case.html", "graph.html", "assistant.html", "sar.html", "settings.html"],
)
def test_pages_have_shell_and_assets(page):
    text = (FRONTEND / page).read_text(encoding="utf-8")
    assert 'id="sidebar"' in text
    assert 'id="topbar"' in text
    assert 'id="content"' in text
    assert "assets/app.css" in text
    for reference in re.findall(r'(?:src|href)="(assets/[^"]+)"', text):
        assert (FRONTEND / reference).is_file(), f"{page} references missing {reference}"


def test_case_and_assistant_pages_do_not_use_a_placeholder_case():
    case_page = (FRONTEND / "case.html").read_text(encoding="utf-8")
    assistant = (FRONTEND / "assistant.html").read_text(encoding="utf-8")
    sar = (FRONTEND / "sar.html").read_text(encoding="utf-8")
    for text in (case_page, assistant, sar):
        assert "UNKNOWN-CASE" not in text
        assert "No report context loaded" not in text


def test_legacy_chat_redirects_instead_of_inventing_a_case():
    text = (FRONTEND / "chat.html").read_text(encoding="utf-8")
    assert "assistant.html" in text
    assert "index.html" in text
    assert "No case is selected" in text


def test_case_js_handles_missing_case_and_real_tabs():
    text = (FRONTEND / "assets/case.js").read_text(encoding="utf-8")
    assert "requireCase(caseId)" in text
    assert "transactions" in text
    assert "notes" in text
    assert "timeline" in text
    assert "renderMarkdown" in text
    assert "Is Laundering" not in text


def test_sar_page_exposes_validation_and_exports():
    text = (FRONTEND / "assets/sar.js").read_text(encoding="utf-8")
    assert "missing_fields" in text
    assert "acknowledge_missing" in text
    assert "export-pdf" in text
    assert "export-docx" in text
    assert "policy" in text.lower()
    assert "No external filing" in text or "does not file" in text


def test_no_tailwind_runtime_is_required():
    for page in ["index.html", "case.html", "graph.html", "assistant.html", "sar.html", "settings.html"]:
        text = (FRONTEND / page).read_text(encoding="utf-8")
        assert "cdn.tailwindcss.com" not in text


def test_frontend_server_forces_javascript_mime():
    from serve_frontend import FrontendHandler

    assert FrontendHandler.extensions_map[".js"].startswith("text/javascript")
    assert FrontendHandler.extensions_map[".css"].startswith("text/css")
