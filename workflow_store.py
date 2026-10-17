"""SQLite-backed workflow state for cases, notes, SAR drafts and audit events.

The AML pipeline outputs remain immutable Parquet artifacts. This store only
holds investigator workflow state. It is deliberately small, stdlib-only, and
local: SQLite in WAL mode with append-only audit events.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional

import config

CASE_STATUSES = ("NEW", "IN_REVIEW", "WAITING_INFO", "ESCALATED", "CLOSED")
NOTE_TYPES = ("NOTE", "FINDING", "HYPOTHESIS")
SAR_STATUSES = ("DRAFT", "IN_REVIEW", "APPROVED", "SUPERSEDED")
AUDIT_ACTIONS = {
    "CASE_VIEWED",
    "CASE_STATUS_CHANGED",
    "NOTE_CREATED",
    "NOTE_UPDATED",
    "SAR_DRAFT_CREATED",
    "SAR_DRAFT_UPDATED",
    "SAR_SUBMITTED_FOR_REVIEW",
    "SAR_APPROVED",
    "SAR_EXPORTED",
    "GRAPH_EXPLORED",
    "GRAPH_ANALYSIS_RUN",
    "AI_QUERY",
}


class WorkflowNotFound(KeyError):
    pass


class WorkflowValidation(ValueError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class WorkflowStore:
    def __init__(self, db_path: Optional[str | Path] = None):
        self.path = Path(db_path or config.WORKFLOW_DB).expanduser().resolve()
        self._lock = RLock()
        self.init_db()

    # -- setup ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schema_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS case_state (
                        run_id TEXT NOT NULL,
                        case_id TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'NEW',
                        assigned_to TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, case_id)
                    );

                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        case_id TEXT NOT NULL,
                        author TEXT NOT NULL,
                        note_type TEXT NOT NULL DEFAULT 'NOTE',
                        body TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1
                    );
                    CREATE INDEX IF NOT EXISTS idx_notes_case
                        ON notes (run_id, case_id, updated_at DESC);

                    CREATE TABLE IF NOT EXISTS note_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        note_id INTEGER NOT NULL,
                        version INTEGER NOT NULL,
                        body TEXT NOT NULL,
                        note_type TEXT NOT NULL,
                        author TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE (note_id, version),
                        FOREIGN KEY (note_id) REFERENCES notes (id)
                    );

                    CREATE TABLE IF NOT EXISTS sar_drafts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        case_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'DRAFT',
                        content_json TEXT NOT NULL,
                        missing_fields_json TEXT NOT NULL DEFAULT '[]',
                        evidence_json TEXT NOT NULL DEFAULT '[]',
                        created_by TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        approved_by TEXT,
                        approved_at TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_sar_case
                        ON sar_drafts (run_id, case_id, version DESC);

                    CREATE TABLE IF NOT EXISTS audit_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        case_id TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        action TEXT NOT NULL,
                        outcome TEXT NOT NULL DEFAULT 'SUCCESS',
                        details_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_audit_case
                        ON audit_events (run_id, case_id, created_at DESC);

                    CREATE TRIGGER IF NOT EXISTS audit_no_update
                    BEFORE UPDATE ON audit_events
                    BEGIN
                        SELECT RAISE(ABORT, 'audit_events is append-only');
                    END;

                    CREATE TRIGGER IF NOT EXISTS audit_no_delete
                    BEFORE DELETE ON audit_events
                    BEGIN
                        SELECT RAISE(ABORT, 'audit_events is append-only');
                    END;
                    """
                )
                conn.execute(
                    "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', '1')"
                )
                conn.commit()
            finally:
                conn.close()

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _case_status(status: str) -> str:
        value = str(status or "").strip().upper()
        if value not in CASE_STATUSES:
            raise WorkflowValidation(f"Unknown case status: {status!r}")
        return value

    @staticmethod
    def _note_type(note_type: str) -> str:
        value = str(note_type or "NOTE").strip().upper()
        if value not in NOTE_TYPES:
            raise WorkflowValidation(f"Unknown note type: {note_type!r}")
        return value

    @staticmethod
    def _body(body: str) -> str:
        text = str(body or "").strip()
        if not text:
            raise WorkflowValidation("Note body must not be empty.")
        if len(text) > 20_000:
            raise WorkflowValidation("Note body is too long (max 20,000 characters).")
        return text

    def _audit(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        case_id: str,
        actor: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "SUCCESS",
    ) -> None:
        if action not in AUDIT_ACTIONS:
            raise WorkflowValidation(f"Unknown audit action: {action!r}")
        conn.execute(
            "INSERT INTO audit_events "
            "(run_id, case_id, actor, action, outcome, details_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, case_id, actor, action, outcome, _json(details or {}), utcnow()),
        )

    # -- case state -------------------------------------------------------
    def get_case_state(self, run_id: str, case_id: str) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM case_state WHERE run_id = ? AND case_id = ?",
                    (run_id, case_id),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return {
                "run_id": run_id,
                "case_id": case_id,
                "status": "NEW",
                "assigned_to": None,
                "updated_at": None,
            }
        return dict(row)

    def list_case_states(self, run_id: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM case_state WHERE run_id = ?", (run_id,)
                ).fetchall()
            finally:
                conn.close()
        return {row["case_id"]: dict(row) for row in rows}

    def set_case_status(
        self,
        run_id: str,
        case_id: str,
        status: str,
        actor: str,
        assigned_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        new_status = self._case_status(status)
        with self._lock:
            conn = self._connect()
            try:
                previous = conn.execute(
                    "SELECT * FROM case_state WHERE run_id = ? AND case_id = ?",
                    (run_id, case_id),
                ).fetchone()
                now = utcnow()
                conn.execute(
                    """
                    INSERT INTO case_state
                        (run_id, case_id, status, assigned_to, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, case_id) DO UPDATE SET
                        status = excluded.status,
                        assigned_to = COALESCE(excluded.assigned_to, case_state.assigned_to),
                        updated_at = excluded.updated_at
                    """,
                    (run_id, case_id, new_status, assigned_to, now),
                )
                if previous is None or previous["status"] != new_status:
                    self._audit(
                        conn,
                        run_id=run_id,
                        case_id=case_id,
                        actor=actor,
                        action="CASE_STATUS_CHANGED",
                        details={
                            "from": previous["status"] if previous else None,
                            "to": new_status,
                        },
                    )
                conn.commit()
            finally:
                conn.close()
        return self.get_case_state(run_id, case_id)

    # -- notes ------------------------------------------------------------
    @staticmethod
    def _note_dict(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        item["note_type"] = str(item.get("note_type") or "NOTE")
        return item

    def create_note(
        self,
        run_id: str,
        case_id: str,
        author: str,
        body: str,
        note_type: str = "NOTE",
    ) -> Dict[str, Any]:
        text = self._body(body)
        kind = self._note_type(note_type)
        now = utcnow()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "INSERT INTO notes "
                    "(run_id, case_id, author, note_type, body, created_at, updated_at, version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                    (run_id, case_id, author, kind, text, now, now),
                )
                note_id = int(cursor.lastrowid)
                conn.execute(
                    "INSERT INTO note_versions "
                    "(note_id, version, body, note_type, author, created_at) "
                    "VALUES (?, 1, ?, ?, ?, ?)",
                    (note_id, text, kind, author, now),
                )
                self._audit(
                    conn,
                    run_id=run_id,
                    case_id=case_id,
                    actor=author,
                    action="NOTE_CREATED",
                    details={"note_id": note_id, "version": 1, "note_type": kind},
                )
                conn.commit()
                row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
            finally:
                conn.close()
        return self._note_dict(row)

    def update_note(
        self,
        note_id: int,
        author: str,
        body: str,
        note_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        text = self._body(body)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
                if row is None:
                    raise WorkflowNotFound(f"Note {note_id} not found")
                kind = self._note_type(note_type or row["note_type"])
                version = int(row["version"]) + 1
                now = utcnow()
                conn.execute(
                    "UPDATE notes SET body = ?, note_type = ?, updated_at = ?, version = ? "
                    "WHERE id = ?",
                    (text, kind, now, version, note_id),
                )
                conn.execute(
                    "INSERT INTO note_versions "
                    "(note_id, version, body, note_type, author, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (note_id, version, text, kind, author, now),
                )
                self._audit(
                    conn,
                    run_id=row["run_id"],
                    case_id=row["case_id"],
                    actor=author,
                    action="NOTE_UPDATED",
                    details={"note_id": note_id, "version": version, "note_type": kind},
                )
                conn.commit()
                updated = conn.execute(
                    "SELECT * FROM notes WHERE id = ?", (note_id,)
                ).fetchone()
            finally:
                conn.close()
        return self._note_dict(updated)

    def list_notes(self, run_id: str, case_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM notes WHERE run_id = ? AND case_id = ? "
                    "ORDER BY updated_at DESC, id DESC",
                    (run_id, case_id),
                ).fetchall()
            finally:
                conn.close()
        return [self._note_dict(row) for row in rows]

    def get_note(self, note_id: int) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
            finally:
                conn.close()
        if row is None:
            raise WorkflowNotFound(f"Note {note_id} not found")
        return self._note_dict(row)

    # -- SAR drafts -------------------------------------------------------
    @staticmethod
    def _draft_dict(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        item["content"] = json.loads(item.pop("content_json") or "{}")
        item["missing_fields"] = json.loads(item.pop("missing_fields_json") or "[]")
        item["evidence"] = json.loads(item.pop("evidence_json") or "[]")
        return item

    def create_sar_draft(
        self,
        run_id: str,
        case_id: str,
        content: Dict[str, Any],
        missing_fields: Iterable[str],
        evidence: Iterable[Dict[str, Any]],
        actor: str,
    ) -> Dict[str, Any]:
        now = utcnow()
        missing = list(dict.fromkeys(str(item) for item in missing_fields))
        evidence_list = list(evidence)
        with self._lock:
            conn = self._connect()
            try:
                version = int(
                    conn.execute(
                        "SELECT COALESCE(MAX(version), 0) FROM sar_drafts "
                        "WHERE run_id = ? AND case_id = ?",
                        (run_id, case_id),
                    ).fetchone()[0]
                ) + 1
                cursor = conn.execute(
                    "INSERT INTO sar_drafts "
                    "(run_id, case_id, version, status, content_json, missing_fields_json, "
                    "evidence_json, created_by, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'DRAFT', ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        case_id,
                        version,
                        _json(content),
                        _json(missing),
                        _json(evidence_list),
                        actor,
                        now,
                        now,
                    ),
                )
                draft_id = int(cursor.lastrowid)
                self._audit(
                    conn,
                    run_id=run_id,
                    case_id=case_id,
                    actor=actor,
                    action="SAR_DRAFT_CREATED",
                    details={"draft_id": draft_id, "version": version},
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM sar_drafts WHERE id = ?", (draft_id,)
                ).fetchone()
            finally:
                conn.close()
        return self._draft_dict(row)

    def update_sar_draft(
        self,
        draft_id: int,
        actor: str,
        content: Dict[str, Any],
        missing_fields: Iterable[str],
        evidence: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        now = utcnow()
        missing = list(dict.fromkeys(str(item) for item in missing_fields))
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM sar_drafts WHERE id = ?", (draft_id,)
                ).fetchone()
                if row is None:
                    raise WorkflowNotFound(f"SAR draft {draft_id} not found")
                version = int(row["version"]) + 1
                conn.execute(
                    "UPDATE sar_drafts SET version = ?, status = 'DRAFT', "
                    "content_json = ?, missing_fields_json = ?, evidence_json = ?, "
                    "updated_at = ?, approved_by = NULL, approved_at = NULL "
                    "WHERE id = ?",
                    (
                        version,
                        _json(content),
                        _json(missing),
                        _json(list(evidence)),
                        now,
                        draft_id,
                    ),
                )
                self._audit(
                    conn,
                    run_id=row["run_id"],
                    case_id=row["case_id"],
                    actor=actor,
                    action="SAR_DRAFT_UPDATED",
                    details={"draft_id": draft_id, "version": version},
                )
                conn.commit()
                updated = conn.execute(
                    "SELECT * FROM sar_drafts WHERE id = ?", (draft_id,)
                ).fetchone()
            finally:
                conn.close()
        return self._draft_dict(updated)

    def submit_sar_for_review(self, draft_id: int, actor: str) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM sar_drafts WHERE id = ?", (draft_id,)
                ).fetchone()
                if row is None:
                    raise WorkflowNotFound(f"SAR draft {draft_id} not found")
                if row["status"] not in ("DRAFT", "IN_REVIEW"):
                    raise WorkflowValidation(
                        f"Draft {draft_id} cannot be submitted from {row['status']}."
                    )
                conn.execute(
                    "UPDATE sar_drafts SET status = 'IN_REVIEW', updated_at = ? WHERE id = ?",
                    (utcnow(), draft_id),
                )
                self._audit(
                    conn,
                    run_id=row["run_id"],
                    case_id=row["case_id"],
                    actor=actor,
                    action="SAR_SUBMITTED_FOR_REVIEW",
                    details={"draft_id": draft_id},
                )
                conn.commit()
                updated = conn.execute(
                    "SELECT * FROM sar_drafts WHERE id = ?", (draft_id,)
                ).fetchone()
            finally:
                conn.close()
        return self._draft_dict(updated)

    def approve_sar_draft(
        self,
        draft_id: int,
        actor: str,
        acknowledge_missing: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM sar_drafts WHERE id = ?", (draft_id,)
                ).fetchone()
                if row is None:
                    raise WorkflowNotFound(f"SAR draft {draft_id} not found")
                if row["status"] != "IN_REVIEW":
                    raise WorkflowValidation(
                        "The draft must be submitted for review before it can be approved."
                    )
                missing = json.loads(row["missing_fields_json"] or "[]")
                if missing and not acknowledge_missing:
                    raise WorkflowValidation(
                        "This draft has missing fields. Acknowledge them before approval."
                    )
                now = utcnow()
                conn.execute(
                    "UPDATE sar_drafts SET status = 'APPROVED', approved_by = ?, "
                    "approved_at = ?, updated_at = ? WHERE id = ?",
                    (actor, now, now, draft_id),
                )
                self._audit(
                    conn,
                    run_id=row["run_id"],
                    case_id=row["case_id"],
                    actor=actor,
                    action="SAR_APPROVED",
                    details={
                        "draft_id": draft_id,
                        "acknowledged_missing": bool(missing),
                    },
                )
                conn.commit()
                updated = conn.execute(
                    "SELECT * FROM sar_drafts WHERE id = ?", (draft_id,)
                ).fetchone()
            finally:
                conn.close()
        return self._draft_dict(updated)

    def get_sar_draft(self, draft_id: int) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM sar_drafts WHERE id = ?", (draft_id,)
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            raise WorkflowNotFound(f"SAR draft {draft_id} not found")
        return self._draft_dict(row)

    def list_sar_drafts(self, run_id: str, case_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM sar_drafts WHERE run_id = ? AND case_id = ? "
                    "ORDER BY version DESC, id DESC",
                    (run_id, case_id),
                ).fetchall()
            finally:
                conn.close()
        return [self._draft_dict(row) for row in rows]

    # -- audit ------------------------------------------------------------
    def add_event(
        self,
        run_id: str,
        case_id: str,
        actor: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "SUCCESS",
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                self._audit(
                    conn,
                    run_id=run_id,
                    case_id=case_id,
                    actor=actor,
                    action=action,
                    details=details,
                    outcome=outcome,
                )
                conn.commit()
            finally:
                conn.close()

    def list_audit(
        self, run_id: str, case_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM audit_events WHERE run_id = ? AND case_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (run_id, case_id, limit),
                ).fetchall()
            finally:
                conn.close()
        events = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json") or "{}")
            events.append(item)
        return events
