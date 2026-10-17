"""Read-only case and transaction access over the pipeline's Parquet outputs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime, time, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

import config
from workflow_store import WorkflowStore

_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_CASE_RE = re.compile(r"^CASE-([A-Za-z0-9]+)-(\d{8})$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CUSTOMER_RE = re.compile(r"\*\*Customer\*\*:\s*([A-Za-z0-9_.:\-]+)")
_GRAPH_DATE_RE = re.compile(r"\*\*Graph Date\*\*:\s*(\d{4}-\d{2}-\d{2})")


class CaseNotFound(KeyError):
    pass


class DataUnavailable(RuntimeError):
    pass


class InvalidCase(ValueError):
    pass


def normalise_account(value: Any) -> str:
    account = str(value or "").strip()
    if not account or len(account) > 64 or not _ACCOUNT_RE.match(account):
        raise InvalidCase(f"Invalid account identifier: {value!r}")
    return account


def build_case_id(account: str, graph_date: str) -> str:
    account = normalise_account(account)
    day = normalise_date(graph_date)
    return f"CASE-{account}-{day.replace('-', '')}"


def normalise_date(value: Any) -> str:
    text = str(value or "").strip()
    if not _DATE_RE.match(text):
        raise InvalidCase(f"Date must be YYYY-MM-DD: {value!r}")
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        raise InvalidCase(f"Not a real date: {value!r}")
    return text


def parse_case_id(case_id: str) -> Tuple[str, str]:
    match = _CASE_RE.match(str(case_id or "").strip())
    if not match:
        raise InvalidCase("Case id must look like CASE-<account>-YYYYMMDD")
    account, compact = match.groups()
    try:
        day = datetime.strptime(compact, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        raise InvalidCase("Case id contains an invalid date")
    return account, day


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date_cls)):
        return value.isoformat()
    return str(value)


def _money(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, 2)


def _float(value: Any, digits: int = 6) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, digits)


def _int(value: Any) -> Optional[int]:
    number = _float(value, 2)
    if number is None:
        return None
    return int(number) if float(number).is_integer() else number


def transaction_ref(
    timestamp: Any,
    sender: str,
    receiver: str,
    amount_paid: Any,
    currency: Any,
    payment_format: Any,
) -> str:
    """Deterministic provenance reference; the source has no native transaction id."""
    canonical = "|".join(
        [
            str(timestamp),
            str(sender),
            str(receiver),
            f"{float(amount_paid or 0.0):.6f}",
            str(currency or ""),
            str(payment_format or ""),
        ]
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20].upper()
    return f"TX-{digest}"


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    output_dir: Path
    risk_path: Path
    transactions_path: Optional[Path]
    summary: Dict[str, Any] = field(default_factory=dict)
    legacy_reports: Dict[Tuple[str, str], str] = field(default_factory=dict)
    created_at: Optional[str] = None


class CaseService:
    def __init__(self, workflow: WorkflowStore, output_root: Optional[str | Path] = None):
        self.workflow = workflow
        self.output_root = Path(output_root or config.OUTPUT_ROOT).resolve()
        self._lock = RLock()
        self._run_cache: Dict[str, RunInfo] = {}
        self._index_cache: Dict[Tuple[str, int, int], pd.DataFrame] = {}
        self._holder_cache: Dict[Tuple[str, str], Optional[str]] = {}

    # -- runs -------------------------------------------------------------
    def _read_summary(self, output_dir: Path) -> Dict[str, Any]:
        path = output_dir / "summary.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _read_legacy_reports(self, output_dir: Path) -> Dict[Tuple[str, str], str]:
        path = output_dir / "sar_reports.jsonl"
        if not path.is_file():
            return {}
        reports: Dict[Tuple[str, str], str] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(record.get("report") or "")
                customer = _CUSTOMER_RE.search(text)
                graph_date = _GRAPH_DATE_RE.search(text)
                if customer and graph_date:
                    reports[(customer.group(1), graph_date.group(1))] = text
        except OSError:
            pass
        return reports

    def _run_from_dir(self, output_dir: Path) -> Optional[RunInfo]:
        risk_path = output_dir / "risk_scores.parquet"
        if not risk_path.is_file():
            return None
        summary = self._read_summary(output_dir)
        tx_raw = summary.get("input_parquet")
        tx_path: Optional[Path] = None
        if tx_raw:
            candidate = Path(str(tx_raw)).expanduser()
            if not candidate.is_absolute():
                candidate = config.REPO_ROOT / candidate
            if candidate.is_file():
                tx_path = candidate.resolve()
        if tx_path is None:
            default = Path(config.DEFAULT_PARQUET_PATH)
            if default.is_file():
                tx_path = default.resolve()
        mtime = datetime.fromtimestamp(risk_path.stat().st_mtime).isoformat(timespec="seconds")
        return RunInfo(
            run_id=output_dir.name,
            output_dir=output_dir,
            risk_path=risk_path,
            transactions_path=tx_path,
            summary=summary,
            legacy_reports=self._read_legacy_reports(output_dir),
            created_at=mtime,
        )

    def list_runs(self) -> List[RunInfo]:
        if not self.output_root.is_dir():
            return []
        runs = []
        for directory in self.output_root.iterdir():
            if directory.is_dir():
                run = self._run_from_dir(directory)
                if run is not None:
                    runs.append(run)
        runs.sort(key=lambda item: item.risk_path.stat().st_mtime, reverse=True)
        return runs

    def get_run(self, run_id: Optional[str] = None) -> RunInfo:
        if run_id:
            if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", str(run_id)):
                raise InvalidCase("Invalid run id")
            with self._lock:
                cached = self._run_cache.get(run_id)
            if cached is not None:
                return cached
            directory = (self.output_root / str(run_id)).resolve()
            if self.output_root not in directory.parents and directory != self.output_root:
                raise InvalidCase("Run path escapes the output directory")
            run = self._run_from_dir(directory)
            if run is None:
                raise CaseNotFound(f"Run not found: {run_id}")
        else:
            runs = self.list_runs()
            if not runs:
                raise CaseNotFound("No completed pipeline run is available")
            run = runs[0]
        with self._lock:
            self._run_cache[run.run_id] = run
        return run

    # -- case index -------------------------------------------------------
    def _case_index(self, run: RunInfo) -> pd.DataFrame:
        stat = run.risk_path.stat()
        key = (str(run.risk_path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._index_cache.get(key)
        if cached is not None:
            return cached

        sql = """
            SELECT
                "Account" AS account_id,
                CAST("graph_date" AS DATE) AS graph_date,
                "final_risk_score" AS risk_score,
                "risk_label" AS risk_label,
                "investigation_tier" AS tier,
                "base_risk" AS base_risk,
                "graph_penalty" AS graph_penalty,
                "hop1_count" AS hop1_count,
                "hop2_count" AS hop2_count,
                "hop3_count" AS hop3_count,
                "bidir_count" AS bidir_count,
                "fan_score" AS fan_score,
                "daily_degree" AS daily_degree,
                "is_high_risk_seed" AS is_high_risk_seed,
                "received_from_high_risk" AS received_from_high_risk,
                "xgb_prob" AS xgb_prob,
                "anomaly_score" AS anomaly_score,
                "behavioral_score" AS behavioral_score,
                "meta_score" AS meta_score
            FROM read_parquet(?)
            WHERE "investigation_flag"
        """
        con = duckdb.connect()
        try:
            frame = con.execute(sql, [run.risk_path.as_posix()]).df()
        finally:
            con.close()

        if frame.empty:
            raise DataUnavailable(f"Run {run.run_id} has no investigation cases")
        frame["graph_date"] = pd.to_datetime(frame["graph_date"]).dt.strftime("%Y-%m-%d")
        frame["account_id"] = frame["account_id"].astype(str)
        frame["case_id"] = [
            build_case_id(account, day)
            for account, day in zip(frame["account_id"], frame["graph_date"])
        ]
        for column in ("hop1_count", "hop2_count", "hop3_count", "bidir_count", "daily_degree"):
            frame[column] = frame[column].apply(_int)
        for column in ("risk_score", "base_risk", "graph_penalty", "fan_score", "xgb_prob",
                       "anomaly_score", "behavioral_score", "meta_score"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["is_high_risk_seed"] = frame["is_high_risk_seed"].astype(bool)
        frame["received_from_high_risk"] = frame["received_from_high_risk"].astype(bool)
        with self._lock:
            self._index_cache[key] = frame
            if len(self._index_cache) > 4:
                self._index_cache.pop(next(iter(self._index_cache)))
        return frame

    def _case_rows(self, run: RunInfo, case_id: str) -> pd.Series:
        account, day = parse_case_id(case_id)
        frame = self._case_index(run)
        rows = frame[(frame["account_id"] == account) & (frame["graph_date"] == day)]
        if rows.empty:
            raise CaseNotFound(f"Case not found: {case_id}")
        return rows.iloc[0]

    def _case_id_for_row(self, row: pd.Series) -> str:
        return str(row["case_id"])

    # -- holder names -----------------------------------------------------
    def holder_names(self, run: RunInfo, accounts: Iterable[str]) -> Dict[str, Optional[str]]:
        wanted = sorted({str(a) for a in accounts if a})
        result: Dict[str, Optional[str]] = {}
        missing = []
        for account in wanted:
            cache_key = (run.run_id, account)
            with self._lock:
                if cache_key in self._holder_cache:
                    result[account] = self._holder_cache[cache_key]
                else:
                    missing.append(account)
        if not missing or run.transactions_path is None:
            return result

        placeholders = ", ".join(["?"] * len(missing))
        sql = f"""
            WITH names AS (
                SELECT "Account" AS account, "From_Entity_Name" AS name
                FROM read_parquet(?)
                WHERE "Account" IN ({placeholders})
                UNION ALL
                SELECT "Account_duplicated_0" AS account, "To_Entity_Name" AS name
                FROM read_parquet(?)
                WHERE "Account_duplicated_0" IN ({placeholders})
            )
            SELECT account, min(name) AS name
            FROM names
            WHERE name IS NOT NULL
            GROUP BY account
        """
        params = [run.transactions_path.as_posix(), *missing,
                  run.transactions_path.as_posix(), *missing]
        con = duckdb.connect()
        try:
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()
        found = {str(row[0]): str(row[1]) for row in rows if row[1] is not None}
        with self._lock:
            for account in missing:
                value = found.get(account)
                self._holder_cache[(run.run_id, account)] = value
                result[account] = value
        return result

    def holder_name(self, run: RunInfo, account: str) -> Optional[str]:
        return self.holder_names(run, [account]).get(account)

    # -- case queries -----------------------------------------------------
    def list_cases(
        self,
        *,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        tier: Optional[str] = None,
        risk_label: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        account: Optional[str] = None,
        q: Optional[str] = None,
        sort: str = "score_desc",
        page: int = 1,
        page_size: int = config.CASE_PAGE_SIZE_DEFAULT,
    ) -> Dict[str, Any]:
        run = self.get_run(run_id)
        frame = self._case_index(run).copy()
        states = self.workflow.list_case_states(run.run_id)

        frame["status"] = frame["case_id"].map(
            lambda value: str(states.get(value, {}).get("status", "NEW"))
        )
        frame["assigned_to"] = frame["case_id"].map(
            lambda value: states.get(value, {}).get("assigned_to")
        )

        base = frame
        if tier:
            base = base[base["tier"] == str(tier).strip().upper()]
        if risk_label:
            base = base[base["risk_label"].str.lower() == str(risk_label).strip().lower()]
        if date_from:
            base = base[base["graph_date"] >= normalise_date(date_from)]
        if date_to:
            base = base[base["graph_date"] <= normalise_date(date_to)]
        if min_score is not None:
            base = base[base["risk_score"] >= float(min_score)]
        if max_score is not None:
            base = base[base["risk_score"] <= float(max_score)]
        if account:
            needle = str(account).strip().upper()
            base = base[base["account_id"].str.upper().str.contains(needle, regex=False)]
        if q:
            needle = str(q).strip().upper()
            base = base[
                base["account_id"].str.upper().str.contains(needle, regex=False)
                | base["case_id"].str.upper().str.contains(needle, regex=False)
            ]

        status_counts = {
            name: int((base["status"] == name).sum())
            for name in ("NEW", "IN_REVIEW", "WAITING_INFO", "ESCALATED", "CLOSED")
        }
        if status:
            base = base[base["status"] == str(status).strip().upper()]

        sort_map = {
            "score_desc": ("risk_score", False),
            "score_asc": ("risk_score", True),
            "date_desc": ("graph_date", False),
            "date_asc": ("graph_date", True),
            "account": ("account_id", True),
        }
        column, ascending = sort_map.get(sort, sort_map["score_desc"])
        base = base.sort_values(by=[column, "account_id"], ascending=[ascending, True])

        total = int(len(base))
        page_size = max(1, min(int(page_size), config.CASE_PAGE_SIZE_MAX))
        page = max(1, int(page))
        start = (page - 1) * page_size
        page_rows = base.iloc[start:start + page_size]
        holders = self.holder_names(run, page_rows["account_id"].tolist())
        available_dates = sorted(frame["graph_date"].dropna().astype(str).unique().tolist())

        items = []
        for _, row in page_rows.iterrows():
            items.append(self._summary_dict(run, row, holders.get(row["account_id"])))
        return {
            "run_id": run.run_id,
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "page_count": max(1, (total + page_size - 1) // page_size) if total else 0,
            "status_counts": status_counts,
            "available_dates": available_dates,
        }

    def _summary_dict(
        self, run: RunInfo, row: pd.Series, holder: Optional[str]
    ) -> Dict[str, Any]:
        return {
            "run_id": run.run_id,
            "case_id": str(row["case_id"]),
            "account_id": str(row["account_id"]),
            "graph_date": str(row["graph_date"]),
            "holder_name": holder,
            "risk_score": float(row["risk_score"]),
            "risk_label": str(row["risk_label"]),
            "tier": str(row["tier"]),
            "status": str(row.get("status", "NEW")),
            "assigned_to": (
                None
                if row.get("assigned_to") is None or pd.isna(row.get("assigned_to"))
                else str(row.get("assigned_to"))
            ),
            "base_risk": _float(row.get("base_risk")),
            "graph_penalty": _float(row.get("graph_penalty")),
            "hop1_count": _int(row.get("hop1_count")),
            "hop2_count": _int(row.get("hop2_count")),
            "bidir_count": _int(row.get("bidir_count")),
            "fan_score": _float(row.get("fan_score")),
            "daily_degree": _int(row.get("daily_degree")),
            "is_high_risk_seed": bool(row.get("is_high_risk_seed", False)),
            "received_from_high_risk": bool(row.get("received_from_high_risk", False)),
        }

    def get_case(self, case_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        run = self.get_run(run_id)
        row = self._case_rows(run, case_id)
        holder = self.holder_name(run, str(row["account_id"]))
        state = self.workflow.get_case_state(run.run_id, case_id)
        drafts = self.workflow.list_sar_drafts(run.run_id, case_id)
        notes = self.workflow.list_notes(run.run_id, case_id)
        summary = self._transaction_summary(run, str(row["account_id"]), str(row["graph_date"]))
        legacy = run.legacy_reports.get((str(row["account_id"]), str(row["graph_date"])))

        detail = self._summary_dict(run, row, holder)
        detail.update(
            {
                "status": state.get("status", "NEW"),
                "assigned_to": state.get("assigned_to"),
                "thresholds": {
                    "investigation": round(float(config.INVESTIGATION_THRESHOLD), 6),
                    "medium_risk": round(float(config.MEDIUM_RISK_THRESHOLD), 6),
                    "tier_1_high": round(float(config.TIER_1_THRESHOLD), 6),
                    "tier_2_medium": round(float(config.TIER_2_THRESHOLD), 6),
                },
                "model_scores": {
                    "xgboost": _float(row.get("xgb_prob")),
                    "anomaly": _float(row.get("anomaly_score")),
                    "behavioral": _float(row.get("behavioral_score")),
                    "meta": _float(row.get("meta_score")),
                },
                "graph_features": {
                    "hop1_count": _int(row.get("hop1_count")),
                    "hop2_count": _int(row.get("hop2_count")),
                    "hop3_count": _int(row.get("hop3_count")),
                    "bidir_count": _int(row.get("bidir_count")),
                    "fan_score": _float(row.get("fan_score")),
                    "daily_degree": _int(row.get("daily_degree")),
                    "is_high_risk_seed": bool(row.get("is_high_risk_seed", False)),
                    "received_from_high_risk": bool(row.get("received_from_high_risk", False)),
                },
                "transaction_summary": summary,
                "sar": {
                    "available": bool(drafts),
                    "draft_count": len(drafts),
                    "latest": (
                        {
                            "id": drafts[0]["id"],
                            "version": drafts[0]["version"],
                            "status": drafts[0]["status"],
                            "updated_at": drafts[0]["updated_at"],
                            "approved_at": drafts[0].get("approved_at"),
                        }
                        if drafts
                        else None
                    ),
                    "legacy_available": legacy is not None,
                },
                "legacy_report": legacy,
                "notes_count": len(notes),
                "created_at": run.created_at,
            }
        )
        return detail

    # -- transactions -----------------------------------------------------
    def _tx_path(self, run: RunInfo) -> Path:
        if run.transactions_path is None or not run.transactions_path.is_file():
            raise DataUnavailable(
                "The transactions parquet for this run is not available. "
                "Set FINDETECT_DEFAULT_PARQUET or re-run the pipeline."
            )
        return run.transactions_path

    def _window(
        self,
        day: str,
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
    ) -> Tuple[datetime, datetime]:
        start = datetime.combine(datetime.strptime(day, "%Y-%m-%d").date(), time.min)
        end = datetime.combine(datetime.strptime(day, "%Y-%m-%d").date(), time.max)
        if from_value:
            start = _parse_datetime(from_value, end_of_day=False)
        if to_value:
            end = _parse_datetime(to_value, end_of_day=True)
        if start > end:
            raise InvalidCase("from must be before to")
        if end - start > timedelta(days=config.GRAPH_MAX_WINDOW_DAYS):
            raise InvalidCase("Time window is too large")
        return start, end

    def _tx_where(
        self,
        account: str,
        start: datetime,
        end: datetime,
        *,
        direction: str = "both",
        counterparty: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
    ) -> Tuple[str, List[Any]]:
        clauses = [
            '("Account" = ? OR "Account_duplicated_0" = ?)',
            '"Timestamp" >= ?',
            '"Timestamp" <= ?',
        ]
        params: List[Any] = [account, account, start, end]
        direction = (direction or "both").lower()
        if direction == "in":
            clauses.append('"Account_duplicated_0" = ?')
            params.append(account)
        elif direction == "out":
            clauses.append('"Account" = ?')
            params.append(account)
        elif direction not in ("both", "all"):
            raise InvalidCase("direction must be both, in or out")

        if counterparty:
            other = normalise_account(counterparty)
            clauses.append('("Account" = ? OR "Account_duplicated_0" = ?)')
            params.extend([other, other])
        amount_expr = 'CASE WHEN "Account_duplicated_0" = ? THEN "Amount Received" ELSE "Amount Paid" END'
        if min_amount is not None:
            clauses.append(f"{amount_expr} >= ?")
            params.extend([account, float(min_amount)])
        if max_amount is not None:
            clauses.append(f"{amount_expr} <= ?")
            params.extend([account, float(max_amount)])
        return " AND ".join(clauses), params

    def _transaction_summary(self, run: RunInfo, account: str, day: str) -> Dict[str, Any]:
        path = self._tx_path(run)
        start, end = self._window(day)
        con = duckdb.connect()
        try:
            sql = """
                SELECT
                    CASE
                        WHEN "Account" = ? AND "Account_duplicated_0" = ? THEN 'self'
                        WHEN "Account_duplicated_0" = ? THEN 'in'
                        ELSE 'out'
                    END AS direction,
                    CASE
                        WHEN "Account_duplicated_0" = ? THEN "Receiving Currency"
                        ELSE "Payment Currency"
                    END AS currency,
                    count(*) AS n,
                    sum(CASE
                        WHEN "Account_duplicated_0" = ? THEN "Amount Received"
                        ELSE "Amount Paid"
                    END) AS total
                FROM read_parquet(?)
                WHERE ("Account" = ? OR "Account_duplicated_0" = ?)
                  AND "Timestamp" >= ? AND "Timestamp" <= ?
                GROUP BY 1, 2
                ORDER BY 1, 2
            """
            rows = con.execute(
                sql,
                [account, account, account, account, account, path.as_posix(),
                 account, account, start, end],
            ).fetchall()
            max_row = con.execute(
                """
                SELECT max("Amount Paid") FROM read_parquet(?)
                WHERE ("Account" = ? OR "Account_duplicated_0" = ?)
                  AND "Timestamp" >= ? AND "Timestamp" <= ?
                """,
                [path.as_posix(), account, account, start, end],
            ).fetchone()
        finally:
            con.close()

        aggregates = [
            {
                "direction": str(row[0]),
                "currency": str(row[1]) if row[1] is not None else None,
                "count": int(row[2]),
                "total": _money(row[3]) or 0.0,
            }
            for row in rows
        ]
        currencies = sorted({item["currency"] for item in aggregates if item["currency"]})
        return {
            "date": day,
            "transaction_count": sum(item["count"] for item in aggregates),
            "currencies": currencies,
            "aggregates": aggregates,
            "largest_transaction": _money(max_row[0]) if max_row and max_row[0] is not None else None,
            "has_multiple_currencies": len(currencies) > 1,
        }

    def list_transactions(
        self,
        case_id: str,
        *,
        run_id: Optional[str] = None,
        direction: str = "both",
        counterparty: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
        page: int = 1,
        page_size: int = 25,
    ) -> Dict[str, Any]:
        run = self.get_run(run_id)
        row = self._case_rows(run, case_id)
        account = str(row["account_id"])
        day = str(row["graph_date"])
        path = self._tx_path(run)
        start, end = self._window(day, from_value, to_value)
        page_size = max(1, min(int(page_size), config.CASE_PAGE_SIZE_MAX))
        page = max(1, int(page))
        where, params = self._tx_where(
            account,
            start,
            end,
            direction=direction,
            counterparty=counterparty,
            min_amount=min_amount,
            max_amount=max_amount,
        )

        con = duckdb.connect()
        try:
            total = int(
                con.execute(
                    f'SELECT count(*) FROM read_parquet(?) WHERE {where}',
                    [path.as_posix(), *params],
                ).fetchone()[0]
            )
            rows = con.execute(
                f"""
                SELECT
                    "Timestamp" AS timestamp,
                    "Account" AS sender,
                    "Account_duplicated_0" AS receiver,
                    "Amount Paid" AS amount_paid,
                    "Amount Received" AS amount_received,
                    "Payment Currency" AS payment_currency,
                    "Receiving Currency" AS receiving_currency,
                    "Payment Format" AS payment_format,
                    "From Bank" AS from_bank,
                    "To Bank" AS to_bank,
                    "From_Entity_ID" AS from_entity_id,
                    "From_Entity_Name" AS from_entity_name,
                    "To_Entity_ID" AS to_entity_id,
                    "To_Entity_Name" AS to_entity_name
                FROM read_parquet(?)
                WHERE {where}
                ORDER BY "Timestamp" DESC
                LIMIT ? OFFSET ?
                """,
                [path.as_posix(), *params, page_size, (page - 1) * page_size],
            ).df()
            aggregate_rows = con.execute(
                f"""
                SELECT
                    CASE
                        WHEN "Account" = ? AND "Account_duplicated_0" = ? THEN 'self'
                        WHEN "Account_duplicated_0" = ? THEN 'in'
                        ELSE 'out'
                    END AS direction,
                    CASE
                        WHEN "Account_duplicated_0" = ? THEN "Receiving Currency"
                        ELSE "Payment Currency"
                    END AS currency,
                    count(*) AS n,
                    sum(CASE
                        WHEN "Account_duplicated_0" = ? THEN "Amount Received"
                        ELSE "Amount Paid"
                    END) AS total
                FROM read_parquet(?)
                WHERE {where}
                GROUP BY 1, 2 ORDER BY 1, 2
                """,
                [account, account, account, account, account, path.as_posix(), *params],
            ).fetchall()
        finally:
            con.close()

        items = []
        for _, item in rows.iterrows():
            sender = str(item["sender"])
            receiver = str(item["receiver"])
            if sender == account and receiver == account:
                side = "self"
                counterparty_value = sender
                amount = float(item["amount_paid"] or 0.0)
                currency = item["payment_currency"]
            elif receiver == account:
                side = "in"
                counterparty_value = sender
                amount = float(item["amount_received"] or 0.0)
                currency = item["receiving_currency"]
            else:
                side = "out"
                counterparty_value = receiver
                amount = float(item["amount_paid"] or 0.0)
                currency = item["payment_currency"]
            items.append(
                {
                    "transaction_ref": transaction_ref(
                        item["timestamp"], sender, receiver, item["amount_paid"],
                        item["payment_currency"], item["payment_format"],
                    ),
                    "timestamp": (
                        item["timestamp"].isoformat(timespec="seconds")
                        if pd.notna(item["timestamp"]) else ""
                    ),
                    "direction": side,
                    "from_account": sender,
                    "to_account": receiver,
                    "from_entity_name": (
                        None if pd.isna(item["from_entity_name"]) else str(item["from_entity_name"])
                    ),
                    "to_entity_name": (
                        None if pd.isna(item["to_entity_name"]) else str(item["to_entity_name"])
                    ),
                    "counterparty": counterparty_value,
                    "amount": _money(amount) or 0.0,
                    "currency": None if pd.isna(currency) else str(currency),
                    "amount_paid": _money(item["amount_paid"]) or 0.0,
                    "amount_received": _money(item["amount_received"]) or 0.0,
                    "payment_currency": (
                        None if pd.isna(item["payment_currency"]) else str(item["payment_currency"])
                    ),
                    "receiving_currency": (
                        None if pd.isna(item["receiving_currency"]) else str(item["receiving_currency"])
                    ),
                    "payment_format": (
                        None if pd.isna(item["payment_format"]) else str(item["payment_format"])
                    ),
                    "from_bank": _int(item["from_bank"]),
                    "to_bank": _int(item["to_bank"]),
                    "source_file": path.name,
                }
            )
        aggregates = [
            {
                "direction": str(row[0]),
                "currency": None if row[1] is None else str(row[1]),
                "count": int(row[2]),
                "total": _money(row[3]) or 0.0,
            }
            for row in aggregate_rows
        ]
        return {
            "run_id": run.run_id,
            "case_id": case_id,
            "account_id": account,
            "graph_date": day,
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "page_count": max(1, (total + page_size - 1) // page_size) if total else 0,
            "aggregates": aggregates,
            "filters": {
                "direction": direction,
                "counterparty": counterparty,
                "min_amount": min_amount,
                "max_amount": max_amount,
                "from": start.isoformat(timespec="seconds"),
                "to": end.isoformat(timespec="seconds"),
            },
        }

    # -- graph edge access ------------------------------------------------
    def edges_for_accounts(
        self,
        run: RunInfo,
        accounts: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        limit: int,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        if not accounts:
            return []
        path = self._tx_path(run)
        accounts = [normalise_account(value) for value in accounts]
        placeholders = ", ".join(["?"] * len(accounts))
        clauses = []
        params: List[Any] = []
        if direction == "in":
            clauses.append(f'"Account_duplicated_0" IN ({placeholders})')
            params.extend(accounts)
        elif direction == "out":
            clauses.append(f'"Account" IN ({placeholders})')
            params.extend(accounts)
        else:
            clauses.append(
                f'("Account" IN ({placeholders}) OR "Account_duplicated_0" IN ({placeholders}))'
            )
            params.extend(accounts)
            params.extend(accounts)
        clauses.extend(['"Timestamp" >= ?', '"Timestamp" <= ?'])
        params.extend([start, end])
        sql = f"""
            SELECT
                "Timestamp" AS timestamp,
                "Account" AS sender,
                "Account_duplicated_0" AS receiver,
                "Amount Paid" AS amount_paid,
                "Amount Received" AS amount_received,
                "Payment Currency" AS payment_currency,
                "Receiving Currency" AS receiving_currency,
                "Payment Format" AS payment_format,
                "From Bank" AS from_bank,
                "To Bank" AS to_bank,
                "From_Entity_ID" AS from_entity_id,
                "From_Entity_Name" AS from_entity_name,
                "To_Entity_ID" AS to_entity_id,
                "To_Entity_Name" AS to_entity_name
            FROM read_parquet(?)
            WHERE {' AND '.join(clauses)}
            ORDER BY "Timestamp"
            LIMIT ?
        """
        params_full = [path.as_posix(), *params, max(1, int(limit))]
        con = duckdb.connect()
        try:
            rows = con.execute(sql, params_full).df()
        finally:
            con.close()
        edges = []
        for _, item in rows.iterrows():
            sender = str(item["sender"])
            receiver = str(item["receiver"])
            edges.append(
                {
                    "id": transaction_ref(
                        item["timestamp"], sender, receiver, item["amount_paid"],
                        item["payment_currency"], item["payment_format"],
                    ),
                    "transaction_ref": transaction_ref(
                        item["timestamp"], sender, receiver, item["amount_paid"],
                        item["payment_currency"], item["payment_format"],
                    ),
                    "source": sender,
                    "target": receiver,
                    "timestamp": item["timestamp"].isoformat(timespec="seconds"),
                    "amount": _money(item["amount_paid"]) or 0.0,
                    "currency": (
                        None if pd.isna(item["payment_currency"]) else str(item["payment_currency"])
                    ),
                    "payment_format": (
                        None if pd.isna(item["payment_format"]) else str(item["payment_format"])
                    ),
                    "from_entity_id": (
                        None if pd.isna(item["from_entity_id"]) else str(item["from_entity_id"])
                    ),
                    "from_entity_name": (
                        None if pd.isna(item["from_entity_name"]) else str(item["from_entity_name"])
                    ),
                    "to_entity_id": (
                        None if pd.isna(item["to_entity_id"]) else str(item["to_entity_id"])
                    ),
                    "to_entity_name": (
                        None if pd.isna(item["to_entity_name"]) else str(item["to_entity_name"])
                    ),
                    "from_bank": _int(item["from_bank"]),
                    "to_bank": _int(item["to_bank"]),
                    "source_file": path.name,
                }
            )
        return edges

    # -- misc -------------------------------------------------------------
    def legacy_report(self, run: RunInfo, case_id: str) -> Optional[str]:
        account, day = parse_case_id(case_id)
        return run.legacy_reports.get((account, day))

    def transaction_counts(self, run: RunInfo, case_ids: Sequence[str]) -> Dict[str, int]:
        """Transaction counts for a small page of cases."""
        if not case_ids:
            return {}
        path = self._tx_path(run)
        pairs = []
        for case_id in case_ids:
            account, day = parse_case_id(case_id)
            pairs.append((account, day))
        values = ", ".join(["(?, ?)"] * len(pairs))
        params: List[Any] = []
        for account, day in pairs:
            params.extend([account, day])
        sql = f"""
            WITH keys(account, d) AS (VALUES {values})
            SELECT k.account, CAST(t."Timestamp" AS DATE) AS d, count(*) AS n
            FROM read_parquet(?) t
            JOIN keys k ON (
                (t."Account" = k.account OR t."Account_duplicated_0" = k.account)
                AND CAST(t."Timestamp" AS DATE) = CAST(k.d AS DATE)
            )
            GROUP BY 1, 2
        """
        con = duckdb.connect()
        try:
            rows = con.execute(sql, [*params, path.as_posix()]).fetchall()
        finally:
            con.close()
        counts = {}
        for account, day, count in rows:
            counts[build_case_id(str(account), str(day))] = int(count)
        return counts


def _parse_datetime(value: str, *, end_of_day: bool) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise InvalidCase("Empty datetime")
    try:
        if len(text) == 10 and _DATE_RE.match(text):
            day = datetime.strptime(text, "%Y-%m-%d").date()
            return datetime.combine(day, time.max if end_of_day else time.min)
        return datetime.fromisoformat(text)
    except ValueError:
        raise InvalidCase(f"Invalid datetime: {value!r}")
