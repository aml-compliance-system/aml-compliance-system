"""
Agent tools for the FINDETECT AML compliance system.

This module gives an LLM investigator read-only access to the evidence the
pipeline already produced:

    get_account_scores(account_id, date=None)
        Account/day risk scores, model components, graph features and the
        operating-point interpretation (tier, investigation flag, percentile).

    get_transactions(account_id, date=None, direction="both", limit=25, ...)
        Aggregate in/out flow plus the most recent transactions and top
        counterparties for one account/day.

    get_graph_exposure(account_id, date=None)
        One-hop network exposure to original high-risk seeds, pre-computed
        multi-hop counts, mule/layering indicators and the graph penalty
        breakdown.

Design notes
------------
* Every tool returns a JSON-serialisable envelope, never raises:
      {"ok": true,  "tool": ..., "account_id": ..., "data": {...}, "meta": {...}}
      {"ok": false, "tool": ..., "error": {"code": ..., "message": ...}}
  That lets the agent feed failures back to the model instead of crashing.
* Data comes from either an in-memory pipeline run (fast path) or the saved
  run on disk (``outputs/FINDETECT_Pipeline_Output/<run>/risk_scores.parquet``
  plus the input transactions parquet). Disk access goes through DuckDB so a
  single account can be queried without loading 1.5M risk rows or 2.7M
  transactions into memory.
* Tool arguments are validated and bounded (row limits, date format, account
  format) before any query runs. All SQL is parameterised.
* Results are deliberately compact: aggregates plus a bounded number of rows,
  so they fit comfortably inside an LLM context window.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import date as _date, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config

log = config.get_logger("tools")

# ---------------------------------------------------------------------------
# Limits / constants
# ---------------------------------------------------------------------------
ACCOUNT_MAX_LEN = 64
DEFAULT_TX_LIMIT = max(1, config.AGENT_TOOL_TX_LIMIT)
MAX_TX_LIMIT = max(DEFAULT_TX_LIMIT, config.AGENT_TOOL_TX_MAX_LIMIT)
MAX_DATES_PER_ACCOUNT = 31
TOP_COUNTERPARTIES = 5
MAX_SEED_NEIGHBOURS = 5

_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,63}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_IN_DIRECTIONS = {"in", "inbound", "incoming", "received", "receive", "credit"}
_OUT_DIRECTIONS = {"out", "outbound", "outgoing", "sent", "send", "debit"}
_BOTH_DIRECTIONS = {"", "both", "all", "any"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class ToolInputError(Exception):
    """Raised for invalid tool arguments; surfaced to the model as a hint."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, _date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if pd.isna(value):
        return None
    return str(value)


def _round(value: Any, digits: int = 6) -> Optional[float]:
    """Round a numeric value, converting NaN/inf to None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, digits)


def _money(value: Any) -> Optional[float]:
    return _round(value, 2)


def _count(value: Any) -> Optional[int]:
    number = _round(value, 2)
    if number is None:
        return None
    return int(number) if float(number).is_integer() else number


def _as_bool(value: Any) -> Optional[bool]:
    """Boolean conversion that treats NaN/NaT as False (unlike bool(nan))."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _iso_date(value: Any) -> Optional[str]:
    """Normalise any date-like value to ``YYYY-MM-DD`` (or None)."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if len(text) >= 10 and _DATE_RE.match(text[:10]):
            return text[:10]
        try:
            return pd.to_datetime(text, errors="raise").strftime("%Y-%m-%d")
        except Exception:
            return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _normalise_account(raw: Any) -> str:
    if not isinstance(raw, str):
        if isinstance(raw, (int, np.integer)):
            raw = str(int(raw))
        else:
            raise ToolInputError("INVALID_ACCOUNT_ID", "account_id must be a string.")
    account = raw.strip().strip("\"'")
    if not account:
        raise ToolInputError("INVALID_ACCOUNT_ID", "account_id must not be empty.")
    if len(account) > ACCOUNT_MAX_LEN or not _ACCOUNT_RE.match(account):
        raise ToolInputError(
            "INVALID_ACCOUNT_ID",
            f"account_id must be 1-{ACCOUNT_MAX_LEN} characters of letters, digits, "
            "dot, colon, underscore or hyphen (for example 100428660).",
        )
    return account


def _normalise_date(raw: Any) -> Optional[str]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not _DATE_RE.match(text):
            raise ToolInputError("INVALID_DATE", "date must be in YYYY-MM-DD format.")
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            raise ToolInputError("INVALID_DATE", f"{text!r} is not a real calendar date.")
        return text
    normalised = _iso_date(raw)
    if normalised is None:
        raise ToolInputError("INVALID_DATE", "date must be in YYYY-MM-DD format.")
    return normalised


def _normalise_direction(raw: Any) -> str:
    if raw is None:
        return "both"
    text = str(raw).strip().lower()
    if text in _BOTH_DIRECTIONS:
        return "both"
    if text in _IN_DIRECTIONS:
        return "in"
    if text in _OUT_DIRECTIONS:
        return "out"
    raise ToolInputError(
        "INVALID_DIRECTION",
        "direction must be one of: both, in, out.",
    )


def _normalise_limit(raw: Any) -> int:
    if raw is None:
        return DEFAULT_TX_LIMIT
    if isinstance(raw, bool):
        raise ToolInputError("INVALID_LIMIT", "limit must be an integer.")
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        raise ToolInputError("INVALID_LIMIT", "limit must be an integer.")
    if limit < 1:
        raise ToolInputError("INVALID_LIMIT", "limit must be at least 1.")
    return min(limit, MAX_TX_LIMIT)


def _normalise_min_amount(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        amount = float(raw)
    except (TypeError, ValueError):
        raise ToolInputError("INVALID_MIN_AMOUNT", "min_amount must be a number.")
    if np.isnan(amount) or np.isinf(amount):
        raise ToolInputError("INVALID_MIN_AMOUNT", "min_amount must be finite.")
    return max(0.0, amount)


def _date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _tool_envelope(name: str):
    """Guarantee that a tool returns an envelope even when called directly.

    ``dispatch_tool`` also catches errors, but tools are public functions used
    by scripts and tests, so they must not raise on bad input either.
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            account = kwargs.get("account_id", args[0] if args else None)
            account_id = str(account) if account is not None else None
            try:
                return func(*args, **kwargs)
            except ToolInputError as exc:
                return _error(
                    name, exc.code, exc.message, account_id=account_id
                )
            except TypeError as exc:
                return _error(
                    name,
                    "INVALID_ARGUMENTS",
                    f"Invalid arguments for {name}: {exc}",
                    account_id=account_id,
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("Tool %s failed", name)
                return _error(
                    name,
                    "TOOL_ERROR",
                    f"{type(exc).__name__}: {exc}",
                    account_id=account_id,
                )

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Data store
# ---------------------------------------------------------------------------
class AMLDataStore:
    """Read-only access to one pipeline run's risk scores and transactions.

    The store can be backed by:

    * an in-memory ``risk_scores`` DataFrame and/or ``transactions`` DataFrame
      (used by tests and by a live pipeline run), or
    * parquet paths (used by the API after a run has been saved, and by the
      CLI against historical runs).

    If both a DataFrame and a path are provided, the DataFrame wins.
    """

    def __init__(
        self,
        *,
        risk_scores_df: Optional[pd.DataFrame] = None,
        risk_scores_path: Optional[str | Path] = None,
        transactions_df: Optional[pd.DataFrame] = None,
        transactions_path: Optional[str | Path] = None,
        run_label: Optional[str] = None,
        input_parquet: Optional[str | Path] = None,
        reports: Optional[Mapping[str, str]] = None,
    ):
        self._lock = RLock()
        self._risk_df: Optional[pd.DataFrame] = None
        self._risk_path: Optional[Path] = None
        self._risk_dates: Optional[pd.Series] = None
        self._tx_df: Optional[pd.DataFrame] = None
        self._tx_path: Optional[Path] = None
        self._tx_columns: Optional[set] = None
        self.run_label = run_label
        self.input_parquet = str(input_parquet) if input_parquet else None
        self._case_reports: Dict[str, str] = dict(reports or {})

        self._tx_cache: "OrderedDict[Tuple[str, str], pd.DataFrame]" = OrderedDict()
        self._seed_cache: Dict[str, frozenset] = {}

        self.register(
            risk_scores_df=risk_scores_df,
            risk_scores_path=risk_scores_path,
            transactions_df=transactions_df,
            transactions_path=transactions_path,
            run_label=run_label,
            input_parquet=input_parquet,
            clear_reports=False,
        )
        if reports:
            self._case_reports = dict(reports)

    # -- registration -----------------------------------------------------
    def register(
        self,
        *,
        risk_scores_df: Optional[pd.DataFrame] = None,
        risk_scores_path: Optional[str | Path] = None,
        transactions_df: Optional[pd.DataFrame] = None,
        transactions_path: Optional[str | Path] = None,
        run_label: Optional[str] = None,
        input_parquet: Optional[str | Path] = None,
        clear_reports: bool = True,
    ) -> "AMLDataStore":
        """Replace the data currently served by the store."""
        with self._lock:
            self._risk_df = risk_scores_df if risk_scores_df is not None else None
            self._risk_dates = (
                _date_series(self._risk_df["graph_date"])
                if self._risk_df is not None and "graph_date" in self._risk_df.columns
                else None
            )
            self._risk_path = Path(risk_scores_path).expanduser() if risk_scores_path else None
            self._tx_df = transactions_df if transactions_df is not None else None
            self._tx_path = Path(transactions_path).expanduser() if transactions_path else None
            self._tx_columns = None
            if run_label:
                self.run_label = run_label
            if input_parquet:
                self.input_parquet = str(input_parquet)
            if clear_reports:
                self._case_reports = {}
            self._tx_cache.clear()
            self._seed_cache.clear()
            return self

    def register_pipeline_run(
        self,
        *,
        final_state: Optional[Mapping[str, Any]] = None,
        risk_scores_df: Optional[pd.DataFrame] = None,
        risk_scores_path: Optional[str | Path] = None,
        transactions_df: Optional[pd.DataFrame] = None,
        transactions_path: Optional[str | Path] = None,
        reports: Optional[Mapping[str, str]] = None,
        run_label: Optional[str] = None,
        input_parquet: Optional[str | Path] = None,
    ) -> "AMLDataStore":
        """Register the output of ``LangGraph.app.invoke`` for agent use.

        Pass a path when the run was saved to disk (preferred: keeps memory
        flat), or a DataFrame when it was not. ``reports`` maps account id to
        the SAR draft generated for it.
        """
        if final_state:
            if risk_scores_df is None:
                candidate = final_state.get("risk_scores")
                if isinstance(candidate, pd.DataFrame):
                    risk_scores_df = candidate
        self.register(
            risk_scores_df=risk_scores_df,
            risk_scores_path=risk_scores_path,
            transactions_df=transactions_df,
            transactions_path=transactions_path,
            run_label=run_label,
            input_parquet=input_parquet,
            clear_reports=True,
        )
        with self._lock:
            self._case_reports = {str(k): str(v) for k, v in (reports or {}).items()}
        self._tx_columns = None
        return self

    # -- introspection ----------------------------------------------------
    @property
    def has_risk_data(self) -> bool:
        return self._risk_df is not None or self._risk_path is not None

    @property
    def has_transaction_data(self) -> bool:
        return self._tx_df is not None or self._tx_path is not None

    @property
    def source(self) -> str:
        if self._risk_df is not None:
            return "pipeline-cache"
        if self._risk_path is not None:
            return "parquet"
        return "unbound"

    def case_report(self, account_id: str) -> Optional[str]:
        with self._lock:
            return self._case_reports.get(str(account_id))

    def available_dates(self) -> List[str]:
        """Dates present in the risk-score data (sorted, newest last)."""
        with self._lock:
            if self._risk_df is not None and self._risk_dates is not None:
                dates = sorted(set(self._risk_dates.dropna().tolist()))
                return dates[-MAX_DATES_PER_ACCOUNT:]
            if self._risk_path is not None:
                rows = self._query(
                    'SELECT DISTINCT CAST("graph_date" AS DATE) AS d '
                    'FROM read_parquet(?) ORDER BY d',
                    [self._risk_path.as_posix()],
                )
                return [str(r[0]) for r in rows if r[0] is not None][-MAX_DATES_PER_ACCOUNT:]
        return []

    # -- low-level query helpers -----------------------------------------
    def _query(self, sql: str, params: Sequence[Any]) -> List[tuple]:
        import duckdb

        con = duckdb.connect()
        try:
            return con.execute(sql, list(params)).fetchall()
        finally:
            con.close()

    def _query_df(self, sql: str, params: Sequence[Any]) -> pd.DataFrame:
        import duckdb

        con = duckdb.connect()
        try:
            return con.execute(sql, list(params)).df()
        finally:
            con.close()

    # -- risk scores ------------------------------------------------------
    def score_rows(self, account_id: str, date: Optional[str] = None) -> pd.DataFrame:
        """Return risk-score rows for one account (optionally one date)."""
        with self._lock:
            if not self.has_risk_data:
                raise ToolInputError(
                    "NO_DATA",
                    "No pipeline run is bound. Run the pipeline first.",
                )

            if self._risk_df is not None:
                df = self._risk_df
                mask = df["Account"].astype(str) == account_id
                if date is not None and self._risk_dates is not None:
                    mask &= self._risk_dates == date
                rows = df.loc[mask].copy()
            else:
                sql = 'SELECT * FROM read_parquet(?) WHERE "Account" = ?'
                params: List[Any] = [self._risk_path.as_posix(), account_id]
                if date is not None:
                    sql += ' AND CAST("graph_date" AS DATE) = CAST(? AS DATE)'
                    params.append(date)
                rows = self._query_df(sql, params)

            if rows.empty:
                return rows
            if "graph_date" in rows.columns:
                rows["graph_date"] = _date_series(rows["graph_date"])
            if "final_risk_score" in rows.columns:
                rows = rows.sort_values(
                    ["final_risk_score"], ascending=False, kind="stable"
                )
            return rows.reset_index(drop=True)

    def risk_percentile(self, date: str, score: float) -> Optional[float]:
        """Percentage of account/day rows on ``date`` at or below ``score``."""
        try:
            with self._lock:
                if self._risk_df is not None:
                    if self._risk_dates is None:
                        return None
                    same_day = self._risk_df.loc[self._risk_dates == date]
                    if same_day.empty:
                        return None
                    values = pd.to_numeric(
                        same_day["final_risk_score"], errors="coerce"
                    ).dropna()
                    if values.empty:
                        return None
                    return round(float((values <= score).mean() * 100.0), 2)
                if self._risk_path is not None:
                    row = self._query(
                        'SELECT 100.0 * SUM(CASE WHEN "final_risk_score" <= ? '
                        'THEN 1 ELSE 0 END) / COUNT(*) '
                        'FROM read_parquet(?) '
                        'WHERE CAST("graph_date" AS DATE) = CAST(? AS DATE)',
                        [float(score), self._risk_path.as_posix(), date],
                    )
                    if row and row[0][0] is not None:
                        return round(float(row[0][0]), 2)
        except Exception:  # pragma: no cover - interpretive extra, never fatal
            log.debug("percentile lookup failed", exc_info=True)
        return None

    def seed_accounts(self, date: Optional[str]) -> frozenset:
        """Accounts that are original high-risk seeds on ``date``."""
        if not date or not self.has_risk_data:
            return frozenset()
        with self._lock:
            if date in self._seed_cache:
                return self._seed_cache[date]
            try:
                if self._risk_df is not None:
                    if self._risk_dates is None or "is_high_risk_seed" not in self._risk_df.columns:
                        seeds = frozenset()
                    else:
                        mask = (self._risk_dates == date) & self._risk_df[
                            "is_high_risk_seed"
                        ].astype(bool)
                        seeds = frozenset(self._risk_df.loc[mask, "Account"].astype(str))
                else:
                    rows = self._query(
                        'SELECT DISTINCT "Account" FROM read_parquet(?) '
                        'WHERE "is_high_risk_seed" AND '
                        'CAST("graph_date" AS DATE) = CAST(? AS DATE)',
                        [self._risk_path.as_posix(), date],
                    )
                    seeds = frozenset(str(r[0]) for r in rows if r[0] is not None)
            except Exception:
                log.warning("seed lookup failed for %s", date, exc_info=True)
                seeds = frozenset()
            self._seed_cache[date] = seeds
            return seeds

    # -- transactions -----------------------------------------------------
    def _tx_select_columns(self) -> List[str]:
        """Columns available in the transactions parquet (cached)."""
        if self._tx_columns is not None:
            return sorted(self._tx_columns)
        assert self._tx_path is not None
        rows = self._query(
            'DESCRIBE SELECT * FROM read_parquet(?)',
            [self._tx_path.as_posix()],
        )
        self._tx_columns = {str(r[0]) for r in rows}
        return sorted(self._tx_columns)

    def _tx_frame(self, account_id: str, date: Optional[str]) -> pd.DataFrame:
        """Transactions where ``account_id`` is sender or receiver."""
        cache_key = (account_id, date or "")
        with self._lock:
            cached = self._tx_cache.get(cache_key)
            if cached is not None:
                self._tx_cache.move_to_end(cache_key)
                return cached

            if self._tx_df is not None:
                raw = self._tx_df
                if not {"Account", "Account_duplicated_0"}.issubset(raw.columns):
                    raise ToolInputError(
                        "NO_DATA", "Transactions data is missing required columns."
                    )
                mask = (raw["Account"].astype(str) == account_id) | (
                    raw["Account_duplicated_0"].astype(str) == account_id
                )
                if date is not None and "Timestamp" in raw.columns:
                    mask &= _date_series(raw["Timestamp"]) == date
                frame = raw.loc[mask].copy()
            else:
                if self._tx_path is None or not self._tx_path.exists():
                    raise ToolInputError(
                        "NO_TRANSACTION_DATA",
                        "No transactions parquet is bound to the data store.",
                    )
                available = set(self._tx_select_columns())
                wanted = {
                    "Timestamp": "timestamp",
                    "Account": "sender",
                    "Account_duplicated_0": "receiver",
                    "Amount Paid": "amount_paid",
                    "Amount Received": "amount_received",
                    "Payment Currency": "payment_currency",
                    "Receiving Currency": "receiving_currency",
                    "Payment Format": "payment_format",
                    "From Bank": "from_bank",
                    "To Bank": "to_bank",
                    "Is Laundering": "is_laundering",
                }
                select_parts = [
                    f'"{src}" AS "{dst}"' for src, dst in wanted.items() if src in available
                ]
                if not {"Timestamp", "Account", "Account_duplicated_0"}.issubset(available):
                    raise ToolInputError(
                        "NO_DATA",
                        "Transactions parquet is missing Account/Timestamp columns.",
                    )
                sql = (
                    f"SELECT {', '.join(select_parts)} FROM read_parquet(?) "
                    'WHERE ("Account" = ? OR "Account_duplicated_0" = ?)'
                )
                params: List[Any] = [
                    self._tx_path.as_posix(),
                    account_id,
                    account_id,
                ]
                if date is not None:
                    sql += ' AND CAST("Timestamp" AS DATE) = CAST(? AS DATE)'
                    params.append(date)
                frame = self._query_df(sql, params)

            frame = self._normalise_tx_frame(frame)
            self._tx_cache[cache_key] = frame
            self._tx_cache.move_to_end(cache_key)
            while len(self._tx_cache) > 4:
                self._tx_cache.popitem(last=False)
            return frame

    @staticmethod
    def _normalise_tx_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Give the transactions frame stable, lower-case column names."""
        rename = {
            "Timestamp": "timestamp",
            "Account": "sender",
            "Account_duplicated_0": "receiver",
            "Amount Paid": "amount_paid",
            "Amount Received": "amount_received",
            "Payment Currency": "payment_currency",
            "Receiving Currency": "receiving_currency",
            "Payment Format": "payment_format",
            "From Bank": "from_bank",
            "To Bank": "to_bank",
            "Is Laundering": "is_laundering",
        }
        frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
        for col in (
            "timestamp", "sender", "receiver", "amount_paid", "amount_received",
            "payment_currency", "receiving_currency", "payment_format",
            "from_bank", "to_bank", "is_laundering",
        ):
            if col not in frame.columns:
                frame[col] = None
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["amount_paid"] = pd.to_numeric(frame["amount_paid"], errors="coerce").fillna(0.0)
        frame["amount_received"] = pd.to_numeric(
            frame["amount_received"], errors="coerce"
        ).fillna(0.0)
        frame["sender"] = frame["sender"].astype(str)
        frame["receiver"] = frame["receiver"].astype(str)
        return frame

    # -- convenience ------------------------------------------------------
    def latest_date_for_account(self, account_id: str) -> Optional[str]:
        try:
            rows = self.score_rows(account_id)
        except ToolInputError:
            return None
        if rows.empty or "graph_date" not in rows.columns:
            return None
        dates = rows["graph_date"].dropna().astype(str)
        return dates.max() if not dates.empty else None


# ---------------------------------------------------------------------------
# Tool 1: account scores
# ---------------------------------------------------------------------------
@_tool_envelope("get_account_scores")
def get_account_scores(
    account_id: Any,
    date: Any = None,
    *,
    store: Optional[AMLDataStore] = None,
) -> Dict[str, Any]:
    """Return account/day risk scores and the operating-point interpretation."""
    account = _normalise_account(account_id)
    day = _normalise_date(date)
    store = store or get_default_store()

    rows = store.score_rows(account, day)
    if rows.empty:
        return _error(
            "get_account_scores",
            "ACCOUNT_NOT_FOUND",
            f"Account {account!r} has no risk scores"
            + (f" on {day}." if day else " in the bound pipeline run."),
            account_id=account,
            hint="Check the account id, or call without date to see all scored dates.",
        )

    records = [_score_record(r) for _, r in rows.head(MAX_DATES_PER_ACCOUNT).iterrows()]
    records.sort(key=lambda r: r.get("date") or "", reverse=True)
    summary = records[0]  # latest scored day
    highest = max(records, key=lambda r: r.get("final_risk_score") or 0.0)
    percentile = store.risk_percentile(
        summary.get("date") or "", summary.get("final_risk_score") or 0.0
    )
    summary["population_percentile"] = percentile

    thresholds = {
        "investigation": round(float(config.INVESTIGATION_THRESHOLD), 6),
        "medium_risk": round(float(config.MEDIUM_RISK_THRESHOLD), 6),
        "tier_1_high": round(float(config.TIER_1_THRESHOLD), 6),
        "tier_2_medium": round(float(config.TIER_2_THRESHOLD), 6),
    }

    return _ok(
        "get_account_scores",
        account_id=account,
        data={
            "summary": summary,
            "history": records,
            "highest_risk_day": {
                "date": highest.get("date"),
                "final_risk_score": highest.get("final_risk_score"),
                "risk_label": highest.get("risk_label"),
            },
            "thresholds": thresholds,
            "interpretation": _score_interpretation(summary, thresholds),
        },
        meta={
            "run_label": store.run_label,
            "source": store.source,
            "dates_returned": len(records),
            "summary_selected": "latest_date",
        },
    )


def _score_record(row: pd.Series) -> Dict[str, Any]:
    def pick(key: str, digits: int = 6) -> Optional[float]:
        return _round(row.get(key), digits) if key in row.index else None

    return _jsonable(
        {
            "date": row.get("graph_date"),
            "risk_label": row.get("risk_label"),
            "investigation_tier": row.get("investigation_tier"),
            "investigation_flag": _as_bool(row.get("investigation_flag")),
            "final_risk_score": pick("final_risk_score"),
            "base_risk": pick("base_risk"),
            "graph_penalty": pick("graph_penalty"),
            "meta_score": pick("meta_score"),
            "model_scores": {
                "xgboost": pick("xgb_prob"),
                "anomaly": pick("anomaly_score"),
                "behavioral": pick("behavioral_score"),
            },
            "graph_features": {
                "is_high_risk_seed": _as_bool(row.get("is_high_risk_seed")),
                "received_from_high_risk": _as_bool(
                    row.get("received_from_high_risk")
                ),
                "hop1_count": _count(row.get("hop1_count")),
                "hop2_count": _count(row.get("hop2_count")),
                "hop3_count": _count(row.get("hop3_count")),
                "bidir_count": _count(row.get("bidir_count")),
                "fan_score": pick("fan_score"),
                "daily_degree": _count(row.get("daily_degree")),
                "total_volume": _money(row.get("total_volume")),
            },
            "behaviour_features": {
                "amount_vs_own_mean": pick("amount_vs_own_mean", 2),
                "amount_zscore_vs_own": pick("amount_zscore_vs_own", 2),
                "velocity_vs_own_avg": pick("velocity_vs_own_avg", 3),
            },
        }
    )


def _score_interpretation(
    summary: Mapping[str, Any], thresholds: Mapping[str, float]
) -> List[str]:
    notes: List[str] = []
    score = summary.get("final_risk_score") or 0.0
    if score >= thresholds["investigation"]:
        notes.append(
            f"Final score {score:.3f} is at or above the investigation threshold "
            f"({thresholds['investigation']:.3f})."
        )
    else:
        notes.append(
            f"Final score {score:.3f} is below the investigation threshold "
            f"({thresholds['investigation']:.3f})."
        )
    if summary.get("investigation_tier"):
        notes.append(f"Tier: {summary['investigation_tier']}.")
    if summary.get("investigation_flag"):
        notes.append("The pipeline routed this account/day for investigation.")
    if summary.get("graph_penalty") and summary["graph_penalty"] > 0:
        notes.append(
            f"Graph exposure added {summary['graph_penalty']:.3f} to the base risk."
        )
    return notes


# ---------------------------------------------------------------------------
# Tool 2: transactions
# ---------------------------------------------------------------------------
@_tool_envelope("get_transactions")
def get_transactions(
    account_id: Any,
    date: Any = None,
    direction: Any = "both",
    limit: Any = None,
    min_amount: Any = None,
    *,
    store: Optional[AMLDataStore] = None,
) -> Dict[str, Any]:
    """Return flow aggregates plus the most recent transactions."""
    account = _normalise_account(account_id)
    day = _normalise_date(date)
    side = _normalise_direction(direction)
    row_limit = _normalise_limit(limit)
    floor = _normalise_min_amount(min_amount)
    store = store or get_default_store()

    date_source = "explicit" if day else None
    if day is None:
        day = store.latest_date_for_account(account)
        if day:
            date_source = "latest_risk_date"

    frame = store._tx_frame(account, day)
    if frame.empty:
        return _ok(
            "get_transactions",
            account_id=account,
            data={
                "date": day,
                "direction": side,
                "summary": _empty_tx_summary(),
                "transactions": [],
                "top_counterparties": [],
                "note": "No transactions found for this account"
                + (f" on {day}." if day else "."),
            },
            meta={"source": store.source, "date_source": date_source or "none"},
        )

    frame = frame.copy()
    frame["direction"] = np.select(
        [
            (frame["sender"] == account) & (frame["receiver"] == account),
            frame["receiver"] == account,
            frame["sender"] == account,
        ],
        ["self", "in", "out"],
        default="out",
    )
    if floor is not None:
        amount = np.where(frame["direction"] == "in", frame["amount_received"], frame["amount_paid"])
        frame = frame.loc[amount >= floor]
    if side != "both":
        frame = frame.loc[frame["direction"] == side]

    if frame.empty:
        return _ok(
            "get_transactions",
            account_id=account,
            data={
                "date": day,
                "direction": side,
                "summary": _empty_tx_summary(),
                "transactions": [],
                "top_counterparties": [],
                "note": "No transactions matched the requested filters.",
            },
            meta={"source": store.source, "date_source": date_source or "none"},
        )

    summary = _tx_summary(frame, account)
    counterparties = _top_counterparties(frame)

    recent = frame.sort_values("timestamp", ascending=False, kind="stable").head(row_limit)
    rows = [
        {
            "timestamp": (
                None if pd.isna(r["timestamp"])
                else pd.Timestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
            ),
            "direction": r["direction"],
            "counterparty": r["sender"] if r["direction"] == "in" else r["receiver"],
            "amount": _money(
                r["amount_received"] if r["direction"] == "in" else r["amount_paid"]
            ),
            "currency": (
                r.get("receiving_currency") if r["direction"] == "in"
                else r.get("payment_currency")
            ),
            "payment_format": r.get("payment_format"),
            "from_bank": _jsonable(r.get("from_bank")),
            "to_bank": _jsonable(r.get("to_bank")),
        }
        for _, r in recent.iterrows()
    ]

    return _ok(
        "get_transactions",
        account_id=account,
        data={
            "date": day,
            "direction": side,
            "summary": summary,
            "transactions": _jsonable(rows),
            "top_counterparties": counterparties,
            "rows_returned": len(rows),
            "rows_available": int(len(frame)),
        },
        meta={
            "source": store.source,
            "date_source": date_source or "explicit",
            "limit": row_limit,
        },
    )


def _empty_tx_summary() -> Dict[str, Any]:
    return {
        "transaction_count": 0,
        "inbound": {"count": 0, "total": 0.0},
        "outbound": {"count": 0, "total": 0.0},
        "net_flow": 0.0,
        "largest_transaction": 0.0,
        "first_transaction": None,
        "last_transaction": None,
        "unique_counterparties": 0,
    }


def _tx_summary(frame: pd.DataFrame, account: str) -> Dict[str, Any]:
    inbound = frame.loc[frame["direction"] == "in"]
    outbound = frame.loc[frame["direction"] == "out"]
    in_total = float(inbound["amount_received"].sum())
    out_total = float(outbound["amount_paid"].sum())
    amounts = np.where(
        frame["direction"] == "in", frame["amount_received"], frame["amount_paid"]
    )
    counterparties = np.where(
        frame["direction"] == "in", frame["sender"], frame["receiver"]
    )
    unique = len(set(counterparties))
    timestamps = frame["timestamp"].dropna()
    formats = frame["payment_format"].dropna().astype(str).value_counts().head(3)
    return _jsonable(
        {
            "transaction_count": int(len(frame)),
            "inbound": {"count": int(len(inbound)), "total": _money(in_total)},
            "outbound": {"count": int(len(outbound)), "total": _money(out_total)},
            "net_flow": _money(in_total - out_total),
            "largest_transaction": _money(np.max(amounts)) if len(amounts) else 0.0,
            "average_transaction": _money(np.mean(amounts)) if len(amounts) else 0.0,
            "first_transaction": (
                None if timestamps.empty else timestamps.min().strftime("%Y-%m-%d %H:%M")
            ),
            "last_transaction": (
                None if timestamps.empty else timestamps.max().strftime("%Y-%m-%d %H:%M")
            ),
            "unique_counterparties": int(unique),
            "payment_formats": {str(k): int(v) for k, v in formats.items()},
        }
    )


def _top_counterparties(frame: pd.DataFrame, top_n: int = TOP_COUNTERPARTIES) -> List[Dict[str, Any]]:
    inbound = frame.loc[frame["direction"] == "in"]
    outbound = frame.loc[frame["direction"] == "out"]

    in_agg = (
        inbound.groupby("sender")
        .agg(in_total=("amount_received", "sum"), in_count=("amount_received", "size"))
        .reset_index()
        .rename(columns={"sender": "counterparty"})
    )
    out_agg = (
        outbound.groupby("receiver")
        .agg(out_total=("amount_paid", "sum"), out_count=("amount_paid", "size"))
        .reset_index()
        .rename(columns={"receiver": "counterparty"})
    )
    merged = in_agg.merge(out_agg, on="counterparty", how="outer").fillna(0)
    if merged.empty:
        return []
    merged["total"] = merged["in_total"] + merged["out_total"]
    merged = merged.sort_values("total", ascending=False).head(top_n)
    return _jsonable(
        [
            {
                "counterparty": r["counterparty"],
                "inbound_total": _money(r["in_total"]),
                "outbound_total": _money(r["out_total"]),
                "inbound_count": int(r["in_count"]),
                "outbound_count": int(r["out_count"]),
            }
            for _, r in merged.iterrows()
        ]
    )


# ---------------------------------------------------------------------------
# Tool 3: graph exposure
# ---------------------------------------------------------------------------
@_tool_envelope("get_graph_exposure")
def get_graph_exposure(
    account_id: Any,
    date: Any = None,
    *,
    store: Optional[AMLDataStore] = None,
) -> Dict[str, Any]:
    """Return one-hop network exposure and pre-computed multi-hop risk features."""
    account = _normalise_account(account_id)
    day = _normalise_date(date)
    store = store or get_default_store()

    date_source = "explicit" if day else None
    if day is None:
        day = store.latest_date_for_account(account)
        if day:
            date_source = "latest_risk_date"

    rows = store.score_rows(account, day)
    if rows.empty:
        return _error(
            "get_graph_exposure",
            "ACCOUNT_NOT_FOUND",
            f"Account {account!r} has no graph/risk features"
            + (f" on {day}." if day else " in the bound pipeline run."),
            account_id=account,
            hint="Check the account id and date; graph features are produced per account/day.",
        )

    row = rows.iloc[0]
    hop1 = int(_round(row.get("hop1_count"), 2) or 0)
    hop2 = int(_round(row.get("hop2_count"), 2) or 0)
    hop3 = int(_round(row.get("hop3_count"), 2) or 0)
    bidir = int(_round(row.get("bidir_count"), 2) or 0)
    fan_score = _round(row.get("fan_score"), 6) or 0.0
    volume_factor = _round(row.get("volume_factor"), 4) or 1.0
    daily_degree = _round(row.get("daily_degree"), 2) or 0.0

    seeds = store.seed_accounts(day)
    frame = store._tx_frame(account, day)
    counterparties: List[Dict[str, Any]] = []
    seed_neighbours: List[Dict[str, Any]] = []
    seed_inbound: set = set()
    seed_outbound: set = set()
    if not frame.empty:
        frame = frame.copy()
        frame["direction"] = np.select(
            [frame["receiver"] == account, frame["sender"] == account],
            ["in", "out"],
            default="out",
        )
        inbound_mask = (frame["direction"] == "in") & frame["sender"].isin(seeds)
        outbound_mask = (frame["direction"] == "out") & frame["receiver"].isin(seeds)
        seed_inbound = set(frame.loc[inbound_mask, "sender"].astype(str))
        seed_outbound = set(frame.loc[outbound_mask, "receiver"].astype(str))
        seed_neighbours = _seed_neighbours(frame, account, seeds)
        counterparties = _top_counterparties(frame, top_n=TOP_COUNTERPARTIES)

    penalty_raw = {
        "direct_hop1": hop1 * config.GRAPH_RECEIVING_PENALTY * volume_factor,
        "indirect_hop2": hop2 * config.GRAPH_INDIRECT_PENALTY,
        "tertiary_hop3": hop3 * config.GRAPH_INDIRECT_PENALTY * config.TERTIARY_PENALTY_FACTOR,
        "bidirectional": bidir * config.BIDIR_PENALTY,
        "fan_pattern": fan_score * config.FAN_BOOST,
    }
    raw_total = float(sum(penalty_raw.values()))
    cap = float(config.GRAPH_PENALTY_CAP)
    cap_applied = raw_total > cap > 0
    scale = cap / raw_total if cap_applied else 1.0
    penalty_components = {
        key: _round(value * scale, 6) for key, value in penalty_raw.items()
    }
    penalty_share_pct = {
        key: _round((value / raw_total) * 100.0, 2) if raw_total > 0 else 0.0
        for key, value in penalty_raw.items()
    }

    flags = {
        "is_high_risk_seed": bool(_as_bool(row.get("is_high_risk_seed"))),
        "received_from_high_risk": bool(_as_bool(row.get("received_from_high_risk"))),
        "direct_seed_exposure": hop1 > 0,
        "indirect_exposure": hop2 > 0,
        "layering_bidirectional": bidir > 0,
        "mule_fan_pattern": fan_score > 0,
        "high_degree_hub": daily_degree > 50,
    }

    return _ok(
        "get_graph_exposure",
        account_id=account,
        data={
            "date": day,
            "exposure": {
                "hop1_direct_seed_count": hop1,
                "hop2_indirect_count": hop2,
                "hop3_tertiary_count": hop3,
                "bidirectional_relationship_count": bidir,
                "fan_score": fan_score,
                "daily_degree": daily_degree,
                "total_seed_volume": _money(row.get("total_volume")),
                "volume_factor": volume_factor,
                "seed_inbound_accounts": len(seed_inbound),
                "seed_outbound_accounts": len(seed_outbound),
            },
            "graph_penalty": _round(row.get("graph_penalty"), 6),
            "penalty_cap": cap,
            "penalty_cap_applied": cap_applied,
            "penalty_components": penalty_components,
            "penalty_share_pct": penalty_share_pct,
            "flags": flags,
            "seed_neighbours": seed_neighbours,
            "top_counterparties": counterparties,
            "interpretation": _graph_interpretation(flags, hop1, seed_inbound, seed_outbound),
            "note": (
                "Hop 2/3 counts are pre-computed aggregates; only one-hop counterparty "
                "identities are available. Penalty components are rescaled to the "
                "capped penalty so their shares remain meaningful. "
                "'received_from_high_risk' is the pipeline's stored flag; use "
                "seed_neighbours to see whether exposure is inbound or outbound."
            ),
        },
        meta={
            "run_label": store.run_label,
            "source": store.source,
            "date_source": date_source or "explicit",
            "seeds_on_date": len(seeds),
        },
    )


def _seed_neighbours(
    frame: pd.DataFrame, account: str, seeds: frozenset
) -> List[Dict[str, Any]]:
    if not seeds:
        return []
    inbound = frame.loc[(frame["direction"] == "in") & frame["sender"].isin(seeds)]
    outbound = frame.loc[(frame["direction"] == "out") & frame["receiver"].isin(seeds)]

    rows: List[Dict[str, Any]] = []
    if not inbound.empty:
        agg = (
            inbound.groupby("sender")
            .agg(total=("amount_received", "sum"), count=("amount_received", "size"))
            .reset_index()
            .rename(columns={"sender": "seed_account"})
        )
        for _, r in agg.sort_values("total", ascending=False).head(MAX_SEED_NEIGHBOURS).iterrows():
            rows.append(
                {
                    "seed_account": r["seed_account"],
                    "direction": "inbound_from_seed",
                    "total": _money(r["total"]),
                    "count": int(r["count"]),
                }
            )
    if not outbound.empty:
        agg = (
            outbound.groupby("receiver")
            .agg(total=("amount_paid", "sum"), count=("amount_paid", "size"))
            .reset_index()
            .rename(columns={"receiver": "seed_account"})
        )
        for _, r in agg.sort_values("total", ascending=False).head(MAX_SEED_NEIGHBOURS).iterrows():
            rows.append(
                {
                    "seed_account": r["seed_account"],
                    "direction": "outbound_to_seed",
                    "total": _money(r["total"]),
                    "count": int(r["count"]),
                }
            )
    return _jsonable(rows)


def _graph_interpretation(
    flags: Mapping[str, bool],
    hop1: int,
    seed_inbound: Iterable[str] = (),
    seed_outbound: Iterable[str] = (),
) -> List[str]:
    notes: List[str] = []
    if flags["is_high_risk_seed"]:
        notes.append(
            "This account is itself an original high-risk seed "
            "(base risk above the seed threshold with normal connectivity)."
        )
    if hop1:
        inbound_count = len(set(seed_inbound))
        outbound_count = len(set(seed_outbound))
        direction = []
        if inbound_count:
            direction.append(f"{inbound_count} inbound")
        if outbound_count:
            direction.append(f"{outbound_count} outbound")
        detail = f" ({', '.join(direction)})" if direction else ""
        notes.append(
            f"Direct one-hop relationships with {hop1} distinct high-risk seed "
            f"account(s) on this date{detail}."
        )
    if flags["indirect_exposure"]:
        notes.append("Indirect (two-hop) exposure to high-risk seeds was detected.")
    if flags["layering_bidirectional"]:
        notes.append(
            "Bidirectional flows with counterparties are consistent with layering."
        )
    if flags["mule_fan_pattern"]:
        notes.append(
            "Fan-in/fan-out topology is consistent with a mule/collection account."
        )
    if flags["high_degree_hub"]:
        notes.append(
            "Very high daily degree (hub/super-node): direct exposure may be "
            "less discriminating than burst or value anomalies."
        )
    if not notes:
        notes.append("No direct high-risk seed exposure or network pattern was found.")
    return notes


# ---------------------------------------------------------------------------
# Envelope helpers / registry
# ---------------------------------------------------------------------------
def _ok(tool: str, *, account_id: str, data: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _jsonable(
        {
            "ok": True,
            "tool": tool,
            "account_id": account_id,
            "data": data,
            "meta": meta or {},
        }
    )


def _error(
    tool: str,
    code: str,
    message: str,
    *,
    account_id: Optional[str] = None,
    hint: Optional[str] = None,
) -> Dict[str, Any]:
    return _jsonable(
        {
            "ok": False,
            "tool": tool,
            "account_id": account_id,
            "error": {"code": code, "message": message, "hint": hint},
        }
    )


_TOOL_FUNCTIONS = {
    "get_account_scores": get_account_scores,
    "get_transactions": get_transactions,
    "get_graph_exposure": get_graph_exposure,
}


def tool_names() -> List[str]:
    return list(_TOOL_FUNCTIONS)


def dispatch_tool(
    name: str,
    arguments: Optional[Mapping[str, Any]] = None,
    *,
    store: Optional[AMLDataStore] = None,
) -> Dict[str, Any]:
    """Execute a tool by name, always returning a JSON-safe envelope."""
    if not isinstance(name, str) or name not in _TOOL_FUNCTIONS:
        return _error(
            str(name),
            "UNKNOWN_TOOL",
            f"Unknown tool {name!r}.",
            hint=f"Available tools: {', '.join(tool_names())}.",
        )

    if arguments is None:
        args: Dict[str, Any] = {}
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return _error(name, "INVALID_ARGUMENTS", "Tool arguments are not valid JSON.")
        if not isinstance(parsed, dict):
            return _error(name, "INVALID_ARGUMENTS", "Tool arguments must be a JSON object.")
        args = parsed
    elif isinstance(arguments, Mapping):
        args = dict(arguments)
    else:
        return _error(name, "INVALID_ARGUMENTS", "Tool arguments must be a JSON object.")

    try:
        return _TOOL_FUNCTIONS[name](store=store or get_default_store(), **args)
    except ToolInputError as exc:
        return _error(name, exc.code, exc.message)
    except TypeError as exc:
        return _error(
            name,
            "INVALID_ARGUMENTS",
            f"Invalid arguments for {name}: {exc}",
        )
    except Exception as exc:  # never let a tool crash the agent loop
        log.exception("Tool %s failed", name)
        return _error(name, "TOOL_ERROR", f"{type(exc).__name__}: {exc}")


TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "get_account_scores",
        "description": (
            "Get account/day AML risk scores, tiers, model components and graph "
            "features. Call this first for any case; omit date to get every scored day."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier, e.g. 100428660.",
                },
                "date": {
                    "type": "string",
                    "description": "Optional day in YYYY-MM-DD format (e.g. 2022-10-23).",
                },
            },
            "required": ["account_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_transactions",
        "description": (
            "Get inbound/outbound totals, net flow, top counterparties and recent "
            "transactions for one account/day. Use it to verify flows behind a score."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier, e.g. 100428660.",
                },
                "date": {
                    "type": "string",
                    "description": "Optional day in YYYY-MM-DD format. Defaults to the latest scored day.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["both", "in", "out"],
                    "description": "Filter by flow direction. Default both.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max transaction rows to return (default {DEFAULT_TX_LIMIT}, max {MAX_TX_LIMIT}).",
                },
                "min_amount": {
                    "type": "number",
                    "description": "Optional minimum transaction amount filter.",
                },
            },
            "required": ["account_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_graph_exposure",
        "description": (
            "Get one-hop high-risk seed exposure, two/three-hop counts, layering and "
            "mule indicators, and the graph penalty breakdown for one account/day."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier, e.g. 100428660.",
                },
                "date": {
                    "type": "string",
                    "description": "Optional day in YYYY-MM-DD format. Defaults to the latest scored day.",
                },
            },
            "required": ["account_id"],
            "additionalProperties": False,
        },
    },
]

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"type": "function", "function": spec} for spec in TOOL_SPECS
]


def tool_catalog_text() -> str:
    """Compact human/LLM readable catalog used in the system prompt."""
    lines = []
    for spec in TOOL_SPECS:
        params = ", ".join(spec["parameters"]["properties"].keys())
        short = spec["description"].split(". ")[0].strip()
        if not short.endswith("."):
            short += "."
        lines.append(f"- {spec['name']}({params}): {short}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default store (used by the FastAPI app and the CLI)
# ---------------------------------------------------------------------------
_DEFAULT_STORE = AMLDataStore()
_STORE_LOCK = RLock()


def get_default_store() -> AMLDataStore:
    return _DEFAULT_STORE


def set_default_store(store: AMLDataStore) -> AMLDataStore:
    global _DEFAULT_STORE
    with _STORE_LOCK:
        _DEFAULT_STORE = store
    return store


def latest_output_dir() -> Optional[Path]:
    """Newest saved run that contains ``risk_scores.parquet``."""
    root = Path(config.OUTPUT_ROOT)
    if not root.is_dir():
        return None
    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and (p / "risk_scores.parquet").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "risk_scores.parquet").stat().st_mtime)


def resolve_transactions_path(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """Pick the transactions parquet to join against a saved run."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    latest = latest_output_dir()
    if latest is not None:
        summary_path = latest / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                recorded = summary.get("input_parquet")
                if recorded:
                    candidate = Path(recorded).expanduser()
                    if not candidate.is_absolute():
                        candidate = Path(config.REPO_ROOT) / candidate
                    if candidate.is_file():
                        return candidate
            except Exception:
                log.debug("Could not read %s", summary_path, exc_info=True)
    fallback = Path(config.DEFAULT_PARQUET_PATH)
    return fallback if fallback.is_file() else None


def _load_saved_reports(run_dir: Path) -> Dict[str, str]:
    """Map account id -> SAR draft from ``sar_reports.jsonl`` (for restarts)."""
    path = run_dir / "sar_reports.jsonl"
    if not path.is_file():
        return {}
    pattern = re.compile(r"\*\*Customer\*\*:\s*([A-Za-z0-9_.:\-]+)")
    reports: Dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("report") or ""
                match = pattern.search(text)
                if match:
                    reports[match.group(1)] = text
    except Exception:
        log.warning("Could not read %s", path, exc_info=True)
    return reports


def store_from_latest_run(
    *,
    output_dir: Optional[str | Path] = None,
    transactions_path: Optional[str | Path] = None,
) -> AMLDataStore:
    """Build a store from a saved pipeline run (used by the CLI)."""
    run_dir = Path(output_dir) if output_dir else latest_output_dir()
    if run_dir is None:
        return AMLDataStore()
    risk_path = run_dir / "risk_scores.parquet"
    if not risk_path.is_file():
        return AMLDataStore()
    tx_path = resolve_transactions_path(transactions_path)
    return AMLDataStore(
        risk_scores_path=risk_path,
        transactions_path=tx_path,
        run_label=run_dir.name,
        input_parquet=tx_path,
        reports=_load_saved_reports(run_dir),
    )
