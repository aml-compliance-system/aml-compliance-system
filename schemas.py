"""Typed request/response models for the investigator API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

CASE_ID_PATTERN = r"^CASE-[A-Za-z0-9]+-\d{8}$"
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"


class RunInfo(BaseModel):
    run_id: str
    risk_path: str
    transactions_path: Optional[str]
    summary: Dict[str, Any]
    created_at: Optional[str] = None
    has_legacy_reports: bool = False


class CaseSummary(BaseModel):
    run_id: str
    case_id: str = Field(pattern=CASE_ID_PATTERN)
    account_id: str
    graph_date: str
    holder_name: Optional[str] = None
    risk_score: float
    risk_label: str
    tier: str
    status: str = "NEW"
    assigned_to: Optional[str] = None
    base_risk: Optional[float] = None
    graph_penalty: Optional[float] = None
    hop1_count: Optional[int] = None
    hop2_count: Optional[int] = None
    bidir_count: Optional[int] = None
    fan_score: Optional[float] = None
    daily_degree: Optional[int] = None
    is_high_risk_seed: bool = False
    received_from_high_risk: bool = False


class CaseListResponse(BaseModel):
    run_id: str
    items: List[CaseSummary]
    total: int
    page: int
    page_size: int
    page_count: int
    status_counts: Dict[str, int] = Field(default_factory=dict)
    available_dates: List[str] = Field(default_factory=list)


class CaseDetail(CaseSummary):
    thresholds: Dict[str, float] = Field(default_factory=dict)
    model_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    graph_features: Dict[str, Any] = Field(default_factory=dict)
    transaction_summary: Dict[str, Any] = Field(default_factory=dict)
    sar: Dict[str, Any] = Field(default_factory=dict)
    legacy_report: Optional[str] = None
    notes_count: int = 0
    created_at: Optional[str] = None


class TransactionItem(BaseModel):
    transaction_ref: str
    timestamp: str
    direction: Literal["in", "out", "self"]
    from_account: str
    to_account: str
    from_entity_name: Optional[str] = None
    to_entity_name: Optional[str] = None
    counterparty: str
    amount: float
    currency: Optional[str] = None
    amount_paid: float
    amount_received: float
    payment_currency: Optional[str] = None
    receiving_currency: Optional[str] = None
    payment_format: Optional[str] = None
    from_bank: Optional[int] = None
    to_bank: Optional[int] = None
    source_file: str


class TransactionAggregate(BaseModel):
    direction: Literal["in", "out", "self"]
    currency: Optional[str] = None
    count: int
    total: float


class TransactionPage(BaseModel):
    run_id: str
    case_id: str
    account_id: str
    graph_date: str
    items: List[TransactionItem]
    total: int
    page: int
    page_size: int
    page_count: int
    aggregates: List[TransactionAggregate] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    type: Literal["account", "holder"] = "account"
    label: str
    account_id: Optional[str] = None
    holder_name: Optional[str] = None
    entity_id: Optional[str] = None
    bank: Optional[int] = None
    is_focal: bool = False
    hop: int = 0


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Literal["money", "holder"] = "money"
    timestamp: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    payment_format: Optional[str] = None
    transaction_ref: Optional[str] = None
    source_file: Optional[str] = None


class GraphLimitInfo(BaseModel):
    max_nodes: int
    max_edges: int
    max_depth: int


class GraphResponse(BaseModel):
    run_id: str
    case_id: str
    window: Dict[str, str]
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    truncated: bool = False
    truncation_reason: Optional[str] = None
    limits: GraphLimitInfo
    note: Optional[str] = None


class GraphPathFinding(BaseModel):
    algorithm: str
    rule: str
    nodes: List[str]
    edges: List[Dict[str, Any]]
    start: Optional[str] = None
    end: Optional[str] = None


class GraphCycleFinding(BaseModel):
    algorithm: str
    rule: str
    nodes: List[str]
    edges: List[Dict[str, Any]]


class SharedCounterpartyFinding(BaseModel):
    counterparty: str
    account_count: int
    accounts: List[str]
    evidence: List[Dict[str, Any]]


class GraphAnalysisResponse(BaseModel):
    run_id: str
    case_id: str
    window: Dict[str, str]
    algorithm: str
    rule: str
    findings: List[Dict[str, Any]]
    truncated: bool = False
    limits: Dict[str, Any] = Field(default_factory=dict)


class GraphQueryParams(BaseModel):
    depth: int = Field(default=1, ge=1, le=2)
    limit_nodes: int = Field(default=100, ge=1, le=100)
    limit_edges: int = Field(default=300, ge=1, le=300)
    include_holders: bool = False


class PathRequest(BaseModel):
    source: str = Field(min_length=1, max_length=64)
    target: str = Field(min_length=1, max_length=64)
    max_depth: int = Field(default=4, ge=1, le=4)


class CycleRequest(BaseModel):
    max_length: int = Field(default=4, ge=2, le=5)


class SharedCounterpartyRequest(BaseModel):
    accounts: List[str] = Field(min_length=2, max_length=10)
    min_shared: int = Field(default=2, ge=2, le=10)


class ComponentRequest(BaseModel):
    account: Optional[str] = Field(default=None, max_length=64)


class ExpandRequest(BaseModel):
    account: str = Field(min_length=1, max_length=64)
    direction: Literal["both", "in", "out"] = "both"


class NoteCreate(BaseModel):
    body: str = Field(min_length=1, max_length=20_000)
    note_type: Literal["NOTE", "FINDING", "HYPOTHESIS"] = "NOTE"
    confirm: bool = False


class NoteUpdate(BaseModel):
    body: str = Field(min_length=1, max_length=20_000)
    note_type: Optional[Literal["NOTE", "FINDING", "HYPOTHESIS"]] = None
    confirm: bool = False


class NoteRead(BaseModel):
    id: int
    run_id: str
    case_id: str
    author: str
    note_type: str
    body: str
    created_at: str
    updated_at: str
    version: int


class CaseStatusUpdate(BaseModel):
    status: Literal["NEW", "IN_REVIEW", "WAITING_INFO", "ESCALATED", "CLOSED"]
    assigned_to: Optional[str] = Field(default=None, max_length=64)
    confirm: bool = False


class TimelineEvent(BaseModel):
    id: int
    run_id: str
    case_id: str
    actor: str
    action: str
    outcome: str
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class SarDraftRead(BaseModel):
    id: int
    run_id: str
    case_id: str
    version: int
    status: str
    content: Dict[str, Any]
    missing_fields: List[str]
    evidence: List[Dict[str, Any]]
    created_by: str
    created_at: str
    updated_at: str
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    validation: Dict[str, Any] = Field(default_factory=dict)


class SarDraftCreate(BaseModel):
    confirm: bool = False
    include_legacy_report: bool = True


class SarDraftUpdate(BaseModel):
    content: Dict[str, Any]
    confirm: bool = False


class SarApprove(BaseModel):
    confirm: bool = False
    acknowledge_missing: bool = False


class SarConfirm(BaseModel):
    confirm: bool = False


class CaseChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class MetaResponse(BaseModel):
    run_id: Optional[str]
    run_summary: Dict[str, Any] = Field(default_factory=dict)
    auth_mode: Literal["local", "token"]
    investigator: str
    model: str
    native_tools: bool
    policy_validation: Dict[str, Any]
    limits: Dict[str, Any]
    capabilities: Dict[str, bool]


class MessageResponse(BaseModel):
    message: str
    case_id: Optional[str] = None


class CaseChatResponse(BaseModel):
    message: str
    case_id: str
    run_id: str
    response: str
    answer: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    steps: int = 0
    model: str = ""
    degraded: bool = False
    warnings: List[str] = Field(default_factory=list)
    elapsed_ms: int = 0
