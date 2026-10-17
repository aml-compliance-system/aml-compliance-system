"""Evidence-grounded SAR drafting, validation, approval and export.

The first draft is deterministic and derived from verified case data, not from
an LLM. It lists external information the pipeline does not contain (KYC,
identity documents, filing institution, reporter identity) rather than inventing
it. An investigator must review, edit and explicitly approve the draft.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import config
from case_service import CaseService, InvalidCase, RunInfo
from docx_writer import build_docx
from pdf_writer import SimplePDF
from workflow_store import WorkflowStore, WorkflowValidation

POLICY_VALIDATION = {
    "available": False,
    "reason": "Policy retrieval/RAG is not configured in this release.",
}

_EXTERNAL_MISSING = [
    "subject_date_of_birth",
    "subject_national_id_or_registration",
    "subject_address",
    "kyc_customer_profile",
    "filing_institution_identity",
    "reporter_identity_and_authority",
    "regulatory_policy_validation",
]


class SARService:
    def __init__(self, cases: CaseService, workflow: WorkflowStore):
        self.cases = cases
        self.workflow = workflow

    # -- drafting ---------------------------------------------------------
    def build_draft(
        self,
        case_id: str,
        actor: str,
        *,
        run_id: Optional[str] = None,
        include_legacy_report: bool = True,
    ) -> Dict[str, Any]:
        run = self.cases.get_run(run_id)
        case = self.cases.get_case(case_id, run.run_id)
        transactions = self.cases.list_transactions(
            case_id,
            run_id=run.run_id,
            page_size=config.SAR_MAX_EVIDENCE_ROWS,
        )
        notes = self.workflow.list_notes(run.run_id, case_id)
        findings = [note for note in notes if note.get("note_type") == "FINDING"]
        content = self._content(
            case=case,
            transactions=transactions,
            findings=findings,
            legacy_report=case.get("legacy_report") if include_legacy_report else None,
            run=run,
        )
        validation = self.validate(content)
        missing = validation["missing_fields"]
        evidence = content.get("evidence", [])
        draft = self.workflow.create_sar_draft(
            run.run_id, case_id, content, missing, evidence, actor
        )
        return self._decorate(draft, validation)

    def _content(
        self,
        *,
        case: Dict[str, Any],
        transactions: Dict[str, Any],
        findings: Sequence[Dict[str, Any]],
        legacy_report: Optional[str],
        run: RunInfo,
    ) -> Dict[str, Any]:
        account = case["account_id"]
        holder = case.get("holder_name")
        subject = holder or "Source-record holder name unavailable"
        reason_codes = self._reason_codes(case)
        aggregates = transactions.get("aggregates") or []
        tx_items = transactions.get("items") or []
        currency_lines = [
            {
                "direction": item["direction"],
                "currency": item["currency"],
                "count": item["count"],
                "total": item["total"],
            }
            for item in aggregates
        ]
        evidence = [
            {
                "transaction_ref": item["transaction_ref"],
                "timestamp": item["timestamp"],
                "direction": item["direction"],
                "counterparty": item["counterparty"],
                "amount": item["amount"],
                "currency": item["currency"],
                "payment_format": item.get("payment_format"),
                "source_file": item.get("source_file"),
            }
            for item in tx_items
        ]
        narrative = self._narrative(case, reason_codes, currency_lines, evidence, findings)
        content = {
            "schema": "findetect.sar.v1",
            "case": {
                "case_id": case["case_id"],
                "run_id": run.run_id,
                "account_id": account,
                "holder_name": holder,
                "holder_name_source": "transaction source record (not verified KYC)",
                "graph_date": case["graph_date"],
            },
            "risk": {
                "final_risk_score": case["risk_score"],
                "risk_label": case["risk_label"],
                "tier": case["tier"],
                "base_risk": case.get("base_risk"),
                "graph_penalty": case.get("graph_penalty"),
                "model_scores": case.get("model_scores") or {},
                "thresholds": case.get("thresholds") or {},
                "graph_features": case.get("graph_features") or {},
            },
            "reason_codes": reason_codes,
            "activity": {
                "period": {
                    "from": f"{case['graph_date']}T00:00:00",
                    "to": f"{case['graph_date']}T23:59:59",
                },
                "transaction_count": transactions.get("total", 0),
                "currencies": transactions.get("currency_list")
                or sorted({row["currency"] for row in currency_lines if row.get("currency")}),
                "aggregates": currency_lines,
            },
            "evidence": evidence,
            "evidence_refs": [item["transaction_ref"] for item in evidence],
            "investigator_findings": [
                {
                    "note_id": note["id"],
                    "author": note["author"],
                    "version": note["version"],
                    "created_at": note["created_at"],
                    "body": note["body"],
                }
                for note in findings
            ],
            "narrative": narrative,
            "legacy_pipeline_report": legacy_report,
            "missing_fields": [],
            "policy_validation": dict(POLICY_VALIDATION),
            "disclaimer": (
                "This is an automated decision-support draft, not a legal "
                "determination. A human investigator must review, edit and approve it. "
                "No report is submitted to a regulator by this application."
            ),
            "status": "DRAFT",
        }
        content["missing_fields"] = self._missing_fields(content)
        return content

    def _reason_codes(self, case: Dict[str, Any]) -> List[Dict[str, str]]:
        codes = []
        if case.get("is_high_risk_seed"):
            codes.append(
                {
                    "code": "RC-ML-SEED",
                    "description": "Account exceeded the combined ML high-risk seed threshold.",
                }
            )
        if case.get("received_from_high_risk"):
            codes.append(
                {
                    "code": "RC-GRAPH-1HOP",
                    "description": "Account has a direct one-hop link to an original high-risk seed.",
                }
            )
        if not codes:
            codes.append(
                {
                    "code": "RC-RISK-THRESHOLD",
                    "description": "Account crossed the operational investigation risk threshold.",
                }
            )
        return codes

    def _narrative(
        self,
        case: Dict[str, Any],
        reason_codes: Sequence[Dict[str, str]],
        currency_lines: Sequence[Dict[str, Any]],
        evidence: Sequence[Dict[str, Any]],
        findings: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        holder = case.get("holder_name") or "the source-record entity"
        activity_lines = []
        for row in currency_lines:
            activity_lines.append(
                f"{row['direction']} {row['count']} transactions totalling "
                f"{row['total']:,.2f} {row.get('currency') or 'unknown currency'}"
            )
        activity_text = "; ".join(activity_lines) if activity_lines else "No transactions were returned."
        codes = ", ".join(item["code"] for item in reason_codes)
        return {
            "summary": (
                f"Account {case['account_id']} ({holder}) was scored "
                f"{case['risk_score']:.3f} ({case['risk_label']}, {case['tier']}) "
                f"for {case['graph_date']} by the FINDETECT pipeline. "
                f"Triggered indicators: {codes}."
            ),
            "suspicious_activity": (
                "The following are analytical indicators and require investigator "
                "verification. The pipeline score is a prioritisation aid, not a "
                "finding that money laundering occurred. "
                f"Activity in the selected day: {activity_text}."
            ),
            "evidence_summary": (
                f"{len(evidence)} transactions were retrieved with reusable evidence "
                "references. Amounts are reported by their source currency; no FX "
                "conversion was applied."
            ),
            "network_summary": (
                f"Direct high-risk seed links: {case['graph_features'].get('hop1_count')}; "
                f"two-hop links: {case['graph_features'].get('hop2_count')}; "
                f"bidirectional relationships: {case['graph_features'].get('bidir_count')}; "
                f"graph penalty: {case.get('graph_penalty')}. A graph link is a risk "
                "indicator and does not establish common ownership or intent."
            ),
            "investigator_findings": (
                f"{len(findings)} investigator-confirmed finding(s) are linked to this draft."
                if findings
                else "No investigator-confirmed findings have been recorded for this case."
            ),
            "conclusion": (
                "The draft is incomplete until the investigator reviews the evidence, "
                "records findings and confirms the missing-information section. "
                "No conclusion of criminal conduct is made by this system."
            ),
        }

    def _missing_fields(self, content: Dict[str, Any]) -> List[str]:
        missing = list(_EXTERNAL_MISSING)
        if not content.get("case", {}).get("holder_name"):
            missing.append("subject_holder_name")
        if not content.get("evidence"):
            missing.append("transaction_evidence")
        if not content.get("risk", {}).get("final_risk_score"):
            missing.append("risk_assessment")
        if not content.get("investigator_findings"):
            missing.append("investigator_findings")
        return list(dict.fromkeys(missing))

    # -- validation -------------------------------------------------------
    def validate(self, content: Dict[str, Any]) -> Dict[str, Any]:
        blocking = []
        warnings = []
        case = content.get("case") or {}
        risk = content.get("risk") or {}
        activity = content.get("activity") or {}
        narrative = content.get("narrative") or {}
        if not case.get("case_id"):
            blocking.append("case_id")
        if not case.get("account_id"):
            blocking.append("account_id")
        if not case.get("graph_date"):
            blocking.append("graph_date")
        if risk.get("final_risk_score") is None:
            blocking.append("risk_score")
        if not activity.get("transaction_count"):
            blocking.append("transaction_evidence")
        if not narrative.get("summary") or not narrative.get("suspicious_activity"):
            blocking.append("narrative")
        missing = list(dict.fromkeys(
            list(content.get("missing_fields") or []) + blocking + _EXTERNAL_MISSING
        ))
        if not content.get("evidence"):
            warnings.append("No transaction rows were included in the draft.")
        if not content.get("investigator_findings"):
            warnings.append("No investigator-confirmed findings are linked.")
        return {
            "valid": not blocking,
            "missing_fields": missing,
            "blocking_missing": blocking,
            "warnings": warnings,
            "policy_validation": dict(POLICY_VALIDATION),
        }

    # -- persistence ------------------------------------------------------
    def _decorate(self, draft: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(draft)
        item["validation"] = validation
        return item

    def list_drafts(self, case_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        run = self.cases.get_run(run_id)
        case = self.cases.get_case(case_id, run.run_id)
        drafts = self.workflow.list_sar_drafts(run.run_id, case_id)
        return {
            "run_id": run.run_id,
            "case_id": case_id,
            "drafts": drafts,
            "legacy_report": case.get("legacy_report"),
            "policy_validation": dict(POLICY_VALIDATION),
        }

    def get_draft(self, draft_id: int) -> Dict[str, Any]:
        draft = self.workflow.get_sar_draft(draft_id)
        return self._decorate(draft, self.validate(draft["content"]))

    def update_draft(self, draft_id: int, actor: str, content: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(content, dict):
            raise InvalidCase("SAR content must be an object")
        validation = self.validate(content)
        draft = self.workflow.update_sar_draft(
            draft_id,
            actor,
            content,
            validation["missing_fields"],
            content.get("evidence") or [],
        )
        return self._decorate(draft, validation)

    def submit_for_review(self, draft_id: int, actor: str) -> Dict[str, Any]:
        draft = self.workflow.submit_sar_for_review(draft_id, actor)
        return self._decorate(draft, self.validate(draft["content"]))

    def approve(
        self, draft_id: int, actor: str, *, acknowledge_missing: bool
    ) -> Dict[str, Any]:
        draft = self.workflow.approve_sar_draft(
            draft_id, actor, acknowledge_missing=acknowledge_missing
        )
        return self._decorate(draft, self.validate(draft["content"]))

    # -- export -----------------------------------------------------------
    def _blocks(self, draft: Dict[str, Any]) -> Tuple[str, str, List[Tuple[str, Any]]]:
        content = draft["content"]
        case = content.get("case") or {}
        risk = content.get("risk") or {}
        activity = content.get("activity") or {}
        narrative = content.get("narrative") or {}
        graph = risk.get("graph_features") or {}
        blocks: List[Tuple[str, Any]] = []

        blocks.append(("heading", "Case & Entity Overview"))
        blocks.append(("table", (
            ["Field", "Value"],
            [
                ["Case ID", case.get("case_id")],
                ["Account", case.get("account_id")],
                ["Holder (source record)", case.get("holder_name")],
                ["Activity date", case.get("graph_date")],
                ["Run", case.get("run_id")],
                ["Draft version", draft.get("version")],
                ["Status", draft.get("status")],
            ],
        )))

        blocks.append(("heading", "Risk Assessment"))
        blocks.append(("table", (
            ["Measure", "Value"],
            [
                ["Final risk score", f"{float(risk.get('final_risk_score') or 0):.3f}"],
                ["Risk label", risk.get("risk_label")],
                ["Tier", risk.get("tier")],
                ["Base risk", risk.get("base_risk")],
                ["Graph penalty", risk.get("graph_penalty")],
                ["Investigation threshold", (risk.get("thresholds") or {}).get("investigation")],
            ],
        )))
        blocks.append(("paragraph", "Model components (stored outputs; no per-feature attribution is available):"))
        blocks.append(("table", (
            ["Component", "Score"],
            [[key, value] for key, value in (risk.get("model_scores") or {}).items()],
        )))

        blocks.append(("heading", "Triggered Indicators"))
        blocks.append(("bullets", [
            f"{item.get('code')}: {item.get('description')}"
            for item in content.get("reason_codes") or []
        ]))

        blocks.append(("heading", "Activity Summary"))
        blocks.append(("paragraph", narrative.get("suspicious_activity", "")))
        blocks.append(("table", (
            ["Direction", "Currency", "Transactions", "Total (source currency)"],
            [
                [row.get("direction"), row.get("currency"), row.get("count"), row.get("total")]
                for row in activity.get("aggregates") or []
            ],
        )))
        blocks.append(("paragraph", "Amounts are not converted across currencies."))

        blocks.append(("heading", "Transaction Evidence"))
        evidence = content.get("evidence") or []
        blocks.append(("paragraph", narrative.get("evidence_summary", "")))
        blocks.append(("table", (
            ["Reference", "Timestamp", "Direction", "Counterparty", "Amount", "Currency", "Format"],
            [
                [
                    item.get("transaction_ref"),
                    item.get("timestamp"),
                    item.get("direction"),
                    item.get("counterparty"),
                    item.get("amount"),
                    item.get("currency"),
                    item.get("payment_format"),
                ]
                for item in evidence[:40]
            ],
        )))
        if len(evidence) > 40:
            blocks.append(("paragraph", f"Only the first 40 of {len(evidence)} evidence rows are printed. All references are retained in the draft JSON."))

        blocks.append(("heading", "Network Exposure"))
        blocks.append(("paragraph", narrative.get("network_summary", "")))
        blocks.append(("table", (
            ["Feature", "Value"],
            [
                ["Direct seed links", graph.get("hop1_count")],
                ["Two-hop links", graph.get("hop2_count")],
                ["Bidirectional relationships", graph.get("bidir_count")],
                ["Fan score", graph.get("fan_score")],
                ["Daily degree", graph.get("daily_degree")],
            ],
        )))

        blocks.append(("heading", "Investigator Findings"))
        findings = content.get("investigator_findings") or []
        blocks.append(("bullets", [
            f"Note {item.get('note_id')} ({item.get('author')}, {item.get('created_at')}): {item.get('body')}"
            for item in findings
        ] or ["No investigator-confirmed findings recorded."]))

        blocks.append(("heading", "Missing Information"))
        blocks.append(("bullets", [
            field.replace("_", " ") for field in (draft.get("missing_fields") or [])
        ]))
        blocks.append(("paragraph", "These fields are not present in the pipeline source data. They must be completed or explicitly acknowledged by the investigator."))

        blocks.append(("heading", "Policy Validation"))
        blocks.append(("paragraph", "Policy retrieval and regulatory validation are not available in this release. No policy citation or validation claim is made."))

        blocks.append(("heading", "Review and Approval"))
        blocks.append(("table", (
            ["Field", "Value"],
            [
                ["Status", draft.get("status")],
                ["Created by", draft.get("created_by")],
                ["Created at", draft.get("created_at")],
                ["Approved by", draft.get("approved_by")],
                ["Approved at", draft.get("approved_at")],
            ],
        )))
        blocks.append(("paragraph", content.get("disclaimer", "")))

        title = f"Suspicious Activity Report — {case.get('case_id')}"
        subtitle = (
            f"Account {case.get('account_id')} · {case.get('graph_date')} · "
            f"FINDETECT investigator workspace"
        )
        return title, subtitle, blocks

    def export(self, draft_id: int, fmt: str) -> Tuple[bytes, str, str]:
        draft = self.workflow.get_sar_draft(draft_id)
        title, subtitle, blocks = self._blocks(draft)
        status = str(draft.get("status") or "DRAFT")
        if fmt == "docx":
            payload = build_docx(title=title, subtitle=subtitle, status=status, blocks=blocks)
            media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            extension = "docx"
        elif fmt == "pdf":
            pdf = SimplePDF(status=status)
            pdf.heading(title, size=15)
            pdf.paragraph(subtitle)
            pdf.paragraph(f"Status: {status} — draft for investigator review", size=8)
            pdf.spacer()
            for kind, value in blocks:
                if kind == "heading":
                    pdf.heading(value)
                elif kind == "subheading":
                    pdf.subheading(value)
                elif kind == "paragraph":
                    pdf.paragraph(value)
                elif kind == "bullets":
                    pdf.bullets(value or [])
                elif kind == "table":
                    headers, rows = value
                    pdf.table(headers, rows)
                else:
                    pdf.spacer()
            payload = pdf.render()
            media = "application/pdf"
            extension = "pdf"
        else:
            raise InvalidCase("format must be pdf or docx")
        case_id = (draft.get("content") or {}).get("case", {}).get("case_id", "case")
        filename = f"{case_id}-SAR-v{draft.get('version', 1)}-{status.lower()}.{extension}"
        return payload, media, filename

    def policy_validation(self) -> Dict[str, Any]:
        return dict(POLICY_VALIDATION)
