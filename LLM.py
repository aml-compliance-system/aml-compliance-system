# llm.py
import json

def generate_report(structured_case: dict) -> str:
    """
    Local deterministic SAR template generator.

    The payload is produced from the account/day FINDETECT
    risk and graph results.
    """

    case_id = structured_case.get(
        "case_id",
        "CASE-UNKNOWN"
    )

    customer = structured_case.get(
        "customer",
        {}
    )

    cust_id = customer.get(
        "customer_id",
        "N/A"
    )

    name = customer.get(
        "name_masked",
        "Redacted Entity"
    )

    segment = customer.get(
        "segment",
        "Transaction Monitoring"
    )

    risk_score = float(
        structured_case.get(
            "risk_score",
            0.0
        )
    )

    risk_tier = structured_case.get(
        "risk_tier",
        "UNKNOWN"
    )

    # ---------------------------------------------
    # Reason codes
    # ---------------------------------------------
    reason_codes = structured_case.get(
        "reason_codes",
        []
    )

    if reason_codes:
        rc_text = "\n".join(
            [
                f"- `{rc.get('code', 'RC-UNKNOWN')}`: "
                f"{rc.get('description', 'No description')}"
                for rc in reason_codes
            ]
        )
    else:
        rc_text = "- No reason codes recorded."

    # ---------------------------------------------
    # Risk / graph telemetry
    # ---------------------------------------------
    model_scores = structured_case.get(
        "model_scores",
        {}
    )

    base_risk = float(
        model_scores.get(
            "combined_base_risk",
            0.0
        )
    )

    graph_features = structured_case.get(
        "graph_features",
        {}
    )

    graph_penalty = float(
        graph_features.get(
            "graph_penalty",
            0.0
        )
    )

    is_seed = bool(
        graph_features.get(
            "is_high_risk_seed",
            False
        )
    )

    received_from_high_risk = bool(
        graph_features.get(
            "received_from_high_risk",
            False
        )
    )

    graph_date = graph_features.get(
        "graph_date",
        "N/A"
    )

    # ---------------------------------------------
    # Transaction / graph evidence
    # ---------------------------------------------
    transactions = structured_case.get(
        "transactions",
        []
    )

    tx_lines = []

    for tx in transactions:
        ts = tx.get(
            "timestamp",
            graph_date
        )

        from_acc = tx.get(
            "from_account",
            "N/A"
        )

        to_acc = tx.get(
            "to_account",
            "N/A"
        )

        amount = float(
            tx.get(
                "amount_sar",
                0.0
            )
        )

        direction = tx.get(
            "direction",
            "unknown"
        )

        tx_lines.append(
            f"- {direction.title()} transaction of "
            f"{amount:,.2f} SAR from {from_acc} "
            f"to {to_acc} on {ts}."
        )

    if tx_lines:
        tx_text = "\n".join(tx_lines)
    else:
        tx_text = (
            "- No direct graph exposure transaction "
            "was recorded for this case."
        )

    # ---------------------------------------------
    # Report
    # ---------------------------------------------
    report = f"""**Case & Entity Overview**:
- **Case ID**: {case_id}
- **Customer**: {cust_id} ({name}) | Segment: {segment}
- **Graph Date**: {graph_date}
- **Risk Assessment**: Score {risk_score:.3f} (Tier: {risk_tier})

**Triggered Reason Codes**:
{rc_text}

**Risk & Graph Intelligence**:
- **Combined Base ML Risk**: {base_risk:.3f}
- **Graph Penalty**: {graph_penalty:.3f}
- **Original High-Risk Seed**: {is_seed}
- **Received Directly From High-Risk Seed**: {received_from_high_risk}

**Observed Financial / Network Evidence**:
{tx_text}

**Data Limitations**:
- Risk is generated from the FINDETECT account/day pipeline.
- Graph propagation is restricted to direct one-hop exposure from original high-risk seeds.
- A graph connection is a risk indicator and does not by itself establish money laundering.

Notice: This is an automated preliminary risk analysis generated for investigator decision-support and requires manual human review."""

    return report


