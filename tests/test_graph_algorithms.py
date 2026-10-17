"""Deterministic tests for bounded graph algorithms."""

from __future__ import annotations

from graph_algorithms import (
    connected_component,
    find_time_ordered_cycles,
    find_time_ordered_paths,
    nodes_and_edges,
    shared_counterparties,
)


def edge(ref, source, target, timestamp, amount=10.0, currency="USD"):
    return {
        "id": ref,
        "transaction_ref": ref,
        "source": source,
        "target": target,
        "timestamp": timestamp,
        "amount": amount,
        "currency": currency,
        "payment_format": "Wire",
        "source_file": "tx.parquet",
    }


EDGES = [
    edge("TX-1", "A", "B", "2022-01-01T09:00:00", 100),
    edge("TX-2", "B", "C", "2022-01-01T10:00:00", 90),
    edge("TX-3", "C", "A", "2022-01-01T11:00:00", 80),
    edge("TX-4", "A", "C", "2022-01-01T08:00:00", 20),
    edge("TX-5", "D", "C", "2022-01-01T09:30:00", 15),
]


def test_time_ordered_paths():
    paths = find_time_ordered_paths(EDGES, "A", "C", max_depth=3)
    assert paths
    via_b = next(path for path in paths if path["nodes"] == ["A", "B", "C"])
    assert [edge["transaction_ref"] for edge in via_b["edges"]] == ["TX-1", "TX-2"]
    assert via_b["totals_by_currency"]["USD"]["amount"] == 190.0


def test_paths_respect_time_order():
    edges = [
        edge("TX-LATE", "A", "B", "2022-01-01T11:00:00"),
        edge("TX-EARLY", "B", "C", "2022-01-01T09:00:00"),
    ]
    assert find_time_ordered_paths(edges, "A", "C", max_depth=3) == []


def test_cycle_detection_is_time_ordered():
    cycles = find_time_ordered_cycles(EDGES, "A", max_length=4)
    assert len(cycles) == 2  # A->B->C->A and A->C->A
    via_b = next(cycle for cycle in cycles if cycle["nodes"] == ["A", "B", "C", "A"])
    assert [edge["transaction_ref"] for edge in via_b["edges"]] == ["TX-1", "TX-2", "TX-3"]


def test_no_cycle_when_time_reverses():
    edges = [
        edge("TX-A", "A", "B", "2022-01-01T11:00:00"),
        edge("TX-B", "B", "A", "2022-01-01T10:00:00"),
    ]
    assert find_time_ordered_cycles(edges, "A") == []


def test_shared_counterparties():
    findings = shared_counterparties(EDGES, ["A", "B"], min_shared=2)
    # A and B both touch C, but C itself is not a third-party counterparty of both
    # in this fixture (A->C, B->C) -> yes, C is shared.
    assert findings
    assert findings[0]["counterparty"] == "C"
    assert findings[0]["account_count"] == 2
    assert {item["account"] for item in findings[0]["evidence"]} == {"A", "B"}


def test_connected_component():
    component = connected_component(EDGES, "A")
    assert component["nodes"] == ["A", "B", "C", "D"]
    assert component["size"] == 4
    assert len(component["edges"]) == 5


def test_nodes_and_edges_hops_and_holders():
    nodes, graph_edges = nodes_and_edges(
        EDGES,
        "A",
        holders={"B": {"name": "Beta Corp", "entity_id": "E-B"}},
        include_holders=True,
    )
    by_id = {node["id"]: node for node in nodes}
    assert by_id["A"]["is_focal"] is True
    assert by_id["B"]["hop"] == 1
    assert by_id["C"]["hop"] == 1
    assert by_id["D"]["hop"] == 2
    assert "HOLDER:E-B" in by_id
    assert any(edge["type"] == "holder" for edge in graph_edges)
