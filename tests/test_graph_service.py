"""Graph service tests against the synthetic case/transaction fixture."""

from __future__ import annotations

from graph_service import GraphService


def test_case_graph_is_case_scoped_and_bounded(synthetic_cases):
    graph = GraphService(synthetic_cases)
    result = graph.case_graph("CASE-A1-20220101", depth=1)
    assert result["case_id"] == "CASE-A1-20220101"
    assert len(result["nodes"]) == 3
    assert len(result["edges"]) == 4
    assert result["truncated"] is False
    assert all(edge["type"] == "money" for edge in result["edges"])
    assert all(edge["transaction_ref"].startswith("TX-") for edge in result["edges"])


def test_graph_time_filter_changes_edges(synthetic_cases):
    graph = GraphService(synthetic_cases)
    result = graph.case_graph(
        "CASE-A1-20220101", from_value="2022-01-01T09:30:00", to_value="2022-01-01T10:30:00"
    )
    assert len(result["edges"]) == 1
    assert result["edges"][0]["amount"] == 300.0
    assert result["edges"][0]["source"] == "A1"
    assert result["edges"][0]["target"] == "A3"


def test_paths_are_directed_and_time_ordered(synthetic_cases):
    graph = GraphService(synthetic_cases)
    result = graph.paths("CASE-A1-20220101", source="A2", target="A3", max_depth=3)
    assert len(result["findings"]) == 1
    assert result["findings"][0]["nodes"] == ["A2", "A1", "A3"]
    assert len(result["findings"][0]["edges"]) == 2


def test_cycle_detection(synthetic_cases):
    graph = GraphService(synthetic_cases)
    result = graph.cycles("CASE-A1-20220101", max_length=3)
    assert len(result["findings"]) == 1
    assert result["findings"][0]["nodes"] == ["A1", "A3", "A1"]


def test_components_and_shared(synthetic_cases):
    graph = GraphService(synthetic_cases)
    component = graph.components("CASE-A1-20220101")
    assert component["findings"][0]["size"] == 3
    shared = graph.shared("CASE-A1-20220101", accounts=["A1", "A2"])
    assert isinstance(shared["findings"], list)


def test_expand_and_holders(synthetic_cases):
    graph = GraphService(synthetic_cases)
    expanded = graph.expand("CASE-A1-20220101", "A3")
    assert len(expanded["edges"]) == 2
    with_holders = graph.case_graph("CASE-A1-20220101", include_holders=True)
    assert any(node["type"] == "holder" for node in with_holders["nodes"])
