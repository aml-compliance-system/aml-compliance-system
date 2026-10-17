"""Pure, bounded graph algorithms over transaction edges.

No external graph dependency is used. Cycles and paths are time-ordered:
an edge can only be followed by an edge whose timestamp is the same or later.
This makes "A -> B -> C -> A" mean funds could have moved around the cycle in
the selected window, rather than an arbitrary undirected loop.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

_MAX_TIME = datetime.max


def _ts(edge: Mapping[str, Any]) -> datetime:
    value = edge.get("timestamp")
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return _MAX_TIME


def _edge_view(edge: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "transaction_ref": edge.get("transaction_ref") or edge.get("id"),
        "timestamp": edge.get("timestamp"),
        "source": edge.get("source"),
        "target": edge.get("target"),
        "amount": edge.get("amount"),
        "currency": edge.get("currency"),
        "payment_format": edge.get("payment_format"),
        "source_file": edge.get("source_file"),
    }


def _totals(edges: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for edge in edges:
        currency = str(edge.get("currency") or "UNKNOWN")
        bucket = totals.setdefault(currency, {"amount": 0.0, "count": 0})
        bucket["amount"] = round(bucket["amount"] + float(edge.get("amount") or 0.0), 2)
        bucket["count"] = int(bucket["count"]) + 1
    return totals


def _adjacency(edges: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    adjacency: Dict[str, List[Mapping[str, Any]]] = {}
    for edge in edges:
        adjacency.setdefault(str(edge["source"]), []).append(edge)
    for values in adjacency.values():
        values.sort(key=_ts)
    return adjacency


def find_time_ordered_paths(
    edges: Sequence[Mapping[str, Any]],
    source: str,
    target: str,
    *,
    max_depth: int = 4,
    max_paths: int = 20,
    max_visited: int = 1000,
) -> List[Dict[str, Any]]:
    """Find up to ``max_paths`` simple directed paths with non-decreasing time."""
    if source == target:
        return []
    adjacency = _adjacency(edges)
    results: List[Dict[str, Any]] = []
    visited_count = 0

    def walk(
        node: str,
        current_time: datetime,
        path_edges: List[Mapping[str, Any]],
        seen: Set[str],
    ) -> None:
        nonlocal visited_count
        if len(results) >= max_paths or visited_count >= max_visited:
            return
        visited_count += 1
        if node == target and path_edges:
            edge_views = [_edge_view(edge) for edge in path_edges]
            results.append(
                {
                    "algorithm": "time_ordered_paths",
                    "rule": (
                        "Directed simple path; each edge timestamp is greater than or "
                        "equal to the previous edge and all edges fall inside the "
                        "selected window."
                    ),
                    "nodes": [source] + [str(edge["target"]) for edge in path_edges],
                    "edges": edge_views,
                    "start": source,
                    "end": target,
                    "totals_by_currency": _totals(path_edges),
                }
            )
            return
        if len(path_edges) >= max_depth:
            return
        for edge in adjacency.get(node, []):
            edge_time = _ts(edge)
            if edge_time < current_time:
                continue
            next_node = str(edge["target"])
            if next_node in seen:
                continue
            walk(
                next_node,
                edge_time,
                [*path_edges, edge],
                {*seen, next_node},
            )
            if len(results) >= max_paths or visited_count >= max_visited:
                return

    walk(source, datetime.min, [], {source})
    return results


def find_time_ordered_cycles(
    edges: Sequence[Mapping[str, Any]],
    start: str,
    *,
    max_length: int = 4,
    max_cycles: int = 20,
    max_visited: int = 1000,
) -> List[Dict[str, Any]]:
    """Find time-ordered simple cycles beginning and ending at ``start``."""
    adjacency = _adjacency(edges)
    results: List[Dict[str, Any]] = []
    visited_count = 0

    def walk(
        node: str,
        current_time: datetime,
        path_edges: List[Mapping[str, Any]],
        seen: Set[str],
    ) -> None:
        nonlocal visited_count
        if len(results) >= max_cycles or visited_count >= max_visited:
            return
        visited_count += 1
        if node == start and len(path_edges) >= 2:
            edge_views = [_edge_view(edge) for edge in path_edges]
            results.append(
                {
                    "algorithm": "time_ordered_cycles",
                    "rule": (
                        "Directed simple cycle starting and ending at the selected "
                        "account; edge timestamps are non-decreasing, so funds could "
                        "have moved around the loop within the window."
                    ),
                    "nodes": [start] + [str(edge["target"]) for edge in path_edges],
                    "edges": edge_views,
                    "totals_by_currency": _totals(path_edges),
                }
            )
            return
        if len(path_edges) >= max_length:
            return
        for edge in adjacency.get(node, []):
            edge_time = _ts(edge)
            if edge_time < current_time:
                continue
            next_node = str(edge["target"])
            if next_node == start:
                if len(path_edges) + 1 < 2:
                    continue
                walk(start, edge_time, [*path_edges, edge], seen)
            elif next_node not in seen:
                walk(next_node, edge_time, [*path_edges, edge], {*seen, next_node})
            if len(results) >= max_cycles or visited_count >= max_visited:
                return

    for edge in adjacency.get(start, []):
        walk(
            str(edge["target"]),
            _ts(edge),
            [edge],
            {start, str(edge["target"])},
        )
        if len(results) >= max_cycles:
            break
    return results


def shared_counterparties(
    edges: Sequence[Mapping[str, Any]],
    accounts: Sequence[str],
    *,
    min_shared: int = 2,
    max_findings: int = 20,
) -> List[Dict[str, Any]]:
    """Find counterparties connected to at least ``min_shared`` selected accounts."""
    selected = {str(account) for account in accounts}
    mapping: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        if source in selected and target not in selected:
            mapping.setdefault(target, {}).setdefault(source, []).append(edge)
        elif target in selected and source not in selected:
            mapping.setdefault(source, {}).setdefault(target, []).append(edge)

    findings = []
    for counterparty, per_account in mapping.items():
        if len(per_account) < min_shared:
            continue
        evidence = []
        for account, items in per_account.items():
            for edge in items:
                evidence.append(
                    {
                        "account": account,
                        "counterparty": counterparty,
                        "transaction_ref": edge.get("transaction_ref") or edge.get("id"),
                        "timestamp": edge.get("timestamp"),
                        "direction": "out" if str(edge["source"]) == account else "in",
                        "amount": edge.get("amount"),
                        "currency": edge.get("currency"),
                        "source_file": edge.get("source_file"),
                    }
                )
        findings.append(
            {
                "counterparty": counterparty,
                "account_count": len(per_account),
                "accounts": sorted(per_account.keys()),
                "evidence": evidence[:100],
            }
        )
    findings.sort(key=lambda item: (-item["account_count"], item["counterparty"]))
    return findings[:max_findings]


def connected_component(
    edges: Sequence[Mapping[str, Any]], start: str
) -> Dict[str, Any]:
    """Undirected connected component containing ``start`` in the supplied edges."""
    adjacency: Dict[str, Set[str]] = {}
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    if start not in adjacency:
        return {"nodes": [start], "edges": [], "size": 1}
    seen = {start}
    queue = [start]
    while queue:
        node = queue.pop()
        for neighbour in adjacency.get(node, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    component_edges = [
        edge for edge in edges
        if str(edge["source"]) in seen and str(edge["target"]) in seen
    ]
    return {
        "nodes": sorted(seen),
        "edges": component_edges,
        "size": len(seen),
        "totals_by_currency": _totals(component_edges),
    }


def nodes_and_edges(
    edges: Sequence[Mapping[str, Any]],
    focal: str,
    *,
    holders: Optional[Mapping[str, Mapping[str, Any]]] = None,
    include_holders: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build UI graph nodes/edges with optional holder nodes."""
    adjacency: Dict[str, Set[str]] = {focal: set()}
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    hops: Dict[str, int] = {focal: 0}
    queue = deque([focal])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency.get(node, ()):
            if neighbour not in hops:
                hops[neighbour] = hops[node] + 1
                queue.append(neighbour)
    for account in adjacency:
        hops.setdefault(account, 1)

    nodes = []
    for account, hop in sorted(hops.items(), key=lambda item: (item[1], item[0])):
        info = (holders or {}).get(account) or {}
        nodes.append(
            {
                "id": account,
                "type": "account",
                "label": account,
                "account_id": account,
                "holder_name": info.get("name"),
                "entity_id": info.get("entity_id"),
                "bank": info.get("bank"),
                "is_focal": account == focal,
                "hop": int(hop),
            }
        )
    graph_edges = [
        {
            "id": str(edge.get("id")),
            "source": str(edge["source"]),
            "target": str(edge["target"]),
            "type": "money",
            "timestamp": edge.get("timestamp"),
            "amount": edge.get("amount"),
            "currency": edge.get("currency"),
            "payment_format": edge.get("payment_format"),
            "transaction_ref": edge.get("transaction_ref") or edge.get("id"),
            "source_file": edge.get("source_file"),
        }
        for edge in edges
    ]
    if include_holders:
        existing = {node["id"] for node in nodes}
        for account, info in (holders or {}).items():
            if account not in hops or not (info.get("entity_id") or info.get("name")):
                continue
            holder_id = f"HOLDER:{info.get('entity_id') or info.get('name')}"
            if holder_id not in existing:
                nodes.append(
                    {
                        "id": holder_id,
                        "type": "holder",
                        "label": str(info.get("name") or info.get("entity_id")),
                        "account_id": None,
                        "holder_name": info.get("name"),
                        "entity_id": info.get("entity_id"),
                        "bank": None,
                        "is_focal": False,
                        "hop": 1,
                    }
                )
                existing.add(holder_id)
            graph_edges.append(
                {
                    "id": f"HAS-{account}-{holder_id}",
                    "source": account,
                    "target": holder_id,
                    "type": "holder",
                    "timestamp": None,
                    "amount": None,
                    "currency": None,
                    "payment_format": None,
                    "transaction_ref": None,
                    "source_file": None,
                }
            )
    return nodes, graph_edges
