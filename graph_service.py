"""Bounded, evidence-backed graph exploration for one case."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import config
from case_service import CaseService, InvalidCase, RunInfo, normalise_account
from graph_algorithms import (
    connected_component,
    find_time_ordered_cycles,
    find_time_ordered_paths,
    nodes_and_edges,
    shared_counterparties,
)


@dataclass
class GraphWindow:
    start: datetime
    end: datetime

    def as_dict(self) -> Dict[str, str]:
        return {
            "from": self.start.isoformat(timespec="seconds"),
            "to": self.end.isoformat(timespec="seconds"),
        }


class GraphService:
    def __init__(self, cases: CaseService):
        self.cases = cases

    # -- resolution -------------------------------------------------------
    def _resolve(self, case_id: str, run_id: Optional[str]) -> Tuple[RunInfo, str, str]:
        run = self.cases.get_run(run_id)
        row = self.cases._case_rows(run, case_id)  # same validated lookup
        return run, str(row["account_id"]), str(row["graph_date"])

    def _window(
        self,
        day: str,
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
    ) -> GraphWindow:
        start, end = self.cases._window(day, from_value, to_value)
        return GraphWindow(start=start, end=end)

    # -- bounded loading --------------------------------------------------
    def _load_neighborhood(
        self,
        run: RunInfo,
        focal: str,
        window: GraphWindow,
        *,
        depth: int,
        direction: str = "both",
        max_nodes: int = config.GRAPH_MAX_NODES,
        max_edges: int = config.GRAPH_MAX_EDGES,
    ) -> Dict[str, Any]:
        depth = max(1, min(int(depth), max(config.GRAPH_MAX_PATH_DEPTH, 5)))
        visited: Set[str] = {focal}
        frontier: Set[str] = {focal}
        edges: List[Dict[str, Any]] = []
        seen_edges: Set[str] = set()
        truncated = False
        reason = None

        for _ in range(depth):
            if not frontier:
                break
            batch = self.cases.edges_for_accounts(
                run,
                sorted(frontier),
                window.start,
                window.end,
                limit=max_edges + 1,
                direction=direction,
            )
            if len(batch) > max_edges:
                batch = batch[:max_edges]
                truncated = True
                reason = (
                    "The graph query exceeded the edge limit. Narrow the time or "
                    "amount filters, or expand one node at a time."
                )
            next_frontier: Set[str] = set()
            for edge in batch:
                edge_id = str(edge.get("id"))
                if edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)
                edges.append(edge)
                for account in (str(edge["source"]), str(edge["target"])):
                    if account not in visited:
                        next_frontier.add(account)
                if len(edges) >= max_edges:
                    truncated = True
                    reason = reason or "Graph edge limit reached."
                    break
            if len(edges) >= max_edges:
                break
            visited.update(next_frontier)
            frontier = next_frontier
            if len(visited) > max_nodes:
                truncated = True
                reason = reason or "Graph node limit reached."
                break

        # Keep the graph inside the node cap, preserving the lowest-hop nodes.
        nodes, graph_edges = nodes_and_edges(edges, focal)
        if len(nodes) > max_nodes:
            allowed = {node["id"] for node in nodes[:max_nodes]}
            graph_edges = [
                edge for edge in graph_edges
                if edge["source"] in allowed and edge["target"] in allowed
            ]
            nodes = [node for node in nodes if node["id"] in allowed]
            truncated = True
            reason = reason or "Graph node limit reached."
        return {
            "edges": edges,
            "graph_edges": graph_edges,
            "nodes": nodes,
            "truncated": truncated,
            "truncation_reason": reason,
        }

    # -- graph responses --------------------------------------------------
    def case_graph(
        self,
        case_id: str,
        *,
        run_id: Optional[str] = None,
        depth: int = 1,
        limit_nodes: int = config.GRAPH_MAX_NODES,
        limit_edges: int = config.GRAPH_MAX_EDGES,
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
        include_holders: bool = False,
    ) -> Dict[str, Any]:
        run, account, day = self._resolve(case_id, run_id)
        window = self._window(day, from_value, to_value)
        max_nodes = max(1, min(int(limit_nodes), config.GRAPH_MAX_NODES))
        max_edges = max(1, min(int(limit_edges), config.GRAPH_MAX_EDGES))
        loaded = self._load_neighborhood(
            run, account, window, depth=depth, max_nodes=max_nodes, max_edges=max_edges
        )
        node_accounts = [
            node["id"] for node in loaded["nodes"] if node["type"] == "account"
        ]
        names = self.cases.holder_names(run, node_accounts)
        holders = {key: {"name": value} for key, value in names.items() if value}
        nodes, graph_edges = nodes_and_edges(
            loaded["edges"], account, holders=holders, include_holders=include_holders
        )
        if len(nodes) > max_nodes:
            allowed = {node["id"] for node in nodes[:max_nodes]}
            graph_edges = [
                edge for edge in graph_edges
                if edge["source"] in allowed and edge["target"] in allowed
            ]
            nodes = [node for node in nodes if node["id"] in allowed]
            loaded["truncated"] = True
        return {
            "run_id": run.run_id,
            "case_id": case_id,
            "window": window.as_dict(),
            "nodes": nodes,
            "edges": graph_edges,
            "truncated": bool(loaded["truncated"]),
            "truncation_reason": loaded.get("truncation_reason"),
            "limits": {
                "max_nodes": max_nodes,
                "max_edges": max_edges,
                "max_depth": min(int(depth), config.GRAPH_MAX_DEPTH),
            },
            "note": (
                "Edges are real transactions in the selected window. Holder nodes "
                "come from source-record entity names, not a verified KYC master. "
                "A graph connection is a risk indicator, not proof of laundering."
            ),
        }

    def expand(
        self,
        case_id: str,
        account: str,
        *,
        run_id: Optional[str] = None,
        direction: str = "both",
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        run, _, day = self._resolve(case_id, run_id)
        account = normalise_account(account)
        window = self._window(day, from_value, to_value)
        edges = self.cases.edges_for_accounts(
            run,
            [account],
            window.start,
            window.end,
            limit=config.GRAPH_MAX_EDGES,
            direction=direction,
        )
        nodes, graph_edges = nodes_and_edges(edges, account)
        return {
            "run_id": run.run_id,
            "case_id": case_id,
            "window": window.as_dict(),
            "nodes": nodes,
            "edges": graph_edges,
            "truncated": len(edges) >= config.GRAPH_MAX_EDGES,
            "truncation_reason": (
                "Edge limit reached for this account; narrow the time window."
                if len(edges) >= config.GRAPH_MAX_EDGES else None
            ),
            "limits": {
                "max_nodes": config.GRAPH_MAX_NODES,
                "max_edges": config.GRAPH_MAX_EDGES,
                "max_depth": 1,
            },
            "note": "One-hop expansion for the selected account.",
        }

    def paths(
        self,
        case_id: str,
        *,
        source: str,
        target: str,
        run_id: Optional[str] = None,
        max_depth: int = 4,
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        run, focal, day = self._resolve(case_id, run_id)
        source = normalise_account(source)
        target = normalise_account(target)
        window = self._window(day, from_value, to_value)
        max_depth = max(1, min(int(max_depth), config.GRAPH_MAX_PATH_DEPTH))
        loaded = self._load_neighborhood(
            run,
            source,
            window,
            depth=max_depth,
            max_nodes=config.GRAPH_MAX_NODES,
            max_edges=config.GRAPH_MAX_EDGES,
        )
        findings = find_time_ordered_paths(
            loaded["edges"],
            source,
            target,
            max_depth=max_depth,
            max_paths=config.GRAPH_MAX_PATHS,
            max_visited=config.GRAPH_MAX_VISITED,
        )
        return {
            "run_id": run.run_id,
            "case_id": case_id,
            "window": window.as_dict(),
            "algorithm": "time_ordered_paths",
            "rule": (
                "Directed simple paths with non-decreasing transaction timestamps. "
                "A path is a money-flow lead, not proof of laundering."
            ),
            "findings": findings,
            "truncated": bool(loaded["truncated"]),
            "limits": {
                "max_depth": max_depth,
                "max_paths": config.GRAPH_MAX_PATHS,
                "max_edges": config.GRAPH_MAX_EDGES,
            },
        }

    def cycles(
        self,
        case_id: str,
        *,
        run_id: Optional[str] = None,
        max_length: int = 4,
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        run, focal, day = self._resolve(case_id, run_id)
        window = self._window(day, from_value, to_value)
        max_length = max(2, min(int(max_length), 5))
        loaded = self._load_neighborhood(
            run,
            focal,
            window,
            depth=max_length,
            max_nodes=config.GRAPH_MAX_NODES,
            max_edges=config.GRAPH_MAX_EDGES,
        )
        findings = find_time_ordered_cycles(
            loaded["edges"],
            focal,
            max_length=max_length,
            max_cycles=config.GRAPH_MAX_CYCLES,
            max_visited=config.GRAPH_MAX_VISITED,
        )
        return {
            "run_id": run.run_id,
            "case_id": case_id,
            "window": window.as_dict(),
            "algorithm": "time_ordered_cycles",
            "rule": (
                "Directed simple cycles starting and ending at the case account, "
                "with non-decreasing timestamps. A cycle is a layering lead, not "
                "evidence of intent."
            ),
            "findings": findings,
            "truncated": bool(loaded["truncated"]),
            "limits": {
                "max_length": max_length,
                "max_cycles": config.GRAPH_MAX_CYCLES,
                "max_edges": config.GRAPH_MAX_EDGES,
            },
        }

    def shared(
        self,
        case_id: str,
        *,
        accounts: Sequence[str],
        run_id: Optional[str] = None,
        min_shared: int = 2,
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        run, _, day = self._resolve(case_id, run_id)
        window = self._window(day, from_value, to_value)
        cleaned = [normalise_account(account) for account in accounts]
        if len(cleaned) < 2:
            raise InvalidCase("At least two accounts are required")
        edges = self.cases.edges_for_accounts(
            run,
            cleaned,
            window.start,
            window.end,
            limit=config.GRAPH_MAX_EDGES,
            direction="both",
        )
        findings = shared_counterparties(
            edges, cleaned, min_shared=min_shared, max_findings=20
        )
        return {
            "run_id": run.run_id,
            "case_id": case_id,
            "window": window.as_dict(),
            "algorithm": "shared_counterparties",
            "rule": (
                "Counterparties connected to at least the requested number of "
                "selected accounts in the window, using real transaction edges."
            ),
            "findings": findings,
            "truncated": len(edges) >= config.GRAPH_MAX_EDGES,
            "limits": {
                "max_edges": config.GRAPH_MAX_EDGES,
                "min_shared": min_shared,
            },
        }

    def components(
        self,
        case_id: str,
        *,
        run_id: Optional[str] = None,
        account: Optional[str] = None,
        from_value: Optional[str] = None,
        to_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        run, focal, day = self._resolve(case_id, run_id)
        start = normalise_account(account) if account else focal
        window = self._window(day, from_value, to_value)
        loaded = self._load_neighborhood(
            run,
            start,
            window,
            depth=2,
            max_nodes=config.GRAPH_MAX_NODES,
            max_edges=config.GRAPH_MAX_EDGES,
        )
        component = connected_component(loaded["edges"], start)
        findings = [
            {
                "algorithm": "connected_component",
                "rule": (
                    "Undirected connected component in the selected window. "
                    "A shared component is a network lead, not proof of a common owner."
                ),
                "nodes": component["nodes"],
                "size": component["size"],
                "edges": [
                    {
                        "transaction_ref": edge.get("transaction_ref") or edge.get("id"),
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "timestamp": edge.get("timestamp"),
                        "amount": edge.get("amount"),
                        "currency": edge.get("currency"),
                    }
                    for edge in component["edges"][:200]
                ],
                "totals_by_currency": component.get("totals_by_currency", {}),
            }
        ]
        return {
            "run_id": run.run_id,
            "case_id": case_id,
            "window": window.as_dict(),
            "algorithm": "connected_component",
            "rule": "Undirected connected component around the selected account.",
            "findings": findings,
            "truncated": bool(loaded["truncated"]),
            "limits": {
                "max_nodes": config.GRAPH_MAX_NODES,
                "max_edges": config.GRAPH_MAX_EDGES,
            },
        }
