"""Compact, auditable summaries of a MiroFish/Zep graph."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


GENERIC_LABELS = {"Entity", "Node"}


def summarize_graph(
    graph_payload: Dict[str, Any],
    *,
    expected_actors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize graph shape and whether every specified actor is represented."""

    nodes = graph_payload.get("nodes") or []
    edges = graph_payload.get("edges") or []
    type_counts: Counter[str] = Counter()
    names_by_type: Dict[str, List[str]] = {}

    for node in nodes:
        labels = [
            str(label)
            for label in (node.get("labels") or [])
            if str(label) not in GENERIC_LABELS
        ]
        for label in labels:
            type_counts[label] += 1
            name = str(node.get("name") or "").strip()
            if name and name not in names_by_type.setdefault(label, []):
                names_by_type[label].append(name)

    actor_coverage = []
    for actor in expected_actors:
        label = str(actor["label"])
        aliases = [label]
        aliases.extend(str(alias) for alias in actor.get("coverage_aliases", []))
        matched_nodes = [
            str(node.get("name") or "")
            for node in nodes
            if any(
                alias.lower() in str(node.get("name") or "").lower()
                or str(node.get("name") or "").lower() in alias.lower()
                for alias in aliases
                if alias and node.get("name")
            )
        ]
        actor_coverage.append(
            {
                "actor_id": actor["id"],
                "label": label,
                "covered": bool(matched_nodes),
                "matched_nodes": sorted(set(matched_nodes)),
            }
        )

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "type_counts": dict(sorted(type_counts.items())),
        "names_by_type": {
            label: sorted(names) for label, names in sorted(names_by_type.items())
        },
        "actor_coverage": actor_coverage,
        "covered_actor_count": sum(item["covered"] for item in actor_coverage),
        "expected_actor_count": len(actor_coverage),
        "missing_actor_ids": [
            item["actor_id"] for item in actor_coverage if not item["covered"]
        ],
    }
