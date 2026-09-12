"""Bounded, extractive interaction input; never infer a change from silence."""

from __future__ import annotations

import json
from typing import Any


OPERATIONAL_ACTIONS = {
    "SEARCH_POSTS", "SEARCH_USER", "LIKE_COMMENT", "LIKE_POST", "FOLLOW",
    "DO_NOTHING", "SOCIAL_ENDORSEMENT",
}


def compact_interactions(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = result.get("simulation_graph") or {}
    actor_by_node = {
        edge["target"]: edge["source"]
        for edge in graph.get("edges", [])
        if edge.get("relation") == "SIMULATED_ACTION"
    }
    raw = [node for node in graph.get("nodes", []) if node.get("kind") == "simulated_action"]
    kept = []
    seen = set()
    operational = duplicates = empty = 0
    # Prefer recent messages within each role, then restore recorded order.
    per_actor: dict[str, int] = {}
    budget_dropped = 0
    chars = 0
    for index, node in reversed(list(enumerate(raw))):
        action_type = str(node.get("action_type") or "")
        summary = str(node.get("summary") or "").strip()
        actor_id = actor_by_node.get(node.get("id"))
        if action_type in OPERATIONAL_ACTIONS:
            operational += 1
            continue
        if not summary:
            empty += 1
            continue
        key = (actor_id, action_type, " ".join(summary.split()))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if len(kept) >= 32 or per_actor.get(actor_id or "unknown", 0) >= 3 or chars >= 6000:
            budget_dropped += 1
            continue
        text = summary[:min(800, 6000 - chars)]
        item = {
            "id": node["id"], "actor_id": actor_id,
            "action_type": action_type, "summary": text,
        }
        if len(text) < len(summary):
            item["summary_truncated"] = True
        kept.append((index, item))
        chars += len(text)
        per_actor[actor_id or "unknown"] = per_actor.get(actor_id or "unknown", 0) + 1
    selected = [item for _, item in sorted(kept)]
    return selected, {
        "version": "extractive-interactions-v2",
        "input_count": len(raw), "retained_count": len(selected),
        "operational_count": operational, "duplicate_count": duplicates,
        "empty_count": empty, "budget_dropped_count": budget_dropped,
        "retained_summary_chars": chars,
        "retained_input_chars": len(json.dumps(selected, ensure_ascii=False)),
        "semantics": "selected_simulated_messages_not_observed_facts_or_inferred_influence",
    }
