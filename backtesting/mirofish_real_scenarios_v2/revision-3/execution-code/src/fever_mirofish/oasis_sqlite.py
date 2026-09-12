"""Read-only export of OASIS Reddit actions and smoke SimulationResults."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .actor_mapping import resolve_actor_id
from .contracts import canonical_sha256, validate_result


INITIALIZATION_ACTIONS = {"sign_up"}
SYSTEM_ACTIONS = {"refresh", "trend"}
ELICITATION_ACTIONS = {"interview"}
TEXT_ACTIONS = {"create_post", "create_comment", "quote_post", "repost"}


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if not exists:
        return 0
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _parse_info(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_text": raw}
    return value if isinstance(value, dict) else {"value": value}


def _summary_for_action(
    action: str,
    info: Dict[str, Any],
    post_contents: Dict[int, str],
) -> str:
    content = info.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    post_id = info.get("post_id")
    if action == "like_post":
        target = post_contents.get(post_id, "")
        return f"点赞帖子 #{post_id}" + (f"：{target}" if target else "")
    if action == "refresh":
        posts = info.get("posts")
        count = len(posts) if isinstance(posts, list) else 0
        return f"刷新信息流，读取 {count} 条帖子"
    if action == "trend":
        return "读取平台趋势"
    return action


def export_oasis_sqlite(
    database_path: Path,
    config_path: Path,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Export trace rows without mutating the OASIS database.

    MiroFish's current run monitor reads a JSONL file, while the Reddit
    environment persists its authoritative actions in SQLite. This adapter
    treats SQLite as the source of truth and separates seed/system activity
    from autonomous agent behavior.
    """

    database_path = Path(database_path).resolve()
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    agent_entities = {
        int(agent["agent_id"]): str(agent.get("entity_name") or "")
        for agent in config.get("agent_configs") or []
        if isinstance(agent, dict) and "agent_id" in agent
    }
    initial_posts = {
        (int(item["poster_agent_id"]), str(item["content"]))
        for item in (config.get("event_config") or {}).get("initial_posts") or []
        if isinstance(item, dict)
        and "poster_agent_id" in item
        and isinstance(item.get("content"), str)
    }

    connection = sqlite3.connect(
        f"file:{database_path.as_posix()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        table_counts = {
            table: _table_count(connection, table)
            for table in (
                "user",
                "post",
                "comment",
                "like",
                "dislike",
                "follow",
                "trace",
                "refresh",
                "trend",
            )
        }
        post_contents = {
            int(row["post_id"]): str(row["content"] or "")
            for row in connection.execute("SELECT post_id, content FROM post")
        }
        rows = connection.execute(
            """
            SELECT t.rowid AS trace_rowid, t.user_id, u.agent_id, u.name,
                   t.created_at, t.action, t.info
            FROM trace AS t
            LEFT JOIN user AS u ON u.user_id = t.user_id
            ORDER BY t.created_at, t.rowid
            """
        ).fetchall()
    finally:
        connection.close()

    actions = []
    unresolved_entities = set()
    for row in rows:
        agent_id = int(row["agent_id"]) if row["agent_id"] is not None else None
        entity_name = agent_entities.get(agent_id, "")
        actor_id = resolve_actor_id(entity_name, spec["actors"])
        if not actor_id and entity_name:
            unresolved_entities.add(entity_name)

        action = str(row["action"])
        info = _parse_info(row["info"])
        is_seed_post = (
            action == "create_post"
            and agent_id is not None
            and (agent_id, str(info.get("content") or "")) in initial_posts
        )
        if action in INITIALIZATION_ACTIONS or is_seed_post:
            origin = "initialization"
        elif action in SYSTEM_ACTIONS:
            origin = "system"
        elif action in ELICITATION_ACTIONS:
            origin = "elicitation"
        else:
            origin = "autonomous"

        compact_details: Dict[str, Any] = {}
        for key in ("post_id", "comment_id", "like_id"):
            if key in info:
                compact_details[key] = info[key]
        if action == "refresh" and isinstance(info.get("posts"), list):
            compact_details["observed_post_count"] = len(info["posts"])

        actions.append(
            {
                "id": f"trace-{row['trace_rowid']}",
                "trace_rowid": int(row["trace_rowid"]),
                "wall_clock_observed_at": str(row["created_at"]),
                "simulation_round": None,
                "user_id": row["user_id"],
                "agent_id": agent_id,
                "entity_name": entity_name or None,
                "platform_name": row["name"],
                "actor_id": actor_id,
                "action": action,
                "origin": origin,
                "has_text_semantics": action in TEXT_ACTIONS,
                "summary": _summary_for_action(action, info, post_contents),
                "details": compact_details,
            }
        )

    action_counts = Counter(item["action"] for item in actions)
    origin_counts = Counter(item["origin"] for item in actions)
    autonomous = [item for item in actions if item["origin"] == "autonomous"]
    autonomous_text = [item for item in autonomous if item["has_text_semantics"]]
    active_actor_ids = sorted(
        {item["actor_id"] for item in autonomous if item["actor_id"]}
    )
    runtime = config.get("fever_mirofish_runtime") or {}
    return {
        "schema_version": "0.1.0",
        "simulation_id": config.get("simulation_id"),
        "source": "oasis_reddit_sqlite",
        "database_path": str(database_path),
        "config_path": str(config_path),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "round_metadata": {
            "configured_rounds": (
                (config.get("fever_mirofish_smoke_override") or {}).get("rounds")
                or (config.get("time_config") or {}).get("total_simulation_hours")
            ),
            "per_action_round_available": False,
        },
        "randomness": {
            "scheduler_seed": runtime.get("scheduler_seed"),
            "seed_scope": runtime.get(
                "seed_scope",
                "python_random_agent_scheduler_only",
            ),
            "provider_sampling_seeded": bool(
                runtime.get("provider_sampling_seeded", False)
            ),
        },
        "table_counts": table_counts,
        "trace_action_counts": dict(sorted(action_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "autonomous_action_count": len(autonomous),
        "autonomous_text_action_count": len(autonomous_text),
        "autonomous_social_signal_count": len(autonomous) - len(autonomous_text),
        "active_actor_ids": active_actor_ids,
        "unresolved_entity_names": sorted(unresolved_entities),
        "actions": actions,
    }


def build_smoke_simulation_result(
    spec: Dict[str, Any],
    action_export: Dict[str, Any],
    *,
    model: str,
    engine_version: str,
    raw_run_artifacts: list[str],
) -> Dict[str, Any]:
    """Build an honest partial SimulationResult from a compatibility run."""

    autonomous = [
        action
        for action in action_export["actions"]
        if action["origin"] == "autonomous" and action.get("actor_id")
    ]
    events = []
    nodes: list[Dict[str, Any]] = []
    edges: list[Dict[str, Any]] = []
    actor_nodes = sorted({action["actor_id"] for action in autonomous})
    for actor_id in actor_nodes:
        actor = next(item for item in spec["actors"] if item["id"] == actor_id)
        nodes.append({"id": actor_id, "kind": "actor", "label": actor["label"]})
    for action in autonomous:
        action_type = {
            "create_post": "COMMUNICATE",
            "create_comment": "COMMUNICATE",
            "like_post": "SOCIAL_ENDORSEMENT",
        }.get(action["action"], action["action"].upper())
        events.append(
            {
                # OASIS SQLite does not persist the scheduler round per action.
                "round": 0,
                "actor_id": action["actor_id"],
                "action_type": action_type,
                "summary": action["summary"],
                "visibility": "public",
                "evidence_refs": [],
            }
        )
        nodes.append(
            {
                "id": action["id"],
                "kind": "simulated_action",
                "action_type": action_type,
                "summary": action["summary"],
            }
        )
        edges.append(
            {
                "source": action["actor_id"],
                "target": action["id"],
                "relation": "SIMULATED_ACTION",
            }
        )

    warnings = [
        "Integration smoke result only; it is not a calibrated financial forecast.",
        "The upstream run monitor reported zero actions; SQLite was used as the authoritative source.",
        "OASIS did not persist per-action round numbers, so exported events use round 0.",
        "Rule-generated institutional profiles contain consumer-social attributes and require a financial profile compiler.",
        "No forecast target result or scenario frequency was produced from this single run.",
    ]
    if action_export.get("unresolved_entity_names"):
        warnings.append(
            "Some OASIS entities could not be mapped to SimulationSpec actors: "
            + ", ".join(action_export["unresolved_entity_names"])
        )

    randomness = action_export.get("randomness") or {}
    scheduler_seed = randomness.get("scheduler_seed")
    run_warnings = []
    if isinstance(scheduler_seed, int) and not isinstance(scheduler_seed, bool):
        run_warnings.append(
            "The recorded seed controls Python agent scheduling only; "
            "provider-side LLM sampling remains unseeded."
        )
    else:
        scheduler_seed = -1
        run_warnings.append(
            "Random seed was not exposed by the upstream runner; "
            "-1 denotes unavailable."
        )

    result = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial",
        "runs": [
            {
                "run_id": str(action_export.get("simulation_id") or "smoke-run"),
                "seed": scheduler_seed,
                "status": "completed",
                "events": events,
                "warnings": run_warnings,
            }
        ],
        "scenarios": [],
        "forecast_target_results": [],
        "simulation_graph": {"nodes": nodes, "edges": edges},
        "warnings": warnings,
        "provenance": {
            "engine": "MiroFish/OASIS",
            "engine_version": engine_version,
            "model": model,
            "raw_run_artifacts": raw_run_artifacts,
        },
    }
    validate_result(result, spec)
    return result
