"""Compile stable finance-oriented OASIS profiles from a SimulationSpec."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from .actor_mapping import map_agent_configs_to_actors
from .contracts import canonical_sha256, validate_spec


OVERLAY_VERSION = "0.2.0"

ACTIVITY_BY_KIND: Dict[str, Dict[str, Any]] = {
    "government": {
        "activity_level": 0.35,
        "posts_per_hour": 0.2,
        "comments_per_hour": 0.1,
        "active_hours": list(range(8, 19)),
        "response_delay_min": 60,
        "response_delay_max": 240,
        "stance": "neutral",
        "influence_weight": 3.0,
    },
    "regulator": {
        "activity_level": 0.4,
        "posts_per_hour": 0.25,
        "comments_per_hour": 0.15,
        "active_hours": list(range(8, 19)),
        "response_delay_min": 60,
        "response_delay_max": 240,
        "stance": "neutral",
        "influence_weight": 3.0,
    },
    "exchange": {
        "activity_level": 0.4,
        "posts_per_hour": 0.25,
        "comments_per_hour": 0.15,
        "active_hours": list(range(8, 19)),
        "response_delay_min": 45,
        "response_delay_max": 180,
        "stance": "neutral",
        "influence_weight": 2.8,
    },
    "broker": {
        "activity_level": 0.7,
        "posts_per_hour": 0.6,
        "comments_per_hour": 1.0,
        "active_hours": list(range(8, 23)),
        "response_delay_min": 5,
        "response_delay_max": 30,
        "stance": "neutral",
        "influence_weight": 1.8,
    },
    "institutional_investor": {
        "activity_level": 0.6,
        "posts_per_hour": 0.4,
        "comments_per_hour": 0.6,
        "active_hours": list(range(8, 22)),
        "response_delay_min": 15,
        "response_delay_max": 60,
        "stance": "neutral",
        "influence_weight": 2.0,
    },
    "labor_union": {
        "activity_level": 0.65,
        "posts_per_hour": 0.45,
        "comments_per_hour": 0.55,
        "active_hours": list(range(6, 23)),
        "response_delay_min": 10,
        "response_delay_max": 60,
        "stance": "neutral",
        "influence_weight": 2.6,
    },
    "foreign_investor": {
        "activity_level": 0.55,
        "posts_per_hour": 0.35,
        "comments_per_hour": 0.5,
        "active_hours": list(range(8, 22)),
        "response_delay_min": 20,
        "response_delay_max": 90,
        "stance": "neutral",
        "influence_weight": 2.1,
    },
    "retail_cohort": {
        "activity_level": 0.9,
        "posts_per_hour": 1.0,
        "comments_per_hour": 2.0,
        "active_hours": list(range(8, 24)),
        "response_delay_min": 1,
        "response_delay_max": 15,
        "stance": "neutral",
        "influence_weight": 1.0,
    },
    "issuer": {
        "activity_level": 0.4,
        "posts_per_hour": 0.2,
        "comments_per_hour": 0.15,
        "active_hours": list(range(8, 19)),
        "response_delay_min": 60,
        "response_delay_max": 240,
        "stance": "neutral",
        "influence_weight": 2.3,
    },
    "media": {
        "activity_level": 0.8,
        "posts_per_hour": 1.0,
        "comments_per_hour": 1.5,
        "active_hours": list(range(7, 24)),
        "response_delay_min": 5,
        "response_delay_max": 30,
        "stance": "observer",
        "influence_weight": 2.4,
    },
    "analyst": {
        "activity_level": 0.7,
        "posts_per_hour": 0.8,
        "comments_per_hour": 1.0,
        "active_hours": list(range(7, 23)),
        "response_delay_min": 5,
        "response_delay_max": 45,
        "stance": "observer",
        "influence_weight": 2.0,
    },
    "supplier": {
        "activity_level": 0.45,
        "posts_per_hour": 0.25,
        "comments_per_hour": 0.25,
        "active_hours": list(range(7, 21)),
        "response_delay_min": 30,
        "response_delay_max": 180,
        "stance": "neutral",
        "influence_weight": 2.0,
    },
    "customer": {
        "activity_level": 0.65,
        "posts_per_hour": 0.6,
        "comments_per_hour": 1.2,
        "active_hours": list(range(7, 24)),
        "response_delay_min": 2,
        "response_delay_max": 30,
        "stance": "neutral",
        "influence_weight": 1.2,
    },
    "competitor": {
        "activity_level": 0.5,
        "posts_per_hour": 0.3,
        "comments_per_hour": 0.4,
        "active_hours": list(range(7, 22)),
        "response_delay_min": 20,
        "response_delay_max": 90,
        "stance": "neutral",
        "influence_weight": 1.8,
    },
}

ENTITY_TYPE_BY_KIND = {
    "government": "GovernmentAgency",
    "regulator": "Regulator",
    "exchange": "Exchange",
    "broker": "Broker",
    "institutional_investor": "InstitutionalInvestor",
    "labor_union": "LaborUnion",
    "foreign_investor": "InstitutionalInvestor",
    "retail_cohort": "RetailInvestor",
    "issuer": "Issuer",
    "media": "FinancialMedia",
    "analyst": "Analyst",
    "supplier": "Supplier",
    "customer": "Customer",
    "competitor": "Competitor",
}


def _profile_persona(actor: Dict[str, Any], spec: Dict[str, Any]) -> str:
    facts = {
        fact["id"]: fact["statement"]
        for fact in spec["facts"]
        if fact["id"] in actor.get("observable_fact_ids", [])
    }
    parts = [
        f"身份：{actor['label']}（{actor['kind']}，{actor['aggregation']}）。",
        "目标：" + "；".join(actor["goals"]) + "。",
        "约束：" + ("；".join(actor["constraints"]) or "无额外约束") + "。",
        "截至信息：" + "；".join(f"{key}={value}" for key, value in facts.items()) + "。",
        "假设：" + ("；".join(actor["assumptions"]) or "无额外假设") + "。",
        "行为规则：只使用截至时点可见事实；区分公开表达与实际金融动作；"
        "不得把社交热度直接解释为价格或概率。年龄、性别和MBTI仅为OASIS兼容占位符，"
        "不参与决策。",
    ]
    return "".join(parts)


def compile_financial_actor_overlay(
    config: Dict[str, Any],
    spec: Dict[str, Any],
    *,
    actor_ids: tuple[str, ...] | None = None,
) -> tuple[Dict[str, Any], list[Dict[str, Any]], Dict[str, Any]]:
    """Return a deduplicated config, deterministic profiles, and metadata."""

    validate_spec(spec)
    requested_actor_ids = (
        list(actor_ids)
        if actor_ids is not None
        else [actor["id"] for actor in spec["actors"]]
    )
    all_actor_ids = {actor["id"] for actor in spec["actors"]}
    if (
        len(requested_actor_ids) < 2
        or len(requested_actor_ids) != len(set(requested_actor_ids))
        or not set(requested_actor_ids) <= all_actor_ids
    ):
        raise ValueError(
            "financial actor selection must contain at least two unique spec actors"
        )
    selected_actor_ids = [
        actor["id"]
        for actor in spec["actors"]
        if actor["id"] in set(requested_actor_ids)
    ]
    selected_actor_id_set = set(selected_actor_ids)
    selected_actors = [
        actor
        for actor in spec["actors"]
        if actor["id"] in selected_actor_id_set
    ]
    source_configs = config.get("agent_configs") or []
    old_to_actor, unresolved = map_agent_configs_to_actors(
        source_configs,
        spec["actors"],
    )
    source_by_actor: Dict[str, list[Dict[str, Any]]] = {
        actor["id"]: [] for actor in selected_actors
    }
    for source in source_configs:
        actor_id = old_to_actor.get(int(source.get("agent_id", -1)))
        if actor_id in selected_actor_id_set:
            source_by_actor[actor_id].append(source)

    actor_to_new_id = {
        actor["id"]: index for index, actor in enumerate(selected_actors)
    }
    financial_configs = []
    profiles = []
    as_of_date = str(spec["as_of"]).split("T", 1)[0]
    market_topics = [item["name"] for item in spec["market"]["instruments"]]

    for actor in selected_actors:
        actor_id = actor["id"]
        agent_id = actor_to_new_id[actor_id]
        sources = source_by_actor[actor_id]
        activity = deepcopy(ACTIVITY_BY_KIND[actor["kind"]])
        activity.update(
            {
                "agent_id": agent_id,
                "actor_id": actor_id,
                "entity_uuid": (
                    sources[0].get("entity_uuid") if sources else f"spec:{actor_id}"
                ),
                "entity_name": actor_id,
                "entity_type": ENTITY_TYPE_BY_KIND[actor["kind"]],
                "spec_kind": actor["kind"],
                "spec_aggregation": actor["aggregation"],
                "sentiment_bias": 0.0,
                "source_entity_uuids": [
                    item.get("entity_uuid")
                    for item in sources
                    if item.get("entity_uuid")
                ],
                "merged_entity_names": sorted(
                    {
                        str(item.get("entity_name"))
                        for item in sources
                        if item.get("entity_name")
                    }
                ),
            }
        )
        financial_configs.append(activity)

        country = "全球" if actor["kind"] == "foreign_investor" else "中国"
        profiles.append(
            {
                "user_id": agent_id,
                "username": actor_id,
                "name": actor["label"],
                "bio": (
                    f"{actor['label']}；目标：{'、'.join(actor['goals'])}；"
                    f"约束：{'、'.join(actor['constraints']) or '无额外约束'}"
                )[:150],
                "persona": _profile_persona(actor, spec),
                "karma": int(activity["influence_weight"] * 1000),
                "created_at": as_of_date,
                "age": 30,
                "gender": "other",
                "mbti": "ISTJ",
                "country": country,
                "profession": actor["kind"],
                "interested_topics": market_topics
                + ["金融政策", "风险管理", "市场微观结构"],
                "fever_actor_id": actor_id,
                "profile_source": "SimulationSpec",
            }
        )

    updated = deepcopy(config)
    updated["agent_configs"] = financial_configs
    remapped_posts = []
    dropped_posts = []
    for post in (updated.get("event_config") or {}).get("initial_posts") or []:
        post = deepcopy(post)
        old_agent_id = post.get("poster_agent_id")
        actor_id = (
            old_to_actor.get(int(old_agent_id))
            if isinstance(old_agent_id, int)
            else None
        )
        if actor_id not in selected_actor_id_set:
            dropped_posts.append(post)
            continue
        post["source_poster_agent_id"] = old_agent_id
        post["poster_actor_id"] = actor_id
        post["poster_agent_id"] = actor_to_new_id[actor_id]
        remapped_posts.append(post)
    updated.setdefault("event_config", {})["initial_posts"] = remapped_posts

    metadata = {
        "version": OVERLAY_VERSION,
        "spec_sha256": canonical_sha256(spec),
        "source_agent_count": len(source_configs),
        "financial_actor_count": len(financial_configs),
        "selected_actor_ids": [actor["id"] for actor in selected_actors],
        "actor_selection_sha256": canonical_sha256(
            {"actor_ids": [actor["id"] for actor in selected_actors]}
        ),
        "source_to_actor_id": {
            str(key): value for key, value in sorted(old_to_actor.items())
        },
        "actor_to_agent_id": actor_to_new_id,
        "unresolved_source_entities": unresolved,
        "dropped_initial_post_count": len(dropped_posts),
        "profile_source": "deterministic SimulationSpec compiler",
    }
    updated["fever_financial_actor_overlay"] = metadata
    return updated, profiles, metadata


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def apply_financial_actor_overlay(
    config_path: Path,
    profiles_path: Path,
    spec: Dict[str, Any],
    *,
    actor_ids: tuple[str, ...] | None = None,
) -> Dict[str, Any]:
    """Back up generated files once, then apply the deterministic overlay."""

    config_path = Path(config_path)
    profiles_path = Path(profiles_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = canonical_sha256(spec)
    requested_actor_ids = (
        list(actor_ids)
        if actor_ids is not None
        else [actor["id"] for actor in spec["actors"]]
    )
    all_actor_ids = {actor["id"] for actor in spec["actors"]}
    if (
        len(requested_actor_ids) < 2
        or len(requested_actor_ids) != len(set(requested_actor_ids))
        or not set(requested_actor_ids) <= all_actor_ids
    ):
        raise ValueError(
            "financial actor selection must contain at least two unique spec actors"
        )
    selected_actor_ids = [
        actor["id"]
        for actor in spec["actors"]
        if actor["id"] in set(requested_actor_ids)
    ]
    selection_hash = canonical_sha256({"actor_ids": selected_actor_ids})
    existing = config.get("fever_financial_actor_overlay") or {}
    if (
        existing.get("version") == OVERLAY_VERSION
        and existing.get("spec_sha256") == expected_hash
        and existing.get("actor_selection_sha256") == selection_hash
    ):
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        if len(profiles) != len(selected_actor_ids):
            raise ValueError("financial profile count does not match actor selection")
        return existing

    config_backup = config_path.with_name(
        "simulation_config.pre_financial_overlay.json"
    )
    profiles_backup = profiles_path.with_name(
        "reddit_profiles.pre_financial_overlay.json"
    )
    if not config_backup.exists():
        config_backup.write_bytes(config_path.read_bytes())
    if profiles_path.exists() and not profiles_backup.exists():
        profiles_backup.write_bytes(profiles_path.read_bytes())

    updated, profiles, metadata = compile_financial_actor_overlay(
        config,
        spec,
        actor_ids=actor_ids,
    )
    _atomic_write_json(config_path, updated)
    _atomic_write_json(profiles_path, profiles)
    return metadata
