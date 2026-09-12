"""Deterministic mapping between MiroFish entities and SimulationSpec actors."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


DIRECT_ALIASES = {
    "A股上市公司": "actor_issuers",
    "上市公司": "actor_issuers",
    "财政部": "actor_mof",
    "税务总局": "actor_mof",
    "中国证监会": "actor_csrc",
    "交易所": "actor_exchanges",
    "沪深北证券交易所": "actor_exchanges",
    "证券公司群体": "actor_brokers",
    "境内机构投资者": "actor_domestic_institutions",
    "境外机构投资者": "actor_foreign_institutions",
    "个人投资者群体": "actor_retail",
    "财经媒体": "actor_financial_media",
}


def resolve_actor_id(
    entity_name: str,
    actors: Iterable[Dict[str, Any]],
) -> Optional[str]:
    """Resolve one generated graph/config entity to a specification actor."""

    actors = list(actors)
    actor_ids = {actor["id"] for actor in actors}
    if entity_name in actor_ids:
        return entity_name

    direct = DIRECT_ALIASES.get(entity_name)
    if direct in actor_ids:
        return direct

    lowered = entity_name.lower()
    kind_candidates = {
        "证监会": "regulator",
        "交易所": "exchange",
        "证券公司": "broker",
        "券商": "broker",
        "境内机构": "institutional_investor",
        "境外机构": "foreign_investor",
        "个人投资": "retail_cohort",
        "上市公司": "issuer",
        "财经媒体": "media",
        "分析师": "analyst",
        "财政部": "government",
        "税务": "government",
        "民航局": "regulator",
        "监管局": "regulator",
        "波音": "supplier",
        "供应商": "supplier",
        "旅客": "customer",
        "客户": "customer",
        "同业航空": "competitor",
    }
    for marker, kind in kind_candidates.items():
        if marker.lower() in lowered:
            candidates = [actor["id"] for actor in actors if actor.get("kind") == kind]
            if len(candidates) == 1:
                return candidates[0]

    for actor in actors:
        label = str(actor.get("label") or "")
        if entity_name and (
            entity_name.lower() in label.lower()
            or label.lower() in entity_name.lower()
        ):
            return str(actor["id"])
    return None


def map_agent_configs_to_actors(
    agent_configs: Iterable[Dict[str, Any]],
    actors: Iterable[Dict[str, Any]],
) -> tuple[Dict[int, str], list[Dict[str, Any]]]:
    """Return old agent-id mappings plus auditable unresolved records."""

    actors = list(actors)
    mapping: Dict[int, str] = {}
    unresolved = []
    for config in agent_configs:
        if not isinstance(config, dict) or "agent_id" not in config:
            continue
        agent_id = int(config["agent_id"])
        entity_name = str(config.get("entity_name") or "")
        actor_id = resolve_actor_id(entity_name, actors)
        if actor_id:
            mapping[agent_id] = actor_id
        else:
            unresolved.append(
                {
                    "agent_id": agent_id,
                    "entity_name": entity_name,
                    "entity_type": config.get("entity_type"),
                }
            )
    return mapping, unresolved
