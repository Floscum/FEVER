"""Preserve explicitly named, evidence-backed transaction parties.

This is an extractive convenience, not general entity recognition. A ticker in
the question supplies an identity hint; admitted facts must contain the name.
Multiple names become separate principals only when facts connect them in a
transaction. Comparisons and mere mentions must not create extra issuers.
"""
from __future__ import annotations

from copy import deepcopy
import re


TICKER_NAME = re.compile(r"([A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff .&-]{1,60})[（(]([03468]\d{5}|[A-Z][A-Z0-9.]{0,8})[）)]")
TRANSACTION = re.compile(r"吸收合并|换股|收购|并购|合并|acqui(?:re|sition)|merg(?:e|er)", re.I)


def grounded_names(question: str, facts: list[dict], market: dict | None = None) -> list[dict]:
    candidates = [(match[1].strip(), match[2]) for match in TICKER_NAME.finditer(question)]
    candidates += [(item.get("name", ""), item.get("symbol", "")) for item in (market or {}).get("instruments", [])]
    result, seen = [], set()
    for raw, symbol in candidates:
        raw = re.sub(r"^(?:与|及|以及)", "", raw)
        if not raw or not symbol or symbol in seen or raw.startswith(("A股 ", "事件研究")):
            continue
        # Strip only explicit question framing. Arbitrary suffix matching could
        # turn an absent company into a shared suffix such as “证券”.
        name = re.sub(r"^(?:(?:请|请你|帮我|请帮我)?(?:分析|比较|研究|看看)|对于|关于|compare\s+|analyze\s+)", "", raw, flags=re.I).strip()
        if len(name) < 2 or not any(name in fact["statement"] for fact in facts):
            continue
        refs = [fact["id"] for fact in facts if name in fact["statement"]]
        seen.add(symbol)
        result.append({"name": name, "symbol": symbol, "evidence_refs": refs})
    return result


def apply_entity_roles(actors: list[dict], facts: list[dict], question: str, market: dict | None = None) -> tuple[list[dict], dict]:
    names = grounded_names(question, facts, market)
    if not names:
        return actors, {"identified_entity_count": 0, "configured_entity_count": 0, "unconfigured_entity_names": []}
    principals = [names[0]]
    for candidate in names[1:]:
        if any(names[0]["name"] in fact["statement"] and candidate["name"] in fact["statement"] and TRANSACTION.search(fact["statement"]) for fact in facts):
            principals.append(candidate)
    # Keep room for at least one other stakeholder under the user's actor cap.
    selected = principals[:len(actors) - 1]
    template = next(actor for actor in actors if actor["kind"] == "issuer")
    named = []
    for index, entity in enumerate(selected):
        actor = deepcopy(template)
        actor["id"] = "actor_issuer" if index == 0 else "actor_issuer_" + re.sub(r"[^a-z0-9]", "_", entity["symbol"].lower())
        actor["label"] = f"{entity['name']}（{entity['symbol']}）"
        actor["identity"] = entity
        actor["goals"][0] = f"代表{entity['name']}独立评估经营与交易条件，维护自身利益"
        actor["constraints"].append(f"只代表{entity['name']}作出自身权限内的决策；不能代替交易对方、股东或监管机构批准交易")
        actor["selection_reason"] = f"问题或市场信息提供名称与代码；主体名称在 {'、'.join(entity['evidence_refs'])} 中可核对。身份匹配不代表已验证证券代码或交易权限"
        named.append(actor)
    remaining = [actor for actor in actors if actor["kind"] != "issuer"]
    result = named + remaining[:len(actors) - len(named)]
    return result, {"identified_entity_count": len(names), "configured_entity_count": len(named), "unconfigured_entity_names": [item["name"] for item in names if item not in selected]}
