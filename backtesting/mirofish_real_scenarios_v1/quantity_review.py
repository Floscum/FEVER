"""Add review context to existing scenarios without generating new claims."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timedelta
import re

# A review aid, not an entailment checker: equal numbers can still refer to
# different objects, and spelled-out quantities are outside this first pass.
QUANTITY = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:%|％|个百分点|个?(?:自然日|交易日|工作日|季度|小时|分钟|个月)|天|日|月|年|轮|万亿元|亿元|亿美元|万元|万美元|美元|家|篇|条|次|份)")
HORIZON_UNITS = {"calendar_days": "个自然日", "trading_days": "个交易日", "rounds": "轮"}


def _quantity_key(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("％", "%").replace("个自然日", "天").replace("自然日", "天").replace("个交易日", "交易日").replace("个工作日", "工作日").replace("个月", "月")


def _strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    return []


def unsupported_quantities(spec: dict, branch: dict) -> list[str]:
    basis = [fact["statement"] for fact in spec["facts"]]
    horizon = spec["horizon"]
    basis.append(f"{horizon['value']}{HORIZON_UNITS[horizon['kind']]}")
    # The user's actual window end is already specified, including when the
    # model renders an ISO date as Chinese month/day text.
    try:
        end = datetime.fromisoformat(str(horizon.get("end_at", "")).replace("Z", "+00:00"))
    except ValueError:
        pass
    else:
        basis.append(f"{end.year}年{end.month}月{end.day}日")
    try:
        cutoff = datetime.fromisoformat(str(spec.get("as_of", "")).replace("Z", "+00:00"))
    except ValueError:
        pass
    else:
        # "The next seven days" may be written as next-day through end-date.
        for boundary in (cutoff, cutoff + timedelta(days=1)):
            basis.append(f"{boundary.year}年{boundary.month}月{boundary.day}日")
    known = {_quantity_key(match.group()) for text in basis for match in QUANTITY.finditer(text)}
    fields = ("summary", "trigger_conditions", "triggers", "invalidation_conditions", "consequences", "conditional_responses", "observations")
    findings = {}
    for field in fields:
        for text in _strings(branch.get(field)):
            for match in QUANTITY.finditer(text):
                quantity = _quantity_key(match.group())
                if quantity not in known:
                    findings.setdefault(quantity, match.group().strip())
    return list(findings.values())


def prepare_scenarios(spec: dict, result: dict) -> list[dict]:
    """Preserve old scenario prose; attach only data already in the result."""
    scenarios = deepcopy(result.get("scenarios") or [])
    nodes = {node["id"]: node for node in result.get("simulation_graph", {}).get("nodes", [])}
    owners = {edge["target"]: edge["source"] for edge in result.get("simulation_graph", {}).get("edges", []) if edge.get("relation") == "SIMULATED_FINANCIAL_DECISION"}
    unit = HORIZON_UNITS[spec["horizon"]["kind"]]
    for scenario in scenarios:
        if "starting_decisions" not in scenario:
            scenario["starting_decisions"] = [
                {"actor_id": owners[ref], "action_type": nodes[ref]["action_type"], "decision_status": nodes[ref].get("decision_status", ""), "rationale": nodes[ref].get("summary", ""), "decision_ref": ref}
                for ref in scenario.get("simulation_refs", [])
                if ref in owners and ref in nodes and nodes[ref].get("action_type")
            ]
        if "observations" not in scenario:
            refs = scenario.get("evidence_refs", [])
            source = f"依据资料 {'、'.join(refs)} 的原始来源及后续更新" if refs else "待确认：先补充可核对的原始资料"
            scenario["observations"] = [
                {"kind": kind, "signal": signal, "source": source, "window": f"未来 {spec['horizon']['value']} {unit}", "evidence_refs": list(refs)}
                for kind, signals in (("trigger", scenario.get("triggers", scenario.get("trigger_conditions", []))), ("invalidation", scenario.get("invalidation_conditions", [])))
                for signal in dict.fromkeys(value for value in (signals if isinstance(signals, list) else []) if isinstance(value, str) and value.strip())
            ]
        findings = unsupported_quantities(spec, scenario)
        scenario["review_notices"] = ([f"这些数值未在输入事实或观察窗口中找到相同依据，使用前请核对：{'、'.join(findings[:10])}。"] if findings else [])
    return scenarios
