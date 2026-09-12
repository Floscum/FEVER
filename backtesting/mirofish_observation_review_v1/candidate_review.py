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


def review_observation(spec: dict, observation: dict) -> dict:
    """Triage concrete, previously observed failure modes, not truth or utility.

    A source label and a matching number cannot establish entailment. Keep the
    original condition visible and attach reasons; never silently rewrite it.
    """
    signal, source = str(observation.get("signal", "")), str(observation.get("source", ""))
    refs = observation.get("evidence_refs") or []
    cited_spec = {**spec, "facts": [fact for fact in spec["facts"] if fact.get("id") in refs]}
    quantities = unsupported_quantities(cited_spec, {"triggers": [signal]})
    findings = []
    unsupported = unsupported_quantities(spec, {"triggers": [signal]})
    citation_gaps = [value for value in quantities if value not in unsupported]
    if unsupported:
        findings.append({"code": "unsupported_quantity", "message": f"输入资料及观察窗口未提供这些数值：{'、'.join(unsupported)}；先核对数值含义及来源"})
    if citation_gaps:
        findings.append({"code": "quantity_citation_gap", "message": f"这些数值出现在其他输入资料中，但当前引用未包含：{'、'.join(citation_gaps)}；补全引用并核对数值所指对象"})
    known_refs = {fact.get("id") for fact in spec["facts"]}
    if not refs or any(ref not in known_refs for ref in refs):
        findings.append({"code": "evidence_reference_gap", "message": "观察条件缺少有效的资料引用，需补上本次输入中的依据编号"})
    if re.search(r"内部|私下|闭门|未公开|非公开|匿名|传闻|备忘录|投委会", source) or (
        re.search(r"内部|私下|闭门|非公开|投委会|日常沟通.{0,24}传递", signal)
        and not re.search(r"(?:公开披露|公开发布|公告披露|公告确认).{0,12}(?:内部|私下|闭门|非公开|投委会)", signal)
    ):
        findings.append({"code": "disclosure_required", "message": "该渠道或信号涉及非公开信息，需找到公开披露后才能复核"})
    horizon = spec["horizon"]
    if re.search(r"13\s*F", signal + source, re.I) and (horizon["kind"] == "calendar_days" and horizon["value"] <= 30):
        findings.append({"code": "reporting_period_mismatch", "message": "13F 反映季度末持仓，不能据此确认本次短期窗口内的交易；需核对报告时点"})
    if not source.strip():
        findings.append({"code": "missing_source", "message": "尚未明确可查询的观察渠道"})
    if observation.get("kind") == "invalidation" and silence_as_invalidation(signal):
        findings.append({"code": "silence_is_not_invalidation", "message": "这里含有尚未观察到或未披露消息的条件；请区分“仍待观察”和“路径已被推翻”。如有明确披露义务及截止日，应按该承诺复核"})
    findings.extend(trading_state_findings(spec, signal))
    return {"review_status": "needs_review" if findings else "no_rule_findings", "review_findings": findings}


def silence_as_invalidation(text: str) -> bool:
    # A reported negative decision is different from an absence of reports.
    # Keep scope within each clause, so a public withdrawal in one clause does
    # not excuse an alternative "no news" condition in another.
    for clause in re.split(r"[，。；;]|；或|，或", text):
        if re.search(r"(?:公告|公开声明|公开披露|正式报告|公开数据)(?:中|已|明确|的数据|数据)?(?:确认|证实|显示|明确|指出|表示)", clause):
            continue
        if re.search(
            r"未(?:见|发现|观察到|查到)|没有(?:看到|观察到|查到)|保持沉默|"
            r"(?:未|没有|不)(?:在[^，。；]{0,16})?(?:公开|新增)?(?:发布|披露|公布|发表|提供|发出|提及|表示|强调)|"
            r"未新增[^，。；]{0,16}披露|"
            r"无(?:任何|新的|新增|公开)[^，。；]{0,32}(?:消息|信息|意见|发布|披露|拒绝|报道)",
            clause,
        ):
            return True
    return False


def trading_state_findings(spec: dict, text: str) -> list[dict]:
    facts = "\n".join(fact["statement"] for fact in spec["facts"])
    # Merely mentioning that a company has NOT halted is not halt evidence.
    positive = []
    for clause in re.split(r"[，。；;\n]", facts):
        if re.search(r"停牌|复牌|trading halt|trading suspension", clause, re.I) and not re.search(
            r"未(?:曾|被|发生|宣布|披露|涉及)?(?:停牌|复牌|停复牌)|没有[^，。；]{0,8}停牌|无需停牌|否认[^，。；]{0,8}停牌|不涉及[^，。；]{0,8}停牌|(?:no|not).{0,15}trading (?:halt|suspension)", clause, re.I
        ):
            positive.append(clause)
    for clause in re.split(r"[。；;\n]", text):
        if not re.search(r"复牌后|停牌期间|已停牌", clause):
            continue
        if re.search(r"假设[^，。；]{0,20}停牌|(?:若|如果)(?:未来|后续|临时)?停牌|(?:若|如果)[^，。；]{0,20}(?:宣布|发生|出现|启动|决定)停牌", clause):
            continue
        # Restrict support to an explicitly named company when both the facts
        # and the scenario name it. A counterparty's halt is not transferable.
        names = [actor.get("identity", {}).get("name") for actor in spec.get("actors", [])]
        mentioned = [name for name in names if name and name in clause]
        supported = bool(positive) and (not mentioned or all(any(name in fact for fact in positive) for name in mentioned))
        if not supported:
            return [{"code": "unsupported_trading_state", "message": "输入未提供支持这里所述主体停复牌状态的肯定披露；请核实主体和交易安排"}]
    return []


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
        prose = "\n".join(_strings({key: scenario.get(key) for key in ("summary", "triggers", "trigger_conditions", "consequences", "invalidation_conditions")}))
        scenario["review_notices"].extend(item["message"] for item in trading_state_findings(spec, prose))
        facts = "\n".join(fact["statement"] for fact in spec["facts"])
        if re.search(r"比较数据.*已.*(?:调整|重述)", facts) and re.search(r"口径调整后|剔除.{0,16}(?:口径|保证费用)|补充.{0,20}可比数据", prose):
            scenario["review_notices"].append("输入已说明比较数据按同一口径调整；请核实这里是否重复调整，或误将可比数据当成尚未提供。")
        for observation in scenario["observations"]:
            observation.update(review_observation(spec, observation))
    return scenarios
