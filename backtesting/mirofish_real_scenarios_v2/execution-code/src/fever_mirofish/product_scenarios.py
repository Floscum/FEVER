"""Explicit starting decisions, conditional responses and observation cards."""
from __future__ import annotations

from typing import Any


def product_prompt(branch_count: int) -> str:
    return f"""你是事件研究助手。根据输入生成恰好 {branch_count} 条有实质区别的条件情景，帮助用户决定接下来查什么。

事实边界：
- facts 是截至 as_of 的资料；其他内容是模拟、角色假设或已有模拟决策，均不是新事实。
- 不得补充截止时间之后的真实结果、价格、公告。不得输出价格目标、收益预测或发生概率。
- 角色目标是建模假设，不代表已证实的动机；不得把私下意图当作公开观察。

行动的时间顺序：
- financial_decisions 是模拟结束时的起点。WAIT 是合法的当前等待，不代表未来永远不行动。
- 按 branch_slots 顺序生成；每条 conditional_responses 为该槽位 actor_ids 中每个角色写且只写一项，并准确复制 actor_id。
- 每项包含 condition（什么新条件出现）和 response（该条件下可能如何响应）。这些响应全是待验证假设。
- 明确一方的行为怎样改变另一方的选择。受约束而继续等待也是一种响应，但说明在等什么。
- response 必须用“我”或“我方”开头，以 actor_id 指定角色的第一人称描述自己的响应。不要把其他人的行动放在我的 response 中。
- 不得把未来响应描述为已经发生；不得改写起点决策。不要生成 decision_ref 或 actions。
- 没有依据支持响应时写明待补信息，不凑造交易条款、资源、权限或精确阈值。

观察清单：
- 每条 observations 写 2–4 项，至少一个 trigger 和一个 invalidation。
- signal 是具体可核对的观察：谁发布什么、哪项条款落实、哪项业务指标如何变化。
- source 是可以核对的渠道或文件类型，例如监管决定、公司公告、运营状态页；不要编造网址。
- 观察窗口由系统沿用用户选择的 horizon，不生成 window 字段，不在其他文字中另造期限。来源未明确时写“待确认：需要哪类资料”。
- trigger 支持进入这条分支；invalidation 应推翻核心机制，不能只写“触发没出现”。
- 每项 evidence_refs 只引用支持该观察设计的 F 编号；引用不代表未来信号已经发生。
- 摘要、条件响应、观察和后果必须相互一致；区分可观察信号与尚需验证的机制。
- confidence 仅为内部连贯性，不是概率。novelty_claim 简述多方推演多发现了哪段传导。

只输出 JSON，每条分支简明、便于阅读：
{{"branches":[{{"label":"简短名称","summary":"若…则…",
"conditional_responses":[{{"actor_id":"复制本槽位中的ID","condition":"若…","response":"我可能…"}}],
"observations":[{{"kind":"trigger","signal":"…","source":"…","evidence_refs":["F1"]}},
{{"kind":"invalidation","signal":"…","source":"…","evidence_refs":["F1"]}}],
"consequences":["可能后果"],"evidence_refs":["F1"],"novelty_claim":"…","confidence":0.5}}],"warnings":[]}}。
"""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def normalize_product_fields(raw: dict, decisions: list[dict], horizon: dict | None = None) -> dict:
    responses = raw.get("conditional_responses")
    if not isinstance(responses, list) or len(responses) != len(decisions):
        raise ValueError("conditional_responses must have one entry per slot actor")
    if not all(isinstance(response, dict) for response in responses):
        raise ValueError("conditional response must be an object")
    ids = [response.get("actor_id") for response in responses]
    expected = [decision["actor_id"] for decision in decisions]
    if any(not isinstance(actor, str) for actor in ids) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"conditional response actor_ids must match this slot: {expected}")
    by_actor = {response["actor_id"]: response for response in responses}
    normalized = []
    for decision in decisions:
        response = by_actor[decision["actor_id"]]
        response_text = _text(response.get("response"), "conditional response")
        if not response_text.startswith("我"):
            raise ValueError("response must start with 我 and describe that actor's own possible action")
        normalized.append({
            "actor_id": decision["actor_id"],
            "condition": _text(response.get("condition"), "response condition"),
            "response": response_text,
        })
    observations = raw.get("observations")
    if not isinstance(observations, list) or not 2 <= len(observations) <= 4:
        raise ValueError("observations must have 2 to 4 entries")
    cards = []
    for observation in observations:
        if not isinstance(observation, dict) or observation.get("kind") not in {"trigger", "invalidation"}:
            raise ValueError("observation kind must be trigger or invalidation")
        refs = observation.get("evidence_refs")
        if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) and ref for ref in refs):
            raise ValueError("observation evidence_refs must be non-empty references")
        cards.append({"kind": observation["kind"], **{
            field: _text(observation.get(field), f"observation {field}")
            for field in ("signal", "source")
        }, "window": (f"未来 {horizon['value']} 个自然日" if horizon else _text(observation.get("window"), "observation window")), "evidence_refs": list(dict.fromkeys(refs))})
    if {card["kind"] for card in cards} != {"trigger", "invalidation"}:
        raise ValueError("observations require both trigger and invalidation")
    return {
        "starting_decisions": [{key: decision[key] for key in ("actor_id", "action_type", "decision_status", "rationale")} | {"decision_ref": decision["id"]} for decision in decisions],
        "conditional_responses": normalized,
        "response_semantics": "conditional_hypothesis_not_observed_action",
        "observations": cards,
        "trigger_conditions": [card["signal"] for card in cards if card["kind"] == "trigger"],
        "invalidation_conditions": [card["signal"] for card in cards if card["kind"] == "invalidation"],
    }


def validate_product_fields(branch: dict) -> None:
    actors = branch["actor_ids"]
    starts = branch.get("starting_decisions", [])
    responses = branch.get("conditional_responses", [])
    if [item.get("actor_id") for item in starts] != actors or [item.get("actor_id") for item in responses] != actors:
        raise ValueError("starting and conditional responses must match slot actor order")
    if branch.get("response_semantics") != "conditional_hypothesis_not_observed_action":
        raise ValueError("future responses must be labeled conditional hypotheses")
    decisions = [{**item, "id": item.get("decision_ref")} for item in starts]
    rebuilt = normalize_product_fields(branch, decisions)
    for key in ("trigger_conditions", "invalidation_conditions"):
        if branch[key] != rebuilt[key]:
            raise ValueError("observation cards and legacy conditions disagree")
    for observation in rebuilt["observations"]:
        if not set(observation["evidence_refs"]) <= set(branch["evidence_refs"]):
            raise ValueError("observation references must be included in branch evidence_refs")
    for start, action in zip(starts, branch["actions"]):
        if any(start.get(key) != action.get(key) for key in ("actor_id", "action_type", "decision_ref")):
            raise ValueError("starting decisions must preserve referenced simulation actions")
