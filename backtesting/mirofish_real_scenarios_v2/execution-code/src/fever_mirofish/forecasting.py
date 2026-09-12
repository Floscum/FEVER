"""Frozen forecast aggregation shared by the B1, B2, and B3 arms."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

from .benchmark import validate_forecast_submission
from .contracts import canonical_sha256, validate_result, validate_spec


PROMPT_VERSION = "forecast-aggregator-v2-target-routed"


def simulation_context(
    result: Dict[str, Any],
    spec: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], set[str]]:
    """Return bounded simulation claims and their allowed reference ids."""

    validate_result(result, spec)
    allowed_kinds = {"simulated_action", "financial_decision", "scenario"}
    nodes = [
        {
            key: node.get(key)
            for key in (
                "id",
                "kind",
                "action_type",
                "decision_status",
                "direction",
                "intensity",
                "probability_semantics",
                "summary",
                "triggers",
                "consequences",
                "invalidation_conditions",
                "novelty_claim",
            )
            if node.get(key) is not None
        }
        for node in result["simulation_graph"]["nodes"]
        if node.get("kind") in allowed_kinds
    ]
    nodes = nodes[:120]
    return nodes, {str(node["id"]) for node in nodes}


def build_forecast_prompt(
    spec: Dict[str, Any],
    *,
    arm: str,
    simulation_result: Dict[str, Any] | None = None,
) -> tuple[str, str, set[str]]:
    """Build identical prediction instructions with arm-specific context."""

    validate_spec(spec)
    if arm not in {"B1", "B2", "B3"}:
        raise ValueError("forecast arm must be B1, B2, or B3")
    if arm == "B1" and simulation_result is not None:
        raise ValueError("B1 must not receive simulation output")
    if arm in {"B2", "B3"} and simulation_result is None:
        raise ValueError(f"{arm} requires a SimulationResult")

    simulation_nodes: list[Dict[str, Any]] = []
    valid_simulation_refs: set[str] = set()
    if simulation_result is not None:
        simulation_nodes, valid_simulation_refs = simulation_context(
            simulation_result,
            spec,
        )

    compact_input = {
        "case_id": spec["case_id"],
        "as_of": spec["as_of"],
        "question": spec["question"],
        "horizon": spec["horizon"],
        "market": spec["market"],
        "facts": [
            {
                "id": item["id"],
                "observed_at": item["observed_at"],
                "statement": item["statement"],
                "source_kind": item["source_kind"],
            }
            for item in spec["facts"]
        ],
        "actors": [
            {
                "id": item["id"],
                "label": item["label"],
                "kind": item["kind"],
                "goals": item["goals"],
                "constraints": item["constraints"],
            }
            for item in spec["actors"]
        ],
        "forecast_targets": spec["forecast_targets"],
    }
    system = """你是冻结历史回放的预测聚合器。你的输入永远以 as_of 截止，禁止补充你记忆中
as_of 之后的真实公告、价格、人员结果或其他后验。模拟节点只是待验证假设，不是证据，
其中的 confidence 也不是预测概率；情景节点的连贯性分数已从输入中移除。

逐个覆盖所有 forecast_targets：
- scoring=brier/log_loss 时，prediction_kind=binary_probability，value 是 0 到 1；
- scoring=direction_accuracy 时，prediction_kind=categorical，value 只能是 up/down/flat；
- 只引用输入中存在的 F 编号；
- B1 的 simulation_refs 必须为空；
- B2/B3 只能引用给出的模拟节点 id；没有相关节点时可以为空；
- 概率不能由点赞数、发帖数或 Agent 投票直接换算。
- 禁止把多个情景的 confidence、frequency 或“分支分数”相加、归一化、比较成发生概率；
- B3 的 market 目标由程序复用同一次已封存 B1 预测，模拟只允许改变 event 目标；
- event 目标可用情景的触发、失效条件和主体动作修正概率，但仍须以冻结事实为锚。

只输出 JSON：{"target_predictions":[{"target_id":"T1","prediction_kind":"...",
"value":0.5,"rationale":"说明传导、反方条件与不确定性","evidence_refs":["F1"],
"simulation_refs":[]}],"warnings":[]}。不得输出 Markdown 或额外字段。"""
    user = (
        f"实验臂：{arm}\n"
        f"冻结输入：{json.dumps(compact_input, ensure_ascii=False)}\n"
        f"模拟节点：{json.dumps(simulation_nodes, ensure_ascii=False)}"
    )
    return system, user, valid_simulation_refs


def _extract_payload(raw_response: Any) -> Dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("forecast response is empty")
    candidate = raw_response.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        candidate = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("forecast response does not contain a JSON object")


def build_forecast_submission(
    raw_response: Any,
    spec: Dict[str, Any],
    *,
    arm: str,
    model_id: str,
    valid_simulation_refs: Iterable[str] = (),
    replication_ids: Iterable[str | int] = (),
    baseline_submission: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Normalize and validate one provider response as a sealed submission."""

    payload = _extract_payload(raw_response)
    predictions = payload.get("target_predictions")
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise ValueError("forecast warnings must be a string list")
    if baseline_submission is not None:
        validate_forecast_submission(baseline_submission, spec)
        if baseline_submission["arm"] != "B1":
            raise ValueError("target-routing baseline must be a B1 submission")
        if arm != "B3":
            raise ValueError("target-routing baseline is only valid for B3")
        baseline_by_target = {
            item["target_id"]: item
            for item in baseline_submission["target_predictions"]
        }
        market_target_ids = {
            item["id"]
            for item in spec["forecast_targets"]
            if item["kind"] == "market"
        }
        if not isinstance(predictions, list):
            raise ValueError("forecast target_predictions must be a list")
        predictions = [
            (
                deepcopy(baseline_by_target[item.get("target_id")])
                if isinstance(item, dict)
                and item.get("target_id") in market_target_ids
                else item
            )
            for item in predictions
        ]
        warnings = list(warnings) + [
            "Market targets were copied from the sealed B1 baseline; "
            "simulation was routed only to event targets."
        ]
    submission = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "arm": arm,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "replication_ids": list(replication_ids),
        "target_predictions": predictions,
        "warnings": warnings,
    }
    validate_forecast_submission(submission, spec)
    allowed = set(valid_simulation_refs)
    used = {
        ref
        for prediction in submission["target_predictions"]
        for ref in prediction["simulation_refs"]
    }
    if arm in {"B2", "B3"} and not used <= allowed:
        unknown = ", ".join(sorted(used - allowed))
        raise ValueError(f"forecast contains unknown simulation refs: {unknown}")
    return submission
