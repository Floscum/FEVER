"""Structured post-simulation interviews for finance-specific decisions."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

from .contracts import (
    canonical_sha256,
    validate_financial_actions,
    validate_result,
)


ACTION_TYPES_BY_KIND = {
    "government": ["COMMUNICATE", "REGULATE", "WAIT"],
    "regulator": ["COMMUNICATE", "REGULATE", "WAIT"],
    "exchange": ["COMMUNICATE", "REGULATE", "OPERATE", "WAIT"],
    "broker": ["COMMUNICATE", "OPERATE", "ALLOCATE", "WAIT"],
    "institutional_investor": ["COMMUNICATE", "ALLOCATE", "WAIT"],
    "labor_union": ["COMMUNICATE", "NEGOTIATE", "OPERATE", "WAIT"],
    "foreign_investor": ["COMMUNICATE", "ALLOCATE", "WAIT"],
    "retail_cohort": ["COMMUNICATE", "ALLOCATE", "WAIT"],
    "issuer": ["COMMUNICATE", "OPERATE", "WAIT"],
    "media": ["COMMUNICATE", "WAIT"],
    "analyst": ["COMMUNICATE", "WAIT"],
    "supplier": ["COMMUNICATE", "OPERATE", "NEGOTIATE", "WAIT"],
    "customer": ["COMMUNICATE", "NEGOTIATE", "WAIT"],
    "competitor": ["COMMUNICATE", "OPERATE", "WAIT"],
}


def build_financial_interview_plan(
    spec: Dict[str, Any],
    actor_to_agent_id: Dict[str, int],
    *,
    platform: str = "reddit",
    decision_round: int = 0,
) -> list[Dict[str, Any]]:
    """Build one bounded, JSON-only decision interview per specification actor."""

    if decision_round < 0:
        raise ValueError("decision_round must not be negative")
    facts = "\n".join(
        f"- {fact['id']}: {fact['statement']}" for fact in spec["facts"]
    )
    instruments = ", ".join(
        f"{item['symbol']}({item['name']})"
        for item in spec["market"]["instruments"]
    )
    plan = []
    for actor in spec["actors"]:
        actor_id = actor["id"]
        if actor_id not in actor_to_agent_id:
            continue
        constraints = [
            {
                "ref": f"{actor_id}:constraint:{index}",
                "text": value,
            }
            for index, value in enumerate(actor["constraints"])
        ]
        allowed_types = ACTION_TYPES_BY_KIND[actor["kind"]]
        prompt = f"""你是 {actor['label']}，actor_id 必须是 {actor_id}。
这是一次金融行为记录，不是价格预测，也不是社交媒体发帖任务。请根据你在模拟中的记忆、
截至时点事实、目标和约束，选择未来一个决策周期内最可能采取的一项主要动作。

输入事实截止时点：{spec['as_of']}
当前是该截止时点之后的抽象模拟第 {decision_round} 轮结束；模拟轮次没有绑定真实日期。
不得把截止时点后发生的真实事件写成已观察事实。事实中已公告但尚未生效的安排，仍须使用
“将生效”或“计划实施”等前瞻表述；模拟记忆只能标为模拟状态或假设。

可选 action_type（只能选一个）：{json.dumps(allowed_types, ensure_ascii=False)}
目标：{json.dumps(actor['goals'], ensure_ascii=False)}
约束及其引用：{json.dumps(constraints, ensure_ascii=False)}
可用事实：
{facts}
可用 instrument_refs：{instruments}

只返回一个 JSON 对象，不要 Markdown、解释前缀或额外字段：
{{
  "actor_id": "{actor_id}",
  "action_type": "{allowed_types[-1]}",
  "decision_status": "intended|conditional|no_action",
  "direction": "increase|decrease|maintain|conditional|not_applicable",
  "intensity": 0.0,
  "instrument_refs": [],
  "rationale": "说明动机、传导机制和主要不确定性",
  "constraint_refs": [],
  "evidence_refs": [],
  "visibility": "public|private",
  "confidence": 0.0
}}

intensity 表示动作强度，必须是 0.0 到 1.0 的小数；若不适用必须为 null。
confidence 必须是 0.0 到 1.0 的小数，仅表示你对该行为选择的清晰度，不是事件或价格预测
概率。evidence_refs 只能从上方列出的 F 编号逐字选择，不得创造新的 F 编号；
constraint_refs 只能使用上面列出的 ref；instrument_refs 只能使用证券代码。"""
        plan.append(
            {
                "agent_id": int(actor_to_agent_id[actor_id]),
                "actor_id": actor_id,
                "platform": platform,
                "prompt": prompt,
                "allowed_action_types": allowed_types,
            }
        )
    return plan


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("interview response is empty")
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
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
    raise ValueError("interview response does not contain a JSON object")


def _find_result_map(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    results = value.get("results")
    if isinstance(results, dict):
        return results
    for key in ("data", "result"):
        nested = _find_result_map(value.get(key))
        if nested:
            return nested
    return {}


def _bounded_float(
    value: Any,
    field: str,
    *,
    nullable: bool = False,
    repair_ten_point_scale: bool = False,
    warnings: list[str] | None = None,
):
    if value is None and nullable:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if repair_ten_point_scale and 1 < numeric <= 10:
        numeric /= 10
        if warnings is not None:
            warnings.append(
                f"{field}: normalized an apparent 0–10 score to 0–1"
            )
    if not 0 <= numeric <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return numeric


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return value


def _normalize_decision(
    payload: Dict[str, Any],
    *,
    plan_item: Dict[str, Any],
    spec: Dict[str, Any],
    round_num: int,
    source_ref: str,
    normalization_warnings: list[str],
) -> Dict[str, Any]:
    actor_id = plan_item["actor_id"]
    if payload.get("actor_id") != actor_id:
        raise ValueError("interview actor_id does not match requested actor")
    action_type = payload.get("action_type")
    if action_type not in plan_item["allowed_action_types"]:
        raise ValueError("action_type is not allowed for this actor kind")
    if payload.get("decision_status") not in {
        "intended",
        "conditional",
        "no_action",
    }:
        raise ValueError("decision_status is invalid")
    if payload.get("direction") not in {
        "increase",
        "decrease",
        "maintain",
        "conditional",
        "not_applicable",
    }:
        raise ValueError("direction is invalid")
    if payload.get("visibility") not in {"public", "private"}:
        raise ValueError("visibility is invalid")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")

    instrument_refs = _string_list(
        payload.get("instrument_refs"),
        "instrument_refs",
    )
    evidence_refs = _string_list(payload.get("evidence_refs"), "evidence_refs")
    constraint_refs = _string_list(
        payload.get("constraint_refs"),
        "constraint_refs",
    )
    valid_facts = {item["id"] for item in spec["facts"]}
    unknown_evidence_refs = sorted(set(evidence_refs) - valid_facts)
    if unknown_evidence_refs:
        evidence_refs = [item for item in evidence_refs if item in valid_facts]
        if not evidence_refs:
            raise ValueError("evidence_refs contains only unknown facts")
        normalization_warnings.append(
            f"{actor_id}: dropped unknown evidence_refs "
            + ", ".join(unknown_evidence_refs)
        )
    decision_warnings: list[str] = []
    decision = {
        "id": f"decision-{actor_id}-r{round_num}",
        "round": round_num,
        "actor_id": actor_id,
        "action_type": action_type,
        "decision_status": payload["decision_status"],
        "direction": payload["direction"],
        "intensity": _bounded_float(
            payload.get("intensity"),
            "intensity",
            nullable=True,
            repair_ten_point_scale=True,
            warnings=decision_warnings,
        ),
        "instrument_refs": instrument_refs,
        "rationale": rationale.strip(),
        "constraint_refs": constraint_refs,
        "evidence_refs": evidence_refs,
        "source_kind": "post_simulation_interview",
        "source_ref": source_ref,
        "visibility": payload["visibility"],
        "confidence": _bounded_float(
            payload.get("confidence"),
            "confidence",
            repair_ten_point_scale=True,
            warnings=decision_warnings,
        ),
        "probability_semantics": "decision_clarity_not_forecast_probability",
    }

    # Validate references early so one malformed actor response does not poison
    # the complete artifact.
    valid_instruments = {
        item["symbol"] for item in spec["market"]["instruments"]
    }
    actor = next(item for item in spec["actors"] if item["id"] == actor_id)
    valid_constraints = {
        f"{actor_id}:constraint:{index}"
        for index, _ in enumerate(actor["constraints"])
    }
    unknown_instrument_refs = sorted(set(instrument_refs) - valid_instruments)
    if unknown_instrument_refs:
        instrument_refs = [
            item for item in instrument_refs if item in valid_instruments
        ]
        decision["instrument_refs"] = instrument_refs
        normalization_warnings.append(
            f"{actor_id}: dropped unknown instrument_refs "
            + ", ".join(unknown_instrument_refs)
        )
    if not set(constraint_refs) <= valid_constraints:
        raise ValueError("constraint_refs contains an unknown constraint")
    normalization_warnings.extend(
        f"{actor_id}: {warning}" for warning in decision_warnings
    )
    return decision


def parse_financial_interviews(
    raw_response: Dict[str, Any],
    interview_plan: Iterable[Dict[str, Any]],
    spec: Dict[str, Any],
    *,
    simulation_id: str,
    round_num: int,
) -> Dict[str, Any]:
    """Parse batch Interview output into an auditable partial-safe artifact."""

    interview_plan = list(interview_plan)
    result_map = _find_result_map(raw_response)
    decisions = []
    failures = []
    normalization_warnings = []
    for item in interview_plan:
        agent_id = int(item["agent_id"])
        result = result_map.get(str(agent_id), result_map.get(agent_id))
        if not isinstance(result, dict):
            failures.append(
                {
                    "agent_id": agent_id,
                    "actor_id": item["actor_id"],
                    "error": "missing interview result",
                }
            )
            continue
        try:
            payload = _extract_json_object(result.get("response"))
            decisions.append(
                _normalize_decision(
                    payload,
                    plan_item=item,
                    spec=spec,
                    round_num=round_num,
                    source_ref=f"interview:agent:{agent_id}",
                    normalization_warnings=normalization_warnings,
                )
            )
        except (TypeError, ValueError) as error:
            failures.append(
                {
                    "agent_id": agent_id,
                    "actor_id": item["actor_id"],
                    "error": str(error)[:300],
                }
            )

    if decisions and not failures:
        status = "completed"
    elif decisions:
        status = "partial"
    else:
        status = "failed"
    artifact = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "simulation_id": simulation_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decisions": decisions,
        "failures": failures,
        "warnings": [
            "Decision confidence expresses action clarity, not forecast probability.",
            "Interview decisions are simulated statements, not observed evidence.",
        ]
        + normalization_warnings,
    }
    validate_financial_actions(artifact, spec)
    return artifact


def append_financial_decisions_to_result(
    result: Dict[str, Any],
    financial_actions: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach validated decisions to a SimulationResult behavior graph."""

    validate_result(result, spec)
    validate_financial_actions(financial_actions, spec)
    updated = deepcopy(result)
    if not updated["runs"]:
        raise ValueError("SimulationResult must contain a run")
    run = updated["runs"][0]
    nodes = updated["simulation_graph"]["nodes"]
    edges = updated["simulation_graph"]["edges"]
    existing_node_ids = {node.get("id") for node in nodes}
    for decision in financial_actions["decisions"]:
        run["events"].append(
            {
                "round": decision["round"],
                "actor_id": decision["actor_id"],
                "action_type": decision["action_type"],
                "summary": decision["rationale"],
                "visibility": decision["visibility"],
                "evidence_refs": decision["evidence_refs"],
            }
        )
        if decision["actor_id"] not in existing_node_ids:
            actor = next(
                item for item in spec["actors"] if item["id"] == decision["actor_id"]
            )
            nodes.append(
                {
                    "id": decision["actor_id"],
                    "kind": "actor",
                    "label": actor["label"],
                }
            )
            existing_node_ids.add(decision["actor_id"])
        nodes.append(
            {
                "id": decision["id"],
                "kind": "financial_decision",
                "action_type": decision["action_type"],
                "decision_status": decision["decision_status"],
                "direction": decision["direction"],
                "intensity": decision["intensity"],
                "confidence": decision["confidence"],
                "probability_semantics": decision["probability_semantics"],
                "summary": decision["rationale"],
            }
        )
        edges.append(
            {
                "source": decision["actor_id"],
                "target": decision["id"],
                "relation": "SIMULATED_FINANCIAL_DECISION",
            }
        )
    updated["warnings"].append(
        "Structured financial decisions were elicited post-simulation and remain simulated claims."
    )
    validate_result(updated, spec)
    return updated
