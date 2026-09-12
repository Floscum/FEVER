"""Compile multi-actor decisions into auditable, falsifiable event branches."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict

from .contracts import (
    canonical_sha256,
    validate_financial_actions,
    validate_result,
    validate_scenario_branches,
    validate_spec,
)
from .trace_digest import compact_interactions
from .product_scenarios import normalize_product_fields, product_prompt
from .focused_scenarios import FOCUSED_PROMPT_VERSION, candidate_slots, focused_prompt, normalize_focused_fields


PROMPT_VERSION = "scenario-branch-compiler-v6"
PRODUCT_PROMPT_VERSION = "scenario-branch-compiler-v8"


def _require_generated_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"scenario {field} must be a non-empty string")
    return value.strip()


def _require_generated_text_list(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"scenario {field} must be a non-empty string list")
    return [item.strip() for item in value]


def build_scenario_branch_retry_prompt(
    system_prompt: str,
    user_prompt: str,
    validation_error: BaseException,
) -> tuple[str, str]:
    """Build one outcome-free repair prompt after structural validation fails."""

    message = " ".join(str(validation_error).split())[:240]
    repair = (
        "\n\n上一次输出未通过结构契约。不要参考或复述上一次回答；"
        "请仅根据原始匿名输入重新生成完整 JSON。验证错误："
        f"{message}。必须严格遵守原始提示中的分支数量，并满足每条分支的全部字段。"
    )
    return system_prompt + repair, user_prompt


def _required_branch_count(financial_actions: Dict[str, Any]) -> int:
    """Keep small cases compact while reserving four slots for 4+ actors."""

    actor_count = len(
        {
            str(item.get("actor_id"))
            for item in financial_actions.get("decisions", [])
            if item.get("actor_id")
        }
    )
    return min(4, max(2, actor_count))


def _build_branch_slots(
    spec: Dict[str, Any],
    simulation_result: Dict[str, Any],
    financial_actions: Dict[str, Any],
    *,
    count: int = 4,
    product: bool = False,
) -> list[Dict[str, Any]]:
    """Choose traceable actor pairs without asking the provider to copy IDs."""

    decisions_by_actor = {
        item["actor_id"]: item for item in financial_actions["decisions"]
    }
    active_counts: Dict[str, int] = {}
    for edge in simulation_result["simulation_graph"]["edges"]:
        if edge.get("relation") in {
            "SIMULATED_ACTION",
            "SIMULATED_FINANCIAL_DECISION",
        }:
            actor_id = edge.get("source")
            if actor_id in decisions_by_actor:
                active_counts[actor_id] = active_counts.get(actor_id, 0) + 1

    def actor_score(actor_id: str) -> float:
        decision = decisions_by_actor[actor_id]
        status_score = {
            "intended": 3.0,
            "conditional": 2.0,
            "no_action": 0.0,
        }.get(decision.get("decision_status"), 0.0)
        action_score = 0.0 if decision.get("action_type") == "WAIT" else 1.0
        intensity = decision.get("intensity")
        intensity_score = float(intensity) if isinstance(
            intensity, (int, float)
        ) and not isinstance(intensity, bool) else 0.0
        return (
            status_score
            + action_score
            + intensity_score
            + min(active_counts.get(actor_id, 0), 3) * 0.25
        )

    candidates = []
    seen_pairs = set()
    for source_index, relationship in enumerate(spec["relationships"]):
        source = relationship["source_actor_id"]
        target = relationship["target_actor_id"]
        if source not in decisions_by_actor or target not in decisions_by_actor:
            continue
        pair_key = tuple(sorted((source, target)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        candidates.append(
            {
                "actor_ids": [source, target],
                "relationship_kind": relationship["kind"],
                "score": actor_score(source) + actor_score(target),
                "source_index": source_index,
            }
        )

    ranked_actors = sorted(
        decisions_by_actor,
        key=lambda actor_id: (-actor_score(actor_id), actor_id),
    )
    for left_index, source in enumerate(ranked_actors):
        for target in ranked_actors[left_index + 1 :]:
            pair_key = tuple(sorted((source, target)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            candidates.append(
                {
                    "actor_ids": [source, target],
                    "relationship_kind": "candidate_interaction",
                    "score": actor_score(source) + actor_score(target),
                    "source_index": len(spec["relationships"])
                    + len(candidates),
                }
            )

    if not candidates:
        raise ValueError("scenario compilation requires at least two actors")
    candidates.sort(key=lambda item: (-item["score"], item["source_index"]))
    slots = []
    covered_actors: set[str] = set()
    available = list(candidates)
    for index in range(count):
        if not available:
            available = list(candidates)
        candidate = min(
            available,
            key=lambda item: (
                -len(set(item["actor_ids"]) - covered_actors),
                -item["score"],
                item["source_index"],
            ),
        )
        available.remove(candidate)
        actor_ids = list(candidate["actor_ids"])
        covered_actors.update(actor_ids)
        slots.append(
            {
                "slot_id": f"slot-{index + 1}",
                "actor_ids": actor_ids,
                "relationship_kind": candidate["relationship_kind"],
                "decision_refs": [
                    decisions_by_actor[actor_id]["id"]
                    for actor_id in actor_ids
                ],
            }
        )
    if product:
        # A fifth branch is unnecessary: extend related pairs into short chains.
        for actor_id in ranked_actors:
            if actor_id in covered_actors:
                continue
            related = {
                other
                for relationship in spec["relationships"]
                for other in (relationship["source_actor_id"], relationship["target_actor_id"])
                if actor_id in (relationship["source_actor_id"], relationship["target_actor_id"])
                and other != actor_id
            }
            eligible = [slot for slot in slots if len(slot["actor_ids"]) < 3]
            if not eligible:
                raise ValueError("too many actors for four short scenario chains")
            slot = min(eligible, key=lambda item: (
                -len(set(item["actor_ids"]) & related), len(item["actor_ids"]), item["slot_id"],
            ))
            slot["actor_ids"].append(actor_id)
            slot["decision_refs"].append(decisions_by_actor[actor_id]["id"])
            covered_actors.add(actor_id)
    return slots


def build_scenario_branch_prompt(
    spec: Dict[str, Any],
    simulation_result: Dict[str, Any],
    financial_actions: Dict[str, Any],
    *,
    product: bool = False,
    product_version: str = PRODUCT_PROMPT_VERSION,
) -> tuple[str, str, set[str]]:
    """Build a bounded synthesis prompt without outcomes or forecast targets."""

    validate_spec(spec)
    if product and product_version not in {"scenario-branch-compiler-v7", PRODUCT_PROMPT_VERSION, FOCUSED_PROMPT_VERSION}:
        raise ValueError("unsupported product compiler version")
    validate_result(simulation_result, spec)
    validate_financial_actions(financial_actions, spec)
    decision_ids = {
        item["id"] for item in financial_actions["decisions"]
    }
    actor_by_node = {
        edge["target"]: edge["source"]
        for edge in simulation_result["simulation_graph"]["edges"]
        if edge.get("relation")
        in {"SIMULATED_ACTION", "SIMULATED_FINANCIAL_DECISION"}
    }
    simulation_nodes = []
    for node in simulation_result["simulation_graph"]["nodes"]:
        if node.get("kind") != "simulated_action":
            continue
        compact = {
            key: node.get(key)
            for key in ("id", "kind", "action_type", "summary")
            if node.get(key) is not None
        }
        compact["actor_id"] = actor_by_node.get(node.get("id"))
        simulation_nodes.append(compact)
    simulation_nodes = simulation_nodes[:80]
    if product:
        simulation_nodes, _ = compact_interactions(simulation_result)
    allowed_refs = decision_ids | {
        str(item["id"]) for item in simulation_nodes
    }
    required_branch_count = _required_branch_count(financial_actions)
    branch_slots = candidate_slots(spec, financial_actions) if product and product_version == FOCUSED_PROMPT_VERSION else _build_branch_slots(
        spec,
        simulation_result,
        financial_actions,
        count=required_branch_count,
        product=product,
    )
    compact_input = {
        "case_id": spec["case_id"],
        "as_of": spec["as_of"],
        "question": spec["question"],
        "horizon": spec["horizon"],
        "facts": [
            {"id": item["id"], "statement": item["statement"]}
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
        "financial_decisions": financial_actions["decisions"],
        "simulated_actions": simulation_nodes,
        "branch_slots": branch_slots,
    }
    system = (
        "你是金融事件分支编译器，不是价格预测器。只使用输入中 as_of 时点可得事实\n"
        "和明确标为模拟的行为，生成恰好 "
        f"{required_branch_count} 条彼此有实质区别、未来可证伪的条件化事件分支。\n"
        """

硬性要求：
- 不得补充 as_of 之后的真实事件、公告、价格或人员结果；
- 每条分支至少串联两个不同 actor 的动作，不能只是改写一个人的观点；
- trigger_conditions 必须是未来可观察的“若……则进入该分支”条件；
- consequences 是机制结果，不是已发生事实；
- invalidation_conditions 必须说明什么观察会推翻该分支；
- evidence_refs 只能引用输入中的 F 编号；
- 按 branch_slots 的顺序生成分支；第 N 条分支必须描述第 N 个槽位中的两个角色及其关系；
- 不要输出 actor_ids、actions、decision_ref 或 simulation_refs；系统会按数组位置确定性绑定
  槽位中的角色及其已有 financial_decision，禁止自行概括、改写或新造动作；
- confidence 只表示分支内部连贯性，不是发生概率；
- novelty_claim 说明该分支比事实直接复述多出了哪段多主体传导；
- 不得输出目标价格、收益率概率或 Agent 投票频率。

只输出 JSON：
{"branches":[{"label":"简短名称","summary":"条件化摘要",
"trigger_conditions":["..."],"consequences":["..."],
"invalidation_conditions":["..."],"evidence_refs":["F1"],
"novelty_claim":"...","confidence":0.5}],"warnings":[]}。"""
    )
    user = json.dumps(compact_input, ensure_ascii=False)
    if product:
        system = system.replace("两个角色及其关系", "所有角色及其相互影响")
        system += (
            "\n互动输入已经按角色去重和筛选；它们仍是模拟消息，不是新增事实。"
            "不得凭空推断立场变化或因果影响。每条分支说明一方行动如何影响另一方的可行行动。"
            "触发和失效条件尽量指出观察对象与时间窗口；没有依据时不编造精确阈值。"
            "对缺少行动依据的传导使用条件表述，不把等待决策改写成已经采取新行动。"
        )
        if product_version == PRODUCT_PROMPT_VERSION:
            system = product_prompt(required_branch_count)
        elif product_version == FOCUSED_PROMPT_VERSION:
            system = focused_prompt()
    return system, user, allowed_refs


def _extract_payload(raw_response: Any) -> Dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("scenario branch response is empty")
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
    raise ValueError("scenario branch response does not contain a JSON object")


def build_scenario_branch_set(
    raw_response: Any,
    spec: Dict[str, Any],
    simulation_result: Dict[str, Any],
    financial_actions: Dict[str, Any],
    *,
    model_id: str,
    product: bool = False,
    product_version: str = PRODUCT_PROMPT_VERSION,
) -> Dict[str, Any]:
    """Normalize one provider response and enforce traceable branch structure."""

    build_scenario_branch_prompt(
        spec,
        simulation_result,
        financial_actions,
        product=product,
        product_version=product_version,
    )
    payload = _extract_payload(raw_response)
    raw_branches = payload.get("branches")
    required_branch_count = _required_branch_count(financial_actions)
    focused = product and product_version == FOCUSED_PROMPT_VERSION
    if (
        not isinstance(raw_branches, list)
        or (len(raw_branches) > 4 if focused else len(raw_branches) != required_branch_count)
    ):
        raise ValueError(
            "scenario response must contain exactly "
            f"{required_branch_count} branches"
        )
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise ValueError("scenario warnings must be a string list")

    decisions_by_actor: Dict[str, list[Dict[str, Any]]] = {}
    for decision in financial_actions["decisions"]:
        decisions_by_actor.setdefault(decision["actor_id"], []).append(decision)
    branch_slots = candidate_slots(spec, financial_actions) if focused else _build_branch_slots(
        spec,
        simulation_result,
        financial_actions,
        count=required_branch_count,
        product=product,
    )
    branches = []
    normalization_warnings = []
    for index, raw in enumerate(raw_branches, start=1):
        if not isinstance(raw, dict):
            raise ValueError("scenario branch must be an object")
        if focused:
            slot = next((slot for slot in branch_slots if slot["slot_id"] == raw.get("slot_id")), None)
            if slot is None:
                raise ValueError("v9 slot_id must identify an input candidate relationship")
            actor_ids = slot["actor_ids"]
        else:
            actor_ids = branch_slots[index - 1]["actor_ids"]
        selected_decisions = []
        for actor_id in actor_ids:
            candidates = decisions_by_actor.get(actor_id, [])
            if len(candidates) != 1:
                raise ValueError(
                    "scenario actor must have exactly one financial decision"
                )
            selected_decisions.append(candidates[0])
        normalized_actions = [
            {
                "actor_id": decision["actor_id"],
                "action_type": decision["action_type"],
                "decision_ref": decision["id"],
            }
            for decision in selected_decisions
        ]
        simulation_refs = [decision["id"] for decision in selected_decisions]
        normalization_warnings.append(
            f"branch-{index}: bound deterministic actor slot to unique "
            "financial decisions"
        )
        raw_label = raw.get("label")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raw_label = f"情景 {index}"
            normalization_warnings.append(
                f"branch-{index}: filled missing display label deterministically"
            )
        confidence = raw.get("confidence")
        if confidence is None:
            confidence = 0.5
            normalization_warnings.append(
                f"branch-{index}: filled missing coherence confidence with 0.5"
            )
        elif not isinstance(confidence, (int, float)) or isinstance(
            confidence, bool
        ):
            raise ValueError("scenario confidence must be numeric")
        product_fields = normalize_product_fields(raw, selected_decisions, spec["horizon"]) if product and product_version == PRODUCT_PROMPT_VERSION else {}
        if focused:
            product_fields = normalize_focused_fields(raw, selected_decisions, spec)
        conditions = {**raw, **product_fields}
        branch = {
            "id": f"branch-{index}",
            "label": raw_label.strip(),
            "summary": _require_generated_text(raw.get("summary"), "summary"),
            "actor_ids": actor_ids,
            "trigger_conditions": _require_generated_text_list(
                conditions.get("trigger_conditions"), "trigger_conditions"
            ),
            "actions": normalized_actions,
            "consequences": _require_generated_text_list(
                raw.get("consequences"), "consequences"
            ),
            "invalidation_conditions": _require_generated_text_list(
                conditions.get("invalidation_conditions"),
                "invalidation_conditions",
            ),
            "evidence_refs": _require_generated_text_list(
                raw.get("evidence_refs"), "evidence_refs"
            ),
            "simulation_refs": simulation_refs,
            "novelty_claim": _require_generated_text(
                raw.get("novelty_claim"), "novelty_claim"
            ),
            "confidence": float(confidence),
            "confidence_semantics": (
                "branch_coherence_not_forecast_probability"
            ),
        }
        branch.update(product_fields)
        branches.append(branch)

    run_id = simulation_result["runs"][0]["run_id"]
    artifact = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "simulation_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "prompt_version": product_version if product else PROMPT_VERSION,
        "status": "completed" if branches else "partial",
        "branches": branches,
        "warnings": warnings
        + normalization_warnings
        + [
            "Branches are simulated hypotheses, not observed evidence.",
            "Branch confidence expresses coherence, not occurrence probability.",
        ],
    }
    if focused:
        covered = {actor for branch in branches for actor in branch["actor_ids"]}
        omitted = [actor["label"] for actor in spec["actors"] if actor["id"] not in covered]
        if omitted:
            artifact["warnings"].append("未强行纳入情景的参与方：" + "、".join(omitted) + "。角色有决策不代表存在有用的传导路径。")
        if not branches:
            artifact["warnings"].append("候选关系不足以形成可跟踪情景；已保留各方决策，请补充影响关系或公开观察渠道。")
    validate_scenario_branches(
        artifact,
        spec,
        simulation_result,
    )
    return artifact


def append_scenario_branches_to_result(
    result: Dict[str, Any],
    branch_set: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach branch hypotheses to a copy of a SimulationResult."""

    validate_result(result, spec)
    validate_scenario_branches(branch_set, spec, result)
    updated = deepcopy(result)
    nodes = updated["simulation_graph"]["nodes"]
    edges = updated["simulation_graph"]["edges"]
    for branch in branch_set["branches"]:
        scenario = {
            "id": branch["id"],
            "label": branch["label"],
            "summary": branch["summary"],
            "run_count": 1,
            "frequency": 1.0,
            "probability_semantics": "uncalibrated_simulation_frequency",
            "triggers": branch["trigger_conditions"],
            "consequences": branch["consequences"],
            "invalidation_conditions": branch["invalidation_conditions"],
            "actor_ids": branch["actor_ids"],
            "evidence_refs": branch["evidence_refs"],
            "simulation_refs": branch["simulation_refs"],
            "novelty_claim": branch["novelty_claim"],
            "confidence": branch["confidence"],
            "confidence_semantics": branch["confidence_semantics"],
        }
        for field in ("starting_decisions", "conditional_responses", "response_semantics", "observations", "assumptions"):
            if field in branch:
                scenario[field] = deepcopy(branch[field])
        updated["scenarios"].append(scenario)
        nodes.append(
            {
                "id": branch["id"],
                "kind": "scenario",
                "summary": branch["summary"],
                "triggers": branch["trigger_conditions"],
                "consequences": branch["consequences"],
                "invalidation_conditions": branch[
                    "invalidation_conditions"
                ],
                "novelty_claim": branch["novelty_claim"],
                "confidence": branch["confidence"],
                "probability_semantics": branch["confidence_semantics"],
            }
        )
        for ref in branch["simulation_refs"]:
            edges.append(
                {
                    "source": ref,
                    "target": branch["id"],
                    "relation": "CONTRIBUTES_TO_SCENARIO",
                }
            )
    updated["warnings"] = [
        warning
        for warning in updated["warnings"]
        if warning
        != "No forecast target result or scenario frequency was produced from this single run."
    ]
    updated["warnings"].append(
        "Single-run scenario frequency is uncalibrated and must not be read as probability."
    )
    updated["warnings"].extend(branch_set.get("warnings", []))
    if not branch_set["branches"]:
        updated["status"] = "partial"
    validate_result(updated, spec)
    return updated
