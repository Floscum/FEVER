"""Outcome-safe evaluation helpers for financial scenario decision support."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

from .contracts import canonical_sha256, validate_spec


SCHEMA_VERSION = "0.1.0"
EVALUATION_VERSION = "decision-support-retrospective-v1"
SINGLE_AGENT_PROMPT_VERSION = "single-agent-scenario-baseline-v1"
QUALITY_JUDGE_VERSION = "scenario-quality-pairwise-v2"
OUTCOME_JUDGE_VERSION = "outcome-path-recall-v1"
ABSOLUTE_QUALITY_JUDGE_VERSION = "scenario-quality-absolute-v1"
ABSOLUTE_OUTCOME_JUDGE_VERSION = "outcome-path-recall-absolute-v1"
BRANCH_COUNT = 4
QUALITY_SCORE_KEYS = (
    "mechanism_coherence",
    "monitoring_actionability",
    "falsifiability",
    "evidence_discipline",
    "scenario_diversity",
)
BRANCH_QUALIFICATION_KEYS = (
    "multi_actor_causal_chain",
    "observable_trigger",
    "specific_invalidation",
    "decision_relevant",
    "evidence_grounded",
)
OUTCOME_STATUSES = ("full", "partial", "miss")


def protocol_descriptor() -> Dict[str, Any]:
    """Return the frozen, machine-readable scoring rules."""

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "classification": "retrospective_method_validation",
        "independent_unit": "historical_event_case",
        "scenario_budget_per_arm": BRANCH_COUNT,
        "arms": {
            "A": "single_agent_from_frozen_evidence",
            "B": "frozen_multi_agent_scenario_artifact",
        },
        "multi_agent_selection_rule": (
            "Within the replication_ids sealed in the final B3 submission, "
            "select the earliest replication whose frozen scenario artifact "
            "contains exactly four branches. The rule does not inspect outcomes."
        ),
        "quality_judging": {
            "outcome_hidden": True,
            "anonymous_labels": ["X", "Y"],
            "order_swapped": True,
            "score_keys": list(QUALITY_SCORE_KEYS),
            "score_range": [0, 4],
            "branch_qualification_keys": list(BRANCH_QUALIFICATION_KEYS),
            "branch_qualification_rule": (
                "A branch qualifies only when both swapped-order judgments mark "
                "all five qualification keys true."
            ),
            "pairwise_win_rule": (
                "An arm wins a case only when its summed quality score is higher "
                "in both swapped-order judgments; all other cases are ties."
            ),
        },
        "outcome_path_judging": {
            "market_targets_excluded": True,
            "event_target_statuses": list(OUTCOME_STATUSES),
            "full_rule": (
                "Both swapped-order judgments must rate the target full."
            ),
            "partial_rule": (
                "Both judgments must rate the target at least partial, while the "
                "full rule is not met. All other disagreements resolve to miss."
            ),
            "negative_target_rule": (
                "A false outcome is covered only by an explicit prevention, delay, "
                "or status-quo mechanism; mere omission of the event is not a hit."
            ),
            "primary_metric": "full_event_target_recall_at_4",
        },
        "interpretation_limits": [
            "The operator knew the historical outcomes before this protocol was created.",
            "Generation and outcome-free quality judging never open outcome files.",
            "Results are descriptive method validation, not a new blind holdout.",
            "Scenario recall and quality are decision-support metrics, not trading returns.",
            "The frozen B arm includes the complete simulation-plus-compiler system effect.",
        ],
    }


def _extract_payload(raw_response: Any) -> Dict[str, Any]:
    if isinstance(raw_response, dict):
        return deepcopy(raw_response)
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("model response is empty")
    candidate = raw_response.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        candidate = "\n".join(lines).strip()
    try:
        whole = json.loads(candidate)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, dict):
        return whole
    decoder = json.JSONDecoder()
    decoded: list[tuple[int, Dict[str, Any]]] = []
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            decoded.append((end, value))
    preferred = [
        item
        for item in decoded
        if _find_labeled_sets(item[1]) is not None
        or "branches" in item[1]
        or "target_predictions" in item[1]
    ]
    if preferred:
        return max(preferred, key=lambda item: item[0])[1]
    if decoded:
        return max(decoded, key=lambda item: item[0])[1]
    raise ValueError("model response does not contain a JSON object")


def _find_labeled_sets(value: Any) -> Dict[str, Any] | None:
    """Find harmless provider wrappers around anonymous X/Y set objects."""

    if isinstance(value, dict):
        nested_sets = value.get("sets")
        if (
            isinstance(nested_sets, dict)
            and isinstance(nested_sets.get("X"), dict)
            and isinstance(nested_sets.get("Y"), dict)
        ):
            return {"X": nested_sets["X"], "Y": nested_sets["Y"]}
        x_key = next(
            (
                key
                for key in ("X", "x", "set_X", "scenario_set_X")
                if isinstance(value.get(key), dict)
            ),
            None,
        )
        y_key = next(
            (
                key
                for key in ("Y", "y", "set_Y", "scenario_set_Y")
                if isinstance(value.get(key), dict)
            ),
            None,
        )
        if x_key is not None and y_key is not None:
            return {"X": value[x_key], "Y": value[y_key]}
        for child in value.values():
            found = _find_labeled_sets(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_labeled_sets(child)
            if found is not None:
                return found
    return None


def _compact_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
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
        "relationships": spec["relationships"],
        "allowed_action_types": [
            item["type"] for item in spec["allowed_actions"]
        ],
    }


def build_single_agent_scenario_prompt(
    spec: Dict[str, Any],
) -> tuple[str, str]:
    """Build a four-branch baseline prompt that cannot access outcomes."""

    validate_spec(spec)
    system = f"""你是金融事件情景规划分析师，不是价格预测器。仅使用 as_of 时点已经给出的
事实、参与方与关系，独立生成恰好 {BRANCH_COUNT} 条未来情景。你看不到任何历史结果，
不得补充 as_of 之后的真实公告、价格、人员变动或其他后验知识。

硬性要求：
- 四条情景必须在关键参与方、行动链或触发机制上有实质区别，不能只换措辞；
- 每条情景至少包含两个不同 actor，actions 至少覆盖两个 actor；
- actor_id、action_type 必须逐字使用输入中的 ID 与 allowed_action_types；
- summary 与 consequences 必须表达“参与方行动 → 对方反应 → 事件后果”的条件化机制；
- trigger_conditions 必须是未来可观察信号，不能写成模糊情绪；
- invalidation_conditions 必须说明什么未来观察会推翻该分支；
- evidence_refs 只能引用输入中的 F 编号，且不得把情景当作事实；
- 不输出目标价、收益率概率、发生概率或置信度。

只输出 JSON：
{{"branches":[{{"id":"branch-1","label":"简短名称","summary":"条件化机制摘要",
"actor_ids":["actor_a","actor_b"],"actions":[{{"actor_id":"actor_a",
"action_type":"COMMUNICATE"}},{{"actor_id":"actor_b","action_type":"WAIT"}}],
"trigger_conditions":["未来可观察条件"],"consequences":["条件成立后的机制后果"],
"invalidation_conditions":["推翻条件"],"evidence_refs":["F1"]}}],"warnings":[]}}。
不得输出 Markdown 或额外字段。"""
    return system, json.dumps(_compact_spec(spec), ensure_ascii=False)


def build_semantic_retry_prompt(
    system_prompt: str,
    validation_error: BaseException,
) -> str:
    message = " ".join(str(validation_error).split())[:240]
    return (
        system_prompt
        + "\n\n上一次输出未通过结构校验。不要参考或复述上一次回答；"
        + "请仅根据原始匿名输入重新生成完整 JSON。验证错误："
        + message
    )


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_string_list(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    return [item.strip() for item in value]


def _normalize_branch(
    raw: Dict[str, Any],
    *,
    index: int,
    actor_ids: set[str],
    action_types: set[str],
    fact_ids: set[str],
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("scenario branch must be an object")
    branch_actor_ids = _require_string_list(
        raw.get("actor_ids"), f"branch-{index} actor_ids"
    )
    if len(branch_actor_ids) < 2 or len(branch_actor_ids) != len(
        set(branch_actor_ids)
    ):
        raise ValueError("scenario branch must name at least two unique actors")
    unknown_actors = set(branch_actor_ids) - actor_ids
    if unknown_actors:
        raise ValueError(
            "scenario branch contains unknown actors: "
            + ", ".join(sorted(unknown_actors))
        )
    raw_actions = raw.get("actions")
    if not isinstance(raw_actions, list) or len(raw_actions) < 2:
        raise ValueError("scenario branch must contain at least two actions")
    actions = []
    action_actor_ids = set()
    for action in raw_actions:
        if not isinstance(action, dict):
            raise ValueError("scenario action must be an object")
        actor_id = _require_string(action.get("actor_id"), "action actor_id")
        action_type = _require_string(
            action.get("action_type"), "action action_type"
        )
        if actor_id not in branch_actor_ids:
            raise ValueError("scenario action actor must appear in actor_ids")
        if action_type not in action_types:
            raise ValueError(f"unknown scenario action type: {action_type}")
        normalized_action = {
            "actor_id": actor_id,
            "action_type": action_type,
        }
        decision_ref = action.get("decision_ref")
        if decision_ref is not None:
            normalized_action["decision_ref"] = _require_string(
                decision_ref, "action decision_ref"
            )
        actions.append(normalized_action)
        action_actor_ids.add(actor_id)
    if len(action_actor_ids) < 2:
        raise ValueError("scenario actions must cover at least two actors")
    evidence_refs = _require_string_list(
        raw.get("evidence_refs"), f"branch-{index} evidence_refs"
    )
    unknown_facts = set(evidence_refs) - fact_ids
    if unknown_facts:
        raise ValueError(
            "scenario branch contains unknown facts: "
            + ", ".join(sorted(unknown_facts))
        )
    return {
        "id": f"branch-{index}",
        "label": _require_string(raw.get("label"), f"branch-{index} label"),
        "summary": _require_string(
            raw.get("summary"), f"branch-{index} summary"
        ),
        "actor_ids": branch_actor_ids,
        "actions": actions,
        "trigger_conditions": _require_string_list(
            raw.get("trigger_conditions"),
            f"branch-{index} trigger_conditions",
        ),
        "consequences": _require_string_list(
            raw.get("consequences"), f"branch-{index} consequences"
        ),
        "invalidation_conditions": _require_string_list(
            raw.get("invalidation_conditions"),
            f"branch-{index} invalidation_conditions",
        ),
        "evidence_refs": evidence_refs,
    }


def _validate_common_scenario_set(
    artifact: Dict[str, Any], spec: Dict[str, Any]
) -> None:
    validate_spec(spec)
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported decision-support scenario schema")
    if artifact.get("evaluation_version") != EVALUATION_VERSION:
        raise ValueError("decision-support evaluation version mismatch")
    if artifact.get("case_id") != spec["case_id"]:
        raise ValueError("scenario case_id does not match spec")
    if artifact.get("spec_sha256") != canonical_sha256(spec):
        raise ValueError("scenario spec hash does not match")
    if artifact.get("arm") not in {"A", "B"}:
        raise ValueError("scenario arm must be A or B")
    branches = artifact.get("branches")
    if not isinstance(branches, list) or len(branches) != BRANCH_COUNT:
        raise ValueError(f"scenario set must contain exactly {BRANCH_COUNT} branches")
    actor_ids = {item["id"] for item in spec["actors"]}
    action_types = {item["type"] for item in spec["allowed_actions"]}
    fact_ids = {item["id"] for item in spec["facts"]}
    normalized = [
        _normalize_branch(
            item,
            index=index,
            actor_ids=actor_ids,
            action_types=action_types,
            fact_ids=fact_ids,
        )
        for index, item in enumerate(branches, start=1)
    ]
    if normalized != branches:
        raise ValueError("scenario branches are not canonically normalized")


def build_single_agent_scenario_set(
    raw_response: Any,
    spec: Dict[str, Any],
    *,
    model_id: str,
) -> Dict[str, Any]:
    """Normalize a model response into the common A-arm representation."""

    validate_spec(spec)
    payload = _extract_payload(raw_response)
    raw_branches = payload.get("branches")
    if not isinstance(raw_branches, list) or len(raw_branches) != BRANCH_COUNT:
        raise ValueError(
            f"single-agent response must contain exactly {BRANCH_COUNT} branches"
        )
    actor_ids = {item["id"] for item in spec["actors"]}
    action_types = {item["type"] for item in spec["allowed_actions"]}
    fact_ids = {item["id"] for item in spec["facts"]}
    branches = [
        _normalize_branch(
            item,
            index=index,
            actor_ids=actor_ids,
            action_types=action_types,
            fact_ids=fact_ids,
        )
        for index, item in enumerate(raw_branches, start=1)
    ]
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise ValueError("scenario warnings must be a string list")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "arm": "A",
        "source_kind": "single_agent_from_frozen_evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "prompt_version": SINGLE_AGENT_PROMPT_VERSION,
        "branches": branches,
        "warnings": warnings,
    }
    _validate_common_scenario_set(artifact, spec)
    return artifact


def normalize_multi_agent_scenario_set(
    source: Dict[str, Any],
    spec: Dict[str, Any],
    *,
    source_sha256: str | None = None,
) -> Dict[str, Any]:
    """Map one frozen four-branch B artifact into the common representation."""

    validate_spec(spec)
    if source.get("case_id") != spec["case_id"]:
        raise ValueError("multi-agent scenario case_id does not match spec")
    raw_branches = source.get("branches")
    if not isinstance(raw_branches, list) or len(raw_branches) != BRANCH_COUNT:
        raise ValueError(
            f"multi-agent source must contain exactly {BRANCH_COUNT} branches"
        )
    actor_ids = {item["id"] for item in spec["actors"]}
    action_types = {item["type"] for item in spec["allowed_actions"]}
    fact_ids = {item["id"] for item in spec["facts"]}
    branches = [
        _normalize_branch(
            item,
            index=index,
            actor_ids=actor_ids,
            action_types=action_types,
            fact_ids=fact_ids,
        )
        for index, item in enumerate(raw_branches, start=1)
    ]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "arm": "B",
        "source_kind": "frozen_multi_agent_scenario_artifact",
        "generated_at": source.get("generated_at"),
        "model_id": source.get("model_id"),
        "prompt_version": source.get("prompt_version"),
        "source_simulation_id": source.get("simulation_id"),
        "source_artifact_sha256": source_sha256 or canonical_sha256(source),
        "branches": branches,
        "warnings": [
            "This common representation preserves a pre-existing, outcome-blind artifact."
        ],
    }
    _validate_common_scenario_set(artifact, spec)
    return artifact


def _judge_view(
    scenario_set: Dict[str, Any], spec: Dict[str, Any]
) -> Dict[str, Any]:
    actor_labels = {item["id"]: item["label"] for item in spec["actors"]}
    return {
        "branches": [
            {
                "id": branch["id"],
                "label": branch["label"],
                "summary": branch["summary"],
                "actors": [actor_labels[item] for item in branch["actor_ids"]],
                "actions": [
                    {
                        "actor": actor_labels[item["actor_id"]],
                        "action_type": item["action_type"],
                    }
                    for item in branch["actions"]
                ],
                "trigger_conditions": branch["trigger_conditions"],
                "consequences": branch["consequences"],
                "invalidation_conditions": branch["invalidation_conditions"],
                "evidence_refs": branch["evidence_refs"],
            }
            for branch in scenario_set["branches"]
        ]
    }


def build_quality_judge_prompt(
    spec: Dict[str, Any],
    scenario_x: Dict[str, Any],
    scenario_y: Dict[str, Any],
) -> tuple[str, str]:
    """Build an outcome-free anonymous pairwise quality prompt."""

    _validate_common_scenario_set(scenario_x, spec)
    _validate_common_scenario_set(scenario_y, spec)
    system = """你是金融情景规划的匿名质量评审。你只能评估 D0 事实与两组候选情景，
绝不能猜测或补充真实历史结果，也不能根据文风推断生成方法。X、Y 每组固定四条。

分别按 0–4 整数评分：
- mechanism_coherence：参与方动作、反应与后果是否构成一致的因果链；
- monitoring_actionability：触发条件是否能转成具体观察清单；
- falsifiability：失效条件是否具体，能否明确推翻分支；
- evidence_discipline：是否以 D0 事实为锚，并把未来内容保持为条件假设；
- scenario_diversity：四条分支是否覆盖实质不同机制，而不是同义改写。

逐条给出五个布尔判断：multi_actor_causal_chain、observable_trigger、
specific_invalidation、decision_relevant、evidence_grounded。标准应严格：只要动作标签与摘要
明显矛盾、触发无法观察、失效只是反义复述或后果对决策无用，相应项就为 false。
最后选总体更有助于“接下来观察什么、什么情况下改变判断”的一组；难分高下时 winner=tie。

只输出 JSON。X 和 Y 都必须完整输出五项 scores，并各自完整评估 branch-1 至 branch-4：
{"sets":{"X":{"scores":{"mechanism_coherence":0,"monitoring_actionability":0,
"falsifiability":0,"evidence_discipline":0,"scenario_diversity":0},
"branches":[{"id":"branch-1","multi_actor_causal_chain":true,
"observable_trigger":true,"specific_invalidation":true,"decision_relevant":true,
"evidence_grounded":true}]},"Y":{"scores":{"mechanism_coherence":0,
"monitoring_actionability":0,"falsifiability":0,"evidence_discipline":0,
"scenario_diversity":0},"branches":[{"id":"branch-1",
"multi_actor_causal_chain":true,"observable_trigger":true,
"specific_invalidation":true,"decision_relevant":true,"evidence_grounded":true}]}},
"winner":"X","winner_reason":"一句话"}。不得输出 Markdown 或额外字段。"""
    compact = _compact_spec(spec)
    compact.pop("allowed_action_types", None)
    user = json.dumps(
        {
            "frozen_input": compact,
            "scenario_set_X": _judge_view(scenario_x, spec),
            "scenario_set_Y": _judge_view(scenario_y, spec),
        },
        ensure_ascii=False,
    )
    return system, user


def validate_quality_judgment(
    raw_response: Any,
    *,
    case_id: str,
    label_to_arm: Dict[str, str],
) -> Dict[str, Any]:
    payload = _extract_payload(raw_response)
    sets = _find_labeled_sets(payload)
    if not isinstance(sets, dict) or set(sets) != {"X", "Y"}:
        raise ValueError("quality judgment must contain sets X and Y")
    normalized_sets: Dict[str, Any] = {}
    for label in ("X", "Y"):
        value = sets[label]
        if not isinstance(value, dict):
            raise ValueError("quality set judgment must be an object")
        scores = value.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(
            QUALITY_SCORE_KEYS
        ):
            raise ValueError("quality score keys do not match protocol")
        normalized_scores = {}
        for key in QUALITY_SCORE_KEYS:
            score = scores[key]
            if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
                raise ValueError(f"quality score {key} must be an integer from 0 to 4")
            normalized_scores[key] = score
        branches = value.get("branches")
        if not isinstance(branches, list) or len(branches) != BRANCH_COUNT:
            raise ValueError("quality judgment must assess all four branches")
        normalized_branches = []
        expected_ids = {f"branch-{index}" for index in range(1, BRANCH_COUNT + 1)}
        seen_ids = set()
        for branch in branches:
            if not isinstance(branch, dict):
                raise ValueError("quality branch assessment must be an object")
            branch_id = branch.get("id")
            if branch_id not in expected_ids or branch_id in seen_ids:
                raise ValueError("quality branch ids must be unique branch-1..branch-4")
            seen_ids.add(branch_id)
            normalized = {"id": branch_id}
            for key in BRANCH_QUALIFICATION_KEYS:
                flag = branch.get(key)
                if not isinstance(flag, bool):
                    raise ValueError(f"quality branch flag {key} must be boolean")
                normalized[key] = flag
            normalized_branches.append(normalized)
        normalized_branches.sort(key=lambda item: item["id"])
        normalized_sets[label] = {
            "arm": label_to_arm[label],
            "scores": normalized_scores,
            "total_score": sum(normalized_scores.values()),
            "branches": normalized_branches,
        }
    winner = payload.get("winner")
    if winner not in {"X", "Y", "tie"}:
        score_x = sum(normalized_sets["X"]["scores"].values())
        score_y = sum(normalized_sets["Y"]["scores"].values())
        winner = "X" if score_x > score_y else "Y" if score_y > score_x else "tie"
    reason = payload.get("winner_reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "Winner label was derived deterministically from the supplied scores."
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "case_id": case_id,
        "judge_version": QUALITY_JUDGE_VERSION,
        "outcomes_read": False,
        "label_to_arm": dict(label_to_arm),
        "sets": normalized_sets,
        "winner_label": winner,
        "winner_arm": label_to_arm.get(winner) if winner != "tie" else "tie",
        "winner_reason": reason.strip(),
    }


def build_absolute_quality_judge_prompt(
    spec: Dict[str, Any],
    scenario_set: Dict[str, Any],
) -> tuple[str, str]:
    """Build an outcome-free, position-independent quality prompt."""

    _validate_common_scenario_set(scenario_set, spec)
    system = """你是金融情景规划的绝对质量评审。你只评估一组固定四条的候选情景 S，
不能与任何未展示的方案比较，也绝不能猜测或补充真实历史结果。

分别按 0–4 整数评分：
- mechanism_coherence：参与方动作、反应与后果是否构成一致的因果链；
- monitoring_actionability：触发条件是否能转成具体观察清单；
- falsifiability：失效条件是否具体，能否明确推翻分支；
- evidence_discipline：是否以 D0 事实为锚，并把未来内容保持为条件假设；
- scenario_diversity：四条分支是否覆盖实质不同机制，而不是同义改写。

逐条给出五个布尔判断：multi_actor_causal_chain、observable_trigger、
specific_invalidation、decision_relevant、evidence_grounded。标准应严格：actions 是系统绑定的
模拟起始动作；如果摘要或后果与这些动作直接矛盾、只有单方叙述、触发无法观察、失效只是反义
复述或后果对决策无用，相应项就为 false。不要因文字更长或语气更肯定而提高评分。

只输出 JSON，必须完整输出五项 scores，并评估 branch-1 至 branch-4：
{"set":{"scores":{"mechanism_coherence":0,"monitoring_actionability":0,
"falsifiability":0,"evidence_discipline":0,"scenario_diversity":0},
"branches":[{"id":"branch-1","multi_actor_causal_chain":true,
"observable_trigger":true,"specific_invalidation":true,"decision_relevant":true,
"evidence_grounded":true}]},"overall_reason":"一句话"}。
不得输出 Markdown 或额外字段。"""
    compact = _compact_spec(spec)
    compact.pop("allowed_action_types", None)
    user = json.dumps(
        {
            "frozen_input": compact,
            "scenario_set_S": _judge_view(scenario_set, spec),
        },
        ensure_ascii=False,
    )
    return system, user


def validate_absolute_quality_judgment(
    raw_response: Any,
    *,
    case_id: str,
    variant: str,
) -> Dict[str, Any]:
    """Validate one absolute quality judgment without an X/Y position."""

    payload = _extract_payload(raw_response)
    value = payload.get("set")
    if not isinstance(value, dict):
        value = payload.get("S")
    if not isinstance(value, dict):
        raise ValueError("absolute quality judgment must contain set S")
    scores = value.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(QUALITY_SCORE_KEYS):
        raise ValueError("absolute quality score keys do not match protocol")
    normalized_scores: Dict[str, int] = {}
    for key in QUALITY_SCORE_KEYS:
        score = scores[key]
        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 4
        ):
            raise ValueError(
                f"absolute quality score {key} must be an integer from 0 to 4"
            )
        normalized_scores[key] = score
    branches = value.get("branches")
    if not isinstance(branches, list) or len(branches) != BRANCH_COUNT:
        raise ValueError("absolute quality judgment must assess all four branches")
    expected_ids = {
        f"branch-{index}" for index in range(1, BRANCH_COUNT + 1)
    }
    seen_ids = set()
    normalized_branches = []
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("absolute quality branch assessment must be an object")
        branch_id = branch.get("id")
        if branch_id not in expected_ids or branch_id in seen_ids:
            raise ValueError(
                "absolute quality branch ids must be unique branch-1..branch-4"
            )
        seen_ids.add(branch_id)
        normalized = {"id": branch_id}
        for key in BRANCH_QUALIFICATION_KEYS:
            flag = branch.get(key)
            if not isinstance(flag, bool):
                raise ValueError(
                    f"absolute quality branch flag {key} must be boolean"
                )
            normalized[key] = flag
        normalized_branches.append(normalized)
    normalized_branches.sort(key=lambda item: item["id"])
    reason = payload.get("overall_reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No free-form reason supplied."
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "case_id": case_id,
        "judge_version": ABSOLUTE_QUALITY_JUDGE_VERSION,
        "outcomes_read": False,
        "variant": variant,
        "set": {
            "scores": normalized_scores,
            "total_score": sum(normalized_scores.values()),
            "branches": normalized_branches,
        },
        "overall_reason": reason.strip(),
    }


def _event_target_outcomes(
    spec: Dict[str, Any], outcome: Dict[str, Any]
) -> list[Dict[str, Any]]:
    target_by_id = {item["id"]: item for item in spec["forecast_targets"]}
    rows = []
    for result in outcome.get("target_results", []):
        target = target_by_id.get(result.get("target_id"))
        if target and target.get("kind") == "event":
            rows.append(
                {
                    "target_id": target["id"],
                    "definition": target["definition"],
                    "horizon": target["horizon"],
                    "observed": result.get("observed"),
                }
            )
    if not rows:
        raise ValueError("outcome contains no event targets")
    return rows


def build_outcome_judge_prompt(
    spec: Dict[str, Any],
    outcome: Dict[str, Any],
    scenario_x: Dict[str, Any],
    scenario_y: Dict[str, Any],
) -> tuple[str, str, list[str]]:
    """Build an anonymous post-unseal event-path recall prompt."""

    _validate_common_scenario_set(scenario_x, spec)
    _validate_common_scenario_set(scenario_y, spec)
    if outcome.get("case_id") != spec["case_id"]:
        raise ValueError("outcome case_id does not match spec")
    if outcome.get("spec_sha256") != canonical_sha256(spec):
        raise ValueError("outcome spec hash does not match")
    event_rows = _event_target_outcomes(spec, outcome)
    target_ids = [item["target_id"] for item in event_rows]
    system = """你是历史事件路径覆盖评审。现在结果已经解封，但任务不是评价价格预测，
而是判断每组固定四条事前情景是否覆盖了每个事件目标的实际发展机制。

对 X、Y 的每个 event target 独立标记：
- full：至少一条分支在事前明确给出与结果一致的关键参与方行动、对方反应和事件后果；
- partial：方向或邻近机制相关，但缺少关键参与方、传导环节或具体结果；
- miss：没有覆盖，或只有宽泛到几乎任何结果都能套用的描述。

严格规则：observed.value=false 时，只有分支明确描述阻止、推迟或维持现状的机制才算覆盖；
仅仅没有提到该事件不能算命中。不要因为一组写得更长而提高等级。matched_branch_ids 只能使用
该组 branch-1 到 branch-4；miss 时必须为空。逐目标分别判断，不输出概率。

只输出 JSON。X 和 Y 都必须完整输出输入中的每一个 event target：
{"sets":{"X":{"targets":[{"target_id":"T4","status":"full",
"matched_branch_ids":["branch-2"],"reason":"一句话"}]},
"Y":{"targets":[{"target_id":"T4","status":"partial",
"matched_branch_ids":["branch-1"],"reason":"一句话"}]}}}。
不得输出 Markdown 或额外字段。"""
    compact = _compact_spec(spec)
    compact.pop("allowed_action_types", None)
    user = json.dumps(
        {
            "frozen_input": compact,
            "observed_event_targets": event_rows,
            "scenario_set_X": _judge_view(scenario_x, spec),
            "scenario_set_Y": _judge_view(scenario_y, spec),
        },
        ensure_ascii=False,
    )
    return system, user, target_ids


def validate_outcome_judgment(
    raw_response: Any,
    *,
    case_id: str,
    label_to_arm: Dict[str, str],
    target_ids: Iterable[str],
) -> Dict[str, Any]:
    payload = _extract_payload(raw_response)
    sets = _find_labeled_sets(payload)
    if not isinstance(sets, dict) or set(sets) != {"X", "Y"}:
        raise ValueError("outcome judgment must contain sets X and Y")
    expected_targets = set(target_ids)
    normalized_sets: Dict[str, Any] = {}
    valid_branch_ids = {
        f"branch-{index}" for index in range(1, BRANCH_COUNT + 1)
    }
    for label in ("X", "Y"):
        value = sets[label]
        targets = value.get("targets") if isinstance(value, dict) else None
        if not isinstance(targets, list) or len(targets) != len(expected_targets):
            raise ValueError("outcome judgment must assess every event target")
        normalized_targets = []
        seen_targets = set()
        for target in targets:
            if not isinstance(target, dict):
                raise ValueError("outcome target judgment must be an object")
            target_id = target.get("target_id")
            if target_id not in expected_targets or target_id in seen_targets:
                raise ValueError("outcome target ids must exactly match the input")
            seen_targets.add(target_id)
            status = target.get("status")
            if status not in OUTCOME_STATUSES:
                raise ValueError("outcome status must be full, partial, or miss")
            refs = target.get("matched_branch_ids")
            if not isinstance(refs, list) or not all(
                isinstance(item, str) for item in refs
            ):
                raise ValueError("matched_branch_ids must be a string list")
            if len(refs) != len(set(refs)) or not set(refs) <= valid_branch_ids:
                raise ValueError("matched_branch_ids contain duplicates or unknown ids")
            if status == "miss" and refs:
                raise ValueError("miss judgments must not name matched branches")
            if status != "miss" and not refs:
                raise ValueError("full/partial judgments must name a matched branch")
            normalized_targets.append(
                {
                    "target_id": target_id,
                    "status": status,
                    "matched_branch_ids": refs,
                    "reason": _require_string(
                        target.get("reason"), "outcome target reason"
                    ),
                }
            )
        normalized_targets.sort(key=lambda item: item["target_id"])
        normalized_sets[label] = {
            "arm": label_to_arm[label],
            "targets": normalized_targets,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "case_id": case_id,
        "judge_version": OUTCOME_JUDGE_VERSION,
        "outcomes_read": True,
        "label_to_arm": dict(label_to_arm),
        "sets": normalized_sets,
    }


def build_absolute_outcome_judge_prompt(
    spec: Dict[str, Any],
    outcome: Dict[str, Any],
    scenario_set: Dict[str, Any],
) -> tuple[str, str, list[str]]:
    """Build a single-set post-unseal event-path recall prompt."""

    _validate_common_scenario_set(scenario_set, spec)
    if outcome.get("case_id") != spec["case_id"]:
        raise ValueError("outcome case_id does not match spec")
    if outcome.get("spec_sha256") != canonical_sha256(spec):
        raise ValueError("outcome spec hash does not match")
    event_rows = _event_target_outcomes(spec, outcome)
    target_ids = [item["target_id"] for item in event_rows]
    system = """你是历史事件路径覆盖的绝对评审。你只评估一组固定四条的事前情景 S，
不能与任何未展示的方案比较。现在结果已经解封，但任务不是评价价格预测，而是判断 S 是否
覆盖了每个事件目标的实际发展机制。

对每个 event target 独立标记：
- full：至少一条分支在事前明确给出与结果一致的关键参与方行动、对方反应和事件后果；
- partial：方向或邻近机制相关，但缺少关键参与方、传导环节或具体结果；
- miss：没有覆盖，或只有宽泛到几乎任何结果都能套用的描述。

严格规则：observed.value=false 时，只有分支明确描述阻止、推迟或维持现状的机制才算覆盖；
仅仅没有提到该事件不能算命中。不要因为文字更长而提高等级。matched_branch_ids 只能使用
branch-1 到 branch-4；miss 时必须为空。逐目标分别判断，不输出概率。

只输出 JSON，必须完整输出输入中的每一个 event target：
{"targets":[{"target_id":"T4","status":"full",
"matched_branch_ids":["branch-2"],"reason":"一句话"}]}。
不得输出 Markdown 或额外字段。"""
    compact = _compact_spec(spec)
    compact.pop("allowed_action_types", None)
    user = json.dumps(
        {
            "frozen_input": compact,
            "observed_event_targets": event_rows,
            "scenario_set_S": _judge_view(scenario_set, spec),
        },
        ensure_ascii=False,
    )
    return system, user, target_ids


def validate_absolute_outcome_judgment(
    raw_response: Any,
    *,
    case_id: str,
    variant: str,
    target_ids: Iterable[str],
) -> Dict[str, Any]:
    """Validate one absolute event-path judgment."""

    payload = _extract_payload(raw_response)
    targets = payload.get("targets")
    expected_targets = set(target_ids)
    if not isinstance(targets, list) or len(targets) != len(expected_targets):
        raise ValueError("absolute outcome judgment must assess every event target")
    valid_branch_ids = {
        f"branch-{index}" for index in range(1, BRANCH_COUNT + 1)
    }
    seen_targets = set()
    normalized_targets = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("absolute outcome target judgment must be an object")
        target_id = target.get("target_id")
        if target_id not in expected_targets or target_id in seen_targets:
            raise ValueError("absolute outcome target ids must exactly match input")
        seen_targets.add(target_id)
        status = target.get("status")
        if status not in OUTCOME_STATUSES:
            raise ValueError("absolute outcome status must be full, partial, or miss")
        refs = target.get("matched_branch_ids")
        if not isinstance(refs, list) or not all(
            isinstance(item, str) for item in refs
        ):
            raise ValueError("absolute matched_branch_ids must be a string list")
        if len(refs) != len(set(refs)) or not set(refs) <= valid_branch_ids:
            raise ValueError(
                "absolute matched_branch_ids contain duplicates or unknown ids"
            )
        if status == "miss" and refs:
            raise ValueError("absolute miss target must not name matched branches")
        reason = target.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "No free-form reason supplied."
        normalized_targets.append(
            {
                "target_id": target_id,
                "status": status,
                "matched_branch_ids": sorted(refs),
                "reason": reason.strip(),
            }
        )
    normalized_targets.sort(key=lambda item: item["target_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "case_id": case_id,
        "judge_version": ABSOLUTE_OUTCOME_JUDGE_VERSION,
        "outcomes_read": True,
        "variant": variant,
        "targets": normalized_targets,
    }


def arm_view(judgment: Dict[str, Any], arm: str) -> Dict[str, Any]:
    """Return the anonymous set restored to its underlying arm."""

    for value in judgment["sets"].values():
        if value["arm"] == arm:
            return value
    raise ValueError(f"judgment does not contain arm {arm}")


def reconcile_quality_passes(
    first: Dict[str, Any], second: Dict[str, Any]
) -> Dict[str, Any]:
    """Conservatively combine a swapped-order quality pair."""

    if first["case_id"] != second["case_id"]:
        raise ValueError("quality pass case ids do not match")
    arms: Dict[str, Any] = {}
    for arm in ("A", "B"):
        left = arm_view(first, arm)
        right = arm_view(second, arm)
        scores = {
            key: (left["scores"][key] + right["scores"][key]) / 2
            for key in QUALITY_SCORE_KEYS
        }
        right_branches = {item["id"]: item for item in right["branches"]}
        branches = []
        for branch in left["branches"]:
            other = right_branches[branch["id"]]
            flags = {
                key: branch[key] and other[key]
                for key in BRANCH_QUALIFICATION_KEYS
            }
            branches.append(
                {
                    "id": branch["id"],
                    **flags,
                    "qualified": all(flags.values()),
                }
            )
        arms[arm] = {
            "scores": scores,
            "total_score": sum(scores.values()),
            "branches": branches,
            "qualified_branch_count": sum(
                item["qualified"] for item in branches
            ),
        }
    first_delta = arm_view(first, "B")["total_score"] - arm_view(
        first, "A"
    )["total_score"]
    second_delta = arm_view(second, "B")["total_score"] - arm_view(
        second, "A"
    )["total_score"]
    winner = "tie"
    if first_delta > 0 and second_delta > 0:
        winner = "B"
    elif first_delta < 0 and second_delta < 0:
        winner = "A"
    return {
        "case_id": first["case_id"],
        "arms": arms,
        "pass_score_deltas_b_minus_a": [first_delta, second_delta],
        "order_consistent_winner": winner,
    }


def reconcile_outcome_passes(
    first: Dict[str, Any], second: Dict[str, Any]
) -> Dict[str, Any]:
    """Conservatively combine a swapped-order event-path pair."""

    if first["case_id"] != second["case_id"]:
        raise ValueError("outcome pass case ids do not match")
    arms: Dict[str, Any] = {}
    rank = {"miss": 0, "partial": 1, "full": 2}
    for arm in ("A", "B"):
        left_targets = {
            item["target_id"]: item for item in arm_view(first, arm)["targets"]
        }
        right_targets = {
            item["target_id"]: item for item in arm_view(second, arm)["targets"]
        }
        if set(left_targets) != set(right_targets):
            raise ValueError("outcome pass targets do not match")
        targets = []
        for target_id in sorted(left_targets):
            left_status = left_targets[target_id]["status"]
            right_status = right_targets[target_id]["status"]
            if left_status == right_status == "full":
                status = "full"
            elif rank[left_status] >= 1 and rank[right_status] >= 1:
                status = "partial"
            else:
                status = "miss"
            targets.append(
                {
                    "target_id": target_id,
                    "status": status,
                    "pass_statuses": [left_status, right_status],
                    "matched_branch_ids": sorted(
                        set(left_targets[target_id]["matched_branch_ids"])
                        | set(right_targets[target_id]["matched_branch_ids"])
                    )
                    if status != "miss"
                    else [],
                }
            )
        arms[arm] = {
            "targets": targets,
            "full_count": sum(item["status"] == "full" for item in targets),
            "partial_count": sum(
                item["status"] == "partial" for item in targets
            ),
            "miss_count": sum(item["status"] == "miss" for item in targets),
        }
    winner = "tie"
    score_a = (arms["A"]["full_count"], arms["A"]["partial_count"])
    score_b = (arms["B"]["full_count"], arms["B"]["partial_count"])
    if score_b > score_a:
        winner = "B"
    elif score_a > score_b:
        winner = "A"
    return {
        "case_id": first["case_id"],
        "arms": arms,
        "recall_winner": winner,
    }


def reconcile_absolute_quality_passes(
    first: Dict[str, Any], second: Dict[str, Any]
) -> Dict[str, Any]:
    """Conservatively combine two position-independent quality judgments."""

    if first["case_id"] != second["case_id"]:
        raise ValueError("absolute quality pass case ids do not match")
    if first["variant"] != second["variant"]:
        raise ValueError("absolute quality pass variants do not match")
    left = first["set"]
    right = second["set"]
    scores = {
        key: (left["scores"][key] + right["scores"][key]) / 2
        for key in QUALITY_SCORE_KEYS
    }
    right_branches = {item["id"]: item for item in right["branches"]}
    branches = []
    flag_agreements = []
    for branch in left["branches"]:
        other = right_branches[branch["id"]]
        flags = {
            key: branch[key] and other[key]
            for key in BRANCH_QUALIFICATION_KEYS
        }
        flag_agreements.extend(
            branch[key] == other[key] for key in BRANCH_QUALIFICATION_KEYS
        )
        branches.append(
            {
                "id": branch["id"],
                **flags,
                "qualified": all(flags.values()),
            }
        )
    score_pairs = [
        [left["scores"][key], right["scores"][key]]
        for key in QUALITY_SCORE_KEYS
    ]
    return {
        "case_id": first["case_id"],
        "variant": first["variant"],
        "scores": scores,
        "total_score": sum(scores.values()),
        "pass_total_scores": [left["total_score"], right["total_score"]],
        "branches": branches,
        "qualified_branch_count": sum(item["qualified"] for item in branches),
        "reliability": {
            "score_exact_agreement_count": sum(
                values[0] == values[1] for values in score_pairs
            ),
            "score_within_one_count": sum(
                abs(values[0] - values[1]) <= 1 for values in score_pairs
            ),
            "score_comparison_count": len(score_pairs),
            "branch_flag_agreement_count": sum(flag_agreements),
            "branch_flag_comparison_count": len(flag_agreements),
        },
    }


def reconcile_absolute_outcome_passes(
    first: Dict[str, Any], second: Dict[str, Any]
) -> Dict[str, Any]:
    """Conservatively combine two position-independent path judgments."""

    if first["case_id"] != second["case_id"]:
        raise ValueError("absolute outcome pass case ids do not match")
    if first["variant"] != second["variant"]:
        raise ValueError("absolute outcome pass variants do not match")
    left_targets = {item["target_id"]: item for item in first["targets"]}
    right_targets = {item["target_id"]: item for item in second["targets"]}
    if set(left_targets) != set(right_targets):
        raise ValueError("absolute outcome pass targets do not match")
    rank = {"miss": 0, "partial": 1, "full": 2}
    targets = []
    agreement_count = 0
    for target_id in sorted(left_targets):
        left = left_targets[target_id]
        right = right_targets[target_id]
        left_status = left["status"]
        right_status = right["status"]
        agreement_count += left_status == right_status
        if left_status == right_status == "full":
            status = "full"
        elif rank[left_status] >= 1 and rank[right_status] >= 1:
            status = "partial"
        else:
            status = "miss"
        targets.append(
            {
                "target_id": target_id,
                "status": status,
                "pass_statuses": [left_status, right_status],
                "matched_branch_ids": sorted(
                    set(left["matched_branch_ids"])
                    | set(right["matched_branch_ids"])
                )
                if status != "miss"
                else [],
            }
        )
    return {
        "case_id": first["case_id"],
        "variant": first["variant"],
        "targets": targets,
        "full_count": sum(item["status"] == "full" for item in targets),
        "partial_count": sum(item["status"] == "partial" for item in targets),
        "miss_count": sum(item["status"] == "miss" for item in targets),
        "reliability": {
            "status_agreement_count": agreement_count,
            "status_comparison_count": len(targets),
        },
    }


def deterministic_structure_metrics(
    scenario_set: Dict[str, Any], spec: Dict[str, Any]
) -> Dict[str, Any]:
    """Measure only contract-checkable properties without semantic grading."""

    _validate_common_scenario_set(scenario_set, spec)
    fact_ids = {item["id"] for item in spec["facts"]}
    branches = scenario_set["branches"]
    return {
        "branch_count": len(branches),
        "multi_actor_branch_rate": sum(
            len(set(item["actor_ids"])) >= 2 for item in branches
        )
        / len(branches),
        "trigger_present_rate": sum(
            bool(item["trigger_conditions"]) for item in branches
        )
        / len(branches),
        "invalidation_present_rate": sum(
            bool(item["invalidation_conditions"]) for item in branches
        )
        / len(branches),
        "valid_evidence_ref_rate": sum(
            bool(item["evidence_refs"])
            and set(item["evidence_refs"]) <= fact_ids
            for item in branches
        )
        / len(branches),
    }
