"""Extract simulation signals without pretending they are calibrated probabilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

from .benchmark import validate_forecast_submission
from .contracts import (
    _parse_datetime,
    _references_exist,
    _require,
    canonical_sha256,
    validate_result,
    validate_spec,
)
from .forecasting import simulation_context


PROMPT_VERSION = "probability-signal-extractor-v1"
UPDATE_DIRECTIONS = {"increase", "decrease", "unchanged", "ambiguous"}
UPDATE_STRENGTHS = {"weak", "moderate", "strong"}


def build_probability_signal_retry_prompt(
    system_prompt: str,
    user_prompt: str,
    validation_error: BaseException,
) -> tuple[str, str]:
    """Build one content-free repair prompt after structural validation fails.

    The rejected provider response is deliberately excluded.  The retry sees
    only the original anonymous input and a bounded validation message, so it
    cannot learn an outcome or be nudged toward a selected prior answer.
    """

    message = " ".join(str(validation_error).split())[:240]
    repair = (
        "\n\n上一次输出未通过结构契约。不要参考或复述上一次回答；"
        "请仅根据原始匿名输入重新生成完整 JSON。验证错误："
        f"{message}。必须让 updates 对每个 event target 恰好出现一次。"
    )
    return system_prompt + repair, user_prompt


def _event_targets(spec: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [
        item
        for item in spec["forecast_targets"]
        if item["kind"] == "event"
        and item["scoring"] in {"brier", "log_loss"}
    ]


def build_probability_signal_prompt(
    spec: Dict[str, Any],
    simulation_result: Dict[str, Any],
    baseline_submission: Dict[str, Any],
) -> tuple[str, str, set[str]]:
    """Build a qualitative signal prompt that cannot request a new probability."""

    validate_spec(spec)
    validate_result(simulation_result, spec)
    validate_forecast_submission(baseline_submission, spec)
    if baseline_submission["arm"] != "B1":
        raise ValueError("probability signal baseline must be a B1 submission")
    targets = _event_targets(spec)
    if not targets:
        raise ValueError("probability signal extraction requires event targets")
    baseline_by_target = {
        item["target_id"]: item
        for item in baseline_submission["target_predictions"]
    }
    simulation_nodes, allowed_refs = simulation_context(
        simulation_result,
        spec,
    )
    compact_input = {
        "case_id": spec["case_id"],
        "as_of": spec["as_of"],
        "facts": [
            {"id": item["id"], "statement": item["statement"]}
            for item in spec["facts"]
        ],
        "event_targets": [
            {
                "id": target["id"],
                "definition": target["definition"],
                "baseline_probability": baseline_by_target[target["id"]][
                    "value"
                ],
            }
            for target in targets
        ],
        "simulation_nodes": simulation_nodes,
    }
    system = """你是概率信号提取器，不是概率预测器。输入中的 B1 probability 是冻结事实
基线，模拟节点只是未校准假设。逐个事件目标判断模拟相对 B1 提供的更新方向和信号强度。

硬性要求：
- 禁止输出新的概率、概率区间、百分点变化或数值 delta；
- update_direction 只能是 increase/decrease/unchanged/ambiguous；
- update_strength 只能是 weak/moderate/strong，且不对应固定百分点；
- support_simulation_refs 和 counter_simulation_refs 只能引用输入节点；
- 必须同时考虑支持和反对该事件的信号；没有信号时使用空数组；
- 禁止把 Agent 数量、发帖数、情景 confidence 或 frequency 当作概率；
- 不得补充 as_of 之后的真实结果。

只输出 JSON：
{"updates":[{"target_id":"T1","update_direction":"ambiguous",
"update_strength":"weak","support_simulation_refs":[],
"counter_simulation_refs":[],"rationale":"..."}],"warnings":[]}。"""
    return (
        system,
        json.dumps(compact_input, ensure_ascii=False),
        allowed_refs,
    )


def _extract_payload(raw_response: Any) -> Dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("probability signal response is empty")
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
    raise ValueError(
        "probability signal response does not contain a JSON object"
    )


def build_uncalibrated_probability_updates(
    raw_response: Any,
    spec: Dict[str, Any],
    simulation_result: Dict[str, Any],
    baseline_submission: Dict[str, Any],
    *,
    model_id: str,
    valid_simulation_refs: Iterable[str],
) -> Dict[str, Any]:
    """Seal qualitative signals while withholding adjusted probabilities."""

    validate_spec(spec)
    validate_result(simulation_result, spec)
    validate_forecast_submission(baseline_submission, spec)
    if baseline_submission["arm"] != "B1":
        raise ValueError("probability signal baseline must be a B1 submission")
    payload = _extract_payload(raw_response)
    updates = payload.get("updates")
    warnings = payload.get("warnings", [])
    if not isinstance(updates, list):
        raise ValueError("probability signal updates must be a list")
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise ValueError("probability signal warnings must be a string list")
    targets = {item["id"]: item for item in _event_targets(spec)}
    baseline_by_target = {
        item["target_id"]: item
        for item in baseline_submission["target_predictions"]
    }
    allowed = set(valid_simulation_refs)
    expected_fields = {
        "target_id",
        "update_direction",
        "update_strength",
        "support_simulation_refs",
        "counter_simulation_refs",
        "rationale",
    }
    normalized = []
    normalization_warnings = []
    for item in updates:
        if not isinstance(item, dict):
            raise ValueError("probability signal update must be an object")
        extra = set(item) - expected_fields
        if extra:
            raise ValueError(
                "uncalibrated probability signal contains forbidden fields: "
                + ", ".join(sorted(extra))
            )
        target_id = item.get("target_id")
        if target_id not in targets:
            raise ValueError("probability signal contains an unknown target")
        support = item.get("support_simulation_refs")
        counter = item.get("counter_simulation_refs")
        _require(
            isinstance(support, list)
            and len(support) == len(set(support)),
            "support_simulation_refs must be a unique list",
        )
        _require(
            isinstance(counter, list)
            and len(counter) == len(set(counter)),
            "counter_simulation_refs must be a unique list",
        )
        overlap = set(support) & set(counter)
        if overlap:
            support = [ref for ref in support if ref not in overlap]
            counter = [ref for ref in counter if ref not in overlap]
            normalization_warnings.append(
                f"{target_id}: removed {len(overlap)} simulation ref(s) "
                "listed as both support and counter"
            )
        _references_exist(
            support + counter,
            allowed,
            "probability signal simulation refs",
        )
        _require(
            item.get("update_direction") in UPDATE_DIRECTIONS,
            "probability signal update_direction is invalid",
        )
        _require(
            item.get("update_strength") in UPDATE_STRENGTHS,
            "probability signal update_strength is invalid",
        )
        _require(
            isinstance(item.get("rationale"), str)
            and bool(item["rationale"].strip()),
            "probability signal rationale is required",
        )
        normalized.append(
            {
                **item,
                "support_simulation_refs": support,
                "counter_simulation_refs": counter,
                "baseline_probability": baseline_by_target[target_id][
                    "value"
                ],
                "adjusted_probability": None,
                "probability_semantics": "withheld_until_calibrated",
            }
        )
    if {item["target_id"] for item in normalized} != set(targets):
        raise ValueError(
            "probability signals must cover every event target exactly once"
        )
    if len(normalized) != len(targets):
        raise ValueError("probability signal target ids must be unique")
    artifact = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "simulation_id": simulation_result["runs"][0]["run_id"],
        "baseline_submission_sha256": canonical_sha256(
            baseline_submission
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "status": "uncalibrated",
        "calibration_version": None,
        "updates": normalized,
        "warnings": warnings
        + normalization_warnings
        + [
            "Adjusted probabilities are withheld until a calibration "
            "version is frozen on separate development data."
        ],
    }
    validate_uncalibrated_probability_updates(
        artifact,
        spec,
        simulation_result,
        baseline_submission,
        valid_simulation_refs=allowed,
    )
    return artifact


def validate_uncalibrated_probability_updates(
    artifact: Dict[str, Any],
    spec: Dict[str, Any],
    simulation_result: Dict[str, Any],
    baseline_submission: Dict[str, Any],
    *,
    valid_simulation_refs: Iterable[str],
) -> None:
    """Validate that an uncalibrated artifact cannot carry a new probability."""

    validate_spec(spec)
    validate_result(simulation_result, spec)
    validate_forecast_submission(baseline_submission, spec)
    _require(
        artifact.get("schema_version") == "0.1.0",
        "unsupported probability update schema_version",
    )
    _require(
        artifact.get("case_id") == spec["case_id"],
        "probability update case_id does not match spec",
    )
    _require(
        artifact.get("spec_sha256") == canonical_sha256(spec),
        "probability update spec_sha256 does not match spec",
    )
    _require(
        artifact.get("simulation_id")
        == simulation_result["runs"][0]["run_id"],
        "probability update simulation_id does not match result",
    )
    _require(
        artifact.get("baseline_submission_sha256")
        == canonical_sha256(baseline_submission),
        "probability update baseline hash does not match submission",
    )
    _parse_datetime(artifact.get("generated_at"), "generated_at")
    _require(
        artifact.get("status") == "uncalibrated"
        and artifact.get("calibration_version") is None,
        "uncalibrated probability update must not name calibration",
    )
    _require(
        artifact.get("prompt_version") == PROMPT_VERSION,
        "unsupported probability signal prompt_version",
    )
    updates = artifact.get("updates")
    _require(
        isinstance(updates, list) and updates,
        "probability updates must be non-empty",
    )
    event_targets = {item["id"] for item in _event_targets(spec)}
    _require(
        {item.get("target_id") for item in updates} == event_targets
        and len(updates) == len(event_targets),
        "probability updates must cover event targets exactly once",
    )
    allowed = set(valid_simulation_refs)
    baseline_by_target = {
        item["target_id"]: item["value"]
        for item in baseline_submission["target_predictions"]
    }
    for item in updates:
        target_id = item["target_id"]
        _require(
            item.get("baseline_probability")
            == baseline_by_target[target_id],
            "probability update baseline value does not match submission",
        )
        _require(
            item.get("adjusted_probability") is None
            and item.get("probability_semantics")
            == "withheld_until_calibrated",
            "uncalibrated probability update must withhold probability",
        )
        _require(
            item.get("update_direction") in UPDATE_DIRECTIONS,
            "probability signal update_direction is invalid",
        )
        _require(
            item.get("update_strength") in UPDATE_STRENGTHS,
            "probability signal update_strength is invalid",
        )
        refs = item.get("support_simulation_refs", []) + item.get(
            "counter_simulation_refs",
            [],
        )
        _require(
            not set(item.get("support_simulation_refs", []))
            & set(item.get("counter_simulation_refs", [])),
            "support and counter simulation refs must be disjoint",
        )
        _references_exist(
            refs,
            allowed,
            "probability signal simulation refs",
        )
