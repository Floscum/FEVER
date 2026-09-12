"""Blind replay validation and scoring for B0/B1/B2/B3 submissions."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional

from .contracts import (
    ContractError,
    _parse_datetime,
    _require,
    canonical_sha256,
    validate_outcome,
    validate_spec,
)


PREDICTION_KIND_BY_SCORING = {
    "brier": "binary_probability",
    "log_loss": "binary_probability",
    "direction_accuracy": "categorical",
    "interval_coverage": "interval",
    "multi_label_f1": "multi_label",
}


def evaluate_promotion_gates(
    *,
    promotion_gates: Dict[str, Any],
    aggregate_by_arm: Dict[str, Dict[str, float]],
    submission_count: int,
    expected_submission_count: int,
    severe_future_leakage_count: int,
    invalid_output_count: int = 0,
    valid_output_count: Optional[int] = None,
    pilot_review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate every preregistered gate without treating missing evidence as pass."""

    completion_rate = (
        submission_count / expected_submission_count
        if expected_submission_count
        else 0.0
    )
    audited_valid_outputs = (
        submission_count if valid_output_count is None else valid_output_count
    )
    total_seen = audited_valid_outputs + invalid_output_count
    invalid_output_rate = (
        invalid_output_count / total_seen if total_seen else 0.0
    )
    review = pilot_review or {}

    def result(
        *,
        actual: Any,
        threshold: Any,
        passed: Optional[bool],
    ) -> Dict[str, Any]:
        return {
            "status": (
                "unassessed"
                if passed is None
                else ("passed" if passed else "failed")
            ),
            "actual": actual,
            "threshold": threshold,
        }

    required_completeness = promotion_gates[
        "required_submission_completeness"
    ]
    maximum_invalid = promotion_gates["maximum_invalid_output_rate"]
    maximum_leakage = promotion_gates[
        "maximum_severe_future_leakage_count"
    ]
    required_usage = promotion_gates["required_local_usage_coverage"]
    minimum_direction_delta = promotion_gates[
        "minimum_b3_minus_b1_direction_accuracy"
    ]
    maximum_brier_delta = promotion_gates[
        "maximum_b3_minus_b1_mean_brier"
    ]
    minimum_scenario_cases = promotion_gates[
        "minimum_cases_with_incremental_testable_scenario"
    ]
    required_scheduler_seed_coverage = promotion_gates.get(
        "required_scheduler_seed_coverage"
    )

    complete = completion_rate >= required_completeness
    b1 = aggregate_by_arm.get("B1", {})
    b3 = aggregate_by_arm.get("B3", {})
    direction_delta = None
    brier_delta = None
    if complete and "direction_accuracy" in b1 and "direction_accuracy" in b3:
        direction_delta = round(
            b3["direction_accuracy"] - b1["direction_accuracy"],
            8,
        )
    if complete and "brier" in b1 and "brier" in b3:
        brier_delta = round(b3["brier"] - b1["brier"], 8)

    local_usage_coverage = review.get("local_usage_coverage")
    scenario_case_ids = review.get(
        "cases_with_incremental_testable_scenario"
    )
    if scenario_case_ids is not None:
        _require(
            isinstance(scenario_case_ids, list)
            and all(isinstance(item, str) for item in scenario_case_ids)
            and len(scenario_case_ids) == len(set(scenario_case_ids)),
            "pilot review scenario case ids must be unique strings",
        )
    scenario_count = (
        len(scenario_case_ids) if scenario_case_ids is not None else None
    )
    scheduler_seed_coverage = review.get("scheduler_seed_coverage")

    gate_results = {
        "submission_completeness": result(
            actual=round(completion_rate, 8),
            threshold={"minimum": required_completeness},
            passed=complete,
        ),
        "invalid_output_rate": result(
            actual=round(invalid_output_rate, 8),
            threshold={"maximum": maximum_invalid},
            passed=invalid_output_rate <= maximum_invalid,
        ),
        "severe_future_leakage_count": result(
            actual=severe_future_leakage_count,
            threshold={"maximum": maximum_leakage},
            passed=severe_future_leakage_count <= maximum_leakage,
        ),
        "local_usage_coverage": result(
            actual=local_usage_coverage,
            threshold={"minimum": required_usage},
            passed=(
                None
                if local_usage_coverage is None
                else local_usage_coverage >= required_usage
            ),
        ),
        "b3_minus_b1_direction_accuracy": result(
            actual=direction_delta,
            threshold={"minimum": minimum_direction_delta},
            passed=(
                None
                if direction_delta is None
                else direction_delta >= minimum_direction_delta
            ),
        ),
        "b3_minus_b1_mean_brier": result(
            actual=brier_delta,
            threshold={"maximum": maximum_brier_delta},
            passed=(
                None
                if brier_delta is None
                else brier_delta <= maximum_brier_delta
            ),
        ),
        "cases_with_incremental_testable_scenario": result(
            actual=scenario_count,
            threshold={"minimum": minimum_scenario_cases},
            passed=(
                None
                if scenario_count is None
                else scenario_count >= minimum_scenario_cases
            ),
        ),
    }
    if required_scheduler_seed_coverage is not None:
        gate_results["scheduler_seed_coverage"] = result(
            actual=scheduler_seed_coverage,
            threshold={"minimum": required_scheduler_seed_coverage},
            passed=(
                None
                if scheduler_seed_coverage is None
                else scheduler_seed_coverage
                >= required_scheduler_seed_coverage
            ),
        )
    performance_gate_names = {
        "b3_minus_b1_direction_accuracy",
        "b3_minus_b1_mean_brier",
    }
    performance_gate_mode = promotion_gates.get(
        "performance_gate_mode",
        "required",
    )
    _require(
        performance_gate_mode in {"required", "exploratory"},
        "performance_gate_mode must be required or exploratory",
    )
    engineering_passed = all(
        item["status"] == "passed"
        for name, item in gate_results.items()
        if name not in performance_gate_names
    )
    all_passed = all(
        item["status"] == "passed" for item in gate_results.values()
    )
    return {
        "performance_gate_mode": performance_gate_mode,
        "all_submissions_complete": complete,
        "eligible_for_promotion_review": complete
        and invalid_output_rate <= maximum_invalid
        and severe_future_leakage_count <= maximum_leakage,
        "eligible_for_next_stage": engineering_passed,
        "eligible_for_formal_experiment": (
            all_passed if performance_gate_mode == "required" else False
        ),
        "gate_results": gate_results,
    }


def validate_forecast_submission(
    submission: Dict[str, Any],
    spec: Dict[str, Any],
) -> None:
    """Enforce complete, scoring-compatible predictions for one frozen spec."""

    validate_spec(spec)
    _require(isinstance(submission, dict), "forecast submission must be an object")
    _require(
        submission.get("schema_version") == "0.1.0",
        "unsupported forecast submission schema_version",
    )
    _require(
        submission.get("case_id") == spec["case_id"],
        "forecast submission case_id does not match spec",
    )
    _require(
        submission.get("spec_sha256") == canonical_sha256(spec),
        "forecast submission spec_sha256 does not match spec",
    )
    _require(
        submission.get("arm") in {"B0", "B1", "B2", "B3"},
        "forecast submission arm is invalid",
    )
    _parse_datetime(submission.get("generated_at"), "generated_at")
    _require(
        isinstance(submission.get("model_id"), str) and submission["model_id"],
        "forecast submission model_id is required",
    )
    _require(
        isinstance(submission.get("prompt_version"), str)
        and submission["prompt_version"],
        "forecast submission prompt_version is required",
    )
    predictions = submission.get("target_predictions")
    _require(
        isinstance(predictions, list) and predictions,
        "target_predictions must be non-empty",
    )
    prediction_ids = [item.get("target_id") for item in predictions]
    _require(
        len(prediction_ids) == len(set(prediction_ids)),
        "target_predictions target_id values must be unique",
    )
    targets = {item["id"]: item for item in spec["forecast_targets"]}
    _require(
        set(prediction_ids) == set(targets),
        "forecast submission must predict every target exactly once",
    )
    fact_ids = {item["id"] for item in spec["facts"]}
    for prediction in predictions:
        target = targets[prediction["target_id"]]
        expected_kind = PREDICTION_KIND_BY_SCORING[target["scoring"]]
        _require(
            prediction.get("prediction_kind") == expected_kind,
            f"{target['id']} requires prediction_kind {expected_kind}",
        )
        value = prediction.get("value")
        if expected_kind == "binary_probability":
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0 <= value <= 1,
                f"{target['id']} probability must be between 0 and 1",
            )
        elif expected_kind == "categorical":
            _require(
                isinstance(value, str)
                and value.strip().lower() in {"up", "down", "flat"},
                f"{target['id']} category must be up, down, or flat",
            )
        elif expected_kind == "interval":
            _require(
                isinstance(value, dict)
                and isinstance(value.get("lower"), (int, float))
                and isinstance(value.get("upper"), (int, float))
                and value["lower"] <= value["upper"],
                f"{target['id']} interval must have lower <= upper",
            )
        elif expected_kind == "multi_label":
            _require(
                isinstance(value, list)
                and all(isinstance(item, str) for item in value)
                and len(value) == len(set(value)),
                f"{target['id']} labels must be unique strings",
            )
        evidence_refs = prediction.get("evidence_refs")
        _require(
            isinstance(evidence_refs, list)
            and set(evidence_refs) <= fact_ids,
            f"{target['id']} contains unknown evidence_refs",
        )
        simulation_refs = prediction.get("simulation_refs")
        _require(
            isinstance(simulation_refs, list)
            and all(isinstance(item, str) for item in simulation_refs),
            f"{target['id']} simulation_refs must be a string list",
        )
        if submission["arm"] in {"B0", "B1"}:
            _require(
                not simulation_refs,
                f"{submission['arm']} may not cite simulation outputs",
            )


def _observed_binary(observed: Dict[str, Any], target_id: str) -> float:
    value = observed.get("value")
    if not isinstance(value, bool):
        raise ContractError(f"{target_id} outcome must contain boolean value")
    return float(value)


def _score_target(
    target: Dict[str, Any],
    prediction: Dict[str, Any],
    observed: Dict[str, Any],
) -> Dict[str, Any]:
    scoring = target["scoring"]
    predicted = prediction["value"]
    if scoring == "brier":
        actual = _observed_binary(observed, target["id"])
        value = (float(predicted) - actual) ** 2
        direction = "lower_is_better"
    elif scoring == "log_loss":
        actual = _observed_binary(observed, target["id"])
        probability = min(max(float(predicted), 1e-15), 1 - 1e-15)
        value = -(
            actual * math.log(probability)
            + (1 - actual) * math.log(1 - probability)
        )
        direction = "lower_is_better"
    elif scoring == "direction_accuracy":
        actual = observed.get("category")
        if not isinstance(actual, str):
            raise ContractError(
                f"{target['id']} outcome must contain category"
            )
        value = float(predicted.strip().lower() == actual.strip().lower())
        direction = "higher_is_better"
    elif scoring == "interval_coverage":
        actual = observed.get("value")
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise ContractError(
                f"{target['id']} outcome must contain numeric value"
            )
        value = float(predicted["lower"] <= actual <= predicted["upper"])
        direction = "higher_is_better"
    elif scoring == "multi_label_f1":
        actual = observed.get("labels")
        if not isinstance(actual, list):
            raise ContractError(f"{target['id']} outcome must contain labels")
        predicted_set = set(predicted)
        actual_set = set(actual)
        if not predicted_set and not actual_set:
            value = 1.0
        elif not predicted_set or not actual_set:
            value = 0.0
        else:
            precision = len(predicted_set & actual_set) / len(predicted_set)
            recall = len(predicted_set & actual_set) / len(actual_set)
            value = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        direction = "higher_is_better"
    else:
        raise ContractError(f"unsupported scoring method: {scoring}")
    return {
        "target_id": target["id"],
        "scoring": scoring,
        "score": round(value, 8),
        "direction": direction,
    }


def score_forecast_submission(
    submission: Dict[str, Any],
    spec: Dict[str, Any],
    outcome: Dict[str, Any],
) -> Dict[str, Any]:
    """Score one sealed submission against its matching hidden outcome."""

    validate_forecast_submission(submission, spec)
    validate_outcome(outcome, spec)
    predictions = {
        item["target_id"]: item for item in submission["target_predictions"]
    }
    observations = {
        item["target_id"]: item["observed"]
        for item in outcome["target_results"]
    }
    target_scores = [
        _score_target(target, predictions[target["id"]], observations[target["id"]])
        for target in spec["forecast_targets"]
    ]
    metric_values: Dict[str, list[float]] = {}
    for item in target_scores:
        metric_values.setdefault(item["scoring"], []).append(item["score"])
    aggregate = {
        metric: round(sum(values) / len(values), 8)
        for metric, values in sorted(metric_values.items())
    }
    return {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "arm": submission["arm"],
        "submission_sha256": canonical_sha256(submission),
        "scored_at": datetime.now().astimezone().isoformat(),
        "target_scores": target_scores,
        "aggregate_by_metric": aggregate,
    }
