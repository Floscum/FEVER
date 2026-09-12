"""Deterministic development-only calibration for qualitative event signals."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .benchmark import validate_forecast_submission
from .contracts import (
    _require,
    canonical_sha256,
    validate_outcome,
    validate_result,
    validate_spec,
)
from .forecasting import simulation_context
from .probability_updates import (
    PROMPT_VERSION as SIGNAL_PROMPT_VERSION,
    validate_uncalibrated_probability_updates,
)
from .probability_ensemble import (
    AGGREGATION_VERSION,
    validate_probability_signal_ensemble,
)


CALIBRATED_ROUTER_VERSION = "calibrated-probability-router-v1"


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_probability_signal(
    update: Dict[str, Any],
    policy: Dict[str, Any],
) -> float:
    """Apply one frozen ordinal policy without using outcome information."""

    baseline = float(update["baseline_probability"])
    direction = update["update_direction"]
    strength = update["update_strength"]
    nested_deltas = policy.get("deltas")
    delta = float(
        nested_deltas[strength]
        if isinstance(nested_deltas, dict)
        else policy[f"{strength}_delta"]
    )
    if direction == "increase":
        adjusted = baseline + delta
    elif direction == "decrease":
        adjusted = baseline - delta
    else:
        adjusted = baseline
    return round(min(max(adjusted, 0.0), 1.0), 8)


def build_calibrated_forecast_submission(
    spec: Dict[str, Any],
    simulation_result: Dict[str, Any],
    baseline_submission: Dict[str, Any],
    probability_signals: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a frozen policy to qualitative signals without another LLM call."""

    validate_spec(spec)
    validate_result(simulation_result, spec)
    validate_forecast_submission(baseline_submission, spec)
    _require(
        baseline_submission.get("arm") == "B1",
        "calibrated submission baseline must be B1",
    )
    _require(
        policy.get("status") == "development_selected_requires_holdout",
        "calibration policy must be frozen before holdout use",
    )
    _require(
        probability_signals.get("prompt_version")
        == policy.get("signal_prompt_version"),
        "probability signal prompt version does not match policy",
    )
    _, valid_refs = simulation_context(simulation_result, spec)
    validate_uncalibrated_probability_updates(
        probability_signals,
        spec,
        simulation_result,
        baseline_submission,
        valid_simulation_refs=valid_refs,
    )
    updates = {
        item["target_id"]: item
        for item in probability_signals["updates"]
    }
    targets = {item["id"]: item for item in spec["forecast_targets"]}
    predictions = []
    for baseline_prediction in baseline_submission["target_predictions"]:
        prediction = deepcopy(baseline_prediction)
        target_id = prediction["target_id"]
        if targets[target_id]["kind"] == "market":
            predictions.append(prediction)
            continue
        update = updates[target_id]
        prediction["value"] = apply_probability_signal(update, policy)
        prediction["simulation_refs"] = list(
            dict.fromkeys(
                update["support_simulation_refs"]
                + update["counter_simulation_refs"]
            )
        )
        prediction["rationale"] = (
            f"{prediction['rationale']} "
            f"冻结定性信号为{update['update_direction']}/"
            f"{update['update_strength']}：{update['rationale']} "
            f"按策略{policy['policy_id']}确定性调整。"
        )
        predictions.append(prediction)

    submission = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "arm": "B3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": probability_signals["model_id"],
        "prompt_version": (
            f"{CALIBRATED_ROUTER_VERSION}:{policy['policy_id']}"
        ),
        "replication_ids": [simulation_result["runs"][0]["run_id"]],
        "target_predictions": predictions,
        "warnings": [
            "Market targets were copied exactly from the sealed B1 baseline.",
            "Event probabilities were produced deterministically from "
            "qualitative signals and a policy frozen before holdout use.",
            f"Calibration policy SHA-256: {canonical_sha256(policy)}",
        ],
    }
    validate_forecast_submission(submission, spec)
    used_refs = {
        ref
        for item in submission["target_predictions"]
        for ref in item["simulation_refs"]
    }
    _require(
        used_refs <= valid_refs,
        "calibrated submission contains unknown simulation refs",
    )
    return submission


def build_ensemble_calibrated_forecast_submission(
    spec: Dict[str, Any],
    baseline_submission: Dict[str, Any],
    probability_ensemble: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply one frozen delta to a three-seed qualitative consensus."""

    validate_spec(spec)
    validate_forecast_submission(baseline_submission, spec)
    _require(
        baseline_submission.get("arm") == "B1",
        "ensemble calibrated submission baseline must be B1",
    )
    _require(
        policy.get("status") == "development_selected_requires_holdout",
        "calibration policy must be frozen before holdout use",
    )
    _require(
        policy.get("signal_prompt_version") == SIGNAL_PROMPT_VERSION,
        "ensemble source signal prompt version does not match policy",
    )
    validate_probability_signal_ensemble(
        probability_ensemble,
        spec,
        baseline_submission,
    )
    updates = {
        item["target_id"]: item
        for item in probability_ensemble["target_updates"]
    }
    targets = {item["id"]: item for item in spec["forecast_targets"]}
    ensemble_sha256 = canonical_sha256(probability_ensemble)
    predictions = []
    for baseline_prediction in baseline_submission["target_predictions"]:
        prediction = deepcopy(baseline_prediction)
        target_id = prediction["target_id"]
        if targets[target_id]["kind"] == "market":
            predictions.append(prediction)
            continue
        aggregate = updates[target_id]
        update = {
            "baseline_probability": aggregate["baseline_probability"],
            "update_direction": aggregate["aggregate_direction"],
            "update_strength": aggregate["aggregate_strength"],
        }
        prediction["value"] = apply_probability_signal(update, policy)
        prediction["simulation_refs"] = []
        prediction["rationale"] = (
            f"{prediction['rationale']} 三随机种子定性聚合为"
            f"{aggregate['aggregate_direction']}/"
            f"{aggregate['aggregate_strength']}，方向计数"
            f"{aggregate['direction_counts']}；按策略"
            f"{policy['policy_id']}仅调整一次。逐种子引用保留在"
            f"聚合产物 {ensemble_sha256}，未跨图压平。"
        )
        predictions.append(prediction)

    replication_ids = [
        item["replication_id"]
        for item in probability_ensemble["replication_signals"]
    ]
    submission = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "arm": "B3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": f"deterministic:{AGGREGATION_VERSION}",
        "prompt_version": (
            f"{CALIBRATED_ROUTER_VERSION}:{policy['policy_id']}:"
            f"{AGGREGATION_VERSION}"
        ),
        "replication_ids": replication_ids,
        "target_predictions": predictions,
        "warnings": [
            "Market targets were copied exactly from the sealed B1 baseline.",
            "Event probabilities were adjusted once after three-seed "
            "qualitative aggregation; seed counts are not probabilities.",
            "Per-seed simulation refs remain namespaced in the ensemble "
            "artifact and are intentionally absent from flattened forecast refs.",
            f"Probability ensemble SHA-256: {ensemble_sha256}",
            f"Calibration policy SHA-256: {canonical_sha256(policy)}",
        ],
    }
    validate_forecast_submission(submission, spec)
    return submission


def evaluate_probability_holdout(
    manifest: Dict[str, Any],
    policy: Dict[str, Any],
    *,
    signal_root: Path,
    submission_root: Path,
    result_root: Path,
    generation_records: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Audit a new holdout without fitting or selecting another policy."""

    budget = manifest["pilot_budget"]
    _require(
        canonical_sha256(policy) == budget["calibration_policy_sha256"],
        "holdout policy hash does not match frozen benchmark",
    )
    _require(
        policy["source_dataset"]["dataset_id"]
        == "probability-calibration-dev-v1",
        "holdout policy must come from the frozen development dataset",
    )
    _require(
        datetime.fromisoformat(policy["frozen_at"])
        <= datetime.fromisoformat(manifest["frozen_at"]),
        "calibration policy must be frozen before benchmark",
    )

    rows = []
    case_reviews = []
    market_copy_valid = True
    deterministic_policy_valid = True
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        spec = _validate_frozen_hash(
            Path(entry["spec_path"]),
            entry["spec_sha256"],
            f"{case_id} spec",
        )
        outcome = _validate_frozen_hash(
            Path(entry["outcome_path"]),
            entry["outcome_sha256"],
            f"{case_id} outcome",
        )
        validate_spec(spec)
        validate_outcome(outcome, spec)
        result = _read(result_root / case_id / "simulation-result.json")
        validate_result(result, spec)
        baseline = _read(submission_root / case_id / "B1.json")
        calibrated = _read(submission_root / case_id / "B3.json")
        validate_forecast_submission(baseline, spec)
        validate_forecast_submission(calibrated, spec)
        signals = _read(signal_root / f"{case_id}.json")
        _, valid_refs = simulation_context(result, spec)
        validate_uncalibrated_probability_updates(
            signals,
            spec,
            result,
            baseline,
            valid_simulation_refs=valid_refs,
        )
        targets = {item["id"]: item for item in spec["forecast_targets"]}
        observed = {
            item["target_id"]: item["observed"]
            for item in outcome["target_results"]
        }
        baseline_by_id = {
            item["target_id"]: item
            for item in baseline["target_predictions"]
        }
        calibrated_by_id = {
            item["target_id"]: item
            for item in calibrated["target_predictions"]
        }
        updates = {
            item["target_id"]: item for item in signals["updates"]
        }
        for target_id, target in targets.items():
            if target["kind"] == "market":
                if calibrated_by_id[target_id] != baseline_by_id[target_id]:
                    market_copy_valid = False
                continue
            update = updates[target_id]
            expected = apply_probability_signal(update, policy)
            actual = float(calibrated_by_id[target_id]["value"])
            if actual != expected:
                deterministic_policy_valid = False
            value = observed[target_id].get("value")
            _require(
                isinstance(value, bool),
                f"{case_id} {target_id} must have a binary outcome",
            )
            baseline_probability = float(
                baseline_by_id[target_id]["value"]
            )
            rows.append(
                {
                    "case_id": case_id,
                    "target_id": target_id,
                    "observed": value,
                    "baseline_probability": baseline_probability,
                    "calibrated_probability": actual,
                    "update_direction": update["update_direction"],
                    "update_strength": update["update_strength"],
                    "baseline_brier": round(
                        _brier(baseline_probability, value),
                        8,
                    ),
                    "calibrated_brier": round(
                        _brier(actual, value),
                        8,
                    ),
                }
            )

    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        case_rows = [row for row in rows if row["case_id"] == case_id]
        baseline_mean = sum(
            row["baseline_brier"] for row in case_rows
        ) / len(case_rows)
        calibrated_mean = sum(
            row["calibrated_brier"] for row in case_rows
        ) / len(case_rows)
        case_reviews.append(
            {
                "case_id": case_id,
                "baseline_mean_brier": round(baseline_mean, 8),
                "calibrated_mean_brier": round(calibrated_mean, 8),
                "delta": round(calibrated_mean - baseline_mean, 8),
            }
        )

    baseline_mean = sum(row["baseline_brier"] for row in rows) / len(rows)
    calibrated_mean = sum(
        row["calibrated_brier"] for row in rows
    ) / len(rows)
    mean_delta = calibrated_mean - baseline_mean
    maximum_case_degradation = max(
        item["delta"] for item in case_reviews
    )
    nonworse_fraction = sum(
        item["delta"] <= 0 for item in case_reviews
    ) / len(case_reviews)
    actionable = [
        row
        for row in rows
        if row["update_direction"] in {"increase", "decrease"}
    ]
    direction_correct = sum(
        (row["observed"] and row["update_direction"] == "increase")
        or (
            not row["observed"]
            and row["update_direction"] == "decrease"
        )
        for row in actionable
    )
    direction_accuracy = (
        direction_correct / len(actionable) if actionable else 0.0
    )
    records = generation_records or []
    invalid_count = sum(
        item.get("status") in {"invalid", "provider_failed"}
        for item in records
    )
    sealed_count = sum(item.get("status") == "sealed" for item in records)
    invalid_rate = (
        invalid_count / (invalid_count + sealed_count)
        if invalid_count + sealed_count
        else 0.0
    )
    positive_count = sum(row["observed"] for row in rows)
    negative_count = len(rows) - positive_count
    gates = policy["holdout_gates"]

    def gate(actual: Any, threshold: Dict[str, Any], passed: bool):
        return {
            "status": "passed" if passed else "failed",
            "actual": actual,
            "threshold": threshold,
        }

    gate_results = {
        "new_case_count": gate(
            len(case_reviews),
            {"minimum": gates["minimum_new_case_count"]},
            len(case_reviews) >= gates["minimum_new_case_count"],
        ),
        "positive_event_outcomes": gate(
            positive_count,
            {"minimum": gates["required_positive_event_outcomes"]},
            positive_count >= gates["required_positive_event_outcomes"],
        ),
        "negative_event_outcomes": gate(
            negative_count,
            {"minimum": gates["required_negative_event_outcomes"]},
            negative_count >= gates["required_negative_event_outcomes"],
        ),
        "mean_brier_delta_vs_b1": gate(
            round(mean_delta, 8),
            {"maximum": gates["maximum_mean_brier_delta_vs_b1"]},
            mean_delta <= gates["maximum_mean_brier_delta_vs_b1"],
        ),
        "maximum_case_mean_brier_degradation": gate(
            round(maximum_case_degradation, 8),
            {"maximum": gates["maximum_case_mean_brier_degradation"]},
            maximum_case_degradation
            <= gates["maximum_case_mean_brier_degradation"],
        ),
        "nonworse_case_fraction": gate(
            round(nonworse_fraction, 8),
            {"minimum": gates["minimum_nonworse_case_fraction"]},
            nonworse_fraction >= gates["minimum_nonworse_case_fraction"],
        ),
        "actionable_direction_accuracy": gate(
            round(direction_accuracy, 8),
            {"minimum": gates["minimum_actionable_direction_accuracy"]},
            direction_accuracy
            >= gates["minimum_actionable_direction_accuracy"],
        ),
        "invalid_output_rate": gate(
            round(invalid_rate, 8),
            {"maximum": gates["maximum_invalid_output_rate"]},
            invalid_rate <= gates["maximum_invalid_output_rate"],
        ),
        "market_targets_copied_from_b1": gate(
            market_copy_valid,
            {"required": True},
            market_copy_valid,
        ),
        "deterministic_policy_application": gate(
            deterministic_policy_valid,
            {"required": True},
            deterministic_policy_valid,
        ),
    }
    performance_gate_names = {
        "mean_brier_delta_vs_b1",
        "maximum_case_mean_brier_degradation",
        "nonworse_case_fraction",
        "actionable_direction_accuracy",
        "market_targets_copied_from_b1",
        "deterministic_policy_application",
    }
    return {
        "schema_version": "0.1.0",
        "benchmark_id": manifest["benchmark_id"],
        "policy_id": policy["policy_id"],
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_reviews),
        "target_count": len(rows),
        "positive_outcome_count": positive_count,
        "negative_outcome_count": negative_count,
        "baseline_mean_brier": round(baseline_mean, 8),
        "calibrated_mean_brier": round(calibrated_mean, 8),
        "mean_brier_delta_vs_b1": round(mean_delta, 8),
        "actionable_direction_count": len(actionable),
        "actionable_direction_correct_count": direction_correct,
        "actionable_direction_accuracy": round(direction_accuracy, 8),
        "performance_gates_passed": all(
            gate_results[name]["status"] == "passed"
            for name in performance_gate_names
        ),
        "all_holdout_gates_passed": all(
            item["status"] == "passed"
            for item in gate_results.values()
        ),
        "eligible_for_formal_experiment": all(
            item["status"] == "passed"
            for item in gate_results.values()
        ),
        "gate_results": gate_results,
        "case_reviews": case_reviews,
        "target_reviews": rows,
        "generation_review": {
            "sealed_count": sealed_count,
            "invalid_or_provider_failed_count": invalid_count,
            "invalid_output_rate": round(invalid_rate, 8),
        },
        "warnings": [
            "Performance gates do not override operational reliability, "
            "budget, or ontology-quality failures.",
            "No policy parameter was selected or changed on this holdout.",
        ],
    }


def _brier(probability: float, observed: bool) -> float:
    return (probability - float(observed)) ** 2


def _validate_frozen_hash(path: Path, expected: str, label: str) -> Dict[str, Any]:
    value = _read(path)
    _require(
        canonical_sha256(value) == expected,
        f"{label} hash does not match frozen calibration dataset",
    )
    return value


def evaluate_probability_calibration(
    manifest: Dict[str, Any],
    *,
    signal_root: Path,
    usage_records: list[Dict[str, Any]] | None = None,
    generation_records: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Evaluate preregistered policies on development cases only."""

    _require(
        manifest.get("role") == "development_only_not_promotion_evidence",
        "calibration dataset must be development-only",
    )
    policies = manifest["candidate_policies"]
    policy_ids = [item["id"] for item in policies]
    _require(
        len(policy_ids) == len(set(policy_ids)),
        "candidate calibration policy ids must be unique",
    )
    rows = []
    case_target_rows: Dict[str, list[Dict[str, Any]]] = {}
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        spec = _validate_frozen_hash(
            Path(entry["spec_path"]),
            entry["spec_sha256"],
            f"{case_id} spec",
        )
        outcome = _validate_frozen_hash(
            Path(entry["outcome_path"]),
            entry["outcome_sha256"],
            f"{case_id} outcome",
        )
        baseline = _validate_frozen_hash(
            Path(entry["baseline_path"]),
            entry["baseline_sha256"],
            f"{case_id} baseline",
        )
        result = _validate_frozen_hash(
            Path(entry["simulation_result_path"]),
            entry["simulation_sha256"],
            f"{case_id} simulation",
        )
        validate_spec(spec)
        validate_outcome(outcome, spec)
        validate_forecast_submission(baseline, spec)
        validate_result(result, spec)
        _, valid_refs = simulation_context(result, spec)
        signals = _read(signal_root / f"{case_id}.json")
        validate_uncalibrated_probability_updates(
            signals,
            spec,
            result,
            baseline,
            valid_simulation_refs=valid_refs,
        )
        signal_target_ids = {
            item["target_id"] for item in signals["updates"]
        }
        outcomes = {
            item["target_id"]: item["observed"]["value"]
            for item in outcome["target_results"]
            if item["target_id"] in signal_target_ids
        }
        case_rows = []
        for update in signals["updates"]:
            target_id = update["target_id"]
            observed = outcomes[target_id]
            _require(
                isinstance(observed, bool),
                f"{case_id} {target_id} must have a binary outcome",
            )
            row = {
                "case_id": case_id,
                "target_id": target_id,
                "observed": observed,
                "baseline_probability": update["baseline_probability"],
                "update_direction": update["update_direction"],
                "update_strength": update["update_strength"],
                "support_ref_count": len(
                    update["support_simulation_refs"]
                ),
                "counter_ref_count": len(
                    update["counter_simulation_refs"]
                ),
                "baseline_brier": round(
                    _brier(update["baseline_probability"], observed),
                    8,
                ),
                "policy_probabilities": {},
            }
            for policy in policies:
                row["policy_probabilities"][policy["id"]] = (
                    apply_probability_signal(update, policy)
                )
            rows.append(row)
            case_rows.append(row)
        case_target_rows[case_id] = case_rows

    policy_reviews = []
    selection = manifest["selection_rule"]
    for policy in policies:
        policy_id = policy["id"]
        adjusted_scores = [
            _brier(row["policy_probabilities"][policy_id], row["observed"])
            for row in rows
        ]
        baseline_scores = [row["baseline_brier"] for row in rows]
        case_reviews = []
        for case_id, case_rows in case_target_rows.items():
            baseline_case = sum(
                item["baseline_brier"] for item in case_rows
            ) / len(case_rows)
            adjusted_case = sum(
                _brier(
                    item["policy_probabilities"][policy_id],
                    item["observed"],
                )
                for item in case_rows
            ) / len(case_rows)
            case_reviews.append(
                {
                    "case_id": case_id,
                    "baseline_mean_brier": round(baseline_case, 8),
                    "adjusted_mean_brier": round(adjusted_case, 8),
                    "delta": round(adjusted_case - baseline_case, 8),
                }
            )
        mean_baseline = sum(baseline_scores) / len(baseline_scores)
        mean_adjusted = sum(adjusted_scores) / len(adjusted_scores)
        mean_delta = mean_adjusted - mean_baseline
        maximum_case_degradation = max(
            item["delta"] for item in case_reviews
        )
        nonworse_fraction = sum(
            item["delta"] <= 0 for item in case_reviews
        ) / len(case_reviews)
        total_absolute_delta = sum(
            abs(row["policy_probabilities"][policy_id]
                - row["baseline_probability"])
            for row in rows
        )
        qualifies = bool(
            mean_delta <= selection["maximum_mean_brier_delta"]
            and maximum_case_degradation
            <= selection["maximum_case_mean_brier_degradation"]
            and nonworse_fraction
            >= selection["minimum_nonworse_case_fraction"]
        )
        policy_reviews.append(
            {
                "policy_id": policy_id,
                "baseline_mean_brier": round(mean_baseline, 8),
                "adjusted_mean_brier": round(mean_adjusted, 8),
                "mean_brier_delta_vs_b1": round(mean_delta, 8),
                "maximum_case_mean_brier_degradation": round(
                    maximum_case_degradation,
                    8,
                ),
                "nonworse_case_fraction": round(nonworse_fraction, 8),
                "total_absolute_delta": round(total_absolute_delta, 8),
                "qualifies_on_development_constraints": qualifies,
                "case_reviews": case_reviews,
            }
        )

    qualifying = [
        item
        for item in policy_reviews
        if item["qualifies_on_development_constraints"]
    ]
    if qualifying:
        chosen = min(
            qualifying,
            key=lambda item: (
                item["mean_brier_delta_vs_b1"],
                item["total_absolute_delta"],
            ),
        )
    else:
        chosen = next(
            item
            for item in policy_reviews
            if item["policy_id"] == selection["fallback_policy"]
        )

    actionable = [
        row
        for row in rows
        if row["update_direction"] in {"increase", "decrease"}
    ]
    correct = sum(
        (row["observed"] and row["update_direction"] == "increase")
        or (not row["observed"] and row["update_direction"] == "decrease")
        for row in actionable
    )
    usage = usage_records or []
    generation = generation_records or []
    completed_usage = [
        item for item in usage if item.get("status") == "completed"
    ]
    return {
        "schema_version": "0.1.0",
        "dataset_id": manifest["dataset_id"],
        "role": manifest["role"],
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_target_rows),
        "target_count": len(rows),
        "positive_outcome_count": sum(row["observed"] for row in rows),
        "negative_outcome_count": sum(
            not row["observed"] for row in rows
        ),
        "signal_diagnostics": {
            "actionable_direction_count": len(actionable),
            "unchanged_count": sum(
                row["update_direction"] == "unchanged" for row in rows
            ),
            "ambiguous_count": sum(
                row["update_direction"] == "ambiguous" for row in rows
            ),
            "actionable_direction_accuracy": round(
                correct / len(actionable) if actionable else 0.0,
                8,
            ),
        },
        "policy_reviews": policy_reviews,
        "selected_policy_id": chosen["policy_id"],
        "selected_policy_is_development_only": True,
        "eligible_as_promotion_evidence": False,
        "target_diagnostics": rows,
        "usage_review": {
            "attempts": len(usage),
            "completed": len(completed_usage),
            "failed": sum(
                item.get("status") == "failed" for item in usage
            ),
            "total_tokens": sum(
                item.get("total_tokens", 0) or 0 for item in usage
            ),
            "estimated_cost": round(
                sum(
                    item.get("estimated_cost", 0) or 0
                    for item in usage
                ),
                8,
            ),
            "completed_call_budget": manifest["extraction_budget"][
                "maximum_calls"
            ],
            "completed_call_budget_exceeded": (
                len(completed_usage)
                > manifest["extraction_budget"]["maximum_calls"]
            ),
            "token_budget": manifest["extraction_budget"][
                "maximum_total_tokens"
            ],
            "token_budget_exceeded": (
                sum(item.get("total_tokens", 0) or 0 for item in usage)
                > manifest["extraction_budget"]["maximum_total_tokens"]
            ),
        },
        "generation_review": {
            "record_count": len(generation),
            "sealed_count": sum(
                item.get("status") == "sealed" for item in generation
            ),
            "invalid_count": sum(
                item.get("status") == "invalid" for item in generation
            ),
            "provider_failed_count": sum(
                item.get("status") == "provider_failed"
                for item in generation
            ),
        },
        "warnings": [
            "This dataset is development-only and cannot support promotion.",
            "Outcome prevalence is reported because a tiny imbalanced "
            "development set can make probability updates look misleadingly "
            "good.",
        ],
    }
