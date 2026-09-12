"""Descriptive paired effect estimates for B1/B3 blind replay scores."""

from __future__ import annotations

import random
from statistics import median
from typing import Any, Dict, Iterable


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def analyze_paired_brier_effects(
    scored_submissions: Iterable[Dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    random_seed: int = 20260731,
    open_source_decision_minimum_cases: int = 12,
    research_minimum_cases: int = 30,
) -> Dict[str, Any]:
    """Compare B3 with B1 while resampling independent cases, not targets."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if open_source_decision_minimum_cases < 1:
        raise ValueError("open_source_decision_minimum_cases must be positive")
    if research_minimum_cases < open_source_decision_minimum_cases:
        raise ValueError(
            "research_minimum_cases must be at least the open-source minimum"
        )
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for submission in scored_submissions:
        case_id = submission.get("case_id")
        arm = submission.get("arm")
        if arm not in {"B1", "B3"}:
            continue
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("scored submission case_id is required")
        case_arms = by_case.setdefault(case_id, {})
        if arm in case_arms:
            raise ValueError(f"duplicate {arm} score for case {case_id}")
        case_arms[arm] = submission
    if not by_case:
        raise ValueError("no B1/B3 scored submissions were provided")

    case_effects = []
    for case_id in sorted(by_case):
        arms = by_case[case_id]
        if set(arms) != {"B1", "B3"}:
            raise ValueError(f"case {case_id} requires paired B1 and B3 scores")
        arm_targets = {}
        for arm in ("B1", "B3"):
            target_scores = arms[arm].get("target_scores")
            if not isinstance(target_scores, list):
                raise ValueError(f"case {case_id} {arm} target_scores is invalid")
            arm_targets[arm] = {
                item.get("target_id"): float(item["score"])
                for item in target_scores
                if item.get("scoring") == "brier"
            }
        if not arm_targets["B1"]:
            raise ValueError(f"case {case_id} has no Brier-scored targets")
        if set(arm_targets["B1"]) != set(arm_targets["B3"]):
            raise ValueError(f"case {case_id} B1/B3 Brier targets do not match")
        target_deltas = [
            {
                "target_id": target_id,
                "b1_brier": round(arm_targets["B1"][target_id], 8),
                "b3_brier": round(arm_targets["B3"][target_id], 8),
                "b3_minus_b1": round(
                    arm_targets["B3"][target_id]
                    - arm_targets["B1"][target_id],
                    8,
                ),
            }
            for target_id in sorted(arm_targets["B1"])
        ]
        case_delta = sum(item["b3_minus_b1"] for item in target_deltas) / len(
            target_deltas
        )
        case_effects.append(
            {
                "case_id": case_id,
                "target_count": len(target_deltas),
                "mean_b3_minus_b1": round(case_delta, 8),
                "target_effects": target_deltas,
            }
        )

    case_deltas = [item["mean_b3_minus_b1"] for item in case_effects]
    mean_delta = sum(case_deltas) / len(case_deltas)
    generator = random.Random(random_seed)
    bootstrap_means = []
    for _ in range(bootstrap_samples):
        sampled = [generator.choice(case_deltas) for _ in case_deltas]
        bootstrap_means.append(sum(sampled) / len(sampled))

    improved = sum(delta < 0 for delta in case_deltas)
    unchanged = sum(delta == 0 for delta in case_deltas)
    target_deltas = [
        target["b3_minus_b1"]
        for case in case_effects
        for target in case["target_effects"]
    ]
    changed_target_deltas = [delta for delta in target_deltas if delta != 0]
    improved_changed_targets = sum(
        delta < 0 for delta in changed_target_deltas
    )
    worsened_changed_targets = sum(
        delta > 0 for delta in changed_target_deltas
    )
    return {
        "schema_version": "0.1.0",
        "metric": "brier_score",
        "effect": "B3_minus_B1",
        "effect_direction": "negative_favors_B3",
        "sampling_unit": "historical_event_case",
        "case_count": len(case_effects),
        "open_source_decision_minimum_case_count": (
            open_source_decision_minimum_cases
        ),
        "open_source_decision_case_count_met": (
            len(case_effects) >= open_source_decision_minimum_cases
        ),
        "research_minimum_case_count": research_minimum_cases,
        "research_case_count_met": len(case_effects) >= research_minimum_cases,
        "inference_status": (
            "research_scale_case_count_met"
            if len(case_effects) >= research_minimum_cases
            else "open_source_decision_grade"
            if len(case_effects) >= open_source_decision_minimum_cases
            else "small_sample_descriptive_only"
        ),
        "target_count": sum(item["target_count"] for item in case_effects),
        "mean_b3_minus_b1": round(mean_delta, 8),
        "median_case_b3_minus_b1": round(median(case_deltas), 8),
        "improved_case_fraction": round(improved / len(case_deltas), 8),
        "unchanged_case_fraction": round(unchanged / len(case_deltas), 8),
        "nonworse_case_fraction": round(
            (improved + unchanged) / len(case_deltas), 8
        ),
        "changed_target_count": len(changed_target_deltas),
        "improved_changed_target_count": improved_changed_targets,
        "worsened_changed_target_count": worsened_changed_targets,
        "improved_changed_target_fraction": (
            round(
                improved_changed_targets / len(changed_target_deltas), 8
            )
            if changed_target_deltas
            else None
        ),
        "cluster_bootstrap_interval_95": {
            "lower": round(_quantile(bootstrap_means, 0.025), 8),
            "upper": round(_quantile(bootstrap_means, 0.975), 8),
            "samples": bootstrap_samples,
            "random_seed": random_seed,
        },
        "case_effects": case_effects,
        "interpretation_limits": [
            "The interval resamples cases because targets within one event are not independent.",
            "More simulation seeds reduce within-case variance but do not increase the number of independent historical events.",
            "A small-case interval is descriptive and must not be read as proof of superiority or equivalence.",
            "Both case-count thresholds are reporting guardrails, not post-hoc power calculations.",
        ],
    }
