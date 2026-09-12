#!/usr/bin/env python3
"""Recompute a product experiment from anonymous outputs; standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

VARIANTS = ("decisions_only", "filtered_interactions", "raw_interactions")
FLAGS = ("multi_actor_causal_chain", "observable_trigger", "specific_invalidation", "decision_relevant", "evidence_grounded")
WATCH_FLAGS = ("observable_trigger", "specific_invalidation", "evidence_grounded")
SCORES = ("mechanism_coherence", "monitoring_actionability", "falsifiability", "evidence_discipline", "scenario_diversity")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(values):
    return statistics.mean(values) if values else None


def quality_metrics(first, second):
    right = {item["id"]: item for item in second["branches"]}
    assert len(first["branches"]) == len(right) == 4
    checks = [{key: branch[key] and right[branch["id"]][key] for key in FLAGS} for branch in first["branches"]]
    return {
        "score": mean([sum(first["scores"].values()), sum(second["scores"].values())]),
        "scores": {key: mean([first["scores"][key], second["scores"][key]]) for key in SCORES},
        "watchlist_usable_rate": mean([all(branch[key] for key in WATCH_FLAGS) for branch in checks]),
        "strict_qualified_rate": mean([all(branch.values()) for branch in checks]),
        "flags": {key: mean([branch[key] for branch in checks]) for key in FLAGS},
        "flag_agreement": mean([branch[key] == right[branch["id"]][key] for branch in first["branches"] for key in FLAGS]),
        "score_within_one": mean([abs(first["scores"][key] - second["scores"][key]) <= 1 for key in SCORES]),
    }


def summarize(cases, manifest):
    result = {"experiment_id": manifest["experiment_id"], "case_count": len(cases), "registered_case_count": manifest["registered_case_count"], "complete": True, "variants": {}, "cases": []}
    for case in cases:
        case_metrics = {"case_id": case["case_id"], "event_type": case["event_type"], "variants": {}}
        for variant in VARIANTS:
            arm = case["variants"][variant]
            judgments = arm["quality"]
            metrics = quality_metrics(*judgments) if len(judgments) == 2 else None
            case_metrics["variants"][variant] = metrics
        result["cases"].append(case_metrics)
    for variant in VARIANTS:
        arms = [case["variants"][variant] for case in cases]
        quality = [case["variants"][variant] for case in result["cases"] if case["variants"][variant] is not None]
        generated = [arm for arm in arms if arm["generation"].get("status") == "sealed"]
        result["complete"] &= len(generated) == len(quality) == len(cases)
        generation_usage = [row for case in cases for row in case["usage"] if row["variant"] == variant and row["phase"] == "generation"]
        branch_count = sum(len(arm["branches"]) for arm in generated)
        coverage = []
        for case in cases:
            arm = case["variants"][variant]
            if arm["generation"].get("status") == "sealed":
                covered = {actor for branch in arm["branches"] for actor in branch["actor_ids"]}
                coverage.append(len(covered & set(case["decision_actor_ids"])) / len(case["decision_actor_ids"]))
        costs = [row["estimated_cost"] for row in generation_usage if isinstance(row.get("estimated_cost"), (int, float))]
        result["variants"][variant] = {
            "completed_generations": len(generated), "scheduled_generations": len(cases),
            "generation_completion_rate": len(generated) / len(cases), "quality_case_count": len(quality), "branch_count": branch_count,
            "decision_actor_coverage": mean(coverage),
            "mean_quality_score_out_of_20": mean([row["score"] for row in quality]),
            "watchlist_usable_rate": mean([row["watchlist_usable_rate"] for row in quality]),
            "strict_qualified_rate": mean([row["strict_qualified_rate"] for row in quality]),
            "mean_dimension_scores": {key: mean([row["scores"][key] for row in quality]) for key in SCORES},
            "conservative_branch_flags": {key: mean([row["flags"][key] for row in quality]) for key in FLAGS},
            "judge_flag_agreement": mean([row["flag_agreement"] for row in quality]),
            "judge_score_within_one": mean([row["score_within_one"] for row in quality]),
            "median_generation_seconds": statistics.median([arm["generation"]["elapsed_seconds"] for arm in generated]) if generated else None,
            "generation_calls": len(generation_usage),
            "generation_failures": sum(row["status"] == "failed" for row in generation_usage),
            "semantic_retries": sum(arm["generation"].get("semantic_retries", 0) for arm in generated),
            "generation_prompt_tokens": sum(row.get("prompt_tokens", 0) for row in generation_usage),
            "generation_completion_tokens": sum(row.get("completion_tokens", 0) for row in generation_usage),
            "generation_estimated_cost": sum(costs) if costs else None,
            "visible_interaction_count": sum(arm["generation"].get("visible_interactions", 0) for arm in arms),
            "initial_prompt_chars": sum(arm["generation"].get("prompt_chars", 0) for arm in arms),
        }
    usage = [row for case in cases for row in case["usage"]]
    costs = [row["estimated_cost"] for row in usage if isinstance(row.get("estimated_cost"), (int, float))]
    result["usage"] = {"attempts": len(usage), "completed": sum(row["status"] == "completed" for row in usage), "failed": sum(row["status"] == "failed" for row in usage), "total_tokens": sum(row.get("total_tokens", 0) for row in usage), "estimated_cost": sum(costs) if costs else None, "cost_source": "provider_response; currency not inferred"}
    reliable = result["complete"] and all(min(arm["judge_flag_agreement"], arm["judge_score_within_one"]) >= 0.8 for arm in result["variants"].values())
    result["semantic_verdict"] = "inspect_paired_differences_exploratory" if reliable else "inconclusive_incomplete_or_low_repeat_agreement"
    result["paired_differences"] = {}
    for baseline in ("decisions_only", "raw_interactions"):
        pairs = [(case["variants"]["filtered_interactions"], case["variants"][baseline]) for case in result["cases"] if case["variants"]["filtered_interactions"] is not None and case["variants"][baseline] is not None]
        result["paired_differences"][f"filtered_minus_{baseline}"] = {"paired_cases": len(pairs), "mean_quality_difference": mean([a["score"] - b["score"] for a, b in pairs]), "mean_watchlist_rate_difference": mean([a["watchlist_usable_rate"] - b["watchlist_usable_rate"] for a, b in pairs])}
    return result


def assert_same_metrics(observed, expected):
    """Accept machine-rounding differences across Python versions, not changed scores."""
    if isinstance(observed, dict) and isinstance(expected, dict):
        assert observed.keys() == expected.keys(), "metric keys differ"
        for key in observed:
            assert_same_metrics(observed[key], expected[key])
    elif isinstance(observed, list) and isinstance(expected, list):
        assert len(observed) == len(expected), "metric list lengths differ"
        for left, right in zip(observed, expected):
            assert_same_metrics(left, right)
    elif isinstance(observed, bool) or isinstance(expected, bool):
        assert type(observed) is type(expected) and observed == expected
    elif isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        assert math.isfinite(observed) and math.isfinite(expected)
        assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12), "numeric metric differs"
    else:
        assert observed == expected, "metric differs"


def verify(root):
    root = Path(root).resolve()
    hashes = read(root / "checksums.json")
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    manifest = read(root / "manifest.json")
    observed = summarize([read(root / path) for path in manifest["cases"]], manifest)
    assert_same_metrics(observed, read(root / "report.json"))
    return {"verified_files": len(hashes), **observed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = verify(args.bundle)
    result.pop("cases", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
