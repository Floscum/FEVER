#!/usr/bin/env python3
"""Portable, offline recomputation of product iteration v2; standard library only."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
from pathlib import Path

VARIANTS = ("v7", "v8")
CRITERIA = ("action_time_clear", "observable", "falsifiable", "grounded", "multi_actor")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(cases, calls, registered_count):
    result = {"case_count": len(cases), "registered_case_count": registered_count, "variants": {}}
    for variant in VARIANTS:
        outputs = [case["variants"][variant] for case in cases if variant in case["variants"]]
        judgments = [output for output in outputs if len(output.get("judgments", [])) == 2]
        agree = total = usable = all_pass = branch_count = 0
        passed = {key: 0 for key in CRITERIA}
        for output in judgments:
            for left, right in zip(output["judgments"][0]["branches"], output["judgments"][1]["branches"]):
                branch_count += 1
                flags = {}
                for key in CRITERIA:
                    a, b = left["criteria"][key]["pass"], right["criteria"][key]["pass"]
                    agree += a == b
                    total += 1
                    flags[key] = a and b
                    passed[key] += flags[key]
                usable += all(flags[key] for key in CRITERIA[:4])
                all_pass += all(flags.values())
        usage = [call for call in calls if call["variant"] == variant and call["phase"] == "generation"]
        covered = eligible = configured = cards = all_branches = 0
        for case in cases:
            if variant not in case["variants"]:
                continue
            branches = case["variants"][variant]["branches"]
            covered += len({actor for branch in branches for actor in branch["actor_ids"]})
            eligible += case["decision_actor_count"]
            configured += case["configured_actor_count"]
            all_branches += len(branches)
            cards += sum(bool(branch.get("observations")) for branch in branches)
        result["variants"][variant] = {
            "generated_cases": len(outputs), "judged_cases": len(judgments),
            "branch_count": all_branches, "judged_branches": branch_count,
            "decision_actor_coverage": covered / eligible if eligible else None,
            "configured_actor_coverage": covered / configured if configured else None,
            "observation_card_branches": cards,
            "usable_watchlist_rate": usable / branch_count if branch_count else None,
            "all_criteria_rate": all_pass / branch_count if branch_count else None,
            "criterion_rates": {key: passed[key] / branch_count if branch_count else None for key in CRITERIA},
            "repeat_agreement": agree / total if total else None,
            "median_generation_seconds": statistics.median(output["generation"]["elapsed_seconds"] for output in outputs) if outputs else None,
            "generation_retries": sum(output["generation"]["semantic_retries"] for output in outputs),
            "generation_calls": len(usage),
            "generation_cost_usd": sum(call.get("estimated_cost") or 0 for call in usage),
            "cost_recorded_calls": sum(call.get("estimated_cost") is not None for call in usage),
        }
    result["complete"] = len(cases) == registered_count and all(item["judged_cases"] == registered_count for item in result["variants"].values())
    result["quality_direction_reliable"] = result["complete"] and all((item["repeat_agreement"] or 0) >= 0.8 for item in result["variants"].values())
    result["calls"] = len(calls)
    result["provider_failures"] = sum(call["status"] == "failed" for call in calls)
    result["interrupted_requests_without_usage"] = sum(call["status"] == "interrupted" for call in calls)
    result["interrupted_cost_upper_bound_usd"] = sum(call.get("reserved_cost_usd") or 0 for call in calls if call["status"] == "interrupted")
    result["total_tokens"] = sum(call.get("total_tokens") or 0 for call in calls)
    result["estimated_cost_usd"] = sum(call.get("estimated_cost") or 0 for call in calls)
    result["cost_recorded_calls"] = sum(call.get("estimated_cost") is not None for call in calls)
    return result


def same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int,float)) and isinstance(right, (int,float)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)
    return left == right


def verify(directory):
    directory = Path(directory)
    for name, expected in read(directory / "checksums.json").items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {name}")
    manifest = read(directory / "manifest.json")
    cases = [read(directory / "cases" / f"{case}.json") for case in manifest["included_case_ids"]]
    result = summarize(cases, read(directory / "calls.json"), manifest["protocol"]["case_count"])
    if not same(result, read(directory / "report.json")):
        raise ValueError("report differs from sealed case recomputation")
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent), ensure_ascii=False, indent=2))
