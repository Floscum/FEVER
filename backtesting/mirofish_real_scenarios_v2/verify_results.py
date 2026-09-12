#!/usr/bin/env python3
"""Portable recomputation. Rule findings are triage metrics, not quality scores."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def equivalent(left, right):
    # Python 3.12 improves sum(float) precision. Currency differences at the
    # 17th decimal place must not make an unchanged bundle unverifiable.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    return left == right


def analyzer(directory):
    module_path = directory / "quantity_review.py"
    if not module_path.exists():
        module_path = Path(__file__).resolve().parents[1] / "src/fever_mirofish/scenario_presentation.py"
    module_spec = importlib.util.spec_from_file_location("v2_quantity_review", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def metrics_for_branch(spec, branch, review):
    cards = review.prepare_scenarios(spec, {"scenarios": [branch]})[0]["observations"]
    quantity_conditions = sum(bool(review.unsupported_quantities(spec, {"triggers": [card["signal"]]})) for card in cards)
    findings = [item for card in cards for item in review.review_observation(spec, card)["review_findings"]]
    return {"unsupported_quantities": review.unsupported_quantities(spec, branch), "condition_count": len(cards),
            "conditions_with_unsupported_quantities": quantity_conditions,
            "conditions_needing_review": sum(review.review_observation(spec, card)["review_status"] == "needs_review" for card in cards),
            "finding_codes": dict(Counter(item["code"] for item in findings))}


def verify_branch_set(spec, actions, artifact):
    if artifact["spec_sha256"] != canonical(spec):
        raise ValueError("branch spec hash mismatch")
    decisions = {d["id"]: d for d in actions["decisions"]}
    fact_ids = {fact["id"] for fact in spec["facts"]}
    for branch in artifact["branches"]:
        if not set(branch["evidence_refs"]) <= fact_ids:
            raise ValueError("unknown branch evidence")
        if set(branch["simulation_refs"]) != {a["decision_ref"] for a in branch["actions"]}:
            raise ValueError("branch decision references mismatch")
        if set(branch["actor_ids"]) != {a["actor_id"] for a in branch["actions"]}:
            raise ValueError("branch actors mismatch")
        for action in branch["actions"]:
            d = decisions[action["decision_ref"]]
            if (action["actor_id"], action["action_type"]) != (d["actor_id"], d["action_type"]):
                raise ValueError("branch changed a starting decision")


def summarize(directory):
    directory = Path(directory)
    review = analyzer(directory)
    calls = read(directory / "calls.json") if (directory / "calls.json").exists() else []
    reports, rows = {}, []
    for revision, variants in ((directory, ("v7", "v9")), (directory / "revision-2", ("v7", "v10")), (directory / "revision-3", ("v7", "v11")), (directory / "revision-4", ("v7", "v11"))):
        if not (revision / "manifest.json").exists():
            continue
        manifest = read(revision / "manifest.json")
        revision_name = revision.name if revision.name.startswith("revision-") else "revision-1"
        for cohort in ("public_pilot", "replay_extension"):
            entries = [entry for entry in manifest["paired_cases"] if entry["cohort"] == cohort]
            for variant in variants:
                outputs, branch_metrics, first_metrics, ids, coverage, available = [], [], [], [], 0, 0
                for entry in entries:
                    pair = revision / "paired" / entry["id"]
                    for name, expected in entry["input_sha256"].items():
                        if hashlib.sha256((pair / (name + ".json")).read_bytes()).hexdigest() != expected:
                            raise ValueError("frozen paired input changed")
                    if not (pair / (variant + ".json")).exists():
                        continue
                    output, spec, actions = read(pair / (variant + ".json")), read(pair / "spec.json"), read(pair / "financial-actions.json")
                    attempts = read(pair / (variant + "-attempts.json"))
                    if output["request_ids"] != [a["request_id"] for a in attempts]:
                        raise ValueError("attempt accounting differs")
                    ids += output["request_ids"]
                    outputs.append(output)
                    first = []
                    try:
                        first = json.loads(attempts[0].get("raw_response", ""))["branches"]
                    except (ValueError, KeyError):
                        pass
                    first_metrics.extend(metrics_for_branch(spec, b, review) for b in first if isinstance(b, dict))
                    artifact = output.get("artifact")
                    if artifact:
                        verify_branch_set(spec, actions, artifact)
                        metrics = [metrics_for_branch(spec, b, review) for b in artifact["branches"]]
                        branch_metrics += metrics
                        coverage += len({actor for b in artifact["branches"] for actor in b["actor_ids"]})
                        available += len({a["actor_id"] for a in actions["decisions"]})
                        rows.append({"revision": revision_name, "cohort": cohort, "case_id": entry["id"], "variant": variant, "branch_metrics": metrics})
                if not outputs:
                    continue
                usage = [c for c in calls if c["request_id"] in ids]
                key = f"{revision_name}/{cohort}/{variant}"
                issues = Counter()
                for metric in branch_metrics: issues.update(metric["finding_codes"])
                reports[key] = {"registered_cases": len(entries), "attempted_cases": len(outputs),
                    "valid_cases": sum(bool(o.get("artifact")) for o in outputs),
                    "nonempty_cases": sum(bool((o.get("artifact") or {}).get("branches")) for o in outputs),
                    "first_attempt_valid_cases": sum(o.get("artifact") is not None and o["attempt_count"] == 1 for o in outputs),
                    "failed_cases": [o["case_id"] for o in outputs if o.get("artifact") is None],
                    "branch_count": len(branch_metrics), "first_response_branch_count": len(first_metrics),
                    "withheld_branches": sum(len((o.get("artifact") or {}).get("review", {}).get("withheld_branches", [])) for o in outputs),
                    "partial_cases": sum((o.get("artifact") or {}).get("status") == "partial" for o in outputs),
                    "first_response_branches_with_unsupported_quantities": sum(bool(m["unsupported_quantities"]) for m in first_metrics),
                    "branches_with_unsupported_quantities": sum(bool(m["unsupported_quantities"]) for m in branch_metrics),
                    "condition_count": sum(m["condition_count"] for m in branch_metrics),
                    "conditions_with_unsupported_quantities": sum(m["conditions_with_unsupported_quantities"] for m in branch_metrics),
                    "conditions_needing_review": sum(m["conditions_needing_review"] for m in branch_metrics),
                    "finding_codes": dict(issues), "covered_decision_actors": coverage, "available_decision_actors": available,
                    "request_count": len(ids), "known_usage_requests": sum(c["usage_known"] for c in usage),
                    "returned_cost_usd": sum(c["usage"].get("estimated_cost", 0) for c in usage),
                    "median_elapsed_seconds": statistics.median(o["elapsed_seconds"] for o in outputs)}
    budget = read(directory / "budget.json") if (directory / "budget.json").exists() else {}
    live_runs = []
    for job_path in sorted((directory / "revision-4/cases").glob("*/entity-v7/job.json")):
        run = job_path.parent
        job, spec, actions, artifact = read(job_path), read(run / "spec.json"), read(run / "financial-actions.json"), read(run / "scenario-branches.json")
        verify_branch_set(spec, actions, artifact)
        payload = job.get("result") or {}
        expected_display = {**payload, "scenarios": review.prepare_scenarios(spec, payload)}
        if read(run / "reviewed-display.json") != expected_display:
            raise ValueError("post-run display differs from frozen helper recomputation")
        case_id = run.parent.name
        usage = [c for c in calls if c["case_id"] == case_id and c["run_label"] == "entity-v7"]
        branch_metrics = [metrics_for_branch(spec, b, review) for b in artifact["branches"]]
        live_runs.append({"case_id": case_id, "status": job["status"], "configured_actors": len(spec["actors"]),
                         "named_entities": [a["identity"]["name"] for a in spec["actors"] if a.get("identity")],
                         "valid_decisions": len(actions["decisions"]), "action_types": dict(Counter(d["action_type"] for d in actions["decisions"])),
                         "branches": len(artifact["branches"]), "branch_metrics": branch_metrics,
                         "end_to_end_seconds": read(run / "timing.json")["end_to_end_seconds"],
                         "provider_calls": len(usage), "returned_cost_usd": sum(c["usage"].get("estimated_cost", 0) for c in usage),
                         "unknown_usage_calls": sum(not c["usage_known"] for c in usage)})
    return {"comparisons": reports, "case_reviews": rows, "shared_budget": budget,
            "live_runs": live_runs,
            "all_calls": len(calls), "returned_cost_usd": sum(c["usage"].get("estimated_cost", 0) for c in calls),
            "unknown_usage_requests": sum(not c["usage_known"] for c in calls),
            "unknown_usage_reserved_usd": sum(c["charged_budget_usd"] for c in calls if not c["usage_known"]),
            "semantic_quality_verified": False, "metrics_note": "Presentation findings are recomputed offline with the bundled post-run helper. Unsupported-quantity checks are heuristic; v10/v11 acceptance enforces that rule, so zero findings is not independent evidence of semantic quality. Read first responses, failures, retained branches and case reviews together. No 24-case extension was executed."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.write:
        for name, digest in read(args.directory / "checksums.json").items():
            if hashlib.sha256((args.directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError("checksum mismatch: " + name)
    result = summarize(args.directory)
    if args.write:
        (args.directory / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    elif not equivalent(result, read(args.directory / "metrics.json")):
        raise ValueError("recomputed metrics mismatch")
    print(json.dumps({"comparisons": result["comparisons"], "all_calls": result["all_calls"], "returned_cost_usd": result["returned_cost_usd"], "semantic_quality_verified": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
