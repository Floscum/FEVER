#!/usr/bin/env python3
"""Portable consistency checks for real-case acceptance; no provider calls."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def verify(root):
    hashes = read(root / "checksums.json")
    for name, expected in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("checksum mismatch: " + name)
    manifest, summary, calls = (read(root / name) for name in ("manifest.json", "metrics.json", "calls.json"))
    sys.path.insert(0, str(root))
    from quantity_review import prepare_scenarios
    presentation_review = []
    assert len(calls) == summary["provider_calls"] == summary["budget"]["attempts"]
    assert len({call["request_id"] for call in calls}) == len(calls)
    assert not summary["budget"]["pending"]
    assert len(calls) <= manifest["max_calls"]
    charged = sum(call["charged_budget_usd"] for call in calls)
    assert math.isclose(charged, summary["budget"]["settled_usd"], abs_tol=1e-12)
    assert charged <= manifest["max_cost_usd"]
    returned = sum(call["usage"].get("estimated_cost") or 0 for call in calls)
    assert math.isclose(returned, summary["returned_cost_usd"], abs_tol=1e-12)
    assert sum(not call["usage_known"] for call in calls) == summary["unknown_usage_calls"]
    for case in manifest["cases"]:
        directory = root / "cases" / case["id"]
        assert hashlib.sha256((directory / "request.json").read_bytes()).hexdigest() == case["request_sha256"]
        original = read(directory / "baseline-spec.json")
        request = read(directory / "request.json")
        assert [fact["statement"] for fact in original["facts"]] == [node["source_data"]["content"] for node in request["evidence_graph"]["nodes"]]
    for row in summary["runs"]:
        directory = root / "cases" / row["case_id"]
        path = directory / row["run_label"]
        original, spec = read(directory / "baseline-spec.json"), read(path / "spec.json")
        assert spec["facts"] == original["facts"] and spec["horizon"] == original["horizon"]
        job = read(path / "job.json")
        assert job["status"] == row["status"]
        output = job.get("result") or {}
        for before, after in zip(output.get("scenarios", []), prepare_scenarios(spec, output)):
            presentation_review.append({"case_id": row["case_id"], "run_label": row["run_label"], "branch_id": before["id"], "original_notices": before.get("review_notices", []), "current_notices": after["review_notices"], "model_rerun": False})
        assert len(spec["actors"]) == row["configured_actors"]
        assert len(output.get("scenarios", [])) == row["branches"]
        decisions = read(path / "financial-actions.json")["decisions"]
        assert len(decisions) == row["valid_decisions"]
        by_id = {decision["id"]: decision for decision in decisions}
        actors = {actor["id"] for actor in spec["actors"]}
        facts = {fact["id"] for fact in spec["facts"]}
        covered = set()
        for scenario in output.get("scenarios", []):
            assert set(scenario["actor_ids"]) <= actors
            assert set(scenario["evidence_refs"]) <= facts
            covered.update(scenario["actor_ids"])
            for decision in scenario["starting_decisions"]:
                source = by_id[decision["decision_ref"]]
                assert decision["actor_id"] == source["actor_id"] and decision["action_type"] == source["action_type"]
        assert len(covered) == row["scenario_actors"]
        usage = [call for call in calls if (call["case_id"], call["run_label"]) == (row["case_id"], row["run_label"])]
        assert usage == read(path / "calls.json")
        assert len(usage) == row["provider_calls"]
        assert math.isclose(sum(call["usage"].get("estimated_cost") or 0 for call in usage), row["returned_cost_usd"], abs_tol=1e-12)
        assert math.isclose(read(path / "timing.json")["end_to_end_seconds"], row["end_to_end_seconds"], abs_tol=1e-12)
    assert presentation_review == read(root / "presentation-review.json")
    return {"verified_files": len(hashes), "cases": len(manifest["cases"]), "runs": len(summary["runs"]), "provider_calls": len(calls), "budget_charged_usd": charged, "semantic_quality_verified": False}


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    print(json.dumps(verify(directory), ensure_ascii=False, indent=2))
