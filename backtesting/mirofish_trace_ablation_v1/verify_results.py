#!/usr/bin/env python3
"""Verify a portable trace-ablation bundle using only the standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FLAGS = ("multi_actor_causal_chain", "observable_trigger", "specific_invalidation", "decision_relevant", "evidence_grounded")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root):
    checksums = read(root / "checksums.json")
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("checksum path escapes bundle")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    manifest = read(root / "manifest.json")
    report = read(root / "report.json")
    result = {}
    for variant in ("with_traces", "without_traces"):
        totals = []; qualified = branches = agreements = comparisons = score_near = score_count = 0
        for relative in manifest["cases"]:
            entry = read(root / relative)
            arm = entry["variants"][variant]
            first, second = arm["quality"]
            totals.extend([first["total_score"], second["total_score"]])
            for key in first["scores"]:
                score_count += 1
                score_near += abs(first["scores"][key] - second["scores"][key]) <= 1
            second_by_id = {branch["id"]: branch for branch in second["branches"]}
            assert {branch["id"] for branch in first["branches"]} == set(second_by_id)
            assert {branch["id"] for branch in arm["branches"]} == set(second_by_id)
            for first_branch in first["branches"]:
                second_branch = second_by_id[first_branch["id"]]
                branches += 1
                qualified += all(first_branch[flag] is True and second_branch[flag] is True for flag in FLAGS)
                for flag in FLAGS:
                    comparisons += 1
                    agreements += first_branch[flag] == second_branch[flag]
        observed = {"mean_quality_score_out_of_20": sum(totals) / len(totals), "strict_qualified_branch_rate": qualified / branches, "score_within_one_rate": score_near / score_count, "branch_flag_agreement_rate": agreements / comparisons}
        expected = report["variants"][variant]
        for key, value in observed.items():
            reference = expected.get(key, expected["quality_repeat_reliability"].get(key))
            assert abs(value - reference) < 0.000001, (variant, key, value, reference)
        result[variant] = observed
    assert len(manifest["cases"]) == report["case_count"]
    return {"verified_cases": len(manifest["cases"]), "verified_files": len(checksums), "recomputed_metrics": result, "semantic_verdict": report["semantic_verdict"]["verdict"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle), ensure_ascii=False, indent=2))
