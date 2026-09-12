#!/usr/bin/env python3
"""Recompute a frozen, offline comparison of observation review rules."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rule(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(directory):
    baseline = load_rule(directory / "baseline_review.py", "baseline_review")
    candidate = load_rule(directory / "candidate_review.py", "candidate_review")
    cases = read(directory / "cases.json")
    totals = {"before": Counter(), "after": Counter()}
    rows = []
    case_rows = []
    for case in cases:
        case_rows_start = len(rows)
        for i, scenario in enumerate(case["scenarios"]):
            for j, observation in enumerate(scenario["observations"]):
                saved = deepcopy(observation)
                before = baseline.review_observation(case["spec"], observation)
                after = candidate.review_observation(case["spec"], observation)
                assert saved == observation, "review changed the input condition"
                assert before["review_findings"] == observation["review_findings"], "baseline does not match sealed real-case result"
                for label, result in (("before", before), ("after", after)):
                    totals[label].update(item["code"] for item in result["review_findings"])
                rows.append({
                    "case_id": case["case_id"], "scenario_index": i, "observation_index": j,
                    "scenario": scenario["label"], "kind": observation["kind"],
                    "signal": observation["signal"], "source": observation.get("source"),
                    "before": before, "after": after,
                })
        selected = rows[case_rows_start:]
        case_rows.append({"case_id": case["case_id"], "scenarios": len(case["scenarios"]), "observations": len(selected),
                          **{label: sum(bool(row[label]["review_findings"]) for row in selected) for label in totals}})
    return {
        "classification": "offline_review_rule_regression_not_generation_quality_or_prediction_accuracy",
        "model_calls": 0, "case_count": len(cases), "scenario_count": sum(row["scenarios"] for row in case_rows),
        "observation_count": len(rows), "input_text_preserved": True,
        "flagged_before": sum(bool(row["before"]["review_findings"]) for row in rows),
        "flagged_after": sum(bool(row["after"]["review_findings"]) for row in rows),
        "newly_flagged": sum(not row["before"]["review_findings"] and bool(row["after"]["review_findings"]) for row in rows),
        "no_longer_flagged": sum(bool(row["before"]["review_findings"]) and not row["after"]["review_findings"] for row in rows),
        "finding_counts": {label: dict(sorted(value.items())) for label, value in totals.items()},
        "cases": case_rows, "conditions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--write", action="store_true", help="write comparison before sealing the bundle")
    args = parser.parse_args()
    checksums = args.directory / "checksums.json"
    if args.write:
        if checksums.exists():
            raise ValueError("sealed comparison cannot be overwritten")
    else:
        for name, digest in read(checksums).items():
            if hashlib.sha256((args.directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError("checksum mismatch: " + name)
    result = compare(args.directory)
    if args.write:
        (args.directory / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif read(args.directory / "report.json") != result:
        raise ValueError("recomputed observation review differs from the report")
    print(json.dumps({key: value for key, value in result.items() if key != "conditions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
