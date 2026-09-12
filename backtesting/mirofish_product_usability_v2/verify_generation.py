#!/usr/bin/env python3
"""Portable verification of the post-pilot generation report, without models."""
from __future__ import annotations
import json
import sys
from pathlib import Path


def verify(directory):
    directory=Path(directory)
    sys.path.insert(0,str(directory))
    from verify_results import verify as verify_parent, read, same
    from quantity_review import unsupported_quantities
    result=verify_parent(directory)
    manifest=read(directory/"manifest.json")
    protocol=read(directory/"generation-stress-protocol.json")
    result["original_quality_protocol_complete"]=result.pop("complete")
    result["generation_regression_complete"]=all(value["generated_cases"]==24 for value in result["variants"].values())
    result["classification"]=protocol["classification"]
    cases=[read(directory/"cases"/f"{case_id}.json") for case_id in manifest["included_case_ids"]]
    for case in cases:
        facts={fact["id"] for fact in case["input"]["facts"]}
        actors={actor["id"] for actor in case["input"]["actors"]}
        decisions={decision["id"]:decision for decision in case["input"]["financial_decisions"]}
        for variant in case["variants"].values():
            for branch in variant["branches"]:
                assert set(branch["evidence_refs"])<=facts
                assert set(branch["actor_ids"])<=actors
                for action in branch["actions"]:
                    original=decisions[action["decision_ref"]]
                    assert (action["actor_id"],action["action_type"])==(original["actor_id"],original["action_type"])
    for variant,metrics in result["variants"].items():
        flagged=[]
        for case in cases:
            if variant not in case["variants"]:continue
            for branch in case["variants"][variant]["branches"]:
                numbers=unsupported_quantities(case["input"],branch)
                if numbers:flagged.append({"case_id":case["anonymous_case_id"],"branch_id":branch["id"],"quantities":numbers})
        metrics["quantity_review_branches"]=len(flagged)
        metrics["quantity_review_rate"]=len(flagged)/metrics["branch_count"] if metrics["branch_count"] else None
        metrics["quantity_review_findings"]=flagged
    if not same(result,read(directory/"generation-stress-report.json")):
        raise ValueError("generation report differs from sealed case recomputation")
    calls=read(directory/"calls.json")
    expected={(case_id,variant) for case_id in manifest["included_case_ids"] for variant in ("v7","v8")}
    attempted={(call["case_id"],call["variant"]) for call in calls if call["phase"]=="generation"}
    result["all_case_variants_attempted"]=expected<=attempted
    result["verified_files"]=len(read(directory/"checksums.json"))
    return result


if __name__=="__main__":
    output=verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent)
    print(json.dumps({key:value for key,value in output.items() if key!="variants"},ensure_ascii=False,indent=2))
