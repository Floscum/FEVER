#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze, run and package a bounded 24-case v7/v8 product comparison."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fever_mirofish.contracts import canonical_sha256, validate_result, validate_financial_actions
from fever_mirofish.scenario_branches import build_scenario_branch_prompt, build_scenario_branch_set, build_scenario_branch_retry_prompt
from run_trace_ablation_experiment import _sanitize_text, _assert_outbound_prompt_is_anonymous
from run_compiler_decision_support_diagnostic import ModelCaller, _with_retry
from product_experiment_budget import ExperimentBudget
from verify_product_iteration import VARIANTS, CRITERIA, read, summarize, verify

EXPERIMENT = "product-usability-experiment-v2"
OUTPUT = ROOT / ".data/benchmarks" / EXPERIMENT
DELIVERY = ROOT / "artifacts" / EXPERIMENT
PILOT_NUMBERS = (5, 18, 24)
LOCK = threading.Lock()
CODE_FILES = ("scripts/run_product_iteration.py", "scripts/verify_product_iteration.py", "scripts/product_experiment_budget.py", "src/fever_mirofish/product_scenarios.py", "src/fever_mirofish/scenario_branches.py", "src/fever_mirofish/trace_digest.py", "src/fever_mirofish/contracts.py", "scripts/run_trace_ablation_experiment.py", "scripts/run_compiler_decision_support_diagnostic.py", "src/fever_mirofish/usage_ledger.py")
JUDGE_SYSTEM = """你是事件研究产物的复核者。只用给定事实、约束和模拟决策，逐条判断输出是否便于研究者使用。不要猜测真实结果，也不要按文风、篇幅或字段数量打分。
当前模拟决策与未来条件响应属于不同时间：WAIT 是合法起点，明确条件下转为行动可以合理；不得把它写成已采取的行动。角色假设、模拟消息不是事实。
逐条判断五项：
action_time_clear：清楚区分事实、当前模拟起点与未来假设，没有把未来响应说成已发生。
observable：触发能指明可核对的事件或指标，并能判断去哪里或在何时复核；散文或独立字段都可，不能只说“出现变化”“市场反应”。
falsifiable：失效条件能推翻该分支的核心传导，而非只重复“触发未出现”或“预期不及”。
grounded：关键机制与引用事实或明确角色约束有依据，没有捏造已知资源、数值、期限或真实结果；假设本身不要求已经发生，但不能违反给定约束。
multi_actor：至少两方存在明确的行动响应关系，说明一方怎样改变另一方的选择，而非罗列角色或只表达共同观望。
每项返回 pass 布尔值、quote（从该分支输出中逐字摘取一个短片段，不能拼接或加省略号）和 reason（简述理由）。缺少要素时 quote 摘取与缺失最相关的原句，reason 指出缺失。每个 quote 和 reason 各不超过60个汉字。
不要输出总分。只输出 JSON：{"branches":[{"index":1,"criteria":{"action_time_clear":{"pass":true,"quote":"…","reason":"…"},"observable":{"pass":true,"quote":"…","reason":"…"},"falsifiable":{"pass":true,"quote":"…","reason":"…"},"grounded":{"pass":true,"quote":"…","reason":"…"},"multi_actor":{"pass":true,"quote":"…","reason":"…"}}}]}。"""


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def append(path, value):
    with LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def select_cases():
    candidates = sorted((ROOT / ".data/smoke-runs").glob("blind*-b3/financial-actions.json")) + sorted((ROOT / ".data/formal-runs").glob("*/blind*/core/replication-00/financial-actions.json"))
    selected = {}
    for path in candidates:
        decisions = read(path)
        case_id = decisions["case_id"]
        number = int(case_id.rsplit("_", 1)[1])
        if number > 25 or case_id in selected:
            continue
        spec_path = ROOT / "examples/replays" / case_id / "spec.json"
        simulation_path = path.with_name("simulation-result.json")
        spec, simulation = read(spec_path), read(simulation_path)
        validate_result(simulation, spec)
        validate_financial_actions(decisions, spec)
        entry = {"case_id": case_id, "number": number, "anonymous_case_id": f"anonymous-case-{number:02d}", "configured_actor_count": len(spec["actors"]), "decision_actor_count": len(decisions["decisions"]), "fact_count": len(spec["facts"]), "fact_chars": sum(len(fact["statement"]) for fact in spec["facts"])}
        for name, file, value in (("spec", spec_path, spec), ("simulation_result", simulation_path, simulation), ("financial_actions", path, decisions)):
            entry[f"{name}_path"] = str(file.relative_to(ROOT))
            entry[f"{name}_sha256"] = canonical_sha256(value)
        selected[case_id] = entry
    entries = sorted(selected.values(), key=lambda entry: entry["number"])
    if len(entries) != 24:
        raise ValueError("expected 24 existing replay cases numbered 1–25, excluding 16 without a completed decision run")
    return entries


def inputs(entry):
    values = []
    for key in ("spec", "simulation_result", "financial_actions"):
        value = read(ROOT / entry[f"{key}_path"])
        if canonical_sha256(value) != entry[f"{key}_sha256"]:
            raise ValueError(f"frozen {key} changed")
        values.append(value)
    return values


def anonymize(value, entry):
    if isinstance(value, str):
        return _sanitize_text(value, entry)
    if isinstance(value, list):
        return [anonymize(item, entry) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "case_id":
            result[key] = entry["anonymous_case_id"]
        elif key in {"as_of", "observed_at"}:
            result[key] = "D0"
        elif key == "end_at":
            # Preserve the numeric horizon; the legacy sanitizer replaced
            # every question with three trading days, which is wrong here.
            result[key] = "D0 + stated horizon"
        elif key in {"source_ref", "source_url"}:
            result[key] = "anonymized-source"
        else:
            result[key] = anonymize(item, entry)
    return result


def generation_prompt(entry, variant):
    system, user, _ = build_scenario_branch_prompt(*inputs(entry), product=True, product_version=f"scenario-branch-compiler-{variant}")
    user = json.dumps(anonymize(json.loads(user), entry), ensure_ascii=False)
    _assert_outbound_prompt_is_anonymous({"system": system, "user": user}, entry)
    return {"system": system, "user": user}


def register():
    entries = select_cases()
    protocol = {
        "case_count": 24, "pilot": [f"anonymous-case-{number:02d}" for number in PILOT_NUMBERS], "variants": list(VARIANTS),
        "selection": "Existing anonymized replay cases 001–025, excluding 016 because no completed decision run exists; no outcome files or prior quality scores used",
        "controls": "Identical facts, question, horizon, actors, decisions, filtered interactions, deterministic slots and provider; input JSON equality checked",
        "question_anonymization": "Preserve original question and numeric horizon, remove dates, security codes, local paths and source URLs",
        "scope": "Compiler product regression using completed simulations; does not rerun simulation, test new role selection, or measure prediction accuracy",
        "pilot_gate": "All 6 generations and 12 quote-validated judgments complete; inspect action timing and observation cards before expanding. Pilot remains included and reported separately.",
        "quality_repeats": 2, "criteria": list(CRITERIA), "judge_system": JUDGE_SYSTEM,
        "watchlist_rule": "First four criteria true in BOTH passes; multi_actor reported separately; only compare direction if repeat agreement >=80% in both arms",
        "limitations": ["same model generates and judges", "development replay cases, not a new holdout", "one generation per case/version", "field structure reveals format, version names hidden from judge", "model review is not human acceptance", "four facts per case remain concise"],
        "provider": "DeepInfra (api.deepinfra.com)", "model": "deepseek-ai/DeepSeek-V4-Flash",
        "calls_without_retries": 144, "max_calls": 192, "max_total_tokens": 1200000, "max_estimated_usd": 0.5,
        "workers": 3, "sdk_retries": 0, "max_semantic_retries": 1,
        "per_request": "30000 UTF-8 prompt bytes, max 5000 output tokens; conservative dollar reservation including in-flight requests",
        "token_threshold": "Checked before dispatch; at most three in-flight requests may pass the token threshold",
        "pricing_source": "https://deepinfra.com/deepseek-ai/DeepSeek-V4-Flash", "pricing_checked": "2026-09-06",
        "authorization": "User approved DeepInfra/DeepSeek anonymous event inputs and requested further iteration and larger testing; incremental cap $0.50 keeps old and new authorized experiment caps under $1 of actual reserved spend",
    }
    plans = {entry["anonymous_case_id"]: {variant: generation_prompt(entry, variant) for variant in VARIANTS} for entry in entries}
    for prompts in plans.values():
        if prompts["v7"]["user"] != prompts["v8"]["user"]:
            raise ValueError("comparison inputs differ")
        for prompt in prompts.values():
            if sum(len(value.encode()) for value in prompt.values()) > 29000:
                raise ValueError("generation leaves insufficient repair headroom")
    candidate = {"experiment_id": EXPERIMENT, "revision": 2, "prior_manifest_sha256": canonical_sha256(read(OUTPUT / "manifest-revision-1.json")),
        "pilot_revision_reason": "Draft v8 assigned some responses to the wrong role and invented observation deadlines. Revision 2 requires explicit actor IDs and first-person responses; windows are assigned from the requested horizon. Existing costs and call counts carry forward under the unchanged $0.50 / 192-call ceiling.",
        "protocol": protocol, "protocol_sha256": canonical_sha256(protocol), "cases": entries,
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES},
        "generation_prompt_sha256": {case: {variant: canonical_sha256(prompt) for variant,prompt in prompts.items()} for case,prompts in plans.items()}}
    path = OUTPUT / "manifest.json"
    if path.exists():
        existing = read(path)
        if {key:value for key,value in existing.items() if key != "registered_at"} != candidate:
            raise ValueError("registered protocol or execution code changed; preserve run and use a new version")
        return existing
    candidate["registered_at"] = datetime.now(timezone.utc).isoformat()
    write(path, candidate)
    for name in CODE_FILES:
        dest = OUTPUT / "execution-code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    for case,prompts in plans.items():
        for variant,prompt in prompts.items():
            write(OUTPUT / "planned-requests" / case / f"{variant}.json", prompt)
    return candidate


class Caller(ModelCaller):
    def __init__(self, manifest):
        protocol = manifest["protocol"]
        os.environ["FEVER_COMPILER_DIAGNOSTIC_MAX_LLM_CALLS"] = str(protocol["max_calls"])
        os.environ["FEVER_COMPILER_DIAGNOSTIC_MAX_TOTAL_TOKENS"] = str(protocol["max_total_tokens"])
        super().__init__(OUTPUT, protocol["model"])
        self.client = self.client.with_options(max_retries=0, timeout=180)
        if urlparse(str(self.client.base_url)).hostname != "api.deepinfra.com":
            raise ValueError("provider differs from approved endpoint")
        self.context = threading.local()
        self.budget = ExperimentBudget(OUTPUT / "dollar-budget.json", protocol["max_estimated_usd"])

    def call(self, system, user, *, max_tokens):
        meta, entry = self.context.value
        _assert_outbound_prompt_is_anonymous({"system": system, "user": user}, entry)
        request_id = uuid.uuid4().hex
        self.budget.reserve(request_id, system, user, max_tokens)
        write(OUTPUT / "requests" / f"{request_id}.json", {**meta, "system": system, "user": user})
        row = {**meta, "revision": 2, "request_id": request_id, "request_sha256": canonical_sha256({"system": system, "user": user}), "started_at": datetime.now(timezone.utc).isoformat()}
        started, settled = time.monotonic(), None
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{"role":"system","content":system},{"role":"user","content":user}], response_format={"type":"json_object"}, max_tokens=max_tokens)
            usage = self.ledger._usage_to_dict(response.usage)
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                settled = usage
            content = response.choices[0].message.content or ""
            row.update({"status":"completed", **{key:usage.get(key) for key in ("prompt_tokens","completion_tokens","total_tokens","estimated_cost")}, "finish_reason":response.choices[0].finish_reason})
            write(OUTPUT / "responses" / f"{request_id}.json", {"content":content,"finish_reason":row["finish_reason"]})
            return content
        except Exception as error:
            row.update(status="failed",error_type=type(error).__name__)
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            append(OUTPUT / "calls.jsonl", row)
            self.budget.settle(request_id, settled)


def path_for(entry, variant, phase, repeat=0):
    return OUTPUT / f"{phase}-r2" / entry["anonymous_case_id"] / f"{variant}{f'-{repeat}' if repeat else ''}.json"


def generate(entry, variant, caller, manifest):
    path = path_for(entry, variant, "generation")
    prompt = generation_prompt(entry, variant)
    expected = manifest["generation_prompt_sha256"][entry["anonymous_case_id"]][variant]
    if canonical_sha256(prompt) != expected:
        raise ValueError("prompt drift")
    if path.exists():
        if read(path)["prompt_sha256"] != expected:
            raise ValueError("cached output drift")
        return "reused"
    caller.context.value = ({"case_id":entry["anonymous_case_id"],"variant":variant,"phase":"generation"},entry)
    artifact,retries,elapsed = _with_retry(caller,**prompt,max_tokens=5000,parser=lambda raw:build_scenario_branch_set(raw,*inputs(entry),model_id=caller.model,product=True,product_version=f"scenario-branch-compiler-{variant}"),retry_builder=build_scenario_branch_retry_prompt)
    write(path,{"artifact":artifact,"prompt_sha256":expected,"generation":{"semantic_retries":retries,"elapsed_seconds":elapsed}})
    return f"{elapsed:.1f}s; retries={retries}"


def judge_prompt(entry, generated):
    payload = json.loads(generation_prompt(entry,"v7")["user"])
    payload.pop("simulated_actions")
    payload.pop("branch_slots")
    keys = ("label","summary","actor_ids","trigger_conditions","invalidation_conditions","consequences","evidence_refs","conditional_responses","observations")
    payload["branches"] = anonymize([{key:branch[key] for key in keys if key in branch} for branch in generated["artifact"]["branches"]],entry)
    return {"system":JUDGE_SYSTEM,"user":json.dumps(payload,ensure_ascii=False)}


def text_leaves(value):
    if isinstance(value,str):
        return [value]
    if isinstance(value,list):
        return [text for item in value for text in text_leaves(item)]
    if isinstance(value,dict):
        return [text for item in value.values() for text in text_leaves(item)]
    return []


def validate_judgment(raw, branches):
    result = json.loads(raw) if isinstance(raw,str) else raw
    rows = result.get("branches")
    if not isinstance(rows,list) or len(rows) != len(branches):
        raise ValueError("judgment requires one entry per branch")
    for index,(row,branch) in enumerate(zip(rows,branches),1):
        if row.get("index") != index or set(row.get("criteria",{})) != set(CRITERIA):
            raise ValueError("judgment index or criteria mismatch")
        for criterion in row["criteria"].values():
            if type(criterion.get("pass")) is not bool or not isinstance(criterion.get("reason"),str) or not criterion["reason"].strip():
                raise ValueError("criterion needs boolean pass and reason")
            quote = criterion.get("quote")
            if not isinstance(quote,str) or not quote.strip() or not any(quote in leaf for leaf in text_leaves(branch)):
                raise ValueError("quote must be an exact contiguous excerpt from that branch, without added ellipses")
    return result


def judge(entry, variant, repeat, caller):
    generated = read(path_for(entry,variant,"generation"))
    source_hash = canonical_sha256(generated)
    path = path_for(entry,variant,"quality",repeat)
    if path.exists():
        if read(path)["generation_sha256"] != source_hash:
            raise ValueError("cached judgment drift")
        return "reused"
    prompt = judge_prompt(entry,generated)
    caller.context.value = ({"case_id":entry["anonymous_case_id"],"variant":variant,"phase":"quality","repeat":repeat},entry)
    judgment,retries,elapsed = _with_retry(caller,**prompt,max_tokens=5000,parser=lambda raw:validate_judgment(raw,json.loads(prompt["user"])["branches"]),retry_builder=build_scenario_branch_retry_prompt)
    write(path,{"judgment":judgment,"generation_sha256":source_hash,"semantic_retries":retries,"elapsed_seconds":elapsed})
    return f"{elapsed:.1f}s; retries={retries}"


def run_phase(jobs, worker):
    jobs.sort(key=lambda job:canonical_sha256([job[0]["anonymous_case_id"],*job[1:]]))
    failures=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(worker,*job):job for job in jobs}
        for future in as_completed(pending):
            job=pending[future];label=" ".join(map(str,[job[0]["anonymous_case_id"],*job[1:]]))
            try:
                print(f"ok {label}: {future.result()}",flush=True)
            except Exception as error:
                failures.append(label)
                append(OUTPUT/"failures.jsonl",{"job":label,"error_type":type(error).__name__,"reason":str(error)[:240] if isinstance(error,ValueError) else "inspect local request record"})
                print(f"failed {label}: {type(error).__name__}",flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; sealed outputs retained")


def package(manifest, entries):
    all_calls=[json.loads(line) for line in (OUTPUT/"calls.jsonl").read_text().splitlines() if line.strip()] if (OUTPUT/"calls.jsonl").exists() else []
    ids={entry["anonymous_case_id"] for entry in entries}
    calls=[call for call in all_calls if call["case_id"] in ids]
    cases=[]
    for entry in entries:
        case={key:entry[key] for key in ("anonymous_case_id","configured_actor_count","decision_actor_count","fact_count","fact_chars")}
        case["input"]=json.loads(generation_prompt(entry,"v7")["user"])
        case["variants"]={}
        for variant in VARIANTS:
            path=path_for(entry,variant,"generation")
            if not path.exists():
                continue
            generated=read(path)
            case["variants"][variant]={"branches":anonymize(generated["artifact"]["branches"],entry),"generation":generated["generation"],"judgments":[read(path_for(entry,variant,"quality",repeat))["judgment"] for repeat in (1,2) if path_for(entry,variant,"quality",repeat).exists()]}
        cases.append(case)
        write(DELIVERY/"cases"/f"{entry['anonymous_case_id']}.json",case)
    report=summarize(cases,calls,manifest["protocol"]["case_count"])
    public_manifest={key:manifest[key] for key in ("experiment_id","revision","prior_manifest_sha256","pilot_revision_reason","registered_at","protocol","protocol_sha256","code_sha256","generation_prompt_sha256")}
    public_manifest["included_case_ids"]=[case["anonymous_case_id"] for case in cases]
    public_manifest["input_sha256"]={entry["anonymous_case_id"]:{key:entry[key] for key in entry if key.endswith("_sha256")} for entry in entries}
    write(DELIVERY/"manifest.json",public_manifest);write(DELIVERY/"report.json",report);write(DELIVERY/"calls.json",calls)
    shutil.copy2(ROOT/"scripts/verify_product_iteration.py",DELIVERY/"verify_results.py")
    files=[path for path in DELIVERY.rglob("*") if path.is_file() and path.name!="checksums.json"]
    write(DELIVERY/"checksums.json",{str(path.relative_to(DELIVERY)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)})
    verify(DELIVERY)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("plan","pilot","expand","report"))
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=register()
        if args.action=="plan":
            print(json.dumps({"protocol":manifest["protocol"],"prepared_generation_prompts":48,"max_prompt_bytes":max(sum(len(value.encode()) for value in generation_prompt(entry,variant).values()) for entry in manifest["cases"] for variant in VARIANTS)},ensure_ascii=False,indent=2));return
        entries=manifest["cases"] if args.action!="pilot" else [entry for entry in manifest["cases"] if entry["number"] in PILOT_NUMBERS]
        if args.action in {"pilot","expand"}:
            if args.action=="expand":
                for entry in manifest["cases"]:
                    if entry["number"] in PILOT_NUMBERS:
                        for variant in VARIANTS:
                            for repeat in (1,2):
                                if not path_for(entry,variant,"quality",repeat).exists():
                                    raise ValueError("pilot must be completed before expansion")
            caller=Caller(manifest)
            run_phase([(entry,variant) for entry in entries for variant in VARIANTS],lambda entry,variant:generate(entry,variant,caller,manifest))
            run_phase([(entry,variant,repeat) for entry in entries for variant in VARIANTS for repeat in (1,2)],lambda entry,variant,repeat:judge(entry,variant,repeat,caller))
        print(json.dumps(package(manifest,entries),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
