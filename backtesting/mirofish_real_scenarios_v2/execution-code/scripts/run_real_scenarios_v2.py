#!/usr/bin/env python3
"""Frozen paired compiler checks plus isolated real simulations; one hard cap."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_real_scenarios as live
from run_product_iteration import anonymize, inputs
from run_trace_ablation_experiment import _assert_outbound_prompt_is_anonymous
from fever_mirofish.scenario_branches import build_scenario_branch_prompt, build_scenario_branch_set, build_scenario_branch_retry_prompt
from fever_mirofish.contracts import canonical_sha256

OUTPUT = ROOT / "artifacts/real-scenarios-v2"
DATA = ROOT / ".data/real-scenarios-v2"
live.ARTIFACTS, live.DATA = OUTPUT, DATA
read = lambda path: json.loads(Path(path).read_text())
write = live.write
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_code():
    names = [*sorted((ROOT / "src/fever_mirofish").glob("*.py")), ROOT / "scripts/generate_scenario_branches.py", ROOT / "scripts/run_real_scenarios.py", Path(__file__).resolve()]
    hashes = {}
    for source in names:
        target = OUTPUT / "execution-code" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("execution code changed; keep existing outputs and register a new revision")
        target.write_bytes(source.read_bytes())
        hashes[str(source.relative_to(ROOT))] = sha(source)
    return hashes


def prepare():
    if (OUTPUT / "manifest.json").exists():
        print("Frozen v2 inputs already prepared", flush=True)
        return
    manifest = {"version": 2, "prepared_at": "2026-09-12", "provider": "DeepInfra", "model": live.MODEL,
                "max_calls": 192, "max_cost_usd": .50, "max_output_tokens_per_request": 5000,
                "max_request_bytes": 131072, "temperature": 0, "cases": [], "paired_cases": [],
                "design": "3 named public development cases then 24 existing anonymous replay cases, each freshly compiled by v7 and v9 from identical frozen simulations; separately 3 fresh entity-aware quick simulations using v9. No outcomes or model judging."}
    old = ROOT / "artifacts/real-scenarios-v1"
    for case in read(old / "manifest.json")["cases"]:
        case_id = case["id"]
        prior = old / "cases" / case_id
        directory = OUTPUT / "cases" / case_id
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("request.json", "source-review.json"):
            shutil.copy2(prior / name, directory / name)
        manifest["cases"].append({"id": case_id, "request_sha256": sha(directory / "request.json")})
        label = "baseline" if case_id == "guotai-haitong-2024" else "role-fix"
        run = prior / label
        values = [read(run / "spec.json"), read(Path(read(run / "timing.json")["job_dir"]) / "simulation-result.json"), read(run / "financial-actions.json")]
        register_pair(manifest, case_id, values, {"cohort": "public_pilot", "prior_run": label})
    for entry in read(ROOT / ".data/benchmarks/product-usability-experiment-v2/manifest.json")["cases"]:
        register_pair(manifest, entry["anonymous_case_id"], inputs(entry), {"cohort": "replay_extension", "anonymization": entry})
    manifest["code_sha256"] = freeze_code()
    write(OUTPUT / "manifest.json", manifest)
    print(json.dumps({"paired_cases": len(manifest["paired_cases"]), "fresh_simulations": len(manifest["cases"]), "max_calls": 192, "max_cost_usd": .50}), flush=True)


def register_pair(manifest, case_id, values, metadata):
    directory = OUTPUT / "paired" / case_id
    hashes = {}
    for name, value in zip(("spec", "simulation-result", "financial-actions"), values):
        write(directory / (name + ".json"), value)
        hashes[name] = sha(directory / (name + ".json"))
    manifest["paired_cases"].append({"id": case_id, "input_sha256": hashes, **metadata})


def paired(stage):
    from dotenv import dotenv_values
    manifest = read(OUTPUT / "manifest.json")
    if freeze_code() != manifest["code_sha256"]:
        raise ValueError("frozen code hash mismatch")
    credentials = dotenv_values(ROOT / ".env")
    if (credentials.get("LLM_BASE_URL") or "").rstrip("/") != "https://api.deepinfra.com/v1/openai" or not credentials.get("LLM_API_KEY"):
        raise ValueError("reviewed provider credentials unavailable")
    relay = live.Relay(credentials["LLM_API_KEY"], manifest)
    selected = [entry for entry in manifest["paired_cases"] if entry["cohort"] == ("public_pilot" if stage == "pilot" else "replay_extension")]
    if stage == "extension":
        gate = read(OUTPUT / "pilot-review.json")
        if not gate.get("expand") or gate["manifest_sha256"] != sha(OUTPUT / "manifest.json"):
            raise ValueError("inspect the paired public pilot before expansion")
    for index, entry in enumerate(selected):
        directory = OUTPUT / "paired" / entry["id"]
        for name, expected in entry["input_sha256"].items():
            if sha(directory / (name + ".json")) != expected:
                raise ValueError("paired input changed")
        spec, result, actions = [read(directory / (name + ".json")) for name in ("spec", "simulation-result", "financial-actions")]
        for variant in (("v7", "v9") if index % 2 == 0 else ("v9", "v7")):
            destination = directory / (variant + ".json")
            if destination.exists():
                print(f"sealed {entry['id']} {variant}: reuse", flush=True)
                continue
            system, user, _ = build_scenario_branch_prompt(spec, result, actions, product=True, product_version=f"scenario-branch-compiler-{variant}")
            if "anonymization" in entry:
                user = json.dumps(anonymize(json.loads(user), entry["anonymization"]), ensure_ascii=False)
                _assert_outbound_prompt_is_anonymous({"system": system, "user": user}, entry["anonymization"])
            write(directory / (variant + "-prompt.json"), {"system": system, "user": user})
            relay.case, relay.run_label = entry["id"], "paired-" + variant
            started, attempts, artifact = time.monotonic(), [], None
            current_system = system
            for attempt in range(2):
                request_id, raw, reserved = relay.reserve({"model": live.MODEL, "temperature": 0, "messages": [{"role": "system", "content": current_system}, {"role": "user", "content": user}], "response_format": {"type": "json_object"}, "max_tokens": 5000})
                began = time.monotonic()
                try:
                    with urlopen(Request(live.ENDPOINT, raw, {"Authorization": "Bearer " + relay.key, "Content-Type": "application/json"}), timeout=175) as response:
                        body, code = response.read(), response.status
                except HTTPError as error:
                    body, code = error.read(), error.code
                except Exception as error:
                    body, code = json.dumps({"error": {"type": type(error).__name__}}).encode(), 502
                relay.complete(request_id, reserved, code, body, time.monotonic() - began)
                record = {"request_id": request_id, "http_status": code}
                try:
                    if code != 200:
                        raise RuntimeError(f"provider HTTP {code}")
                    response = json.loads(body)
                    record["raw_response"] = response["choices"][0]["message"]["content"]
                    artifact = build_scenario_branch_set(record["raw_response"], spec, result, actions, model_id=live.MODEL, product=True, product_version=f"scenario-branch-compiler-{variant}")
                except Exception as error:
                    record["error"] = str(error)
                    current_system, user = build_scenario_branch_retry_prompt(system, user, error)
                attempts.append(record)
                write(directory / (variant + "-attempts.json"), attempts)
                if artifact is not None:
                    break
            output = {"case_id": entry["id"], "variant": variant, "cohort": entry["cohort"], "artifact": artifact,
                      "elapsed_seconds": time.monotonic() - started, "attempt_count": len(attempts), "request_ids": [item["request_id"] for item in attempts]}
            write(destination, output)
            print(f"finished {entry['id']} {variant}: {'valid' if artifact else 'failed'}; {len(artifact['branches']) if artifact else 0} branches; budget=${relay.state['settled_usd']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "extension", "live"))
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            prepare()
        elif args.stage == "live":
            freeze_code()
            os.environ["FEVER_SCENARIO_PRODUCT_VERSION"] = "v9"
            live.execute(SimpleNamespace(label="entity-v9", case=[]))
        else:
            paired(args.stage)


if __name__ == "__main__":
    main()
