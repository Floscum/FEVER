#!/usr/bin/env python3
"""Measure whether raw multi-agent interaction traces improve final scenarios."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fever_mirofish.contracts import canonical_sha256  # noqa: E402
from fever_mirofish.decision_support_eval import (  # noqa: E402
    BRANCH_QUALIFICATION_KEYS,
    QUALITY_SCORE_KEYS,
    build_absolute_quality_judge_prompt,
    deterministic_structure_metrics,
    normalize_multi_agent_scenario_set,
    reconcile_absolute_quality_passes,
    validate_absolute_quality_judgment,
)
from fever_mirofish.scenario_branches import (  # noqa: E402
    PROMPT_VERSION,
    build_scenario_branch_prompt,
    build_scenario_branch_retry_prompt,
    build_scenario_branch_set,
)
from run_compiler_decision_support_diagnostic import (  # noqa: E402
    ModelCaller,
    _with_retry,
)


EXPERIMENT_ID = "decision-support-trace-ablation-v1"
DEFAULT_OUTPUT = ROOT / ".data" / "benchmarks" / EXPERIMENT_ID
SCALE_MANIFEST = (
    ROOT / ".data" / "experiments" / "scenario-coverage-v2" / "manifest.json"
)
SCALE_RUNS = (
    ROOT / ".data" / "experiments" / "scenario-coverage-v2" / "runs.jsonl"
)
ENGINEERING_RUNS = (
    ROOT / ".data" / "experiments" / "actor-scale-v1" / "runs.jsonl"
)
ENGINEERING_REPORT = (
    ROOT / ".data" / "experiments" / "actor-scale-v1" / "report.json"
)
VARIANTS = ("with_traces", "without_traces")
SELECTED_CASE_IDS = (
    "scale_001_seed_cn_603259__9603b4d460",
    "scale_004_seed_cn_sh516650__747b7e18cb",
    "scale_006_seed_cn_300620_event_e2cfab102a",
    "scale_007_seed_cn_sz399005__d09d0bcaf1",
    "scale_009_seed_cn_600508_event_610059cf64",
    "scale_011_seed_cn_sh512980__7d18ffa35e",
    "scale_013_seed_us_blk__ddc3006ab0",
    "scale_016_seed_us_spy_event_47ebaf7f54__dup13",
    "scale_018_seed_us_msft_event_f6f1e4d8cc",
    "scale_019_seed_us_xlre__e2f4713286",
    "scale_021_seed_us_mmm__c9cd6c9787",
    "scale_023_seed_us_lqd__8e86e41e37",
)
PRINT_LOCK = threading.Lock()
AUDIT_LOCK = threading.Lock()


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    with AUDIT_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _progress(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * proportion)))
    return ordered[index]


def _protocol() -> Dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "experiment_id": EXPERIMENT_ID,
        "classification": "outcome_free_component_ablation",
        "independent_unit": "frozen_event_case",
        "selection": {
            "case_ids": list(SELECTED_CASE_IDS),
            "market_balance": {"CN": 6, "US": 6},
            "actor_budget_balance": {"4": 4, "6": 4, "8": 4},
            "event_type_count": 6,
            "cases_per_event_type": 2,
            "selected_using_outcomes": False,
            "selected_using_quality_scores": False,
        },
        "variants": {
            "with_traces": (
                "Current v6 compiler with the frozen autonomous interaction "
                "summaries visible in the prompt."
            ),
            "without_traces": (
                "The identical v6 prompt inputs and branch slots, except the "
                "autonomous interaction summaries are replaced by an empty list."
            ),
        },
        "controls": {
            "same_spec": True,
            "same_simulation_result_for_validation": True,
            "same_financial_decisions": True,
            "same_branch_slots": True,
            "same_model": True,
            "same_four_branch_budget": True,
            "both_variants_regenerated": True,
            "outcomes_read": False,
            "outbound_prompts_anonymized": True,
            "outbound_fields_removed": [
                "original_case_id",
                "gateway_job_id",
                "security_symbols",
                "benchmark_symbols",
                "internal_source_refs",
                "absolute_paths",
                "calendar_dates",
            ],
        },
        "quality": {
            "absolute_single_set_judging": True,
            "independent_repeats": 2,
            "score_keys": list(QUALITY_SCORE_KEYS),
            "branch_qualification_keys": list(BRANCH_QUALIFICATION_KEYS),
            "qualification_rule": "all flags true in both repeats",
        },
        "decision_rule": {
            "reliability_floor": 0.8,
            "material_score_difference": 1.0,
            "with_traces_add_value": (
                "with_traces mean score is at least one point higher and its "
                "strict qualified rate is not lower"
            ),
            "no_material_increment": "absolute mean score difference is below one",
            "raw_traces_may_add_noise": (
                "without_traces mean score is at least one point higher and its "
                "strict qualified rate is not lower"
            ),
            "low_reliability": (
                "if either variant has score-within-one or branch-flag repeat "
                "agreement below 80%, semantic direction is inconclusive"
            ),
        },
        "interpretation_limits": [
            "This isolates trace visibility at compilation time, not the effect of simulation on the post-simulation decisions.",
            "One scenario generation draw is used per variant and case.",
            "The same economical model family generates and judges the scenarios.",
            "No historical outcomes, price labels, forecast calibration, or trading returns are evaluated.",
        ],
    }


def _build_manifest() -> Dict[str, Any]:
    coverage_manifest = _read(SCALE_MANIFEST)
    coverage_entries = {
        item["experiment_case_id"]: item for item in coverage_manifest["cases"]
    }
    coverage_runs = {
        item["experiment_case_id"]: item for item in _read_jsonl(SCALE_RUNS)
    }
    cases = []
    models = set()
    for case_id in SELECTED_CASE_IDS:
        source = coverage_entries[case_id]
        prior_run = coverage_runs[case_id]
        if prior_run["candidate"]["scenario_count"] != 4:
            raise ValueError(f"{case_id} prior v6 artifact did not contain four branches")
        spec_path = Path(source["inputs"]["spec"])
        simulation_path = Path(source["inputs"]["simulation_result"])
        actions_path = Path(source["inputs"]["financial_actions"])
        prior_v6_path = _resolve(prior_run["output"])
        spec = _read(spec_path)
        simulation = _read(simulation_path)
        actions = _read(actions_path)
        prior_v6 = _read(prior_v6_path)
        models.add(prior_v6["model_id"])
        _, user, _ = build_scenario_branch_prompt(spec, simulation, actions)
        prompt_input = json.loads(user)
        cases.append(
            {
                "case_id": case_id,
                "anonymous_case_id": f"anonymous-case-{len(cases) + 1:02d}",
                "market": source["market"],
                "event_type_l2": source["event_type_l2"],
                "actor_budget": source["actor_budget"],
                "spec_path": _relative(spec_path),
                "spec_sha256": canonical_sha256(spec),
                "simulation_result_path": _relative(simulation_path),
                "simulation_result_sha256": canonical_sha256(simulation),
                "financial_actions_path": _relative(actions_path),
                "financial_actions_sha256": canonical_sha256(actions),
                "prior_v6_path": _relative(prior_v6_path),
                "prior_v6_sha256": canonical_sha256(prior_v6),
                "decision_actor_count": len(
                    {item["actor_id"] for item in actions["decisions"]}
                ),
                "simulated_action_count": len(prompt_input["simulated_actions"]),
            }
        )
    if len(models) != 1:
        raise ValueError("selected prior v6 artifacts do not use one model")
    protocol = _protocol()
    return {
        "schema_version": "0.1.0",
        "experiment_id": EXPERIMENT_ID,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "required_model": next(iter(models)),
        "case_count": len(cases),
        "cases": cases,
        "outcome_data_read": False,
    }


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("trace ablation manifest experiment id mismatch")
    protocol = _protocol()
    if manifest.get("protocol") != protocol:
        raise ValueError("trace ablation protocol changed after registration")
    if manifest.get("protocol_sha256") != canonical_sha256(protocol):
        raise ValueError("trace ablation protocol hash mismatch")
    if manifest.get("outcome_data_read") is not False:
        raise ValueError("trace ablation must remain outcome-free")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != len(SELECTED_CASE_IDS):
        raise ValueError("trace ablation case count changed")
    if tuple(item["case_id"] for item in cases) != SELECTED_CASE_IDS:
        raise ValueError("trace ablation selected cases changed")
    if {item["market"] for item in cases} != {"CN", "US"}:
        raise ValueError("trace ablation markets changed")
    for key, expected in (
        ("spec_path", "spec_sha256"),
        ("simulation_result_path", "simulation_result_sha256"),
        ("financial_actions_path", "financial_actions_sha256"),
        ("prior_v6_path", "prior_v6_sha256"),
    ):
        for entry in cases:
            value = _read(_resolve(entry[key]))
            if canonical_sha256(value) != entry[expected]:
                raise ValueError(f"{entry['case_id']} frozen input changed: {key}")
    by_market = {
        market: sum(item["market"] == market for item in cases)
        for market in ("CN", "US")
    }
    by_budget = {
        budget: sum(item["actor_budget"] == budget for item in cases)
        for budget in (4, 6, 8)
    }
    by_type = {
        value: sum(item["event_type_l2"] == value for item in cases)
        for value in {item["event_type_l2"] for item in cases}
    }
    if by_market != {"CN": 6, "US": 6}:
        raise ValueError("trace ablation market balance changed")
    if by_budget != {4: 4, 6: 4, 8: 4}:
        raise ValueError("trace ablation actor budget balance changed")
    if len(by_type) != 6 or set(by_type.values()) != {2}:
        raise ValueError("trace ablation event type balance changed")


def _ensure_manifest(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.exists():
        _write(path, _build_manifest())
    manifest = _read(path)
    _validate_manifest(manifest)
    return manifest


def _scenario_path(
    output_dir: Path, case_id: str, variant: str, *, source: bool = False
) -> Path:
    suffix = "-source" if source else ""
    return output_dir / "scenarios" / case_id / f"{variant}{suffix}.json"


def _judgment_path(
    output_dir: Path, case_id: str, variant: str, pass_number: int
) -> Path:
    return output_dir / "quality" / case_id / f"{variant}-pass-{pass_number}.json"


def _privacy_tokens(entry: Dict[str, Any]) -> set[str]:
    """Collect case-specific identifiers that must never enter a provider prompt."""

    spec = _read(_resolve(entry["spec_path"]))
    actions = _read(_resolve(entry["financial_actions_path"]))
    tokens = {entry["case_id"]}
    question = str(spec.get("question", ""))
    relative_match = re.search(
        r"影响\s+(\S+)\s+相对\s+(\S+)\s+表现", question
    )
    if relative_match:
        tokens.update(relative_match.groups())
    for decision in actions.get("decisions", []):
        tokens.update(
            str(item)
            for item in decision.get("instrument_refs", [])
            if item
        )
    return {item for item in tokens if item}


def _sanitize_text(value: str, entry: Dict[str, Any]) -> str:
    text = value
    replacements = {
        token: f"匿名标的{index}"
        for index, token in enumerate(sorted(_privacy_tokens(entry)), start=1)
    }
    for token, replacement in replacements.items():
        text = re.sub(re.escape(token), replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\b(?:sh|sz)?\d{6}\b", "匿名标的", text)
    text = re.sub(r"\bsimjob_[0-9a-f]+\b", "anonymous-job", text)
    text = re.sub(r"fever://\S+", "anonymized-source", text)
    text = re.sub(r"/Users/\S+", "anonymized-path", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "D0-date", text)
    return text


def _sanitize_prompt_payload(value: Any, entry: Dict[str, Any]) -> Any:
    """Recursively remove local and security identifiers before outbound calls."""

    if isinstance(value, str):
        return _sanitize_text(value, entry)
    if isinstance(value, list):
        return [_sanitize_prompt_payload(item, entry) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized = {}
    for key, item in value.items():
        if key == "case_id":
            sanitized[key] = entry["anonymous_case_id"]
        elif key in {"as_of", "observed_at"}:
            sanitized[key] = "D0"
        elif key == "question":
            sanitized[key] = (
                "严格基于 D0 时点可得信息，推演未来三个交易日"
                "各参与方行动及条件化传导机制。"
            )
        elif key in {"source_ref", "source_url"}:
            sanitized[key] = "anonymized-source"
        elif key == "instrument_refs":
            sanitized[key] = (
                ["ANON_ASSET"] if isinstance(item, list) and item else []
            )
        else:
            sanitized[key] = _sanitize_prompt_payload(item, entry)
    return sanitized


def _assert_outbound_prompt_is_anonymous(
    payload: Dict[str, Any], entry: Dict[str, Any]
) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden = {
        entry["case_id"],
        str(ROOT),
        "fever://",
        "simjob_",
        *(_privacy_tokens(entry)),
    }
    leaked = [item for item in forbidden if item and item.lower() in serialized.lower()]
    if leaked:
        raise ValueError(
            f"{entry['case_id']} outbound prompt contains identifiers: {leaked}"
        )
    if re.search(r"(?i)\b(?:sh|sz)?\d{6}\b", serialized):
        raise ValueError(f"{entry['case_id']} outbound prompt contains a security code")
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", serialized):
        raise ValueError(f"{entry['case_id']} outbound prompt contains a calendar date")


def _generation_job(
    entry: Dict[str, Any],
    variant: str,
    output_dir: Path,
    caller: ModelCaller,
) -> str:
    case_id = entry["case_id"]
    source_path = _scenario_path(output_dir, case_id, variant, source=True)
    normalized_path = _scenario_path(output_dir, case_id, variant)
    spec = _read(_resolve(entry["spec_path"]))
    simulation = _read(_resolve(entry["simulation_result_path"]))
    actions = _read(_resolve(entry["financial_actions_path"]))
    if source_path.exists() and normalized_path.exists():
        existing = _read(normalized_path)
        deterministic_structure_metrics(existing, spec)
        return f"reuse generation {case_id} {variant}"

    system, user, _ = build_scenario_branch_prompt(spec, simulation, actions)
    user_payload = json.loads(user)
    original_trace_count = len(user_payload["simulated_actions"])
    if variant == "without_traces":
        user_payload["simulated_actions"] = []
    outbound_payload = _sanitize_prompt_payload(user_payload, entry)
    _assert_outbound_prompt_is_anonymous(outbound_payload, entry)
    request_user = json.dumps(outbound_payload, ensure_ascii=False)

    def parse(raw: str) -> Dict[str, Any]:
        source = build_scenario_branch_set(
            raw,
            spec,
            simulation,
            actions,
            model_id=caller.model,
        )
        normalize_multi_agent_scenario_set(
            source,
            spec,
            source_sha256=canonical_sha256(source),
        )
        return source

    source, retries, elapsed = _with_retry(
        caller,
        system=system,
        user=request_user,
        max_tokens=5000,
        parser=parse,
        retry_builder=build_scenario_branch_retry_prompt,
    )
    normalized = normalize_multi_agent_scenario_set(
        source,
        spec,
        source_sha256=canonical_sha256(source),
    )
    _write(source_path, source)
    _write(normalized_path, normalized)
    _append_jsonl(
        output_dir / "generation-audit.jsonl",
        {
            "case_id": case_id,
            "variant": variant,
            "status": "sealed",
            "visible_simulated_action_count": (
                original_trace_count if variant == "with_traces" else 0
            ),
            "original_simulated_action_count": original_trace_count,
            "prompt_sha256": canonical_sha256(
                {"system": system, "user": request_user}
            ),
            "semantic_retries": retries,
            "elapsed_seconds": round(elapsed, 6),
            "artifact_sha256": canonical_sha256(source),
        },
    )
    return f"generated {case_id} {variant} ({elapsed:.1f}s)"


def _quality_job(
    entry: Dict[str, Any],
    variant: str,
    pass_number: int,
    output_dir: Path,
    caller: ModelCaller,
) -> str:
    case_id = entry["case_id"]
    path = _judgment_path(output_dir, case_id, variant, pass_number)
    if path.exists():
        value = _read(path)
        if value.get("case_id") != case_id or value.get("variant") != variant:
            raise ValueError(f"{case_id} existing quality metadata mismatch")
        return f"reuse quality {case_id} {variant} pass={pass_number}"
    spec = _read(_resolve(entry["spec_path"]))
    scenario = _read(_scenario_path(output_dir, case_id, variant))
    deterministic_structure_metrics(scenario, spec)
    system, user = build_absolute_quality_judge_prompt(spec, scenario)
    outbound_payload = _sanitize_prompt_payload(json.loads(user), entry)
    _assert_outbound_prompt_is_anonymous(outbound_payload, entry)
    user = json.dumps(outbound_payload, ensure_ascii=False)
    judgment, retries, elapsed = _with_retry(
        caller,
        system=system,
        user=user,
        max_tokens=3500,
        parser=lambda raw: validate_absolute_quality_judgment(
            raw,
            case_id=case_id,
            variant=variant,
        ),
    )
    _write(path, judgment)
    _append_jsonl(
        output_dir / "judging-audit.jsonl",
        {
            "kind": "quality",
            "case_id": case_id,
            "variant": variant,
            "pass": pass_number,
            "semantic_retries": retries,
            "elapsed_seconds": round(elapsed, 6),
        },
    )
    return f"quality {case_id} {variant} pass={pass_number} ({elapsed:.1f}s)"


def _parallel(
    jobs: list[Any], worker: Callable[[Any], str], workers: int
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            _progress(future.result())


def _run_generation(
    manifest: Dict[str, Any],
    output_dir: Path,
    caller: ModelCaller,
    workers: int,
) -> None:
    jobs = [
        (entry, variant)
        for entry in manifest["cases"]
        for variant in VARIANTS
    ]
    jobs.sort(
        key=lambda item: canonical_sha256(
            {"experiment_id": EXPERIMENT_ID, "case_id": item[0]["case_id"], "variant": item[1]}
        )
    )
    _parallel(
        jobs,
        lambda job: _generation_job(job[0], job[1], output_dir, caller),
        workers,
    )


def _run_quality(
    manifest: Dict[str, Any],
    output_dir: Path,
    caller: ModelCaller,
    workers: int,
) -> None:
    for entry in manifest["cases"]:
        for variant in VARIANTS:
            if not _scenario_path(output_dir, entry["case_id"], variant).exists():
                raise RuntimeError("run trace ablation generation before quality")
    jobs = [
        (entry, variant, pass_number)
        for entry in manifest["cases"]
        for variant in VARIANTS
        for pass_number in (1, 2)
    ]
    jobs.sort(
        key=lambda item: canonical_sha256(
            {
                "experiment_id": EXPERIMENT_ID,
                "case_id": item[0]["case_id"],
                "variant": item[1],
                "pass": item[2],
            }
        )
    )
    _parallel(
        jobs,
        lambda job: _quality_job(
            job[0], job[1], job[2], output_dir, caller
        ),
        workers,
    )


def _decision_actor_coverage(
    scenario: Dict[str, Any], entry: Dict[str, Any]
) -> float:
    covered = {
        actor_id
        for branch in scenario["branches"]
        for actor_id in branch["actor_ids"]
    }
    denominator = entry["decision_actor_count"]
    return len(covered) / denominator if denominator else 0.0


def _usage_summary(output_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(output_dir / "llm-usage.jsonl")
    completed = [item for item in rows if item.get("status") == "completed"]
    failed = [item for item in rows if item.get("status") != "completed"]
    costs = [
        float(item["estimated_cost"])
        for item in completed
        if isinstance(item.get("estimated_cost"), (int, float))
    ]
    generation_calls = sum(
        1 + int(item.get("semantic_retries", 0))
        for item in _read_jsonl(output_dir / "generation-audit.jsonl")
    )
    quality_calls = sum(
        1 + int(item.get("semantic_retries", 0))
        for item in _read_jsonl(output_dir / "judging-audit.jsonl")
    )
    unsealed_completed_calls = max(
        0, len(completed) - generation_calls - quality_calls
    )
    return {
        "attempt_count": len(rows),
        "completed_call_count": len(completed),
        "failed_call_count": len(failed),
        "total_tokens": sum(int(item.get("total_tokens", 0)) for item in completed),
        "estimated_cost": round(sum(costs), 8) if costs else None,
        "stage_breakdown": {
            "sealed_scenario_generation_calls": generation_calls,
            "absolute_quality_calls": quality_calls,
            "completed_calls_without_sealed_audit_record": (
                unsealed_completed_calls
            ),
        },
        "clean_rerun_expected_call_count_without_semantic_retries": (
            len(SELECTED_CASE_IDS) * len(VARIANTS) * 3
        ),
    }


def _trace_input_diagnostic(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the frozen traces without assigning them semantic quality.

    This is deliberately post-hoc.  Tool/navigation actions can still carry a
    weak behavioural signal, so the report calls them low-context rather than
    useless.
    """

    low_context_types = {
        "SEARCH_POSTS",
        "SEARCH_USER",
        "LIKE_COMMENT",
        "FOLLOW",
        "DO_NOTHING",
    }
    action_type_counts: Dict[str, int] = {}
    total = 0
    low_context = 0
    social_endorsements = 0
    communications = 0
    summary_characters = 0
    within_case_exact_repeats = 0
    for entry in manifest["cases"]:
        spec = _read(_resolve(entry["spec_path"]))
        simulation = _read(_resolve(entry["simulation_result_path"]))
        actions = _read(_resolve(entry["financial_actions_path"]))
        _, user, _ = build_scenario_branch_prompt(spec, simulation, actions)
        traces = json.loads(user).get("simulated_actions", [])
        seen_summaries = set()
        for trace in traces:
            action_type = str(trace.get("action_type", "UNKNOWN"))
            summary = re.sub(
                r"\s+", " ", str(trace.get("summary", "")).strip().lower()
            )
            action_type_counts[action_type] = (
                action_type_counts.get(action_type, 0) + 1
            )
            total += 1
            low_context += action_type in low_context_types
            social_endorsements += action_type == "SOCIAL_ENDORSEMENT"
            communications += action_type == "COMMUNICATE"
            summary_characters += len(summary)
            if summary in seen_summaries:
                within_case_exact_repeats += 1
            else:
                seen_summaries.add(summary)
    denominator = total or 1
    case_count = len(manifest["cases"]) or 1
    return {
        "post_hoc": True,
        "trace_count": total,
        "mean_trace_count_per_case": round(total / case_count, 6),
        "action_type_counts": dict(sorted(action_type_counts.items())),
        "low_context_operational_trace_count": low_context,
        "low_context_operational_trace_rate": round(
            low_context / denominator, 8
        ),
        "social_endorsement_count": social_endorsements,
        "social_endorsement_rate": round(
            social_endorsements / denominator, 8
        ),
        "communication_count": communications,
        "communication_rate": round(communications / denominator, 8),
        "within_case_exact_repeat_count": within_case_exact_repeats,
        "within_case_exact_repeat_rate": round(
            within_case_exact_repeats / denominator, 8
        ),
        "summary_character_count": summary_characters,
        "mean_summary_characters_per_case": round(
            summary_characters / case_count, 6
        ),
        "interpretation": (
            "This composition check was added after the semantic result. "
            "It describes prompt volume and repetition; it does not prove "
            "that any action type is useless or caused the score difference."
        ),
    }


def _engineering_scale_summary() -> Dict[str, Any]:
    rows = _read_jsonl(ENGINEERING_RUNS)
    report = _read(ENGINEERING_REPORT)
    costs = report["usage"]["simulation_by_actor_budget"]
    by_budget = {}
    for budget in (4, 6, 8):
        group = [item for item in rows if item["actor_budget"] == budget]
        by_budget[str(budget)] = {
            "run_count": len(group),
            "completion_rate": sum(item["status"] == "completed" for item in group)
            / len(group),
            "active_actor_rate": _safe_mean(
                [float(item["active_actor_ratio"]) for item in group]
            ),
            "valid_decision_rate": _safe_mean(
                [float(item["valid_decision_ratio"]) for item in group]
            ),
            "mean_autonomous_action_count": round(
                _safe_mean([float(item["autonomous_action_count"]) for item in group]),
                6,
            ),
            "median_wall_seconds": round(
                statistics.median([float(item["wall_seconds"]) for item in group]),
                6,
            ),
            "p90_wall_seconds": round(
                _percentile([float(item["wall_seconds"]) for item in group], 0.9),
                6,
            ),
            "mean_completed_calls": costs[str(budget)]["mean_completed_calls"],
            "mean_total_tokens": costs[str(budget)]["mean_total_tokens"],
            "mean_estimated_cost": costs[str(budget)]["mean_estimated_cost"],
        }
    four = by_budget["4"]
    eight = by_budget["8"]
    return {
        "source_run_count": len(rows),
        "all_completed": all(item["status"] == "completed" for item in rows),
        "by_actor_budget": by_budget,
        "four_to_eight": {
            "action_multiplier": round(
                eight["mean_autonomous_action_count"]
                / four["mean_autonomous_action_count"],
                6,
            ),
            "cost_multiplier": round(
                eight["mean_estimated_cost"] / four["mean_estimated_cost"],
                6,
            ),
            "median_latency_multiplier": round(
                eight["median_wall_seconds"] / four["median_wall_seconds"],
                6,
            ),
        },
    }


def _aggregate_variant(
    variant: str,
    case_results: list[Dict[str, Any]],
    generation_rows: list[Dict[str, Any]],
) -> Dict[str, Any]:
    quality = [item["variants"][variant]["quality"] for item in case_results]
    structures = [
        item["variants"][variant]["structure"] for item in case_results
    ]
    score_comparisons = sum(
        item["reliability"]["score_comparison_count"] for item in quality
    )
    flag_comparisons = sum(
        item["reliability"]["branch_flag_comparison_count"] for item in quality
    )
    elapsed = [
        float(item["elapsed_seconds"])
        for item in generation_rows
        if item["variant"] == variant
    ]
    branch_count = sum(item["branch_count"] for item in structures)
    qualified = sum(item["qualified_branch_count"] for item in quality)
    return {
        "mean_quality_score_out_of_20": round(
            _safe_mean([item["total_score"] for item in quality]), 6
        ),
        "strict_qualified_branch_rate": round(
            qualified / branch_count if branch_count else 0.0, 8
        ),
        "decision_actor_coverage": round(
            _safe_mean(
                [item["decision_actor_coverage"] for item in structures]
            ),
            8,
        ),
        "multi_actor_branch_rate": round(
            _safe_mean([item["multi_actor_branch_rate"] for item in structures]),
            8,
        ),
        "trigger_present_rate": round(
            _safe_mean([item["trigger_present_rate"] for item in structures]), 8
        ),
        "invalidation_present_rate": round(
            _safe_mean(
                [item["invalidation_present_rate"] for item in structures]
            ),
            8,
        ),
        "valid_evidence_ref_rate": round(
            _safe_mean(
                [item["valid_evidence_ref_rate"] for item in structures]
            ),
            8,
        ),
        "quality_repeat_reliability": {
            "score_within_one_rate": round(
                sum(
                    item["reliability"]["score_within_one_count"]
                    for item in quality
                )
                / score_comparisons,
                8,
            ),
            "branch_flag_agreement_rate": round(
                sum(
                    item["reliability"]["branch_flag_agreement_count"]
                    for item in quality
                )
                / flag_comparisons,
                8,
            ),
        },
        "generation_latency_seconds": {
            "count": len(elapsed),
            "median": round(statistics.median(elapsed), 6),
            "p90_observed": round(_percentile(elapsed, 0.9), 6),
            "maximum": round(max(elapsed), 6),
        },
    }


def _semantic_verdict(variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    with_traces = variants["with_traces"]
    without = variants["without_traces"]
    reliability_floor = min(
        with_traces["quality_repeat_reliability"]["score_within_one_rate"],
        with_traces["quality_repeat_reliability"]["branch_flag_agreement_rate"],
        without["quality_repeat_reliability"]["score_within_one_rate"],
        without["quality_repeat_reliability"]["branch_flag_agreement_rate"],
    )
    delta = (
        with_traces["mean_quality_score_out_of_20"]
        - without["mean_quality_score_out_of_20"]
    )
    if reliability_floor < 0.8:
        verdict = "inconclusive_low_repeat_agreement"
    elif (
        delta >= 1.0
        and with_traces["strict_qualified_branch_rate"]
        >= without["strict_qualified_branch_rate"]
    ):
        verdict = "with_traces_add_value"
    elif (
        delta <= -1.0
        and without["strict_qualified_branch_rate"]
        >= with_traces["strict_qualified_branch_rate"]
    ):
        verdict = "raw_traces_may_add_noise"
    elif abs(delta) < 1.0:
        verdict = "no_material_trace_increment"
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "mean_quality_delta_with_minus_without": round(delta, 6),
        "minimum_repeat_reliability": round(reliability_floor, 8),
        "decision_rule_was_preregistered": True,
    }


def _post_hoc_subgroups(
    case_results: list[Dict[str, Any]],
) -> Dict[str, Any]:
    dimensions = ("market", "actor_budget", "event_type_l2")
    output: Dict[str, Any] = {
        "post_hoc": True,
        "interpretation": (
            "Each event-type subgroup has only two cases. These estimates are "
            "hypothesis-generating and must not be treated as routing evidence."
        ),
    }
    for dimension in dimensions:
        grouped: Dict[str, list[Dict[str, Any]]] = {}
        for item in case_results:
            grouped.setdefault(str(item[dimension]), []).append(item)
        output[dimension] = {
            key: {
                "case_count": len(items),
                "mean_quality_delta_with_minus_without": round(
                    _safe_mean(
                        [
                            float(
                                item[
                                    "quality_delta_with_minus_without"
                                ]
                            )
                            for item in items
                        ]
                    ),
                    6,
                ),
                "with_traces_win_count": sum(
                    item["quality_winner"] == "with_traces"
                    for item in items
                ),
                "without_traces_win_count": sum(
                    item["quality_winner"] == "without_traces"
                    for item in items
                ),
                "tie_count": sum(
                    item["quality_winner"] == "tie" for item in items
                ),
            }
            for key, items in sorted(grouped.items())
        }
    return output


def _build_report(
    manifest: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    case_results = []
    for entry in manifest["cases"]:
        variants = {}
        for variant in VARIANTS:
            spec = _read(_resolve(entry["spec_path"]))
            scenario = _read(_scenario_path(output_dir, entry["case_id"], variant))
            structure = deterministic_structure_metrics(scenario, spec)
            structure["decision_actor_coverage"] = _decision_actor_coverage(
                scenario, entry
            )
            quality = reconcile_absolute_quality_passes(
                _read(_judgment_path(output_dir, entry["case_id"], variant, 1)),
                _read(_judgment_path(output_dir, entry["case_id"], variant, 2)),
            )
            variants[variant] = {"structure": structure, "quality": quality}
        delta = (
            variants["with_traces"]["quality"]["total_score"]
            - variants["without_traces"]["quality"]["total_score"]
        )
        winner = "tie"
        if delta > 0:
            winner = "with_traces"
        elif delta < 0:
            winner = "without_traces"
        case_results.append(
            {
                "case_id": entry["case_id"],
                "market": entry["market"],
                "event_type_l2": entry["event_type_l2"],
                "actor_budget": entry["actor_budget"],
                "simulated_action_count": entry["simulated_action_count"],
                "variants": variants,
                "quality_delta_with_minus_without": delta,
                "quality_winner": winner,
            }
        )
    generation_rows = _read_jsonl(output_dir / "generation-audit.jsonl")
    variants = {
        variant: _aggregate_variant(variant, case_results, generation_rows)
        for variant in VARIANTS
    }
    paired = {
        value: sum(item["quality_winner"] == value for item in case_results)
        for value in ("with_traces", "without_traces", "tie")
    }
    return {
        "schema_version": "0.1.0",
        "experiment_id": EXPERIMENT_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "outcome_free_component_ablation",
        "case_count": len(case_results),
        "variants": variants,
        "paired_quality": {
            "with_traces_win_count": paired["with_traces"],
            "without_traces_win_count": paired["without_traces"],
            "tie_count": paired["tie"],
        },
        "semantic_verdict": _semantic_verdict(variants),
        "post_hoc_trace_diagnostic": _trace_input_diagnostic(manifest),
        "post_hoc_subgroups": _post_hoc_subgroups(case_results),
        "engineering_scale": _engineering_scale_summary(),
        "usage": _usage_summary(output_dir),
        "case_results": case_results,
        "interpretation_limits": manifest["protocol"]["interpretation_limits"],
    }


def _write_markdown(report: Dict[str, Any], output_dir: Path) -> None:
    with_traces = report["variants"]["with_traces"]
    without = report["variants"]["without_traces"]
    verdict = report["semantic_verdict"]
    paired = report["paired_quality"]
    scale = report["engineering_scale"]
    usage = report["usage"]
    trace_diagnostic = report["post_hoc_trace_diagnostic"]
    subgroups = report["post_hoc_subgroups"]
    verdict_text = {
        "inconclusive_low_repeat_agreement": "评审重复一致性不足，互动文本的语义增量暂不能确认。",
        "with_traces_add_value": "保留互动文本达到预先设定的增量标准。",
        "no_material_trace_increment": "未看到互动文本带来实质质量增量。",
        "raw_traces_may_add_noise": "隐藏原始互动后反而更好，互动文本可能需先筛选。",
        "mixed": "质量分和严格合格率方向不一致，结果混合。",
    }[verdict["verdict"]]
    lines = [
        "# 互动轨迹增量消融：12 案例",
        "",
        "## 一句话结论",
        "",
        verdict_text,
        "",
        (
            f"保留互动的质量均分为 {with_traces['mean_quality_score_out_of_20']:.2f}/20，"
            f"隐藏互动为 {without['mean_quality_score_out_of_20']:.2f}/20，差值 "
            f"{verdict['mean_quality_delta_with_minus_without']:+.2f}。"
        ),
        "",
        "## 消融设计",
        "",
        "- 12 个案例：CN/US 各 6；4/6/8 参与方各 4；六类事件各 2。",
        "- 两组使用相同事实、参与方、金融决策、分支槽位、模型和四分支预算。",
        "- 唯一区别是编译提示是否看到多智能体自主互动文本。",
        "- 不读取后验结果；每组独立绝对评分两次。",
        "",
        "## 主要数字",
        "",
        "| 指标 | 保留互动 | 隐藏互动 |",
        "|---|---:|---:|",
        f"| 有效决策角色覆盖 | {_pct(with_traces['decision_actor_coverage'])} | {_pct(without['decision_actor_coverage'])} |",
        f"| 质量均分（满分20） | {with_traces['mean_quality_score_out_of_20']:.2f} | {without['mean_quality_score_out_of_20']:.2f} |",
        f"| 严格合格分支 | {_pct(with_traces['strict_qualified_branch_rate'])} | {_pct(without['strict_qualified_branch_rate'])} |",
        f"| 可观察触发存在 | {_pct(with_traces['trigger_present_rate'])} | {_pct(without['trigger_present_rate'])} |",
        f"| 失效条件存在 | {_pct(with_traces['invalidation_present_rate'])} | {_pct(without['invalidation_present_rate'])} |",
        f"| 合法证据引用 | {_pct(with_traces['valid_evidence_ref_rate'])} | {_pct(without['valid_evidence_ref_rate'])} |",
        "",
        (
            f"逐案例质量分：保留互动胜 {paired['with_traces_win_count']}，"
            f"隐藏互动胜 {paired['without_traces_win_count']}，平 {paired['tie_count']}。"
        ),
        "",
        "## 评审稳定性",
        "",
        "| 指标 | 保留互动 | 隐藏互动 |",
        "|---|---:|---:|",
        f"| 分数差 ≤1 的重复一致率 | {_pct(with_traces['quality_repeat_reliability']['score_within_one_rate'])} | {_pct(without['quality_repeat_reliability']['score_within_one_rate'])} |",
        f"| 分支布尔标签重复一致率 | {_pct(with_traces['quality_repeat_reliability']['branch_flag_agreement_rate'])} | {_pct(without['quality_repeat_reliability']['branch_flag_agreement_rate'])} |",
        "",
        f"预先设定的一致性护栏为 80%；本轮最低一致率为 {_pct(verdict['minimum_repeat_reliability'])}。",
        "",
        "## 为什么原始互动可能没有帮上忙（后验诊断）",
        "",
        (
            f"12 个案例共有 {trace_diagnostic['trace_count']} 条互动记录，"
            f"平均每例 {trace_diagnostic['mean_trace_count_per_case']:.2f} 条。"
        ),
        "",
        (
            f"其中 {trace_diagnostic['low_context_operational_trace_count']} 条"
            f"（{_pct(trace_diagnostic['low_context_operational_trace_rate'])}）只是搜索、"
            "点赞、关注或无动作等低上下文操作；"
            f"{trace_diagnostic['within_case_exact_repeat_count']} 条"
            f"（{_pct(trace_diagnostic['within_case_exact_repeat_rate'])}）在同一案例内"
            "与前文完全重复。"
        ),
        "",
        (
            "这说明把整段互动原样交给下游，会同时带入不少噪声。"
            "这是结果出来后增加的诊断，只能解释一个可能原因，不能证明这些操作没有价值。"
        ),
        "",
        "### 事件类型方向（每类仅 2 例）",
        "",
        "| 事件类型 | 保留互动－隐藏互动质量分 |",
        "|---|---:|",
    ]
    for event_type, item in subgroups["event_type_l2"].items():
        lines.append(
            f"| {event_type} | {item['mean_quality_delta_with_minus_without']:+.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "并购类的点估计略偏正，而数据发布类多数偏负；每类只有 2 例且评审稳定性不足，"
                "这里只能用来提出“复杂博弈事件优先启用完整互动”的下一轮假设。"
            ),
            "",
            "## 4/6/8 参与方工程尺度",
            "",
            "| 参与方 | 运行数 | 活跃率 | 有效决策率 | 平均自主行动 | 中位耗时 | 平均成本 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in ("4", "6", "8"):
        item = scale["by_actor_budget"][budget]
        lines.append(
            f"| {budget} | {item['run_count']} | {_pct(item['active_actor_rate'])} | "
            f"{_pct(item['valid_decision_rate'])} | {item['mean_autonomous_action_count']:.2f} | "
            f"{item['median_wall_seconds']:.1f}s | ${item['mean_estimated_cost']:.4f} |"
        )
    multiplier = scale["four_to_eight"]
    lines.extend(
        [
            "",
            (
                f"4→8 参与方时，平均自主行动增至 {multiplier['action_multiplier']:.2f} 倍，"
                f"估算成本增至 {multiplier['cost_multiplier']:.2f} 倍，而中位耗时为 "
                f"{multiplier['median_latency_multiplier']:.2f} 倍。"
            ),
            "",
            "## 开销",
            "",
            f"- 完成调用 {usage['completed_call_count']}，失败 {usage['failed_call_count']}，tokens {usage['total_tokens']:,}。",
            f"- 供应商估算费用：{usage['estimated_cost'] if usage['estimated_cost'] is not None else '未提供'}。",
            (
                f"- 情景生成封存链路 {usage['stage_breakdown']['sealed_scenario_generation_calls']} 次，"
                f"质量评审 {usage['stage_breakdown']['absolute_quality_calls']} 次；"
                f"另有 {usage['stage_breakdown']['completed_calls_without_sealed_audit_record']} 次"
                "返回内容未通过结构校验，未封存为实验结果。"
            ),
            f"- 无语义重试时，干净复现预计 {usage['clean_rerun_expected_call_count_without_semantic_retries']} 次调用。",
            "",
            "## 解释边界",
            "",
            "- 本实验只测试“编译情景时是否看到原始互动文本”，没有移除互动之后形成的结构化角色决策。",
            "- 每个案例、每个版本只生成一次情景，仍可能受单次模型采样影响。",
            "- 生成和评审使用同一廉价模型家族，评审并非人工金标准。",
            "- 本轮不读取历史结果、价格标签，也不计算预测校准或交易收益。",
            "",
            "## 复现",
            "",
            "```bash",
            "python3 scripts/run_trace_ablation_experiment.py --stage report",
            "```",
            "",
            "该命令只汇总已封存产物，不产生模型费用。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_svg(report: Dict[str, Any], output_dir: Path) -> None:
    with_traces = report["variants"]["with_traces"]
    without = report["variants"]["without_traces"]
    metrics = [
        (
            "质量均分（满分 20）",
            with_traces["mean_quality_score_out_of_20"],
            without["mean_quality_score_out_of_20"],
            20.0,
            lambda value: f"{value:.2f}",
        ),
        (
            "严格合格分支",
            with_traces["strict_qualified_branch_rate"],
            without["strict_qualified_branch_rate"],
            1.0,
            _pct,
        ),
        (
            "有效决策角色覆盖",
            with_traces["decision_actor_coverage"],
            without["decision_actor_coverage"],
            1.0,
            _pct,
        ),
    ]
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">',
        '<title id="title">原始互动轨迹增量消融结果</title>',
        '<desc id="desc">12 个均衡案例中，对比保留与隐藏多智能体互动轨迹后的情景质量、严格合格率和角色覆盖。</desc>',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#172033}",
        ".title{font-size:30px;font-weight:600}.subtitle{font-size:17px;fill:#596579}.panel{font-size:20px;font-weight:600}",
        ".label{font-size:16px;fill:#374151}.value{font-size:17px;font-weight:600}.track{fill:#e8edf3}.note{font-size:15px;fill:#596579}",
        "</style>",
        '<rect width="1200" height="720" fill="#ffffff"/>',
        '<text x="64" y="60" class="title">互动文本是否真的带来增量？</text>',
        '<text x="64" y="92" class="subtitle">12 案例 · CN/US 均衡 · 4/6/8 参与方均衡 · 不读取市场结果</text>',
    ]
    for index, (label, first, second, maximum, formatter) in enumerate(metrics):
        top = 146 + index * 158
        first_width = 760 * max(0.0, min(first / maximum, 1.0))
        second_width = 760 * max(0.0, min(second / maximum, 1.0))
        pieces.extend(
            [
                f'<text x="64" y="{top}" class="panel">{label}</text>',
                f'<text x="88" y="{top + 50}" class="label">保留互动</text>',
                f'<rect x="220" y="{top + 30}" width="760" height="22" rx="3" class="track"/>',
                f'<rect x="220" y="{top + 30}" width="{first_width:.2f}" height="22" rx="3" fill="#2563eb"/>',
                f'<text x="998" y="{top + 48}" class="value">{formatter(first)}</text>',
                f'<text x="88" y="{top + 96}" class="label">隐藏互动</text>',
                f'<rect x="220" y="{top + 76}" width="760" height="22" rx="3" class="track"/>',
                f'<rect x="220" y="{top + 76}" width="{second_width:.2f}" height="22" rx="3" fill="#64748b"/>',
                f'<text x="998" y="{top + 94}" class="value">{formatter(second)}</text>',
            ]
        )
    verdict = report["semantic_verdict"]
    pieces.extend(
        [
            '<line x1="64" y1="650" x2="1136" y2="650" stroke="#d5dce5"/>',
            f'<text x="64" y="684" class="note">语义判断：{verdict["verdict"]}；最低重复一致率 {_pct(verdict["minimum_repeat_reliability"])}，预设护栏 80%。</text>',
            "</svg>",
        ]
    )
    (output_dir / "results.svg").write_text("\n".join(pieces), encoding="utf-8")


def _run_report(manifest: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    report = _build_report(manifest, output_dir)
    _write(output_dir / "report.json", report)
    _write_markdown(report, output_dir)
    _write_svg(report, output_dir)
    return report


def _plan(manifest: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    generation = sum(
        not _scenario_path(output_dir, entry["case_id"], variant).exists()
        for entry in manifest["cases"]
        for variant in VARIANTS
    )
    quality = sum(
        not _judgment_path(
            output_dir, entry["case_id"], variant, pass_number
        ).exists()
        for entry in manifest["cases"]
        for variant in VARIANTS
        for pass_number in (1, 2)
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "case_count": manifest["case_count"],
        "missing_billable_calls": {
            "scenario_generation": generation,
            "absolute_quality": quality,
            "total": generation + quality,
        },
        "outcomes_opened_by_plan": False,
        "output_dir": _relative(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("manifest", "plan", "generate", "quality", "report", "all"),
        default="plan",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-billable", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    output_dir = args.output_dir.resolve()
    manifest = _ensure_manifest(output_dir)
    if args.stage == "manifest":
        print(
            json.dumps(
                {
                    "manifest": _relative(output_dir / "manifest.json"),
                    "case_count": manifest["case_count"],
                    "protocol_sha256": manifest["protocol_sha256"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    plan = _plan(manifest, output_dir)
    if args.stage == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.stage in {"generate", "quality", "all"} and not args.allow_billable:
        print(
            json.dumps(
                {
                    **plan,
                    "billable": False,
                    "next_action": "rerun with --allow-billable",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    caller = None
    if args.stage in {"generate", "quality", "all"}:
        caller = ModelCaller(output_dir, manifest["required_model"])
    if args.stage in {"generate", "all"}:
        assert caller is not None
        _run_generation(manifest, output_dir, caller, args.workers)
    if args.stage in {"quality", "all"}:
        assert caller is not None
        _run_quality(manifest, output_dir, caller, args.workers)
    if args.stage in {"report", "all"}:
        report = _run_report(manifest, output_dir)
        print(
            json.dumps(
                {
                    "report": _relative(output_dir / "report.json"),
                    "semantic_verdict": report["semantic_verdict"],
                    "variants": report["variants"],
                    "paired_quality": report["paired_quality"],
                    "engineering_scale": report["engineering_scale"],
                    "usage": report["usage"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
