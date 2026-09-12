"""Outcome-free execution planning for the multi-seed formal experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .contracts import _require, canonical_sha256, validate_spec
from .probability_ensemble import AGGREGATION_VERSION
from .randomness import resolve_replication_seeds


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _command(
    command_id: str,
    stage: str,
    argv: list[str],
    *,
    prerequisites: list[str],
    billable: bool,
    output_paths: list[Path],
    env: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    return {
        "command_id": command_id,
        "stage": stage,
        "prerequisites": prerequisites,
        "billable": billable,
        "env": env or {"PYTHONPATH": "src"},
        "argv": argv,
        "output_paths": [str(path) for path in output_paths],
    }


def _actor_sets(
    spec: Dict[str, Any],
    case_id: str,
    registered: Dict[str, Any],
) -> Dict[str, list[str]]:
    available = [actor["id"] for actor in spec["actors"]]
    available_set = set(available)
    entry = registered.get(case_id)
    _require(isinstance(entry, dict), f"{case_id} actor set is missing")
    core = entry.get("core")
    _require(
        isinstance(core, list)
        and 5 <= len(core) <= 7
        and len(core) == len(set(core))
        and set(core) <= available_set,
        f"{case_id} core actor set must contain 5 to 7 unique spec actors",
    )
    normalized = {
        "core": [actor_id for actor_id in available if actor_id in set(core)]
    }
    expanded = entry.get("expanded")
    if expanded is not None:
        _require(
            isinstance(expanded, list)
            and 8 <= len(expanded) <= 10
            and len(expanded) == len(set(expanded))
            and set(expanded) <= available_set
            and set(core) < set(expanded),
            f"{case_id} expanded actor set must be a strict 8 to 10 actor superset",
        )
        normalized["expanded"] = [
            actor_id for actor_id in available if actor_id in set(expanded)
        ]
    return normalized


def build_formal_execution_plan(
    manifest: Dict[str, Any],
    *,
    run_root: Path = Path(".data/formal-runs"),
    scenario_root: Path = Path(".data/formal-scenarios"),
    probability_root: Path = Path(".data/probability-calibration"),
    benchmark_root: Path = Path(".data/benchmarks"),
) -> Dict[str, Any]:
    """Build commands and paths without opening any registered outcome file."""

    _require(
        manifest.get("schema_version") == "0.1.0",
        "unsupported formal benchmark schema_version",
    )
    _require(
        manifest.get("status") == "input_frozen",
        "formal benchmark inputs must be frozen before planning",
    )
    benchmark_id = manifest.get("benchmark_id")
    _require(
        isinstance(benchmark_id, str) and benchmark_id,
        "formal benchmark_id is required",
    )
    cases = manifest.get("cases")
    budget = manifest.get("pilot_budget") or {}
    _require(isinstance(cases, list) and cases, "formal cases are required")
    _require(
        len(cases) == budget.get("formal_case_count"),
        "formal case count does not match frozen budget",
    )
    case_ids = [entry.get("case_id") for entry in cases]
    _require(
        len(case_ids) == len(set(case_ids)),
        "formal case ids must be unique",
    )
    _require(
        budget.get("core_replications_per_case") == 3,
        "formal core requires exactly three replications per case",
    )
    _require(
        budget.get("signal_aggregation_version") == AGGREGATION_VERSION,
        "formal signal aggregation version is not frozen",
    )
    b3_arm = next(
        (arm for arm in manifest.get("arms", []) if arm.get("id") == "B3"),
        None,
    )
    _require(
        b3_arm is not None
        and b3_arm.get("pilot_replications_per_case") == 3,
        "formal B3 arm must register three replications per case",
    )
    policy_path = Path(budget.get("calibration_policy_path", ""))
    _require(policy_path.is_file(), "frozen calibration policy is missing")
    _require(
        canonical_sha256(_read(policy_path))
        == budget.get("calibration_policy_sha256"),
        "frozen calibration policy hash does not match",
    )
    actor_set_registry = budget.get("actor_sets")
    _require(
        isinstance(actor_set_registry, dict),
        "formal actor sets must be frozen",
    )
    expected_ablation_cases = budget.get("expanded_ablation_case_count")
    _require(
        isinstance(expected_ablation_cases, int)
        and not isinstance(expected_ablation_cases, bool)
        and expected_ablation_cases >= 0,
        "expanded ablation case count must be frozen",
    )

    benchmark_dir = benchmark_root / benchmark_id
    generation_audit = benchmark_dir / "generation-audit.jsonl"
    common_model_env = {
        "PYTHONPATH": "src",
        "FEVER_BENCHMARK_MAX_LLM_CALLS": str(
            budget.get(
                "forecast_transport_attempt_ceiling",
                budget["new_b1_forecast_aggregation_calls"]
                * (1 + budget.get("maximum_transport_retries_per_call", 0)),
            )
        ),
        "FEVER_BENCHMARK_MAX_TOTAL_TOKENS": str(
            budget["forecast_total_token_ceiling"]
        ),
        "FEVER_SCENARIO_MAX_LLM_CALLS": str(
            budget.get(
                "scenario_transport_attempt_ceiling",
                budget["new_scenario_compilation_calls"]
                * (1 + budget.get("maximum_transport_retries_per_call", 0)),
            )
        ),
        "FEVER_SCENARIO_MAX_TOTAL_TOKENS": str(
            budget["scenario_total_token_ceiling"]
        ),
        "FEVER_PROBABILITY_SIGNAL_MAX_LLM_CALLS": str(
            budget.get(
                "probability_signal_transport_attempt_ceiling",
                budget["new_probability_signal_extraction_calls"]
                * (1 + budget.get("maximum_transport_retries_per_call", 0)),
            )
        ),
        "FEVER_PROBABILITY_SIGNAL_MAX_TOTAL_TOKENS": str(
            budget["probability_signal_total_token_ceiling"]
        ),
    }
    commands = []
    case_plans = []
    expanded_case_count = 0
    total_replications = 0

    for entry in cases:
        case_id = entry["case_id"]
        spec_path = Path(entry["spec_path"])
        spec = _read(spec_path)
        validate_spec(spec)
        _require(spec["case_id"] == case_id, f"{case_id} spec id mismatch")
        _require(
            canonical_sha256(spec) == entry["spec_sha256"],
            f"{case_id} frozen spec hash does not match",
        )
        variants = _actor_sets(
            spec,
            case_id,
            actor_set_registry,
        )
        if "expanded" in variants:
            expanded_case_count += 1
        seeds = resolve_replication_seeds(spec, count=3)
        case_run_root = run_root / benchmark_id / case_id
        research_dir = case_run_root / "research"
        submission_dir = benchmark_dir / "submissions" / case_id
        baseline_path = submission_dir / "B1.json"
        b1_command_id = f"{case_id}:B1"
        graph_command_id = f"{case_id}:research_graph"
        commands.append(
            _command(
                b1_command_id,
                "baseline_forecast",
                [
                    "upstreams/FEVER/.venv/bin/python",
                    "scripts/generate_forecast_submission.py",
                    str(spec_path),
                    "--arm",
                    "B1",
                    "--output",
                    str(baseline_path),
                    "--benchmark-id",
                    benchmark_id,
                    "--generation-audit",
                    str(generation_audit),
                    "--allow-billable",
                ],
                prerequisites=[],
                billable=True,
                output_paths=[baseline_path],
                env=common_model_env,
            )
        )
        commands.append(
            _command(
                graph_command_id,
                "research_graph",
                [
                    "python3",
                    "scripts/run_smoke.py",
                    str(spec_path),
                    "--through",
                    "graph",
                    "--run-dir",
                    str(research_dir),
                    "--allow-billable",
                ],
                prerequisites=[],
                billable=True,
                output_paths=[research_dir / "state.json"],
            )
        )
        variant_plans = []
        for variant, actor_ids in variants.items():
            signal_paths = []
            replication_plans = []
            replication_command_ids = []
            for index, seed in enumerate(seeds):
                total_replications += 1
                replication_id = f"{case_id}-{variant}-r{index:02d}"
                replication_dir = (
                    case_run_root / variant / f"replication-{index:02d}"
                )
                scenario_dir = (
                    scenario_root
                    / benchmark_id
                    / case_id
                    / variant
                    / f"replication-{index:02d}"
                )
                raw_result = replication_dir / "simulation-result.json"
                financial_actions = replication_dir / "financial-actions.json"
                branches = scenario_dir / "scenario-branches.json"
                updated_result = scenario_dir / "simulation-result.json"
                signal_path = (
                    probability_root
                    / benchmark_id
                    / "signals"
                    / case_id
                    / variant
                    / f"replication-{index:02d}.json"
                )
                signal_paths.append(signal_path)
                simulation_command_id = f"{replication_id}:simulation"
                scenario_command_id = f"{replication_id}:scenario"
                signal_command_id = f"{replication_id}:signal"
                simulation_argv = [
                    "python3",
                    "scripts/run_smoke.py",
                    str(spec_path),
                    "--financial",
                    "--through",
                    "simulation",
                    "--run-dir",
                    str(replication_dir),
                    "--reuse-research-state",
                    str(research_dir / "state.json"),
                    "--scheduler-seed",
                    str(seed),
                ]
                for actor_id in actor_ids:
                    simulation_argv.extend(
                        ["--financial-actor-id", actor_id]
                    )
                simulation_argv.append("--allow-billable")
                commands.append(
                    _command(
                        simulation_command_id,
                        "multi_agent_simulation",
                        simulation_argv,
                        prerequisites=[graph_command_id],
                        billable=True,
                        output_paths=[
                            raw_result,
                            financial_actions,
                            replication_dir / "sqlite-actions.json",
                        ],
                    )
                )
                commands.append(
                    _command(
                        scenario_command_id,
                        "scenario_compilation",
                        [
                            "upstreams/FEVER/.venv/bin/python",
                            "scripts/generate_scenario_branches.py",
                            str(spec_path),
                            "--simulation-result",
                            str(raw_result),
                            "--financial-actions",
                            str(financial_actions),
                            "--output",
                            str(branches),
                            "--updated-result",
                            str(updated_result),
                            "--benchmark-id",
                            benchmark_id,
                            "--generation-audit",
                            str(generation_audit),
                            "--allow-billable",
                        ],
                        prerequisites=[simulation_command_id],
                        billable=True,
                        output_paths=[branches, updated_result],
                        env=common_model_env,
                    )
                )
                commands.append(
                    _command(
                        signal_command_id,
                        "probability_signal_extraction",
                        [
                            "upstreams/FEVER/.venv/bin/python",
                            "scripts/generate_probability_signals.py",
                            str(spec_path),
                            "--simulation-result",
                            str(updated_result),
                            "--baseline-submission",
                            str(baseline_path),
                            "--output",
                            str(signal_path),
                            "--dataset-id",
                            benchmark_id,
                            "--generation-audit",
                            str(generation_audit),
                            "--allow-billable",
                        ],
                        prerequisites=[scenario_command_id, b1_command_id],
                        billable=True,
                        output_paths=[signal_path],
                        env=common_model_env,
                    )
                )
                replication_command_ids.append(signal_command_id)
                replication_plans.append(
                    {
                        "replication_index": index,
                        "replication_id": replication_id,
                        "scheduler_seed": seed,
                        "run_dir": str(replication_dir),
                        "scenario_dir": str(scenario_dir),
                        "probability_signal_path": str(signal_path),
                    }
                )

            ensemble_path = (
                probability_root
                / benchmark_id
                / "ensembles"
                / case_id
                / f"{variant}.json"
            )
            aggregate_command_id = f"{case_id}:{variant}:aggregate"
            aggregate_argv = [
                "python3",
                "scripts/aggregate_probability_signals.py",
                str(spec_path),
                "--baseline-submission",
                str(baseline_path),
            ]
            for signal_path in signal_paths:
                aggregate_argv.extend(
                    ["--probability-signal", str(signal_path)]
                )
            aggregate_argv.extend(
                [
                    "--output",
                    str(ensemble_path),
                    "--benchmark-id",
                    benchmark_id,
                    "--generation-audit",
                    str(generation_audit),
                ]
            )
            commands.append(
                _command(
                    aggregate_command_id,
                    "signal_aggregation",
                    aggregate_argv,
                    prerequisites=replication_command_ids,
                    billable=False,
                    output_paths=[ensemble_path],
                )
            )
            submission_path = (
                submission_dir / "B3.json"
                if variant == "core"
                else benchmark_dir
                / "ablations"
                / case_id
                / "B3-expanded.json"
            )
            b3_command_id = f"{case_id}:{variant}:B3"
            commands.append(
                _command(
                    b3_command_id,
                    "ensemble_calibration",
                    [
                        "python3",
                        "scripts/build_ensemble_calibrated_submission.py",
                        str(spec_path),
                        "--baseline-submission",
                        str(baseline_path),
                        "--probability-ensemble",
                        str(ensemble_path),
                        "--policy",
                        str(policy_path),
                        "--output",
                        str(submission_path),
                        "--benchmark-id",
                        benchmark_id,
                        "--generation-audit",
                        str(generation_audit),
                    ],
                    prerequisites=[aggregate_command_id],
                    billable=False,
                    output_paths=[submission_path],
                )
            )
            variant_plans.append(
                {
                    "variant": variant,
                    "actor_ids": actor_ids,
                    "replications": replication_plans,
                    "ensemble_path": str(ensemble_path),
                    "submission_path": str(submission_path),
                }
            )
        case_plans.append(
            {
                "case_id": case_id,
                "spec_path": str(spec_path),
                "spec_sha256": entry["spec_sha256"],
                "research_dir": str(research_dir),
                "baseline_submission_path": str(baseline_path),
                "variants": variant_plans,
            }
        )

    _require(
        expanded_case_count == expected_ablation_cases,
        "expanded actor-set count does not match frozen budget",
    )
    expected_runs = len(cases) * 3 + expanded_case_count * 3
    expected_cost_units = {
        "new_research_graph_builds": len(cases),
        "new_multi_agent_runs": expected_runs,
        "new_scenario_compilation_calls": expected_runs,
        "new_probability_signal_extraction_calls": expected_runs,
        "new_b1_forecast_aggregation_calls": len(cases),
        "new_signal_aggregations": len(cases) + expanded_case_count,
        "new_b3_local_calibrations": len(cases) + expanded_case_count,
    }
    for field, expected in expected_cost_units.items():
        _require(
            budget.get(field) == expected,
            f"formal budget {field} does not match execution plan",
        )
    output_paths = [
        path
        for command in commands
        for path in command["output_paths"]
    ]
    _require(
        len(output_paths) == len(set(output_paths)),
        "formal execution outputs must use unique paths",
    )
    _require(
        total_replications == expected_runs,
        "formal replication accounting mismatch",
    )
    return {
        "schema_version": "0.1.0",
        "benchmark_id": benchmark_id,
        "manifest_frozen_at": manifest["frozen_at"],
        "case_count": len(cases),
        "core_replications_per_case": 3,
        "expanded_ablation_case_count": expanded_case_count,
        "total_multi_agent_runs": total_replications,
        "signal_aggregation_version": AGGREGATION_VERSION,
        "provider_sampling_seeded": False,
        "scheduler_seed_scope": "python_random_agent_scheduler_only",
        "service_start": {
            "env": {
                "BENCHMARK": benchmark_id,
                "FEVER_MIROFISH_HOST_LEDGER_PATH": str(
                    benchmark_dir / "mirofish-host-llm-usage.jsonl"
                ),
                "FEVER_MIROFISH_MAX_LLM_CALLS": str(
                    budget.get("host_max_llm_calls", 80)
                ),
                "FEVER_MIROFISH_MAX_TOTAL_TOKENS": str(
                    budget.get("host_total_token_ceiling", 150000)
                ),
                "FEVER_MIROFISH_OASIS_MAX_LLM_CALLS": str(
                    budget.get("oasis_max_llm_calls_per_run", 40)
                ),
                "FEVER_MIROFISH_OASIS_MAX_TOTAL_TOKENS": str(
                    budget.get("oasis_max_tokens_per_run", 120000)
                ),
            },
            "argv": ["make", "run-mirofish", f"BENCHMARK={benchmark_id}"],
        },
        "cost_units": expected_cost_units,
        "case_plans": case_plans,
        "commands": commands,
        "execution_rule": (
            "Execute commands only after every frozen hash and budget check "
            "passes; respect prerequisites; stop on the first invalid output."
        ),
        "limitations": [
            "Registered result files are never opened or referenced.",
            "Three scheduler seeds reduce within-case instability but remain "
            "one independent historical event.",
            "Provider-side model sampling is not seed-controlled.",
        ],
    }
