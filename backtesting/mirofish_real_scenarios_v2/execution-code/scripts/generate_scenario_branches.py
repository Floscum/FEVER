#!/usr/bin/env python3
"""Generate one bounded scenario-branch artifact and optionally attach it."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fever_mirofish.scenario_branches import (  # noqa: E402
    append_scenario_branches_to_result,
    build_scenario_branch_prompt,
    build_scenario_branch_retry_prompt,
    build_scenario_branch_set,
)
from fever_mirofish.contracts import (  # noqa: E402
    canonical_sha256,
    validate_result,
    validate_scenario_branches,
)
from fever_mirofish.generation_audit import (  # noqa: E402
    acquire_generation_lock,
    append_generation_audit,
)
from fever_mirofish.usage_ledger import (  # noqa: E402
    UsageLedger,
    attach_openai_usage_ledger,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--simulation-result", type=Path, required=True)
    parser.add_argument("--financial-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updated-result", type=Path)
    parser.add_argument("--product", action="store_true", help="Use the stable compact product compiler.")
    parser.add_argument("--product-version", choices=("v7", "v8", "v9"), default="v7", help="v9 selects focused paths; v8 is experimental. The default is promoted only after paired checks.")
    parser.add_argument(
        "--benchmark-id",
        default="engineering",
        help="Names the isolated ledger and generation-audit namespace.",
    )
    parser.add_argument("--generation-audit", type=Path)
    parser.add_argument(
        "--max-semantic-retries",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Retry one structurally invalid response using only the original "
            "anonymous input and validation error."
        ),
    )
    parser.add_argument("--allow-billable", action="store_true")
    args = parser.parse_args()

    spec = _read(args.spec)
    result = _read(args.simulation_result)
    actions = _read(args.financial_actions)
    audit_path = args.generation_audit or (
        ROOT
        / ".data"
        / "benchmarks"
        / args.benchmark_id
        / "generation-audit.jsonl"
    )
    system, user, allowed_refs = build_scenario_branch_prompt(
        spec,
        result,
        actions,
        product=args.product,
        product_version=f"scenario-branch-compiler-{args.product_version}",
    )
    plan = {
        "case_id": spec["case_id"],
        "simulation_id": result["runs"][0]["run_id"],
        "prompt_chars": len(system) + len(user),
        "allowed_simulation_ref_count": len(allowed_refs),
        "output": str(args.output),
        "updated_result": (
            str(args.updated_result) if args.updated_result else None
        ),
        "billable": bool(args.allow_billable),
        "max_semantic_retries": args.max_semantic_retries,
        "compiler_profile": f"product-{args.product_version}" if args.product else "research-v6",
    }
    if not args.allow_billable:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    # Keep this handle alive until the process exits. This prevents two shell
    # sessions from paying for the same artifact while the first is in flight.
    generation_lock = acquire_generation_lock(args.output)
    if args.output.exists():
        existing = _read(args.output)
        validate_scenario_branches(existing, spec, result)
        if args.updated_result:
            if not args.updated_result.exists():
                raise RuntimeError(
                    "sealed scenario output exists but updated result is missing"
                )
            validate_result(_read(args.updated_result), spec)
        plan["branch_count"] = len(existing["branches"])
        plan["artifact_sha256"] = canonical_sha256(existing)
        plan["billable"] = False
        plan["reused_sealed_output"] = True
        plan["reused_prompt_version"] = existing["prompt_version"]
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    base_url = os.environ.get("ARK_API_URL") or os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("LLM_API_KEY")
    model = (
        os.environ.get("ARK_MODEL")
        or os.environ.get("LLM_MODEL_NAME")
        or "deepseek-ai/DeepSeek-V4-Flash"
    )
    if not base_url or not api_key:
        raise RuntimeError("scenario provider endpoint/key is missing")

    from openai import OpenAI
    from openai.resources.chat.completions.completions import (
        AsyncCompletions,
        Completions,
    )

    ledger = UsageLedger(
        ROOT
        / ".data"
        / "benchmarks"
        / args.benchmark_id
        / "scenario-llm-usage.jsonl",
        max_calls=int(os.environ.get("FEVER_SCENARIO_MAX_LLM_CALLS", "12")),
        max_total_tokens=int(
            os.environ.get("FEVER_SCENARIO_MAX_TOTAL_TOKENS", "120000")
        ),
    )
    attach_openai_usage_ledger(Completions, AsyncCompletions, ledger)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
    request_system = system
    semantic_attempts = 0
    while True:
        response_received = False
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": request_system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=5000,
            )
            response_received = True
            branch_set = build_scenario_branch_set(
                response.choices[0].message.content or "",
                spec,
                result,
                actions,
                model_id=model,
                product=args.product,
                product_version=f"scenario-branch-compiler-{args.product_version}",
            )
            break
        except Exception as error:
            append_generation_audit(
                audit_path,
                benchmark_id=args.benchmark_id,
                case_id=spec["case_id"],
                artifact_kind="scenario_branches",
                status="invalid" if response_received else "provider_failed",
                model_id=model,
                output_path=args.output,
                error=error,
            )
            if not response_received or semantic_attempts >= args.max_semantic_retries:
                raise
            semantic_attempts += 1
            request_system, user = build_scenario_branch_retry_prompt(
                system,
                user,
                error,
            )
    _write(args.output, branch_set)
    if args.updated_result:
        updated = append_scenario_branches_to_result(
            result,
            branch_set,
            spec,
        )
        updated["provenance"]["raw_run_artifacts"].append(
            str(args.output.resolve())
        )
        _write(args.updated_result, updated)
    branch_sha256 = canonical_sha256(branch_set)
    append_generation_audit(
        audit_path,
        benchmark_id=args.benchmark_id,
        case_id=spec["case_id"],
        artifact_kind="scenario_branches",
        status="sealed",
        model_id=model,
        output_path=args.output,
        artifact_sha256=branch_sha256,
    )
    plan["branch_count"] = len(branch_set["branches"])
    plan["artifact_sha256"] = branch_sha256
    plan["billable"] = True
    plan["semantic_retries_used"] = semantic_attempts
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
