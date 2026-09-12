"""Seed planning and audit helpers for MiroFish/OASIS replications."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict

from .contracts import canonical_sha256, validate_spec


MAX_SCHEDULER_SEED = 2_147_483_647
SEED_SCOPE = "python_random_agent_scheduler_only"


def resolve_replication_seeds(
    spec: Dict[str, Any],
    *,
    count: int | None = None,
) -> list[int]:
    """Resolve stable, case-bound seeds without reading any outcome."""

    validate_spec(spec)
    run_config = spec["run_config"]
    configured_count = int(run_config["replications"])
    requested_count = configured_count if count is None else count
    if (
        not isinstance(requested_count, int)
        or isinstance(requested_count, bool)
        or not 1 <= requested_count <= configured_count
    ):
        raise ValueError(
            "replication count must be between 1 and run_config.replications"
        )

    if run_config["seed_strategy"] == "explicit":
        return [int(value) for value in run_config["seeds"][:requested_count]]

    spec_hash = canonical_sha256(spec)
    seeds = []
    for index in range(requested_count):
        digest = hashlib.sha256(
            f"fever-mirofish:{spec_hash}:{index}".encode("utf-8")
        ).digest()
        seeds.append(int.from_bytes(digest[:8], "big") % (MAX_SCHEDULER_SEED + 1))
    return seeds


def seed_scheduler_from_config(config_path: Path) -> Dict[str, Any]:
    """Seed the global Python RNG used by the upstream agent scheduler.

    This deliberately does not claim to seed provider-side LLM sampling.
    """

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = config.get("fever_mirofish_runtime") or {}
    seed = runtime.get("scheduler_seed")
    if seed is None:
        return {
            "scheduler_seed": None,
            "seed_applied": False,
            "seed_scope": SEED_SCOPE,
            "provider_sampling_seeded": False,
        }
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= MAX_SCHEDULER_SEED
    ):
        raise ValueError(
            "simulation config scheduler_seed must be an integer between "
            f"0 and {MAX_SCHEDULER_SEED}"
        )
    random.seed(seed)
    return {
        "scheduler_seed": seed,
        "seed_applied": True,
        "seed_scope": SEED_SCOPE,
        "provider_sampling_seeded": False,
    }


def build_replication_plan(
    spec: Dict[str, Any],
    *,
    arm: str,
    count: int,
    run_root: Path,
) -> Dict[str, Any]:
    """Build a non-billable, deterministic execution plan."""

    if arm not in {"B2", "B3"}:
        raise ValueError("simulation replication arm must be B2 or B3")
    seeds = resolve_replication_seeds(spec, count=count)
    spec_hash = canonical_sha256(spec)
    suffix = arm.lower()
    return {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": spec_hash,
        "arm": arm,
        "replication_count": count,
        "seed_strategy": spec["run_config"]["seed_strategy"],
        "seed_scope": SEED_SCOPE,
        "provider_sampling_seeded": False,
        "replications": [
            {
                "replication_index": index,
                "replication_id": f"{spec['case_id']}-{suffix}-r{index:02d}",
                "scheduler_seed": seed,
                "run_dir": str(
                    Path(run_root)
                    / spec["case_id"]
                    / suffix
                    / f"replication-{index:02d}"
                ),
            }
            for index, seed in enumerate(seeds)
        ],
        "limitations": [
            "The seed controls Python random scheduling and actor selection.",
            "Provider-side LLM sampling is not seed-controlled and exact text "
            "replay is not guaranteed.",
        ],
    }
