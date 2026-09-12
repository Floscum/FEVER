"""Deterministic runtime overrides for a bounded MiroFish smoke run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def apply_smoke_runtime_config(
    config_path: Path,
    *,
    rounds: int,
    scheduler_seed: int | None = None,
) -> Dict[str, Any]:
    """Make the first ``rounds`` hours active and cap agents per round.

    MiroFish's generated schedule begins at hour zero. Financial agents are
    usually generated with daytime active hours, so merely truncating to three
    rounds can otherwise produce zero autonomous actions.
    """

    if not 1 <= rounds <= 5:
        raise ValueError("smoke runtime rounds must be between 1 and 5")
    if scheduler_seed is not None and (
        not isinstance(scheduler_seed, int)
        or isinstance(scheduler_seed, bool)
        or not 0 <= scheduler_seed <= 2_147_483_647
    ):
        raise ValueError(
            "scheduler_seed must be an integer between 0 and 2147483647"
        )
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    backup_path = config_path.with_name("simulation_config.generated.json")
    if not backup_path.exists():
        backup_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    active_hours = list(range(rounds))
    configured_actor_count = len(config.get("agent_configs") or [])
    active_actor_target = max(1, min(10, configured_actor_count or 4))
    time_config = config.setdefault("time_config", {})
    time_config.update(
        {
            "total_simulation_hours": rounds,
            "minutes_per_round": 60,
            # Quick mode is already bounded to at most ten financial actors and
            # five rounds.  Scheduling every configured actor prevents a 6/8/10
            # actor specification from silently behaving like a four-actor run.
            "agents_per_hour_min": active_actor_target,
            "agents_per_hour_max": active_actor_target,
            "peak_hours": active_hours,
            "peak_activity_multiplier": 1.0,
            "off_peak_hours": [],
            "off_peak_activity_multiplier": 1.0,
        }
    )
    for agent in config.get("agent_configs") or []:
        agent["active_hours"] = active_hours
        agent["activity_level"] = 1.0

    config["fever_mirofish_smoke_override"] = {
        "rounds": rounds,
        "active_hours": active_hours,
        "configured_actor_count": configured_actor_count,
        "agents_per_round_min": active_actor_target,
        "agents_per_round_max": active_actor_target,
        "purpose": "bounded integration smoke test; not a research replication",
    }
    config["fever_mirofish_runtime"] = {
        "scheduler_seed": scheduler_seed,
        "seed_scope": "python_random_agent_scheduler_only",
        "provider_sampling_seeded": False,
        "implementation": "oasis_wrapper_v1",
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Upstream's restart readiness check currently requires both platform
    # profile files even when Twitter is disabled. A header-only placeholder
    # satisfies that existence check and is never consumed by a Reddit run.
    twitter_placeholder = config_path.parent / "twitter_profiles.csv"
    if not twitter_placeholder.exists():
        twitter_placeholder.write_text("user_id\n", encoding="utf-8")
    return config["fever_mirofish_smoke_override"]
