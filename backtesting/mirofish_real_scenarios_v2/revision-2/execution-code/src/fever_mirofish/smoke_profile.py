"""Conservative execution settings for a low-cost MiroFish smoke run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SmokeProfile:
    """A deliberately small compatibility run, not a research replication."""

    rounds: int = 3
    platform: str = "reddit"
    enable_twitter: bool = False
    enable_reddit: bool = True
    use_llm_for_profiles: bool = False
    parallel_profile_count: int = 1
    enable_graph_memory_update: bool = False
    financial_actor_overlay: bool = False
    collect_financial_decisions: bool = False
    financial_actor_ids: Optional[tuple[str, ...]] = None
    scheduler_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 1 <= self.rounds <= 5:
            raise ValueError("smoke rounds must be between 1 and 5")
        if self.platform not in {"twitter", "reddit"}:
            raise ValueError("smoke profile must use exactly one platform")
        if self.platform == "reddit" and not self.enable_reddit:
            raise ValueError("reddit platform must be enabled")
        if self.platform == "twitter" and not self.enable_twitter:
            raise ValueError("twitter platform must be enabled")
        if self.enable_twitter and self.enable_reddit:
            raise ValueError("smoke profile must not enable both platforms")
        if self.parallel_profile_count != 1:
            raise ValueError("smoke profile parallel_profile_count must be 1")
        if self.enable_graph_memory_update:
            raise ValueError("smoke profile must not mutate graph memory")
        if self.collect_financial_decisions and not self.financial_actor_overlay:
            raise ValueError(
                "financial decisions require the deterministic actor overlay"
            )
        if self.financial_actor_ids is not None:
            if not self.financial_actor_overlay:
                raise ValueError(
                    "financial_actor_ids require the deterministic actor overlay"
                )
            if (
                not isinstance(self.financial_actor_ids, tuple)
                or not 2 <= len(self.financial_actor_ids) <= 12
                or len(self.financial_actor_ids)
                != len(set(self.financial_actor_ids))
                or not all(
                    isinstance(item, str) and item
                    for item in self.financial_actor_ids
                )
            ):
                raise ValueError(
                    "financial_actor_ids must be a unique tuple of 2 to 12 ids"
                )
        if self.scheduler_seed is not None and (
            not isinstance(self.scheduler_seed, int)
            or isinstance(self.scheduler_seed, bool)
            or not 0 <= self.scheduler_seed <= 2_147_483_647
        ):
            raise ValueError(
                "scheduler_seed must be an integer between 0 and 2147483647"
            )

    def create_payload(self, project_id: str, graph_id: str | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "enable_twitter": self.enable_twitter,
            "enable_reddit": self.enable_reddit,
        }
        if graph_id:
            payload["graph_id"] = graph_id
        return payload

    def prepare_payload(self, simulation_id: str) -> Dict[str, Any]:
        return {
            "simulation_id": simulation_id,
            "use_llm_for_profiles": self.use_llm_for_profiles,
            "parallel_profile_count": self.parallel_profile_count,
            "force_regenerate": False,
        }

    def start_payload(self, simulation_id: str) -> Dict[str, Any]:
        return {
            "simulation_id": simulation_id,
            "platform": self.platform,
            "max_rounds": self.rounds,
            "enable_graph_memory_update": self.enable_graph_memory_update,
            "force": False,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
