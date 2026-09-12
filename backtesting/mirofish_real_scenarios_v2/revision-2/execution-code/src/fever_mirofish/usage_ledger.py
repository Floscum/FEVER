"""Token/cost observability for the local MiroFish smoke server."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type


class UsageBudgetExceeded(RuntimeError):
    """Raised before a request when the local smoke budget is exhausted."""


class UsageLedger:
    """Track model usage without persisting prompts, responses, or credentials."""

    def __init__(
        self,
        path: Path,
        *,
        max_calls: int = 80,
        max_total_tokens: int = 150_000,
    ):
        if max_calls < 1 or max_total_tokens < 1:
            raise ValueError("usage budgets must be positive")
        self.path = Path(path)
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self.attempts = 0
        self.calls = 0
        self.failures = 0
        self.total_tokens = 0
        self._lock = threading.Lock()
        self._restore_existing()

    def before_request(self) -> None:
        with self._lock:
            if self.attempts >= self.max_calls:
                raise UsageBudgetExceeded(
                    f"local smoke budget exhausted at {self.attempts} LLM calls"
                )
            if self.total_tokens >= self.max_total_tokens:
                raise UsageBudgetExceeded(
                    "local smoke token budget exhausted at "
                    f"{self.total_tokens} tokens"
                )
            # Reserve before dispatch so concurrent OASIS agents cannot all
            # pass the same stale call-count check.
            self.attempts += 1

    def record(self, *, model: str, response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        usage_data = self._usage_to_dict(usage)
        prompt_tokens = self._integer(
            usage_data.get("prompt_tokens"),
            getattr(usage, "prompt_tokens", 0),
        )
        completion_tokens = self._integer(
            usage_data.get("completion_tokens"),
            getattr(usage, "completion_tokens", 0),
        )
        total_tokens = self._integer(
            usage_data.get("total_tokens"),
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens),
        )
        estimated_cost = usage_data.get("estimated_cost")
        if estimated_cost is None:
            details = usage_data.get("model_extra")
            if isinstance(details, dict):
                estimated_cost = details.get("estimated_cost")

        with self._lock:
            if self.attempts <= self.calls + self.failures:
                # Support direct record() calls in small utilities/tests.
                self.attempts += 1
            self.calls += 1
            self.total_tokens += total_tokens
            completed_attempt_number = self.calls + self.failures
            record = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "status": "completed",
                "attempt_number": completed_attempt_number,
                "call_number": self.calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "session_total_tokens": self.total_tokens,
                "estimated_cost": estimated_cost,
            }
            self._append(record)
        return record

    def record_failure(self, *, model: str, error: BaseException) -> Dict[str, Any]:
        """Persist only an exception type; never persist provider text."""

        with self._lock:
            if self.attempts <= self.calls + self.failures:
                self.attempts += 1
            self.failures += 1
            completed_attempt_number = self.calls + self.failures
            record = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "status": "failed",
                "attempt_number": completed_attempt_number,
                "failure_number": self.failures,
                "error_type": type(error).__name__,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "session_total_tokens": self.total_tokens,
                "estimated_cost": None,
            }
            self._append(record)
        return record

    def _append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    def _restore_existing(self) -> None:
        if not self.path.exists():
            return
        attempts = calls = failures = total_tokens = 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    attempts = max(
                        attempts,
                        self._integer(record.get("attempt_number"), attempts + 1),
                    )
                    if record.get("status", "completed") == "failed":
                        failures += 1
                    else:
                        calls += 1
                    total_tokens += self._integer(record.get("total_tokens"), 0)
        except (OSError, json.JSONDecodeError):
            # A corrupt ledger must fail closed rather than reset the budget.
            self.attempts = self.max_calls
            self.total_tokens = self.max_total_tokens
            return
        self.attempts = attempts
        self.calls = calls
        self.failures = failures
        self.total_tokens = total_tokens

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        model_dump = getattr(usage, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                model_extra = getattr(usage, "model_extra", None)
                if isinstance(model_extra, dict):
                    dumped.update(model_extra)
                return dumped
        return {}

    @staticmethod
    def _integer(primary: Any, fallback: Any) -> int:
        for value in (primary, fallback):
            if isinstance(value, int) and value >= 0:
                return value
        return 0


def attach_usage_ledger(
    llm_client_class: Type[Any],
    ledger: UsageLedger,
) -> Callable[..., Any]:
    """Wrap MiroFish ``LLMClient._create_completion`` once per process."""

    original = llm_client_class._create_completion
    if getattr(original, "_fever_mirofish_usage_wrapper", False):
        return original

    def tracked(client_self: Any, **kwargs: Any) -> Any:
        ledger.before_request()
        try:
            response = original(client_self, **kwargs)
        except Exception as error:
            ledger.record_failure(model=str(client_self.model), error=error)
            raise
        ledger.record(model=str(client_self.model), response=response)
        return response

    tracked._fever_mirofish_usage_wrapper = True  # type: ignore[attr-defined]
    llm_client_class._create_completion = tracked
    return tracked


def attach_model_backend_usage_ledger(
    model_backend_class: Type[Any],
    ledger: UsageLedger,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Wrap CAMEL's sync and async model entry points in an OASIS process."""

    original_run = model_backend_class.run
    original_arun = model_backend_class.arun
    if getattr(original_run, "_fever_mirofish_usage_wrapper", False):
        return original_run, original_arun

    def model_name(instance: Any) -> str:
        return str(
            getattr(instance, "model_type", None)
            or getattr(instance, "model", None)
            or "unknown"
        )

    def tracked_run(instance: Any, *args: Any, **kwargs: Any) -> Any:
        ledger.before_request()
        try:
            response = original_run(instance, *args, **kwargs)
        except Exception as error:
            ledger.record_failure(model=model_name(instance), error=error)
            raise
        ledger.record(model=model_name(instance), response=response)
        return response

    async def tracked_arun(instance: Any, *args: Any, **kwargs: Any) -> Any:
        ledger.before_request()
        try:
            response = await original_arun(instance, *args, **kwargs)
        except Exception as error:
            ledger.record_failure(model=model_name(instance), error=error)
            raise
        ledger.record(model=model_name(instance), response=response)
        return response

    tracked_run._fever_mirofish_usage_wrapper = True  # type: ignore[attr-defined]
    tracked_arun._fever_mirofish_usage_wrapper = True  # type: ignore[attr-defined]
    model_backend_class.run = tracked_run
    model_backend_class.arun = tracked_arun
    return tracked_run, tracked_arun


def attach_openai_usage_ledger(
    completions_class: Type[Any],
    async_completions_class: Type[Any],
    ledger: UsageLedger,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Wrap OpenAI SDK completion resources used directly by MiroFish."""

    original_create = completions_class.create
    original_async_create = async_completions_class.create
    if getattr(original_create, "_fever_mirofish_usage_wrapper", False):
        return original_create, original_async_create

    def tracked_create(instance: Any, *args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "unknown")
        ledger.before_request()
        try:
            response = original_create(instance, *args, **kwargs)
        except Exception as error:
            ledger.record_failure(model=model, error=error)
            raise
        ledger.record(model=model, response=response)
        return response

    async def tracked_async_create(
        instance: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        model = str(kwargs.get("model") or "unknown")
        ledger.before_request()
        try:
            response = await original_async_create(instance, *args, **kwargs)
        except Exception as error:
            ledger.record_failure(model=model, error=error)
            raise
        ledger.record(model=model, response=response)
        return response

    tracked_create._fever_mirofish_usage_wrapper = True  # type: ignore[attr-defined]
    tracked_async_create._fever_mirofish_usage_wrapper = True  # type: ignore[attr-defined]
    completions_class.create = tracked_create
    async_completions_class.create = tracked_async_create
    return tracked_create, tracked_async_create
