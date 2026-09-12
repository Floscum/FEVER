"""Content-free audit records for model-output parsing and sealing."""

from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def acquire_generation_lock(output_path: Path):
    """Hold a non-blocking process lock for one billable output path.

    The lock file is intentionally retained as harmless metadata; the kernel
    lock is released automatically when the returned handle closes or the
    process exits. Callers must keep the handle alive for the whole request.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.generation.lock")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"generation already active for output: {output_path}"
        ) from error
    return handle


def append_generation_audit(
    path: Path,
    *,
    benchmark_id: str,
    case_id: str,
    artifact_kind: str,
    status: str,
    model_id: str,
    output_path: Path,
    error: Optional[BaseException] = None,
    artifact_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Append metadata only; prompts and model responses are never persisted."""

    if status not in {"sealed", "invalid", "provider_failed"}:
        raise ValueError(f"unsupported generation status: {status}")
    record: Dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": benchmark_id,
        "case_id": case_id,
        "artifact_kind": artifact_kind,
        "status": status,
        "model_id": model_id,
        "output_path": str(output_path),
        "artifact_sha256": artifact_sha256,
        "error_type": type(error).__name__ if error is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
    return record
