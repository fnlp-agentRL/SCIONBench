#!/usr/bin/env python3
"""Persist token usage emitted by ``codex exec --json``.

Codex has emitted two relevant JSONL shapes over time: cumulative
``token_count`` snapshots and the newer ``turn.completed`` event with a
``usage`` object.  The last snapshot in one process is therefore the value to
persist; summing snapshots would count the same tokens repeatedly.  This
module is intentionally dependency-free so the shell supervisor can use it in
the same environments as the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - the pipeline currently runs on Linux.
    fcntl = None  # type: ignore[assignment]


USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
SCHEMA_VERSION = 1
USAGE_EVENT_TYPES = {
    "turn.completed",
    "turn.failed",
    "response.completed",
    "response.failed",
    "response.done",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_usage() -> dict[str, int | None]:
    return {field: None for field in USAGE_FIELDS}


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    return None


def normalize_usage(value: Any) -> dict[str, int | None]:
    if not isinstance(value, dict):
        return empty_usage()

    normalized = {field: _integer(value.get(field)) for field in USAGE_FIELDS}
    # Responses API versions may put cache/reasoning details in nested
    # ``*_details`` objects, while CCR's turn.completed currently emits the
    # flat names.  Accept both without counting cached input twice.
    input_details = value.get("input_tokens_details")
    if not isinstance(input_details, dict):
        # Responses API uses the singular ``input_token_details`` in some
        # response envelopes (while newer turn events use the plural form).
        input_details = value.get("input_token_details")
    if isinstance(input_details, dict):
        if normalized["cached_input_tokens"] is None:
            normalized["cached_input_tokens"] = _integer(
                input_details.get("cached_tokens")
            )
        if normalized["cache_write_input_tokens"] is None:
            normalized["cache_write_input_tokens"] = _integer(
                input_details.get("cache_write_tokens")
            )
    output_details = value.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = value.get("output_token_details")
    if isinstance(output_details, dict) and normalized["reasoning_output_tokens"] is None:
        normalized["reasoning_output_tokens"] = _integer(
            output_details.get("reasoning_tokens")
        )
    if normalized["cached_input_tokens"] is None:
        normalized["cached_input_tokens"] = _integer(value.get("cache_read_input_tokens"))
    if normalized["cache_write_input_tokens"] is None:
        normalized["cache_write_input_tokens"] = _integer(
            value.get("cache_creation_input_tokens")
        )
    if normalized["reasoning_output_tokens"] is None:
        normalized["reasoning_output_tokens"] = _integer(value.get("reasoning_tokens"))

    # ``turn.completed.usage`` does not include total_tokens.  In the
    # Responses schema input_tokens already includes cached input, so the
    # unambiguous total is input + output (cached input is a subset, not an
    # additional quantity).  Leave total_tokens null when either component is
    # unavailable rather than inventing a partial total.
    if normalized["total_tokens"] is None:
        input_tokens = normalized["input_tokens"]
        output_tokens = normalized["output_tokens"]
        if input_tokens is not None and output_tokens is not None:
            normalized["total_tokens"] = input_tokens + output_tokens
    return normalized


def _normalized_usage(value: Any) -> dict[str, int | None] | None:
    """Normalize a candidate and reject objects that contain no token data."""

    normalized = normalize_usage(value)
    if not any(item is not None for item in normalized.values()):
        return None
    return normalized


def _merge_usage_snapshots(
    previous: dict[str, int | None], current: dict[str, int | None]
) -> dict[str, int | None]:
    """Keep known fields when a failure event reports only a partial usage."""

    merged = {
        field: current.get(field) if current.get(field) is not None else previous.get(field)
        for field in USAGE_FIELDS
    }
    if current.get("total_tokens") is None:
        input_tokens = merged["input_tokens"]
        output_tokens = merged["output_tokens"]
        if input_tokens is not None and output_tokens is not None:
            merged["total_tokens"] = input_tokens + output_tokens
    return merged


def _token_usage_from_event(event: Any) -> dict[str, int | None] | None:
    """Extract a usage object from one JSONL event if present."""

    if not isinstance(event, dict):
        return None

    # Current Codex JSONL shape:
    # {"type":"event_msg", "payload":{"type":"token_count",
    #   "info":{"total_token_usage": {...}}}
    payload = event.get("payload")
    if isinstance(payload, dict) and payload.get("type") == "token_count":
        info = payload.get("info")
        if isinstance(info, dict):
            total = info.get("total_token_usage")
            if isinstance(total, dict):
                return _normalized_usage(total)

    # Be tolerant of direct token_count events from older/newer CLI versions.
    if event.get("type") == "token_count":
        info = event.get("info")
        if isinstance(info, dict):
            total = info.get("total_token_usage")
            if isinstance(total, dict):
                return _normalized_usage(total)
        total = event.get("total_token_usage")
        if isinstance(total, dict):
            return _normalized_usage(total)

    # Current Codex/CCR JSONL shape:
    # {"type":"turn.completed", "usage":{"input_tokens": ..., ...}}
    # Some wrappers put the same event under ``payload``.  Failed turns are
    # accepted too so a partial snapshot can still be retained as a lower
    # bound by the supervisor.
    event_type = event.get("type")
    if event_type in USAGE_EVENT_TYPES:
        usage = _normalized_usage(event.get("usage"))
        if usage is not None:
            return usage
    if isinstance(payload, dict) and payload.get("type") in USAGE_EVENT_TYPES:
        usage = _normalized_usage(payload.get("usage"))
        if usage is not None:
            return usage

    # Responses streaming/non-streaming envelopes sometimes place usage under
    # ``response`` (for example ``response.done.response.usage``) or under a
    # generic ``data`` wrapper.  Accept those forms without treating ordinary
    # response metadata as a token snapshot.
    for envelope_key in ("response", "data"):
        envelope = event.get(envelope_key)
        if not isinstance(envelope, dict):
            continue
        usage = _normalized_usage(envelope.get("usage"))
        if usage is not None:
            return usage
        nested = envelope.get("response")
        if isinstance(nested, dict):
            usage = _normalized_usage(nested.get("usage"))
            if usage is not None:
                return usage

    return None


def usage_from_jsonl(path: Path | None) -> tuple[dict[str, int | None], bool, int]:
    """Return the last cumulative usage snapshot, availability, and event count."""

    usage = empty_usage()
    event_count = 0
    if path is not None and not isinstance(path, Path):
        path = Path(path)
    if path is None or not path.is_file():
        return usage, False, event_count
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    # stderr and partial lines can be interleaved with JSONL.
                    continue
                snapshot = _token_usage_from_event(event)
                if snapshot is not None:
                    usage = (
                        snapshot
                        if event_count == 0
                        else _merge_usage_snapshots(usage, snapshot)
                    )
                    event_count += 1
    except OSError:
        return usage, event_count > 0, event_count
    return usage, event_count > 0, event_count


@contextmanager
def _locked(usage_file: Path) -> Iterator[None]:
    """Serialize updates from independent supervisor processes."""

    usage_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = usage_file.with_name(f".{usage_file.name}.lock")
    lock_handle = lock_file.open("a+", encoding="utf-8")
    locked = False
    try:
        if fcntl is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                locked = True
            except OSError:
                # Some remote filesystems do not implement advisory locks;
                # the atomic replace still prevents readers from seeing a
                # partial JSON document.
                locked = False
        yield
    finally:
        if fcntl is not None and locked:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def _read_document(path: Path, format_name: str, mode: str, number: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "format": format_name,
            "mode": mode,
            "number": str(number),
            "runs": [],
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A broken/partially written old file must not prevent a new run from
        # being recorded.  The atomic replacement below repairs it.
        value = None
    if isinstance(value, dict) and isinstance(value.get("runs"), list):
        document = dict(value)
        document["runs"] = [entry for entry in document["runs"] if isinstance(entry, dict)]
        return document
    if isinstance(value, list):
        # Accept the early experimental list-only shape without losing entries.
        return {
            "schema_version": SCHEMA_VERSION,
            "format": format_name,
            "mode": mode,
            "number": str(number),
            "runs": [entry for entry in value if isinstance(entry, dict)],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "format": format_name,
        "mode": mode,
        "number": str(number),
        "runs": [],
    }


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _entry_key(entry: dict[str, Any]) -> tuple[str, int | str]:
    attempt = entry.get("attempt", 0)
    parsed_attempt = _integer(attempt)
    return str(entry.get("run_id", "")), parsed_attempt if parsed_attempt is not None else str(attempt)


def _merge_entry(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge duplicate writes without discarding a usage snapshot."""

    merged = dict(existing)
    existing_status = str(existing.get("status", ""))
    incoming_status = str(incoming.get("status", ""))
    # A parent-side timeout/wait fallback must not relabel an attempt that the
    # supervisor already finalized.  It may still enrich that entry with a
    # token snapshot discovered after the first write.
    preserve_terminal_status = (
        existing_status in {"success", "failed", "interrupted"}
        and incoming_status in {"timeout", "supervisor_wait_failed"}
        and existing.get("finished_at")
    )
    for key, value in incoming.items():
        if preserve_terminal_status and key in {"status", "reason", "exit_code", "finished_at"}:
            continue
        if key in {"usage", "total_usage"}:
            old_usage = existing.get("total_usage", existing.get("usage"))
            old_available = bool(existing.get("usage_available"))
            new_available = bool(incoming.get("usage_available"))
            incoming_usage = incoming.get("total_usage", incoming.get("usage"))
            old_total = _integer(old_usage.get("total_tokens")) if isinstance(old_usage, dict) else None
            new_total = _integer(incoming_usage.get("total_tokens")) if isinstance(incoming_usage, dict) else None
            incoming_is_newer = (
                new_total is None
                or old_total is None
                or new_total >= old_total
            )
            if (new_available and incoming_is_newer) or not old_available:
                merged["total_usage"] = incoming_usage
                merged["usage"] = incoming_usage
            elif isinstance(old_usage, dict):
                merged["total_usage"] = old_usage
                merged["usage"] = old_usage
            continue
        if key == "usage_available":
            merged[key] = bool(existing.get(key)) or bool(value)
            continue
        if key == "usage_complete":
            merged[key] = bool(existing.get(key)) or bool(value)
            continue
        if key == "usage_lower_bound":
            merged[key] = bool(existing.get(key)) or bool(value)
            continue
        if key == "started_at" and existing.get(key):
            continue
        if value is not None or key not in existing:
            merged[key] = value
    # A later fallback can discover usage that an earlier write missed.
    if merged.get("usage_available"):
        merged["usage_source"] = "codex_jsonl"
        if merged.get("usage_lower_bound"):
            merged["usage_reason"] = "last_snapshot_before_process_termination"
        else:
            merged.pop("usage_reason", None)
    else:
        merged["usage_source"] = "none"
        merged["usage_reason"] = "no_token_count_event"
    return merged


def record_usage(
    usage_file: Path,
    *,
    events_file: Path | None,
    format_name: str,
    mode: str,
    number: str,
    run_id: str,
    attempt: int,
    status: str,
    exit_code: int | None = None,
    reason: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    usage_complete: bool = False,
    usage_lower_bound: bool = False,
) -> dict[str, Any]:
    """Upsert one Codex attempt and return the stored entry."""

    usage, available, event_count = usage_from_jsonl(events_file)
    entry: dict[str, Any] = {
        "run_id": str(run_id),
        "attempt": int(attempt),
        "codex_started": int(attempt) > 0,
        "status": status,
        "exit_code": exit_code,
        "started_at": started_at or now(),
        "finished_at": finished_at or now(),
        "usage_available": available,
        "usage_complete": bool(available and usage_complete),
        "usage_lower_bound": bool(usage_lower_bound and available),
        "usage_source": "codex_jsonl" if available else "none",
        # ``total_usage`` is the canonical name; ``usage`` is retained as a
        # short alias for consumers that prefer it.
        "total_usage": usage,
        "usage": usage,
        "token_event_count": event_count,
        "events_file": str(events_file) if events_file is not None else None,
    }
    if reason:
        entry["reason"] = reason
    if not available:
        entry["usage_reason"] = "no_token_count_event"
    elif usage_lower_bound:
        entry["usage_reason"] = "last_snapshot_before_process_termination"

    with _locked(usage_file):
        document = _read_document(usage_file, format_name, mode, str(number))
        document["schema_version"] = SCHEMA_VERSION
        document.setdefault("format", format_name)
        document.setdefault("mode", mode)
        document.setdefault("number", str(number))
        runs = document.setdefault("runs", [])
        key = _entry_key(entry)
        matching_index = next(
            (index for index, existing in enumerate(runs) if _entry_key(existing) == key),
            None,
        )
        if matching_index is None:
            runs.append(entry)
            stored = entry
        else:
            stored = _merge_entry(runs[matching_index], entry)
            runs[matching_index] = stored
        document["updated_at"] = now()
        _atomic_write(usage_file, document)
    return stored


def _resolve_events_file(usage_file: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    events_file = Path(value)
    if not events_file.is_absolute():
        events_file = usage_file.parent / events_file
    return events_file


def backfill_usage_file(
    usage_file: Path, *, statuses: set[str] | None = None
) -> dict[str, int | str]:
    """Enrich existing entries from their raw JSONL logs.

    Only entries whose status is in ``statuses`` are considered when a set is
    supplied.  The default caller uses ``{"success"}``, which avoids changing
    terminal failure semantics while still repairing historical successful
    reports.  The operation is idempotent and preserves run IDs, attempts and
    timestamps.
    """

    considered = 0
    updated = 0
    unavailable = 0
    with _locked(usage_file):
        document = _read_document(usage_file, "unknown", "unknown", "unknown")
        runs = document.setdefault("runs", [])
        for entry in runs:
            if statuses is not None and str(entry.get("status", "")) not in statuses:
                continue
            considered += 1
            events_file = _resolve_events_file(usage_file, entry.get("events_file"))
            usage, available, event_count = usage_from_jsonl(events_file)
            if not available:
                unavailable += 1
                continue
            entry["usage_available"] = True
            entry["usage_source"] = "codex_jsonl"
            entry["total_usage"] = usage
            entry["usage"] = usage
            entry["token_event_count"] = event_count
            if events_file is not None:
                entry["events_file"] = str(events_file)
            if str(entry.get("status", "")) == "success":
                entry["usage_complete"] = True
            if entry.get("usage_lower_bound"):
                entry["usage_reason"] = "last_snapshot_before_process_termination"
            else:
                entry.pop("usage_reason", None)
            updated += 1
        if updated:
            document["updated_at"] = now()
            _atomic_write(usage_file, document)
    return {
        "usage_file": str(usage_file),
        "considered": considered,
        "updated": updated,
        "unavailable": unavailable,
    }


def backfill_usage_tree(
    root: Path, *, statuses: set[str] | None = None
) -> dict[str, int]:
    """Backfill every ``codex_token_usage.json`` below ``root``."""

    paths = (
        [root]
        if root.name == "codex_token_usage.json" and root.is_file()
        else sorted(root.rglob("codex_token_usage.json"))
    )
    totals = {"files": 0, "files_changed": 0, "considered": 0, "updated": 0, "unavailable": 0}
    for usage_file in paths:
        result = backfill_usage_file(usage_file, statuses=statuses)
        totals["files"] += 1
        totals["files_changed"] += int(result["updated"] > 0)
        for key in ("considered", "updated", "unavailable"):
            totals[key] += int(result[key])
    return totals


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record", help="parse one JSONL attempt and append it")
    record.add_argument("--usage-file", type=Path, required=True)
    record.add_argument("--events-file", type=Path)
    record.add_argument("--format", dest="format_name", default="unknown")
    record.add_argument("--mode", default="unknown")
    record.add_argument("--number", default="unknown")
    record.add_argument("--run-id", required=True)
    record.add_argument("--attempt", type=int, required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--exit-code", type=int)
    record.add_argument("--reason")
    record.add_argument("--started-at")
    record.add_argument("--finished-at")
    record.add_argument("--usage-complete", action="store_true")
    record.add_argument("--usage-lower-bound", action="store_true")
    backfill = subparsers.add_parser(
        "backfill", help="recover usage in existing records from their JSONL logs"
    )
    backfill.add_argument("--root", type=Path, required=True)
    backfill.add_argument(
        "--all-statuses",
        action="store_true",
        help="also inspect failed/interrupted/timeout records (default: success only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "record":
        record_usage(
            args.usage_file,
            events_file=args.events_file,
            format_name=args.format_name,
            mode=args.mode,
            number=str(args.number),
            run_id=args.run_id,
            attempt=args.attempt,
            status=args.status,
            exit_code=args.exit_code,
            reason=args.reason,
            started_at=args.started_at,
            finished_at=args.finished_at,
            usage_complete=args.usage_complete,
            usage_lower_bound=args.usage_lower_bound,
        )
        return 0

    if args.command == "backfill":
        statuses = None if args.all_statuses else {"success"}
        print(json.dumps(backfill_usage_tree(args.root, statuses=statuses), ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
