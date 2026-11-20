#!/usr/bin/env python3
"""Write one test-task duration and retry summary atomically."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_analysis(
    analysis_file: Path,
    *,
    total_elapsed_seconds: float,
    attempt_count: int,
    status: str,
    started_at: str | None,
    finished_at: str | None = None,
    format_name: str = "unknown",
    mode: str = "unknown",
    number: str = "unknown",
    run_id: str = "",
) -> dict[str, object]:
    """Write the summary for one complete supervisor run."""

    elapsed = max(0.0, float(total_elapsed_seconds))
    attempts = max(0, int(attempt_count))
    payload: dict[str, object] = {
        "total_elapsed_seconds": round(elapsed, 3),
        "retry_count": max(0, attempts - 1),
        "attempt_count": attempts,
        "status": str(status),
        "started_at": started_at,
        "finished_at": finished_at or now(),
        "format": str(format_name),
        "mode": str(mode),
        "number": str(number),
        "run_id": str(run_id),
    }
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = analysis_file.with_name(f".{analysis_file.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, analysis_file)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-file", type=Path, required=True)
    parser.add_argument("--started-monotonic-ns", type=int, required=True)
    parser.add_argument("--started-at")
    parser.add_argument("--attempt-count", type=int, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--format", dest="format_name", default="unknown")
    parser.add_argument("--mode", default="unknown")
    parser.add_argument("--number", default="unknown")
    parser.add_argument("--run-id", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.started_monotonic_ns < 0:
        raise SystemExit("started monotonic time cannot be negative")
    if args.attempt_count < 0:
        raise SystemExit("attempt count cannot be negative")
    elapsed_seconds = max(
        0.0,
        (time.monotonic_ns() - args.started_monotonic_ns) / 1_000_000_000,
    )
    write_analysis(
        args.analysis_file,
        total_elapsed_seconds=elapsed_seconds,
        attempt_count=args.attempt_count,
        status=args.status,
        started_at=args.started_at,
        format_name=args.format_name,
        mode=args.mode,
        number=str(args.number),
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
