#!/usr/bin/env python3
"""Prepare the method-mode agent input and held-out gold data."""

from __future__ import annotations

import sys
from pathlib import Path


JUDGE_DIR = Path(__file__).resolve().parents[1]
if str(JUDGE_DIR) not in sys.path:
    sys.path.insert(0, str(JUDGE_DIR))

from _split_common import run_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_cli("method-mode"))
