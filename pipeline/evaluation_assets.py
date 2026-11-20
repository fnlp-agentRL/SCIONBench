"""Relocate split evaluation files into their numbered judge-input scopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class EvaluationMoveError(RuntimeError):
    """Raised when an expected evaluation output cannot be relocated."""


@dataclass(frozen=True)
class EvaluationMoveStats:
    moved_files: int


def _validate_component(value: str, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise EvaluationMoveError(f"invalid {label} path component: {value!r}")
    return value


def move_evaluation_files(
    *,
    preprocess_dir: Path,
    numbers: Iterable[str],
    modes: Iterable[str],
) -> EvaluationMoveStats:
    """Move each split ``evaluation.json`` to ``judge-input/<number>/<mode>``."""

    preprocess_dir = Path(preprocess_dir)
    requested_numbers = tuple(
        _validate_component(str(number), "article number") for number in numbers
    )
    requested_modes = tuple(_validate_component(str(mode), "mode") for mode in modes)
    moves: list[tuple[Path, Path]] = []

    # Validate the complete set before moving anything so a missing split output
    # does not leave an avoidable partially relocated preprocessing result.
    for number in requested_numbers:
        for mode in requested_modes:
            source = (
                preprocess_dir
                / ".work"
                / "split-json"
                / mode
                / number
                / "evaluation.json"
            )
            destination = (
                preprocess_dir
                / "judge-input"
                / number
                / mode
                / "evaluation.json"
            )
            if not source.is_file():
                raise EvaluationMoveError(
                    f"evaluation output does not exist for {mode}/{number}: {source}"
                )
            if destination.is_dir():
                raise EvaluationMoveError(
                    f"evaluation destination is a directory: {destination}"
                )
            moves.append((source, destination))

    for source, destination in moves:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        except OSError as exc:
            raise EvaluationMoveError(
                f"cannot move evaluation output {source} to {destination}: {exc}"
            ) from exc

    return EvaluationMoveStats(moved_files=len(moves))
