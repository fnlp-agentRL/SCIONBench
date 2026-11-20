"""Copy each paper's source data directory beside generated JSON inputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class DataAssetError(RuntimeError):
    """Raised when source data cannot be mapped or copied safely."""


@dataclass(frozen=True)
class DataCopyStats:
    source_directories: int
    copied_targets: int
    missing_numbers: tuple[str, ...]


def _load_source_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataAssetError(f"source manifest does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DataAssetError(f"cannot read source manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataAssetError(f"source manifest must be a JSON object: {path}")
    return payload


def _validate_component(value: str, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise DataAssetError(f"invalid {label} path component: {value!r}")
    return value


def _remove_existing(path: Path) -> None:
    try:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except OSError as exc:
        raise DataAssetError(
            f"cannot replace existing data destination {path}: {exc}"
        ) from exc


def copy_source_data_directories(
    *,
    preprocess_dir: Path,
    numbers: Iterable[str],
    modes: Iterable[str],
) -> DataCopyStats:
    """Mirror source ``data/`` trees beside every Markdown agent input."""

    preprocess_dir = Path(preprocess_dir)
    manifest_path = preprocess_dir / "source-manifest.json"
    manifest = _load_source_manifest(manifest_path)
    requested_numbers = tuple(
        _validate_component(str(number), "article number") for number in numbers
    )
    requested_modes = tuple(_validate_component(str(mode), "mode") for mode in modes)

    copied_sources = 0
    copied_targets = 0
    missing_numbers: list[str] = []

    for number in requested_numbers:
        graph_value = manifest.get(number)
        if not isinstance(graph_value, str) or not graph_value.strip():
            raise DataAssetError(
                f"source manifest has no graph path for article {number}: {manifest_path}"
            )
        graph_path = Path(graph_value).expanduser()
        if not graph_path.is_absolute():
            graph_path = manifest_path.parent / graph_path
        graph_path = graph_path.resolve()
        if not graph_path.is_file():
            raise DataAssetError(
                f"source graph for article {number} does not exist: {graph_path}"
            )

        source_data = graph_path.parent / "data"
        destinations = tuple(
            preprocess_dir / "test-input" / "md" / mode / number / "data"
            for mode in requested_modes
        )

        if not source_data.exists():
            if source_data.is_symlink():
                raise DataAssetError(
                    f"source data directory is a broken symlink: {source_data}"
                )
            for destination in destinations:
                _remove_existing(destination)
            missing_numbers.append(number)
            continue
        if not source_data.is_dir():
            raise DataAssetError(f"source data path is not a directory: {source_data}")

        copied_sources += 1
        for destination in destinations:
            _remove_existing(destination)
            try:
                shutil.copytree(
                    source_data,
                    destination,
                    symlinks=True,
                    copy_function=shutil.copy2,
                )
            except OSError as exc:
                raise DataAssetError(
                    f"cannot copy source data {source_data} to {destination}: {exc}"
                ) from exc
            copied_targets += 1

    return DataCopyStats(
        source_directories=copied_sources,
        copied_targets=copied_targets,
        missing_numbers=tuple(missing_numbers),
    )
