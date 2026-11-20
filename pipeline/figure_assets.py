#!/usr/bin/env python3
"""Materialize figure files next to ScionBench graph and agent artifacts.

Figure paths in extracted graphs point back to ``row4sci/md``.  This module
resolves those paths within the corresponding paper subtree, copies the files
into a sibling ``images/`` directory, deduplicates them by SHA-256, and rewrites
figure references to ``images/<name>``.

The module is both a reusable pipeline helper and a standalone migration tool
for existing ``sci-paper-extrac`` and ``sci-postproc`` trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator
from urllib.parse import unquote, urlparse


SCIONBENCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_EXTRAC_ROOT = Path(os.environ.get("PIPELINE_PAPER_EXTRAC_ROOT", "."))
DEFAULT_POSTPROC_ROOT = Path(os.environ.get("PIPELINE_PREPROCESS_OUTPUT_DIR", "."))
DEFAULT_SOURCE_MD_ROOT = Path(os.environ.get("PIPELINE_SOURCE_MD_ROOT", "."))

IMAGE_SUFFIXES = frozenset(
    {
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".jpeg",
        ".jpg",
        ".png",
        ".svg",
        ".tif",
        ".tiff",
        ".webp",
    }
)
IMAGE_PATH_FIELDS = frozenset(
    {
        "asset_file",
        "asset_path",
        "figure_file",
        "figure_path",
        "image_file",
        "image_path",
        "plot_file",
        "plot_path",
    }
)
JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(\s*(?:<(?P<angle>[^>]+)>|(?P<plain>[^)\s]+))"
    r"(?:\s+[\"'][^)]*[\"'])?\s*\)"
)
MARKDOWN_IMAGE_FILE_RE = re.compile(
    r"\*\*(?:Image|Asset) file:\*\*\s*`(?P<path>[^`]+)`",
    re.IGNORECASE,
)
ASSET_FIELD_RE = re.compile(
    r'"(?:asset_file|asset_path|figure_file|figure_path|image_file|image_path|'
    r'plot_file|plot_path)"\s*:\s*("(?:[^"\\]|\\.)*")'
)
FIGURE_INDEX_START = "<!-- scionbench-figure-assets:start -->"
FIGURE_INDEX_END = "<!-- scionbench-figure-assets:end -->"


class FigureAssetError(RuntimeError):
    """Raised when figure assets cannot be organized safely."""


@dataclass
class MutableFigureReference:
    raw: str
    figure_id: str
    container: dict[str, Any]
    key: str


@dataclass(frozen=True)
class FigureReference:
    raw: str
    figure_id: str = ""


@dataclass
class AssetStats:
    graphs: int = 0
    postproc_runs: int = 0
    targets: int = 0
    artifacts: int = 0
    references: int = 0
    copied: int = 0
    reused: int = 0
    rewritten: int = 0
    unresolved: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "graphs": self.graphs,
            "postproc_runs": self.postproc_runs,
            "targets": self.targets,
            "artifacts": self.artifacts,
            "references": self.references,
            "copied": self.copied,
            "reused_or_deduplicated": self.reused,
            "rewritten": self.rewritten,
            "unresolved_count": len(self.unresolved),
            "unresolved": self.unresolved,
        }


@dataclass(frozen=True)
class PaperContext:
    graph_path: Path
    source_scope: Path
    expected_hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_image_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip().strip("<>"))
    return Path(unquote(parsed.path)).suffix.lower() in IMAGE_SUFFIXES


def _reference_basename(value: str) -> str:
    parsed = urlparse(value.strip().strip("<>"))
    normalized = unquote(parsed.path).replace("\\", "/")
    return PurePosixPath(normalized).name


def _is_image_field(key: Any, value: Any) -> bool:
    if not isinstance(key, str) or not _looks_like_image_path(value):
        return False
    normalized = key.strip().lower()
    return normalized in IMAGE_PATH_FIELDS or normalized.endswith(
        ("_asset", "_file", "_image", "_path")
    )


def _references_in_figure(
    value: Any,
    *,
    figure_id: str,
) -> Iterator[MutableFigureReference]:
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_image_field(key, item):
                yield MutableFigureReference(
                    raw=item,
                    figure_id=figure_id,
                    container=value,
                    key=key,
                )
            elif isinstance(item, (dict, list)):
                yield from _references_in_figure(item, figure_id=figure_id)
    elif isinstance(value, list):
        for item in value:
            yield from _references_in_figure(item, figure_id=figure_id)


def iter_mutable_figure_references(value: Any) -> Iterator[MutableFigureReference]:
    """Yield mutable image fields only from objects below ``figures`` lists."""

    if isinstance(value, dict):
        for key, item in value.items():
            if key == "figures" and isinstance(item, list):
                for figure in item:
                    if not isinstance(figure, dict):
                        continue
                    figure_id = figure.get("id")
                    yield from _references_in_figure(
                        figure,
                        figure_id=figure_id if isinstance(figure_id, str) else "",
                    )
                continue
            if isinstance(item, (dict, list)):
                yield from iter_mutable_figure_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_mutable_figure_references(item)


def figure_references(value: Any) -> list[FigureReference]:
    seen: set[tuple[str, str]] = set()
    result: list[FigureReference] = []
    for reference in iter_mutable_figure_references(value):
        key = (reference.raw, reference.figure_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(FigureReference(reference.raw, reference.figure_id))
    return result


def collect_markdown_figure_references(text: str) -> list[FigureReference]:
    """Collect figure paths from JSON fences and Markdown image syntax."""

    result: list[FigureReference] = []
    seen: set[tuple[str, str]] = set()

    def add(reference: FigureReference) -> None:
        key = (reference.raw, reference.figure_id)
        if key not in seen and _looks_like_image_path(reference.raw):
            seen.add(key)
            result.append(reference)

    parsed_any_fence = False
    for match in JSON_FENCE_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        parsed_any_fence = True
        for reference in figure_references(payload):
            add(reference)

    if not parsed_any_fence:
        for match in ASSET_FIELD_RE.finditer(text):
            try:
                raw = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(raw, str):
                add(FigureReference(raw))

    for match in MARKDOWN_IMAGE_RE.finditer(text):
        raw = match.group("angle") or match.group("plain")
        add(FigureReference(raw))
    for match in MARKDOWN_IMAGE_FILE_RE.finditer(text):
        add(FigureReference(match.group("path")))
    return result


def collect_markdown_image_paths(markdown_path: Path) -> list[Path]:
    """Resolve locally materialized images referenced by one Markdown file."""

    text = markdown_path.read_text(encoding="utf-8")
    result: list[Path] = []
    seen: set[Path] = set()
    for reference in collect_markdown_figure_references(text):
        raw_path = Path(unquote(urlparse(reference.raw).path))
        candidates = [markdown_path.parent / raw_path]
        if raw_path.parent == Path("."):
            candidates.append(markdown_path.parent / "images" / raw_path.name)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
                break
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FigureAssetError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _path_is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


class SourceLocator:
    def __init__(self, source_md_root: Path) -> None:
        self.source_md_root = source_md_root.resolve()
        self._basename_indexes: dict[Path, dict[str, list[Path]]] = {}
        self._hash_cache: dict[Path, str] = {}

    def digest(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._hash_cache:
            self._hash_cache[resolved] = sha256_file(resolved)
        return self._hash_cache[resolved]

    def _index(self, scope: Path) -> dict[str, list[Path]]:
        scope = scope.resolve()
        if scope not in self._basename_indexes:
            index: dict[str, list[Path]] = {}
            if scope.is_dir():
                for path in scope.rglob("*"):
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                        index.setdefault(path.name, []).append(path.resolve())
            self._basename_indexes[scope] = index
        return self._basename_indexes[scope]

    @staticmethod
    def _normalized_parts(value: str) -> list[str]:
        parsed = urlparse(value.strip().strip("<>"))
        parts = [
            part.casefold()
            for part in PurePosixPath(
                unquote(parsed.path).replace("\\", "/")
            ).parts
            if part not in {"", "/", "."} and part.casefold() != "vlm"
        ]
        collapsed: list[str] = []
        for part in parts:
            if not collapsed or collapsed[-1] != part:
                collapsed.append(part)
        return collapsed

    @classmethod
    def _suffix_score(cls, reference: str, candidate: Path, scope: Path) -> int:
        reference_parts = cls._normalized_parts(reference)
        try:
            relative = candidate.resolve().relative_to(scope.resolve())
        except ValueError:
            relative = candidate
        candidate_parts = cls._normalized_parts(relative.as_posix())
        score = 0
        for left, right in zip(reversed(reference_parts), reversed(candidate_parts)):
            if left != right:
                break
            score += 1
        return score

    def _direct_candidates(self, scope: Path, reference: str) -> list[Path]:
        parsed = urlparse(reference.strip().strip("<>"))
        raw_parts = [
            part
            for part in PurePosixPath(
                unquote(parsed.path).replace("\\", "/")
            ).parts
            if part not in {"", "/", ".", ".."}
        ]
        variants: list[list[str]] = [raw_parts]
        without_vlm = [part for part in raw_parts if part.casefold() != "vlm"]
        variants.append(without_vlm)
        collapsed: list[str] = []
        for part in without_vlm:
            if not collapsed or collapsed[-1].casefold() != part.casefold():
                collapsed.append(part)
        variants.append(collapsed)

        result: list[Path] = []
        seen: set[Path] = set()
        for parts in variants:
            candidate = scope.joinpath(*parts).resolve()
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
        return result

    def locate(self, context: PaperContext, reference: str) -> tuple[Path | None, str]:
        basename = _reference_basename(reference)
        if not basename:
            return None, "reference has no filename"

        scope = context.source_scope
        candidates = self._direct_candidates(scope, reference)
        for candidate in self._index(scope).get(basename, []):
            if candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            return None, f"no {basename!r} below {scope}"

        expected = _expected_digest(context.expected_hashes, reference)
        if expected:
            matching = [path for path in candidates if self.digest(path) == expected]
            if len(matching) == 1:
                return matching[0], ""
            if matching:
                candidates = matching

        if len(candidates) == 1:
            return candidates[0], ""

        scores = {
            path: self._suffix_score(reference, path, scope) for path in candidates
        }
        best_score = max(scores.values())
        best = [path for path, score in scores.items() if score == best_score]
        if len(best) == 1:
            return best[0], ""

        digests = {self.digest(path) for path in best}
        if len(digests) == 1:
            return sorted(best)[0], ""
        names = ", ".join(str(path) for path in sorted(best))
        return None, f"ambiguous candidates with different content: {names}"


def _expected_digest(expected_hashes: dict[str, str], reference: str) -> str:
    if reference in expected_hashes:
        return expected_hashes[reference]
    basename = _reference_basename(reference)
    normalized = f"images/{basename}"
    if normalized in expected_hashes:
        return expected_hashes[normalized]
    matches = {
        digest
        for path, digest in expected_hashes.items()
        if _reference_basename(path) == basename
    }
    return next(iter(matches)) if len(matches) == 1 else ""


def _source_scope(
    graph_path: Path,
    graph: dict[str, Any],
    *,
    paper_extrac_root: Path,
    source_md_root: Path,
) -> Path:
    candidates: list[Path] = []
    try:
        relative_parent = graph_path.resolve().relative_to(
            paper_extrac_root.resolve()
        ).parent
        candidates.append(source_md_root / relative_parent)
    except ValueError:
        pass

    paper = graph.get("paper")
    source_dir_raw = paper.get("source_dir") if isinstance(paper, dict) else None
    if isinstance(source_dir_raw, str) and source_dir_raw.strip():
        source_dir = Path(source_dir_raw).expanduser()
        if source_dir.is_dir() and _path_is_below(source_dir, source_md_root):
            candidates.append(source_dir)

        parts = list(source_dir.parts)
        if "md" in parts:
            marker = len(parts) - 1 - list(reversed(parts)).index("md")
            candidates.append(source_md_root.joinpath(*parts[marker + 1 :]))
        for index, part in enumerate(parts):
            if part.endswith("-md") and index + 1 < len(parts):
                candidates.append(
                    source_md_root
                    / part[: -len("-md")]
                    / Path(*parts[index + 1 :])
                )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    matching_dirs = sorted(
        path.resolve()
        for path in source_md_root.rglob(graph_path.parent.name)
        if path.is_dir()
    )
    if len(matching_dirs) == 1:
        return matching_dirs[0]
    if not source_md_root.is_dir():
        raise FigureAssetError(f"source Markdown root does not exist: {source_md_root}")
    return source_md_root.resolve()


class ContextRegistry:
    def __init__(
        self,
        *,
        paper_extrac_root: Path,
        source_md_root: Path,
        locator: SourceLocator,
    ) -> None:
        self.paper_extrac_root = paper_extrac_root.resolve()
        self.source_md_root = source_md_root.resolve()
        self.locator = locator
        self._contexts: dict[Path, PaperContext] = {}

    def get(self, graph_path: Path) -> PaperContext:
        graph_path = graph_path.expanduser().resolve()
        if graph_path in self._contexts:
            return self._contexts[graph_path]
        graph = _load_json(graph_path)
        if not isinstance(graph, dict):
            raise FigureAssetError(f"graph JSON root must be an object: {graph_path}")
        manifest = graph.get("source_manifest")
        image_hashes = manifest.get("image_sha256") if isinstance(manifest, dict) else {}
        expected_hashes = {
            str(path): str(digest)
            for path, digest in image_hashes.items()
            if isinstance(path, str) and isinstance(digest, str)
        } if isinstance(image_hashes, dict) else {}
        context = PaperContext(
            graph_path=graph_path,
            source_scope=_source_scope(
                graph_path,
                graph,
                paper_extrac_root=self.paper_extrac_root,
                source_md_root=self.source_md_root,
            ),
            expected_hashes=expected_hashes,
        )
        self._contexts[graph_path] = context
        return context


class AssetBundle:
    def __init__(
        self,
        artifact_dir: Path,
        *,
        locator: SourceLocator,
        stats: AssetStats,
    ) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.images_dir = self.artifact_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.locator = locator
        self.stats = stats
        self.entries: list[dict[str, str]] = []
        self.artifacts: set[str] = set()
        self._by_digest: dict[str, Path] = {}
        self._name_digest: dict[str, str] = {}
        for path in sorted(self.images_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = self.locator.digest(path)
            self._by_digest.setdefault(digest, path)
            self._name_digest[path.name] = digest
        stats.targets += 1

    def add_artifact(self, path: Path) -> None:
        self.artifacts.add(str(path.resolve()))

    def add(
        self,
        reference: FigureReference,
        *,
        context: PaperContext,
        artifact: Path,
    ) -> str | None:
        self.stats.references += 1
        source, error = self.locator.locate(context, reference.raw)
        if source is None:
            self.stats.unresolved.append(
                {
                    "artifact": str(artifact.resolve()),
                    "graph": str(context.graph_path),
                    "figure_id": reference.figure_id,
                    "reference": reference.raw,
                    "reason": error,
                }
            )
            return None

        digest = self.locator.digest(source)
        target = self._by_digest.get(digest)
        if target is not None:
            self.stats.reused += 1
        else:
            name = source.name
            existing_digest = self._name_digest.get(name)
            if existing_digest is not None and existing_digest != digest:
                stem = Path(name).stem
                suffix = Path(name).suffix
                name = f"{stem}-{digest[:12]}{suffix}"
                counter = 2
                while name in self._name_digest and self._name_digest[name] != digest:
                    name = f"{stem}-{digest[:12]}-{counter}{suffix}"
                    counter += 1
            target = self.images_dir / name
            if target.is_file():
                if self.locator.digest(target) != digest:
                    raise FigureAssetError(
                        f"refusing to overwrite different image content: {target}"
                    )
                self.stats.reused += 1
            else:
                shutil.copy2(source, target)
                self.stats.copied += 1
            self._by_digest[digest] = target
            self._name_digest[target.name] = digest

        relative_target = f"images/{target.name}"
        self.entries.append(
            {
                "artifact": str(artifact.resolve()),
                "figure_id": reference.figure_id,
                "reference": reference.raw,
                "source": str(source),
                "target": relative_target,
                "sha256": digest,
            }
        )
        return relative_target

    def finalize(self) -> None:
        manifest = {
            "artifacts": sorted(self.artifacts),
            "images_dir": str(self.images_dir),
            "references": self.entries,
            "unique_images": sorted(
                {entry["target"] for entry in self.entries}
            ),
        }
        _write_json(self.artifact_dir / "figure-assets-manifest.json", manifest)


def _process_json_artifact(
    path: Path,
    *,
    context: PaperContext,
    bundle: AssetBundle,
) -> list[FigureReference]:
    document = _load_json(path)
    bundle.add_artifact(path)
    bundle.stats.artifacts += 1
    references: list[FigureReference] = []
    changed = False
    for mutable in iter_mutable_figure_references(document):
        reference = FigureReference(mutable.raw, mutable.figure_id)
        references.append(reference)
        target = bundle.add(reference, context=context, artifact=path)
        if target is not None and mutable.raw != target:
            mutable.container[mutable.key] = target
            bundle.stats.rewritten += 1
            changed = True
    if changed:
        _write_json(path, document)
    return references


def _replace_reference(text: str, old: str, new: str) -> str:
    old_encoded = json.dumps(old, ensure_ascii=False)[1:-1]
    new_encoded = json.dumps(new, ensure_ascii=False)[1:-1]
    text = text.replace(old_encoded, new_encoded)
    if old_encoded != old:
        text = text.replace(old, new)
    return text


def _figure_index(mapping: list[tuple[FigureReference, str]]) -> str:
    lines = [FIGURE_INDEX_START, "## Figure Files", ""]
    for reference, target in mapping:
        label = f"`{reference.figure_id}`" if reference.figure_id else "Figure"
        lines.append(f"- {label}: `{target}`")
    lines.append(FIGURE_INDEX_END)
    return "\n".join(lines)


def _replace_figure_index(text: str, index: str) -> str:
    start = text.find(FIGURE_INDEX_START)
    end = text.find(FIGURE_INDEX_END)
    if start >= 0 and end >= start:
        end += len(FIGURE_INDEX_END)
        return text[:start].rstrip() + "\n\n" + index + text[end:]
    return text.rstrip() + "\n\n" + index + "\n"


def _process_markdown_artifact(
    path: Path,
    *,
    context: PaperContext,
    bundle: AssetBundle,
    fallback_references: Iterable[FigureReference] = (),
    add_fallback_index: bool = False,
) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FigureAssetError(f"cannot read Markdown {path}: {exc}") from exc
    bundle.add_artifact(path)
    bundle.stats.artifacts += 1
    embedded = collect_markdown_figure_references(text)
    references = embedded or list(fallback_references)
    mapping: list[tuple[FigureReference, str]] = []
    changed = False
    for reference in references:
        target = bundle.add(reference, context=context, artifact=path)
        if target is None:
            continue
        mapping.append((reference, target))
        if embedded and reference.raw != target:
            updated = _replace_reference(text, reference.raw, target)
            if updated != text:
                text = updated
                bundle.stats.rewritten += 1
                changed = True
    if add_fallback_index and not embedded and mapping:
        updated = _replace_figure_index(text, _figure_index(mapping))
        if updated != text:
            text = updated
            bundle.stats.rewritten += 1
            changed = True
    if changed:
        path.write_text(text, encoding="utf-8")


def _bundle_for(
    bundles: dict[Path, AssetBundle],
    artifact_dir: Path,
    *,
    locator: SourceLocator,
    stats: AssetStats,
) -> AssetBundle:
    key = artifact_dir.resolve()
    if key not in bundles:
        bundles[key] = AssetBundle(key, locator=locator, stats=stats)
    return bundles[key]


def organize_graphs(
    graph_paths: Iterable[Path],
    *,
    registry: ContextRegistry,
    stats: AssetStats,
) -> None:
    for graph_path in sorted({path.resolve() for path in graph_paths}):
        context = registry.get(graph_path)
        bundle = AssetBundle(
            graph_path.parent,
            locator=registry.locator,
            stats=stats,
        )
        _process_json_artifact(graph_path, context=context, bundle=bundle)
        bundle.finalize()
        stats.graphs += 1


def _load_source_manifest(path: Path) -> dict[str, Path]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise FigureAssetError(f"source manifest must be an object: {path}")
    result: dict[str, Path] = {}
    for number, graph_path in payload.items():
        if isinstance(number, str) and isinstance(graph_path, str):
            resolved = Path(graph_path).expanduser().resolve()
            if resolved.is_file():
                result[number] = resolved
    if not result:
        raise FigureAssetError(f"source manifest has no usable graph paths: {path}")
    return result


def _number_for_path(path: Path, root: Path, numbers: set[str]) -> str | None:
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return None
    for part in parts:
        if part in numbers:
            return part
    return None


def organize_postproc_run(
    run_root: Path,
    *,
    registry: ContextRegistry,
    stats: AssetStats,
) -> None:
    run_root = run_root.resolve()
    manifest_path = run_root / "source-manifest.json"
    number_graphs = _load_source_manifest(manifest_path)
    contexts = {number: registry.get(path) for number, path in number_graphs.items()}
    numbers = set(contexts)
    bundles: dict[Path, AssetBundle] = {}

    first_root = run_root / "first"
    first_files = sorted(path for path in first_root.glob("*.json") if path.stem in numbers)
    if first_files:
        first_bundle = _bundle_for(
            bundles, first_root, locator=registry.locator, stats=stats
        )
        for path in first_files:
            _process_json_artifact(
                path,
                context=contexts[path.stem],
                bundle=first_bundle,
            )

    json_references: dict[tuple[str, str], list[FigureReference]] = {}
    json_root = run_root / ".work" / "split-json"
    for path in sorted(json_root.glob("*/*/agent_input.json")):
        number = path.parent.name
        if number not in contexts:
            continue
        mode = path.parent.parent.name
        bundle = _bundle_for(
            bundles, path.parent, locator=registry.locator, stats=stats
        )
        json_references[(mode, number)] = _process_json_artifact(
            path,
            context=contexts[number],
            bundle=bundle,
        )

    md_root = run_root / "test-input" / "md"
    for path in sorted(md_root.glob("*/*/agent_input.md")):
        number = path.parent.name
        if number not in contexts:
            continue
        mode = path.parent.parent.name
        bundle = _bundle_for(
            bundles, path.parent, locator=registry.locator, stats=stats
        )
        _process_markdown_artifact(
            path,
            context=contexts[number],
            bundle=bundle,
            fallback_references=json_references.get((mode, number), ()),
            add_fallback_index=True,
        )

    judge_root = run_root / "judge-input"
    judge_files = sorted(
        path
        for path in judge_root.rglob("*.md")
        if "images" not in path.parts
    ) if judge_root.is_dir() else []
    for path in judge_files:
        number = _number_for_path(path, judge_root, numbers)
        if number is None:
            continue
        bundle = _bundle_for(
            bundles, path.parent, locator=registry.locator, stats=stats
        )
        _process_markdown_artifact(
            path,
            context=contexts[number],
            bundle=bundle,
        )

    for bundle in bundles.values():
        bundle.finalize()
    stats.postproc_runs += 1


def _write_report(path: Path, stats: AssetStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, stats.as_dict())


def organize_article_assets(
    *,
    input_dir: Path,
    preprocess_dir: Path,
    paper_extrac_root: Path,
    source_md_root: Path,
    strict: bool = True,
) -> AssetStats:
    """Organize source graphs and one newly generated postproc run."""

    stats = AssetStats()
    locator = SourceLocator(source_md_root)
    registry = ContextRegistry(
        paper_extrac_root=paper_extrac_root,
        source_md_root=source_md_root,
        locator=locator,
    )
    graph_paths = sorted(input_dir.rglob("paper_graph.json"))
    if not graph_paths:
        raise FigureAssetError(f"no paper_graph.json below {input_dir}")
    organize_graphs(graph_paths, registry=registry, stats=stats)
    organize_postproc_run(preprocess_dir, registry=registry, stats=stats)
    _write_report(preprocess_dir / "figure-assets-report.json", stats)
    if strict and stats.unresolved:
        raise FigureAssetError(
            f"{len(stats.unresolved)} figure reference(s) could not be resolved; "
            f"see {preprocess_dir / 'figure-assets-report.json'}"
        )
    return stats


def organize_existing_trees(
    *,
    paper_extrac_root: Path,
    postproc_root: Path,
    source_md_root: Path,
    strict: bool = False,
) -> AssetStats:
    """Migrate all existing extraction graphs and postproc runs."""

    stats = AssetStats()
    locator = SourceLocator(source_md_root)
    registry = ContextRegistry(
        paper_extrac_root=paper_extrac_root,
        source_md_root=source_md_root,
        locator=locator,
    )
    graph_paths = sorted(paper_extrac_root.rglob("paper_graph.json"))
    organize_graphs(graph_paths, registry=registry, stats=stats)
    for manifest_path in sorted(postproc_root.rglob("source-manifest.json")):
        organize_postproc_run(manifest_path.parent, registry=registry, stats=stats)
    report_path = postproc_root / "figure-assets-report.json"
    _write_report(report_path, stats)
    if strict and stats.unresolved:
        raise FigureAssetError(
            f"{len(stats.unresolved)} figure reference(s) could not be resolved; "
            f"see {report_path}"
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-extrac-root",
        type=Path,
        default=DEFAULT_PAPER_EXTRAC_ROOT,
    )
    parser.add_argument(
        "--postproc-root",
        type=Path,
        default=DEFAULT_POSTPROC_ROOT,
    )
    parser.add_argument(
        "--source-md-root",
        type=Path,
        default=DEFAULT_SOURCE_MD_ROOT,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return an error when any figure reference cannot be resolved",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stats = organize_existing_trees(
            paper_extrac_root=args.paper_extrac_root.expanduser().resolve(),
            postproc_root=args.postproc_root.expanduser().resolve(),
            source_md_root=args.source_md_root.expanduser().resolve(),
            strict=args.strict,
        )
    except (FigureAssetError, OSError) as exc:
        print(f"figure asset error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
