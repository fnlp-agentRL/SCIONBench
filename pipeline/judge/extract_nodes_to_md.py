#!/usr/bin/env python3
"""Extract nodes from ScionBench evaluation / paper-graph JSON into per-node Markdown.

Reads one or more JSON files (e.g. judge-split ``evaluation.json``), collects
nodes from both ``gold_nodes`` and ``nodes`` (deduplicated by ``id``), keeps only
``id`` / ``node_type`` / ``article_content`` / ``evaluation_plan``, and writes
one Markdown file per node under stage subfolders of the output directory:

    <OUTPUT_DIR>/
      method/
      design/        # research_stage == experiment_design
      experiment/
      conclusion/

Edges, evidence, method_relations, method_coverage and other non-node fields
are ignored.

Fill ``INPUT_DIR`` / ``OUTPUT_DIR`` below, or pass ``--input-dir`` /
``--output-dir`` on the command line (CLI overrides the constants).

Examples:
    python3 extract_nodes_to_md.py

    python3 extract_nodes_to_md.py \\
        --input-dir /path/to/json_dir \\
        --output-dir /path/to/out_dir

    python3 extract_nodes_to_md.py \\
        --input-file /path/to/evaluation.json \\
        --output-dir /path/to/out_dir
    使用单斜杠即可
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fill these paths before running (CLI flags override when provided).
# ---------------------------------------------------------------------------
INPUT_DIR = ""
OUTPUT_DIR = ""

KEEP_FIELDS = ("id", "node_type", "article_content", "evaluation_plan")

# research_stage -> output subdirectory name
STAGE_DIR = {
    "method": "method",
    "experiment_design": "design",
    "experiment": "experiment",
    "conclusion": "conclusion",
}

NODE_LIST_KEYS = ("gold_nodes", "nodes")


def slim_node(node: dict[str, Any]) -> dict[str, Any]:
    """Keep only the requested node fields, in stable order."""
    slim: dict[str, Any] = {}
    for key in KEEP_FIELDS:
        if key in node:
            slim[key] = node[key]
    return slim


def stage_of(node: dict[str, Any]) -> str | None:
    node_type = node.get("node_type")
    if not isinstance(node_type, dict):
        return None
    stage = node_type.get("research_stage")
    return stage if isinstance(stage, str) and stage else None


def collect_nodes_from_obj(obj: Any) -> list[dict[str, Any]]:
    """Collect raw node dicts from a loaded JSON object."""
    found: list[dict[str, Any]] = []

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "id" in item and "node_type" in item:
                found.append(item)
        return found

    if not isinstance(obj, dict):
        return found

    # Prefer explicit node lists used by judge-split evaluation.json.
    saw_list = False
    for key in NODE_LIST_KEYS:
        value = obj.get(key)
        if isinstance(value, list):
            saw_list = True
            for item in value:
                if isinstance(item, dict) and "id" in item:
                    found.append(item)

    if saw_list:
        return found

    # Fallback: single node object, or a bare paper_graph-style {"nodes": ...}
    # already handled above; also accept a lone node dict.
    if "id" in obj and "node_type" in obj:
        found.append(obj)

    return found


def dedupe_nodes(
    nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Deduplicate by id. First occurrence wins; later conflicts are warned."""
    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            warnings.append("skip node without string id")
            continue
        if node_id not in by_id:
            by_id[node_id] = node
            ordered.append(node)
            continue
        if json.dumps(by_id[node_id], sort_keys=True, ensure_ascii=False) != json.dumps(
            node, sort_keys=True, ensure_ascii=False
        ):
            warnings.append(
                f"duplicate id {node_id!r} differs from first copy; keeping first"
            )
        # identical or conflicting: still only keep the first

    return ordered, warnings


def safe_filename(node_id: str) -> str:
    """Map a node id to a filesystem-safe markdown filename."""
    name = node_id.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    if not name:
        name = "unnamed_node"
    if not name.endswith(".md"):
        name = f"{name}.md"
    return name


def render_markdown(slim: dict[str, Any]) -> str:
    node_id = slim.get("id", "unknown")
    stage = stage_of(slim) or "unknown"
    payload = json.dumps(slim, indent=2, ensure_ascii=False)
    return (
        f"# {node_id}\n\n"
        f"- research_stage: `{stage}`\n\n"
        f"```json\n{payload}\n```\n"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_input_paths(input_dir: str, input_file: str | None) -> list[Path]:
    if input_file:
        path = Path(input_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
        return [path]

    if not input_dir or not str(input_dir).strip():
        raise ValueError(
            "INPUT_DIR is empty. Set INPUT_DIR at the top of the script "
            "or pass --input-dir / --input-file."
        )

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory not found: {root}")

    paths = sorted(p for p in root.rglob("*.json") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no .json files under: {root}")
    return paths


def write_nodes(nodes: list[dict[str, Any]], output_dir: Path) -> dict[str, int]:
    counts = {name: 0 for name in STAGE_DIR.values()}
    skipped: list[str] = []

    for node in nodes:
        slim = slim_node(node)
        stage = stage_of(slim)
        if stage not in STAGE_DIR:
            skipped.append(f"{slim.get('id')}: unsupported research_stage={stage!r}")
            continue

        subdir = output_dir / STAGE_DIR[stage]
        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / safe_filename(str(slim["id"]))
        out_path.write_text(render_markdown(slim), encoding="utf-8")
        counts[STAGE_DIR[stage]] += 1

    if skipped:
        print("skipped nodes:", file=sys.stderr)
        for line in skipped:
            print(f"  - {line}", file=sys.stderr)

    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract gold_nodes/nodes from ScionBench JSON into per-node "
            "Markdown files under method/design/experiment/conclusion."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory of JSON files to scan recursively (overrides INPUT_DIR).",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Single JSON file (overrides --input-dir / INPUT_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides OUTPUT_DIR).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_dir = args.input_dir if args.input_dir is not None else INPUT_DIR
    output_dir_raw = args.output_dir if args.output_dir is not None else OUTPUT_DIR

    if not output_dir_raw or not str(output_dir_raw).strip():
        print(
            "OUTPUT_DIR is empty. Set OUTPUT_DIR at the top of the script "
            "or pass --output-dir.",
            file=sys.stderr,
        )
        return 2

    try:
        input_paths = resolve_input_paths(input_dir, args.input_file)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir = Path(output_dir_raw).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_nodes: list[dict[str, Any]] = []
    for path in input_paths:
        try:
            obj = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"failed to read {path}: {exc}", file=sys.stderr)
            return 1
        collected = collect_nodes_from_obj(obj)
        print(f"read {path}: {len(collected)} node record(s)")
        raw_nodes.extend(collected)

    nodes, warnings = dedupe_nodes(raw_nodes)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"unique nodes after dedupe: {len(nodes)}")

    counts = write_nodes(nodes, output_dir)
    total = sum(counts.values())
    print(f"wrote {total} markdown file(s) under {output_dir}")
    for name in ("method", "design", "experiment", "conclusion"):
        print(f"  {name}/: {counts[name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
