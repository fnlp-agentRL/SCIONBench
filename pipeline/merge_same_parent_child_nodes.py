#!/usr/bin/env python3
"""Merge graph nodes that have the same immediate parents and children.

The graph relation used by this script is intentionally independent of the
edge ``relation`` value: for every edge ``source -> target``, ``source`` is a
parent of ``target`` and ``target`` is a child of ``source``.

Nodes are grouped by the exact pair ``(parent_ids, child_ids)`` calculated from
all edges in the input graph.  For a group containing more than one node, one
new node is emitted.  Its regular node fields are copied from the first node
in the group so that the result keeps the original node shape.  The complete,
unchanged original node objects are additionally stored in ``merged_nodes``;
therefore conflicting fields are never overwritten or discarded.

Usage:
    1. Fill INPUT_DIR and OUTPUT_DIR below, then run this file; or
    2. Pass --input-dir and --output-dir explicitly.

The output directory receives:
    paper_graph_unit.json  The graph after node IDs and edge endpoints are
                            remapped.
    record.json             The groups and original-to-merged ID mapping.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


# ============================ 可填写配置区 ============================
# 也可以在运行时通过 --input-dir / --output-dir 覆盖这两个值。
INPUT_DIR = ""
OUTPUT_DIR = ""

INPUT_FILENAME = "paper_graph.json"
OUTPUT_GRAPH_FILENAME = "paper_graph_unit.json"
OUTPUT_RECORD_FILENAME = "record.json"
MERGED_ID_PREFIX = "merged_node_"

MERGE_RULE_NAME = "same_parent_and_child"


class GraphFormatError(ValueError):
    """Raised when the input does not satisfy the minimum graph contract."""


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphFormatError(f"{location} must be a non-empty string")
    return value


def _validate_graph(graph: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and return the node and edge arrays used by the algorithm."""

    if not isinstance(graph, dict):
        raise GraphFormatError("the JSON root must be an object")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list):
        raise GraphFormatError("$.nodes must be an array")
    if not isinstance(edges, list):
        raise GraphFormatError("$.edges must be an array")

    node_ids: set[str] = set()
    normalized_nodes: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise GraphFormatError(f"$.nodes[{index}] must be an object")
        node_id = _require_string(node.get("id"), f"$.nodes[{index}].id")
        if node_id in node_ids:
            raise GraphFormatError(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        normalized_nodes.append(node)

    normalized_edges: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise GraphFormatError(f"$.edges[{index}] must be an object")
        source = _require_string(edge.get("source"), f"$.edges[{index}].source")
        target = _require_string(edge.get("target"), f"$.edges[{index}].target")
        if source not in node_ids:
            raise GraphFormatError(
                f"$.edges[{index}].source refers to unknown node id: {source}"
            )
        if target not in node_ids:
            raise GraphFormatError(
                f"$.edges[{index}].target refers to unknown node id: {target}"
            )
        normalized_edges.append(edge)

    return normalized_nodes, normalized_edges


def _build_adjacency(
    nodes: Iterable[dict[str, Any]], edges: Iterable[dict[str, Any]]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build parent and child sets using every directed edge."""

    node_ids = [node["id"] for node in nodes]
    parents = {node_id: set() for node_id in node_ids}
    children = {node_id: set() for node_id in node_ids}

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        parents[target].add(source)
        children[source].add(target)

    return parents, children


def _signature(
    node_id: str,
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a deterministic signature for one node."""

    return tuple(sorted(parents[node_id])), tuple(sorted(children[node_id]))


def _ordered_groups(
    nodes: list[dict[str, Any]],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> list[list[str]]:
    """Group IDs by signature while retaining input-node order."""

    groups_by_signature: dict[
        tuple[tuple[str, ...], tuple[str, ...]], list[str]
    ] = defaultdict(list)
    for node in nodes:
        node_id = node["id"]
        groups_by_signature[_signature(node_id, parents, children)].append(node_id)

    seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    ordered: list[list[str]] = []
    for node in nodes:
        node_id = node["id"]
        signature = _signature(node_id, parents, children)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        ordered.append(groups_by_signature[signature])
    return ordered


def _new_merged_id(
    prefix: str,
    counter: int,
    occupied_ids: set[str],
) -> tuple[str, int]:
    """Create a deterministic ID that cannot collide with an input ID."""

    while True:
        candidate = f"{prefix}{counter:03d}"
        counter += 1
        if candidate not in occupied_ids:
            occupied_ids.add(candidate)
            return candidate, counter


def _remap_reference(value: Any, node_id_mapping: dict[str, str]) -> Any:
    """Remap a scalar or list of node IDs while preserving its value shape."""

    if isinstance(value, str):
        return node_id_mapping.get(value, value)
    if isinstance(value, list):
        return [
            node_id_mapping.get(item, item) if isinstance(item, str) else deepcopy(item)
            for item in value
        ]
    return deepcopy(value)


def _remap_known_node_references(
    record: dict[str, Any],
    fields: set[str],
    node_id_mapping: dict[str, str],
) -> dict[str, Any]:
    """Copy a record and remap only fields known to contain node IDs."""

    result = deepcopy(record)
    for field in fields:
        if field in result:
            result[field] = _remap_reference(result[field], node_id_mapping)
    return result


def _unique_in_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def merge_graph(
    graph: dict[str, Any],
    *,
    merged_id_prefix: str = MERGED_ID_PREFIX,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge same-signature nodes and return ``(output_graph, record)``.

    Grouping is performed once from the original graph.  This makes the
    result deterministic and ensures that a node is merged only when its
    original parent and child sets are exactly equal to another node's sets.
    """

    nodes, edges = _validate_graph(graph)
    node_by_id = {node["id"]: node for node in nodes}
    original_node_ids = [node["id"] for node in nodes]
    original_id_set = set(original_node_ids)
    parents, children = _build_adjacency(nodes, edges)
    groups = _ordered_groups(nodes, parents, children)

    node_id_mapping: dict[str, str] = {}
    occupied_ids = set(original_id_set)
    next_merged_number = 1

    for group in groups:
        if len(group) == 1:
            node_id_mapping[group[0]] = group[0]
            continue
        merged_id, next_merged_number = _new_merged_id(
            merged_id_prefix, next_merged_number, occupied_ids
        )
        for original_id in group:
            node_id_mapping[original_id] = merged_id

    output_nodes: list[dict[str, Any]] = []
    merge_records: list[dict[str, Any]] = []

    for group in groups:
        first_id = group[0]
        if len(group) == 1:
            # Keep a non-merged node in its original shape.  Only reference
            # fields are remapped when another node it points to was merged.
            output_node = deepcopy(node_by_id[first_id])
            if "parent_id" in output_node:
                output_node["parent_id"] = _remap_reference(
                    output_node["parent_id"], node_id_mapping
                )
            if "parent_ids" in output_node:
                output_node["parent_ids"] = _remap_reference(
                    output_node["parent_ids"], node_id_mapping
                )
            if "depends_on" in output_node:
                output_node["depends_on"] = _remap_reference(
                    output_node["depends_on"], node_id_mapping
                )
            output_nodes.append(output_node)
            continue

        merged_id = node_id_mapping[first_id]
        original_nodes = [deepcopy(node_by_id[node_id]) for node_id in group]

        # Start with the first node to retain the same top-level node shape.
        # Every original object is also copied below without any modification.
        merged_node = deepcopy(original_nodes[0])
        merged_node["id"] = merged_id

        # If optional fields occur only on later members, retain them at the
        # top level too.  Their exact original values remain in merged_nodes.
        for original_node in original_nodes[1:]:
            for key, value in original_node.items():
                if key not in merged_node:
                    merged_node[key] = deepcopy(value)

        mapped_parent_ids = sorted(
            {
                node_id_mapping[parent_id]
                for member_id in group
                for parent_id in parents[member_id]
            }
        )
        mapped_child_ids = sorted(
            {
                node_id_mapping[child_id]
                for member_id in group
                for child_id in children[member_id]
            }
        )
        original_parent_ids = sorted(
            {
                parent_id
                for member_id in group
                for parent_id in parents[member_id]
            }
        )
        original_child_ids = sorted(
            {
                child_id
                for member_id in group
                for child_id in children[member_id]
            }
        )

        # These are graph-level references in the merged representation.  The
        # unmodified source values are available in merged_node["merged_nodes"].
        merged_node["parent_ids"] = mapped_parent_ids
        if "depends_on" in merged_node or any(
            "depends_on" in original_node for original_node in original_nodes
        ):
            merged_depends_on = _unique_in_order(
                node_id_mapping.get(depends_on, depends_on)
                for original_node in original_nodes
                for depends_on in original_node.get("depends_on", [])
                if isinstance(depends_on, str)
            )
            merged_node["depends_on"] = merged_depends_on
        if "evidence_refs" in merged_node or any(
            "evidence_refs" in original_node for original_node in original_nodes
        ):
            merged_node["evidence_refs"] = _unique_in_order(
                evidence_ref
                for original_node in original_nodes
                for evidence_ref in original_node.get("evidence_refs", [])
                if isinstance(evidence_ref, str)
            )

        merged_node["merge_info"] = {
            "rule": MERGE_RULE_NAME,
            "original_node_ids": list(group),
            "original_parent_ids": original_parent_ids,
            "original_child_ids": original_child_ids,
            "parent_ids_after_merge": mapped_parent_ids,
            "child_ids_after_merge": mapped_child_ids,
        }
        # This list contains exact deep copies, including each original id and
        # every original field.  It is the lossless part of the merge output.
        merged_node["merged_nodes"] = original_nodes
        output_nodes.append(merged_node)

        merge_records.append(
            {
                "merged_node_id": merged_id,
                "original_node_ids": list(group),
                "original_parent_ids": original_parent_ids,
                "original_child_ids": original_child_ids,
                "parent_ids_after_merge": mapped_parent_ids,
                "child_ids_after_merge": mapped_child_ids,
            }
        )

    output_graph = deepcopy(graph)
    output_graph["nodes"] = output_nodes
    output_graph["edges"] = []
    for edge in edges:
        output_edge = deepcopy(edge)
        output_edge["source"] = node_id_mapping[edge["source"]]
        output_edge["target"] = node_id_mapping[edge["target"]]
        output_graph["edges"].append(output_edge)

    if isinstance(output_graph.get("paper"), dict):
        output_graph["paper"] = _remap_known_node_references(
            output_graph["paper"],
            {"root_motivation_id", "overall_conclusion_id"},
            node_id_mapping,
        )

    if isinstance(output_graph.get("method_relations"), list):
        output_graph["method_relations"] = [
            _remap_known_node_references(
                item,
                {"method_a", "method_b"},
                node_id_mapping,
            )
            if isinstance(item, dict)
            else deepcopy(item)
            for item in output_graph["method_relations"]
        ]

    if isinstance(output_graph.get("method_coverage"), list):
        remapped_coverage = []
        for coverage in output_graph["method_coverage"]:
            if not isinstance(coverage, dict):
                remapped_coverage.append(deepcopy(coverage))
                continue
            remapped = _remap_known_node_references(
                coverage, {"method_id"}, node_id_mapping
            )
            if isinstance(remapped.get("entries"), list):
                remapped["entries"] = [
                    _remap_known_node_references(
                        entry,
                        {"design_id", "result_id", "conclusion_id"},
                        node_id_mapping,
                    )
                    if isinstance(entry, dict)
                    else deepcopy(entry)
                    for entry in remapped["entries"]
                ]
            remapped_coverage.append(remapped)
        output_graph["method_coverage"] = remapped_coverage

    record = {
        "record_schema": "scionbench.same_parent_child_merge.v1",
        "merge_rule": {
            "name": MERGE_RULE_NAME,
            "parent_definition": "edge.source is the parent of edge.target",
            "relation_definition": "every edge counts regardless of edge.relation",
            "group_definition": "exactly equal immediate parent and child ID sets in the original graph",
        },
        "summary": {
            "original_node_count": len(nodes),
            "output_node_count": len(output_nodes),
            "merged_group_count": len(merge_records),
            "original_nodes_in_merged_groups": sum(
                len(group["original_node_ids"]) for group in merge_records
            ),
            "nodes_removed_by_merge": sum(len(group["original_node_ids"]) - 1 for group in merge_records),
            "original_edge_count": len(edges),
            "output_edge_count": len(output_graph["edges"]),
        },
        "merged_groups": merge_records,
        "node_id_mapping": node_id_mapping,
        "unchanged_node_ids": [
            node_id for node_id in original_node_ids if node_id_mapping[node_id] == node_id
        ],
    }
    return output_graph, record


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError as exc:
        raise GraphFormatError(f"input file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GraphFormatError(
            f"input file is not valid JSON: {path} (line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(value, dict):
        raise GraphFormatError("the JSON root must be an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge nodes with identical immediate parent and child sets."
    )
    parser.add_argument(
        "--input-dir",
        default=INPUT_DIR,
        help="Directory containing paper_graph.json (or the configured INPUT_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for paper_graph_unit.json and record.json (or OUTPUT_DIR).",
    )
    parser.add_argument(
        "--input-name",
        default=INPUT_FILENAME,
        help=f"Input filename inside --input-dir (default: {INPUT_FILENAME}).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Optional direct input path; when set, --input-dir and --input-name are ignored.",
    )
    parser.add_argument(
        "--merged-id-prefix",
        default=MERGED_ID_PREFIX,
        help=f"Prefix for generated merged node IDs (default: {MERGED_ID_PREFIX}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.input_file is not None:
        input_path = args.input_file
    else:
        if not args.input_dir:
            print(
                "请在脚本顶部填写 INPUT_DIR，或运行时传入 --input-dir。",
                file=sys.stderr,
            )
            return 2
        input_path = Path(args.input_dir) / args.input_name

    if not args.output_dir:
        print(
            "请在脚本顶部填写 OUTPUT_DIR，或运行时传入 --output-dir。",
            file=sys.stderr,
        )
        return 2

    try:
        graph = _load_json(input_path)
        output_graph, record = merge_graph(
            graph, merged_id_prefix=args.merged_id_prefix
        )
        output_dir = Path(args.output_dir)
        output_graph_path = output_dir / OUTPUT_GRAPH_FILENAME
        record_path = output_dir / OUTPUT_RECORD_FILENAME
        _write_json(output_graph_path, output_graph)
        record["files"] = {
            "input": str(input_path),
            "merged_graph": str(output_graph_path),
            "record": str(record_path),
        }
        _write_json(record_path, record)
    except (GraphFormatError, OSError) as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1

    print(f"合并完成：{len(record['merged_groups'])} 个节点组。")
    print(f"合并后的图：{output_graph_path}")
    print(f"合并记录：{record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
