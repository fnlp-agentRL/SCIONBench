#!/usr/bin/env python3
"""Merge all branch-synthesis conclusions into one overall conclusion.

The script keeps the existing ``overall_conclusion`` node as the surviving
node.  Branch text, data, evidence references and evaluation metrics are
folded into it; branch-to-overall graph edges are replaced by direct edges to
the surviving node, and the branch nodes are removed.  All unrelated top-level
records and edge fields are copied unchanged.

Usage:
    python3 merge_branch_conclusions.py INPUT.json OUTPUT.json

If OUTPUT.json is omitted, ``INPUT_stem_merged.json`` is written next to the
input file.  The input is never overwritten unless the output path explicitly
equals the input path.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable


class MergeError(ValueError):
    """Raised when the input graph cannot be merged unambiguously."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise MergeError(f"input file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MergeError(
            f"input file is not valid JSON: {path} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    except OSError as exc:
        raise MergeError(f"cannot read input file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError("the JSON root must be an object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise MergeError(f"cannot write output file {path}: {exc}") from exc


def _unique_extend(target: list[Any], values: Iterable[Any]) -> None:
    """Append deep-copied values while preserving first-seen order."""

    for value in values:
        if value not in target:
            target.append(copy.deepcopy(value))


def _remap_ids(
    values: Any,
    branch_ids: set[str],
    overall_id: str,
    *,
    drop_overall: bool = True,
) -> Any:
    if not isinstance(values, list):
        return copy.deepcopy(values)
    result: list[Any] = []
    for value in values:
        mapped = overall_id if value in branch_ids else value
        if drop_overall and mapped == overall_id:
            # A surviving overall node must not refer to itself as a parent or
            # dependency after its branch parents have been absorbed.
            continue
        if mapped not in result:
            result.append(copy.deepcopy(mapped))
    return result


def _merge_completeness(
    overall_content: dict[str, Any], branch_content: dict[str, Any]
) -> None:
    overall = overall_content.get("completeness")
    branch = branch_content.get("completeness")
    if not isinstance(overall, dict) or not isinstance(branch, dict):
        return

    for key, branch_value in branch.items():
        if key not in overall:
            overall[key] = copy.deepcopy(branch_value)
        elif isinstance(overall[key], list) and isinstance(branch_value, list):
            _unique_extend(overall[key], branch_value)
        elif isinstance(overall[key], bool) and isinstance(branch_value, bool):
            # The merged content is complete only if every merged source was
            # complete.
            overall[key] = overall[key] and branch_value


def _merge_article_content(
    overall_node: dict[str, Any], branch_node: dict[str, Any]
) -> None:
    overall_content = overall_node.get("article_content")
    branch_content = branch_node.get("article_content")
    if not isinstance(overall_content, dict) or not isinstance(branch_content, dict):
        raise MergeError(
            f"node {branch_node.get('id', '<unknown>')} has invalid article_content"
        )

    overall_text = overall_content.get("text")
    branch_text = branch_content.get("text")
    if not isinstance(overall_text, dict) or not isinstance(branch_text, dict):
        raise MergeError(
            f"node {branch_node.get('id', '<unknown>')} has invalid article_content.text"
        )

    branch_label = f"[Merged from {branch_node['id']}"
    if branch_node.get("branch_id"):
        branch_label += f" / {branch_node['branch_id']}"
    branch_label += "] "
    branch_statement = branch_text.get("statement")
    if isinstance(branch_statement, str) and branch_statement:
        overall_text.setdefault("facts", [])
        if isinstance(overall_text["facts"], list):
            _unique_extend(
                overall_text["facts"],
                [f"{branch_label}Branch statement: {branch_statement}"],
            )

    for key in ("facts", "definitions", "conditions", "reported_interpretation"):
        branch_values = branch_text.get(key, [])
        if not isinstance(branch_values, list):
            continue
        overall_text.setdefault(key, [])
        if isinstance(overall_text[key], list):
            _unique_extend(
                overall_text[key],
                [f"{branch_label}{value}" for value in branch_values],
            )

    overall_data = overall_content.get("data")
    branch_data = branch_content.get("data")
    if isinstance(overall_data, dict) and isinstance(branch_data, dict):
        for key in ("tables", "figures", "numeric_records", "formulas"):
            branch_values = branch_data.get(key, [])
            if not isinstance(branch_values, list):
                continue
            overall_data.setdefault(key, [])
            if isinstance(overall_data[key], list):
                _unique_extend(overall_data[key], branch_values)

    _merge_completeness(overall_content, branch_content)


def _merge_evaluation_plan(
    overall_node: dict[str, Any], branch_node: dict[str, Any]
) -> None:
    overall_plan = overall_node.get("evaluation_plan")
    branch_plan = branch_node.get("evaluation_plan")
    if not isinstance(overall_plan, dict) or not isinstance(branch_plan, dict):
        return

    for key in ("benchmark_metrics", "paper_scientific_metrics"):
        branch_values = branch_plan.get(key, [])
        if not isinstance(branch_values, list):
            continue
        overall_plan.setdefault(key, [])
        if isinstance(overall_plan[key], list):
            _unique_extend(overall_plan[key], branch_values)

    # Preserve any future evaluation-plan fields without replacing the
    # surviving overall plan's value on a key collision.
    for key, value in branch_plan.items():
        if key not in overall_plan:
            overall_plan[key] = copy.deepcopy(value)


def _merge_node_content(
    overall_node: dict[str, Any], branch_nodes: list[dict[str, Any]]
) -> None:
    parent_source = overall_node.get("parent_ids", [])
    depends_source = overall_node.get("depends_on", [])
    evidence_source = overall_node.get("evidence_refs", [])
    parent_ids: list[Any] = list(parent_source) if isinstance(parent_source, list) else []
    depends_on: list[Any] = list(depends_source) if isinstance(depends_source, list) else []
    evidence_refs: list[Any] = list(evidence_source) if isinstance(evidence_source, list) else []

    for branch_node in branch_nodes:
        branch_parents = branch_node.get("parent_ids", [])
        branch_depends = branch_node.get("depends_on", [])
        branch_evidence = branch_node.get("evidence_refs", [])
        if isinstance(branch_parents, list):
            parent_ids.extend(branch_parents)
        if isinstance(branch_depends, list):
            depends_on.extend(branch_depends)
        if isinstance(branch_evidence, list):
            evidence_refs.extend(branch_evidence)
        _merge_article_content(overall_node, branch_node)
        _merge_evaluation_plan(overall_node, branch_node)

    branch_ids = {node["id"] for node in branch_nodes}
    overall_id = overall_node["id"]
    overall_node["parent_ids"] = _remap_ids(parent_ids, branch_ids, overall_id)
    overall_node["depends_on"] = _remap_ids(depends_on, branch_ids, overall_id)
    overall_node["evidence_refs"] = _remap_ids(
        evidence_refs, set(), overall_id, drop_overall=False
    )


def _redirect_edges(
    edges: list[Any], branch_ids: set[str], overall_id: str
) -> tuple[list[Any], list[str], int, int]:
    """Redirect all branch endpoints and drop the resulting self-edges."""

    output_edges: list[Any] = []
    removed_edge_evidence: list[str] = []
    dropped_self_edge_count = 0
    redirected_edge_count = 0
    for edge in edges:
        if not isinstance(edge, dict):
            output_edges.append(copy.deepcopy(edge))
            continue

        output_edge = copy.deepcopy(edge)
        source = output_edge.get("source")
        target = output_edge.get("target")
        output_edge["source"] = overall_id if source in branch_ids else source
        output_edge["target"] = overall_id if target in branch_ids else target

        if output_edge["source"] == output_edge["target"]:
            dropped_self_edge_count += 1
            refs = output_edge.get("evidence_refs", [])
            if isinstance(refs, list):
                removed_edge_evidence.extend(
                    ref for ref in refs if isinstance(ref, str)
                )
            continue
        if output_edge["source"] != source or output_edge["target"] != target:
            redirected_edge_count += 1
        output_edges.append(output_edge)

    return (
        output_edges,
        removed_edge_evidence,
        dropped_self_edge_count,
        redirected_edge_count,
    )


def _remap_surviving_node_references(
    node: dict[str, Any], branch_ids: set[str], overall_id: str
) -> dict[str, Any]:
    result = copy.deepcopy(node)
    for key in ("parent_ids", "depends_on"):
        if key in result:
            result[key] = _remap_ids(result[key], branch_ids, overall_id)
    if result.get("parent_id") in branch_ids:
        result["parent_id"] = overall_id
    return result


def _remap_method_coverage(
    coverage: Any, branch_ids: set[str], overall_id: str
) -> Any:
    if not isinstance(coverage, list):
        return copy.deepcopy(coverage)
    result = copy.deepcopy(coverage)
    for record in result:
        if not isinstance(record, dict) or not isinstance(record.get("entries"), list):
            continue
        for entry in record["entries"]:
            if isinstance(entry, dict) and entry.get("conclusion_id") in branch_ids:
                entry["conclusion_id"] = overall_id
    return result


def merge_graph(graph: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise MergeError("$.nodes must be an array")

    branch_nodes = [
        node
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("node_type"), dict)
        and node["node_type"].get("research_stage") == "conclusion"
        and isinstance(node.get("article_content"), dict)
        and node["article_content"].get("subtype") == "branch_synthesis"
    ]
    overall_nodes = [
        node
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("node_type"), dict)
        and node["node_type"].get("research_stage") == "conclusion"
        and isinstance(node.get("article_content"), dict)
        and node["article_content"].get("subtype") == "overall_conclusion"
    ]

    if len(overall_nodes) != 1:
        raise MergeError(
            "the input must contain exactly one conclusion node with "
            f"subtype overall_conclusion; found {len(overall_nodes)}"
        )

    if not branch_nodes:
        return copy.deepcopy(graph), {
            "branch_nodes_found": 0,
            "branch_nodes_removed": 0,
            "overall_conclusion_id": overall_nodes[0].get("id"),
            "nodes_before": len(nodes),
            "nodes_after": len(nodes),
            "edges_before": len(graph.get("edges", [])) if isinstance(graph.get("edges"), list) else None,
            "edges_after": len(graph.get("edges", [])) if isinstance(graph.get("edges"), list) else None,
        }

    branch_ids = {node["id"] for node in branch_nodes}
    overall_id = overall_nodes[0].get("id")
    if not isinstance(overall_id, str) or not overall_id:
        raise MergeError("the overall_conclusion node must have a non-empty id")

    merged_overall = copy.deepcopy(overall_nodes[0])
    _merge_node_content(merged_overall, branch_nodes)

    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise MergeError("$.edges must be an array")
    (
        merged_edges,
        removed_edge_evidence,
        dropped_self_edge_count,
        redirected_edge_count,
    ) = _redirect_edges(
        edges, branch_ids, overall_id
    )
    _unique_extend(merged_overall["evidence_refs"], removed_edge_evidence)

    merged_nodes: list[Any] = []
    for node in nodes:
        if not isinstance(node, dict):
            merged_nodes.append(copy.deepcopy(node))
            continue
        node_id = node.get("id")
        if node_id in branch_ids:
            continue
        if node_id == overall_id:
            merged_nodes.append(merged_overall)
        else:
            merged_nodes.append(
                _remap_surviving_node_references(node, branch_ids, overall_id)
            )

    output = copy.deepcopy(graph)
    output["nodes"] = merged_nodes
    output["edges"] = merged_edges

    paper = output.get("paper")
    if isinstance(paper, dict):
        paper["overall_conclusion_id"] = overall_id

    if "method_coverage" in output:
        output["method_coverage"] = _remap_method_coverage(
            output.get("method_coverage"), branch_ids, overall_id
        )

    remaining_overall = [
        node
        for node in output["nodes"]
        if isinstance(node, dict)
        and isinstance(node.get("node_type"), dict)
        and node["node_type"].get("research_stage") == "conclusion"
        and isinstance(node.get("article_content"), dict)
        and node["article_content"].get("subtype") == "overall_conclusion"
    ]
    if len(remaining_overall) != 1:
        raise MergeError(
            f"merge produced {len(remaining_overall)} overall_conclusion nodes; expected 1"
        )

    record = {
        "branch_nodes_found": len(branch_nodes),
        "branch_nodes_removed": len(branch_nodes),
        "removed_branch_node_ids": [node["id"] for node in branch_nodes],
        "overall_conclusion_id": overall_id,
        "nodes_before": len(nodes),
        "nodes_after": len(merged_nodes),
        "edges_before": len(edges),
        "edges_after": len(merged_edges),
        "redirected_edge_count": redirected_edge_count,
        "dropped_self_edge_count": dropped_self_edge_count,
    }
    return output, record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all conclusion/branch_synthesis nodes into the single "
            "conclusion/overall_conclusion node."
        )
    )
    parser.add_argument("input_json", type=Path, help="input paper graph JSON")
    parser.add_argument(
        "output_json",
        nargs="?",
        type=Path,
        help="output JSON; defaults to <input_stem>_merged.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_option",
        type=Path,
        help="output JSON (alternative to the positional output path)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output_json is not None and args.output_option is not None:
        print("error: provide output either positionally or with --output, not both", file=sys.stderr)
        return 2
    output_path = args.output_option or args.output_json
    if output_path is None:
        output_path = args.input_json.with_name(
            f"{args.input_json.stem}_merged{args.input_json.suffix}"
        )

    try:
        graph = _load_json(args.input_json)
        merged_graph, record = merge_graph(graph)
        _write_json(output_path, merged_graph)
    except MergeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = {"input": str(args.input_json), "output": str(output_path), **record}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
