#!/usr/bin/env python3
"""Shared implementation for the four ScionBench judge splitters.

The public JSON is deliberately a projection of the paper graph.  It keeps
the expanded scientific content of released nodes, but removes evaluation
plans and every edge/relation that crosses from released input to held-out
targets.  The held-out JSON keeps the complete target records and all edges
that are not entirely inside the released input subgraph.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLIT_SCHEMA_VERSION = "scionbench.judge_split.v1"
SUPPORTED_STAGES = {
    "motivation",
    "method",
    "experiment_design",
    "experiment",
    "conclusion",
}


@dataclass(frozen=True)
class ModeSpec:
    name: str
    input_stages: tuple[str, ...]
    target_stages: tuple[str, ...]
    task: str


MODE_SPECS: dict[str, ModeSpec] = {
    "conclusion-mode": ModeSpec(
        name="conclusion-mode",
        input_stages=("motivation", "method", "experiment_design", "experiment"),
        target_stages=("conclusion",),
        task=(
            "Using only the released motivation, method, experiment-design and "
            "experiment-result information, produce a scientifically grounded "
            "conclusion for every held-out branch and then synthesize the overall "
            "conclusion. Preserve evidence scope, contradictions and uncertainty; "
            "do not use held-out gold records or invent unreported results."
        ),
    ),
    "experiment-mode": ModeSpec(
        name="experiment-mode",
        input_stages=("motivation", "method", "experiment_design"),
        target_stages=("experiment", "conclusion"),
        task=(
            "Using only the released motivation, method and experiment-design "
            "information, execute the declared experiments, record reproducible "
            "results and analyze them into branch and overall conclusions. Keep "
            "observations separate from interpretations and report failures or "
            "missing resources honestly."
        ),
    ),
    "method-mode": ModeSpec(
        name="method-mode",
        input_stages=("motivation", "method"),
        target_stages=("experiment_design", "experiment", "conclusion"),
        task=(
            "Using only the released motivation and method information, design the "
            "experiments, execute them, record reproducible results and derive "
            "branch and overall conclusions. Make variables, controls, metrics, "
            "limitations and uncertainty explicit; do not assume held-out designs, "
            "results or conclusions."
        ),
    ),
    "motivation-mode": ModeSpec(
        name="motivation-mode",
        input_stages=("motivation",),
        target_stages=("method", "experiment_design", "experiment", "conclusion"),
        task=(
            "Using only the released research motivation, carry out the complete "
            "scientific workflow: propose a method, design and execute experiments, "
            "record results, analyze evidence and synthesize conclusions. Make the "
            "workflow reproducible and distinguish observed results from causal or "
            "general claims."
        ),
    ),
}


class SplitError(ValueError):
    """A user-correctable graph or command-line error."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SplitError(f"benchmark JSON does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SplitError(f"invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise SplitError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitError("the benchmark root must be a JSON object")
    return value


def _node_stage(node: dict[str, Any], index: int) -> str:
    node_type = node.get("node_type")
    if not isinstance(node_type, dict):
        raise SplitError(f"nodes[{index}] is missing object node_type")
    stage = node_type.get("research_stage")
    if stage not in SUPPORTED_STAGES:
        raise SplitError(
            f"nodes[{index}] ({node.get('id', '<unknown>')}) has unsupported "
            f"research_stage: {stage!r}"
        )
    return stage


def _validate_and_index(doc: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    nodes = doc.get("nodes")
    if not isinstance(nodes, list):
        raise SplitError("benchmark must contain a nodes array")

    node_by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise SplitError(f"nodes[{index}] must be an object")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise SplitError(f"nodes[{index}] must have a non-empty string id")
        if node_id in node_by_id:
            raise SplitError(f"duplicate node id: {node_id}")
        _node_stage(node, index)
        node_by_id[node_id] = node

    edges = doc.get("edges", [])
    if not isinstance(edges, list):
        raise SplitError("benchmark edges must be an array")
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise SplitError(f"edges[{index}] must be an object")
        for endpoint in ("source", "target"):
            if edge.get(endpoint) not in node_by_id:
                raise SplitError(
                    f"edges[{index}].{endpoint} references an unknown node: "
                    f"{edge.get(endpoint)!r}"
                )

    method_relations = doc.get("method_relations", [])
    if not isinstance(method_relations, list):
        raise SplitError("benchmark method_relations must be an array")
    for index, relation in enumerate(method_relations):
        if not isinstance(relation, dict):
            raise SplitError(f"method_relations[{index}] must be an object")
        for field in ("method_a", "method_b"):
            method_id = relation.get(field)
            if method_id not in node_by_id:
                raise SplitError(
                    f"method_relations[{index}].{field} references an unknown node: "
                    f"{method_id!r}"
                )
            if node_by_id[method_id]["node_type"]["research_stage"] != "method":
                raise SplitError(
                    f"method_relations[{index}].{field} is not a method node: "
                    f"{method_id!r}"
                )

    return nodes, node_by_id


def _public_node(node: dict[str, Any]) -> dict[str, Any]:
    """Project an input node to the explicitly released source fields.

    ``parent_ids`` and ``depends_on`` are copied exactly from the source node.
    They are node metadata, while explicit graph edges are partitioned
    separately and still enforce the no-cross-boundary-edge rule.
    """

    node_type = node.get("node_type")
    article_content = node.get("article_content")
    if not isinstance(node_type, dict) or not isinstance(article_content, dict):
        raise SplitError(f"node {node.get('id', '<unknown>')} has invalid content")

    return {
        "id": copy.deepcopy(node.get("id")),
        "parent_ids": copy.deepcopy(node.get("parent_ids", [])),
        "depends_on": copy.deepcopy(node.get("depends_on", [])),
        "node_type": {
            "research_stage": copy.deepcopy(node_type.get("research_stage")),
            "data_type": copy.deepcopy(node_type.get("data_type")),
        },
        "article_content": {
            "subtype": copy.deepcopy(article_content.get("subtype")),
            "text": copy.deepcopy(article_content.get("text")),
            "data": copy.deepcopy(article_content.get("data")),
        },
    }


def _full_paper_context(doc: dict[str, Any]) -> dict[str, Any]:
    paper = doc.get("paper")
    if not isinstance(paper, dict):
        return {}
    return copy.deepcopy(paper)


def _public_paper_context(doc: dict[str, Any]) -> dict[str, Any]:
    """Expose only non-identifying context; title/source paths stay withheld."""

    paper = doc.get("paper")
    if not isinstance(paper, dict):
        return {}
    result: dict[str, Any] = {}
    if isinstance(paper.get("source_language"), str):
        result["source_language"] = paper["source_language"]
    return result


def _edge_partition(
    edges: list[dict[str, Any]], input_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    input_edges: list[dict[str, Any]] = []
    evaluation_edges: list[dict[str, Any]] = []
    boundary_edges: list[dict[str, Any]] = []
    for edge in edges:
        source_is_input = edge["source"] in input_ids
        target_is_input = edge["target"] in input_ids
        if source_is_input and target_is_input:
            input_edges.append(copy.deepcopy(edge))
        else:
            evaluation_edges.append(copy.deepcopy(edge))
            if source_is_input != target_is_input:
                boundary_edges.append(copy.deepcopy(edge))
    return input_edges, evaluation_edges, boundary_edges


def _relation_partition(
    relations: list[dict[str, Any]], input_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_relations: list[dict[str, Any]] = []
    evaluation_relations: list[dict[str, Any]] = []
    for relation in relations:
        source_is_input = relation["method_a"] in input_ids
        target_is_input = relation["method_b"] in input_ids
        if source_is_input and target_is_input:
            input_relations.append(copy.deepcopy(relation))
        else:
            evaluation_relations.append(copy.deepcopy(relation))
    return input_relations, evaluation_relations


def split_graph(doc: dict[str, Any], spec: ModeSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create the public agent input and held-out evaluation JSON objects."""

    nodes, node_by_id = _validate_and_index(doc)
    input_ids = {
        node_id
        for node_id, node in node_by_id.items()
        if node["node_type"]["research_stage"] in spec.input_stages
    }
    target_ids = {node_id for node_id in node_by_id if node_id not in input_ids}

    input_edges, evaluation_edges, boundary_edges = _edge_partition(
        doc.get("edges", []), input_ids
    )
    input_relations, evaluation_relations = _relation_partition(
        doc.get("method_relations", []), input_ids
    )

    input_nodes = [
        _public_node(node) for node in nodes if node["id"] in input_ids
    ]
    target_nodes = [
        copy.deepcopy(node) for node in nodes if node["id"] in target_ids
    ]

    agent_input = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "mode": spec.name,
        "task": spec.task,
        "paper_context": _public_paper_context(doc),
        "released_stages": list(spec.input_stages),
        "released_node_ids": [node["id"] for node in input_nodes],
        "nodes": input_nodes,
        "edges": input_edges,
        "method_relations": input_relations,
        "release_policy": {
            "node_content": (
                "nodes retain id, parent_ids, depends_on, node_type.research_stage, "
                "node_type.data_type, and article_content.subtype/text/data; "
                "evaluation_plan and other node fields are withheld"
            ),
            "node_relationship_fields": "parent_ids and depends_on are copied exactly from the source node",
            "edge_scope": "only edges whose source and target are both released nodes",
            "method_relation_scope": "only method relations whose two endpoints are released nodes",
            "held_out_stages": list(spec.target_stages),
        },
    }

    evaluation = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "mode": spec.name,
        "released_stages": list(spec.input_stages),
        "target_stages": list(spec.target_stages),
        "released_node_ids": [node["id"] for node in nodes if node["id"] in input_ids],
        "target_node_ids": [node["id"] for node in target_nodes],
        "gold_nodes": target_nodes,
        # Canonical target-side names.  The gold_* aliases above/below are kept
        # for compatibility with the first version of these splitters.
        "nodes": copy.deepcopy(target_nodes),
        # This is the complement of agent_input.edges.  It includes all
        # target-internal edges and all input-to-target boundary edges.
        "gold_edges": evaluation_edges,
        "edges": copy.deepcopy(evaluation_edges),
        "boundary_edges": boundary_edges,
        "gold_method_relations": evaluation_relations,
        # The evaluator may need the complete method-relation table even when
        # the methods themselves were released to the agent.
        "method_relations": copy.deepcopy(doc.get("method_relations", [])),
        "method_coverage": copy.deepcopy(doc.get("method_coverage", [])),
        "evidence": copy.deepcopy(doc.get("evidence", [])),
        "paper": _full_paper_context(doc),
        "source_manifest": copy.deepcopy(doc.get("source_manifest", {})),
        "evaluation_policy": {
            "target_content": "nodes/gold_nodes contain complete held-out nodes, including evaluation_plan",
            "edge_scope": "edges/gold_edges is the complement of the released internal edge set",
            "method_relation_scope": "method_relations contains the complete source method-relation table",
            "boundary_edges_are_held_out": True,
            "agent_must_not_receive_this_file": True,
        },
    }

    return agent_input, evaluation


def _write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise SplitError(f"cannot write {path}: {exc}") from exc


def _build_parser(spec: ModeSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Split a ScionBench paper graph into agent input and held-out "
            f"evaluation data for {spec.name}."
        )
    )
    parser.add_argument(
        "graph",
        nargs="?",
        type=Path,
        help="path to paper_graph.json",
    )
    parser.add_argument(
        "--graph",
        dest="graph_option",
        type=Path,
        help="path to paper_graph.json (alternative to the positional argument)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="directory for agent_input.json and evaluation.json (default: ./<mode>-split)",
    )
    parser.add_argument(
        "--input-output",
        type=Path,
        help="explicit path for the agent input JSON",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        help="explicit path for the held-out evaluation JSON",
    )
    return parser


def run_cli(mode: str, argv: list[str] | None = None) -> int:
    spec = MODE_SPECS[mode]
    parser = _build_parser(spec)
    args = parser.parse_args(argv)

    if args.graph is not None and args.graph_option is not None:
        parser.error("provide the graph path either positionally or with --graph, not both")
    graph_path = args.graph_option or args.graph
    if graph_path is None:
        parser.error("a paper_graph.json path is required")

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.input_output is not None:
        output_dir = args.input_output.parent
    elif args.evaluation_output is not None:
        output_dir = args.evaluation_output.parent
    else:
        output_dir = Path.cwd() / f"{mode}-split"

    input_path = args.input_output or output_dir / "agent_input.json"
    evaluation_path = args.evaluation_output or output_dir / "evaluation.json"

    try:
        doc = _load_json(graph_path)
        agent_input, evaluation = split_graph(doc, spec)
        _write_json(input_path, agent_input)
        _write_json(evaluation_path, evaluation)
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = {
        "mode": mode,
        "agent_input": str(input_path),
        "evaluation": str(evaluation_path),
        "released_nodes": len(agent_input["nodes"]),
        "released_edges": len(agent_input["edges"]),
        "released_method_relations": len(agent_input["method_relations"]),
        "target_nodes": len(evaluation["gold_nodes"]),
        "held_out_edges": len(evaluation["gold_edges"]),
        "held_out_boundary_edges": len(evaluation["boundary_edges"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
