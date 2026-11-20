#!/usr/bin/env python3
"""Convert a ScionBench ``agent_input.json`` into readable Markdown.

The converter keeps the four research stages in the fixed order
``motivation``, ``method``, ``experiment_design`` and ``experiment``.  Nodes
within a stage retain the order in which they occur in the JSON ``nodes``
array.  The meaningful contents of ``article_content`` are rendered as
ordinary Markdown, including nested fields in text/data records, instead of
being emitted as JSON code blocks.

Fields that only identify an extraction artifact or point to a local file
are deliberately omitted.  In particular, image paths, evidence-reference
IDs, source-artifact IDs and extraction-completeness flags do not add useful
scientific content for the agent.  Graph-control fields such as ``parent_ids``
and ``depends_on`` are also not copied into the document body; this converter
is intended to present the released article content in research-stage order.

Examples:
    python3 agent_input_to_markdown.py \
        --input-dir /path/to/input \
        --output-dir /path/to/output \
        --input-name agent_input.json \
        --output-name agent_input.md

    python3 agent_input_to_markdown.py \
        --input-file /path/to/agent_input.json \
        --output-file /path/to/agent_input.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


STAGE_ORDER = ("motivation", "method", "experiment_design", "experiment")
STAGE_TITLES = {
    "motivation": "Motivation",
    "method": "Method",
    "experiment_design": "Experiment Design",
    "experiment": "Experiment",
}
TEXT_LABELS = {
    "statement": "Statement",
    "facts": "Facts",
    "definitions": "Definitions",
    "conditions": "Conditions",
    "reported_interpretation": "Reported Interpretation",
}
DATA_LABELS = {
    "tables": "Tables",
    "figures": "Figures",
    "numeric_records": "Numeric Records",
    "formulas": "Formulas",
}

# These fields are useful to an extraction/evaluation pipeline but do not
# express the scientific content that should be shown to the agent.  The
# suffix rules below also cover future fields such as ``plot_file`` or
# ``source_path`` without requiring a code change for every new name.
IGNORED_FIELD_NAMES = frozenset(
    {
        "asset_file",
        "asset_path",
        "directory",
        "dir",
        "evidence_ref",
        "evidence_refs",
        "file",
        "file_path",
        "filepath",
        "image_file",
        "image_path",
        "path",
        "source_artifact_id",
        "source_dir",
        "source_file",
        "source_path",
        "uri",
        "all_cells_transcribed",
        "all_visible_values_transcribed",
    }
)
IGNORED_FIELD_SUFFIXES = ("_file", "_dir", "_path", "_filepath", "_uri")


class ConversionError(ValueError):
    """Raised for malformed or unsupported agent-input JSON."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ConversionError(f"input JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConversionError(
            f"input JSON is invalid: {path} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    except OSError as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError("the JSON root must be an object")
    return value


def _write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ConversionError(f"cannot write Markdown output {path}: {exc}") from exc


def _node_stage(node: dict[str, Any]) -> Any:
    node_type = node.get("node_type")
    if isinstance(node_type, dict):
        return node_type.get("research_stage")
    # Compatibility with older inputs that used a scalar node.type field.
    return node.get("type")


def _is_ignored_field(key: Any) -> bool:
    """Return whether *key* is extraction metadata or a file reference."""

    if not isinstance(key, str):
        return False
    normalized = key.strip().lower()
    return normalized in IGNORED_FIELD_NAMES or normalized.endswith(
        IGNORED_FIELD_SUFFIXES
    )


def _has_content(value: Any) -> bool:
    """Check whether a value contains anything worth rendering."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(
            not _is_ignored_field(key) and _has_content(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_content(item) for item in value)
    return True


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _scalar_text(value: Any) -> str:
    """Format a scalar without falling back to JSON serialization."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False)
    return _inline_value(value)


def _inline_value(value: Any) -> str:
    """Render an arbitrary value in one line, never as a JSON fragment."""

    if _is_scalar(value):
        return _scalar_text(value).replace("\n", "<br>")
    if isinstance(value, list):
        visible = [_inline_value(item) for item in value if _has_content(item)]
        return "; ".join(visible) if visible else "—"
    if isinstance(value, dict):
        fields = []
        for key, item in value.items():
            if _is_ignored_field(key) or not _has_content(item):
                continue
            fields.append(f"{_humanize_key(key)}: {_inline_value(item)}")
        return "; ".join(fields) if fields else "—"
    return str(value)


def _humanize_key(key: Any) -> str:
    text = str(key).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "Field"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def _visible_items(
    mapping: dict[str, Any], *, skip_keys: frozenset[str] = frozenset()
) -> list[tuple[str, Any]]:
    return [
        (key, value)
        for key, value in mapping.items()
        if key not in skip_keys
        and not _is_ignored_field(key)
        and _has_content(value)
    ]


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _append_list_items(lines: list[str], values: list[Any], indent: str = "") -> None:
    """Render a list as nested Markdown bullets."""

    for index, value in enumerate(values, start=1):
        if not _has_content(value):
            continue
        if _is_scalar(value):
            lines.append(f"{indent}- {_inline_value(value)}")
        elif isinstance(value, dict):
            lines.append(f"{indent}- **Item {index}:**")
            _append_mapping_bullets(lines, value, indent=f"{indent}  ")
        elif isinstance(value, list):
            lines.append(f"{indent}- **Item {index}:**")
            _append_list_items(lines, value, indent=f"{indent}  ")
        else:
            lines.append(f"{indent}- {_inline_value(value)}")


def _append_mapping_bullets(
    lines: list[str],
    mapping: dict[str, Any],
    indent: str = "",
    *,
    skip_keys: frozenset[str] = frozenset(),
) -> None:
    """Render a mapping as readable nested bullets, preserving key order."""

    for key, value in _visible_items(mapping, skip_keys=skip_keys):
        label = _humanize_key(key)
        if _is_scalar(value):
            lines.append(f"{indent}- **{label}:** {_inline_value(value)}")
        else:
            lines.append(f"{indent}- **{label}:**")
            if isinstance(value, dict):
                _append_mapping_bullets(lines, value, indent=f"{indent}  ")
            elif isinstance(value, list):
                _append_list_items(lines, value, indent=f"{indent}  ")
            else:
                lines.append(f"{indent}  - {_inline_value(value)}")


def _append_value_section(
    lines: list[str], heading: str, value: Any, *, level: int = 4
) -> None:
    """Render a field under a Markdown heading at the requested level."""

    if not _has_content(value):
        return
    lines.append(f"{'#' * level} {heading}")
    lines.append("")
    if _is_scalar(value):
        lines.append(_scalar_text(value))
    elif isinstance(value, dict):
        _append_mapping_bullets(lines, value)
    elif isinstance(value, list):
        _append_list_items(lines, value)
    else:
        lines.append(_inline_value(value))
    lines.append("")


def _render_text(lines: list[str], text: Any) -> None:
    if not isinstance(text, dict):
        _append_value_section(lines, "Text", text, level=4)
        return

    for key, label in TEXT_LABELS.items():
        _append_value_section(lines, label, text.get(key), level=4)

    unknown = {
        key: value
        for key, value in text.items()
        if key not in TEXT_LABELS and not _is_ignored_field(key)
    }
    for key, value in unknown.items():
        _append_value_section(lines, _humanize_key(key), value, level=4)


def _table_cell(value: Any) -> str:
    """Use the human-facing cell display while retaining status in details."""

    if not isinstance(value, dict):
        rendered = _inline_value(value)
    else:
        display = value.get("display")
        if not _has_content(display):
            display = value.get("value")
        rendered = _inline_value(display) if _has_content(display) else "—"
        status = value.get("cell_status")
        if _has_content(status):
            rendered = f"{rendered} ({_inline_value(status)})"
    return _escape_table_cell(rendered)


def _column_title(column: Any, index: int) -> str:
    if isinstance(column, dict):
        name = column.get("name", f"Column {index + 1}")
        unit = column.get("unit")
        title = _inline_value(name)
        if _has_content(unit):
            title += f" ({_inline_value(unit)})"
        return title
    return _inline_value(column)


def _render_cell_details(
    lines: list[str], columns: list[Any], rows: list[Any]
) -> None:
    """Render cell-level values/statuses as Markdown rather than JSON."""

    details: list[tuple[str, str, str, str]] = []
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not isinstance(row.get("cells"), list):
            continue
        row_label = _inline_value(row.get("label")) if _has_content(row.get("label")) else f"Row {row_index}"
        for column_index, cell in enumerate(row["cells"]):
            if not isinstance(cell, dict):
                continue
            column = (
                _column_title(columns[column_index], column_index)
                if column_index < len(columns)
                else f"Column {column_index + 1}"
            )
            for key, value in cell.items():
                if _is_ignored_field(key) or not _has_content(value):
                    continue
                # ``display`` is already shown in the main table.  All other
                # cell fields are included here in a compact field/value
                # table, including numeric values and merge/status metadata.
                if key == "display":
                    continue
                details.append((row_label, column, _humanize_key(key), _inline_value(value)))

    if not details:
        return
    lines.append("###### Cell Details")
    lines.append("")
    lines.append("| Row | Column | Field | Value |")
    lines.append("| --- | --- | --- | --- |")
    for row, column, field, value in details:
        lines.append(
            "| "
            + " | ".join(
                _escape_table_cell(item) for item in (row, column, field, value)
            )
            + " |"
        )
    lines.append("")


def _render_table(lines: list[str], table: Any, index: int) -> None:
    if not isinstance(table, dict):
        _append_value_section(lines, f"Table {index + 1}", table, level=5)
        return

    table_id = table.get("id")
    title = f"Table {index + 1}"
    if _has_content(table_id):
        title += f" — `{_inline_value(table_id)}`"
    lines.append(f"##### {title}")
    lines.append("")

    caption = table.get("caption")
    if _has_content(caption):
        if _is_scalar(caption):
            lines.append(f"**Caption:** {_inline_value(caption)}")
            lines.append("")
        else:
            _append_value_section(lines, "Caption", caption, level=6)

    columns = table.get("columns")
    rows = table.get("rows")
    rectangular = (
        isinstance(columns, list)
        and bool(columns)
        and isinstance(rows, list)
        and bool(rows)
        and all(isinstance(row, dict) and isinstance(row.get("cells"), list) for row in rows)
    )
    if rectangular:
        headers = ["Row", *[_column_title(column, i) for i, column in enumerate(columns)]]
        lines.append("| " + " | ".join(_escape_table_cell(item) for item in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            label = row.get("label")
            row_label = _inline_value(label) if _has_content(label) else ""
            cells = row.get("cells", [])
            rendered_cells = [_table_cell(cell) for cell in cells]
            rendered_cells = (rendered_cells + ["—"] * len(columns))[: len(columns)]
            lines.append(
                "| "
                + " | ".join(
                    [_escape_table_cell(row_label), *rendered_cells]
                )
                + " |"
            )
        lines.append("")
        _render_cell_details(lines, columns, rows)
    else:
        if _has_content(columns):
            _append_value_section(lines, "Columns", columns, level=6)
        if _has_content(rows):
            _append_value_section(lines, "Rows", rows, level=6)

    notes = table.get("notes")
    _append_value_section(lines, "Notes", notes, level=6)

    # Preserve meaningful fields added by future extractors.  The usual table
    # structure has already been rendered above, so it is excluded here.
    known = {"id", "caption", "columns", "rows", "notes"}
    extras = {
        key: value
        for key, value in table.items()
        if key not in known and not _is_ignored_field(key)
    }
    for key, value in extras.items():
        _append_value_section(lines, _humanize_key(key), value, level=6)


def _render_diagram_edges(lines: list[str], edges: Any) -> None:
    if not isinstance(edges, list) or not _has_content(edges):
        return
    lines.append("###### Diagram Edges")
    lines.append("")
    for index, edge in enumerate(edges, start=1):
        if not isinstance(edge, dict):
            lines.append(f"- {_inline_value(edge)}")
            continue
        source = edge.get("source")
        target = edge.get("target")
        if _has_content(source) or _has_content(target):
            lines.append(
                f"- {_inline_value(source) if _has_content(source) else '—'} "
                f"→ {_inline_value(target) if _has_content(target) else '—'}"
            )
        else:
            lines.append(f"- **Edge {index}:**")
        extras = {
            key: value
            for key, value in edge.items()
            if key not in {"source", "target"}
        }
        if extras:
            _append_mapping_bullets(lines, extras, indent="  ")
    lines.append("")


def _render_figure(lines: list[str], figure: Any, index: int) -> None:
    if not isinstance(figure, dict):
        _append_value_section(lines, f"Figure {index + 1}", figure, level=5)
        return

    figure_id = figure.get("id")
    title = f"Figure {index + 1}"
    if _has_content(figure_id):
        title += f" — `{_inline_value(figure_id)}`"
    lines.append(f"##### {title}")
    lines.append("")

    asset_file = figure.get("asset_file")
    if _has_content(asset_file):
        lines.append(f"**Image file:** `{_inline_value(asset_file)}`")
        lines.append("")

    for key, label in (
        ("caption", "Caption"),
        ("kind", "Kind"),
        ("semantic_description", "Semantic Description"),
        ("precision", "Precision"),
    ):
        value = figure.get(key)
        if not _has_content(value):
            continue
        if _is_scalar(value):
            lines.append(f"**{label}:** {_inline_value(value)}")
            lines.append("")
        else:
            _append_value_section(lines, label, value, level=6)

    for key, label in (
        ("annotations", "Annotations"),
        ("diagram_nodes", "Diagram Nodes"),
        ("axes", "Axes"),
        ("series", "Series"),
        ("points", "Points"),
    ):
        _append_value_section(lines, label, figure.get(key), level=6)
    _render_diagram_edges(lines, figure.get("diagram_edges"))

    known = {
        "id",
        "caption",
        "kind",
        "semantic_description",
        "precision",
        "annotations",
        "diagram_nodes",
        "diagram_edges",
        "axes",
        "series",
        "points",
    }
    extras = {
        key: value
        for key, value in figure.items()
        if key not in known and not _is_ignored_field(key)
    }
    for key, value in extras.items():
        _append_value_section(lines, _humanize_key(key), value, level=6)


def _render_records(lines: list[str], heading: str, records: Any) -> None:
    if not isinstance(records, list) or not _has_content(records):
        return
    lines.append(f"##### {heading}")
    lines.append("")
    singular = heading[:-1] if heading.endswith("s") else heading
    for index, record in enumerate(records, start=1):
        if isinstance(record, dict):
            record_id = record.get("id")
            title = f"{singular} {index}"
            if _has_content(record_id):
                title += f" — `{_inline_value(record_id)}`"
            lines.append(f"###### {title}")
            lines.append("")
            _append_mapping_bullets(lines, record, skip_keys=frozenset({"id"}))
            lines.append("")
        else:
            lines.append(f"- {_inline_value(record)}")
    lines.append("")


def _render_data(lines: list[str], data: Any) -> None:
    if not _has_content(data):
        return
    if not isinstance(data, dict):
        _append_value_section(lines, "Data", data, level=4)
        return

    lines.append("#### Data")
    lines.append("")
    for index, table in enumerate(_list_or_empty(data.get("tables")), start=1):
        _render_table(lines, table, index - 1)
    for index, figure in enumerate(_list_or_empty(data.get("figures")), start=1):
        _render_figure(lines, figure, index - 1)
    _render_records(lines, DATA_LABELS["numeric_records"], data.get("numeric_records"))
    _render_records(lines, DATA_LABELS["formulas"], data.get("formulas"))

    known = set(DATA_LABELS)
    extras = {
        key: value
        for key, value in data.items()
        if key not in known and not _is_ignored_field(key)
    }
    for key, value in extras.items():
        _append_value_section(lines, _humanize_key(key), value, level=5)
    lines.append("")


def _render_node(lines: list[str], node: dict[str, Any]) -> None:
    node_id = node.get("id")
    node_type = node.get("node_type")
    article_content = node.get("article_content")
    if not isinstance(node_id, str) or not node_id:
        raise ConversionError("every node must have a non-empty string id")
    if not isinstance(node_type, dict):
        node_type = {"research_stage": _node_stage(node)}
    if not isinstance(article_content, dict):
        raise ConversionError(f"node {node_id} has no object article_content")

    lines.append(f"### `{node_id}`")
    lines.append("")
    lines.append(f"- **Research stage:** `{_inline_value(node_type.get('research_stage'))}`")
    if _has_content(node_type.get("data_type")):
        lines.append(f"- **Data type:** `{_inline_value(node_type['data_type'])}`")
    if _has_content(article_content.get("subtype")):
        lines.append(
            f"- **Article subtype:** `{_inline_value(article_content['subtype'])}`"
        )
    lines.append("")

    _render_text(lines, article_content.get("text"))
    _render_data(lines, article_content.get("data"))

    known = {"subtype", "text", "data"}
    extras = {
        key: value
        for key, value in article_content.items()
        if key not in known and not _is_ignored_field(key)
    }
    for key, value in extras.items():
        _append_value_section(lines, _humanize_key(key), value, level=4)


def _render_document_context(lines: list[str], document: dict[str, Any]) -> None:
    mode = document.get("mode")
    task = document.get("task")
    paper_context = document.get("paper_context")
    if _has_content(mode):
        lines.append(f"> **Mode:** `{_inline_value(mode)}`")
    if _has_content(task):
        lines.append(f"> **Task:** {_inline_value(task)}")
    if _has_content(mode) or _has_content(task):
        lines.append("")
    if _has_content(paper_context):
        lines.append("## Paper Context")
        lines.append("")
        if isinstance(paper_context, dict):
            _append_mapping_bullets(lines, paper_context)
        else:
            lines.append(_inline_value(paper_context))
        lines.append("")


def convert_json_to_markdown(document: dict[str, Any]) -> str:
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise ConversionError("JSON must contain a nodes array")

    grouped: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGE_ORDER}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ConversionError(f"nodes[{index}] must be an object")
        stage = _node_stage(node)
        if stage not in grouped:
            raise ConversionError(
                f"node {node.get('id', '<unknown>')} has unsupported stage {stage!r}; "
                f"expected one of {list(STAGE_ORDER)}"
            )
        grouped[stage].append(node)

    lines = ["# Benchmark Input", ""]
    _render_document_context(lines, document)

    for stage in STAGE_ORDER:
        stage_nodes = grouped[stage]
        if not stage_nodes:
            continue
        lines.append(f"## {STAGE_TITLES[stage]}")
        lines.append("")
        for node in stage_nodes:
            _render_node(lines, node)

    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an agent_input.json into ordered four-stage Markdown."
    )
    parser.add_argument("--input-dir", type=Path, help="directory containing the input JSON")
    parser.add_argument("--output-dir", type=Path, help="directory for the output Markdown")
    parser.add_argument("--input-name", default="agent_input.json", help="input filename")
    parser.add_argument(
        "--output-name",
        help="output filename; defaults to the input stem with a .md suffix",
    )
    parser.add_argument("--input-file", type=Path, help="direct input JSON path")
    parser.add_argument("--output-file", type=Path, help="direct output Markdown path")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.input_file and args.input_dir:
        raise ConversionError("use either --input-file or --input-dir, not both")
    if args.output_file and args.output_dir:
        raise ConversionError("use either --output-file or --output-dir, not both")
    if args.input_file:
        input_path = args.input_file
    elif args.input_dir:
        input_path = args.input_dir / args.input_name
    else:
        raise ConversionError("provide --input-file or --input-dir")

    output_name = args.output_name or f"{Path(args.input_name).stem}.md"
    if args.output_file:
        output_path = args.output_file
    elif args.output_dir:
        output_path = args.output_dir / output_name
    else:
        raise ConversionError("provide --output-file or --output-dir")
    return input_path, output_path


def main() -> int:
    args = _parse_args()
    try:
        input_path, output_path = _resolve_paths(args)
        document = _load_json(input_path)
        markdown = convert_json_to_markdown(document)
        _write_text(output_path, markdown)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stage_counts = {
        stage: sum(
            1
            for node in document.get("nodes", [])
            if isinstance(node, dict)
            and _node_stage(node) == stage
        )
        for stage in STAGE_ORDER
    }
    print(
        json.dumps(
            {"input": str(input_path), "output": str(output_path), "stage_counts": stage_counts},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
