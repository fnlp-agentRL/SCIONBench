#!/usr/bin/env python3
"""Validate a ScionBench single-paper extraction.

The validator is deliberately dependency-free.  It checks the machine-level
contract (IDs, graph, stage I/O, evidence, tables/figures and metric rules)
and performs conservative source audits.  It cannot prove that a scientific
interpretation is true; those cases are emitted as warnings for human/LLM
review.  Exit status is 0 only when no error diagnostics remain.

Usage:
  python validate_extraction.py --paper-dir PAPER_DIR --graph paper_graph.json \
      --report validation_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "scionbench.single_paper.v3.2"
LEGACY_SCHEMA_VERSION = "scionbench.single_paper.v3.1"
OLDER_LEGACY_SCHEMA_VERSION = "scionbench.single_paper.v3"
LEGACY_SCHEMA_VERSIONS = {LEGACY_SCHEMA_VERSION, OLDER_LEGACY_SCHEMA_VERSION}
STAGES = {"motivation", "method", "experiment_design", "experiment", "conclusion"}
DATA_TYPES = {"text", "data"}
EVALUABILITY = {"evaluable", "not_evaluable"}
RELATIONS = {
    "contains",
    "depends_on",
    "parallel_branch",
    "strict_serial",
    "provides_input",
    "produces",
    "supports",
    "aggregates",
    "compares",
    "technical_dependency",
    "variant",
    "shared_pipeline",
}
METHOD_RELATIONS = {
    "parallel",
    "strict_serial",
    "technical_dependency",
    "variant",
    "shared_pipeline",
}
METHOD_EDGE_RELATIONS = {
    "parallel": "parallel_branch",
    "strict_serial": "strict_serial",
    "technical_dependency": "technical_dependency",
    "variant": "variant",
    "shared_pipeline": "shared_pipeline",
}
COVERAGE_STATUSES = {
    "directly_tested",
    "indirectly_tested",
    "downstream_use",
    "no_followup",
}
COVERAGE_TYPES = {
    "factor_variation",
    "ablation",
    "component_in_pipeline",
    "downstream_application",
    "mechanism_analysis",
    "qualitative_probe",
    "baseline_comparison",
}
COVERAGE_EDGE_RELATIONS = {"provides_input", "contains", "compares"}
CONFIDENCE = {"high", "medium", "low"}
EVIDENCE_KINDS = {"text", "table", "figure", "equation", "supplement"}
SCORE_ROLES = {"core", "diagnostic", "not_scored"}
METRIC_CLASSES = {"process", "result", "reasoning", "long_horizon"}
METRIC_ROLES = {"core", "diagnostic"}
SCORING_TYPES = {"rubric", "normalized_error", "directional", "artifact_check", "consistency"}
FIGURE_KINDS = {"plot", "diagram", "photo", "scheme", "flowchart"}
CONCLUSION_SUBTYPES = {"stage_conclusion", "branch_synthesis", "overall_conclusion"}
LEGACY_CONCLUSION_SUBTYPES = CONCLUSION_SUBTYPES | {"limitation"}
GRAPH_ACTIONS = {"same_parent_siblings", "child_of_a", "child_of_a_conclusion", "same_method_protocol"}
REQUIRED_ARTICLE_KEYS = {"subtype", "text", "data", "completeness"}
REQUIRED_NODE_KEYS = {"id", "node_type", "article_content", "evaluation_plan", "evidence_refs"}
OPTIONAL_NODE_KEYS = {"parent_id", "parent_ids", "depends_on", "branch_id"}
PLACEHOLDER_RE = re.compile(
    r"^(?:see|refer(?:red)?\s+to|consult|look\s+at|见|参见|详见)\s+(?:the\s+)?(?:table|tab\.?|figure|fig\.?|equation|eq\.?)\b",
    re.I,
)


def _diag(items: list[dict[str, Any]], severity: str, code: str, location: str, message: str, fix: str) -> None:
    items.append({"severity": severity, "code": code, "location": location, "message": message, "fix": fix})


def _is_nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _uniq_strings(value: Any) -> bool:
    return isinstance(value, list) and all(_is_nonempty_string(x) for x in value) and len(value) == len(set(value))


def _norm(s: str) -> str:
    s = s.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_relative(path_text: str, root: Path) -> Path | None:
    try:
        candidate = (root / path_text).resolve()
        candidate.relative_to(root.resolve())
        return candidate
    except (OSError, ValueError):
        return None


def _read_sources(paper_dir: Path) -> tuple[list[Path], list[Path], dict[str, str]]:
    markdown = sorted(p for p in paper_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md")
    image_root = paper_dir / "images"
    image_ext = {".jpg", ".jpeg", ".png", ".webp", ".svg", ".gif", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in image_ext) if image_root.is_dir() else []
    text = {}
    for path in markdown:
        try:
            text[path.relative_to(paper_dir).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text[path.relative_to(paper_dir).as_posix()] = ""
    return markdown, images, text


def _index(items: Any, label: str, diagnostics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        _diag(diagnostics, "error", "COLLECTION_TYPE", f"$.{label}", f"{label} must be an array", "Use an array of objects.")
        return {}
    out: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(items):
        loc = f"$.{label}[{i}]"
        if not isinstance(item, dict):
            _diag(diagnostics, "error", "ITEM_TYPE", loc, "item must be an object", "Replace it with an object containing id.")
            continue
        item_id = item.get("id")
        if not _is_nonempty_string(item_id):
            _diag(diagnostics, "error", "ID_INVALID", f"{loc}.id", "id must be a non-empty string", "Provide a stable ASCII id.")
            continue
        if item_id in out:
            _diag(diagnostics, "error", "DUPLICATE_ID", loc, f"duplicate id: {item_id}", "Rename the duplicate and update references.")
        else:
            out[item_id] = item
    return out


def _check_evidence(doc: dict[str, Any], paper_dir: Path, evidence: dict[str, dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    for eid, ev in evidence.items():
        loc = f"$.evidence[{eid}]"
        kind = ev.get("kind")
        if kind not in EVIDENCE_KINDS:
            _diag(diagnostics, "error", "EVIDENCE_KIND", loc, f"invalid evidence kind: {kind}", f"Use one of {sorted(EVIDENCE_KINDS)}.")
        source_file = ev.get("source_file")
        if not _is_nonempty_string(source_file):
            _diag(diagnostics, "error", "EVIDENCE_SOURCE", loc, "source_file is required", "Use a path relative to paper_dir.")
            continue
        source = _safe_relative(source_file, paper_dir)
        if source is None or not source.is_file():
            _diag(diagnostics, "error", "EVIDENCE_FILE_MISSING", loc, f"evidence source does not exist: {source_file}", "Correct source_file or add the missing source asset.")
        if kind in {"text", "table", "equation", "supplement"} and not _is_nonempty_string(ev.get("verbatim")):
            _diag(diagnostics, "error", "EVIDENCE_VERBATIM_EMPTY", loc, "text/table/equation evidence needs verbatim", "Copy the relevant source text verbatim.")
        if kind == "figure":
            asset = ev.get("asset_file")
            if not _is_nonempty_string(asset):
                _diag(diagnostics, "error", "FIGURE_ASSET_EMPTY", loc, "figure evidence needs asset_file", "Point to the actual image under paper_dir.")
            elif (_safe_relative(asset, paper_dir) is None) or not (_safe_relative(asset, paper_dir) or Path("" )).is_file():
                _diag(diagnostics, "error", "FIGURE_ASSET_MISSING", loc, f"figure asset does not exist: {asset}", "Use an existing image relative to paper_dir.")
        if not _is_nonempty_string(ev.get("locator")) or not _is_nonempty_string(ev.get("relevance")):
            _diag(diagnostics, "error", "EVIDENCE_METADATA", loc, "locator and relevance are required", "Describe the exact section/table/figure and scientific fact.")
        if ev.get("confidence") not in CONFIDENCE:
            _diag(diagnostics, "error", "EVIDENCE_CONFIDENCE", loc, "invalid confidence", f"Use one of {sorted(CONFIDENCE)}.")
        if kind in {"text", "table", "equation", "supplement"} and _is_nonempty_string(ev.get("verbatim")) and source_file in _source_text_cache:
            if _norm(ev["verbatim"]) not in _norm(_source_text_cache[source_file]):
                _diag(diagnostics, "warning", "VERBATIM_NOT_FOUND", loc, "verbatim was not found as a normalized substring in the source", "Recheck the quote and locator; keep a warning only when the source is a transformed extraction.")


_source_text_cache: dict[str, str] = {}


def _check_table(table: Any, loc: str, paper_dir: Path, diagnostics: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> None:
    if not isinstance(table, dict):
        _diag(diagnostics, "error", "TABLE_TYPE", loc, "table must be an object", "Expand the table into an object.")
        return
    for key in ("id", "evidence_refs", "caption", "columns", "rows", "notes"):
        if key not in table:
            _diag(diagnostics, "error", "TABLE_FIELD", loc, f"missing table field: {key}", "Provide all table fields, including notes.")
    if not _is_nonempty_string(table.get("caption")):
        _diag(diagnostics, "error", "TABLE_CAPTION", loc, "table caption is empty", "Copy the table caption or a precise description.")
    if not _uniq_strings(table.get("evidence_refs")):
        _diag(diagnostics, "error", "TABLE_EVIDENCE_REFS", loc, "table evidence_refs must be a unique non-empty string list", "Attach the table to its table evidence record.")
    else:
        for eid in table["evidence_refs"]:
            if eid not in evidence:
                _diag(diagnostics, "error", "TABLE_EVIDENCE_UNKNOWN", loc, f"unknown table evidence id: {eid}", "Create the evidence record before using it.")
            elif evidence[eid].get("kind") not in {"table", "text", "supplement"}:
                _diag(diagnostics, "error", "TABLE_EVIDENCE_KIND", loc, f"evidence {eid} is not table/text/supplement evidence", "Use table evidence for table content.")
    cols = table.get("columns")
    rows = table.get("rows")
    if not isinstance(cols, list) or not cols:
        _diag(diagnostics, "error", "TABLE_COLUMNS", loc, "table columns must be a non-empty array", "List every column and unit.")
    if not isinstance(rows, list) or not rows:
        _diag(diagnostics, "error", "TABLE_ROWS", loc, "table rows must be a non-empty array", "List every row and cell, including non-numeric entries.")
    if isinstance(cols, list):
        for j, col in enumerate(cols):
            if not isinstance(col, dict) or not _is_nonempty_string(col.get("name")) or "unit" not in col:
                _diag(diagnostics, "error", "TABLE_COLUMN_SCHEMA", f"{loc}.columns[{j}]", "column needs name and unit (unit may be null)", "Record the visible column header and unit.")
    if isinstance(rows, list) and isinstance(cols, list):
        for j, row in enumerate(rows):
            rloc = f"{loc}.rows[{j}]"
            if not isinstance(row, dict) or not _is_nonempty_string(row.get("label")) or not isinstance(row.get("cells"), list):
                _diag(diagnostics, "error", "TABLE_ROW_SCHEMA", rloc, "row needs label and cells array", "Expand the row exactly.")
                continue
            cells = row["cells"]
            if len(cells) != len(cols):
                _diag(diagnostics, "error", "TABLE_CELL_COUNT", rloc, f"cell count {len(cells)} != column count {len(cols)}", "Add every cell in the original order.")
            for k, cell in enumerate(cells):
                if not isinstance(cell, dict) or "value" not in cell or "display" not in cell or "numeric_value" not in cell:
                    _diag(diagnostics, "error", "TABLE_CELL_SCHEMA", f"{rloc}.cells[{k}]", "cell needs value, numeric_value and display", "Preserve text, numeric interpretation and exact display.")
    if table.get("all_cells_transcribed") is not True:
        _diag(diagnostics, "error", "TABLE_INCOMPLETE", loc, "all_cells_transcribed must be true", "Do not leave table cells implicit.")


def _check_figure(fig: Any, loc: str, paper_dir: Path, diagnostics: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> None:
    if not isinstance(fig, dict):
        _diag(diagnostics, "error", "FIGURE_TYPE", loc, "figure must be an object", "Expand the image into an object.")
        return
    for key in ("id", "evidence_refs", "caption", "asset_file", "kind", "semantic_description", "annotations"):
        if key not in fig:
            _diag(diagnostics, "error", "FIGURE_FIELD", loc, f"missing figure field: {key}", "Provide all figure fields.")
    if fig.get("kind") not in FIGURE_KINDS:
        _diag(diagnostics, "error", "FIGURE_KIND", loc, "invalid figure kind", f"Use one of {sorted(FIGURE_KINDS)}.")
    asset = fig.get("asset_file")
    if not _is_nonempty_string(asset) or (_safe_relative(asset, paper_dir) is None) or not (_safe_relative(asset, paper_dir) or Path("" )).is_file():
        _diag(diagnostics, "error", "FIGURE_ASSET", loc, "figure asset_file does not resolve", "Use an existing image path relative to paper_dir.")
    if not _uniq_strings(fig.get("evidence_refs")):
        _diag(diagnostics, "error", "FIGURE_EVIDENCE_REFS", loc, "figure evidence_refs must be a unique non-empty string list", "Attach the figure to its figure evidence record.")
    else:
        for eid in fig["evidence_refs"]:
            if eid not in evidence:
                _diag(diagnostics, "error", "FIGURE_EVIDENCE_UNKNOWN", loc, f"unknown figure evidence id: {eid}", "Create the evidence record before using it.")
            elif evidence[eid].get("kind") not in {"figure", "text", "supplement"}:
                _diag(diagnostics, "error", "FIGURE_EVIDENCE_KIND", loc, f"evidence {eid} is not figure/text/supplement evidence", "Use figure evidence for image content.")
    if not _is_nonempty_string(fig.get("caption")) or not _is_nonempty_string(fig.get("semantic_description")):
        _diag(diagnostics, "error", "FIGURE_DESCRIPTION", loc, "caption and semantic_description are required", "Describe the scientific content of the image.")
    kind = fig.get("kind")
    if kind == "plot":
        axes = fig.get("axes")
        series = fig.get("series")
        if not isinstance(axes, dict) or not _is_nonempty_string(axes.get("x_label")) or not _is_nonempty_string(axes.get("y_label")):
            _diag(diagnostics, "error", "PLOT_AXES", loc, "plot needs x_label and y_label", "Transcribe axes and units even when values are approximate.")
        if not isinstance(series, list) or not series:
            _diag(diagnostics, "error", "PLOT_SERIES", loc, "plot needs at least one series", "List every visible curve/bar/condition.")
        else:
            for i, ser in enumerate(series):
                if not isinstance(ser, dict) or not _is_nonempty_string(ser.get("name")) or not isinstance(ser.get("points"), list) or not ser["points"]:
                    _diag(diagnostics, "error", "PLOT_POINTS", f"{loc}.series[{i}]", "each plot series needs a name and points", "Transcribe all readable points or a justified approximate range.")
    if kind in {"diagram", "flowchart", "scheme"}:
        if not isinstance(fig.get("diagram_nodes"), list) or not fig.get("diagram_nodes"):
            _diag(diagnostics, "error", "DIAGRAM_NODES", loc, "diagram/flowchart needs diagram_nodes", "List all visible components or steps.")
        if not isinstance(fig.get("diagram_edges"), list):
            _diag(diagnostics, "error", "DIAGRAM_EDGES", loc, "diagram/flowchart needs diagram_edges array", "List all visible directed relations (possibly empty).")
    if fig.get("all_visible_values_transcribed") is not True:
        _diag(diagnostics, "error", "FIGURE_INCOMPLETE", loc, "all_visible_values_transcribed must be true", "Inspect the image and expand visible values/annotations.")


def _check_article_content(
    node: dict[str, Any],
    idx: int,
    paper_dir: Path,
    diagnostics: list[dict[str, Any]],
    artifact_ids: set[str],
    evidence: dict[str, dict[str, Any]],
    *,
    current_schema: bool = True,
) -> None:
    loc = f"$.nodes[{idx}].article_content"
    content = node.get("article_content")
    if not isinstance(content, dict):
        _diag(diagnostics, "error", "ARTICLE_CONTENT_TYPE", loc, "article_content must be an object", "Use the four-part node contract.")
        return
    missing = REQUIRED_ARTICLE_KEYS - set(content)
    extra = set(content) - REQUIRED_ARTICLE_KEYS
    for key in sorted(missing):
        _diag(diagnostics, "error", "ARTICLE_CONTENT_FIELD", loc, f"missing article_content field: {key}", "Provide text, data and completeness fields.")
    if extra:
        _diag(diagnostics, "error", "ARTICLE_CONTENT_EXTRA", loc, f"unexpected article_content fields: {sorted(extra)}", "Keep article_content to the declared contract.")
    subtype = content.get("subtype")
    stage = node.get("node_type", {}).get("research_stage") if isinstance(node.get("node_type"), dict) else None
    allowed_conclusion_subtypes = CONCLUSION_SUBTYPES if current_schema else LEGACY_CONCLUSION_SUBTYPES
    if stage == "conclusion" and subtype not in allowed_conclusion_subtypes:
        _diag(diagnostics, "error", "CONCLUSION_SUBTYPE", loc, f"invalid conclusion subtype: {subtype}", f"Use one of {sorted(CONCLUSION_SUBTYPES)}.")
    if stage != "conclusion" and subtype in LEGACY_CONCLUSION_SUBTYPES:
        _diag(diagnostics, "error", "SUBTYPE_STAGE", loc, "conclusion subtype used outside conclusion stage", "Use a stage-specific subtype.")
    if not current_schema and stage == "conclusion" and subtype == "limitation":
        _diag(diagnostics, "warning", "LEGACY_LIMITATION_SUBTYPE", loc, "legacy graph uses the removed limitation conclusion subtype", "Regenerate this conclusion as stage_conclusion, branch_synthesis or overall_conclusion under v3.2.")
    text = content.get("text")
    if not isinstance(text, dict):
        _diag(diagnostics, "error", "TEXT_SECTION_TYPE", f"{loc}.text", "text must be an object", "Fill the semantic text section.")
    else:
        for key in ("statement", "facts", "definitions", "conditions", "reported_interpretation"):
            if key not in text:
                _diag(diagnostics, "error", "TEXT_SECTION_FIELD", f"{loc}.text", f"missing text field: {key}", "Provide all text fields; use [] when genuinely absent.")
        if not _is_nonempty_string(text.get("statement")):
            _diag(diagnostics, "error", "TEXT_STATEMENT_EMPTY", f"{loc}.text.statement", "statement must be non-empty", "State the current node's bounded paper fact.")
        for key in ("facts", "definitions", "conditions", "reported_interpretation"):
            if key in text and not _uniq_strings(text[key]):
                _diag(diagnostics, "error", "TEXT_LIST_SCHEMA", f"{loc}.text.{key}", "must be a list of unique non-empty strings", "Use [] or explicit atomic facts.")
        for value in [text.get("statement"), *(text.get("facts") or []), *(text.get("definitions") or []), *(text.get("conditions") or []), *(text.get("reported_interpretation") or [])]:
            if isinstance(value, str) and PLACEHOLDER_RE.match(value.strip()):
                _diag(diagnostics, "error", "CONTENT_PLACEHOLDER", f"{loc}.text", "article content is only a source pointer", "Expand the referenced table/figure/text directly.")
    data = content.get("data")
    if not isinstance(data, dict):
        _diag(diagnostics, "error", "DATA_SECTION_TYPE", f"{loc}.data", "data must be an object", "Fill tables, figures, numeric_records and formulas arrays.")
        return
    for key in ("tables", "figures", "numeric_records", "formulas"):
        if key not in data or not isinstance(data[key], list):
            _diag(diagnostics, "error", "DATA_SECTION_FIELD", f"{loc}.data.{key}", "must be an array", "Use [] when absent; never omit the field.")
    for j, table in enumerate(data.get("tables", []) if isinstance(data.get("tables"), list) else []):
        tid = table.get("id") if isinstance(table, dict) else None
        if _is_nonempty_string(tid):
            if tid in artifact_ids:
                _diag(diagnostics, "error", "DUPLICATE_ARTIFACT_ID", f"{loc}.data.tables[{j}]", f"duplicate artifact id: {tid}", "Use a globally unique table id.")
            artifact_ids.add(tid)
        _check_table(table, f"{loc}.data.tables[{j}]", paper_dir, diagnostics, evidence)
    for j, fig in enumerate(data.get("figures", []) if isinstance(data.get("figures"), list) else []):
        fid = fig.get("id") if isinstance(fig, dict) else None
        if _is_nonempty_string(fid):
            if fid in artifact_ids:
                _diag(diagnostics, "error", "DUPLICATE_ARTIFACT_ID", f"{loc}.data.figures[{j}]", f"duplicate artifact id: {fid}", "Use a globally unique figure id.")
            artifact_ids.add(fid)
        _check_figure(fig, f"{loc}.data.figures[{j}]", paper_dir, diagnostics, evidence)
    for j, record in enumerate(data.get("numeric_records", []) if isinstance(data.get("numeric_records"), list) else []):
        rloc = f"{loc}.data.numeric_records[{j}]"
        if not isinstance(record, dict):
            _diag(diagnostics, "error", "NUMERIC_RECORD_TYPE", rloc, "numeric record must be an object", "Record metric, value, unit and condition.")
            continue
        for key in ("id", "metric", "value", "unit", "condition", "uncertainty", "source_artifact_id"):
            if key not in record:
                _diag(diagnostics, "error", "NUMERIC_RECORD_FIELD", rloc, f"missing numeric record field: {key}", "Do not drop units, conditions or source artifact.")
        rid = record.get("id")
        if _is_nonempty_string(rid):
            if rid in artifact_ids:
                _diag(diagnostics, "error", "DUPLICATE_ARTIFACT_ID", rloc, f"duplicate artifact id: {rid}", "Use a globally unique numeric record id.")
            artifact_ids.add(rid)
        if not _is_nonempty_string(record.get("metric")) or not _is_nonempty_string(record.get("condition")):
            _diag(diagnostics, "error", "NUMERIC_RECORD_EMPTY", rloc, "metric and condition must be non-empty", "Make the scientific context explicit.")
    for j, formula in enumerate(data.get("formulas", []) if isinstance(data.get("formulas"), list) else []):
        floc = f"{loc}.data.formulas[{j}]"
        if not isinstance(formula, dict):
            _diag(diagnostics, "error", "FORMULA_TYPE", floc, "formula must be an object", "Record expression and variables.")
            continue
        for key in ("id", "label", "original", "normalized", "variables", "conditions"):
            if key not in formula:
                _diag(diagnostics, "error", "FORMULA_FIELD", floc, f"missing formula field: {key}", "Preserve original and normalized expression plus variables.")
        if not _is_nonempty_string(formula.get("original")) or not _is_nonempty_string(formula.get("normalized")):
            _diag(diagnostics, "error", "FORMULA_EMPTY", floc, "formula expressions must be non-empty", "Copy the complete expression.")
    comp = content.get("completeness")
    if not isinstance(comp, dict):
        _diag(diagnostics, "error", "COMPLETENESS_TYPE", f"{loc}.completeness", "completeness must be an object", "Declare extraction completeness.")
    else:
        for key in ("all_visible_values_transcribed", "all_semantic_text_expanded", "unresolved_items"):
            if key not in comp:
                _diag(diagnostics, "error", "COMPLETENESS_FIELD", f"{loc}.completeness", f"missing completeness field: {key}", "Declare every completeness flag.")
        if comp.get("all_visible_values_transcribed") is not True:
            _diag(diagnostics, "error", "INCOMPLETE_VALUES", f"{loc}.completeness", "all_visible_values_transcribed must be true", "Expand every visible value or repair the extraction.")
        if comp.get("all_semantic_text_expanded") is not True:
            _diag(diagnostics, "error", "INCOMPLETE_TEXT", f"{loc}.completeness", "all_semantic_text_expanded must be true", "Expand the semantic content instead of keeping a source pointer.")
        if not isinstance(comp.get("unresolved_items"), list):
            _diag(diagnostics, "error", "UNRESOLVED_SCHEMA", f"{loc}.completeness.unresolved_items", "must be an array", "Use [] when complete.")
        elif comp["unresolved_items"]:
            _diag(diagnostics, "error", "UNRESOLVED_ITEMS", f"{loc}.completeness.unresolved_items", "unresolved extraction items remain", "Inspect the source image/text and expand the missing content.")


def _check_metric(metric: Any, loc: str, diagnostics: list[dict[str, Any]], scientific: bool = False) -> None:
    if not isinstance(metric, dict):
        _diag(diagnostics, "error", "METRIC_TYPE", loc, "metric must be an object", "Use the detailed metric contract.")
        return
    if scientific:
        required = ("id", "name", "unit", "direction", "transformation", "raw_input_fields", "reference_values", "comparison_policy")
        for key in required:
            if key not in metric:
                _diag(diagnostics, "error", "SCIENTIFIC_METRIC_FIELD", loc, f"missing field: {key}", "Define how Agent raw outputs are normalized.")
        if metric.get("direction") not in {"higher_is_better", "lower_is_better", "target_value", "descriptive"}:
            _diag(diagnostics, "error", "SCIENTIFIC_METRIC_DIRECTION", loc, "invalid direction", "Use higher_is_better, lower_is_better, target_value or descriptive.")
        if not _is_nonempty_string(metric.get("transformation")) or not _is_nonempty_string(metric.get("comparison_policy")):
            _diag(diagnostics, "error", "SCIENTIFIC_METRIC_DETAILS", loc, "transformation and comparison_policy must be detailed", "Specify tokenization/formula/unit and reference comparison policy.")
        if not _uniq_strings(metric.get("raw_input_fields")):
            _diag(diagnostics, "error", "SCIENTIFIC_METRIC_INPUTS", loc, "raw_input_fields must be a non-empty unique string list", "List Agent artifacts consumed by the scientific metric.")
        return
    required = ("id", "name", "class", "target", "role", "definition", "input_fields", "score_range", "scoring", "failure_conditions", "missing_data_policy")
    for key in required:
        if key not in metric:
            _diag(diagnostics, "error", "BENCHMARK_METRIC_FIELD", loc, f"missing field: {key}", "Define the metric and its scoring details completely.")
    if metric.get("class") not in METRIC_CLASSES or metric.get("role") not in METRIC_ROLES:
        _diag(diagnostics, "error", "BENCHMARK_METRIC_ENUM", loc, "invalid metric class or role", f"class={sorted(METRIC_CLASSES)}, role={sorted(METRIC_ROLES)}")
    if not _is_nonempty_string(metric.get("definition")) or len(metric.get("definition", "")) < 20:
        _diag(diagnostics, "error", "BENCHMARK_METRIC_DEFINITION", loc, "definition is missing or too short", "Explain capability target and non-target explicitly.")
    if not _uniq_strings(metric.get("input_fields")):
        _diag(diagnostics, "error", "BENCHMARK_METRIC_INPUTS", loc, "input_fields must be a non-empty unique string list", "Name the external run artifacts or released analysis fields being scored.")
    rng = metric.get("score_range")
    if not isinstance(rng, list) or len(rng) != 2 or not all(isinstance(x, (int, float)) for x in rng) or rng[0] >= rng[1]:
        _diag(diagnostics, "error", "BENCHMARK_METRIC_RANGE", loc, "score_range must be [min,max] with min<max", "Use [0,1] or another explicit numeric interval.")
    scoring = metric.get("scoring")
    if not isinstance(scoring, dict) or scoring.get("type") not in SCORING_TYPES or not _is_nonempty_string(scoring.get("details")) or len(scoring.get("details", "")) < 30:
        _diag(diagnostics, "error", "BENCHMARK_METRIC_SCORING", loc, "scoring.type/details are incomplete", "Give the formula, rubric anchors, tolerances and weighting.")
    for key in ("failure_conditions",):
        if not _uniq_strings(metric.get(key)):
            _diag(diagnostics, "error", "BENCHMARK_METRIC_FAILURES", loc, f"{key} must be a unique string list", "List concrete failure cases.")
    if not _is_nonempty_string(metric.get("missing_data_policy")):
        _diag(diagnostics, "error", "BENCHMARK_METRIC_MISSING_POLICY", loc, "missing_data_policy is required", "State how missing outputs are scored.")


def _check_runtime_free_node(node: dict[str, Any], idx: int, diagnostics: list[dict[str, Any]]) -> None:
    """Ensure the static extraction contains no runtime Agent contract.

    Agent prompts, released inputs, output schemas and run artifacts belong to
    the external benchmark runner. Keeping them out of the paper graph avoids
    mixing paper Golden Answer data with a particular evaluation run.
    """
    loc = f"$.nodes[{idx}]"
    if "agent_content" in node:
        _diag(
            diagnostics,
            "error",
            "AGENT_CONTENT_FORBIDDEN",
            f"{loc}.agent_content",
            "agent_content is forbidden in schema v3 nodes",
            "Remove agent_content; store only node_type, article_content and evaluation_plan (plus graph metadata).",
        )
    plan = node.get("evaluation_plan")
    if isinstance(plan, dict):
        for section in ("benchmark_metrics", "paper_scientific_metrics"):
            metrics = plan.get(section, [])
            if isinstance(metrics, list):
                for j, metric in enumerate(metrics):
                    if not isinstance(metric, dict):
                        continue
                    fields = metric.get("input_fields") if section == "benchmark_metrics" else metric.get("raw_input_fields")
                    if isinstance(fields, list) and any("agent_content" in str(field) for field in fields):
                        _diag(
                            diagnostics,
                            "error",
                            "STALE_AGENT_REFERENCE",
                            f"{loc}.evaluation_plan.{section}[{j}]",
                            "metric references removed agent_content fields",
                            "Reference external run artifacts such as run_artifacts.method_output or run_artifacts.experiment_result.",
                        )


def _check_nodes(
    doc: dict[str, Any],
    paper_dir: Path,
    nodes: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    *,
    current_schema: bool = True,
) -> None:
    motivation = []
    overall = []
    artifact_ids: set[str] = set()
    for idx, node in enumerate(doc.get("nodes", [])):
        loc = f"$.nodes[{idx}]"
        extra = set(node) - REQUIRED_NODE_KEYS - OPTIONAL_NODE_KEYS
        missing = REQUIRED_NODE_KEYS - set(node)
        for key in sorted(missing):
            _diag(diagnostics, "error", "NODE_FIELD", loc, f"missing node field: {key}", "Use id, node_type, article_content, evaluation_plan and evidence_refs, plus optional graph metadata.")
        if extra:
            _diag(diagnostics, "error", "NODE_EXTRA", loc, f"unexpected node fields: {sorted(extra)}", "Keep graph fields plus node_type, article_content and evaluation_plan only.")
        if "parent_id" in node and "parent_ids" in node:
            _diag(diagnostics, "error", "NODE_PARENT_FIELDS_BOTH", loc, "use either legacy parent_id or current parent_ids, not both", "Use parent_ids in the current v3.2 schema.")
        elif current_schema and "parent_id" in node:
            _diag(diagnostics, "error", "NODE_LEGACY_PARENT", f"{loc}.parent_id", "current v3.2 nodes must use parent_ids", "Replace parent_id with a zero-or-more parent_ids list.")
        if current_schema and "parent_ids" not in node:
            _diag(diagnostics, "error", "NODE_PARENTS_MISSING", f"{loc}.parent_ids", "current v3.2 nodes must declare parent_ids, which may be an empty list", "Add parent_ids: [] when no scientifically justified parent exists.")
        if current_schema and "depends_on" not in node:
            _diag(diagnostics, "error", "NODE_DEPENDS_MISSING", f"{loc}.depends_on", "current v3.2 nodes must declare depends_on, which may be an empty list", "Add depends_on: [] when no dependency exists.")
        nt = node.get("node_type")
        if not isinstance(nt, dict) or set(nt) != {"research_stage", "data_type", "evaluability"}:
            _diag(diagnostics, "error", "NODE_TYPE_SCHEMA", f"{loc}.node_type", "node_type must contain exactly research_stage, data_type and evaluability", "Use only marker fields in node_type.")
        else:
            stage = nt.get("research_stage")
            if stage not in STAGES:
                _diag(diagnostics, "error", "NODE_STAGE", f"{loc}.node_type.research_stage", f"invalid stage: {stage}", f"Use one of {sorted(STAGES)}.")
            if nt.get("data_type") not in DATA_TYPES or nt.get("evaluability") not in EVALUABILITY:
                _diag(diagnostics, "error", "NODE_TYPE_ENUM", f"{loc}.node_type", "invalid data_type or evaluability", "Use text/data and evaluable/not_evaluable.")
            if stage == "motivation":
                motivation.append(node)
                if nt.get("evaluability") != "not_evaluable":
                    _diag(diagnostics, "error", "MOTIVATION_EVALUABILITY", f"{loc}.node_type.evaluability", "motivation is an input-only root and must be marked not_evaluable", "Set motivation evaluability to not_evaluable.")
            if stage == "conclusion" and isinstance(node.get("article_content"), dict) and node["article_content"].get("subtype") == "overall_conclusion":
                overall.append(node)
        refs = node.get("evidence_refs")
        if not _uniq_strings(refs):
            _diag(diagnostics, "error", "NODE_EVIDENCE_REFS", f"{loc}.evidence_refs", "evidence_refs must be a unique non-empty string list", "Attach evidence to every node.")
        else:
            for ref in refs:
                if ref not in evidence:
                    _diag(diagnostics, "error", "NODE_EVIDENCE_UNKNOWN", f"{loc}.evidence_refs", f"unknown evidence id: {ref}", "Create the evidence record first.")
        parent_ids = node.get("parent_ids", [])
        if not _uniq_strings(parent_ids):
            _diag(diagnostics, "error", "NODE_PARENTS_SCHEMA", f"{loc}.parent_ids", "parent_ids must be a unique string list", "Use [] when no scientifically justified parent exists, or list every justified parent node ID.")
        else:
            for ref in parent_ids:
                if ref not in nodes:
                    _diag(diagnostics, "error", "NODE_PARENT_UNKNOWN", f"{loc}.parent_ids", f"unknown parent: {ref}", "Reference an existing node or leave parent_ids empty when no relation is evidenced.")
        depends = node.get("depends_on", [])
        if not _uniq_strings(depends):
            _diag(diagnostics, "error", "NODE_DEPENDS_SCHEMA", f"{loc}.depends_on", "depends_on must be a unique string list", "Use [] or existing node IDs.")
        else:
            for ref in depends:
                if ref not in nodes:
                    _diag(diagnostics, "error", "NODE_DEPENDS_UNKNOWN", f"{loc}.depends_on", f"unknown dependency: {ref}", "Reference an existing node.")
        _check_article_content(node, idx, paper_dir, diagnostics, artifact_ids, evidence, current_schema=current_schema)
        _check_runtime_free_node(node, idx, diagnostics)
        plan = node.get("evaluation_plan")
        if not isinstance(plan, dict) or set(plan) != {"score_role", "benchmark_metrics", "paper_scientific_metrics"}:
            _diag(diagnostics, "error", "EVAL_PLAN_SCHEMA", f"{loc}.evaluation_plan", "evaluation_plan must contain score_role, benchmark_metrics and paper_scientific_metrics", "Separate Benchmark metrics from paper metric transformations.")
        else:
            if plan.get("score_role") not in SCORE_ROLES:
                _diag(diagnostics, "error", "EVAL_SCORE_ROLE", f"{loc}.evaluation_plan.score_role", "invalid score_role", f"Use one of {sorted(SCORE_ROLES)}.")
            if not isinstance(plan.get("benchmark_metrics"), list) or not isinstance(plan.get("paper_scientific_metrics"), list):
                _diag(diagnostics, "error", "EVAL_METRIC_LIST", f"{loc}.evaluation_plan", "metric fields must be arrays", "Use [] when no metric is applicable.")
            else:
                for j, metric in enumerate(plan["benchmark_metrics"]):
                    _check_metric(metric, f"{loc}.evaluation_plan.benchmark_metrics[{j}]", diagnostics, scientific=False)
                for j, metric in enumerate(plan["paper_scientific_metrics"]):
                    _check_metric(metric, f"{loc}.evaluation_plan.paper_scientific_metrics[{j}]", diagnostics, scientific=True)
                stage = nt.get("research_stage") if isinstance(nt, dict) else None
                evaluability = nt.get("evaluability") if isinstance(nt, dict) else None
                if stage == "motivation":
                    if plan.get("score_role") != "not_scored" or plan["benchmark_metrics"]:
                        _diag(diagnostics, "error", "MOTIVATION_SCORE_PLAN", f"{loc}.evaluation_plan", "motivation is an input-only node and must not carry Benchmark metrics", "Set score_role=not_scored and benchmark_metrics=[]; keep only article content and evidence.")
                elif stage == "experiment" and evaluability == "not_evaluable":
                    if plan["benchmark_metrics"]:
                        _diag(diagnostics, "error", "NONEXECUTABLE_EXPERIMENT_METRICS", f"{loc}.evaluation_plan.benchmark_metrics", "not_evaluable experiment nodes do not produce Agent outputs and must not carry direct execution metrics", "Move reasoning metrics to the downstream conclusion node; retain paper_scientific_metrics only when needed for the reported result.")
                    if plan.get("score_role") == "core":
                        _diag(diagnostics, "error", "NONEXECUTABLE_EXPERIMENT_CORE", f"{loc}.evaluation_plan.score_role", "not_evaluable experiment cannot be a core Agent-scored node", "Use diagnostic or not_scored for the result-only node.")
                if nt and nt.get("data_type") == "data" and isinstance(node.get("article_content"), dict) and node["article_content"].get("data", {}).get("numeric_records") and not plan["paper_scientific_metrics"]:
                    _diag(diagnostics, "error", "DATA_WITHOUT_SCIENTIFIC_METRIC", f"{loc}.evaluation_plan.paper_scientific_metrics", "numeric article content needs a scientific-metric transformation", "Define how Agent raw artifacts become comparable scientific values.")
    if len(motivation) != 1:
        _diag(diagnostics, "error", "MOTIVATION_COUNT", "$.nodes", f"expected exactly one motivation node, found {len(motivation)}", "Create one root motivation node.")
    if len(overall) != 1:
        _diag(diagnostics, "error", "OVERALL_COUNT", "$.nodes", f"expected exactly one overall_conclusion node, found {len(overall)}", "Create one conclusion node with subtype overall_conclusion.")


def _check_graph(
    doc: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    *,
    current_schema: bool = True,
) -> None:
    edges = doc.get("edges")
    edge_map = _index(edges, "edges", diagnostics)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for eid, edge in edge_map.items():
        loc = f"$.edges[{eid}]"
        for key in ("source", "target", "relation", "evidence_refs", "reasoning", "confidence"):
            if key not in edge:
                _diag(diagnostics, "error", "EDGE_FIELD", loc, f"missing edge field: {key}", "Include source, target, relation, evidence refs, reasoning and confidence.")
        if edge.get("source") not in nodes or edge.get("target") not in nodes:
            _diag(diagnostics, "error", "EDGE_ENDPOINT", loc, "edge source/target not found", "Reference existing node IDs.")
        if edge.get("relation") not in RELATIONS:
            _diag(diagnostics, "error", "EDGE_RELATION", loc, f"invalid edge relation: {edge.get('relation')}", f"Use one of {sorted(RELATIONS)}.")
        if not _uniq_strings(edge.get("evidence_refs")):
            _diag(diagnostics, "error", "EDGE_EVIDENCE", loc, "edge evidence_refs must be a unique non-empty string list", "Provide evidence for the dependency/claim.")
        if not _is_nonempty_string(edge.get("reasoning")) or edge.get("confidence") not in CONFIDENCE:
            _diag(diagnostics, "error", "EDGE_METADATA", loc, "edge reasoning/confidence is incomplete", "Explain why the relation holds and give confidence.")
        incoming[edge.get("target")].append(edge)
    paper = doc.get("paper", {})
    root_id = paper.get("root_motivation_id")
    overall_id = paper.get("overall_conclusion_id")
    if root_id not in nodes or overall_id not in nodes:
        return
    if not current_schema:
        # Preserve the v3/v3.1 single-parent validation path for migration.
        parent: dict[str, str] = {}
        for nid, node in nodes.items():
            if nid == root_id:
                if "parent_id" in node:
                    _diag(diagnostics, "error", "ROOT_PARENT", f"$.nodes[{nid}]", "motivation root must not have parent_id", "Omit parent_id for the root.")
                continue
            pid = node.get("parent_id")
            if not _is_nonempty_string(pid) or pid not in nodes:
                _diag(diagnostics, "error", "PARENT_INVALID", f"$.nodes[{nid}].parent_id", "every non-root legacy node needs an existing parent_id", "Attach the node to its legacy graph parent.")
            else:
                parent[nid] = pid
                if not any(e.get("source") == pid and e.get("target") == nid for e in incoming.get(nid, [])):
                    _diag(diagnostics, "error", "PARENT_EDGE_MISSING", f"$.nodes[{nid}]", "parent_id has no corresponding incoming edge", "Add a structural edge from parent to child.")
        for start in nodes:
            seen: set[str] = set()
            cur = start
            while cur in parent:
                if cur in seen:
                    _diag(diagnostics, "error", "LEGACY_GRAPH_CYCLE", f"$.nodes[{start}]", "legacy parent pointers contain a cycle", "Break the cycle in the legacy single-parent graph.")
                    break
                seen.add(cur)
                cur = parent[cur]
        children: dict[str, list[str]] = defaultdict(list)
        for child, pid in parent.items():
            children[pid].append(child)
        reached: set[str] = set()
        pending = [root_id]
        while pending:
            cur = pending.pop()
            if cur in reached:
                continue
            reached.add(cur)
            pending.extend(children.get(cur, []))
        for nid in nodes:
            if nid not in reached:
                _diag(diagnostics, "error", "UNREACHABLE_NODE", f"$.nodes[{nid}]", "node is not reachable from motivation root", "Connect it to the legacy graph or regenerate with v3.2 explicit graph relations.")
        if children.get(overall_id):
            _diag(diagnostics, "error", "OVERALL_NOT_LEAF", f"$.nodes[{overall_id}]", "overall conclusion must be the legacy graph's terminal node", "Move any synthesis before overall conclusion.")
        if not any(e.get("target") == overall_id and e.get("relation") in {"aggregates", "supports", "contains"} for e in edge_map.values()):
            _diag(diagnostics, "error", "OVERALL_NO_SUPPORT", f"$.nodes[{overall_id}]", "overall conclusion has no incoming synthesis/support edge", "Aggregate stage conclusions before the final node.")
        return
    # parent_ids are optional scientific structure annotations, not a required
    # spanning structure. Every listed parent relationship still needs an explicit
    # graph edge, but nodes may legitimately have no parent or several parents.
    for nid, node in nodes.items():
        parent_ids = node.get("parent_ids", [])
        if not _uniq_strings(parent_ids):
            continue
        if nid == root_id and parent_ids:
            _diag(diagnostics, "error", "ROOT_PARENT", f"$.nodes[{nid}].parent_ids", "motivation root must not have parent_ids", "Use an empty parent_ids list for the designated motivation root.")
        for pid in parent_ids:
            if pid in nodes and not any(e.get("source") == pid and e.get("target") == nid for e in incoming.get(nid, [])):
                _diag(diagnostics, "error", "PARENT_EDGE_MISSING", f"$.nodes[{nid}].parent_ids", f"parent {pid} has no corresponding explicit graph edge", "Add the scientifically justified edge or remove the parent annotation.")

        depends = node.get("depends_on", [])
        if _uniq_strings(depends):
            for dep in depends:
                if dep in nodes and not any(
                    e.get("source") == dep and e.get("target") == nid and e.get("relation") == "depends_on"
                    for e in incoming.get(nid, [])
                ):
                    _diag(diagnostics, "error", "DEPENDS_EDGE_MISSING", f"$.nodes[{nid}].depends_on", f"dependency {dep} has no explicit depends_on edge", "Add a depends_on edge from the dependency to this node.")

    # The field and edge representations of depends_on must agree in both directions.
    for eid, edge in edge_map.items():
        if edge.get("relation") != "depends_on":
            continue
        target = nodes.get(edge.get("target"))
        depends = target.get("depends_on", []) if isinstance(target, dict) else []
        if edge.get("source") not in depends:
            _diag(diagnostics, "error", "DEPENDS_FIELD_MISSING", f"$.edges[{eid}]", "depends_on edge is not mirrored in the target node's depends_on list", "Add the source ID to target.depends_on or remove this edge.")

    # The overall conclusion remains the designated terminal synthesis node,
    # but unrelated components are allowed and no artificial parent edge is
    # required merely to make it the only graph leaf.
    if not any(e.get("target") == overall_id and e.get("relation") in {"aggregates", "supports", "contains"} for e in edge_map.values()):
        _diag(diagnostics, "error", "OVERALL_NO_SUPPORT", f"$.nodes[{overall_id}]", "overall conclusion has no incoming synthesis/support edge", "Aggregate stage conclusions before the final node.")


def _check_method_relations(
    doc: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    evidence_ids: set[str],
    diagnostics: list[dict[str, Any]],
    *,
    current_schema: bool = True,
) -> None:
    methods = sorted(nid for nid, n in nodes.items() if isinstance(n.get("node_type"), dict) and n["node_type"].get("research_stage") == "method")
    relations = doc.get("method_relations")
    if not isinstance(relations, list):
        _diag(diagnostics, "error", "METHOD_RELATIONS_TYPE", "$.method_relations", "method_relations must be an array", "Provide one relation record for each pair of Method nodes.")
        return
    seen: set[tuple[str, str]] = set()
    for i, rel in enumerate(relations):
        loc = f"$.method_relations[{i}]"
        for key in ("method_a", "method_b", "relation", "serial_basis", "output_of_a_used_by_b", "b_can_be_predefined_before_a", "implementation_required", "adaptive_decision", "direct_data_transfer", "evidence_refs", "confidence", "rationale"):
            if key not in rel:
                _diag(diagnostics, "error", "METHOD_RELATION_FIELD", loc, f"missing field: {key}", "Complete the pairwise Method relation record.")
        if current_schema:
            if "graph_action" not in rel and "tree_action" not in rel:
                _diag(diagnostics, "error", "METHOD_RELATION_FIELD", loc, "missing field: graph_action", "Complete the pairwise Method relation record with graph_action.")
            elif "graph_action" in rel and "tree_action" in rel:
                _diag(diagnostics, "error", "METHOD_RELATION_ACTION_BOTH", loc, "use graph_action, not both graph_action and the legacy tree_action field", "Remove tree_action and keep graph_action.")
            elif "tree_action" in rel:
                _diag(diagnostics, "warning", "LEGACY_TREE_ACTION", f"{loc}.tree_action", "legacy method relation uses tree_action", "Rename tree_action to graph_action when regenerating the graph.")
        elif "tree_action" not in rel:
            _diag(diagnostics, "error", "METHOD_RELATION_FIELD", loc, "missing field: tree_action", "Complete the legacy v3/v3.1 Method relation record.")
        action = rel.get("graph_action") if "graph_action" in rel else rel.get("tree_action")
        a, b = rel.get("method_a"), rel.get("method_b")
        if a not in methods or b not in methods or a == b:
            _diag(diagnostics, "error", "METHOD_RELATION_ENDPOINT", loc, "relation endpoints must be two distinct method node IDs", "Reference two Method nodes.")
            continue
        pair = tuple(sorted((a, b)))
        if pair in seen:
            _diag(diagnostics, "error", "METHOD_RELATION_DUPLICATE", loc, f"duplicate relation pair: {pair}", "Keep exactly one record per unordered method pair.")
        seen.add(pair)
        if rel.get("relation") not in METHOD_RELATIONS:
            _diag(diagnostics, "error", "METHOD_RELATION_ENUM", loc, "invalid Method relation", f"Use one of {sorted(METHOD_RELATIONS)}.")
        elif current_schema:
            edge_relation = METHOD_EDGE_RELATIONS[rel["relation"]]
            edges = doc.get("edges", []) if isinstance(doc.get("edges"), list) else []
            if not any(
                isinstance(edge, dict)
                and edge.get("source") == a
                and edge.get("target") == b
                and edge.get("relation") == edge_relation
                for edge in edges
            ):
                _diag(diagnostics, "error", "METHOD_RELATION_EDGE_MISSING", loc, f"Method relation has no explicit {edge_relation} graph edge from {a} to {b}", "Add a graph edge for every method_relations record; parallel maps to parallel_branch.")
        if not _uniq_strings(rel.get("evidence_refs")):
            _diag(diagnostics, "error", "METHOD_RELATION_EVIDENCE", loc, "strict/technical relation needs evidence_refs", "Cite the adaptive decision or data transfer.")
        elif any(eid not in evidence_ids for eid in rel["evidence_refs"]):
            _diag(diagnostics, "error", "METHOD_RELATION_UNKNOWN_EVIDENCE", loc, "relation references unknown evidence", "Create all evidence IDs before relations.")
        for key in ("output_of_a_used_by_b", "b_can_be_predefined_before_a", "implementation_required", "adaptive_decision", "direct_data_transfer"):
            if not isinstance(rel.get(key), bool):
                _diag(diagnostics, "error", "METHOD_RELATION_BOOL", f"{loc}.{key}", "field must be boolean", "Use true or false based on explicit evidence.")
        if rel.get("confidence") not in CONFIDENCE or not _is_nonempty_string(rel.get("rationale")):
            _diag(diagnostics, "error", "METHOD_RELATION_META", loc, "confidence/rationale is incomplete", "Explain the relation in scientific terms.")
        if action not in GRAPH_ACTIONS:
            _diag(diagnostics, "error", "METHOD_RELATION_GRAPH_ACTION", f"{loc}.graph_action", f"invalid graph_action: {action}", f"Use one of {sorted(GRAPH_ACTIONS)}.")
        basis = rel.get("serial_basis")
        if basis not in {"none", "implementation_dependency", "adaptive_result_dependency"}:
            _diag(diagnostics, "error", "METHOD_SERIAL_BASIS", f"{loc}.serial_basis", "invalid serial_basis", "Use none, implementation_dependency or adaptive_result_dependency.")
        if rel.get("relation") == "strict_serial":
            common_gate = rel.get("output_of_a_used_by_b") is True and bool(rel.get("evidence_refs"))
            if not common_gate:
                _diag(diagnostics, "error", "STRICT_SERIAL_GATE", loc, "strict_serial requires explicit use of an A implementation/result artifact and evidence", "Use parallel/technical_dependency unless B really consumes an A implementation or runtime artifact.")
            if basis == "implementation_dependency":
                if rel.get("implementation_required") is not True:
                    _diag(diagnostics, "error", "IMPLEMENTATION_SERIAL_GATE", loc, "implementation_dependency requires implementation_required=true", "Set implementation_required when B cannot run without A's implementation or training artifact.")
                if action != "child_of_a":
                    _diag(diagnostics, "error", "IMPLEMENTATION_SERIAL_GRAPH_ACTION", loc, "implementation_dependency must use graph_action=child_of_a", "Record the evidenced method-to-method graph relation from A to B.")
                elif (
                    a not in nodes.get(b, {}).get("parent_ids", [])
                    if current_schema
                    else nodes.get(b, {}).get("parent_id") != a
                ):
                    parent_field = "parent_ids" if current_schema else "parent_id"
                    _diag(diagnostics, "error", "IMPLEMENTATION_SERIAL_PARENT", loc, f"implementation-dependent method must list method_a in {parent_field}", f"Record method_a in method_b.{parent_field} because this is an evidenced scientific implementation dependency.")
            elif basis == "adaptive_result_dependency":
                if not (rel.get("b_can_be_predefined_before_a") is False and rel.get("adaptive_decision") is True):
                    _diag(diagnostics, "error", "ADAPTIVE_SERIAL_GATE", loc, "adaptive_result_dependency requires an A-result-driven decision", "Set b_can_be_predefined_before_a=false and adaptive_decision=true only with evidence.")
                if rel.get("direct_data_transfer") is not True:
                    _diag(diagnostics, "error", "ADAPTIVE_SERIAL_TRANSFER", loc, "adaptive_result_dependency requires direct transfer of an A result", "Set direct_data_transfer=true only when B consumes A's runtime result.")
                if action != "child_of_a_conclusion":
                    _diag(diagnostics, "error", "ADAPTIVE_SERIAL_GRAPH_ACTION", loc, "adaptive_result_dependency must use graph_action=child_of_a_conclusion", "Record the evidenced result-driven graph relation from A toward B.")
            else:
                _diag(diagnostics, "error", "STRICT_SERIAL_BASIS", loc, "strict_serial needs an explicit serial_basis", "Choose implementation_dependency or adaptive_result_dependency.")
            if not current_schema:
                edges = doc.get("edges", []) if isinstance(doc.get("edges"), list) else []
                if not any(
                    isinstance(edge, dict)
                    and edge.get("source") == a
                    and edge.get("target") == b
                    and edge.get("relation") == "strict_serial"
                    for edge in edges
                ):
                    _diag(diagnostics, "error", "STRICT_SERIAL_EDGE_MISSING", loc, "strict_serial relation has no matching strict_serial graph edge", "Add a strict_serial edge from method_a to method_b.")
        if rel.get("relation") != "strict_serial" and action in {"child_of_a", "child_of_a_conclusion"}:
            _diag(diagnostics, "error", "NONSTRICT_CHILD_ACTION", loc, "non-strict relation cannot use a serial child graph action", "Use same_parent_siblings or same_method_protocol.")
    expected = set(combinations(methods, 2))
    # combinations are sorted by methods, therefore pair order is stable.
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing:
            _diag(diagnostics, "error", "METHOD_RELATION_COVERAGE", "$.method_relations", f"missing method pairs: {missing}", "Add one relation record for every pair of methods.")
        if extra:
            _diag(diagnostics, "error", "METHOD_RELATION_EXTRA", "$.method_relations", f"unknown method pairs: {extra}", "Remove relations whose endpoints are not Method nodes.")


def _check_method_coverage(
    doc: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    evidence_ids: set[str],
    diagnostics: list[dict[str, Any]],
    *,
    current_schema: bool = True,
) -> None:
    coverage = doc.get("method_coverage")
    if not isinstance(coverage, list):
        return
    methods = {
        nid
        for nid, node in nodes.items()
        if isinstance(node.get("node_type"), dict) and node["node_type"].get("research_stage") == "method"
    }
    edges = {
        edge.get("id"): edge
        for edge in doc.get("edges", [])
        if isinstance(edge, dict) and _is_nonempty_string(edge.get("id"))
    }
    seen_methods: set[str] = set()
    for i, record in enumerate(coverage):
        loc = f"$.method_coverage[{i}]"
        if not isinstance(record, dict):
            _diag(diagnostics, "error", "METHOD_COVERAGE_RECORD", loc, "coverage entry must be an object", "Provide one coverage record for each Method.")
            continue
        for key in ("method_id", "coverage_status", "entries", "uncovered_reason", "uncovered_evidence_refs"):
            if key not in record:
                _diag(diagnostics, "error", "METHOD_COVERAGE_FIELD", loc, f"missing field: {key}", "Complete the Method coverage record.")
        method_id = record.get("method_id")
        if method_id not in methods:
            _diag(diagnostics, "error", "METHOD_COVERAGE_METHOD", f"{loc}.method_id", "method_id must reference a Method node", "Reference an existing node with research_stage=method.")
        elif method_id in seen_methods:
            _diag(diagnostics, "error", "METHOD_COVERAGE_DUPLICATE", f"{loc}.method_id", "each Method needs exactly one coverage record", "Merge entries under the same method_id.")
        else:
            seen_methods.add(method_id)
        status = record.get("coverage_status")
        if status not in COVERAGE_STATUSES:
            _diag(diagnostics, "error", "METHOD_COVERAGE_STATUS", f"{loc}.coverage_status", "invalid Method coverage status", f"Use one of {sorted(COVERAGE_STATUSES)}.")
        entries = record.get("entries")
        if not isinstance(entries, list):
            _diag(diagnostics, "error", "METHOD_COVERAGE_ENTRIES", f"{loc}.entries", "entries must be an array", "Use [] only when no direct follow-up is evidenced.")
            entries = []
        reason = record.get("uncovered_reason")
        if not isinstance(reason, str):
            _diag(diagnostics, "error", "METHOD_COVERAGE_REASON", f"{loc}.uncovered_reason", "uncovered_reason must be a string", "Use an empty string when the Method has coverage; otherwise explain the missing follow-up.")
            reason = ""
        uncovered_evidence_refs = record.get("uncovered_evidence_refs")
        if not _uniq_strings(uncovered_evidence_refs):
            _diag(diagnostics, "error", "METHOD_COVERAGE_UNCOVERED_EVIDENCE", f"{loc}.uncovered_evidence_refs", "uncovered_evidence_refs must be a unique string list", "Use [] for covered Methods and cite the exhaustive audit scope for no_followup.")
            uncovered_evidence_refs = []
        elif any(eid not in evidence_ids for eid in uncovered_evidence_refs):
            _diag(diagnostics, "error", "METHOD_COVERAGE_UNKNOWN_UNCOVERED_EVIDENCE", f"{loc}.uncovered_evidence_refs", "uncovered_evidence_refs contains unknown evidence", "Create the evidence records before citing the no-followup audit.")
        if status == "no_followup":
            if entries:
                _diag(diagnostics, "error", "METHOD_COVERAGE_NOFOLLOWUP_ENTRIES", loc, "no_followup cannot contain coverage entries", "Use a covered status when a downstream design or result exists.")
            if not reason.strip():
                _diag(diagnostics, "error", "METHOD_COVERAGE_NOFOLLOWUP_REASON", f"{loc}.uncovered_reason", "no_followup requires an evidence-based reason", "Explain why the source contains no downstream use or validation.")
            if not uncovered_evidence_refs:
                _diag(diagnostics, "error", "METHOD_COVERAGE_NOFOLLOWUP_EVIDENCE", f"{loc}.uncovered_evidence_refs", "no_followup requires evidence references", "Cite the Method definition and the exhaustively checked experiment/result scope.")
        elif not entries:
            _diag(diagnostics, "error", "METHOD_COVERAGE_EMPTY", loc, "a covered Method needs at least one design entry", "Add every experiment design that tests or uses the Method.")
        elif reason.strip():
            _diag(diagnostics, "error", "METHOD_COVERAGE_REASON_ON_COVERED", f"{loc}.uncovered_reason", "uncovered_reason must be empty when the Method has coverage", "Move the explanation to an entry rationale or use no_followup only when no coverage exists.")
        if status != "no_followup" and uncovered_evidence_refs:
            _diag(diagnostics, "error", "METHOD_COVERAGE_UNCOVERED_EVIDENCE_ON_COVERED", f"{loc}.uncovered_evidence_refs", "uncovered_evidence_refs must be empty when the Method has coverage", "Use entry evidence_refs for covered Methods.")
        entry_keys: set[tuple[Any, Any, Any, Any]] = set()
        for j, entry in enumerate(entries):
            eloc = f"{loc}.entries[{j}]"
            if not isinstance(entry, dict):
                _diag(diagnostics, "error", "METHOD_COVERAGE_ENTRY", eloc, "coverage entry must be an object", "Fill design, factor, edge and evidence fields.")
                continue
            for key in ("design_id", "coverage_type", "covered_factors", "edge_id", "evidence_refs", "rationale"):
                if key not in entry:
                    _diag(diagnostics, "error", "METHOD_COVERAGE_ENTRY_FIELD", eloc, f"missing field: {key}", "Complete the Method-to-design coverage entry.")
            design_id = entry.get("design_id")
            design = nodes.get(design_id, {}) if isinstance(design_id, str) else {}
            if not isinstance(design.get("node_type"), dict) or design["node_type"].get("research_stage") != "experiment_design":
                _diag(diagnostics, "error", "METHOD_COVERAGE_DESIGN", f"{eloc}.design_id", "design_id must reference an experiment_design node", "Reference the concrete downstream experiment design.")
            factors = entry.get("covered_factors")
            if not _uniq_strings(factors) or not factors:
                _diag(diagnostics, "error", "METHOD_COVERAGE_FACTORS", f"{eloc}.covered_factors", "covered_factors must be a non-empty unique string list", "List every changed, used, observed or compared factor.")
            coverage_type = entry.get("coverage_type")
            if coverage_type not in COVERAGE_TYPES:
                _diag(diagnostics, "error", "METHOD_COVERAGE_TYPE", f"{eloc}.coverage_type", "invalid coverage_type", f"Use one of {sorted(COVERAGE_TYPES)}.")
            evidence_refs = entry.get("evidence_refs")
            if not _uniq_strings(evidence_refs):
                _diag(diagnostics, "error", "METHOD_COVERAGE_EVIDENCE", f"{eloc}.evidence_refs", "coverage evidence_refs must be a unique non-empty string list", "Cite the table, figure, formula or paragraph supporting the attribution.")
            elif any(eid not in evidence_ids for eid in evidence_refs):
                _diag(diagnostics, "error", "METHOD_COVERAGE_UNKNOWN_EVIDENCE", f"{eloc}.evidence_refs", "coverage references unknown evidence", "Create the evidence before adding the coverage entry.")
            if not _is_nonempty_string(entry.get("rationale")):
                _diag(diagnostics, "error", "METHOD_COVERAGE_RATIONALE", f"{eloc}.rationale", "coverage rationale is required", "Explain the scientific attribution and its causal limits.")
            edge_id = entry.get("edge_id")
            edge = edges.get(edge_id) if isinstance(edge_id, str) else None
            if not isinstance(edge, dict):
                _diag(diagnostics, "error", "METHOD_COVERAGE_EDGE", f"{eloc}.edge_id", "edge_id must reference an existing graph edge", "Add an explicit Method-to-design coverage edge.")
            else:
                if edge.get("source") != method_id or edge.get("target") != design_id:
                    _diag(diagnostics, "error", "METHOD_COVERAGE_EDGE_ENDPOINT", f"{eloc}.edge_id", "coverage edge must run from method_id to design_id", "Use a direct Method→experiment_design edge.")
                if edge.get("relation") not in COVERAGE_EDGE_RELATIONS:
                    _diag(diagnostics, "error", "METHOD_COVERAGE_EDGE_RELATION", f"{eloc}.edge_id", "coverage edge has an invalid relation", f"Use one of {sorted(COVERAGE_EDGE_RELATIONS)}.")
            result_id = entry.get("result_id")
            conclusion_id = entry.get("conclusion_id")
            key = (design_id, result_id, conclusion_id, coverage_type)
            if key in entry_keys:
                _diag(diagnostics, "error", "METHOD_COVERAGE_ENTRY_DUPLICATE", eloc, "duplicate Method coverage entry", "Merge duplicate attribution entries or distinguish their factors.")
            entry_keys.add(key)
            if result_id is not None:
                result = nodes.get(result_id, {}) if isinstance(result_id, str) else {}
                if not isinstance(result.get("node_type"), dict) or result["node_type"].get("research_stage") != "experiment":
                    _diag(diagnostics, "error", "METHOD_COVERAGE_RESULT", f"{eloc}.result_id", "result_id must reference an experiment node", "Reference the result produced by the design.")
                has_result_link = (not current_schema and result.get("parent_id") == design_id) or any(
                    isinstance(edge_item, dict)
                    and edge_item.get("source") == design_id
                    and edge_item.get("target") == result_id
                    and edge_item.get("relation") == "produces"
                    for edge_item in doc.get("edges", [])
                )
                if not has_result_link:
                    _diag(diagnostics, "error", "METHOD_COVERAGE_RESULT_LINK", eloc, "design and result are not connected by an explicit produces edge", "Add a produces edge from design_id to result_id.")
            elif any(
                isinstance(edge_item, dict)
                and edge_item.get("source") == design_id
                and edge_item.get("relation") == "produces"
                for edge_item in doc.get("edges", [])
            ) or (
                not current_schema
                and any(
                    isinstance(candidate, dict)
                    and candidate.get("parent_id") == design_id
                    and isinstance(candidate.get("node_type"), dict)
                    and candidate["node_type"].get("research_stage") == "experiment"
                    for candidate in nodes.values()
                )
            ):
                _diag(diagnostics, "error", "METHOD_COVERAGE_RESULT_OMITTED", eloc, "a design with a produced result must record result_id", "Include the result node in the Method coverage entry.")
            if conclusion_id is not None:
                conclusion = nodes.get(conclusion_id, {}) if isinstance(conclusion_id, str) else {}
                if not isinstance(conclusion.get("node_type"), dict) or conclusion["node_type"].get("research_stage") != "conclusion":
                    _diag(diagnostics, "error", "METHOD_COVERAGE_CONCLUSION", f"{eloc}.conclusion_id", "conclusion_id must reference a conclusion node", "Reference the stage conclusion supported by the result.")
                has_conclusion_link = (not current_schema and conclusion.get("parent_id") == result_id) or any(
                    isinstance(edge_item, dict)
                    and edge_item.get("source") == result_id
                    and edge_item.get("target") == conclusion_id
                    and edge_item.get("relation") in {"supports", "aggregates"}
                    for edge_item in doc.get("edges", [])
                )
                if not has_conclusion_link:
                    _diag(diagnostics, "error", "METHOD_COVERAGE_CONCLUSION_LINK", eloc, "result and conclusion are not connected by an explicit support/aggregation edge", "Add a supports or aggregates edge from result_id to conclusion_id.")
            elif result_id is not None and (
                any(
                    isinstance(edge_item, dict)
                    and edge_item.get("source") == result_id
                    and edge_item.get("relation") in {"supports", "aggregates"}
                    for edge_item in doc.get("edges", [])
                )
                or (
                    not current_schema
                    and any(
                        isinstance(candidate, dict)
                        and candidate.get("parent_id") == result_id
                        and isinstance(candidate.get("node_type"), dict)
                        and candidate["node_type"].get("research_stage") == "conclusion"
                        for candidate in nodes.values()
                    )
                )
            ):
                _diag(diagnostics, "error", "METHOD_COVERAGE_CONCLUSION_OMITTED", eloc, "a result with a stage conclusion must record conclusion_id", "Include the supported conclusion in the Method coverage entry.")
    missing = sorted(methods - seen_methods)
    extra = sorted(seen_methods - methods)
    if missing:
        _diag(diagnostics, "error", "METHOD_COVERAGE_MISSING", "$.method_coverage", f"missing Method coverage records: {missing}", "Add one coverage record for every Method, including explicit no_followup records.")
    if extra:
        _diag(diagnostics, "error", "METHOD_COVERAGE_EXTRA", "$.method_coverage", f"unknown Method coverage records: {extra}", "Remove coverage records whose method_id is not a Method node.")


def _check_manifest(doc: dict[str, Any], paper_dir: Path, diagnostics: list[dict[str, Any]]) -> None:
    manifest = doc.get("source_manifest")
    if not isinstance(manifest, dict):
        _diag(diagnostics, "error", "MANIFEST_TYPE", "$.source_manifest", "source_manifest must be an object", "Inventory all input files.")
        return
    md_paths, img_paths, texts = _read_sources(paper_dir)
    global _source_text_cache
    _source_text_cache = texts
    expected_md = sorted(p.relative_to(paper_dir).as_posix() for p in md_paths)
    expected_img = sorted(p.relative_to(paper_dir).as_posix() for p in img_paths)
    for key, expected in (("markdown_files", expected_md), ("image_files", expected_img)):
        if not _uniq_strings(manifest.get(key)):
            _diag(diagnostics, "error", "MANIFEST_LIST", f"$.source_manifest.{key}", "must be a unique string list", "List every actual input file.")
        elif sorted(manifest[key]) != expected:
            _diag(diagnostics, "error", "MANIFEST_MISMATCH", f"$.source_manifest.{key}", f"manifest differs from directory; expected {expected}, got {sorted(manifest[key])}", "Regenerate the file inventory without dropping images.")
    for key in ("other_files",):
        if key in manifest and not _uniq_strings(manifest[key]) and manifest[key] != []:
            _diag(diagnostics, "error", "MANIFEST_OTHER", f"$.source_manifest.{key}", "other_files must be a unique string list", "Use [] or relative paths.")
    if manifest.get("paper_completeness") not in {"fulltext", "fulltext_without_supplement", "abstract_only", "figure_table_only", "metadata_only"}:
        _diag(diagnostics, "error", "MANIFEST_COMPLETENESS", "$.source_manifest.paper_completeness", "invalid paper_completeness", "Declare how much source material was available.")
    for hash_key, paths in (("markdown_sha256", expected_md), ("image_sha256", expected_img)):
        hashes = manifest.get(hash_key)
        if not isinstance(hashes, dict):
            _diag(diagnostics, "error", "MANIFEST_HASHES", f"$.source_manifest.{hash_key}", "hash map is required", "Record SHA-256 for every listed file.")
            continue
        for rel in paths:
            if not _is_nonempty_string(hashes.get(rel)):
                _diag(diagnostics, "error", "MANIFEST_HASH_MISSING", f"$.source_manifest.{hash_key}", f"missing hash for {rel}", "Compute SHA-256 for all source files.")
            else:
                actual = _sha256(paper_dir / rel)
                if hashes[rel] != actual:
                    _diag(diagnostics, "error", "MANIFEST_HASH_MISMATCH", f"$.source_manifest.{hash_key}.{rel}", "source hash does not match", "Regenerate the manifest after reading the actual input file.")


def validate(doc: Any, paper_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    if not isinstance(doc, dict):
        _diag(diagnostics, "error", "ROOT_TYPE", "$", "graph JSON must be an object", "Write the v3.2 top-level object.")
        return diagnostics, {"nodes": 0, "evidence": 0, "edges": 0}
    required_top = {"schema_version", "paper", "source_manifest", "evidence", "method_relations", "nodes", "edges"}
    schema_version = doc.get("schema_version")
    is_current_schema = schema_version == SCHEMA_VERSION
    is_legacy_schema = schema_version in LEGACY_SCHEMA_VERSIONS
    expected_top = required_top | ({"method_coverage"} if is_current_schema else set())
    allowed_top = required_top | {"method_coverage"}
    missing = expected_top - set(doc)
    extra = set(doc) - allowed_top
    for key in sorted(missing):
        _diag(diagnostics, "error", "TOP_LEVEL_MISSING", "$", f"missing top-level key: {key}", "Use the fixed v3.2 top-level contract, including method_coverage.")
    if extra:
        _diag(diagnostics, "error", "TOP_LEVEL_EXTRA", "$", f"unexpected top-level keys: {sorted(extra)}", "Remove undeclared top-level fields.")
    if not (is_current_schema or is_legacy_schema):
        _diag(diagnostics, "error", "SCHEMA_VERSION", "$.schema_version", f"expected {SCHEMA_VERSION} (legacy {sorted(LEGACY_SCHEMA_VERSIONS)} are accepted for migration)", "Set schema_version to the current v3.2 contract.")
    elif is_legacy_schema:
        _diag(diagnostics, "warning", "LEGACY_EXPLICIT_GRAPH", "$.schema_version", "legacy schema does not enforce multi-parent explicit relation closure", "Regenerate the graph with the v3.2 prompt and explicit graph edges.")
        if "method_coverage" not in doc:
            _diag(diagnostics, "warning", "LEGACY_METHOD_COVERAGE", "$.method_coverage", "legacy graph has no Method coverage matrix", "Regenerate the graph with the v3.1/v3.2 prompt and add explicit Method-to-experiment coverage.")
    paper = doc.get("paper")
    if not isinstance(paper, dict):
        _diag(diagnostics, "error", "PAPER_TYPE", "$.paper", "paper must be an object", "Fill paper metadata.")
        return diagnostics, {"nodes": 0, "evidence": 0, "edges": 0}
    for key in ("paper_id", "title", "source_dir", "source_language", "root_motivation_id", "overall_conclusion_id"):
        if not _is_nonempty_string(paper.get(key)):
            _diag(diagnostics, "error", "PAPER_FIELD", f"$.paper.{key}", "required non-empty string missing", "Fill paper metadata and root/terminal IDs.")
    if _is_nonempty_string(paper.get("source_dir")) and Path(paper["source_dir"]).resolve() != paper_dir.resolve():
        _diag(diagnostics, "error", "PAPER_SOURCE_DIR", "$.paper.source_dir", "source_dir does not match --paper-dir", "Use the exact input directory path.")
    evidence = _index(doc.get("evidence"), "evidence", diagnostics)
    nodes = _index(doc.get("nodes"), "nodes", diagnostics)
    _check_manifest(doc, paper_dir, diagnostics)
    _check_evidence(doc, paper_dir, evidence, diagnostics)
    _check_nodes(doc, paper_dir, nodes, evidence, diagnostics, current_schema=is_current_schema)
    _check_graph(doc, nodes, diagnostics, current_schema=is_current_schema)
    _check_method_relations(doc, nodes, set(evidence), diagnostics, current_schema=is_current_schema)
    _check_method_coverage(doc, nodes, set(evidence), diagnostics, current_schema=is_current_schema)
    # Every evidence reference is used at least once; unreferenced evidence is suspicious.
    used_evidence: set[str] = set()
    for node in nodes.values():
        used_evidence.update(node.get("evidence_refs", []) if isinstance(node.get("evidence_refs"), list) else [])
    for edge in doc.get("edges", []) if isinstance(doc.get("edges"), list) else []:
        used_evidence.update(edge.get("evidence_refs", []) if isinstance(edge.get("evidence_refs"), list) else [])
    for rel in doc.get("method_relations", []) if isinstance(doc.get("method_relations"), list) else []:
        used_evidence.update(rel.get("evidence_refs", []) if isinstance(rel.get("evidence_refs"), list) else [])
    for record in doc.get("method_coverage", []) if isinstance(doc.get("method_coverage"), list) else []:
        if isinstance(record, dict):
            used_evidence.update(record.get("uncovered_evidence_refs", []) if isinstance(record.get("uncovered_evidence_refs"), list) else [])
        for entry in record.get("entries", []) if isinstance(record, dict) and isinstance(record.get("entries"), list) else []:
            used_evidence.update(entry.get("evidence_refs", []) if isinstance(entry, dict) and isinstance(entry.get("evidence_refs"), list) else [])
    for eid in set(evidence) - used_evidence:
        _diag(diagnostics, "warning", "UNUSED_EVIDENCE", f"$.evidence[{eid}]", "evidence is never referenced", "Attach it to a node/edge or remove it.")
    # A warning for image files not represented by figure evidence; logos are allowed but need review.
    img_files = set(doc.get("source_manifest", {}).get("image_files", []) if isinstance(doc.get("source_manifest"), dict) else [])
    represented = {ev.get("asset_file") for ev in evidence.values() if ev.get("kind") == "figure" and _is_nonempty_string(ev.get("asset_file"))}
    for rel in sorted(img_files - represented):
        _diag(diagnostics, "warning", "IMAGE_NOT_MAPPED", "$.source_manifest.image_files", f"image is not mapped to figure evidence: {rel}", "Confirm it is decorative; otherwise create figure evidence and expand it.")
    summary = {
        "nodes": len(nodes),
        "evidence": len(evidence),
        "edges": len(doc.get("edges", []) if isinstance(doc.get("edges"), list) else []),
        "methods": sum(1 for n in nodes.values() if isinstance(n.get("node_type"), dict) and n["node_type"].get("research_stage") == "method"),
        "method_coverage": len(doc.get("method_coverage", []) if isinstance(doc.get("method_coverage"), list) else []),
        "evaluable_nodes": sum(1 for n in nodes.values() if isinstance(n.get("node_type"), dict) and n["node_type"].get("evaluability") == "evaluable"),
        "not_evaluable_nodes": sum(1 for n in nodes.values() if isinstance(n.get("node_type"), dict) and n["node_type"].get("evaluability") == "not_evaluable"),
    }
    return diagnostics, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ScionBench single-paper graph")
    parser.add_argument("--paper-dir", required=True, type=Path)
    parser.add_argument("--graph", dest="graph_path", type=Path, help="path to paper_graph.json")
    parser.add_argument("--tree", dest="legacy_graph_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.graph_path is not None and args.legacy_graph_path is not None:
        parser.error("use only one of --graph or the legacy --tree alias")
    graph_path = args.graph_path or args.legacy_graph_path
    if graph_path is None:
        parser.error("the --graph argument is required")
    diagnostics: list[dict[str, Any]] = []
    if not args.paper_dir.is_dir():
        _diag(diagnostics, "error", "PAPER_DIR_MISSING", str(args.paper_dir), "paper directory does not exist", "Pass an existing input directory.")
        doc: Any = None
        summary = {"nodes": 0, "evidence": 0, "edges": 0}
    elif not graph_path.is_file():
        _diag(diagnostics, "error", "GRAPH_MISSING", str(graph_path), "graph JSON does not exist", "Write paper_graph.json before validating.")
        doc = None
        summary = {"nodes": 0, "evidence": 0, "edges": 0}
    else:
        try:
            doc = json.loads(graph_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _diag(diagnostics, "error", "GRAPH_READ", str(graph_path), f"cannot parse graph JSON: {exc}", "Repair JSON syntax and rerun.")
            doc = None
            summary = {"nodes": 0, "evidence": 0, "edges": 0}
        else:
            diagnostics, summary = validate(doc, args.paper_dir)
    errors = [d for d in diagnostics if d["severity"] == "error"]
    warnings = [d for d in diagnostics if d["severity"] == "warning"]
    report = {"status": "VALID" if not errors else "INVALID", "summary": {**summary, "errors": len(errors), "warnings": len(warnings)}, "diagnostics": diagnostics}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("VALID" if not errors else "INVALID")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
