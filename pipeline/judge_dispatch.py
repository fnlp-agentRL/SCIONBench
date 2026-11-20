#!/usr/bin/env python3
"""Run node-level LLM-as-judge calls with mode and article scheduling.

For one article, the four report modes run in parallel.  Gold stages inside
one mode follow ``MODE_GOLD_STAGES`` and Markdown nodes inside each stage are
evaluated in lexical order, one at a time.  The script deliberately delegates
the HTTP/API details to the repository's existing
``merge_markdown_and_call_luna.py`` implementation so that retries and API
response parsing remain in one place.  ``judge-input/<number>`` is the shared
motivation-split gold tree; each report mode selects its relevant stage
directories, and each gold node selects a prompt based on its own stage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .figure_assets import collect_markdown_image_paths
except ImportError:  # Direct execution: pipeline/ is on sys.path.
    from figure_assets import collect_markdown_image_paths


MODES = ("conclusion", "experiment", "method", "motivation")
# The motivation split's held-out graph contains all target stages.  Keep the
# requested stage order so one report is compared with every applicable gold
# node deterministically.
MODE_GOLD_STAGES = {
    "conclusion": ("conclusion",),
    "experiment": ("conclusion", "experiment"),
    "method": ("conclusion", "experiment", "design"),
    "motivation": ("conclusion", "experiment", "design", "method"),
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCIONBENCH_ROOT = PROJECT_ROOT / "pipeline"
DEFAULT_JUDGE_SCRIPT = SCIONBENCH_ROOT / "judge" / "merge_markdown_and_call_luna.py"
DEFAULT_PROMPT_ROOT = SCIONBENCH_ROOT / "judge" / "judgeagent-inputprompt"
# Gold-node stages map to the capability being evaluated.  A method node
# evaluates the motivation-mode capability, while a design node evaluates the
# method-mode capability; report mode is used only for selecting the node set.
STAGE_PROMPT_RELATIVE_PATH: dict[str, Path] = {
    "method": Path("unified") / "motivation.md",
    "design": Path("unified") / "method.md",
    "experiment": Path("unified") / "experiment.md",
    "conclusion": Path("unified") / "conclusion.md",
}
# MaaS Kimi K3 judge defaults. The base URL may be supplied either as the
# ``/v1`` root or as the full ``/v1/chat/completions`` endpoint.
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "")
DEFAULT_BASE_URL = os.environ.get("JUDGE_API_ENDPOINT", "")
# 第六、八步的唯一手动密钥填写入口。只填写 API key 本体，不要包含
# ``Bearer ``。也可以保持为空，并在运行流水线前导出同名环境变量
# ``AIGW_API_KEY``。密钥不会被拼进命令行；dispatcher 仅通过子进程环境传递，
# HTTP 层再生成 Authorization 请求头。
AIGW_API_KEY = os.environ.get("AIGW_API_KEY", "")
THINKING = False
TEMPERATURE = 1.0
STREAM = False
# The Kimi request follows the supplied curl shape.  Optional legacy gateway
# fields remain disabled unless explicitly reintroduced here.
TOP_P: float | None = None
MAX_TOKENS: int | None = None
DEFAULT_NODE_RETRIES = 10
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_IMAGES_PER_REQUEST = 24
SCORE_LINE_PATTERN = re.compile(
    r"^[ \t]*score[ \t]*=[ \t]*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[ \t]*$",
    re.MULTILINE,
)


def judge_article_root(
    output_root: Path, format_name: str, article_number: str
) -> Path:
    """Return the article-level directory under a dispatcher output root.

    ``--output-root`` is the owner/team-level judge directory.  The dispatcher
    owns the remaining ``<format>/<article-number>`` path components, which is
    easy to lose when constructing paths outside this module.
    """

    if format_name != "md":
        raise ValueError(f"unsupported judge format: {format_name!r}")
    return output_root / format_name / safe_component(str(article_number))


def judge_mode_root(
    output_root: Path, format_name: str, article_number: str, mode: str
) -> Path:
    """Return the output directory for one format/article/mode combination."""

    if mode not in MODES:
        raise ValueError(f"unsupported judge mode: {mode!r}")
    return judge_article_root(output_root, format_name, article_number) / mode


def mode_average_path(
    output_root: Path, format_name: str, article_number: str, mode: str
) -> Path:
    """Return the real ``avg.json`` path written after one judge phase.

    For example, with ``output_root=OUTPUT/sci-judgeoutput/ying-tian-lei`` the
    result is ``.../json/01/method/avg.json``.  Keeping this construction in a
    helper makes the path used for writing, logging, and callers identical.
    """

    return judge_mode_root(output_root, format_name, article_number, mode) / "avg.json"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "node"


def score_from_markdown(path: Path) -> float:
    """Read the last numeric ``score = ...`` line from one judge result.

    Judge models sometimes repeat a final answer while reasoning or include a
    placeholder such as ``score = <final>`` before the actual numeric answer.
    Only numeric lines match the pattern; when several do, the last one is the
    final score emitted by the model.
    """

    content = path.read_text(encoding="utf-8")
    matches = SCORE_LINE_PATTERN.findall(content)
    if not matches:
        raise ValueError(
            f"no valid 'score = <number>' line in {path}"
        )
    score = float(matches[-1])
    if not 0.0 <= score <= 10.0:
        raise ValueError(
            f"last score in {path} is outside the allowed range 0-10: {score}"
        )
    return score


def _write_average_payload(average_path: Path, payload: dict[str, Any]) -> None:
    """Write an average payload atomically so readers never see a partial JSON."""

    average_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = average_path.with_name(
        f".{average_path.name}.tmp.{os.getpid()}"
    )
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(average_path)


def write_mode_average(mode_root: Path) -> dict[str, Any] | None:
    """Average node-result Markdown scores in one mode directory.

    A mode with some malformed result files still receives a ``partial``
    average over valid scores, together with the failed node names.  A mode
    with no valid score receives an explicit ``unavailable`` payload; it is
    never represented as a zero score.  A missing mode directory is treated as
    absent and does not create a file.
    """

    average_path = mode_root / "avg.json"
    if not mode_root.is_dir():
        return None

    result_paths = sorted(path for path in mode_root.glob("*.md") if path.is_file())
    scores: list[float] = []
    failed_nodes: list[dict[str, str]] = []
    for path in result_paths:
        try:
            scores.append(score_from_markdown(path))
        except Exception as exc:  # noqa: BLE001 - isolate one bad node.
            failed_nodes.append({"node_name": path.name, "reason": str(exc)})

    if not scores:
        payload: dict[str, Any] = {
            "status": "unavailable",
            "avg_score": None,
            "node_count": 0,
            "result_count": len(result_paths),
            "reason": "no valid score was found in the mode results",
        }
        if failed_nodes:
            payload["failed_nodes"] = failed_nodes
        _write_average_payload(average_path, payload)
        return payload

    payload: dict[str, float | int] = {
        "avg_score": sum(scores) / len(scores),
        "node_count": len(scores),
    }
    if failed_nodes:
        payload = {
            **payload,
            "status": "partial",
            "result_count": len(result_paths),
            "failed_node_count": len(failed_nodes),
            "failed_nodes": failed_nodes,
        }
    _write_average_payload(average_path, payload)
    return payload


@dataclass(frozen=True)
class ModeJob:
    mode: str
    gold_root: Path
    agent_report: Path
    output_root: Path
    merged_root: Path
    prompt_root: Path
    judge_script: Path
    model: str
    api_mode: str
    base_url: str | None
    api_key_env: str
    timeout: int
    retries: int
    api_key: str
    resume: bool
    stream: bool = STREAM
    temperature: float = TEMPERATURE
    max_request_bytes: int = MAX_REQUEST_BYTES
    max_images: int = MAX_IMAGES_PER_REQUEST
    failed_tasks_file: Path | None = None


def is_nonempty_file(path: Path) -> bool:
    """Return whether path is a regular file containing at least one byte."""

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def is_timeout_report(path: Path) -> bool:
    """Return whether a report is the synthetic 18-hour timeout marker."""

    try:
        return "运行时间超过18小时，强制终止任务" in path.read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False


def _file_signature(path: Path) -> tuple[int, int, int] | None:
    """Capture enough metadata to reject a stale result from an earlier call."""

    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file() or stat.st_size <= 0:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _read_failure_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _write_failure_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_failure(path: Path, entry: dict[str, Any]) -> None:
    """Upsert a failure entry without losing earlier resumed-run failures."""

    entries = _read_failure_entries(path)
    key = (
        entry.get("format"),
        entry.get("article_number"),
        entry.get("mode"),
        entry.get("node_name"),
    )
    entries = [
        existing
        for existing in entries
        if (
            existing.get("format"),
            existing.get("article_number"),
            existing.get("mode"),
            existing.get("node_name"),
        )
        != key
    ]
    entries.append(entry)
    _write_failure_entries(path, entries)


def _is_stale_average_failure(entry: dict[str, Any]) -> bool:
    """Identify dispatcher records emitted by an earlier average attempt."""

    return entry.get("mode") == "dispatcher" and " average:" in str(
        entry.get("reason", "")
    )


def clear_stale_average_failures(path: Path) -> None:
    """Remove obsolete mode-level average errors before recalculating."""

    entries = _read_failure_entries(path)
    remaining = [entry for entry in entries if not _is_stale_average_failure(entry)]
    if remaining != entries:
        _write_failure_entries(path, remaining)


def clear_failure(path: Path, entry: dict[str, Any]) -> None:
    """Remove a previously recorded failure after a later successful resume."""

    entries = _read_failure_entries(path)
    key = (
        entry.get("format"),
        entry.get("article_number"),
        entry.get("mode"),
        entry.get("node_name"),
    )
    remaining = [
        existing
        for existing in entries
        if (
            existing.get("format"),
            existing.get("article_number"),
            existing.get("mode"),
            existing.get("node_name"),
        )
        != key
    ]
    if remaining != entries:
        _write_failure_entries(path, remaining)


def _dedupe_failure_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest record for each format/article/mode/node key."""

    deduped: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for entry in entries:
        key = (
            entry.get("format"),
            entry.get("article_number", entry.get("number")),
            entry.get("mode"),
            entry.get("node_name"),
        )
        deduped[key] = entry
    return list(deduped.values())


def _failure_entry(
    job: ModeJob,
    *,
    node_name: str | None,
    reason: str,
    attempts: int,
) -> dict[str, Any]:
    return {
        "updated_at": timestamp(),
        "format": job.output_root.parents[1].name,
        "article_number": job.output_root.parent.name,
        "mode": job.mode,
        "node_name": node_name,
        "status": "failed",
        "reason": reason,
        "attempts": attempts,
    }


def _test_task_failed(job: ModeJob) -> dict[str, Any] | None:
    if job.failed_tasks_file is None:
        return None
    format_name = job.output_root.parents[1].name
    article_number = job.output_root.parent.name
    for entry in _read_failure_entries(job.failed_tasks_file):
        if (
            entry.get("format") == format_name
            and entry.get("mode") == job.mode
            and str(entry.get("number")) == article_number
            and entry.get("status", "failed") == "failed"
        ):
            return entry
    return None


def _node_failure(path: Path, job: ModeJob, node_name: str) -> dict[str, Any] | None:
    """Return a prior failed evaluation record for one node, if present."""

    format_name = job.output_root.parents[1].name
    article_number = job.output_root.parent.name
    for entry in _read_failure_entries(path):
        if (
            entry.get("format") == format_name
            and str(entry.get("article_number", entry.get("number"))) == article_number
            and entry.get("mode") == job.mode
            and entry.get("node_name") == node_name
            and entry.get("status", "failed") == "failed"
        ):
            return entry
    return None


def markdown_nodes(root: Path, stages: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for stage in stages:
        stage_root = root / stage
        if stage_root.is_dir():
            stage_paths = sorted(
                (path for path in stage_root.rglob("*.md") if path.is_file()),
                key=lambda path: path.relative_to(stage_root).as_posix(),
            )
            paths.extend(stage_paths)
    return paths


def gold_node_stage(node_path: Path, gold_root: Path) -> str:
    """Return the stage directory containing a standard-answer node."""

    try:
        relative = node_path.relative_to(gold_root)
    except ValueError as exc:
        raise RuntimeError(
            f"gold node {node_path} is outside its gold root {gold_root}"
        ) from exc
    if not relative.parts or relative.parts[0] not in STAGE_PROMPT_RELATIVE_PATH:
        raise RuntimeError(
            f"gold node {node_path} is not under a supported stage directory; "
            f"expected one of {tuple(STAGE_PROMPT_RELATIVE_PATH)}"
        )
    return relative.parts[0]


def prompt_for_gold_node(
    node_path: Path, gold_root: Path, prompt_root: Path
) -> Path:
    """Select the unified prompt from the gold node's own stage."""

    stage = gold_node_stage(node_path, gold_root)
    return prompt_root / STAGE_PROMPT_RELATIVE_PATH[stage]


def ensure_unique_node_filenames(nodes: list[Path]) -> None:
    """Reject ambiguous output names instead of overwriting a node result."""

    by_name: dict[str, list[Path]] = {}
    for node in nodes:
        by_name.setdefault(node.name, []).append(node)
    duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if not duplicates:
        return

    details = "; ".join(
        f"{name}: {', '.join(str(path) for path in paths)}"
        for name, paths in sorted(duplicates.items())
    )
    raise RuntimeError(
        "gold nodes contain duplicate filenames, so node-named judge outputs "
        f"would collide: {details}"
    )


def run_mode(job: ModeJob) -> int:
    job.output_root.mkdir(parents=True, exist_ok=True)
    job.merged_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    failure_path = job.output_root / "failed_evaluations.json"

    failed_test = _test_task_failed(job)
    if failed_test is not None:
        entry = _failure_entry(
            job,
            node_name=None,
            reason=(
                "corresponding test task failed; evaluation skipped: "
                f"{failed_test.get('reason', 'unknown reason')}"
            ),
            attempts=0,
        )
        record_failure(failure_path, entry)
        manifest.append({**entry, "status": "skipped_test_failure"})
        (job.output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{timestamp()}] mode={job.mode}: test task failed; evaluation skipped; continuing",
            flush=True,
        )
        return 0

    # Protect against old runs where the test failure manifest was not written
    # but the supervisor did leave its synthetic timeout report behind.
    if is_timeout_report(job.agent_report):
        entry = _failure_entry(
            job,
            node_name=None,
            reason=(
                "agent report is an 18-hour timeout report; evaluation skipped"
            ),
            attempts=0,
        )
        record_failure(failure_path, entry)
        manifest.append({**entry, "status": "skipped_test_timeout"})
        (job.output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{timestamp()}] mode={job.mode}: timeout report detected; "
            "evaluation skipped; continuing",
            flush=True,
        )
        return 0

    if not is_nonempty_file(job.agent_report):
        entry = _failure_entry(
            job,
            node_name=None,
            reason=f"agent report is missing or empty: {job.agent_report}",
            attempts=0,
        )
        record_failure(failure_path, entry)
        manifest.append({**entry, "status": "failed"})
        (job.output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{timestamp()}] mode={job.mode}: missing report recorded; continuing",
            flush=True,
        )
        return 0

    try:
        if not job.gold_root.is_dir():
            raise RuntimeError(f"gold node root does not exist: {job.gold_root}")
        nodes = markdown_nodes(job.gold_root, MODE_GOLD_STAGES[job.mode])
        ensure_unique_node_filenames(nodes)
    except Exception as exc:  # noqa: BLE001 - isolate one mode from other modes.
        entry = _failure_entry(
            job,
            node_name=None,
            reason=str(exc),
            attempts=0,
        )
        record_failure(failure_path, entry)
        manifest.append({**entry, "status": "failed"})
        (job.output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{timestamp()}] mode={job.mode}: setup failure recorded; continuing: {exc}",
            flush=True,
        )
        return 0

    if not nodes:
        entry = _failure_entry(
            job,
            node_name=None,
            reason=(
                "no standard-answer Markdown nodes found for the requested "
                f"stages {MODE_GOLD_STAGES[job.mode]} under {job.gold_root}"
            ),
            attempts=0,
        )
        record_failure(failure_path, entry)
        manifest.append({**entry, "status": "failed"})
        (job.output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{timestamp()}] mode={job.mode}: no gold nodes recorded; continuing",
            file=sys.stderr,
        )
        return 0

    # A previous run may have failed before it reached the node loop (for
    # example, a missing report or an empty gold tree).  Once setup is valid,
    # remove that mode-level marker while retaining any node-specific failures.
    clear_failure(
        failure_path,
        _failure_entry(
            job,
            node_name=None,
            reason="setup recovered",
            attempts=0,
        ),
    )

    for ordinal, node_path in enumerate(nodes, start=1):
        image_paths: list[Path] = []
        result_path = job.output_root / node_path.name
        merged_path = job.merged_root / node_path.name
        manifest_entry = {
            "ordinal": f"{ordinal:02d}",
            "node_name": node_path.name,
            "gold_node": str(node_path),
            "agent_report": str(job.agent_report),
            "merged_prompt": str(merged_path),
            "result": str(result_path),
            "images": [str(path) for path in image_paths],
        }
        previous_failure = _node_failure(failure_path, job, node_path.name)
        if job.resume and is_nonempty_file(result_path) and previous_failure is None:
            clear_failure(
                failure_path,
                _failure_entry(
                    job,
                    node_name=node_path.name,
                    reason="resume found a non-empty result",
                    attempts=0,
                ),
            )
            manifest.append(manifest_entry)
            print(
                f"[{timestamp()}] mode={job.mode} node={ordinal:02d}/{len(nodes):02d} "
                f"name={node_path.name} resume skip -> {result_path}",
                flush=True,
            )
            continue
        if job.resume and previous_failure is not None:
            print(
                f"[{timestamp()}] mode={job.mode} node={ordinal:02d}/{len(nodes):02d} "
                f"previous failure found; retrying: {previous_failure.get('reason', '')}",
                flush=True,
            )

        try:
            image_paths = collect_markdown_image_paths(node_path)
            prompt_file = prompt_for_gold_node(
                node_path, job.gold_root, job.prompt_root
            )
            if not prompt_file.is_file():
                raise RuntimeError(
                    f"missing judge prompt for gold stage "
                    f"{gold_node_stage(node_path, job.gold_root)!r}: {prompt_file}"
                )
            manifest_entry["images"] = [str(path) for path in image_paths]
            command = [
                sys.executable,
                str(job.judge_script),
                "--golden-nodes",
                str(node_path),
                "--agent-output-md",
                str(job.agent_report),
                "--prompt-md",
                str(prompt_file),
                "--merged-md-output",
                str(merged_path),
                "--model-output-md",
                str(result_path),
                "--model",
                job.model,
                "--api-mode",
                job.api_mode,
                "--api-key-env",
                job.api_key_env,
                "--timeout",
                str(job.timeout),
                "--max-retries",
                str(job.retries),
                "--max-request-bytes",
                str(job.max_request_bytes),
                "--max-images",
                str(job.max_images),
                "--thinking",
                "true" if THINKING else "false",
                "--temperature",
                str(job.temperature),
                "--stream",
                "true" if job.stream else "false",
            ]
            if TOP_P is not None:
                command.extend(("--top-p", str(TOP_P)))
            if MAX_TOKENS is not None:
                command.extend(("--max-tokens", str(MAX_TOKENS)))
            if job.base_url:
                command.extend(("--base-url", job.base_url))
            for image_path in image_paths:
                command.extend(("--image-path", str(image_path)))

            # A non-resume attempt must not be allowed to inherit a previous
            # non-empty result when the child exits without writing anything.
            # Remove only this generated node result; ``--resume`` has already
            # returned above for a valid existing result.
            if result_path.exists():
                result_path.unlink()
            result_signature_before = _file_signature(result_path)
            print(
                f"[{timestamp()}] mode={job.mode} node={ordinal:02d}/{len(nodes):02d} "
                f"name={node_path.name} images={len(image_paths)} start gold={node_path}",
                flush=True,
            )
            child_env = os.environ.copy()
            if job.api_key:
                child_env[job.api_key_env] = job.api_key
            completed = subprocess.run(command, check=False, env=child_env)
            result_signature_after = _file_signature(result_path)
            if completed.returncode != 0 or result_signature_after is None:
                raise RuntimeError(
                    f"judge call failed for mode={job.mode}, node={ordinal:02d}; "
                    f"returncode={completed.returncode} after {job.retries} retries"
                )
            if (
                result_signature_before is not None
                and result_signature_after == result_signature_before
            ):
                raise RuntimeError(
                    f"judge call produced no new result for mode={job.mode}, "
                    f"node={ordinal:02d}; stale output was rejected"
                )
            clear_failure(
                failure_path,
                _failure_entry(
                    job,
                    node_name=node_path.name,
                    reason="resolved",
                    attempts=0,
                ),
            )
            manifest.append(manifest_entry)
            print(
                f"[{timestamp()}] mode={job.mode} node={ordinal:02d} "
                f"name={node_path.name} done -> {result_path}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - one node must not stop its mode.
            failure = _failure_entry(
                job,
                node_name=node_path.name,
                reason=str(exc),
                attempts=job.retries + 1,
            )
            record_failure(failure_path, failure)
            manifest.append({**manifest_entry, **failure, "status": "failed"})
            print(
                f"[{timestamp()}] mode={job.mode} node={ordinal:02d} "
                f"name={node_path.name} failed after {job.retries} retries; continuing: {exc}",
                file=sys.stderr,
                flush=True,
            )

    (job.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-input-root", type=Path, required=True,
                        help="judge-input root containing <number>/<stage>/...md")
    parser.add_argument("--agent-root", type=Path, required=True,
                        help="agent output root containing md/")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="judge output root; creates md/<number>/<mode>")
    parser.add_argument("--article-number", required=True,
                        help="number such as 01")
    parser.add_argument("--format", choices=("md",), required=True)
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=DEFAULT_PROMPT_ROOT,
        help="judge prompt root containing unified/<stage-specific>.md files",
    )
    parser.add_argument("--judge-script", type=Path, default=DEFAULT_JUDGE_SCRIPT)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"judge model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-mode", choices=("responses", "chat_completions"),
                        default="chat_completions")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible API root or full endpoint")
    parser.add_argument("--api-key-env", default="AIGW_API_KEY")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--stream",
        choices=("true", "false"),
        default="true" if STREAM else "false",
        help="request an SSE streaming chat-completion response",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_NODE_RETRIES,
        help=(
            "API retries per node after the initial request; a failed node is "
            f"recorded and skipped after {DEFAULT_NODE_RETRIES} retries by default"
        ),
    )
    parser.add_argument("--max-request-bytes", type=int, default=MAX_REQUEST_BYTES)
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES_PER_REQUEST)
    parser.add_argument(
        "--failed-tasks-file",
        type=Path,
        default=None,
        help="JSON list of failed test tasks whose evaluations should be skipped",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip each standard-answer node whose node-named result Markdown is non-empty",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_retries < 0:
        print("max retries cannot be negative", file=sys.stderr)
        return 2
    if args.max_request_bytes <= 0:
        print("max request bytes must be greater than zero", file=sys.stderr)
        return 2
    if args.max_images < 0:
        print("max images cannot be negative", file=sys.stderr)
        return 2
    article = safe_component(args.article_number)
    format_root = args.agent_root / args.format
    output_format_root = judge_article_root(args.output_root, args.format, article)
    merged_root = args.output_root / ".merged-prompts" / args.format / article

    jobs: list[ModeJob] = []
    for mode in MODES:
        shared_gold_root = args.judge_input_root / article
        mode_gold_root = args.judge_input_root / mode / article
        gold_root = shared_gold_root if shared_gold_root.is_dir() else mode_gold_root
        jobs.append(
            ModeJob(
                mode=mode,
                gold_root=gold_root,
                agent_report=format_root / mode / article / "research_report.md",
                output_root=output_format_root / mode,
                merged_root=merged_root / mode,
                prompt_root=args.prompt_root,
                judge_script=args.judge_script,
                model=args.model,
                api_mode=args.api_mode,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                timeout=args.timeout,
                retries=args.max_retries,
                api_key=AIGW_API_KEY,
                resume=args.resume,
                stream=args.stream == "true",
                temperature=args.temperature,
                max_request_bytes=args.max_request_bytes,
                max_images=args.max_images,
                failed_tasks_file=args.failed_tasks_file,
            )
        )

    # Validate shared resources per node inside ``run_mode``.  A missing prompt
    # or child script is recorded for the affected nodes instead of aborting
    # all four mode workers and later article numbers.
    for stage, relative_path in STAGE_PROMPT_RELATIVE_PATH.items():
        prompt_file = args.prompt_root / relative_path
        if not prompt_file.is_file():
            print(
                f"warning: missing judge prompt for gold stage {stage!r}: {prompt_file}",
                file=sys.stderr,
            )
    if not args.judge_script.is_file():
        print(
            f"warning: missing judge script; affected nodes will be recorded: "
            f"{args.judge_script}",
            file=sys.stderr,
        )

    failures: list[str] = []
    # Exactly four workers: modes are parallel; each worker's run_mode loop is
    # intentionally serial over that mode's standard-answer nodes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_mode, job): job.mode for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            mode = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - report all mode failures
                failures.append(f"{mode}: {exc}")

    for mode in MODES:
        mode_root = judge_mode_root(args.output_root, args.format, article, mode)
        average_path = mode_average_path(args.output_root, args.format, article, mode)
        clear_stale_average_failures(mode_root / "failed_evaluations.json")
        try:
            average = write_mode_average(mode_root)
        except Exception as exc:  # noqa: BLE001 - report one mode and continue.
            failure = f"{mode} average: {exc}"
            failures.append(failure)
            print(f"judge average failure recorded; continuing: {failure}", file=sys.stderr)
            continue
        if average is not None:
            print(
                f"[{timestamp()}] mode={mode} status={average.get('status', 'complete')} "
                f"average={average.get('avg_score')} nodes={average['node_count']} "
                f"-> {average_path}",
                flush=True,
            )
            # A malformed result is still a node-level failure even when a
            # partial average can be computed from the remaining scores.
            for failed_node in average.get("failed_nodes", []):
                if not isinstance(failed_node, dict):
                    continue
                node_name = failed_node.get("node_name")
                if not isinstance(node_name, str):
                    continue
                entry = {
                    "updated_at": timestamp(),
                    "format": args.format,
                    "article_number": article,
                    "mode": mode,
                    "node_name": node_name,
                    "status": "failed",
                    "reason": str(
                        failed_node.get("reason", "invalid score output")
                    ),
                    "attempts": 0,
                }
                record_failure(mode_root / "failed_evaluations.json", entry)
            if average.get("status") == "unavailable":
                # Keep an auditable mode-level reason when all expected node
                # results are absent (for example, an upstream test timeout).
                existing_entries = _read_failure_entries(
                    mode_root / "failed_evaluations.json"
                )
                has_mode_failure = any(
                    entry.get("mode") == mode
                    and entry.get("node_name") is None
                    and entry.get("status", "failed") == "failed"
                    for entry in existing_entries
                )
                if not has_mode_failure:
                    entry = {
                        "updated_at": timestamp(),
                        "format": args.format,
                        "article_number": article,
                        "mode": mode,
                        "node_name": None,
                        "status": "failed",
                        "reason": str(
                            average.get("reason", "no valid score was found")
                        ),
                        "attempts": 0,
                    }
                    record_failure(mode_root / "failed_evaluations.json", entry)

    all_failed_entries: list[dict[str, Any]] = []
    for mode in MODES:
        all_failed_entries.extend(
            _read_failure_entries(
                judge_mode_root(args.output_root, args.format, article, mode)
                / "failed_evaluations.json"
            )
        )
    if failures:
        for failure in failures:
            entry = {
                "updated_at": timestamp(),
                "format": args.format,
                "article_number": article,
                "mode": "dispatcher",
                "node_name": None,
                "status": "failed",
                "reason": failure,
                "attempts": 0,
            }
            all_failed_entries.append(entry)
            print(f"judge failure recorded; continuing: {failure}", file=sys.stderr)
    all_failed_entries = [
        entry for entry in all_failed_entries if not _is_stale_average_failure(entry)
    ]
    all_failed_entries = _dedupe_failure_entries(all_failed_entries)
    aggregate_failure_path = output_format_root / "failed_evaluations.json"
    if all_failed_entries or aggregate_failure_path.is_file():
        _write_failure_entries(aggregate_failure_path, all_failed_entries)
    if all_failed_entries:
        print(
            f"[{timestamp()}] article {article} ({args.format}) completed with "
            f"{len(all_failed_entries)} failed evaluation(s); continuing",
            flush=True,
        )

    print(f"[{timestamp()}] all modes completed for article {article} ({args.format})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
