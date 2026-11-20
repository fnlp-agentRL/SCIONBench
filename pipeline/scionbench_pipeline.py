#!/usr/bin/env python3
"""串联 ScionBench 文献处理、Codex 测试和 LLM-as-judge 评估。

默认配置区故意留空，使用前填写 ``INPUT_DIR`` 和三个输出目录，或在命令行
传入同名参数。一次普通运行处理一个文章目录；``--single-paper`` 可将输入
严格限制为一篇文献，并自动在三个输出根目录下追加文献目录名；``--batch``
可将输入根目录下的一级文章目录逐个串行处理。

目录约定（以一个文章目录为例）::

    preprocess-output/
      first/01.json
      test-input/md/<mode>/01/agent_input.md
      test-input/md/<mode>/01/data/...
      judge-input/01/<stage>/*.md
      judge-input/01/<mode>/evaluation.json
    agent-output/
      md/<mode>/01/{research_report.md,codex_token_usage.json,analysis.json}
    judge-output/
      md/01/<mode>/<node-name>.md
      md/01/<mode>/avg.json

    文章编号按 ``paper_graph.json`` 的稳定字典序排序。一个编号内的四种模式
    并行、模式内节点串行；编号之间完整串行完成。Codex 的每次重试由
    codex_supervisor.sh 建立全新的 CODEX_HOME。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from .figure_assets import FigureAssetError, organize_article_assets
    from .codex_token_usage import record_usage as record_codex_usage
    from .data_assets import DataAssetError, copy_source_data_directories
    from .evaluation_assets import EvaluationMoveError, move_evaluation_files
    from .run_analysis import write_analysis as write_run_analysis
except ImportError:  # Direct execution: pipeline/ is on sys.path.
    from figure_assets import FigureAssetError, organize_article_assets
    from codex_token_usage import record_usage as record_codex_usage
    from data_assets import DataAssetError, copy_source_data_directories
    from evaluation_assets import EvaluationMoveError, move_evaluation_files
    from run_analysis import write_analysis as write_run_analysis


# ============================ 可填写配置区 ============================
# 也可以完全通过命令行参数覆盖。这里保持空字符串，避免误处理其他目录。
INPUT_DIR = ""
PREPROCESS_OUTPUT_DIR = ""
AGENT_OUTPUT_DIR = ""
JUDGE_OUTPUT_DIR = ""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCIONBENCH_ROOT = PROJECT_ROOT
PIPELINE_ROOT = PROJECT_ROOT / "pipeline"
EXTRACTION_ROOT = PIPELINE_ROOT
JUDGE_ROOT = PIPELINE_ROOT / "judge"
PAPER_EXTRAC_ROOT = Path(os.environ.get("PIPELINE_PAPER_EXTRAC_ROOT", "."))
SOURCE_MD_ROOT = Path(os.environ.get("PIPELINE_SOURCE_MD_ROOT", "."))

MERGE_SAME_SCRIPT = EXTRACTION_ROOT / "merge_same_parent_child_nodes.py"
MERGE_BRANCH_SCRIPT = EXTRACTION_ROOT / "merge_branch_conclusions.py"
SPLIT_SCRIPTS = {
    "conclusion": JUDGE_ROOT / "conclusion-mode" / "split_conclusion.py",
    "experiment": JUDGE_ROOT / "experiment-mode" / "split_experiment.py",
    "method": JUDGE_ROOT / "method-mode" / "split_method.py",
    "motivation": JUDGE_ROOT / "motivation-mode" / "split_motivation.py",
}
AGENT_INPUT_TO_MD = JUDGE_ROOT / "agent_input_to_markdown.py"
EXTRACT_NODES_TO_MD = JUDGE_ROOT / "extract_nodes_to_md.py"
JUDGE_DISPATCH = PIPELINE_ROOT / "judge_dispatch.py"
SUPERVISOR = PIPELINE_ROOT / "codex_supervisor.sh"
TEST_PROMPT_ROOT = JUDGE_ROOT / "testagent-inputprompt"

MODES = ("conclusion", "experiment", "method", "motivation")
# This must match the model id configured by the active Codex provider.
MODEL = os.environ.get("PIPELINE_MODEL", "gpt-5.6-sol")
# 第六、八步的 MaaS Kimi K3 配置。密钥填写在同目录 judge_dispatch.py 中。
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")
JUDGE_API_MODE = "chat_completions"
JUDGE_API_ENDPOINT = os.environ.get("JUDGE_API_ENDPOINT", "")
JUDGE_API_KEY_ENV = "AIGW_API_KEY"
JUDGE_TEMPERATURE = 1.0
JUDGE_STREAM = False
TASK_TIMEOUT_HOURS = 18
TASK_TIMEOUT_SECONDS = TASK_TIMEOUT_HOURS * 60 * 60
MAX_RESTARTS_LIMIT = 10
DEFAULT_JUDGE_RETRIES = 10
JUDGE_MAX_REQUEST_BYTES = 2 * 1024 * 1024
JUDGE_MAX_IMAGES = 24


class PipelineError(RuntimeError):
    """Expected pipeline failure with an actionable message."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return result or "article"


def ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise PipelineError(f"{description} does not exist: {path}")


def is_nonempty_file(path: Path) -> bool:
    """Return whether path is a regular file containing at least one byte."""

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def is_timeout_report(path: Path) -> bool:
    """Identify the synthetic report written when the 18-hour cap fires."""

    try:
        return "运行时间超过18小时，强制终止任务" in path.read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False


def run_tool(label: str, command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print(f"[{now()}] {label}: {' '.join(str(part) for part in command)}", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise PipelineError(f"{label} failed with exit code {completed.returncode}")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_status(root: Path, state: str, message: str) -> None:
    payload = {
        "updated_at": now(),
        "state": state,
        "message": message,
    }
    path = root / "pipeline-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def failure_list_path(agent_dir: Path, format_name: str) -> Path:
    return agent_dir / f"failed_tasks_{format_name}.json"


def read_failure_entries(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def write_failure_entries(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def record_failure_entry(path: Path, entry: dict[str, object]) -> None:
    entries = read_failure_entries(path)
    key = (entry.get("format"), entry.get("mode"), entry.get("number"))
    entries = [
        existing
        for existing in entries
        if (existing.get("format"), existing.get("mode"), existing.get("number"))
        != key
    ]
    entries.append(entry)
    write_failure_entries(path, entries)


def record_evaluation_failure_entry(path: Path, entry: dict[str, object]) -> None:
    """Upsert a dispatcher-level evaluation failure using judge-list fields."""

    entries = read_failure_entries(path)
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
    write_failure_entries(path, entries)


def clear_failure_entry(path: Path, *, format_name: str, mode: str, number: str) -> None:
    entries = read_failure_entries(path)
    remaining = [
        existing
        for existing in entries
        if (
            existing.get("format"),
            existing.get("mode"),
            existing.get("number"),
        )
        != (format_name, mode, number)
    ]
    if remaining != entries:
        write_failure_entries(path, remaining)


def task_failure_entry(
    path: Path, *, format_name: str, mode: str, number: str
) -> dict[str, object] | None:
    """Return a prior failed test entry for one task, if present."""

    for entry in read_failure_entries(path):
        if (
            entry.get("format") == format_name
            and entry.get("mode") == mode
            and str(entry.get("number")) == str(number)
            and entry.get("status", "failed") == "failed"
        ):
            return entry
    return None


def discover_graphs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise PipelineError(f"input directory does not exist: {input_dir}")
    paths = sorted(path for path in input_dir.rglob("paper_graph.json") if path.is_file())
    if not paths:
        raise PipelineError(f"no paper_graph.json found below: {input_dir}")
    return paths


def article_dirs(input_root: Path) -> list[Path]:
    """Return first-level article directories that contain at least one graph."""

    if not input_root.is_dir():
        raise PipelineError(f"input root does not exist: {input_root}")
    result = []
    for child in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if any(path.is_file() for path in child.rglob("paper_graph.json")):
            result.append(child)
    if not result:
        raise PipelineError(f"no article directory containing paper_graph.json: {input_root}")
    return result


def single_paper_output_dirs(
    input_path: Path,
    preprocess_path: Path,
    agent_path: Path,
    judge_path: Path,
) -> tuple[Path, Path, Path]:
    """Validate a single-paper input and derive its three output directories.

    The regular and ``--batch`` entry points intentionally retain their
    historical output paths.  This helper is only used by ``--single-paper``
    so a team directory cannot accidentally mix multiple articles into one
    result tree.
    """

    if not input_path.is_dir():
        raise PipelineError(f"input directory does not exist: {input_path}")
    graph_paths = sorted(
        path for path in input_path.rglob("paper_graph.json") if path.is_file()
    )
    if len(graph_paths) != 1:
        raise PipelineError(
            "--single-paper requires exactly one paper_graph.json below the "
            f"input directory; found {len(graph_paths)}: {input_path}"
        )
    paper_name = safe_component(input_path.name)
    return (
        preprocess_path / paper_name,
        agent_path / paper_name,
        judge_path / paper_name,
    )


def replace_prompt(template: Path, replacements: dict[str, str], output: Path) -> None:
    ensure_file(template, "prompt template")
    content = template.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        if placeholder not in content:
            raise PipelineError(f"prompt {template} is missing {placeholder}")
        content = content.replace(placeholder, value)
    write_text(output, content)


def preprocess_graphs(input_dir: Path, output_dir: Path) -> list[str]:
    """Run both graph merge scripts once per source graph."""

    graph_paths = discover_graphs(input_dir)
    first_root = output_dir / "first"
    first_root.mkdir(parents=True, exist_ok=True)
    numbers: list[str] = []
    source_manifest: dict[str, str] = {}

    for ordinal, source in enumerate(graph_paths, start=1):
        number = f"{ordinal:02d}"
        numbers.append(number)
        source_manifest[number] = str(source.resolve())
        intermediate_dir = output_dir / ".work" / "merge" / number
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        same_output = intermediate_dir / "paper_graph_unit.json"
        final_output = first_root / f"{number}.json"

        run_tool(
            f"step1 merge-same {number}",
            [
                sys.executable,
                str(MERGE_SAME_SCRIPT),
                "--input-file",
                str(source),
                "--output-dir",
                str(intermediate_dir),
            ],
        )
        run_tool(
            f"step1 merge-branch {number}",
            [sys.executable, str(MERGE_BRANCH_SCRIPT), str(same_output), str(final_output)],
        )
    write_text(
        output_dir / "source-manifest.json",
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
    )

    return numbers


def split_inputs(output_dir: Path, numbers: Iterable[str]) -> None:
    """Run each mode splitter once for each numbered graph."""

    for number in numbers:
        graph = output_dir / "first" / f"{number}.json"
        ensure_file(graph, f"merged graph {number}")
        for mode in MODES:
            mode_output = output_dir / ".work" / "split-json" / mode / number
            run_tool(
                f"step2 split {mode} {number}",
                [
                    sys.executable,
                    str(SPLIT_SCRIPTS[mode]),
                    str(graph),
                    "--output-dir",
                    str(mode_output),
                ],
            )


def convert_inputs_to_markdown(output_dir: Path, numbers: Iterable[str]) -> None:
    for number in numbers:
        for mode in MODES:
            source = output_dir / ".work" / "split-json" / mode / number / "agent_input.json"
            target = output_dir / "test-input" / "md" / mode / number / "agent_input.md"
            ensure_file(source, f"agent input {mode}/{number}")
            run_tool(
                f"step3 json-to-md {mode} {number}",
                [
                    sys.executable,
                    str(AGENT_INPUT_TO_MD),
                    "--input-file",
                    str(source),
                    "--output-file",
                    str(target),
                ],
            )


def extract_gold_nodes(output_dir: Path, numbers: Iterable[str]) -> None:
    # The motivation split's evaluation.json contains the complete held-out
    # target graph (method/design/experiment/conclusion).  It is extracted once
    # into a shared judge-input/<number> tree; judge_dispatch selects the stage
    # subset appropriate to each mode.
    for number in numbers:
        source = output_dir / ".work" / "split-json" / "motivation" / number / "evaluation.json"
        target = output_dir / "judge-input" / number
        ensure_file(source, f"motivation evaluation input {number}")
        run_tool(
            f"step4 extract-gold motivation {number}",
            [
                sys.executable,
                str(EXTRACT_NODES_TO_MD),
                "--input-file",
                str(source),
                "--output-dir",
                str(target),
            ],
        )


@dataclass
class SupervisorJob:
    mode: str
    number: str
    format_name: str
    runner_dir: Path
    state_file: Path
    status_file: Path
    session_name: str
    report_path: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    launcher_log: object | None = None
    launcher: str = "direct"
    started_monotonic: float | None = None
    started_at: str | None = None
    # One run id is shared by all fresh Codex attempts made by this supervisor.
    # The aggregate JSON lives beside the final research report.
    usage_file: Path | None = None
    analysis_file: Path | None = None
    run_id: str = ""


def _usage_attempt_and_log(job: SupervisorJob) -> tuple[int, Path | None]:
    """Find the current Codex attempt and its JSONL log for fallback writes."""

    run_id_file = job.runner_dir / "run_id"
    if run_id_file.is_file() and job.run_id:
        try:
            if run_id_file.read_text(encoding="utf-8").strip() != job.run_id:
                return 0, None
        except OSError:
            return 0, None

    current_attempt = job.runner_dir / "current_attempt"
    if current_attempt.is_file():
        try:
            attempt = int(current_attempt.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0, None
        if attempt is not None:
            candidate = (
                job.runner_dir / "logs" / f"codex_attempt_{attempt}.log"
                if attempt > 0
                else None
            )
            return attempt, candidate

    # Older/in-flight supervisors may not have current_attempt yet.  The state
    # file contains the attempt number while Codex is running or restarting.
    try:
        state = job.state_file.read_text(encoding="utf-8").strip()
    except OSError:
        state = ""
    match = re.search(r"attempt:(\d+)", state)
    if match:
        attempt = int(match.group(1))
        return attempt, job.runner_dir / "logs" / f"codex_attempt_{attempt}.log"

    def attempt_number(path: Path) -> int:
        match = re.search(r"codex_attempt_(\d+)\.log$", path.name)
        return int(match.group(1)) if match else -1

    logs = sorted(job.runner_dir.glob("logs/codex_attempt_*.log"), key=attempt_number)
    if logs:
        latest = logs[-1]
        match = re.search(r"codex_attempt_(\d+)\.log$", latest.name)
        if match:
            return int(match.group(1)), latest
    return 0, None


def record_codex_usage_for_job(
    job: SupervisorJob,
    *,
    status: str,
    reason: str | None = None,
    exit_code: int | None = None,
    usage_complete: bool = False,
    usage_lower_bound: bool = False,
    attempt: int | None = None,
) -> None:
    """Best-effort usage fallback for supervisor exits and setup failures.

    The shell supervisor records normal attempt completion itself.  This helper
    is intentionally idempotent and fills the gap when the parent process has
    to kill the supervisor (for example at the 18-hour timeout).
    """

    if job.usage_file is None:
        return
    run_id = job.run_id or (
        f"legacy:{job.format_name}:{job.mode}:{job.number}:{job.runner_dir}"
    )
    if attempt is None:
        attempt, events_file = _usage_attempt_and_log(job)
    else:
        events_file = (
            job.runner_dir / "logs" / f"codex_attempt_{attempt}.log"
            if attempt > 0
            else None
        )
    try:
        record_codex_usage(
            job.usage_file,
            events_file=events_file,
            format_name=job.format_name,
            mode=job.mode,
            number=str(job.number),
            run_id=run_id,
            attempt=attempt,
            status=status,
            exit_code=exit_code,
            reason=reason,
            usage_complete=usage_complete,
            usage_lower_bound=usage_lower_bound,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry cannot stop later tasks.
        print(
            f"[{now()}] token usage record failed for "
            f"{job.format_name}/{job.mode}/{job.number}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def supervisor_failure_entry(job: SupervisorJob, reason: str) -> dict[str, object]:
    return {
        "updated_at": now(),
        "format": job.format_name,
        "mode": job.mode,
        "number": job.number,
        "status": "failed",
        "reason": reason,
        "report_path": str(job.report_path) if job.report_path else None,
    }


def supervisor_command(
    *,
    prompt: Path,
    input_path: Path,
    output_path: Path,
    runner_dir: Path,
    workdir: Path,
    health_interval: int,
    max_restarts: int,
    usage_file: Path | None = None,
    analysis_file: Path | None = None,
    format_name: str | None = None,
    mode: str | None = None,
    number: str | None = None,
    run_id: str | None = None,
) -> list[str]:
    command = [
        str(SUPERVISOR),
        "--prompt",
        str(prompt),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--runner",
        str(runner_dir),
        "--workdir",
        str(workdir),
        "--model",
        MODEL,
        "--health-interval",
        str(health_interval),
        "--max-restarts",
        str(max_restarts),
    ]
    if usage_file is not None:
        command.extend(["--usage-file", str(usage_file)])
        if format_name is not None:
            command.extend(["--format", format_name])
        if mode is not None:
            command.extend(["--mode", mode])
        if number is not None:
            command.extend(["--number", str(number)])
        if run_id:
            command.extend(["--run-id", run_id])
    if analysis_file is not None:
        command.extend(["--analysis-file", str(analysis_file)])
    return command


def launch_supervisor(
    *,
    job: SupervisorJob,
    command: list[str],
    launcher_preference: str,
) -> None:
    job.runner_dir.mkdir(parents=True, exist_ok=True)
    session = job.session_name
    selected = launcher_preference
    if selected == "auto":
        if shutil.which("tmux"):
            selected = "tmux"
        elif shutil.which("systemd-run"):
            selected = "systemd"
        else:
            selected = "direct"

    if selected == "tmux":
        if not shutil.which("tmux"):
            if launcher_preference == "tmux":
                raise PipelineError("launcher=tmux requested but tmux is not installed")
            selected = "direct"
        else:
            completed = subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, shlex.join(command)],
                check=False,
            )
            if completed.returncode == 0:
                job.launcher = "tmux"
                job.started_monotonic = time.monotonic()
                job.started_at = now()
                write_text(job.runner_dir / "launcher.txt", f"tmux session: {session}\n")
                return
            if launcher_preference == "tmux":
                raise PipelineError(f"could not create tmux session {session}")
            selected = "direct"

    if selected == "systemd":
        if not shutil.which("systemd-run"):
            raise PipelineError("launcher=systemd requested but systemd-run is not installed")
        unit = re.sub(r"[^A-Za-z0-9_.-]+", "_", session)[:180]
        completed = subprocess.run(
            [
                "systemd-run",
                "--user",
                "--unit",
                unit,
                "--collect",
                "--working-directory",
                str(job.runner_dir),
                "--",
                *command,
            ],
            check=False,
        )
        if completed.returncode != 0:
            if launcher_preference == "systemd":
                raise PipelineError(f"could not start systemd unit {unit}")
            selected = "direct"
        else:
            job.launcher = "systemd"
            job.started_monotonic = time.monotonic()
            job.started_at = now()
            write_text(job.runner_dir / "launcher.txt", f"systemd unit: {unit}\n")
            return

    if selected != "direct":
        raise PipelineError(f"unsupported launcher: {selected}")
    launcher_log = (job.runner_dir / "launcher.log").open("ab")
    process = subprocess.Popen(
        command,
        stdout=launcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job.process = process
    job.launcher_log = launcher_log
    job.launcher = "direct"
    job.started_monotonic = time.monotonic()
    job.started_at = now()
    write_text(job.runner_dir / "launcher.txt", f"direct pid: {process.pid}\n")


def read_state(job: SupervisorJob) -> str:
    try:
        return job.state_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "starting"


def external_launcher_alive(job: SupervisorJob) -> bool:
    if job.launcher == "tmux":
        result = subprocess.run(
            ["tmux", "has-session", "-t", job.session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    if job.launcher == "systemd" and shutil.which("systemctl"):
        unit = re.sub(r"[^A-Za-z0-9_.-]+", "_", job.session_name)[:180]
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    return True


def direct_process_alive(job: SupervisorJob) -> bool:
    return job.process is not None and job.process.poll() is None


def signal_direct_process(job: SupervisorJob, sig: signal.Signals) -> None:
    """Signal the direct supervisor and its Codex descendants when possible."""

    if not direct_process_alive(job) or job.process is None:
        return
    try:
        process_group = os.getpgid(job.process.pid)
        if process_group == job.process.pid:
            os.killpg(process_group, sig)
        elif sig == signal.SIGTERM:
            job.process.terminate()
        else:
            job.process.kill()
    except (OSError, ProcessLookupError):
        return


def stop_supervisor(job: SupervisorJob, *, force: bool = False) -> None:
    """Stop a supervisor and wait briefly for its Codex descendants to exit."""

    if job.launcher == "tmux" and shutil.which("tmux"):
        subprocess.run(["tmux", "kill-session", "-t", job.session_name], check=False)
    elif job.launcher == "systemd" and shutil.which("systemctl"):
        unit = re.sub(r"[^A-Za-z0-9_.-]+", "_", job.session_name)[:180]
        if force:
            subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "kill",
                    "--signal=KILL",
                    "--kill-whom=all",
                    unit,
                ],
                check=False,
            )
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
    elif job.launcher == "direct":
        signal_direct_process(job, signal.SIGKILL if force else signal.SIGTERM)

    stop_deadline = time.monotonic() + (1 if force else 10)
    while time.monotonic() < stop_deadline:
        if job.launcher == "direct":
            alive = direct_process_alive(job)
        else:
            alive = external_launcher_alive(job)
        if not alive:
            return
        time.sleep(0.1)

    if job.launcher == "direct" and direct_process_alive(job):
        signal_direct_process(job, signal.SIGKILL)
        if job.process is not None:
            try:
                job.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass


def task_elapsed_seconds(job: SupervisorJob) -> float:
    if job.started_monotonic is None:
        return 0.0
    return max(0.0, time.monotonic() - job.started_monotonic)


def record_run_analysis_for_job(
    job: SupervisorJob,
    *,
    status: str,
    elapsed_seconds: float | None = None,
    attempt_count: int | None = None,
    overwrite: bool = False,
) -> None:
    """Best-effort fallback when the supervisor cannot write its own summary."""

    analysis_file = job.analysis_file
    if analysis_file is None and job.report_path is not None:
        analysis_file = job.report_path.parent / "analysis.json"
    if analysis_file is None or (not overwrite and is_nonempty_file(analysis_file)):
        return
    if attempt_count is None:
        attempt_count, _ = _usage_attempt_and_log(job)
    try:
        write_run_analysis(
            analysis_file,
            total_elapsed_seconds=(
                task_elapsed_seconds(job)
                if elapsed_seconds is None
                else elapsed_seconds
            ),
            attempt_count=attempt_count,
            status=status,
            started_at=job.started_at,
            finished_at=now(),
            format_name=job.format_name,
            mode=job.mode,
            number=str(job.number),
            run_id=job.run_id,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry cannot stop later tasks.
        print(
            f"[{now()}] analysis record failed for "
            f"{job.format_name}/{job.mode}/{job.number}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def write_timeout_report(job: SupervisorJob, elapsed_seconds: float) -> None:
    if job.report_path is None:
        raise PipelineError(
            f"timeout report path is missing for {job.format_name}/{job.mode}/{job.number}"
        )
    write_text(
        job.report_path,
        (
            "# 测试任务超时\n\n"
            "运行时间超过18小时，强制终止任务。\n\n"
            f"- 数据格式：`{job.format_name}`\n"
            f"- 测试模式：`{job.mode}`\n"
            f"- 文章编号：`{job.number}`\n"
            f"- 最长运行时间：`{TASK_TIMEOUT_HOURS} 小时`\n"
            f"- 终止时累计运行时间：`{elapsed_seconds / 3600:.2f} 小时`\n"
            f"- 强制终止时间：`{now()}`\n\n"
            "该报告由流水线在任务超时后自动生成；本任务未正常完成科研测试。\n"
        ),
    )


def timeout_supervisor(job: SupervisorJob) -> None:
    elapsed_seconds = task_elapsed_seconds(job)
    print(
        f"[{now()}] Codex 超时 {job.format_name}/{job.mode}/{job.number}: "
        f"累计运行 {elapsed_seconds / 3600:.2f} 小时，强制终止",
        flush=True,
    )
    stop_supervisor(job, force=True)
    record_codex_usage_for_job(
        job,
        status="timeout",
        reason="task_timeout",
        usage_complete=False,
        usage_lower_bound=True,
    )
    record_run_analysis_for_job(
        job,
        status="timeout",
        elapsed_seconds=elapsed_seconds,
        overwrite=True,
    )
    write_timeout_report(job, elapsed_seconds)
    write_text(job.state_file, "task_timeout\n")
    write_text(
        job.status_file,
        (
            "# Codex 子进程状态报告\n\n"
            f"**更新时间**: {now()}\n\n"
            "**监督器状态**: `TASK_TIMEOUT`\n\n"
            "运行时间超过18小时，强制终止任务。超时报告已写入：\n\n"
            f"`{job.report_path}`\n"
        ),
    )


def wait_supervisors(
    jobs: list[SupervisorJob],
    poll_interval: int,
    task_timeout_seconds: float = TASK_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    if task_timeout_seconds <= 0:
        raise PipelineError("task timeout must be greater than zero")
    wait_started = time.monotonic()
    for job in jobs:
        if job.started_monotonic is None:
            job.started_monotonic = wait_started
        if job.started_at is None:
            job.started_at = now()

    pending = set(range(len(jobs)))
    failures: list[dict[str, object]] = []
    while pending:
        completed_this_round: list[int] = []
        for index in sorted(pending):
            job = jobs[index]
            state = read_state(job)
            if state == "task_complete":
                if (
                    job.report_path is None
                    or not is_nonempty_file(job.report_path)
                    or is_timeout_report(job.report_path)
                ):
                    record_codex_usage_for_job(
                        job,
                        status="failed",
                        reason="supervisor reported task_complete but no valid report was produced",
                        usage_complete=False,
                        usage_lower_bound=True,
                    )
                    record_run_analysis_for_job(job, status="failed", overwrite=True)
                    failures.append(
                        supervisor_failure_entry(
                            job,
                            "supervisor reported task_complete but no valid report was produced",
                        )
                    )
                    completed_this_round.append(index)
                    continue
                record_codex_usage_for_job(
                    job,
                    status="success",
                    reason="task_complete",
                    usage_complete=True,
                )
                record_run_analysis_for_job(job, status="success")
                print(f"[{now()}] Codex 完成 {job.format_name}/{job.mode}/{job.number}", flush=True)
                completed_this_round.append(index)
            elif state == "task_timeout":
                record_codex_usage_for_job(
                    job,
                    status="timeout",
                    reason="task_timeout",
                    usage_complete=False,
                    usage_lower_bound=True,
                )
                record_run_analysis_for_job(job, status="timeout")
                print(f"[{now()}] Codex 已超时 {job.format_name}/{job.mode}/{job.number}", flush=True)
                failures.append(supervisor_failure_entry(job, "task_timeout"))
                completed_this_round.append(index)
            elif state in {"task_failed", "interrupted"}:
                record_codex_usage_for_job(
                    job,
                    status="interrupted" if state == "interrupted" else "failed",
                    reason=state,
                    usage_complete=False,
                    usage_lower_bound=True,
                )
                record_run_analysis_for_job(
                    job,
                    status="interrupted" if state == "interrupted" else "failed",
                )
                failures.append(supervisor_failure_entry(job, state))
                completed_this_round.append(index)
            elif task_elapsed_seconds(job) >= task_timeout_seconds:
                timeout_supervisor(job)
                # The timeout path writes the terminal report and analysis
                # file, but it must also flow through the same failure list as
                # other terminal states so the article cannot be marked DONE
                # when a test task was forcibly stopped.
                failures.append(supervisor_failure_entry(job, "task_timeout"))
                completed_this_round.append(index)
            elif job.process is None and not external_launcher_alive(job):
                record_codex_usage_for_job(
                    job,
                    status="failed",
                    reason="external launcher exited without a terminal state",
                    usage_complete=False,
                )
                record_run_analysis_for_job(job, status="failed", overwrite=True)
                failures.append(
                    supervisor_failure_entry(
                        job,
                        "external launcher exited without a terminal state "
                        f"({state})",
                    )
                )
                completed_this_round.append(index)
            elif job.process is not None and job.process.poll() is not None:
                # A direct launcher can expose a supervisor failure before the
                # state file is flushed; do not silently proceed.
                final_state = read_state(job)
                if final_state == "task_complete":
                    record_codex_usage_for_job(
                        job,
                        status="success",
                        reason="task_complete",
                        usage_complete=True,
                    )
                    record_run_analysis_for_job(job, status="success")
                    completed_this_round.append(index)
                    continue
                record_codex_usage_for_job(
                    job,
                    status="failed",
                    reason=f"supervisor exited with {job.process.returncode}, state={final_state}",
                    usage_complete=False,
                    usage_lower_bound=True,
                )
                record_run_analysis_for_job(job, status="failed", overwrite=True)
                failures.append(
                    supervisor_failure_entry(
                        job,
                        f"supervisor exited with {job.process.returncode}, "
                        f"state={final_state}",
                    )
                )
                completed_this_round.append(index)
            else:
                status_tail = ""
                if job.status_file.is_file():
                    lines = job.status_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    status_tail = next(
                        (line.strip() for line in reversed(lines) if line.strip()),
                        "status updating",
                    )
                print(
                    f"[{now()}] Codex 运行中 {job.format_name}/{job.mode}/{job.number}: "
                    f"{state}; {status_tail}",
                    flush=True,
                )
        pending.difference_update(completed_this_round)
        if pending:
            sleep_seconds = max(0.001, min(float(poll_interval), 30.0))
            remaining_seconds = [
                max(
                    0.001,
                    task_timeout_seconds - task_elapsed_seconds(jobs[index]),
                )
                for index in pending
            ]
            time.sleep(min(sleep_seconds, *remaining_seconds))

    for job in jobs:
        if job.launcher_log is not None:
            job.launcher_log.close()

    return failures


def run_codex_phase(
    *,
    preprocess_dir: Path,
    agent_dir: Path,
    numbers: list[str],
    format_name: str,
    launcher: str,
    poll_interval: int,
    max_restarts: int,
    resume: bool,
) -> list[dict[str, object]]:
    failures_path = failure_list_path(agent_dir, format_name)
    phase_failures: list[dict[str, object]] = []
    for number in numbers:
        jobs: list[SupervisorJob] = []
        for mode in MODES:
            if format_name != "md":
                raise PipelineError(f"unsupported test format: {format_name}")
            input_path = preprocess_dir / "test-input" / "md" / mode / number / "agent_input.md"
            prompt_template = TEST_PROMPT_ROOT / mode / "prompt-md.md"
            report_path = agent_dir / "md" / mode / number / "research_report.md"
            runner_dir = agent_dir / "runners" / format_name / mode / number
            prompt_file = runner_dir / "prompt.md"
            usage_file = report_path.parent / "codex_token_usage.json"
            analysis_file = report_path.parent / "analysis.json"
            job = SupervisorJob(
                mode=mode,
                number=number,
                format_name=format_name,
                runner_dir=runner_dir,
                state_file=runner_dir / "state",
                status_file=runner_dir / "status.md",
                session_name=safe_component(f"scionbench_{format_name}_{mode}_{number}"),
                report_path=report_path,
                usage_file=usage_file,
                analysis_file=analysis_file,
                run_id=uuid.uuid4().hex,
            )
            job.started_monotonic = time.monotonic()
            job.started_at = now()
            try:
                ensure_file(input_path, f"{format_name} test input {mode}/{number}")
                previous_failure = task_failure_entry(
                    failures_path,
                    format_name=format_name,
                    mode=mode,
                    number=number,
                )
                if (
                    resume
                    and is_nonempty_file(report_path)
                    and not is_timeout_report(report_path)
                    and previous_failure is None
                ):
                    clear_failure_entry(
                        failures_path,
                        format_name=format_name,
                        mode=mode,
                        number=number,
                    )
                    print(
                        f"[{now()}] resume 跳过已完成 {format_name}/{mode}/{number}: "
                        f"{report_path}",
                        flush=True,
                    )
                    continue
                if resume and previous_failure is not None:
                    print(
                        f"[{now()}] resume 发现上次失败任务，重新尝试 "
                        f"{format_name}/{mode}/{number}: {previous_failure.get('reason', '')}",
                        flush=True,
                    )

                analysis_file.unlink(missing_ok=True)

                input_placeholder = "{{MD_DIR}}"
                replace_prompt(
                    prompt_template,
                    {input_placeholder: str(input_path), "{{OUTPUT_DIR}}": str(report_path.parent)},
                    prompt_file,
                )
                # The templates specify an output directory, while the supervisor
                # needs one unambiguous completion sentinel.  Keep the requested
                # directory and add the exact filename as an operational detail.
                prompt_content = prompt_file.read_text(encoding="utf-8")
                prompt_content += (
                    "\n\n自动化运行要求：请将最终的单一 Markdown 研究报告写入以下精确文件路径，"
                    "并确保在任务结束前该文件为非空：\n"
                    f"`{report_path}`\n"
                )
                write_text(prompt_file, prompt_content)

                command = supervisor_command(
                    prompt=prompt_file,
                    input_path=input_path,
                    output_path=report_path,
                    runner_dir=runner_dir,
                    workdir=runner_dir / "work",
                    health_interval=poll_interval,
                    max_restarts=max_restarts,
                    usage_file=usage_file,
                    analysis_file=analysis_file,
                    format_name=format_name,
                    mode=mode,
                    number=number,
                    run_id=job.run_id,
                )
                launch_supervisor(job=job, command=command, launcher_preference=launcher)
                jobs.append(job)
            except Exception as exc:  # noqa: BLE001 - isolate one mode/task.
                entry = supervisor_failure_entry(job, f"test setup/launch failed: {exc}")
                record_codex_usage_for_job(
                    job,
                    status="setup_failed",
                    reason=f"test setup/launch failed: {exc}",
                    attempt=0,
                )
                record_run_analysis_for_job(
                    job,
                    status="setup_failed",
                    attempt_count=0,
                    overwrite=True,
                )
                record_failure_entry(failures_path, entry)
                phase_failures.append(entry)
                print(
                    f"[{now()}] Codex 测试失败 {format_name}/{mode}/{number}; "
                    f"已记录并继续：{exc}",
                    file=sys.stderr,
                    flush=True,
                )

        if jobs:
            print(f"[{now()}] 等待文章编号 {number} 的四模式 {format_name} Codex 任务", flush=True)
            try:
                wait_failures = wait_supervisors(jobs, poll_interval) or []
            except Exception as exc:  # noqa: BLE001 - keep later numbers running.
                unresolved = [
                    job
                    for job in jobs
                    if read_state(job) not in {"task_complete", "task_timeout"}
                ]
                for job in unresolved:
                    try:
                        stop_supervisor(job, force=True)
                    except Exception:
                        pass
                for job in jobs:
                    if job.launcher_log is not None:
                        try:
                            job.launcher_log.close()
                        except Exception:
                            pass
                wait_failures = [
                    supervisor_failure_entry(
                        job,
                        f"supervisor wait failed: {exc}",
                    )
                    for job in unresolved
                ]
                for job in unresolved:
                    record_codex_usage_for_job(
                        job,
                        status="supervisor_wait_failed",
                        reason=f"supervisor wait failed: {exc}",
                        usage_complete=False,
                    )
                    record_run_analysis_for_job(
                        job,
                        status="supervisor_wait_failed",
                        overwrite=True,
                    )
            for failure in wait_failures:
                record_failure_entry(failures_path, failure)
                phase_failures.append(failure)
                print(
                    f"[{now()}] Codex 任务失败 {format_name}/{failure['mode']}/"
                    f"{failure['number']}; 已记录并继续：{failure['reason']}",
                    file=sys.stderr,
                    flush=True,
                )
            for job in jobs:
                if (
                    read_state(job) == "task_complete"
                    and job.report_path is not None
                    and is_nonempty_file(job.report_path)
                    and not is_timeout_report(job.report_path)
                ):
                    clear_failure_entry(
                        failures_path,
                        format_name=job.format_name,
                        mode=job.mode,
                        number=job.number,
                    )
    return phase_failures


def run_judge_phase(
    *,
    preprocess_dir: Path,
    agent_dir: Path,
    judge_dir: Path,
    numbers: Iterable[str],
    format_name: str,
    timeout: int,
    retries: int,
    resume: bool,
) -> list[dict[str, object]]:
    if format_name != "md":
        raise PipelineError(f"unsupported judge format: {format_name}")
    phase_failures: list[dict[str, object]] = []
    for number in numbers:
        command = [
            sys.executable,
            str(JUDGE_DISPATCH),
            "--judge-input-root",
            str(preprocess_dir / "judge-input"),
            "--agent-root",
            str(agent_dir),
            "--output-root",
            str(judge_dir),
            "--article-number",
            number,
            "--format",
            format_name,
            "--model",
            JUDGE_MODEL,
            "--api-mode",
            JUDGE_API_MODE,
            "--temperature",
            str(JUDGE_TEMPERATURE),
            "--stream",
            "true" if JUDGE_STREAM else "false",
            "--base-url",
            JUDGE_API_ENDPOINT,
            "--api-key-env",
            JUDGE_API_KEY_ENV,
            "--timeout",
            str(timeout),
            "--max-retries",
            str(retries),
            "--max-request-bytes",
            str(JUDGE_MAX_REQUEST_BYTES),
            "--max-images",
            str(JUDGE_MAX_IMAGES),
            "--failed-tasks-file",
            str(failure_list_path(agent_dir, format_name)),
        ]
        if resume:
            command.append("--resume")
        try:
            run_tool(
                f"step8 judge {number} ({format_name})",
                command,
            )
        except Exception as exc:  # noqa: BLE001 - continue with later numbers.
            entry = {
                "updated_at": now(),
                "format": format_name,
                "article_number": str(number),
                "mode": "dispatcher",
                "node_name": None,
                "status": "failed",
                "reason": f"judge dispatcher failed: {exc}",
                "attempts": 0,
            }
            record_evaluation_failure_entry(
                judge_dir / format_name / str(number) / "failed_evaluations.json",
                entry,
            )
            phase_failures.append(entry)
            print(
                f"[{now()}] judge 调度失败 {format_name}/{number}; "
                f"已记录并继续后续任务：{exc}",
                file=sys.stderr,
                flush=True,
            )
    return phase_failures


def max_restarts_value(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max restarts must be an integer") from exc
    if not 1 <= parsed <= MAX_RESTARTS_LIMIT:
        raise argparse.ArgumentTypeError(
            f"max restarts must be between 1 and {MAX_RESTARTS_LIMIT}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR,
                        help="一个文章目录；--single-paper 要求其中恰好一个 paper_graph.json；--batch 时改为包含一级文章目录的根目录")
    parser.add_argument("--preprocess-output-dir", default=PREPROCESS_OUTPUT_DIR,
                        help="第一至第四步输出目录")
    parser.add_argument("--agent-output-dir", default=AGENT_OUTPUT_DIR,
                        help="Markdown 测试输出目录（步骤 7）")
    parser.add_argument("--judge-output-dir", default=JUDGE_OUTPUT_DIR,
                        help="Markdown 评估输出目录（步骤 8）")
    parser.add_argument(
        "--source-md-root",
        type=Path,
        default=SOURCE_MD_ROOT,
        help="row4sci Markdown 根目录，用于解析并整理 figure 图片",
    )
    parser.add_argument("--batch", action="store_true",
                        help="按输入根目录的一级文章目录串行处理，并把三个输出目录按文章名分开")
    parser.add_argument(
        "--single-paper",
        action="store_true",
        help="只处理输入目录中的一篇文献，并在三个输出根目录下自动追加文献目录名",
    )
    parser.add_argument("--resume", action="store_true",
                        help="跳过 Markdown 测试的非空报告及 Markdown 评估的非空节点评估结果")
    parser.add_argument("--launcher", choices=("auto", "tmux", "systemd", "direct"), default="auto",
                        help="Codex 监督器启动方式（auto 优先 tmux，其次 systemd，再 direct）")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument(
        "--max-restarts",
        type=max_restarts_value,
        default=MAX_RESTARTS_LIMIT,
        help=f"每个 Codex 测试最多尝试次数（1-{MAX_RESTARTS_LIMIT}，默认 {MAX_RESTARTS_LIMIT}）",
    )
    parser.add_argument("--judge-timeout", type=int, default=600)
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=DEFAULT_JUDGE_RETRIES,
        help=f"每个评估节点的 API 重试次数（默认 {DEFAULT_JUDGE_RETRIES}）",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        choices=(4, 5, 6, 7, 8),
        default=8,
        help=(
            "只运行到指定步骤；JSON 测试/评估已禁用，5/7 在 Markdown 测试后停止，"
            "6/8 完成 Markdown 测试和评估"
        ),
    )
    return parser


def resolve_nonempty(value: str, label: str) -> Path:
    if not value.strip():
        raise PipelineError(f"{label} is empty; fill it in the configuration section or pass the CLI option")
    return Path(value).expanduser().resolve()


def run_article(
    *,
    input_dir: Path,
    preprocess_dir: Path,
    agent_dir: Path,
    judge_dir: Path,
    args: argparse.Namespace,
) -> None:
    write_status(preprocess_dir, "RUNNING", f"开始处理文章：{input_dir}")
    numbers = preprocess_graphs(input_dir, preprocess_dir)
    split_inputs(preprocess_dir, numbers)
    convert_inputs_to_markdown(preprocess_dir, numbers)
    extract_gold_nodes(preprocess_dir, numbers)
    try:
        data_stats = copy_source_data_directories(
            preprocess_dir=preprocess_dir,
            numbers=numbers,
            modes=MODES,
        )
    except DataAssetError as exc:
        raise PipelineError(f"source data directory copy failed: {exc}") from exc
    print(
        f"[{now()}] source data ready: sources={data_stats.source_directories}, "
        f"copied_targets={data_stats.copied_targets}, "
        f"missing={len(data_stats.missing_numbers)}",
        flush=True,
    )
    try:
        asset_stats = organize_article_assets(
            input_dir=input_dir,
            preprocess_dir=preprocess_dir,
            paper_extrac_root=PAPER_EXTRAC_ROOT,
            source_md_root=args.source_md_root.expanduser().resolve(),
            strict=True,
        )
    except FigureAssetError as exc:
        raise PipelineError(f"figure asset organization failed: {exc}") from exc
    print(
        f"[{now()}] figure assets ready: copied={asset_stats.copied}, "
        f"reused={asset_stats.reused}, targets={asset_stats.targets}",
        flush=True,
    )
    try:
        evaluation_stats = move_evaluation_files(
            preprocess_dir=preprocess_dir,
            numbers=numbers,
            modes=MODES,
        )
    except EvaluationMoveError as exc:
        raise PipelineError(f"evaluation file move failed: {exc}") from exc
    print(
        f"[{now()}] evaluation files ready: moved={evaluation_stats.moved_files}",
        flush=True,
    )
    # Do not retain an inactive JSON test-input channel after preprocessing.
    shutil.rmtree(preprocess_dir / ".work" / "split-json")
    shutil.rmtree(preprocess_dir / "test-input" / "json", ignore_errors=True)
    write_status(preprocess_dir, "PREPROCESS_DONE", f"步骤 1-4 完成，文章编号：{numbers}")
    if args.stop_after == 4:
        return

    phase_failures: list[dict[str, object]] = []

    def finish_or_raise() -> None:
        if not phase_failures:
            return
        summary = "; ".join(
            f"{item.get('format', '?')}/{item.get('mode', '?')}/"
            f"{item.get('number', item.get('article_number', '?'))}: {item.get('reason', 'failed')}"
            for item in phase_failures[:8]
        )
        write_status(
            preprocess_dir,
            "FAILED",
            f"Markdown 测试/评估存在 {len(phase_failures)} 个失败：{summary}",
        )
        raise PipelineError(
            f"{len(phase_failures)} pipeline task(s) failed; see "
            f"{failure_list_path(agent_dir, 'md')} and judge failure files"
        )

    # Split JSON exists only temporarily to derive Markdown test inputs and
    # held-out evaluation assets. Only the Markdown test and judge lanes run.
    #
    # Keep the historical stop-after values usable: 5 and 7 stop after the
    # sole active test lane, while 6 runs through the sole active evaluation
    # lane.  The aliases avoid silently starting a JSON task for old commands.
    if args.stop_after != 8:
        phase_failures.extend(run_codex_phase(
            preprocess_dir=preprocess_dir,
            agent_dir=agent_dir,
            numbers=numbers,
            format_name="md",
            launcher=args.launcher,
            poll_interval=args.poll_interval,
            max_restarts=args.max_restarts,
            resume=args.resume,
        ))
        if args.stop_after in (5, 7):
            finish_or_raise()
            return
        phase_failures.extend(run_judge_phase(
            preprocess_dir=preprocess_dir,
            agent_dir=agent_dir,
            judge_dir=judge_dir,
            numbers=numbers,
            format_name="md",
            timeout=args.judge_timeout,
            retries=args.judge_retries,
            resume=args.resume,
        ))
        finish_or_raise()
        return

    # The full run remains article/number serial: the Markdown test and
    # evaluation for 01 finish before any task for 02 is launched.  This keeps
    # reports and judge outputs for one paper complete and independently
    # auditable without running the disabled JSON lanes.
    for number in numbers:
        phase_failures.extend(run_codex_phase(
            preprocess_dir=preprocess_dir,
            agent_dir=agent_dir,
            numbers=[number],
            format_name="md",
            launcher=args.launcher,
            poll_interval=args.poll_interval,
            max_restarts=args.max_restarts,
            resume=args.resume,
        ))
        phase_failures.extend(run_judge_phase(
            preprocess_dir=preprocess_dir,
            agent_dir=agent_dir,
            judge_dir=judge_dir,
            numbers=[number],
            format_name="md",
            timeout=args.judge_timeout,
            retries=args.judge_retries,
            resume=args.resume,
        ))
    finish_or_raise()
    write_status(preprocess_dir, "DONE", f"步骤 1-4、7-8（仅 Markdown）完成，文章编号：{numbers}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path = resolve_nonempty(args.input_dir, "INPUT_DIR/--input-dir")
        preprocess_path = resolve_nonempty(
            args.preprocess_output_dir, "PREPROCESS_OUTPUT_DIR/--preprocess-output-dir"
        )
        agent_path = resolve_nonempty(args.agent_output_dir, "AGENT_OUTPUT_DIR/--agent-output-dir")
        judge_path = resolve_nonempty(args.judge_output_dir, "JUDGE_OUTPUT_DIR/--judge-output-dir")

        if args.batch and args.single_paper:
            raise PipelineError("--single-paper cannot be combined with --batch")

        if args.single_paper:
            preprocess_path, agent_path, judge_path = single_paper_output_dirs(
                input_path,
                preprocess_path,
                agent_path,
                judge_path,
            )
            paper_name = safe_component(input_path.name)
            print(
                f"[{now()}] 单篇文献模式：{paper_name}；输出目录已按文献名分层",
                flush=True,
            )
            run_article(
                input_dir=input_path,
                preprocess_dir=preprocess_path,
                agent_dir=agent_path,
                judge_dir=judge_path,
                args=args,
            )
        elif args.batch:
            for article in article_dirs(input_path):
                article_name = safe_component(article.name)
                print(f"[{now()}] 开始文章 {article_name}", flush=True)
                run_article(
                    input_dir=article,
                    preprocess_dir=preprocess_path / article_name,
                    agent_dir=agent_path / article_name,
                    judge_dir=judge_path / article_name,
                    args=args,
                )
        else:
            run_article(
                input_dir=input_path,
                preprocess_dir=preprocess_path,
                agent_dir=agent_path,
                judge_dir=judge_path,
                args=args,
            )
    except (PipelineError, OSError) as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
