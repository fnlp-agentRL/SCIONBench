#!/usr/bin/env bash
# Run one isolated Codex task, monitor it, and restart it after an abnormal exit.
# The pipeline launches this script from tmux when tmux is available.  It is
# also useful on its own, because status.md, supervisor.log, and the optional
# aggregate token-usage JSON are sufficient to inspect a task from another
# shell or a systemd unit.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_USAGE_TOOL="$SCRIPT_DIR/codex_token_usage.py"
RUN_ANALYSIS_TOOL="$SCRIPT_DIR/run_analysis.py"

usage() {
    cat >&2 <<'EOF'
Usage:
  codex_supervisor.sh --prompt FILE --input FILE --output FILE --runner DIR [options]

Options:
  --model NAME                 Codex model (default: glm-5.2)
  --workdir DIR                Codex working directory (default: runner DIR)
  --max-restarts N             Maximum fresh attempts (1-10; default: 10)
  --restart-delay SECONDS      Delay before a restart (default: 10)
  --health-interval SECONDS    Status refresh interval (default: 30)
  --codex-home DIR             Original home used only for auth/config copying
  --usage-file FILE            Aggregate token usage JSON written beside report
  --analysis-file FILE         Duration/retry JSON written beside report
  --format NAME                Pipeline input format for usage metadata
  --mode NAME                  Pipeline test mode for usage metadata
  --number NAME                Pipeline article number for usage metadata
  --run-id ID                  Unique pipeline run identifier
  --help                       Show this help
EOF
}

PROMPT_FILE=""
INPUT_FILE=""
OUTPUT_FILE=""
RUNNER_DIR=""
# Use the model id configured by the active Codex provider.
MODEL="glm-5.2"
WORKDIR=""
MAX_RESTARTS=10
MAX_RESTARTS_LIMIT=10
RESTART_DELAY=10
HEALTH_INTERVAL=30
# Keep verbose shell/tool output from consuming the entire Responses context.
# Codex truncates the model-visible tool result; the raw JSONL log is unchanged.
CODEX_TOOL_OUTPUT_TOKEN_LIMIT="${CODEX_TOOL_OUTPUT_TOKEN_LIMIT:-6000}"
ORIGINAL_CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
USAGE_FILE=""
ANALYSIS_FILE=""
FORMAT_NAME="unknown"
MODE="unknown"
NUMBER="unknown"
RUN_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt) PROMPT_FILE="$2"; shift 2 ;;
        --input) INPUT_FILE="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        --runner) RUNNER_DIR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        --restart-delay) RESTART_DELAY="$2"; shift 2 ;;
        --health-interval) HEALTH_INTERVAL="$2"; shift 2 ;;
        --codex-home) ORIGINAL_CODEX_HOME="$2"; shift 2 ;;
        --usage-file) USAGE_FILE="$2"; shift 2 ;;
        --analysis-file) ANALYSIS_FILE="$2"; shift 2 ;;
        --format) FORMAT_NAME="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --number) NUMBER="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$PROMPT_FILE" || -z "$INPUT_FILE" || -z "$OUTPUT_FILE" || -z "$RUNNER_DIR" ]]; then
    echo "--prompt, --input, --output and --runner are required" >&2
    usage
    exit 2
fi

if [[ ! "$MAX_RESTARTS" =~ ^[1-9][0-9]*$ ]] || (( MAX_RESTARTS > MAX_RESTARTS_LIMIT )); then
    echo "--max-restarts must be an integer between 1 and $MAX_RESTARTS_LIMIT" >&2
    exit 2
fi

if [[ ! "$CODEX_TOOL_OUTPUT_TOKEN_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "CODEX_TOOL_OUTPUT_TOKEN_LIMIT must be a positive integer" >&2
    exit 2
fi

if [[ ! -f "$PROMPT_FILE" ]]; then echo "prompt does not exist: $PROMPT_FILE" >&2; exit 2; fi
if [[ ! -f "$INPUT_FILE" ]]; then echo "input does not exist: $INPUT_FILE" >&2; exit 2; fi

mkdir -p "$RUNNER_DIR/logs" "$(dirname "$OUTPUT_FILE")"
if [[ -z "$WORKDIR" ]]; then WORKDIR="$RUNNER_DIR"; fi
mkdir -p "$WORKDIR"
if [[ -n "$USAGE_FILE" ]]; then mkdir -p "$(dirname "$USAGE_FILE")"; fi
if [[ -z "$ANALYSIS_FILE" ]]; then ANALYSIS_FILE="$(dirname "$OUTPUT_FILE")/analysis.json"; fi
mkdir -p "$(dirname "$ANALYSIS_FILE")"
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="supervisor_$(date +%s%N)_$$"
fi
ANALYSIS_STARTED_MONOTONIC_NS="$(python3 -c 'import time; print(time.monotonic_ns())')"
ANALYSIS_STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

STATE_FILE="$RUNNER_DIR/state"
STATUS_FILE="$RUNNER_DIR/status.md"
SUPERVISOR_LOG="$RUNNER_DIR/supervisor.log"
CURRENT_ATTEMPT_FILE="$RUNNER_DIR/current_attempt"
RUN_ID_FILE="$RUNNER_DIR/run_id"
printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
printf '0\n' > "$CURRENT_ATTEMPT_FILE"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

file_signature() {
    if [[ -s "$OUTPUT_FILE" ]]; then
        stat -c '%Y:%s:%i' "$OUTPUT_FILE" 2>/dev/null || printf 'present\n'
    else
        printf 'missing\n'
    fi
}

failure_reason_from_log() {
    local log_file="$1"
    if [[ -f "$log_file" ]] && grep -Eqi \
        'maximum context length|context length|prompt contains at least|input_tokens.*202[0-9]{3}|上下文.{0,12}(超限|过长)' \
        "$log_file"; then
        printf '上游拒绝：模型上下文超过上限（请压缩输入/命令输出后重试）'
        return 0
    fi
    if [[ -f "$log_file" ]] && grep -Eqi \
        'stream disconnected|stream closed before response|reconnecting|connection reset|timed out|timeout' \
        "$log_file"; then
        printf 'Codex 长连接/上游流中断（监督器将使用全新隔离状态重试）'
        return 0
    fi
    return 1
}

log_msg() {
    local line="[$(timestamp)] $1"
    printf '%s\n' "$line" >> "$SUPERVISOR_LOG"
    printf '%s\n' "$line"
}

write_run_analysis() {
    local status="$1"
    python3 "$RUN_ANALYSIS_TOOL" \
        --analysis-file "$ANALYSIS_FILE" \
        --started-monotonic-ns "$ANALYSIS_STARTED_MONOTONIC_NS" \
        --started-at "$ANALYSIS_STARTED_AT" \
        --attempt-count "$attempt" \
        --status "$status" \
        --format "$FORMAT_NAME" \
        --mode "$MODE" \
        --number "$NUMBER" \
        --run-id "$RUN_ID" \
        >/dev/null 2>>"$SUPERVISOR_LOG" || log_msg "analysis record failed"
}

record_attempt_usage() {
    local status="$1"
    local exit_code="${2:-}"
    local reason="${3:-}"
    local complete="${4:-false}"
    local lower_bound="${5:-false}"
    if [[ -z "$USAGE_FILE" || "$attempt_started" != "true" || "$attempt_recorded" == "true" ]]; then
        return 0
    fi
    local complete_args=()
    local lower_bound_args=()
    if [[ "$complete" == "true" ]]; then complete_args=(--usage-complete); fi
    if [[ "$lower_bound" == "true" ]]; then lower_bound_args=(--usage-lower-bound); fi
    local exit_args=()
    if [[ -n "$exit_code" ]]; then exit_args=(--exit-code "$exit_code"); fi
    local started_args=()
    if [[ -n "$attempt_started_at" ]]; then
        started_args=(--started-at "$attempt_started_at")
    fi
    python3 "$TOKEN_USAGE_TOOL" record \
        --usage-file "$USAGE_FILE" \
        --events-file "$log_file" \
        --format "$FORMAT_NAME" \
        --mode "$MODE" \
        --number "$NUMBER" \
        --run-id "$RUN_ID" \
        --attempt "$attempt" \
        --status "$status" \
        "${exit_args[@]+"${exit_args[@]}"}" \
        --reason "$reason" \
        "${started_args[@]+"${started_args[@]}"}" \
        "${complete_args[@]+"${complete_args[@]}"}" \
        "${lower_bound_args[@]+"${lower_bound_args[@]}"}" \
        >/dev/null 2>>"$SUPERVISOR_LOG" || log_msg "token usage record failed for attempt #$attempt"
    attempt_recorded="true"
}

write_status() {
    local state="$1"
    local attempt="$2"
    local pid="$3"
    local home="$4"
    local log_file="$5"
    local extra="${6:-}"
    local tmp_file="$STATUS_FILE.tmp.$$"
    {
        printf '# Codex 子进程状态报告\n\n'
        printf '**更新时间**: %s\n\n' "$(timestamp)"
        printf '| 项目 | 值 |\n|---|---|\n'
        printf '| 监督器状态 | `%s` |\n' "$state"
        printf '| 当前尝试 | `%s / %s` |\n' "$attempt" "$MAX_RESTARTS"
        printf '| Codex PID | `%s` |\n' "$pid"
        printf '| 模型 | `%s` |\n' "$MODEL"
        printf '| 隔离 CODEX_HOME | `%s` |\n' "$home"
        printf '| 输入 | `%s` |\n' "$INPUT_FILE"
        printf '| 输出 | `%s` |\n' "$OUTPUT_FILE"
        printf '| 当前日志 | `%s` |\n\n' "$log_file"
        printf '## 附加状态\n\n%s\n\n' "$extra"
        printf '## 监督器日志（最后 30 行）\n\n```text\n'
        tail -30 "$SUPERVISOR_LOG" 2>/dev/null || true
        printf '\n```\n\n## Codex 输出（最后 40 行）\n\n```text\n'
        # A terminal failure passes "-" because the attempt log was removed.
        # Never let tail read the supervisor's stdin in that case.
        if [[ "$log_file" == "-" || -z "$log_file" ]]; then
            printf '(日志文件不可用)\n'
        else
            tail -40 -- "$log_file" 2>/dev/null || true
        fi
        printf '\n```\n\n## 输出文件\n\n'
        if [[ -s "$OUTPUT_FILE" ]]; then
            ls -l "$OUTPUT_FILE"
        else
            printf '(尚未生成)\n'
        fi
    } > "$tmp_file"
    mv -f "$tmp_file" "$STATUS_FILE"
}

copy_if_present() {
    local source="$1"
    local target="$2"
    if [[ -f "$source" ]]; then
        cp "$source" "$target" 2>/dev/null || true
    fi
}

catalog_is_usable() {
    local candidate="$1"
    [[ -s "$candidate" ]] || return 1
    # CCR's Codex catalog is an object with a `models` array.  The router's
    # internal dist/models.json has a different schema and must not be copied
    # as a Codex catalog merely because it happens to be available.
    python3 - "$candidate" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError, IndexError):
    raise SystemExit(1)
raise SystemExit(0 if isinstance(value, dict) and isinstance(value.get("models"), list) else 1)
PY
}

find_model_catalog() {
    local configured=""
    local candidate=""
    if [[ -f "$ORIGINAL_CODEX_HOME/config.toml" ]]; then
        configured="$(sed -n \
            's/^[[:space:]]*model_catalog_json[[:space:]]*=[[:space:]]*"\([^"]*\)".*$/\1/p' \
            "$ORIGINAL_CODEX_HOME/config.toml" | head -1)"
        if [[ -n "$configured" && "$configured" != /* ]]; then
            configured="$ORIGINAL_CODEX_HOME/$configured"
        fi
    fi
    local -a candidates=(
        "$configured"
        "$ORIGINAL_CODEX_HOME/ccr-model-catalog.json"
        "$ORIGINAL_CODEX_HOME/.claude-code-router/codex-app-user-data/default-codex/ccr-codex-model-catalog.json"
        "$ORIGINAL_CODEX_HOME/.claude-code-router/ccr-model-catalog.json"
        "${CCR_MODEL_CATALOG_PATH:-}"
    )
    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if catalog_is_usable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    # A previous successful attempt can be the only surviving copy when the
    # original HOME was provisioned without the CCR catalog.
    while IFS= read -r candidate; do
        if catalog_is_usable "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$RUNNER_DIR" -type f -name 'ccr-model-catalog.json' -print 2>/dev/null)
    return 1
}

rewrite_model_catalog_path() {
    local config_file="$1"
    local catalog_path="${2:-}"
    [[ -f "$config_file" ]] || return 0
    # Rewrite only the managed setting and preserve the rest of the user's
    # Codex/CCR configuration.  If no valid catalog was found, remove the
    # stale setting so Codex can use its normal metadata fallback.
    python3 - "$config_file" "$catalog_path" <<'PY'
import os
import re
import sys
from pathlib import Path

config = Path(sys.argv[1])
catalog = sys.argv[2]
text = config.read_text(encoding="utf-8")
escaped = catalog.replace("\\", "\\\\").replace('"', '\\"')
replacement = f'model_catalog_json = "{escaped}"\n' if catalog else ""
lines = text.splitlines(keepends=True)
out = []
replaced = False
pattern = re.compile(r"^[ \\t]*model_catalog_json[ \\t]*=")
for line in lines:
    if pattern.match(line):
        if replacement:
            out.append(replacement)
        replaced = True
    else:
        out.append(line)
if catalog and not replaced:
    out.insert(0, replacement)
temporary = config.with_name(f".{config.name}.tmp.{os.getpid()}")
temporary.write_text("".join(out), encoding="utf-8")
os.replace(temporary, config)
PY
}

prepare_clean_home() {
    local attempt="$1"
    local clean_home="$RUNNER_DIR/codex_home_attempt_${attempt}"
    local catalog_source=""
    # This path is generated below RUNNER_DIR, so removing it only removes the
    # failed attempt's isolated Codex state, including sessions and memories.
    if [[ -d "$clean_home" ]]; then rm -rf "$clean_home"; fi
    mkdir -p "$clean_home"
    copy_if_present "$ORIGINAL_CODEX_HOME/auth.json" "$clean_home/auth.json"
    copy_if_present "$ORIGINAL_CODEX_HOME/config.toml" "$clean_home/config.toml"
    if catalog_source="$(find_model_catalog)"; then
        cp -- "$catalog_source" "$clean_home/ccr-model-catalog.json" 2>/dev/null || true
    else
        printf '[codex_supervisor] no usable CCR model catalog found for attempt %s\n' "$attempt" >&2
    fi
    rewrite_model_catalog_path \
        "$clean_home/config.toml" \
        "$(if [[ -f "$clean_home/ccr-model-catalog.json" ]]; then printf '%s' "$clean_home/ccr-model-catalog.json"; fi)"
    copy_if_present "$ORIGINAL_CODEX_HOME/installation_id" "$clean_home/installation_id"
    if [[ -d "$ORIGINAL_CODEX_HOME/.sii" ]]; then
        cp -r "$ORIGINAL_CODEX_HOME/.sii" "$clean_home/.sii" 2>/dev/null || true
    fi
    if [[ -d "$ORIGINAL_CODEX_HOME/.claude-code-router" ]]; then
        cp -r "$ORIGINAL_CODEX_HOME/.claude-code-router" "$clean_home/.claude-code-router" 2>/dev/null || true
    fi
    printf '%s\n' "$clean_home"
}

child_pid=""
attempt=0
attempt_started="false"
attempt_recorded="false"
attempt_started_at=""
log_file=""
on_signal() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill "$child_pid" 2>/dev/null || true
    fi
    if [[ "$attempt_started" == "true" ]]; then
        record_attempt_usage "interrupted" "" "监督器收到终止信号" "false" "true"
    fi
    write_run_analysis "interrupted"
    printf 'interrupted\n' > "$STATE_FILE"
    write_status "INTERRUPTED" "-" "$child_pid" "-" "-" "监督器收到终止信号。"
    exit 130
}
trap on_signal INT TERM

printf 'supervisor_running\n' > "$STATE_FILE"
log_msg "监督器启动；模型=$MODEL"
log_msg "prompt=$PROMPT_FILE"
log_msg "input=$INPUT_FILE"
log_msg "output=$OUTPUT_FILE"

while [[ "$attempt" -lt "$MAX_RESTARTS" ]]; do
    attempt=$((attempt + 1))
    clean_home="$(prepare_clean_home "$attempt")"
    log_file="$RUNNER_DIR/logs/codex_attempt_${attempt}.log"
    last_message="$RUNNER_DIR/last_message_attempt_${attempt}.txt"
    attempt_started="true"
    attempt_recorded="false"
    attempt_started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '%s\n' "$attempt" > "$CURRENT_ATTEMPT_FILE"
    output_signature_before="$(file_signature)"
    : > "$log_file"
    : > "$last_message"
    write_status "STARTING" "$attempt" "-" "$clean_home" "$log_file" "准备全新的 Codex HOME。"

    log_msg "启动尝试 #${attempt}/${MAX_RESTARTS}，CODEX_HOME=$clean_home"
    # The approval and sandbox flags are deliberately placed before `exec`;
    # this is the syntax supported by the installed Codex CLI.
    CODEX_HOME="$clean_home" "${CODEX_BIN:-codex}" \
        --ask-for-approval never \
        --sandbox danger-full-access \
        -c "tool_output_token_limit=$CODEX_TOOL_OUTPUT_TOKEN_LIMIT" \
        exec \
        --json \
        --model "$MODEL" \
        --skip-git-repo-check \
        -C "$WORKDIR" \
        --add-dir "$(dirname "$INPUT_FILE")" \
        --add-dir "$(dirname "$OUTPUT_FILE")" \
        --output-last-message "$last_message" \
        - < "$PROMPT_FILE" > "$log_file" 2>&1 &
    child_pid=$!
    printf 'codex_running:%s:attempt:%s\n' "$child_pid" "$attempt" > "$STATE_FILE"
    write_status "RUNNING" "$attempt" "$child_pid" "$clean_home" "$log_file" "Codex 正在执行；可从另一终端查看此 status.md 和日志。"

    while kill -0 "$child_pid" 2>/dev/null; do
        write_status "RUNNING" "$attempt" "$child_pid" "$clean_home" "$log_file" "仍在运行，最后一次检查间隔为 ${HEALTH_INTERVAL}s。"
        sleep "$HEALTH_INTERVAL"
    done

    wait "$child_pid" 2>/dev/null
    exit_code=$?
    child_pid=""
    log_msg "Codex 尝试 #$attempt 结束，exit_code=$exit_code"

    output_signature_after="$(file_signature)"
    output_changed="false"
    if [[ "$output_signature_before" == "missing" && "$output_signature_after" != "missing" ]]; then
        output_changed="true"
    elif [[ "$output_signature_before" != "$output_signature_after" ]]; then
        output_changed="true"
    fi

    if [[ "$exit_code" -eq 0 && "$output_signature_after" != "missing" && "$output_changed" == "true" ]]; then
        record_attempt_usage "success" "$exit_code" "Codex 正常退出并生成报告" "true"
        write_run_analysis "success"
        printf 'task_complete\n' > "$STATE_FILE"
        write_status "DONE" "$attempt" "-" "$clean_home" "$log_file" "报告已生成，任务完成。"
        log_msg "任务完成：$OUTPUT_FILE"
        exit 0
    fi

    if [[ "$exit_code" -eq 0 && "$output_signature_after" != "missing" ]]; then
        reason="Codex 正常退出但报告没有被本次任务创建或更新（拒绝使用旧报告）"
    elif [[ "$exit_code" -eq 0 ]]; then
        reason="Codex 正常退出但没有生成非空报告"
    elif reason_from_log="$(failure_reason_from_log "$log_file")"; [[ -n "$reason_from_log" ]]; then
        reason="$reason_from_log"
    else
        reason="Codex 异常退出（常见原因包括网络断连或服务端错误）"
    fi
    # A failed process may have stopped after its last usage snapshot, so keep
    # the value but mark it as a possible lower bound.
    record_attempt_usage "failed" "$exit_code" "$reason" "false" "true"
    log_msg "${reason}；删除本次隔离 CODEX_HOME 后重启"
    if [[ -d "$clean_home" ]]; then rm -rf "$clean_home"; fi
    printf 'restarting:attempt:%s\n' "$attempt" > "$STATE_FILE"
    write_status "RESTARTING" "$attempt" "-" "(已删除)" "$log_file" "${reason}；${RESTART_DELAY}s 后使用全新 memory 重试。"
    if [[ "$attempt" -lt "$MAX_RESTARTS" ]]; then sleep "$RESTART_DELAY"; fi
done

write_run_analysis "failed"
printf 'task_failed\n' > "$STATE_FILE"
write_status "FAILED" "$MAX_RESTARTS" "-" "(已删除)" "-" "达到最大重试次数，任务失败。"
log_msg "达到最大重试次数 ${MAX_RESTARTS}，任务失败"
exit 1
