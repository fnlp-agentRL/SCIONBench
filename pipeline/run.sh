#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$repo_root/.env" ]]; then
  echo "Missing $repo_root/.env; copy .env.example and fill the pipeline settings." >&2
  exit 2
fi
set -a
source "$repo_root/.env"
set +a

required=(PIPELINE_INPUT_DIR PIPELINE_PREPROCESS_OUTPUT_DIR PIPELINE_AGENT_OUTPUT_DIR PIPELINE_JUDGE_OUTPUT_DIR PIPELINE_PAPER_EXTRAC_ROOT PIPELINE_SOURCE_MD_ROOT PIPELINE_MODEL)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Set $name in .env" >&2; exit 2; }
done
for name in PIPELINE_INPUT_DIR PIPELINE_PAPER_EXTRAC_ROOT PIPELINE_SOURCE_MD_ROOT; do
  [[ -d "${!name}" ]] || { echo "$name is not a directory: ${!name}" >&2; exit 2; }
done
command -v "${CODEX_BIN:-codex}" >/dev/null 2>&1 || { echo 'Codex CLI not found; set CODEX_BIN in .env' >&2; exit 2; }

mode="${PIPELINE_MODE:-single-paper}"
case "$mode" in
  single-paper) mode_arg=--single-paper ;;
  batch) mode_arg=--batch ;;
  article) mode_arg= ;;
  *) echo 'PIPELINE_MODE must be single-paper, batch, or article' >&2; exit 2 ;;
esac
stop_after="${PIPELINE_STOP_AFTER:-8}"
case "$stop_after" in 4|5|6|7|8) ;; *) echo 'PIPELINE_STOP_AFTER must be 4, 5, 6, 7, or 8' >&2; exit 2 ;; esac
if [[ "$stop_after" == 6 || "$stop_after" == 8 ]]; then
  for name in JUDGE_MODEL JUDGE_API_ENDPOINT AIGW_API_KEY; do
    [[ -n "${!name:-}" ]] || { echo "Set $name in .env for judge phases" >&2; exit 2; }
  done
fi

args=(
  --input-dir "$PIPELINE_INPUT_DIR"
  --preprocess-output-dir "$PIPELINE_PREPROCESS_OUTPUT_DIR"
  --agent-output-dir "$PIPELINE_AGENT_OUTPUT_DIR"
  --judge-output-dir "$PIPELINE_JUDGE_OUTPUT_DIR"
  --source-md-root "$PIPELINE_SOURCE_MD_ROOT"
  --launcher "${PIPELINE_LAUNCHER:-direct}"
  --stop-after "$stop_after"
)
[[ -z "$mode_arg" ]] || args+=("$mode_arg")
[[ "${PIPELINE_RESUME:-0}" == 1 ]] && args+=(--resume)

exec python3 "$repo_root/pipeline/scionbench_pipeline.py" "${args[@]}" "$@"
