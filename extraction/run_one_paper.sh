#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  printf 'Usage: %s PAPER_DIR OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

paper_dir="$(realpath -e "$1")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$repo_root/.env" ]; then
  # The owner of this checkout controls this Bash-compatible config file.
  set -a
  source "$repo_root/.env"
  set +a
fi
output_dir="$2"
mkdir -p "$output_dir"
output_dir="$(realpath -e "$output_dir")"
validator="$repo_root/extraction/validate_extraction.py"
prompt="$repo_root/extraction/extract_prompt.txt"
codex_bin="${CODEX_BIN:-codex}"
codex_path="$(command -v "$codex_bin" 2>/dev/null || true)"
codex_args=()
if [ -n "${SCIONBENCH_MODEL:-}" ]; then
  codex_args+=(-m "$SCIONBENCH_MODEL")
fi
if [ -n "${SCIONBENCH_MODEL_REASONING_EFFORT:-}" ]; then
  codex_args+=(-c "model_reasoning_effort=\"$SCIONBENCH_MODEL_REASONING_EFFORT\"")
fi

if [ ! -f "$prompt" ] || [ ! -f "$validator" ]; then
  printf 'Missing extraction prompt or validator in %s\n' "$repo_root/extraction" >&2
  exit 2
fi
if [ -z "$codex_path" ]; then
  printf 'Codex CLI not found. Put codex on PATH or set CODEX_BIN to its executable path.\n' >&2
  exit 2
fi

python3 - "$prompt" "$paper_dir" "$output_dir" "$validator" <<'PY' |
import pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text()
for key, value in zip(('PAPER_DIR', 'OUTPUT_DIR', 'VALIDATOR_PATH'), sys.argv[2:]):
    text = text.replace('{{' + key + '}}', value)
sys.stdout.write(text)
PY
"$codex_path" "${codex_args[@]}" \
  -C "$repo_root" --sandbox danger-full-access --ask-for-approval never \
  exec --ephemeral --skip-git-repo-check -

python3 "$validator" --paper-dir "$paper_dir" \
  --graph "$output_dir/paper_graph.json" \
  --report "$output_dir/validation_report.json"
