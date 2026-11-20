#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf 'Usage: %s PDF_FILE_OR_DIR OUTPUT_DIR\n' "$0"
}
if [[ $# -ne 2 || "${1:-}" == -h || "${1:-}" == --help ]]; then
  usage
  [[ "${1:-}" == -h || "${1:-}" == --help ]] && exit 0
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  source "$repo_root/.env"
  set +a
fi

mineru_bin="${PDF2MD_MINERU_BIN:-mineru-kit}"
mineru_path="$(command -v "$mineru_bin" 2>/dev/null || true)"
[[ -n "$mineru_path" ]] || { echo 'MinerU CLI not found. Install pdf2md/requirements.txt and set PDF2MD_MINERU_BIN in .env.' >&2; exit 2; }
tier="${PDF2MD_TIER:-standard}"
ocr_mode="${PDF2MD_OCR_MODE:-}"
input="$1"
output="$2"
[[ -f "$input" || -d "$input" ]] || { echo "Input does not exist: $input" >&2; exit 2; }
mkdir -p -- "$output"
output="$(cd "$output" && pwd -P)"

parse_one() {
  local pdf="$1" relative="$2" stem="$3" target="$4"
  mkdir -p -- "$(dirname "$target")"
  local args=(parse "$pdf" -o "$target" --tier "$tier")
  [[ -z "$ocr_mode" ]] || args+=(--ocr-mode "$ocr_mode")
  printf 'Converting: %s -> %s\n' "$relative" "$target"
  "$mineru_path" "${args[@]}"
  [[ -s "$target" ]] || { echo "MinerU did not create Markdown: $target" >&2; return 1; }
}

if [[ -f "$input" ]]; then
  [[ "${input,,}" == *.pdf ]] || { echo "Input is not a PDF: $input" >&2; exit 2; }
  pdf="$(realpath "$input")"
  name="$(basename "$pdf")"
  parse_one "$pdf" "$name" "${name%.*}" "$output/${name%.*}.md"
else
  input="$(cd "$input" && pwd -P)"
  mapfile -d '' pdfs < <(find "$input" -type f -iname '*.pdf' -print0 | sort -z)
  [[ ${#pdfs[@]} -gt 0 ]] || { echo "No PDF files in: $input" >&2; exit 2; }
  for pdf in "${pdfs[@]}"; do
    relative="${pdf#"$input"/}"
    stem="${relative%.*}"
    parse_one "$pdf" "$relative" "$stem" "$output/$stem.md"
  done
fi
