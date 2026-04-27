#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="${PAPER_DIR:-"$ROOT_DIR/lossless-paper"}"
MAIN_TEX="${MAIN_TEX:-main.tex}"
CONDA_ENV="${CONDA_ENV:-quant}"

cd "$PAPER_DIR"

if [[ ! -f "$MAIN_TEX" ]]; then
  echo "error: cannot find $PAPER_DIR/$MAIN_TEX" >&2
  exit 1
fi

if command -v tectonic >/dev/null 2>&1; then
  TEX_CMD=(tectonic)
elif command -v conda >/dev/null 2>&1; then
  TEX_CMD=(conda run -n "$CONDA_ENV" tectonic)
else
  echo "error: tectonic not found, and conda is not available for env '$CONDA_ENV'" >&2
  exit 1
fi

echo "Compiling $PAPER_DIR/$MAIN_TEX"
echo "Using: ${TEX_CMD[*]}"

set +e
"${TEX_CMD[@]}" --keep-logs --keep-intermediates "$MAIN_TEX" 2>&1 | tee compile.log
status=${PIPESTATUS[0]}
set -e

if [[ $status -ne 0 ]]; then
  echo "error: LaTeX compilation failed; see $PAPER_DIR/compile.log" >&2
  exit "$status"
fi

base="${MAIN_TEX%.tex}"
pdf="$base.pdf"
log="$base.log"
blg="$base.blg"

if [[ ! -s "$pdf" ]]; then
  echo "error: expected output PDF $PAPER_DIR/$pdf was not created" >&2
  exit 1
fi

echo "PDF written: $PAPER_DIR/$pdf ($(du -h "$pdf" | cut -f1))"

pattern='undefined|Citation|Reference|There were undefined|Label\(s\) may have changed|Overfull|LaTeX Error|Emergency stop|Fatal'
files=(compile.log)
[[ -f "$log" ]] && files+=("$log")
[[ -f "$blg" ]] && files+=("$blg")

if command -v rg >/dev/null 2>&1; then
  findings="$(rg -n "$pattern" "${files[@]}" || true)"
else
  findings="$(grep -nE "$pattern" "${files[@]}" || true)"
fi

if [[ -n "$findings" ]]; then
  echo
  echo "Diagnostics needing review:"
  echo "$findings"
else
  echo "No LaTeX errors, undefined refs/citations, or overfull boxes found in logs."
fi
