#!/usr/bin/env bash
# Compile main.tex into main.pdf with no user interaction.
#
# Usage:
#   bash compile.sh           # compiles in this directory
#
# The script runs pdflatex twice so that \ref / \cref / table-of-contents
# style cross-references resolve, then cleans up auxiliary files. If
# `latexmk` is available it is used preferentially; otherwise the script
# falls back to plain pdflatex.
set -euo pipefail

cd "$(dirname "$0")"

TEX_FILE="main.tex"
JOB_NAME="main"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "ERROR: pdflatex not found on PATH." >&2
    exit 1
fi

if command -v latexmk >/dev/null 2>&1; then
    echo "Using latexmk."
    latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
        -jobname="$JOB_NAME" "$TEX_FILE"
    latexmk -c -jobname="$JOB_NAME" >/dev/null 2>&1 || true
else
    echo "latexmk not found; falling back to two passes of pdflatex."
    PDFLATEX_FLAGS="-interaction=nonstopmode -halt-on-error -file-line-error -jobname=$JOB_NAME"
    pdflatex $PDFLATEX_FLAGS "$TEX_FILE"
    pdflatex $PDFLATEX_FLAGS "$TEX_FILE"
    # Tidy up auxiliary files; keep .pdf and .log.
    for ext in aux out toc lof lot bbl blg fls fdb_latexmk synctex.gz; do
        rm -f "${JOB_NAME}.${ext}"
    done
fi

if [[ -f "${JOB_NAME}.pdf" ]]; then
    echo "OK: built ${JOB_NAME}.pdf"
else
    echo "ERROR: ${JOB_NAME}.pdf was not produced. See ${JOB_NAME}.log." >&2
    exit 1
fi
