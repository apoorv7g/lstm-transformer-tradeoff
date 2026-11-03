#!/usr/bin/env bash
# Render slides.html to PDF with headless Edge (same 720x405pt page as CacheMemory.pdf).
# Usage (Git Bash, from repo root): bash docs/slides/build.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
edge="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
out="$here/Progress_Review_2026-10-23_CacheMemory.pdf"
"$edge" --headless=new --disable-gpu --no-pdf-header-footer --no-margins \
  --print-to-pdf="$(cygpath -w "$out")" "file:///$(cygpath -m "$here/slides.html")"
echo "wrote $out"
