#!/usr/bin/env bash
# Render docs/technical-report.md to a PDF: pandoc for the HTML, headless Chrome for the PDF.
#   docs/build-pdf.sh [OUT.pdf]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/technical-report.pdf}"
html="$(mktemp --suffix=.html)"
trap 'rm -f "$html"' EXIT
pandoc "$here/technical-report.md" --standalone --embed-resources --css "$here/report.css" \
    --metadata pagetitle="Ball possession from automatic tracking" -o "$html"
google-chrome --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$out" "$html" 2>/dev/null
echo "wrote $out"
