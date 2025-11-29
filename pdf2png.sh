#!/usr/bin/env bash
# Convert a single PDF to a single PNG (first page only)
# Usage: ./pdf_to_png_single.sh /absolute/path/input.pdf /absolute/path/output.png

set -euo pipefail

pdf="${1:-}"
png="${2:-}"

if [[ -z "$pdf" || -z "$png" ]]; then
  echo "Usage: $0 <input.pdf> <output.png>"
  exit 1
fi

if [[ ! -f "$pdf" ]]; then
  echo "❌ PDF not found: $pdf"
  exit 1
fi

# Create output directory if needed
mkdir -p "$(dirname "$png")"

# Extract base name (pdftopng automatically appends -000001.png)
tmp_prefix="${png%.png}"
pdftopng "$pdf" "$tmp_prefix" >/dev/null 2>&1

# Move first generated page to desired output
first_page="${tmp_prefix}-000001.png"
if [[ -f "$first_page" ]]; then
  mv "$first_page" "$png"
  # Clean up any other pages if they exist
  rm -f "${tmp_prefix}"-0000??.png 2>/dev/null || true
  echo "✅ Saved: $png"
else
  echo "⚠️ No PNG output found for: $pdf"
  exit 1
fi
