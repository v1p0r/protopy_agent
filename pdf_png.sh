#!/usr/bin/env bash
# Recursively convert all PDFs in SRC to PNGs in DST using pdftopng
# Uses system default resolution (typically 150 DPI)
# Usage: ./pdf_to_png_recursive.sh /path/to/source /path/to/destination

set -euo pipefail

src="${1:-}"
dst="${2:-}"

if [[ -z "$src" || -z "$dst" ]]; then
  echo "Usage: $0 <source_dir> <destination_dir>"
  exit 1
fi

# Normalize and verify paths
src="${src%/}"
dst="${dst%/}"

if [[ ! -d "$src" ]]; then
  echo "❌ Source directory not found: $src"
  exit 1
fi
mkdir -p "$dst"

echo "🔎 Searching PDFs in: $src"
echo "📁 Output directory:  $dst"

# Find all PDFs and convert them
find "$src" -type f -iname "*.pdf" | while read -r pdf; do
  rel="${pdf#"$src"/}"             # relative path under src
  rel_dir="$(dirname "$rel")"
  base="$(basename "$pdf" .pdf)"
  out_dir="$dst/$rel_dir"

  mkdir -p "$out_dir"

  echo "➡️  Converting: $rel"
  # Convert to PNG (default DPI)
  pdftopng "$pdf" "$out_dir/${base}" >/dev/null 2>&1

  if ls "$out_dir/${base}"-*.png >/dev/null 2>&1; then
    echo "   ✅ Output: $out_dir/${base}-*.png"
  else
    echo "   ⚠️ No output found for: $pdf"
  fi
done

echo "✅ Done! All PNGs written to: $dst"
