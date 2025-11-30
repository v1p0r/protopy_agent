#!/bin/bash
# copy all .png files recursively into one destination folder
# usage: ./copy_pngs.sh /path/to/source /path/to/destination

src="$1"
dst="$2"

if [ -z "$src" ] || [ -z "$dst" ]; then
  echo "Usage: $0 <source_dir> <destination_dir>"
  exit 1
fi

mkdir -p "$dst"

find "$src" -type f -iname "*.png" | while read -r file; do
  base=$(basename "$file")
  target="$dst/$base"
  i=1
  while [ -e "$target" ]; do
    target="$dst/${base%.*}_$i.png"
    ((i++))
  done
  cp -p "$file" "$target"
done

echo "✅ Copied all PNG files from '$src' → '$dst'"