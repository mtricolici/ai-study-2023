#!/bin/bash
set -e

# Neural network needs all images to be the same size.
# This script convers all images to 100x100 exact size.

function handle() {
  local src_dir=$1
  local dst_dir=$2

  mkdir -p "$dst_dir"

  find $src_dir -type f -name '*.jp*g' |\
  while read -r src_file;
  do
    base_name=$(basename -- "$src_file")
    dst_file="$dst_dir/$base_name"
    if [ ! -f "$dst_file" ]; then
       echo "$dst_file"
       convert "$src_file" \
         -gravity center -background red \
         -resize 100x100 -extent 100x100 \
         "$dst_file"
    fi
  done
}

handle "$HOME/ai-datasets/dogs-cats/train" "$HOME/ai-datasets/dogs-cats/train-normal"
handle "$HOME/ai-datasets/dogs-cats/test1" "$HOME/ai-datasets/dogs-cats/test1-normal"

