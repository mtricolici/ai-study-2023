#!/bin/bash
set -e

# Path with images of different sizes
RAW_DIR=$HOME/temp/raw-images

# output folder to save dataset for image restoration
DATASET_DIR=$HOME/temp/image-restoration

mkdir -p "$DATASET_DIR"

RES="100x70"

count=1
total=$(find $RAW_DIR/ -type f -name '*.png' |wc -l)

find $RAW_DIR/ -type f -name '*.png' -print0 | shuf -z |\
while IFS= read -d '' img; do
  prefix="${DATASET_DIR}/$(printf '%05d' $count)"

  dimensions=$(identify -format "%w %h" "$img")
  read w h <<< "$dimensions"

  rotate=""
  rotation="NO"
  if [ "$h" -gt "$w" ]; then
    rotate="-rotate 90"
    rotation="YES"
  fi

  echo "Creating ${prefix}-good.png (rotation: $rotation)"
  convert "$img" $rotate -resize $RES -filter lanczos -background black -gravity center -extent $RES -strip "${prefix}-good.png"

  echo "${count} of ${total} done"
  ((count++))
done

echo "Done. Dataset good samples are ready ! Now run python to generate bad samples"
