#!/bin/bash
set -e

RAW_DIR=$HOME/temp/raw-images

DATASET_DIR=$HOME/datasets/blurred

RES=120x80

mkdir -p "$DATASET_DIR"

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

  # create a good unblured dataset example
  echo "Creating ${prefix}-good.png (rotation: $rotation)"
  convert "$img" $rotate -resize $RES -filter lanczos -background black -gravity center -extent $RES -strip "${prefix}-good.png"

  # creating a blurred example from good one
  echo "Creating ${prefix}-bad.png"
  convert "${prefix}-good.png" -resize 25% -strip "${prefix}-bad.png"
  convert "${prefix}-bad.png" -resize 400% -strip "${prefix}-bad.png"

  echo "${count} of ${total} done"
  ((count++))
done

echo "Done. Dataset is ready !"

