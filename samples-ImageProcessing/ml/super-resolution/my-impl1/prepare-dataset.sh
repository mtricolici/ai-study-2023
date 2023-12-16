#!/bin/bash
set -e

# Path with images of different sizes
RAW_DIR=$HOME/temp/raw-images

# output folder to save dataset ready for CNN training
DATASET_DIR=$HOME/temp/super-dataset

RSMALL=60x50
RBIG=240x200

# ImageMagic convert filters: "lanczos", "bilinear", "nearestneighbor", "box", "cubic", "mitchell"
OPS="-background black -gravity center -strip"

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
  echo "Creating ${prefix}-small.png (rotation: $rotation)"
  convert "$img" $rotate -resize $RSMALL -filter lanczos -background black -gravity center -extent $RSMALL -strip "${prefix}-small.png"
  echo "Creating ${prefix}-big.png (rotation: $rotation)"
  convert "$img" $rotate -resize $RBIG -filter lanczos -background black -gravity center -extent $RBIG -strip "${prefix}-big.png"

  echo "${count} of ${total} done"
  
  ((count++))
done

echo "Done. Dataset is ready !"

