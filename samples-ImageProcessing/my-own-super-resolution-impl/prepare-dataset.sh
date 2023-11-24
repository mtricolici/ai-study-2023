#!/bin/bash
set -e

# Path with images of different sizes
RAW_DIR=$HOME/temp/raw-images

# output folder to save dataset ready for CNN training
DATASET_DIR=$HOME/temp/super-dataset

RSMALL=160x90
RBIG=640x360

# ImageMagic convert filters: "lanczos", "bilinear", "nearestneighbor", "box", "cubic", "mitchell"
OPS="-background black -gravity center -strip"

count=1
total=$(find $RAW_DIR/ -type f |wc -l)

find $RAW_DIR/ -type f -print0 |\
while IFS= read -d '' img; do
  prefix="${DATASET_DIR}/$(printf '%05d' $count)"

  # create a good unblured dataset example
  echo "Creating ${prefix}-small.png"
  convert "$img" -resize $RSMALL -filter lanczos -background black -gravity center -extent $RSMALL -strip "${prefix}-small.png"
  echo "Creating ${prefix}-big.png"
  convert "$img" -resize $RBIG -filter lanczos -background black -gravity center -extent $RBIG -strip "${prefix}-big.png"

  echo "${count} of ${total} done"
  
  ((count++))
done

echo "Done. Dataset is ready !"

