#!/bin/bash
set -e

# Path with images of different sizes
RAW_DIR=$HOME/temp/raw-images

# output folder to save dataset ready for CNN training
DATASET_DIR=$HOME/temp/dataset

RES="640x360"

# ImageMagic convert filters: "lanczos", "bilinear", "nearestneighbor", "box", "cubic", "mitchell"
OPS="-filter lanczos -filter mitchell -background black -gravity center -strip"

count=1
total=$(find $RAW_DIR/ -type f |wc -l)

find $RAW_DIR/ -type f -print0 |\
while IFS= read -d '' img; do
  prefix="$(printf '%05d' $count)"


  # create a good unblured dataset example
  echo "Creating $DATASET_DIR/${prefix}-good.png"
  convert "$img" -resize $RES $OPS -extent $RES "$DATASET_DIR/${prefix}-good.png"

  echo "Creating $DATASET_DIR/${prefix}-bad.png"
  # scaling down good image to 25%
  convert "$DATASET_DIR/${prefix}-good.png" -resize 50% $OPS "$DATASET_DIR/${prefix}-bad.png"
  # scale up bad image to 400% - as a blurred example
  convert "$DATASET_DIR/${prefix}-bad.png" -resize 200% $OPS "$DATASET_DIR/${prefix}-bad.png"

  echo "${count} of ${total} done"
  
  ((count++))
done

echo "Done. Dataset is ready !"

