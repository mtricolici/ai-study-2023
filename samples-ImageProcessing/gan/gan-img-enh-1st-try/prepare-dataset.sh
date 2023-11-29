#!/bin/bash
set -e

# Path with images of different sizes
RAW_DIR=$HOME/temp/raw-images

# output folder to save dataset ready for GAN
DATASET_DIR=$HOME/temp/gan-dataset

#resolutions=("176x144" "256x192" "320x180" "320x240" "420x240" ) #"640x360" "854x480")
resolutions=("160x90")

RES="?"
PREFIX="?"

# ImageMagic convert filters: "lanczos", "bilinear", "nearestneighbor", "box", "cubic", "mitchell"
OPS="-filter lanczos -filter mitchell -background black -gravity center -strip"

####################################################
function select_random_resolution() {
  local ri=$((RANDOM % ${#resolutions[@]}))
  local el="${resolutions[$ri]}"
  local parts=(${el//x/ })
  local w=${parts[0]}
  local h=${parts[1]}

  RES="${w}x${h}"
  PREFIX="$ri-"
}
####################################################

count=1
total=$(find $RAW_DIR/ -type f |wc -l)

find $RAW_DIR/ -type f -print0 |\
while IFS= read -d '' img; do
  select_random_resolution

  oname="$(printf '%05d' $count).png"

  # create a good unblured dataset example
  echo "Creating $DATASET_DIR/${PREFIX}good-$oname"
  convert "$img" -resize $RES $OPS -extent $RES "$DATASET_DIR/${PREFIX}good-$oname"

  echo "Creating $DATASET_DIR/${PREFIX}bad-$oname"
  # scaling down good image to 25%
  convert "$DATASET_DIR/${PREFIX}good-$oname" -resize 25% $OPS "$DATASET_DIR/${PREFIX}bad-$oname"
  # scale up bad image to 400% - as a blurred example
  convert "$DATASET_DIR/${PREFIX}bad-$oname" -resize 400% $OPS "$DATASET_DIR/${PREFIX}bad-$oname"

  echo "${count} of ${total} done"
  
  ((count++))
done

echo "Done. Dataset is ready !"

