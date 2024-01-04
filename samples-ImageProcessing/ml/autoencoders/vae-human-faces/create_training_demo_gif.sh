#!/bin/bash
set -e

rm -f /tmp/epoch-*.jpg

for fn in content/epoch-*.jpg; do
  bn=$(basename "$fn") # remove path
  epoch="${bn##*-}" # Remove everything up to the last hyphen
  epoch="${epoch%.*}" # remove extension
  epoch=$(awk '{printf("%d", $1)}' <<< "$epoch") # Remove leding zeros

  convert "$fn" -fill white -pointsize 36 -annotate +30+30 "epoch $epoch" "/tmp/$bn"
done

convert -delay 100 -loop 0 /tmp/epoch-*.jpg content/training-animated.gif
