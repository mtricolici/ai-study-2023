#!/bin/bash
set +e

docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -t mykerasimage .

echo $?
