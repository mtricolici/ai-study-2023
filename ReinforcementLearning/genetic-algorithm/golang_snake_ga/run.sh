#!/bin/bash

if [ ! -f /tmp/libsnake.so ]; then
  echo "File /tmp/libsnake.so not found!"
  echo "cd to ../../../c-ui-app/ and invoke ./compile.sh !"
  exit 1
fi

go run -exec "env LD_LIBRARY_PATH=/tmp" main.go "$@"
