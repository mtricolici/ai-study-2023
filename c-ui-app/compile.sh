#!/bin/bash
set -ex

gcc -o ~/temp/libapp.so main.c game.c -lX11 -pthread -shared -fPIC #-fvisibility=hidden -fPIC

echo "List all exported functions:"
nm -D ~/temp/libapp.so
