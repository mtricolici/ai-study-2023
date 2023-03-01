#!/bin/bash
set -ex

gcc -o ~/temp/libapp.so main.c game.c xwrapper.c -lX11 -pthread -shared -fPIC #-fvisibility=hidden

echo "List all exported functions:"
nm -D ~/temp/libapp.so
