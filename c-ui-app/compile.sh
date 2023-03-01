#!/bin/bash
set -ex

gcc -o ~/temp/libsnake.so game.c xwrapper.c -lX11 -pthread -shared -fPIC #-fvisibility=hidden

echo "List all exported functions:"
nm -D ~/temp/libsnake.so
