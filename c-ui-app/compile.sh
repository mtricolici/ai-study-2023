#!/bin/bash
set -ex

rm -f libsnake.so

gcc -o libsnake.so game.c xwrapper.c -lX11 -pthread -shared -fPIC #-fvisibility=hidden

echo "List all exported functions:"
nm -D libsnake.so
