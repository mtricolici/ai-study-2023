#!/bin/bash
set -ex

ofile=/tmp/libsnake.so

rm -f $ofile

gcc -o $ofile game.c xwrapper.c -lX11 -pthread -shared -fPIC #-fvisibility=hidden

echo "List all exported functions:"
nm -D $ofile
