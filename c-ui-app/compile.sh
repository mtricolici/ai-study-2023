#!/bin/bash
set -ex
gcc -o ~/temp/libapp.so main.c -lX11 -pthread
