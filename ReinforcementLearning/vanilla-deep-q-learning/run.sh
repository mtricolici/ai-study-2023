#!/bin/bash

go run -exec "env LD_LIBRARY_PATH=/tmp" main.go "$@"
