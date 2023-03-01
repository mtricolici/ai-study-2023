#!/bin/bash

lp="$(pwd)/../../c-ui-app/"

lp=$(realpath "$lp")

go run -exec "env LD_LIBRARY_PATH=$lp" main.go
