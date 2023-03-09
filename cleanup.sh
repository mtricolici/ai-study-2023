#!/bin/bash

find .|grep -E "(/__pycache__$|\.pyc$|\.pyo$)"|xargs -I{} rm -rf {}
