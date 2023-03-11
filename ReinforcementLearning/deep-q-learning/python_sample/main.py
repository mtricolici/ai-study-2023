#!/usr/bin/env python
import os
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "" # use CPU only

import sys
from demo_game import demo

from training import train, train_continue

def show_usage():
    print("Usage examples:")
    print(f" {sys.argv[0]} train -> invokes new training")
    print(f" {sys.argv[0]} train continue -> loads from file and continue a training")
    print(f" {sys.argv[0]} demo -> play a demo game")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train(8, 1000, "/output/deepql.keras.zzz")
    elif len(sys.argv) == 3 and sys.argv[1] == "train" and sys.argv[2] == "continue":
        train_continue()
    elif len(sys.argv) == 2 and sys.argv[1] == "demo":
        demo(10, "/output/deepql.keras.zzz")
    else:
        show_usage()
