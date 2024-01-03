import os
import sys
import numpy as np
import tensorflow as tf

from image import load_image
from helper import lm

#########################################################
class DataSet:
#########################################################
    def __init__(self, batch_size, split_ratio=0.9):
        self.folder = '/dataset'
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self._load_and_split()
#########################################################
    def _find_files(self):
        files = []
        for root, _, filenames in os.walk(self.folder):
            for filename in filenames:
                if filename.endswith(".png"):
                    files.append(os.path.join(root, filename))

        return files
#########################################################
    def _load_and_split(self):
        files = self._find_files()
        num_files = len(files)
        if num_files == 0:
            print("NO Files in dataset???")
            sys.exit(1)

        split_idx = int(num_files * self.split_ratio)
        self.train_files = files[:split_idx]
        self.valid_files = files[split_idx:]

        lm(f'Training files   : {len(self.train_files)}')
        lm(f'Validation files : {len(self.valid_files)}')
#########################################################
    def _generate_batch(self, indices, files):
        batch = []

        for idx in indices:
            batch.append(load_image(files[idx]))

        return tf.stack(batch)
#########################################################
    def train_samples(self):
        num_files = len(self.train_files)
        while True:
            indices = np.random.choice(num_files, size=self.batch_size, replace=False)
            yield self._generate_batch(indices, self.train_files)
#########################################################
    def validation_samples(self):
        num_files = len(self.valid_files)
        for start in range(0, num_files, self.batch_size):
            end = min(start + self.batch_size, num_files)
            indices = np.arange(start, end)
            yield self._generate_batch(indices, self.valid_files)
#########################################################

