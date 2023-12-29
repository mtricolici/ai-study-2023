import os
import sys
import numpy as np
import tensorflow as tf

from image import load_image

class DataSet:
#########################################################
    def __init__(self, batch_size, split_ratio=0.9):
        self.batch_size  = batch_size
        self.split_ratio = split_ratio
        self._load()
#########################################################
    def _load(self):
        files = [os.path.join('/dataset', f) for f in os.listdir('/dataset') if f.endswith(".png")]

        num_files = len(files)
        if num_files == 0:
            print("NO Files in dataset???")
            sys.exit(1)

        split_index = int(num_files * self.split_ratio)

        self.train_files = files[:split_index]
        self.valid_files = files[split_index:]

        # round up. for 18.6 will be 19

        self.validation_steps = int(tf.math.ceil(len(self.valid_files) / self.batch_size).numpy())

        print("#DataSet:")
        print(f"training-samples   : {len(self.train_files)}")
        print(f"validation-samples : {len(self.valid_files)}")
        print(f"batch-size         : {self.batch_size}")
        print(f"validation-steps   : {self.validation_steps}")
#########################################################
    def generate_batch(self, indices, files):
        batch_input = []
        batch_output = []

        for idx in indices:
            batch_input.append(load_image(files[idx]))
            if files[idx].endswith('good.png'):
                batch_output.append([1.0, 0.0])
            else:
                batch_output.append([0.0, 1.1])

        batch_x = tf.stack(batch_input)
        batch_y = tf.stack(batch_output)

        return (batch_x, batch_y)
#########################################################
    def train_data(self):
        num_files = len(self.train_files)

        while True:
            indices = np.random.choice(num_files, self.batch_size)
            yield self.generate_batch(indices, self.train_files)
#########################################################
    def validation_data(self):
        num_files = len(self.valid_files)

        while True:
            indices = np.random.choice(num_files, self.batch_size)
            yield self.generate_batch(indices, self.valid_files)
#########################################################



