import tensorflow as tf
import datetime

def lm(msg):
  ct = datetime.datetime.now()
  ft = ct.strftime("%H:%M:%S")
  print(f'{ft} {msg}')

def psnr_metric(y_true, y_pred):
  return tf.image.psnr(y_true, y_pred, max_val=1.0)

