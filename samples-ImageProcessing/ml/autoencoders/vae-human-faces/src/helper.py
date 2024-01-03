from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

##############################################################################
def lm(msg, end='\n'):
    tm = datetime.now().strftime("%H:%M:%S")
    print(f'{tm} {msg}', end=end)
##############################################################################
def mse_loss(y_true, y_pred):
    r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return 1000 * r_loss # Why????
##############################################################################
def kl_loss(mean, log_var):
    return -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
##############################################################################
def vae_loss(y_true, y_pred, mean, log_var):
    r_loss = mse_loss(y_true, y_pred)
    kl = kl_loss(mean, log_var)
    return  r_loss + kl
##############################################################################
def _sampling_model(distribution_params):
    mean, log_var = distribution_params
    epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
    return mean + K.exp(log_var / 2) * epsilon
##################################################################################################################
def build_final(vae):
    mean    = tf.keras.Input(shape=(vae.latent_dim,))
    log_var = tf.keras.Input(shape=(vae.latent_dim,))

    out = layers.Lambda(_sampling_model)([mean, log_var])
    return tf.keras.Model([mean,log_var], out)
##################################################################################################################

