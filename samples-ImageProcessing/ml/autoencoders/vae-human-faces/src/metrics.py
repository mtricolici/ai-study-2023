import tensorflow as tf

####################################################################################
class Metrics:
####################################################################################
    def __init__(self):
        self.metrics = {
            'loss'     : tf.keras.metrics.Mean(),
            'kl_loss'  : tf.keras.metrics.Mean(),
            'rec_loss' : tf.keras.metrics.Mean()
        }
####################################################################################
    def collect(self, loss, kl_loss, reconstruction_loss):
        self.metrics['loss'](loss)
        self.metrics['kl_loss'](kl_loss)
        self.metrics['rec_loss'](reconstruction_loss)
####################################################################################
    def loss(self):
        return self.metrics['loss'].result()
####################################################################################
    def as_string(self):
        l1 = self.metrics['loss'].result()
        l2 = self.metrics['kl_loss'].result()
        l3 = self.metrics['rec_loss'].result()
        return f'loss: {l1:.6f} kl: {l2:.6f} rl: {l3:.6f}'
####################################################################################
