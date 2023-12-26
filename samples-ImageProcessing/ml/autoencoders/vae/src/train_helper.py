import tensorflow as tf
import numpy as np
from tensorflow.keras import metrics

from helper import lm

####################################################################################
class TrainHelper:
####################################################################################
    def __init__(self, vae):
        self.vae = vae
        self.optimizer = None
####################################################################################
    def training_start(self, optimizer):
        self.optimizer = optimizer
        self.metrics = {
            'loss': metrics.Mean(),
            'kl_loss': metrics.Mean(),
            'rec_loss': metrics.Mean()
        }
        self.best_loss = float('inf')

        lm(f'latent-dim: {self.vae.latent_dim}')
        lm(f'depths    : {self.vae.depths}')
        lm(f'batch-size: {self.vae.batch_size}')
        lm(f'learn-rate: {optimizer.learning_rate.numpy():.2e}')

####################################################################################
    def on_step_end(self, ep, step, loss, kl_loss, reconstruction_loss):
        self.metrics['loss'](loss)
        self.metrics['kl_loss'](kl_loss)
        self.metrics['rec_loss'](reconstruction_loss)


        ls = f'loss: {loss:.6f} kl: {kl_loss:.6f} rl: {reconstruction_loss:.6f}'
        perc = step / self.vae.steps_per_epoch * 100.0

        lm(f'>>> ep {ep}: {perc:.0f}% [step {step} of {self.vae.steps_per_epoch}] {ls}', "\r")
####################################################################################
    def _get_loss(self):
        ls = []
        for k, m in self.metrics.items():
            v = m.result()
            if k == 'loss':
                loss = v
            ls.append(f'{k}: {v:.6f}')
        return loss, " ".join(ls)
####################################################################################
    def on_epoch_end(self, epoch):
       loss, loss_s = self._get_loss() 
       lr = self.optimizer.learning_rate.numpy()

       lm(f"Epoch {epoch}/{self.vae.epochs} {loss_s} lr: {lr:.2e}")

       # Save best weights only
       if loss < self.best_loss:
          self.best_loss = loss
          self.vae.save_model()

####################################################################################

