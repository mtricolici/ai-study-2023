import tensorflow as tf
import numpy as np
from tensorflow.keras import metrics


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

        print(f'latent-dim: {self.vae.latent_dim}')
        print(f'depths    : {self.vae.depths}')
        print(f'batch-size: {self.vae.batch_size}')
        print(f'learn-rate: {optimizer.learning_rate.numpy():.2e}')

####################################################################################
    def on_step_end(self, ep, step, loss, kl_loss, reconstruction_loss):
        self.metrics['loss'](loss)
        self.metrics['kl_loss'](kl_loss)
        self.metrics['rec_loss'](reconstruction_loss)


        ls = f'loss: {loss:.6f} kl: {kl_loss:.6f} rl: {reconstruction_loss:.6f}'
        perc = step / self.vae.steps_per_epoch * 100.0

        print(f'>>> ep {ep}: {perc:.0f}% [step {step} of {self.vae.steps_per_epoch}] {ls}', end="\r")
####################################################################################
    def _get_loss_str(self):
        loss = []
        for k, m in self.metrics.items():
            loss.append(f'{k}: {m.result():.6f}')
        return " ".join(loss)
####################################################################################
    def on_epoch_end(self, epoch):
       loss = self._get_loss_str() 
       lr = self.optimizer.learning_rate.numpy()

       print(f"Epoch {epoch}/{self.vae.epochs} {loss} lr: {lr:.2e}")
####################################################################################

