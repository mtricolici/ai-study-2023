import tensorflow as tf
import numpy as np
from tensorflow.keras import metrics

from helper import lm
from image import save_image

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
        self.best_loss  = float('inf')
        self.best_ep    = -1
        self.bad_epochs = 0
        self.early_stop_count = 0

        self.random_samples_dim = np.random.normal(size=(1, self.vae.latent_dim))

        lm(f'MODEL-TYPE: {self.vae.model_type.upper()}')

        if self.vae.model_type in ('cnn', 'mlp'):
            lm(f'depths    : {self.vae.depths}')

        lm(f'latent-dim: {self.vae.latent_dim}')
        lm(f'batch-size: {self.vae.batch_size}')
        lm(f'learn-rate: {optimizer.learning_rate.numpy():.2e}')

####################################################################################
    def on_step_end(self, ep, step, loss, kl_loss, reconstruction_loss):
        self.metrics['loss'](loss)
        self.metrics['kl_loss'](kl_loss)
        self.metrics['rec_loss'](reconstruction_loss)

        l1 = self.metrics['loss'].result()
        l2 = self.metrics['kl_loss'].result()
        l3 = self.metrics['rec_loss'].result()

        ls = f'loss: {l1:.6f} kl: {l2:.6f} rl: {l3:.6f}'
        perc = step / self.vae.steps_per_epoch * 100.0
        lr = self.optimizer.learning_rate.numpy()

        lm(f'>>> ep {ep}: {perc:.0f}% [step {step} of {self.vae.steps_per_epoch}] {ls} lr={lr:.2e}', "\r")
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
    def generate_some_samples(self, epoch):
        samples = self.vae.sample(self.random_samples_dim)
        for i, img in enumerate(samples):
            path = f'/content/ep{epoch:03d}-s{i+1}.png'
            save_image(img, path)
####################################################################################
    def reset_metrics(self):
        for _, m in self.metrics.items():
            m.reset_states()
####################################################################################
    def on_epoch_end(self, epoch):
       loss, loss_s = self._get_loss() 

       # We don't need metrics from previous epoches! just remove the history!
       self.reset_metrics()
       lr = self.optimizer.learning_rate.numpy()

       improvement = "+" if loss < self.best_loss else "-"

       lm(f"Epoch {epoch}/{self.vae.epochs} {improvement}{loss_s} lr: {lr:.2e}           ")

       if loss < self.best_loss:
          # Epoch with loss improvement !!!
          self.best_loss = loss
          self.best_ep = epoch

          self.vae.save_model()
          self.generate_some_samples(epoch)

          self.early_stop_count = 0
          self.bad_epochs = 0
       else:
          # Epoch without improvement :(
          self.bad_epochs += 1
          self.early_stop_count += 1

          if self.early_stop_count >= self.vae.early_stop:
              # Training must stop
              lm(f'Early stop. No improvements for {self.early_stop_count} epoches.')
              return True

          if self.bad_epochs > self.vae.learning_rate_patience:
#              lm(f'... Restoring best weights from epoch {self.best_ep} ...')
#              self.vae.load_model()
              self.bad_epochs = 0

              # Decrease learning rate
              new_lr = self.optimizer.learning_rate.numpy() * self.vae.learning_rate_decrease_factor
              if new_lr < self.vae.minimum_learning_rate:
                  new_lr = self.vae.minimum_learning_rate
              self.optimizer.learning_rate = new_lr

       # Training must continue
       return False

####################################################################################

