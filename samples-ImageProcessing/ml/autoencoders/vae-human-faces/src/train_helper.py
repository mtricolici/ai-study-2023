import tensorflow as tf
import numpy as np

from helper import lm
from image import save_images_as_grid

####################################################################################
class TrainHelper:
####################################################################################
    def __init__(self, vae):
        self.vae = vae
####################################################################################
    def training_start(self):
        self.best_loss  = float('inf')
        self.best_ep    = -1
        self.bad_epochs = 0
        self.early_stop_count = 0

        self.random_samples_dim = np.random.normal(size=(28, self.vae.latent_dim))

        lm(f'MODEL-TYPE: {self.vae.model_type.upper()}')

        if self.vae.model_type == 'mlp':
            lm(f'depths    : {self.vae.depths}')

        if self.vae.model_type == 'cnn':
            lm(f'depths    : [{self.vae.f1} {self.vae.f2}]')

        lm(f'latent-dim: {self.vae.latent_dim}')
        lm(f'batch-size: {self.vae.batch_size}')
        lm(f'learn-rate: {self.vae.optimizer.learning_rate.numpy():.2e}')

####################################################################################
    def on_step_end(self, ep, step, loss):
        perc = step / self.vae.steps_per_epoch * 100.0
        lr = self.vae.optimizer.learning_rate.numpy()

        lm(f'>>> ep {ep}: {perc:.0f}% [step {step} of {self.vae.steps_per_epoch}] loss:{loss:.5f} lr={lr:.2e}', "\r")
####################################################################################
    def on_validation_step(self, loss):
        lm(f'>> validation loss: {loss:.5f}  ', "\r")
####################################################################################
    def generate_some_samples(self, epoch):
        samples = self.vae.sample(self.random_samples_dim)
        save_images_as_grid(samples, f'/content/epoch-{epoch:03d}.jpg', 7)
####################################################################################
    def on_epoch_end(self, epoch, loss, val_loss):
       lr = self.vae.optimizer.learning_rate.numpy()

       improvement = "+" if val_loss < self.best_loss else "-"

       m = f'loss: {loss:.5f}, val_loss: {val_loss:.5f}'
       lm(f"Epoch {epoch}/{self.vae.epochs} {improvement}{m} lr: {lr:.2e}           ")

       if val_loss < self.best_loss:
          # Epoch with loss improvement !!!
          self.best_loss = val_loss
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
              lm(f'... Restoring best weights from epoch {self.best_ep} ...')
              self.vae.load_model()
              self.bad_epochs = 0

              # Decrease learning rate
              new_lr = self.vae.optimizer.learning_rate.numpy() * self.vae.learning_rate_decrease_factor
              if new_lr < self.vae.minimum_learning_rate:
                  new_lr = self.vae.minimum_learning_rate

              # Create NEW optimizer with smaller LR
              self.vae.create_optimizer(new_lr)

       # Training must continue
       return False

####################################################################################

