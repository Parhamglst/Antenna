import numpy as np
import torch

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss doesn't improve.
    """
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_model_state = None
        self.best_loss_unaveraged = None

    def step(self, loss, batch_loss, model):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_loss_unaveraged = batch_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1

        return self.counter >= self.patience, self.best_model_state