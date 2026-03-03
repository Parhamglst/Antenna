import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.tools import EarlyStopping

MODEL_PATH = './models/cnn_surrogate.pth'

class CNNSurrogate(nn.Module):
    """
    A simple CNN surrogate model for predicting EM design performance.
    Input: (N, 1, 4, 4) - N samples of 4x4 design matrices
    Output: (N, 2) - N samples of 2 performance metrics (e.g., gain and bandwidth)
    """
    def __init__(self, input_channels=1, output_dim=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),          # 32 * 4 * 4 = 512
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

 
def train_model(model, train_loader, val_loader, y_mean, y_std, epochs=200, lr=1e-3):
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=20, min_delta=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss.mean(dim=0)  # Average over batch
            loss_per_dim = loss.mean()  # Average over output dimensions
            loss_per_dim.backward()
            optimizer.step()
            train_loss += loss_per_dim.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                outputs_denorm = outputs * y_std + y_mean
                targets_denorm = targets * y_std + y_mean
                loss = criterion(outputs_denorm, targets_denorm)
                batch_loss = loss.mean(dim=0)  # Average over batch
                loss_per_dim = batch_loss.mean()  # Average over output dimensions
                val_loss += loss_per_dim.item()
                batch_loss += batch_loss

        val_loss /= len(val_loader)
        batch_loss /= len(val_loader)
        es, model_state = early_stopping.step(val_loss, batch_loss, model)
        if es:
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopping.best_loss_unaveraged} (Unaveraged), {early_stopping.best_loss:.6f} (Averaged)")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f} "
                  f"Val Loss: {val_loss:.6f}")
    torch.save(model_state, MODEL_PATH)


def predict(model, X, mean, std):
    """
    Predict performance metrics for new designs using the trained surrogate model.
        X: numpy array (N, H, W) - New design matrices
        mean: numpy array (2,) - Mean used for normalizing outputs during training
        std: numpy array (2,) - Std used for normalizing outputs during training
        
        Returns: numpy array (N, 2) - Predicted performance metrics (denormalized)
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X[:, None, :, :], dtype=torch.float32)
        predictions = model(inputs).numpy()
        predictions = predictions * std + mean  # Denormalize predictions
    return predictions