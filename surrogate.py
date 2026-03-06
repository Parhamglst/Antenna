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

class MLPSurrogate(nn.Module):
    """
    MLP model for EM design. 
    Input: (N, 1, 4, 4) or (N, 16)
    Output: (N, 2)
    """
    def __init__(self, input_dim=16, output_dim=2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.network(x)

 
def train_model(model, train_loader, val_loader, y_mean, y_std, epochs=200, lr=1e-3, device='cpu'):
    model = model.to(device)
    
    # criterion = nn.MSELoss(reduction='none')
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=20, min_delta=1e-4)

    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss_per_dim = loss.mean(dim=0)  
            batch_loss = loss_per_dim.mean()  
            
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_loss_unaveraged = 0.0  # Accumulator for per-dimension loss

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                outputs_denorm = outputs * y_std_t + y_mean_t
                targets_denorm = targets * y_std_t + y_mean_t
                
                loss = criterion(outputs_denorm, targets_denorm)
                batch_loss_unaveraged = loss.mean(dim=0) # Shape: (2,)
                batch_loss_averaged = batch_loss_unaveraged.mean() # Scalar
                
                val_loss += batch_loss_averaged.item()
                val_loss_unaveraged += batch_loss_unaveraged # Accumulates correctly now

        # Average out over the number of validation batches
        val_loss /= len(val_loader)
        val_loss_unaveraged /= len(val_loader)
        
        es, model_state = early_stopping.step(val_loss, val_loss_unaveraged, model)
        if es:
            print(f"Early stopping at epoch {epoch+1}")
            # Ensure tensor is moved to CPU for clean printing if it was on GPU
            unaveraged_print = early_stopping.best_loss_unaveraged.cpu().numpy() if torch.is_tensor(early_stopping.best_loss_unaveraged) else early_stopping.best_loss_unaveraged
            print(f"Best validation loss: {unaveraged_print} (Unaveraged), {early_stopping.best_loss:.6f} (Averaged)")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f} "
                  f"Val Loss: {val_loss:.6f}")
            
    torch.save(model_state, MODEL_PATH)


def predict(model, X, mean, std, device='cpu'):
    """
    Predict performance metrics for new designs using the trained surrogate model.
        X: numpy array (N, H, W) - New design matrices
        mean: numpy array (2,) - Mean used for normalizing outputs during training
        std: numpy array (2,) - Std used for normalizing outputs during training
        
        Returns: numpy array (N, 2) - Predicted performance metrics (denormalized)
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X[:, None, :, :], dtype=torch.float32).to(device)
        # Added .cpu() to safely handle GPU models before converting to numpy
        predictions = model(inputs).cpu().numpy()
        predictions = predictions * std + mean  # Denormalize predictions
    return predictions