import surrogate
from utils.dataset import EMDesignDataset, normalise_outputs
from scipy.io import loadmat
import numpy as np
import torch


INPUT_DIM = (4, 4)
X_PATH = './data/x5000c.mat'
Y_PATH = './data/y5000c.mat'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = loadmat(X_PATH)['A']
y = loadmat(Y_PATH)['zar']
X = X.reshape(-1, *INPUT_DIM)  # Reshape to (N, H, W) format
y, mean, std = normalise_outputs(y)
with open('./data/normalisation_params.npz', 'wb') as f:
    np.savez(f, mean=mean, std=std)
    print(f"Normalisation parameters: mean={mean}, std={std}")
dataset = EMDesignDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
# model = surrogate.CNNSurrogate(input_channels=1, output_dim=2)
model = surrogate.MLPSurrogate(input_dim=16, output_dim=2)
surrogate.train_model(model, train_loader, val_loader, mean, std, epochs=300, lr=3e-4, device=device)
