from surrogate import MLPSurrogate, CNNSurrogate, predict
import numpy as np
import torch
from scipy.io import loadmat
import utils.dataset

normalisation_params = np.load('./data/normalisation_params.npz')
mean = normalisation_params['mean']
std = normalisation_params['std']

X = loadmat('./data/x5000c.mat')['A']
y = loadmat('./data/y5000c.mat')['zar']

X = X.reshape(-1, 4, 4)  # Reshape to (N, C, H, W) format for CNN input
dataset = utils.dataset.EMDesignDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

random_index = np.random.randint(0, len(val_dataset))
print(f"Randomly selected design index: {random_index}")
print(f"Design matrix (X) for index {random_index}:\n{val_dataset[random_index][0].numpy()}")
print(f"Performance metrics (y) for index {random_index}:\n{val_dataset[random_index][1].numpy()}")

x = val_dataset[random_index][0]  # Add batch dimension
print(x)

# model = CNNSurrogate(input_channels=1, output_dim=2)
model = MLPSurrogate(input_dim=16, output_dim=2)
model.load_state_dict(torch.load('./models/cnn_surrogate.pth'))
model.eval()
pred = predict(model, x, mean, std)
print(f"Predicted performance metrics: {pred}")