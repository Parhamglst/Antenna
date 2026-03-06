from surrogate import MLPSurrogate, CNNSurrogate, predict
import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
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

# Load model
# model = CNNSurrogate(input_channels=1, output_dim=2)
model = MLPSurrogate(input_dim=16, output_dim=2)
model.load_state_dict(torch.load('./models/cnn_surrogate.pth'))
model.eval()

# Collect all validation data
true_y = []
pred_y = []

for idx in range(len(val_dataset)):
    x, y_true = val_dataset[idx]
    true_y.append(y_true.numpy())
    
    # Make prediction for this sample
    pred = predict(model, x.unsqueeze(0).numpy(), mean, std)
    pred_y.append(pred[0])

true_y = np.array(true_y)
pred_y = np.array(pred_y)

print(f"Validation set size: {len(val_dataset)}")
print(f"True y shape: {true_y.shape}")
print(f"Predicted y shape: {pred_y.shape}")

# Calculate Pearson Correlation Coefficient for each output dimension
pcc_1 = pearsonr(true_y[:, 0], pred_y[:, 0])[0]
pcc_2 = pearsonr(true_y[:, 1], pred_y[:, 1])[0]

# Calculate Mean Absolute Error for each output dimension
mae_1 = np.mean(np.abs(true_y[:, 0] - pred_y[:, 0]))
mae_2 = np.mean(np.abs(true_y[:, 1] - pred_y[:, 1]))

print(f"\nPearson Correlation Coefficient (Output 1): {pcc_1:.4f}")
print(f"Pearson Correlation Coefficient (Output 2): {pcc_2:.4f}")
print(f"\nMean Absolute Error (Output 1): {mae_1:.4f}")
print(f"Mean Absolute Error (Output 2): {mae_2:.4f}")

# Plot PCC graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: True vs Predicted for Output 1
axes[0].scatter(true_y[:, 0], pred_y[:, 0], alpha=0.6, s=30)
min_val = min(true_y[:, 0].min(), pred_y[:, 0].min())
max_val = max(true_y[:, 0].max(), pred_y[:, 0].max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('True y (Output 1)', fontsize=12)
axes[0].set_ylabel('Predicted y (Output 1)', fontsize=12)
axes[0].set_title(f'Output 1 - PCC: {pcc_1:.4f}', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: True vs Predicted for Output 2
axes[1].scatter(true_y[:, 1], pred_y[:, 1], alpha=0.6, s=30)
min_val = min(true_y[:, 1].min(), pred_y[:, 1].min())
max_val = max(true_y[:, 1].max(), pred_y[:, 1].max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
axes[1].set_xlabel('True y (Output 2)', fontsize=12)
axes[1].set_ylabel('Predicted y (Output 2)', fontsize=12)
axes[1].set_title(f'Output 2 - PCC: {pcc_2:.4f}', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()