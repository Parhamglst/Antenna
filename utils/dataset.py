import torch
from torch.utils.data import Dataset


class EMDesignDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array (N, H, W)
        y: numpy array (N, 1)
        """
        self.X = torch.tensor(X[:, None, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def normalise_outputs(y):
    """
    Z-score normalise the outputs.
    """
    mean = y.mean(axis=0)
    std = y.std(axis=0) + 1e-8  # Avoid division by zero
    return (y - mean) / std, mean, std