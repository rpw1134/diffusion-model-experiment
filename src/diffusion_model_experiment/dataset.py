from sklearn.datasets import make_moons
import torch
from torch.utils.data import Dataset


def generate_dataset(n_samples=5000, noise=0.05):
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    return torch.from_numpy(X).float()


class DiffusionDataset(Dataset):
    def __init__(self):
        self.points = generate_dataset()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]