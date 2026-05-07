from sklearn.datasets import make_moons
import torch


def generate_dataset(n_samples=5000, noise=0.05):
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    return torch.from_numpy(X).float()
