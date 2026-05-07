from sklearn.datasets import make_moons
import torch


def generate_sample():
    sample = make_moons(n_samples=5000, noise=0.05)
    return torch.from_numpy(sample[0]).float()

if __name__ == '__main__':
    sample = generate_sample()