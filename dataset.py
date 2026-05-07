import numpy as np
from sklearn.datasets import make_moons
import torch


def generate_sample(n_samples=5000, noise=0.1):
    sample = make_moons(n_samples=n_samples, noise=noise)
    return sample[0]

def generate_dataset(n_samples=5000, sample_len=5000, noise=0.1):
    # sample, each of a length, each with x and y
    arr = np.zeros((n_samples, sample_len, 2))
    for i in range(n_samples):
        arr[i, :, :] = generate_sample(n_samples, noise=noise)
    return torch.from_numpy(arr).float()

if __name__ == '__main__':
    sample = generate_dataset()
    print(sample[0])