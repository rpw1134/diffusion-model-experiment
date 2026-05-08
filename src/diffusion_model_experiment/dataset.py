from sklearn.datasets import make_moons
import torch
from torch.utils.data import Dataset
import struct
import numpy as np

def generate_dataset(n_samples=5000, noise=0.05):
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    Y = (X - X.mean(axis=0)) / X.std(axis=0)
    return torch.from_numpy(Y).float()

def load_ubyte_images(filename):
    with open(filename, 'rb') as f:
        # Read header: magic number, number of images, rows, and columns
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read image data as unsigned bytes (ubyte)
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        return data.astype(np.float32)

def load_ubyte_labels(filename):
    with open(filename, 'rb') as f:
        # Read header: magic number and number of items
        magic, num_items = struct.unpack(">II", f.read(8))
        # Read label data
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def load_mnist():
    images = torch.from_numpy(load_ubyte_images("data/mnist/train-images.idx3-ubyte")).float() / 255.0
    labels = torch.from_numpy(load_ubyte_labels("data/mnist/train-labels.idx1-ubyte")).float()
    return images, labels


class DiffusionDataset(Dataset):
    def __init__(self):
        self.points = generate_dataset()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

class MNISTDataset(Dataset):
    def __init__(self):
        self.images, self.labels = load_mnist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

if __name__ == "__main__":
    from diffusion_model_experiment.visualize import visualize_mnist
    images, labels = load_mnist()
    dataset = images.numpy()
    print(f"shape: {dataset.shape}, min: {dataset.min():.3f}, max: {dataset.max():.3f}")
    print(f"Labels 1-16: {labels[:16].numpy()}")
    visualize_mnist(dataset[:16], title="MNIST Training Samples")