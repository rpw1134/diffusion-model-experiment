import matplotlib.pyplot as plt
import torch
from dataset import generate_sample


def visualize_sample(x: torch.Tensor, title: str = "Sample") -> None:
    pts = x.detach().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6)
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample = generate_sample()
    visualize_sample(sample)

