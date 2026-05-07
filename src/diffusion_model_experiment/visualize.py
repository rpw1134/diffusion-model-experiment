import matplotlib.pyplot as plt
import torch
from diffusion_model_experiment.dataset import generate_dataset


def visualize_sample(x: torch.Tensor, title: str = "Sample") -> None:
    pts = x.detach().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6)
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def visualize_samples(xs: list[torch.Tensor], titles: list[str] | None = None) -> None:
    n = len(xs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, (ax, x) in enumerate(zip(axes, xs)):
        pts = x.detach().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6)
        ax.set_title(titles[i] if titles else f"Sample {i}")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    samples = generate_dataset()
    visualize_samples(samples[0:5])

