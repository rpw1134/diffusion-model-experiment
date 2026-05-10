import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from diffusion_model_experiment.dataset import generate_dataset


def visualize_sample(x: torch.Tensor, title: str = "Sample from Distribution") -> None:
    """Scatter plot a single (N, 2) point cloud."""
    pts = x.detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6)
    plt.title(title)
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.show()


def visualize_samples(xs: list[torch.Tensor], titles: list[str] | None = None) -> None:
    """Plot multiple (N, 2) point clouds side by side."""
    n = len(xs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, (ax, x) in enumerate(zip(axes, xs)):
        pts = x.detach().cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6)
        ax.set_title(titles[i] if titles else f"Sample {i}")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def save_gif(snapshots: list[torch.Tensor], labels: list[str], path: str = "diffusion.gif", fps: int = 20, hold_last_ms: int = 2000) -> None:
    """Save a GIF of 2D point cloud snapshots. Holds the last frame for `hold_last_ms` ms."""
    all_pts = [s.detach().cpu().numpy() for s in snapshots]
    all_x = [p[:, 0] for p in all_pts]
    all_y = [p[:, 1] for p in all_pts]
    xlim = (min(x.min() for x in all_x), max(x.max() for x in all_x))
    ylim = (min(y.min() for y in all_y), max(y.max() for y in all_y))

    frame_ms = 1000 // fps
    durations = [frame_ms] * len(snapshots)
    durations[-1] = hold_last_ms

    pil_frames = []
    for pts, label in zip(all_pts, labels):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6)
        ax.set_title(label)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=durations,
    )
    print(f"Saved to {path}")


def visualize_mnist(images: np.ndarray, title: str = "MNIST Samples") -> None:
    """Display a batch of grayscale images as a grid. Expects values in [0, 1]."""
    n = len(images)
    ncols = min(n, 8)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_mnist_sample(x: torch.Tensor, title: str = "MNIST Sample") -> None:
    """Display model output as a grid. Expects (N, 1, H, W) tensor in [-1, 1]."""
    imgs = x.detach().cpu()
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    imgs = np.clip((imgs.squeeze(1).numpy() + 1) / 2, 0, 1)  # [-1, 1] → [0, 1]
    visualize_mnist(imgs, title=title)


def save_mnist_gif(snapshots: list[torch.Tensor], labels: list[str], path: str = "mnist_diffusion.gif", fps: int = 20, hold_last_ms: int = 2000, n_show: int = 16) -> None:
    """
    Save a GIF of the MNIST reverse diffusion process.

    snapshots: list of (N, 1, H, W) tensors in [-1, 1], ordered noise → data
    n_show: how many images to display per frame (max 16)
    """
    frame_ms = 1000 // fps
    durations = [frame_ms] * len(snapshots)
    durations[-1] = hold_last_ms

    pil_frames = []
    for snapshot, label in zip(snapshots, labels):
        imgs = np.clip((snapshot.detach().cpu().squeeze(1).numpy()[:n_show] + 1) / 2, 0, 1)
        n = len(imgs)
        ncols = min(n, 8)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
        axes = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes):
            if i < n:
                ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        fig.suptitle(label)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=durations,
    )
    print(f"Saved to {path}")


if __name__ == "__main__":
    samples = generate_dataset()
    visualize_sample(samples)
