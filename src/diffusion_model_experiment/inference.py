import torch

from diffusion_model_experiment.dataset import generate_dataset
from diffusion_model_experiment.model import DiffusionModel, MnistDiffusionUNet
from diffusion_model_experiment.schedule import generate_schedule
from diffusion_model_experiment.visualize import visualize_sample, visualize_samples, save_gif, save_mnist_gif, visualize_mnist_sample


def _load_model(device, path="diffusion_model.pth"):
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _load_mnist_model(device, path="mnist_diffusion_model.pth"):
    model = MnistDiffusionUNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _reverse_loop(model, initial, T, device):
    """
    Runs the DDPM reverse process from `initial` down to t=0.
    Works for any sample shape — betas/alphas are scalars and broadcast.

    Args:
        model:   noise-prediction model, callable as model(x_t, t_tensor)
        initial: starting noise tensor, shape (N, ...) already on device
        T:       number of diffusion timesteps
        device:  torch device

    Returns:
        final sample tensor, same shape as `initial`
    """
    current = initial
    betas, alphas, cumulative_alphas = [s.to(device) for s in generate_schedule(T=T)]
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            t_tensor = torch.full((current.shape[0],), t, device=device)
            prediction = model(current, t_tensor)
            additive_noise = torch.sqrt(betas[t]) * torch.randn_like(current) if t > 0 else 0
            current = (1 / torch.sqrt(alphas[t])) * (current - (betas[t] / torch.sqrt(1 - cumulative_alphas[t])) * prediction) + additive_noise
    return current


def _reverse_loop_with_snapshots(model, initial, T, device, num_snapshots):
    """
    Same as _reverse_loop but captures evenly-spaced snapshots for visualization.

    Returns:
        (final, snapshots, labels)
        snapshots: list of tensors ordered t=T → t=0 (noise → data)
        labels:    matching list of strings like "t=999"
    """
    current = initial
    betas, alphas, cumulative_alphas = [s.to(device) for s in generate_schedule(T=T)]
    checkpoints = set(range(0, T, max(1, T // num_snapshots)))
    checkpoints.add(T - 1)
    snapshots, labels = [], []
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            t_tensor = torch.full((current.shape[0],), t, device=device)
            prediction = model(current, t_tensor)
            additive_noise = torch.sqrt(betas[t]) * torch.randn_like(current) if t > 0 else 0
            current = (1 / torch.sqrt(alphas[t])) * (current - (betas[t] / torch.sqrt(1 - cumulative_alphas[t])) * prediction) + additive_noise
            if t in checkpoints:
                snapshots.append(current.clone())
                labels.append(f"t={t}")
    snapshots.reverse()
    labels.reverse()
    return current, snapshots, labels


# --- 2D model ---

def sample(num_points=1000, T=1000):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_model(device)
    initial = torch.randn((num_points, 2), device=device)
    return _reverse_loop(model, initial, T, device)


def sample_with_snapshots(num_points=1000, T=1000, num_snapshots=5):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_model(device)
    initial = torch.randn((num_points, 2), device=device)
    return _reverse_loop_with_snapshots(model, initial, T, device, num_snapshots)


# --- MNIST model ---

def sample_mnist(num_images=16, T=1000):
    """
    Sample from the trained MNIST UNet.

    Returns:
        tensor of shape (num_images, 1, 28, 28), pixel values roughly in [0, 1]
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_mnist_model(device)
    initial = torch.randn((num_images, 1, 28, 28), device=device)
    return _reverse_loop(model, initial, T, device)


def sample_mnist_with_snapshots(num_images=16, T=1000, num_snapshots=50, path="mnist_diffusion_model_large.pth"):
    """
    Sample from the trained MNIST UNet, capturing snapshots for a GIF.

    Returns:
        (final, snapshots, labels)
        final:     (num_images, 1, 28, 28)
        snapshots: list of (num_images, 1, 28, 28) tensors, noise → data order
        labels:    matching list of strings
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_mnist_model(device, path=path)
    initial = torch.randn((num_images, 1, 28, 28), device=device)
    return _reverse_loop_with_snapshots(model, initial, T, device, num_snapshots)


if __name__ == "__main__":
    final, snapshots, labels = sample_mnist_with_snapshots(num_images=4, T=1000, num_snapshots=200)
    steady_state = snapshots[0]                                    # cleanest frame (t≈0)
    gif_frames = snapshots[::-1] + [steady_state] * 40            # noisy → clean, then hold
    gif_labels = labels[::-1] + ["t=0"] * 40
    save_mnist_gif(gif_frames, gif_labels, path="mnist_diffusion.gif", fps=20)
    visualize_mnist_sample(final, title="Final samples")
