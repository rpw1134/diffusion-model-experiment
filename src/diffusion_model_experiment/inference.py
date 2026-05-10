import torch

from diffusion_model_experiment.model import DiffusionModel, MnistDiffusionUNet, NULL_CLASS
from diffusion_model_experiment.schedule import generate_schedule
from diffusion_model_experiment.visualize import save_gif, save_mnist_gif, visualize_mnist_sample


def _load_model(device, path="diffusion_model.pth"):
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _load_mnist_model(device, path="mnist_diffusion_model_conditioned.pth"):
    model = MnistDiffusionUNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _reverse_loop(model, initial, T, device):
    """DDPM reverse process (Algorithm 2). Runs from t=T-1 down to t=0."""
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
    """Like _reverse_loop, but also returns evenly-spaced snapshots ordered noise → data."""
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
    """Run the reverse process and return (num_points, 2) samples."""
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_model(device)
    initial = torch.randn((num_points, 2), device=device)
    return _reverse_loop(model, initial, T, device)


def sample_with_snapshots(num_points=1000, T=1000, num_snapshots=5):
    """Like sample(), but also returns (final, snapshots, labels) for GIF generation."""
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_model(device)
    initial = torch.randn((num_points, 2), device=device)
    return _reverse_loop_with_snapshots(model, initial, T, device, num_snapshots)


# --- MNIST model ---

def _reverse_loop_cfg(model, initial, label, T, device, guidance_scale):
    """
    CFG reverse loop. Runs the model twice per step — conditioned on `label` and
    on the null token — then interpolates:
        ε̂ = ε_uncond + guidance_scale * (ε_cond - ε_uncond)

    guidance_scale=1.0 is equivalent to pure conditioned. Higher values push more
    strongly toward the class at the cost of sample diversity.
    """
    current = initial
    N = current.shape[0]
    betas, alphas, cumulative_alphas = [s.to(device) for s in generate_schedule(T=T)]
    c_cond   = torch.full((N,), label,      device=device, dtype=torch.long)
    c_uncond = torch.full((N,), NULL_CLASS, device=device, dtype=torch.long)
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            t_tensor   = torch.full((N,), t, device=device)
            eps_cond   = model(current, t_tensor, c_cond)
            eps_uncond = model(current, t_tensor, c_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            additive_noise = torch.sqrt(betas[t]) * torch.randn_like(current) if t > 0 else 0
            current = (1 / torch.sqrt(alphas[t])) * (current - (betas[t] / torch.sqrt(1 - cumulative_alphas[t])) * eps) + additive_noise
    return current


def _reverse_loop_cfg_with_snapshots(model, initial, label, T, device, guidance_scale, num_snapshots):
    """Like _reverse_loop_cfg, but also returns evenly-spaced snapshots ordered noise → data."""
    current = initial
    N = current.shape[0]
    betas, alphas, cumulative_alphas = [s.to(device) for s in generate_schedule(T=T)]
    c_cond   = torch.full((N,), label,      device=device, dtype=torch.long)
    c_uncond = torch.full((N,), NULL_CLASS, device=device, dtype=torch.long)
    checkpoints = set(range(0, T, max(1, T // num_snapshots)))
    checkpoints.add(T - 1)
    snapshots, snap_labels = [], []
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            t_tensor   = torch.full((N,), t, device=device)
            eps_cond   = model(current, t_tensor, c_cond)
            eps_uncond = model(current, t_tensor, c_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            additive_noise = torch.sqrt(betas[t]) * torch.randn_like(current) if t > 0 else 0
            current = (1 / torch.sqrt(alphas[t])) * (current - (betas[t] / torch.sqrt(1 - cumulative_alphas[t])) * eps) + additive_noise
            if t in checkpoints:
                snapshots.append(current.clone())
                snap_labels.append(f"t={t}")
    snapshots.reverse()
    snap_labels.reverse()
    return current, snapshots, snap_labels


def sample_mnist(num_images=16, label=None, T=1000, guidance_scale=7.5, path="mnist_diffusion_model_conditioned.pth"):
    """
    Sample from the trained MNIST UNet.

    label: int 0-9 for class-conditioned sampling, or None for unconditional.
    guidance_scale: CFG strength (ignored when label is None).
    Returns a (num_images, 1, 28, 28) tensor.
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_mnist_model(device, path=path)
    initial = torch.randn((num_images, 1, 28, 28), device=device)
    if label is None:
        return _reverse_loop_cfg(model, initial, NULL_CLASS, T, device, guidance_scale=1.0)
    return _reverse_loop_cfg(model, initial, label, T, device, guidance_scale)


def sample_mnist_with_snapshots(num_images=16, label=None, T=1000, guidance_scale=7.5, num_snapshots=50, path="mnist_diffusion_model_conditioned.pth"):
    """
    Like sample_mnist(), but also returns (final, snapshots, labels) for GIF generation.

    label: int 0-9 for class-conditioned, or None for unconditional.
    Returns snapshots ordered noise → data.
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_mnist_model(device, path=path)
    initial = torch.randn((num_images, 1, 28, 28), device=device)
    if label is None:
        return _reverse_loop_cfg_with_snapshots(model, initial, NULL_CLASS, T, device, guidance_scale=1.0, num_snapshots=num_snapshots)
    return _reverse_loop_cfg_with_snapshots(model, initial, label, T, device, guidance_scale, num_snapshots)


if __name__ == "__main__":
    import sys
    digit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    final, snapshots, labels = sample_mnist_with_snapshots(num_images=4, label=digit, T=1000, guidance_scale=7.5, num_snapshots=100)
    steady_state = snapshots[0]                          # cleanest frame (t≈0)
    gif_frames = snapshots[::-1] + [steady_state] * 40  # noisy → clean, then hold
    gif_labels = labels[::-1] + ["t=0"] * 40
    label_str = str(digit) if digit is not None else "unconditioned"
    save_mnist_gif(gif_frames, gif_labels, path=f"mnist_{label_str}.gif", fps=20)
    visualize_mnist_sample(final, title=f"Digit {label_str} samples")
