import torch

from diffusion_model_experiment.dataset import generate_dataset
from diffusion_model_experiment.schedule import generate_schedule
from diffusion_model_experiment.visualize import visualize_samples


def generate_normal_noise(shape):
    return torch.randn(size=shape)


def forward_diffusion(samples, schedule, t, noise=None):
    """Closed-form forward step: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε (DDPM Eq. 4)."""
    if noise is None:
        noise = generate_normal_noise(samples.shape)
    alpha_bar_t = schedule[2][t]
    # Reshape to (N, 1, 1, ...) to broadcast over spatial dims for both 2D points and images
    for _ in range(samples.ndim - 1):
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    return torch.sqrt(alpha_bar_t) * samples + torch.sqrt(1 - alpha_bar_t) * noise


if __name__ == "__main__":
    samples = generate_dataset(n_samples=500)
    schedule = generate_schedule(T=100)
    timesteps = [0, 25, 50, 75, 99]
    results = []
    for t in timesteps:
        t_tensor = torch.full((500,), t)
        results.append(forward_diffusion(samples, schedule, t_tensor))
    visualize_samples(results, titles=[f"t={t}" for t in timesteps])
