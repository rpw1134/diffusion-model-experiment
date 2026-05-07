import torch

from diffusion_model_experiment.dataset import generate_dataset
from diffusion_model_experiment.schedule import generate_schedule
from diffusion_model_experiment.visualize import visualize_samples

def generate_normal_noise(shape, T=100):
    n = torch.randn(size=shape)
    return n
def forward_diffusion(samples, schedule, t, noise=None):
    if noise is None:
        noise = generate_normal_noise(samples.shape)
    alpha_bar_t = schedule[2][t].unsqueeze(1)
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
