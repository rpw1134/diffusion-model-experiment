import torch

from diffusion_model_experiment.dataset import generate_dataset
from diffusion_model_experiment.model import DiffusionModel
from diffusion_model_experiment.schedule import generate_schedule
from diffusion_model_experiment.visualize import visualize_sample, visualize_samples, save_gif


def _load_model(device, path="diffusion_model.pth"):
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _reverse_loop(model, device, num_points, T):
    current = torch.randn((num_points, 2), device=device)
    betas, alphas, cumulative_alphas = [s.to(device) for s in generate_schedule(T=T)]
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            t_tensor = torch.full((num_points,), t, device=device)
            prediction = model(current, t_tensor)
            additive_noise = torch.sqrt(betas[t]) * torch.randn_like(current) if t > 0 else 0
            current = (1 / torch.sqrt(alphas[t])) * (current - (betas[t] / torch.sqrt(1 - cumulative_alphas[t])) * prediction) + additive_noise
    return current, betas, alphas, cumulative_alphas


def sample(num_points=1000, T=1000):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_model(device)
    current, *_ = _reverse_loop(model, device, num_points, T)
    return current


def sample_with_snapshots(num_points=1000, T=1000, num_snapshots=5):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_model(device)
    current = torch.randn((num_points, 2), device=device)
    betas, alphas, cumulative_alphas = [s.to(device) for s in generate_schedule(T=T)]
    checkpoints = set(range(0, T, max(1, T // num_snapshots)))
    checkpoints.add(T - 1)
    snapshots, labels = [], []
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            t_tensor = torch.full((num_points,), t, device=device)
            prediction = model(current, t_tensor)
            additive_noise = torch.sqrt(betas[t]) * torch.randn_like(current) if t > 0 else 0
            current = (1 / torch.sqrt(alphas[t])) * (current - (betas[t] / torch.sqrt(1 - cumulative_alphas[t])) * prediction) + additive_noise
            if t in checkpoints:
                snapshots.append(current.clone())
                labels.append(f"t={t}")
    snapshots.reverse()
    labels.reverse()
    return current, snapshots, labels


def test_timestep(T=1000):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = _load_model(device)
    test_point = generate_dataset(n_samples=1).to(device)
    with torch.no_grad():
        for t in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
            prediction = model(test_point, torch.tensor([t], device=device))
            print(f"t={t}: {prediction}")


if __name__ == "__main__":
    final, snapshots, labels = sample_with_snapshots(num_points=1000, T=1000, num_snapshots=200)
    steady_state = snapshots[0]
    gif_frames = snapshots[::-1] + [steady_state] * 40
    gif_labels = labels[::-1] + ["t=0"] * 40
    save_gif(gif_frames, gif_labels, path="diffusion_process.gif", fps=20)
