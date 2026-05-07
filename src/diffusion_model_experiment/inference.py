import torch

from diffusion_model_experiment.dataset import generate_dataset
from diffusion_model_experiment.model import DiffusionModel
from diffusion_model_experiment.schedule import generate_schedule
from diffusion_model_experiment.visualize import visualize_sample, visualize_samples


def test_timestep(T=5000):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
    model.eval()
    test_point = generate_dataset(n_samples=1).to(device)
    with torch.no_grad():
        for t in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
            prediction = model(test_point, torch.tensor([t], device=device))
            print(f"t={t}: {prediction}")

def generate_new_sample(num_points=1000, T=5000):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
    model.eval()

    current = torch.randn((num_points, 2), device=device)
    betas, alphas, cumulative_alphas = [s.to(device) for s in generate_schedule(T=T)]
    checkpoints = {T - 1, 3 * T // 4, T // 2, T // 4, 0}
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


if __name__ == "__main__":
    final, snapshots, labels = generate_new_sample(num_points=1000, T=1000)
    print(labels)
    visualize_samples(snapshots, titles=labels)
