import torch
from torch import nn
from torch.utils.data import random_split

from diffusion_model_experiment.dataset import DiffusionDataset
from diffusion_model_experiment.forward import forward_diffusion, generate_normal_noise
from diffusion_model_experiment.model import DiffusionModel
from diffusion_model_experiment.schedule import generate_uniform_times, generate_schedule


def train(epochs = 25):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = DiffusionModel().to(device)
    dataset = DiffusionDataset()
    schedule = generate_schedule(T=100)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_split, val_split = random_split(dataset, [4200, 800])
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        for batch_idx, data in enumerate(train_loader):
            clean_samples = data
            times = generate_uniform_times(num_samples=clean_samples.shape[0])
            noise = generate_normal_noise(clean_samples.shape)
            noisy_samples = forward_diffusion(samples=clean_samples, schedule=schedule, t=times, noise=noise).to(device)

            predicted_noise = model(torch.concat([noisy_samples, times], dim=1))
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        for batch_idx, data in enumerate(val_loader):
            with torch.no_grad():
                clean_samples = data
                times = generate_uniform_times(num_samples=clean_samples.shape[0])
                noise = generate_normal_noise(clean_samples.shape)
                noisy_samples = forward_diffusion(samples=clean_samples, schedule=schedule, t=times, noise=noise).to(device)

                predicted_noise = model(torch.concat([noisy_samples, times], dim=1))
                loss = criterion(predicted_noise, noise)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return model








