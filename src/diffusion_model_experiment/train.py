import torch
from torch import nn
from torch.utils.data import random_split

from diffusion_model_experiment.dataset import DiffusionDataset, MNISTDataset
from diffusion_model_experiment.forward import forward_diffusion, generate_normal_noise
from diffusion_model_experiment.model import DiffusionModel, MnistDiffusionUNet
from diffusion_model_experiment.schedule import generate_uniform_times, generate_schedule


def train(epochs=100, T=1000, save_path="diffusion_model.pth"):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = DiffusionModel().to(device)
    dataset = DiffusionDataset()
    schedule = tuple(s.to(device) for s in generate_schedule(T=T))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_split, val_split = random_split(dataset, [4200, 800])
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=64, shuffle=True)
    best_val_loss = float("inf")
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        for batch_idx, data in enumerate(train_loader):
            clean_samples = data.to(device)
            times = generate_uniform_times(num_samples=clean_samples.shape[0], T=T).to(device)
            noise = generate_normal_noise(clean_samples.shape).to(device)
            noisy_samples = forward_diffusion(samples=clean_samples, schedule=schedule, t=times, noise=noise).to(device)

            predicted_noise = model(noisy_samples, times)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        for batch_idx, data in enumerate(val_loader):
            with torch.no_grad():
                clean_samples = data.to(device)
                times = generate_uniform_times(num_samples=clean_samples.shape[0], T=T).to(device)
                noise = generate_normal_noise(clean_samples.shape).to(device)
                noisy_samples = forward_diffusion(samples=clean_samples, schedule=schedule, t=times, noise=noise).to(device)

                predicted_noise = model(noisy_samples, times)
                loss = criterion(predicted_noise, noise)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return model


def train_mnist(epochs=100, T=1000, save_path="mnist_diffusion_model.pth"):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = MnistDiffusionUNet().to(device)
    dataset = MNISTDataset()
    schedule = tuple(s.to(device) for s in generate_schedule(T=T))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_split, val_split = random_split(dataset, [58000, 2000])
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=64, shuffle=True)
    best_val_loss = float("inf")
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            clean_samples = images.unsqueeze(1).to(device)  # (N, 1, 28, 28)
            times = generate_uniform_times(num_samples=clean_samples.shape[0], T=T).to(device)
            noise = generate_normal_noise(clean_samples.shape).to(device)
            noisy_samples = forward_diffusion(samples=clean_samples, schedule=schedule, t=times, noise=noise)

            predicted_noise = model(noisy_samples, times)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        for batch_idx, (images, _) in enumerate(val_loader):
            with torch.no_grad():
                clean_samples = images.unsqueeze(1).to(device)  # (N, 1, 28, 28)
                times = generate_uniform_times(num_samples=clean_samples.shape[0], T=T).to(device)
                noise = generate_normal_noise(clean_samples.shape).to(device)
                noisy_samples = forward_diffusion(samples=clean_samples, schedule=schedule, t=times, noise=noise)

                predicted_noise = model(noisy_samples, times)
                loss = criterion(predicted_noise, noise)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return model


if __name__ == "__main__":
    train_mnist(epochs=300, T=1000, save_path="mnist_diffusion_model_large.pth")









