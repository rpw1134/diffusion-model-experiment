import torch
from torch import nn
from torch.utils.data import random_split

from diffusion_model_experiment.dataset import DiffusionDataset
from diffusion_model_experiment.model import DiffusionModel


def train():
    model = DiffusionModel()
    dataset = DiffusionDataset()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_split, val_split = random_split(dataset, [4200, 800])
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=64, shuffle=True)