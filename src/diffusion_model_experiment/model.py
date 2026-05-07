import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, 128)

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (N, half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (N, dim)
        return self.proj(emb)  # (N, 128)


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(dim=32)
        self.model = nn.Sequential(
            nn.Linear(130, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.output_head = nn.Linear(128, 2)

    def forward(self, x, t):
        t_emb = self.time_emb(t)           # (N, 128)
        x = torch.cat([x, t_emb], dim=1)  # (N, 130) — wait, 2 + 128 = 130, first layer takes 130
        x = self.model(x)
        return self.output_head(x)
