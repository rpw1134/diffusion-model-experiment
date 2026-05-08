import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (N, half)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (N, dim)


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(dim=32)
        self.proj = nn.Linear(32, 128)
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
        t_emb = self.proj(self.time_emb(t))   # (N, 128)
        x = torch.cat([x, t_emb], dim=1)      # (N, 130)
        x = self.model(x)
        return self.output_head(x)


class MnistDiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(dim=32)

        # Time projections — target channel dim at each block, broadcast over H and W in forward.
        self.time_proj_init  = nn.Linear(32,  32)
        self.time_proj_down1 = nn.Linear(32,  64)
        self.time_proj_down2 = nn.Linear(32, 128)
        self.time_proj_btn   = nn.Linear(32, 128)
        self.time_proj_up1   = nn.Linear(32,  64)
        self.time_proj_up2   = nn.Linear(32,  32)

        # Encoder — stride=2, kernel=3, padding=1 halves spatial dims cleanly. 28 → 14 → 7
        # GroupNorm(8, C) normalizes within 8 channel groups, stabilizing training across the network.
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),    # (N,  1, 28, 28) → (N,  32, 28, 28)
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (N,  32, 28, 28) → (N,  64, 14, 14)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (N,  64, 14, 14) → (N, 128,  7,  7)
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),           # (N, 128, 7, 7) → (N, 128, 7, 7)
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        # Decoder — ConvTranspose2d with stride=2, kernel=3, padding=1, output_padding=1 exactly
        # inverts the encoder stride=2 conv. After upsample, cat skip (doubles channels), then conv down.
        self.up1_transpose = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # (N, 128,  7,  7) → (N, 64, 14, 14)
        self.up1_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),            # (N, 128, 14, 14) → (N, 64, 14, 14)  — 128 = 64 transpose + 64 skip1
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        self.up2_transpose = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)   # (N,  64, 14, 14) → (N, 32, 28, 28)
        self.up2_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),             # (N,  64, 28, 28) → (N, 32, 28, 28)  — 64 = 32 transpose + 32 skip0
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )

        self.output_head = nn.Conv2d(32, 1, kernel_size=1)           # (N, 32, 28, 28) → (N,  1, 28, 28)

    def forward(self, x, t):
        t_emb = self.time_emb(t)  # (N, 32)

        # Encoder — save skip connections at each resolution
        skip0 = self.init_conv(x) + self.time_proj_init(t_emb).unsqueeze(-1).unsqueeze(-1)   # (N,  32, 28, 28)
        skip1 = self.down1(skip0) + self.time_proj_down1(t_emb).unsqueeze(-1).unsqueeze(-1)  # (N,  64, 14, 14)
        x     = self.down2(skip1) + self.time_proj_down2(t_emb).unsqueeze(-1).unsqueeze(-1)  # (N, 128,  7,  7)

        # Bottleneck
        x = self.bottleneck(x) + self.time_proj_btn(t_emb).unsqueeze(-1).unsqueeze(-1)       # (N, 128,  7,  7)

        # Decoder — upsample, inject time, cat skip, conv
        x = self.up1_transpose(x) + self.time_proj_up1(t_emb).unsqueeze(-1).unsqueeze(-1)   # (N,  64, 14, 14)
        x = self.up1_conv(torch.cat([x, skip1], dim=1))                                      # (N, 128, 14, 14) → (N, 64, 14, 14)

        x = self.up2_transpose(x) + self.time_proj_up2(t_emb).unsqueeze(-1).unsqueeze(-1)   # (N,  32, 28, 28)
        x = self.up2_conv(torch.cat([x, skip0], dim=1))                                      # (N,  64, 28, 28) → (N, 32, 28, 28)

        return self.output_head(x)                                                            # (N,   1, 28, 28)
