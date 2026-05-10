import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """Projects a scalar timestep t to a sinusoidal vector of length `dim`."""

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (N, half)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (N, dim)


class DiffusionModel(nn.Module):
    """Small MLP for 2D noise prediction. Input: concat(x_t, time_emb). Output: ε̂."""

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


NULL_CLASS = 10  # reserved index for unconditional (CFG dropout) token


class MnistDiffusionUNet(nn.Module):
    """
    UNet for MNIST noise prediction with class-conditional guidance.

    Time and class embeddings are injected additively at each resolution block
    (broadcast over H and W), rather than concatenated at the input as in the
    2D MLP. This is the standard UNet conditioning pattern.
    """

    def __init__(self):
        super().__init__()
        self.time_emb  = SinusoidalEmbedding(dim=32)
        self.class_emb = nn.Embedding(11, 32)  # 0-9 = digits, 10 = null token

        # One (time, class) projection pair per block, sized to match that block's channel dim.
        self.time_proj_init  = nn.Linear(32,  32)
        self.time_proj_down1 = nn.Linear(32,  64)
        self.time_proj_down2 = nn.Linear(32, 128)
        self.time_proj_btn   = nn.Linear(32, 128)
        self.time_proj_up1   = nn.Linear(32,  64)
        self.time_proj_up2   = nn.Linear(32,  32)

        self.class_proj_init  = nn.Linear(32,  32)
        self.class_proj_down1 = nn.Linear(32,  64)
        self.class_proj_down2 = nn.Linear(32, 128)
        self.class_proj_btn   = nn.Linear(32, 128)
        self.class_proj_up1   = nn.Linear(32,  64)
        self.class_proj_up2   = nn.Linear(32,  32)

        # Encoder — stride=2 convs halve spatial dims: 28 → 14 → 7
        # GroupNorm(8, C) keeps training stable without large batches.
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        # Decoder — ConvTranspose2d with output_padding=1 exactly inverts the stride-2 encoder convs.
        # After upsampling, skip connections are concatenated (doubling channels) before the conv.
        self.up1_transpose = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),   # 128 = 64 upsampled + 64 skip
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        self.up2_transpose = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),    # 64 = 32 upsampled + 32 skip
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )

        self.output_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t, c):
        t_emb = self.time_emb(t)   # (N, 32)
        c_emb = self.class_emb(c)  # (N, 32)

        def inject(features, t_proj, c_proj):
            return features + t_proj(t_emb).unsqueeze(-1).unsqueeze(-1) \
                            + c_proj(c_emb).unsqueeze(-1).unsqueeze(-1)

        # Encoder
        skip0 = inject(self.init_conv(x),  self.time_proj_init,  self.class_proj_init)
        skip1 = inject(self.down1(skip0),  self.time_proj_down1, self.class_proj_down1)
        x     = inject(self.down2(skip1),  self.time_proj_down2, self.class_proj_down2)

        # Bottleneck
        x = inject(self.bottleneck(x), self.time_proj_btn, self.class_proj_btn)

        # Decoder
        x = inject(self.up1_transpose(x), self.time_proj_up1, self.class_proj_up1)
        x = self.up1_conv(torch.cat([x, skip1], dim=1))

        x = inject(self.up2_transpose(x), self.time_proj_up2, self.class_proj_up2)
        x = self.up2_conv(torch.cat([x, skip0], dim=1))

        return self.output_head(x)
