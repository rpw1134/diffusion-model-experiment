import torch
def forward_diffusion(samples, schedule, T, noise=None):
    if noise is None:
        noise = torch.randn_like(samples)
    t = (T * torch.rand(samples.shape[0])).int()
    alpha_bar_t = schedule[2][t].unsqueeze(1)
    return torch.sqrt(alpha_bar_t) * samples + torch.sqrt(1 - alpha_bar_t) * noise
