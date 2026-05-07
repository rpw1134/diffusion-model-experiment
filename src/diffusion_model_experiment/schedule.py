import torch
def generate_schedule(T=100):
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


if __name__ == "__main__":
    arr, alphas, alpha_bars = generate_schedule()
    print(arr.shape)
    print(alphas.shape)
    print(alpha_bars.shape)