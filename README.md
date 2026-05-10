# Diffusion Model from Scratch

A PyTorch implementation of DDPM (Ho et al. 2020) built from scratch — no diffusion libraries. Two models are included: a toy 2D MLP trained on a two-moons distribution, and a class-conditional UNet trained on MNIST with classifier-free guidance.

## Project structure

```
src/diffusion_model_experiment/
├── dataset.py    — two-moons and MNIST data loading
├── schedule.py   — noise schedule: β_t, α_t, ᾱ_t
├── forward.py    — closed-form forward process (DDPM Eq. 4)
├── model.py      — 2D MLP and MNIST UNet with sinusoidal time embeddings
├── train.py      — training loops for both models
├── inference.py  — reverse diffusion sampling (with and without CFG)
└── visualize.py  — scatter plots and GIF export
```

## Quickstart

```bash
uv sync
```

**Train the 2D model:**
```bash
python -m diffusion_model_experiment.train
```

**Train the MNIST model (300 epochs, ~4h on M3):**
```bash
# edit train.py __main__ or call train_mnist() directly
python -m diffusion_model_experiment.train
```

**Sample from the 2D model:**
```python
from diffusion_model_experiment.inference import sample_with_snapshots
from diffusion_model_experiment.visualize import save_gif

final, snapshots, labels = sample_with_snapshots(num_points=1000, num_snapshots=50)
save_gif(snapshots, labels, path="diffusion.gif")
```

**Sample a digit from the MNIST model:**
```bash
python -m diffusion_model_experiment.inference 8
# generates mnist_8.gif
```

## Architecture

**2D MLP** — takes `(x_t, t)` where `t` is projected to a 128-dim sinusoidal embedding, concatenated with `x_t`, then passed through 4 × 128-unit SiLU layers.

**MNIST UNet** — encoder/decoder with skip connections (28→14→7→14→28), GroupNorm, and sinusoidal time + class embeddings injected additively at each resolution block. Trained with CFG dropout (20% of labels replaced with a null token) so the same model handles both conditioned and unconditional generation.

## Hyperparameters

| | 2D model | MNIST model |
|---|---|---|
| T | 1000 | 1000 |
| β schedule | linear 1e-4 → 0.02 | linear 1e-4 → 0.02 |
| Optimizer | Adam 1e-3 | Adam 1e-3 |
| Batch size | 64 | 64 |
| Epochs | ~300 | ~300 |
| CFG scale (inference) | — | 7.5 |
