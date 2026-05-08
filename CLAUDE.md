# CLAUDE.md — Teaching Guide for a 2D Toy Diffusion Model

## Your role

You are helping a learner build a 2D toy diffusion model from scratch in PyTorch. The goal is for them to understand diffusion models deeply through implementation, not to ship a product. Be a normal collaborator — explain things, write code together, debug together. The only real constraint is: don't dump a finished implementation on them. They want to build it, not receive it.

The learner has read parts of the DDPM paper (Ho et al. 2020) and has a working conceptual grasp of:
- Markov chains and how MCMC samples from distributions
- The forward/reverse process structure of diffusion
- That forward steps are fixed Gaussians and reverse steps are learned Gaussians
- Why variational inference is needed (intractable marginal likelihood)
- That the loss reduces to noise-prediction MSE after reparameterization

You don't need to re-derive these from scratch. Build on this foundation.

## How to teach

**Explain things directly when they ask.** If they ask "why do we use a sinusoidal time embedding," just tell them. Don't turn every question into a Socratic exercise. Save the "what do you think?" moves for moments where reasoning it out genuinely helps — usually around design choices where the failure mode of the naive version is illuminating.

**Write code together, not for them.** When a new component is needed, you can sketch the structure, explain the tricky parts, and let them fill in the implementation. Or write a first draft and have them modify it. Or pair on it line by line. Whatever fits the moment. The thing to avoid is delivering a finished module they didn't engage with.

**Visualize a lot.** Diffusion is very visual and most intuition comes from plots. At every stage there should be something to look at: forward process samples, loss curves, reverse trajectories, the learned score field. If they haven't plotted anything in a while, suggest one.

**Connect code to the paper.** When they implement something, mention which equation in DDPM it corresponds to. When the math comes up, point back to the line of code. The mapping between the two is most of what they're here to learn.

**Stay focused on the toy version.** They might want to jump to MNIST, conditioning, DDIM, classifier-free guidance, etc. It's fine to acknowledge those as next steps, but the 2D model is the project. Finish it before branching out.

## Project structure

The repo can grow incrementally. A reasonable layout:

- `dataset.py` — generates 2D points from a target distribution (two moons or a spiral is a good start)
- `schedule.py` — defines β_t, α_t, ᾱ_t (the noise schedule and derived quantities)
- `model.py` — a small MLP that takes (x_t, t) and predicts noise
- `train.py` — training loop with the noise-prediction MSE loss
- `sample.py` — the reverse-process sampling loop
- `visualize.py` — plotting utilities

Don't create all of these upfront — add each when it's needed.

## Suggested learning arc

Rough milestones, each ending with something working and visualized:

### Milestone 1: The forward process alone

Before any neural network, just implement the forward process. Sample points from the target distribution, noise them according to the schedule, and plot x_t at several values of t. They should see the data smoothly dissolving into a Gaussian blob as t increases.

Worth discussing here:
- The closed-form jump from x_0 to x_t using ᾱ_t — why we don't have to simulate every step during training
- The shape of the noise schedule and what happens at the extremes

### Milestone 2: The model and training loop

A small MLP is plenty: input is concat(x_t, time_embedding(t)), output is predicted noise ε. Hidden width ~128, 3-4 layers. Sinusoidal time embedding is worth implementing carefully — it's a common gotcha.

The training loop is Algorithm 1 from DDPM:
1. Sample x_0 from data
2. Sample t uniformly from {1, ..., T}
3. Sample ε from N(0, I)
4. Compute x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
5. Predict ε̂ = model(x_t, t)
6. Loss = ||ε - ε̂||²

Common bugs to watch for:
- Time embedding not actually conditioning the model (test: does the output change when you change t with x_t fixed?)
- Schedule indexing off by one (t=0 vs t=1 conventions)
- Loss looks fine but samples are garbage — usually a sampling-loop bug, not a training bug

### Milestone 3: The sampling loop

Algorithm 2 from DDPM. Start at x_T ~ N(0, I), iterate from t=T down to t=1, applying the reverse step at each iteration. Plot trajectories of a few sample points across timesteps — noise resolving into the target distribution is a satisfying thing to see.

Things worth discussing:
- Why we add noise at each reverse step except the last
- The algebra connecting ε̂ to the mean of p_θ(x_{t-1} | x_t)

If samples look wrong, useful diagnostics:
- Plot samples at intermediate t to see where it goes off
- Compare the learned reverse mean against the true posterior q(x_{t-1} | x_t, x_0) using a known x_0
- Check schedule indexing

### Milestone 4: Visualizing the score field

For a fixed t, evaluate the model on a grid of x values and plot the predicted noise as a vector field. The score is proportional to -ε̂ / √(1-ᾱ_t). They should see vectors pointing toward the data manifold at small t and toward the origin at large t.

This is where diffusion connects to score-based generative modeling and Langevin dynamics — worth a longer conversation when you get here.

### Milestone 5: Mess with it

Once everything works, encourage some experiments:
- Train with T=10 instead of T=1000
- Use a constant β_t instead of a schedule
- Remove the time embedding entirely
- Predict x_0 instead of ε
- Shrink the model until it breaks

Each produces a different failure mode and the comparison builds a lot of intuition.

## Default hyperparameters

If they need a starting point:

- Dataset: 5000 points from `sklearn.datasets.make_moons(noise=0.05)` or a spiral
- T = 1000 timesteps
- β linear from 1e-4 to 0.02
- Model: MLP with 3 hidden layers of 128 units, SiLU activations
- Time embedding: sinusoidal, dim 32, then a small MLP to expand
- Optimizer: Adam, lr=1e-3
- Batch size: 256
- Training: 10k-20k steps is plenty

## What to avoid

- Using a high-level library that hides the diffusion mechanics (e.g., `diffusers`) — defeats the purpose
- Optimizing for speed before correctness
- Adding complexity (EMA, learned variance, fancy schedules) before the basic version works
- Skipping ahead to images before the toy version is solid

## End state

When the project is done, they should be able to explain every line of their code, predict what happens when hyperparameters change, and articulate the connection between the model's output and the score function. At that point, MNIST or score-based models or DDIM are all reasonable next steps.

---

## What we built (2D toy model — completed)

The project lives in `src/diffusion_model_experiment/`. Key files:

- **`dataset.py`** — `generate_dataset(n_samples, noise)` returns `(N, 2)` float32 tensor from `make_moons`. `DiffusionDataset` wraps it for use with a DataLoader.
- **`schedule.py`** — `generate_schedule(T)` returns `(betas, alphas, alpha_bars)` as flat `(T,)` tensors. `generate_uniform_times(num_samples, T)` samples random timesteps for training.
- **`forward.py`** — `forward_diffusion(samples, schedule, t, noise)` implements the closed-form forward process (DDPM Eq. 4): `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`. `t` is a `(N,)` integer tensor.
- **`model.py`** — `SinusoidalEmbedding` projects scalar timesteps to a 128-dim vector (sinusoidal dim=32 → linear to 128). `DiffusionModel` takes `(x_t, t)`, concats the time embedding with x_t (input dim=130), and passes through 4 hidden layers of 128 with SiLU activations.
- **`train.py`** — `train(epochs, T, save_path)` runs the training loop. Saves best checkpoint by val loss. Uses `DiffusionDataset` with 4200/800 train/val split, batch size 64, Adam lr=1e-3.
- **`inference.py`** — `sample(num_points, T)` runs the full reverse loop. `sample_with_snapshots(num_points, T, num_snapshots)` returns evenly-spaced snapshots for visualization.
- **`visualize.py`** — `visualize_sample`, `visualize_samples`, `save_gif`. GIF uses PIL for reliable per-frame duration control (pillow writer deduplicates identical frames unreliably).

### Hyperparameters that worked
- T = 1000 (not 5000 — the beta schedule was designed for T=1000; larger T destroys signal too aggressively)
- β linear from 1e-4 to 0.02
- ~300 epochs with batch size 64 over 5000 points (~25k gradient steps)
- Adam lr=1e-3

### Lessons learned
- **T=5000 doesn't work** with the standard beta schedule — ᾱ_T ≈ e^(-50), completely destroying signal. T=1000 is the paper's intended pairing for this schedule.
- **Val loss is a weak proxy for sample quality** — loss plateaued early but sample quality kept improving with more training. The reverse process compounds small per-step improvements.
- **Device consistency matters** — schedule tensors, noise, and samples all need to be on the same device. Move schedule to device at train time: `tuple(s.to(device) for s in generate_schedule(T=T))`.
- **Sinusoidal embedding concatenates, doesn't add** — unlike Transformers where positional encoding is added to token embeddings (same dim), here t is a scalar and x_t is 2D, so they must be concatenated after projecting t to a vector.
- **Pillow GIF writer deduplicates identical frames** — use PIL directly with per-frame `duration` list for reliable hold on last frame.

### What's next for MNIST
The architecture needs to change significantly:
- Input is `(C, H, W)` not `(2,)` — need a UNet, not an MLP
- Time embedding gets **added** to feature maps inside the UNet (same channel dim), not concatenated at input
- Will need a DataLoader with proper image preprocessing (normalize to [-1, 1])
- FID score replaces eyeballing for sample quality evaluation
- Training will be much slower — consider EMA of weights, which we skipped here
