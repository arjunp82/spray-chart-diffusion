# Spray Chart Diffusion: Learning Conditional Batted Ball Distributions

A conditional DDPM (Denoising Diffusion Probabilistic Model) that generates baseball spray chart distributions from batter identity and game situation. Rather than producing a single point estimate, the model learns the full distribution over where a batter's batted balls land in fair territory — yielding calibrated uncertainty and meaningful probabilistic predictions.

The key capability is inpainting: given only *k* observed batted balls from a batter never seen during training (e.g., a call-up with 20 plate appearances), the model conditions on that partial evidence and recovers a full spray chart distribution using its learned prior over batter styles.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1 — Download Statcast data (2023)
```bash
python -m src.data.fetch
```
Writes `data/raw/statcast_2023.csv`.

### Step 2 — Preprocess into spray chart images
```bash
python -m src.data.preprocess \
    --raw-dir data/raw \
    --processed-dir data/processed
```
Reads `data/raw/`, writes 64×64 `.npy` density images to `data/processed/`, and saves `data/processed/metadata.csv` with a **per-batter** train/val/test split (no batter appears in more than one split). 12 batters are held out entirely from all training data for the sparse-data evaluation.

### Step 3 — Train the model
```bash
python -m src.training.train --config configs/default.yaml
```
Saves checkpoints to `checkpoints/heldout/`. The best validation checkpoint is `checkpoints/heldout/best.pt`.

### Step 4 — Evaluate
```bash
python scripts/evaluate.py --config configs/default.yaml
```
Runs four tasks and writes figures to `results/`.

---

## Project Structure

```
spray_chart_diffusion/
├── src/
│   ├── data/         preprocess.py · dataset.py
│   ├── model/        unet.py · diffusion.py
│   ├── training/     train.py
│   └── evaluation/   metrics.py · baselines.py
├── configs/          default.yaml · fast.yaml
├── scripts/          evaluate.py · zone_style_charts.py · situation_grid.py
└── results/          generated figures
```

---

## Architecture

- **Backbone**: Conditional U-Net with ResBlocks and self-attention at 8×8.
- **Conditioning**: Timestep embedding injected additively in every ResBlock. Batter identity and situation code injected via cross-attention only at the 8×8 bottleneck.
- **Diffusion**: DDPM with 1000 timesteps, cosine (`squaredcos_cap_v2`) noise schedule, epsilon prediction.
- **Guidance**: Classifier-free guidance (scale 5.0) over both batter and situation dimensions.
- **Inpainting**: Partial chart (k observed events → Gaussian-blurred density) concatenated channel-wise to the noisy input at each reverse step.
- **Batter embeddings**: 640-entry table (index 0 = population prior; indices 1–639 = training batters sorted by MLBAM). 10% dropout during training so index 0 learns a real population prior.

---

## Tasks and Evaluation

**Task 1 — Situational conditioning:** Given a batter MLBAM ID, pitcher handedness, count state, and pitch type, generate a 64×64 spray chart density. Evaluated with KL divergence on val-split batters.

**Task 2 — Sparse-data inpainting:** Given only *k* observed events from a **held-out batter** (never seen during training, embedding index 0), generate a full density. Evaluated at k ∈ {10, 25, 50, 100} against three conditions: HistoricalKDE baseline, embedding-only control (zeros partial), and diffusion with partial conditioning.

**Task 3 — Calibration:** For held-out batters, check what fraction of actual events fall inside the model's X% credible region across X ∈ {50, 70, 80, 90}.

**Task 4 — Inpainting gallery:** Visual comparison of partial → inpainted charts at increasing k for a training batter (Freddie Freeman).

---

## Baselines

- **HistoricalKDE** — KDE over all observed events; no situation adjustment
- **PlatoonKDE** — separate KDE for vs. LHP / vs. RHP
- **SituationalKDE** — KDE restricted to exact (count, handedness, pitch type) bucket

---

*CS 159 Final Project — Arjun Pradhan & Jason Tran — Spring 2026*
