# Spray Chart Diffusion: Learning Conditional Batted Ball Distributions

A conditional DDPM (Denoising Diffusion Probabilistic Model) that generates baseball spray chart distributions from batter identity and game situation. Rather than producing a single point estimate, the model learns the full distribution over where a batter's batted balls land in fair territory — yielding calibrated uncertainty and meaningful probabilistic predictions for fielder positioning.

The key capability that separates this from prior tools like SEAM is inpainting: given only *k* observed batted balls from a new batter (e.g., a mid-season call-up with 20 plate appearances), the model conditions on that partial evidence and recovers a full spray chart distribution using the prior it learned from 1.2 million historical events. Performance is evaluated against four baselines (historical KDE, platoon KDE, situational KDE, and a discriminative CNN), with the model's advantage expected to be largest in the sparse-data regime.

---

## Setup

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Confirm GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Running the Pipeline

### Step 1 — Download Statcast data (2015–2023)
```bash
python scripts/download_data.py
# Single season for testing:
python scripts/download_data.py --years 2023
```
Writes one CSV per season to `data/raw/`.

### Step 2 — Preprocess into spray chart images
```bash
python scripts/build_dataset.py
```
Reads `data/raw/`, writes 64×64 `.npy` images to `data/processed/`, and saves `data/processed/metadata.csv` with train/val/test split assignments (2015–2021 / 2022 / 2023).

### Step 3 — Train the model
```bash
python scripts/run_training.py --config configs/default.yaml
```
Saves checkpoints to `checkpoints/`. The best validation checkpoint is `checkpoints/best.pt`. Sample spray charts are saved to `samples/` every 500 steps.

### Step 4 — Evaluate
Open `notebooks/04_evaluation_results.ipynb` and run all cells.

### Interactive demo
Open `demo.ipynb` (at the project root), set `BATTER_MLBAM` and `SITUATION`, and run all cells to generate and plot a spray chart.

---

## Project Structure

```
spray_chart_diffusion/
├── src/
│   ├── data/         fetch.py · preprocess.py · dataset.py
│   ├── model/        unet.py · embeddings.py · diffusion.py
│   ├── training/     train.py · losses.py
│   ├── inference/    sample.py · inpaint.py
│   ├── evaluation/   metrics.py · baselines.py
│   └── analysis/     latent_space.py · visualize.py
├── configs/default.yaml
├── scripts/          download_data.py · build_dataset.py · run_training.py
├── notebooks/        01_data_exploration … 04_evaluation_results
└── demo.ipynb
```

---

## Tasks

**Task 1 — Situational conditioning:** Given a batter MLBAM ID, pitcher handedness, count state, and pitch type, the model generates a 64×64 spray chart probability density. Evaluated with KL divergence and zone accuracy on 2023 holdout batters.

**Task 2 — Sparse-data inpainting:** Given only *k* observed landing coordinates, the model concatenates a partial spray chart to the denoiser at each reverse-diffusion step and recovers a calibrated distribution. Evaluated at k ∈ {10, 25, 50, 100} against full-season KDE baselines.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **KL divergence** | KL(empirical 2023 density ∥ generated mean density) |
| **Zone accuracy** | Predicted vs. actual probabilities in 9 field zones (pull/center/oppo × GB/LD/FB) |
| **Calibration** | Fraction of actual 2023 events inside model's 80% credible region (target ≈ 0.80) |
| **Sparse-data curve** | KL divergence vs. available PA count at k = 10, 25, 50, 100 |

---

## Baselines

1. **HistoricalKDE** — career average KDE, no situational adjustment
2. **PlatoonKDE** — separate KDE for vs. LHP / vs. RHP
3. **SituationalKDE** — KDE restricted to exact situation; degrades under sparse data
4. **DiscriminativeCNN** — ResNet-style point estimator trained with MSE; controls for model capacity

---

*CS 159 Final Project — Arjun Pradhan & Jason Tran — Spring 2026*
