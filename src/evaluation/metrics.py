"""Evaluation metrics: KL divergence, zone accuracy, calibration, sparse-data eval."""

from __future__ import annotations

import numpy as np
import torch

from src.data.preprocess import normalize_coords

_EPS = 1e-10
_IMG = 64

# ---------------------------------------------------------------------------
# Zone definitions: (pull / center / oppo) × (GB / LD / FB)
# We define zones in the normalised [-1, 1] coordinate system.
# ---------------------------------------------------------------------------

# Lateral thirds (pull = left side for RHB convention, oppo = right)
_X_PULL = (-1.0, -0.25)
_X_CENTER = (-0.25, 0.25)
_X_OPPO = (0.25, 1.0)

# Depth thirds (ground ball = short, line drive = mid, fly ball = deep)
_Y_GB = (0.0, 0.33)
_Y_LD = (0.33, 0.66)
_Y_FB = (0.66, 1.05)

ZONE_NAMES = [
    f"{lat}_{depth}"
    for lat in ("pull", "center", "oppo")
    for depth in ("gb", "ld", "fb")
]

_LIN = np.linspace(-1, 1, _IMG)
_GX, _GY = np.meshgrid(_LIN, _LIN[::-1])   # shape (64, 64), y>0 = outfield


def _zone_masks() -> dict[str, np.ndarray]:
    masks = {}
    for lat, (x0, x1) in zip(("pull", "center", "oppo"), (_X_PULL, _X_CENTER, _X_OPPO)):
        for dep, (y0, y1) in zip(("gb", "ld", "fb"), (_Y_GB, _Y_LD, _Y_FB)):
            masks[f"{lat}_{dep}"] = (
                (_GX >= x0) & (_GX < x1) & (_GY >= y0) & (_GY < y1)
            )
    return masks


_ZONE_MASKS = _zone_masks()


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def kl_divergence(p_generated: np.ndarray, p_empirical: np.ndarray) -> float:
    """
    KL(p_empirical || p_generated).
    Both arrays are 64×64 densities summing to 1.
    """
    p = p_empirical.flatten().astype(np.float64) + _EPS
    q = p_generated.flatten().astype(np.float64) + _EPS
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Zone accuracy
# ---------------------------------------------------------------------------

def zone_accuracy(
    generated_samples: np.ndarray,     # (N, 64, 64) generated densities
    actual_coords: np.ndarray,         # (M, 2) normalised (x, y) landing coords
) -> dict[str, tuple[float, float]]:
    """
    Compare predicted zone probabilities to actual frequencies.
    Returns dict zone_name -> (predicted_prob, actual_freq).
    """
    # Mean generated density
    mean_density = generated_samples.mean(axis=0)                   # (64, 64)
    mean_density = mean_density / (mean_density.sum() + _EPS)

    # Actual zone frequencies
    results: dict[str, tuple[float, float]] = {}
    for name, mask in _ZONE_MASKS.items():
        pred_prob = float(mean_density[mask].sum())

        x_act, y_act = actual_coords[:, 0], actual_coords[:, 1]
        lat, dep = name.split("_")
        x0, x1 = {"pull": _X_PULL, "center": _X_CENTER, "oppo": _X_OPPO}[lat]
        y0, y1 = {"gb": _Y_GB, "ld": _Y_LD, "fb": _Y_FB}[dep]
        in_zone = (x_act >= x0) & (x_act < x1) & (y_act >= y0) & (y_act < y1)
        actual_freq = float(in_zone.mean()) if len(actual_coords) > 0 else 0.0

        results[name] = (pred_prob, actual_freq)
    return results


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_score(
    generated_samples: np.ndarray,     # (N, 64, 64)
    actual_coords: np.ndarray,         # (M, 2) normalised (x, y)
    confidence_level: float = 0.80,
) -> float:
    """
    Build confidence_level% credible region from generated samples.
    Returns fraction of actual batted balls that fall inside it.
    Should be ≈ confidence_level if well-calibrated.
    """
    mean_density = generated_samples.mean(axis=0).flatten()
    order = np.argsort(mean_density)[::-1]
    cumulative = np.cumsum(mean_density[order])
    threshold_idx = np.searchsorted(cumulative, confidence_level)
    region_flat_indices = set(order[: threshold_idx + 1])

    # Map actual coords to pixel indices
    x_act, y_act = actual_coords[:, 0], actual_coords[:, 1]
    px = ((x_act + 1) / 2 * (_IMG - 1)).astype(int)
    py = ((1 - (y_act + 1) / 2) * (_IMG - 1)).astype(int)

    inside = 0
    for xi, yi in zip(px, py):
        xi = np.clip(xi, 0, _IMG - 1)
        yi = np.clip(yi, 0, _IMG - 1)
        if yi * _IMG + xi in region_flat_indices:
            inside += 1

    return inside / max(len(actual_coords), 1)


# ---------------------------------------------------------------------------
# Sparse-data evaluation
# ---------------------------------------------------------------------------

def sparse_data_eval(
    model,
    batter_events: list[tuple[float, float]],   # list of (hc_x, hc_y) raw coords
    batter_idx: int,
    situation_code: int,
    full_chart: np.ndarray,                      # (64, 64) reference density
    pa_thresholds: list[int] | None = None,
    num_samples: int = 50,
    device: torch.device | None = None,
) -> dict[int, float]:
    """
    For each PA threshold, build partial chart, run inpainting, compute KL.
    Returns {pa_count -> kl_divergence}.
    """
    from src.data.preprocess import coords_to_image

    if pa_thresholds is None:
        pa_thresholds = [10, 25, 50, 100]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[int, float] = {}
    events = np.array(batter_events)    # (M, 2) hc_x, hc_y

    bat_t = torch.tensor([batter_idx], device=device)
    sit_t = torch.tensor([situation_code], device=device)

    from src.data.dataset import DENSITY_SCALE

    for k in pa_thresholds:
        if k > len(events):
            continue
        idx = np.random.choice(len(events), k, replace=False)
        sub = events[idx]
        x_n, y_n = normalize_coords(sub[:, 0], sub[:, 1])
        partial_np = coords_to_image(x_n, y_n)
        partial_t = (torch.from_numpy(partial_np).unsqueeze(0).unsqueeze(0).to(device)
                     * DENSITY_SCALE)

        samples = []
        for _ in range(num_samples):
            gen = model.inpaint_sample(bat_t, sit_t, partial_t, num_inference_steps=50, device=device)
            samples.append(gen.cpu().numpy()[0, 0])
        samples_np = np.stack(samples)   # (num_samples, 64, 64)
        mean_gen = samples_np.mean(axis=0)

        results[k] = kl_divergence(mean_gen, full_chart)

    return results
