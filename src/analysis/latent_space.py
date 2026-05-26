"""UMAP visualisation of batter embeddings and nearest-neighbour lookup."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


def extract_embeddings(model, device: torch.device | None = None) -> np.ndarray:
    """
    Extract the full batter embedding matrix from a trained model.
    Returns (N_batters, embed_dim) numpy array.
    """
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        emb = model.unet.batter_emb.embed.weight.detach().cpu().numpy()
    return emb


def run_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Reduce to 2D with UMAP. Returns (N, 2) array."""
    if not _UMAP_AVAILABLE:
        raise ImportError("Install umap-learn: pip install umap-learn")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)


def load_batter_id_map(processed_dir: str | Path) -> dict[int, int]:
    """Returns {mlbam_id -> 0-indexed embed row}."""
    with open(Path(processed_dir) / "batter_id_map.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def plot_latent_space(
    coords_2d: np.ndarray,                         # (N, 2)
    labels: np.ndarray | None = None,              # (N,) int archetype labels
    archetype_names: list[str] | None = None,
    highlight_ids: list[int] | None = None,        # 0-indexed rows to annotate
    id_to_name: dict[int, str] | None = None,      # row_idx -> player name
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot 2D UMAP projection coloured by hitter archetype."""
    fig, ax = plt.subplots(figsize=figsize)
    scatter_kw = dict(s=10, alpha=0.6, linewidths=0)

    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab10")
        for lbl in unique_labels:
            mask = labels == lbl
            name = (archetype_names or {})[lbl] if archetype_names and lbl < len(archetype_names) else str(lbl)
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=[cmap(lbl)], label=name, **scatter_kw)
        ax.legend(markerscale=2, fontsize=9)
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c="#4477AA", **scatter_kw)

    if highlight_ids is not None:
        for row in highlight_ids:
            x, y = coords_2d[row]
            label = (id_to_name or {}).get(row, str(row))
            ax.annotate(label, (x, y), fontsize=7, ha="center",
                        xytext=(0, 5), textcoords="offset points")
            ax.scatter([x], [y], s=60, c="red", zorder=5)

    ax.set_title("Batter embedding space (UMAP projection)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def nearest_neighbors(
    embeddings: np.ndarray,
    query_row: int,
    k: int = 5,
    id_map_inv: dict[int, int] | None = None,
) -> list[dict]:
    """
    Return k nearest batter rows (by cosine distance) to query_row.
    id_map_inv: {embed_row -> mlbam_id}
    """
    query = embeddings[query_row]
    norms = np.linalg.norm(embeddings, axis=1) + 1e-10
    query_norm = np.linalg.norm(query) + 1e-10
    cosine_sim = (embeddings @ query) / (norms * query_norm)
    cosine_sim[query_row] = -2.0   # exclude self
    top_k = np.argsort(cosine_sim)[::-1][:k]

    results = []
    for row in top_k:
        mlbam = (id_map_inv or {}).get(int(row), int(row))
        results.append({"embed_row": int(row), "mlbam_id": mlbam, "cosine_sim": float(cosine_sim[row])})
    return results
