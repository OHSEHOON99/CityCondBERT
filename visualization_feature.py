import os

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import seaborn as sns
import wandb

# Matplotlib: headless backend (safe for servers)
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



# ======================
# Helpers (padding/mask)
# ======================
def _filter_out_padding_and_mask(feature_name, emb_np, ids_np, pad_idx=0, mask_idx=None):
    """
    Remove PAD (=0) and optionally MASK tokens per feature.
    - location: exclude PAD(0) and MASK
    - categorical(day/time/dow/weekday): exclude PAD(0)
    - delta: no filter applied
    """
    if emb_np is None or ids_np is None:
        return emb_np, ids_np

    ids_np = np.asarray(ids_np)
    if feature_name == "location":
        if mask_idx is None:
            valid = (ids_np != pad_idx) & (ids_np >= 1)
        else:
            valid = (ids_np != pad_idx) & (ids_np != mask_idx)
    elif feature_name in {"day", "time", "dow", "weekday"}:
        valid = (ids_np != pad_idx)
    else:
        return emb_np, ids_np

    if not np.any(valid):
        return None, None
    return emb_np[valid], ids_np[valid]


def _to_one_based(ids_np):
    """Convert IDs to 1-based for visualization."""
    return ids_np + 1


def id_to_xy(core_loc_ids, grid_width=200):
    """
    Convert 0-based flattened location IDs to (x, y) 1-based coordinates for plotting.
    Example: core_loc_ids: 0..39999 → returns (x,y): 1..200
    """
    core_loc_ids = np.asarray(core_loc_ids).astype(np.int64)
    x = (core_loc_ids // grid_width) + 1
    y = (core_loc_ids % grid_width) + 1
    return np.stack([x, y], axis=1)


@torch.no_grad()
def compute_feature_codebook(model, feature_name, device,
                             n_loc_samples=2000,
                             delta_max=48*75, delta_steps=256):
    """
    Compute feature-wise embedding codebooks.
    - categorical: exclude PAD(0)
    - location: exclude PAD(0) and MASK
    - delta: follow original sampling
    Returns (emb_np, ids_np)
    """
    model.eval()
    if not hasattr(model, "embedding"):
        raise AttributeError("model.embedding not found.")

    blocks = model.embedding.feature_blocks
    if not blocks or feature_name not in blocks:
        print(f"[Viz] feature '{feature_name}' not found in EmbeddingLayer; skipping codebook.")
        return None, None

    fblock = blocks[feature_name].to(device)

    if feature_name in {"day", "time", "dow", "weekday"}:
        P = {"day": 75, "time": 48, "dow": 7, "weekday": 2}[feature_name]
        ids = torch.arange(1, P + 1, device=device, dtype=torch.long).unsqueeze(0)

    elif feature_name == "location":
        V = int(blocks["location"].categorical_emb.num_embeddings)
        mask_id = V - 1
        valid = torch.arange(1, V - 1, device=device, dtype=torch.long)
        if valid.numel() > n_loc_samples:
            sel = torch.randperm(valid.numel(), device=device)[:n_loc_samples]
            ids = valid[sel].unsqueeze(0)
        else:
            ids = valid.unsqueeze(0)

    elif feature_name == "delta":
        if (delta_steps is None) or (int(delta_steps) >= int(delta_max) + 1):
            vals = torch.arange(0, int(delta_max) + 1, device=device, dtype=torch.long)
        else:
            lin = torch.linspace(0, int(delta_max), steps=int(delta_steps), device=device)
            vals = torch.round(lin).to(torch.long).unique(sorted=True)
        ids = vals.unsqueeze(0)

    else:
        raise ValueError(f"Unknown feature: {feature_name}")

    emb = fblock(ids)
    emb_np = emb.squeeze(0).detach().cpu().numpy()
    ids_np = ids.squeeze(0).detach().cpu().numpy()

    return emb_np, ids_np


def log_codebook_tsne(feature_name, emb_np, ids_np, epoch=0, perplexity=15, grid_width=200, num_location_ids=None):
    """
    Codebook embedding t-SNE visualization.

    location → 3 subplots:
      (1) color by x only
      (2) color by y only
      (3) color by flattened location id (1-based label)

    day → 3 subplots:
      (1) color by weekday (Mon~Sun)
      (2) color by week index (continuous colormap)
      (3) original: weekday colors + week number text labels

    others → single plot.
    """
    if emb_np is None or ids_np is None:
        return

    mask_idx = None
    if feature_name == "location" and num_location_ids is not None:
        mask_idx = int(num_location_ids) - 1
    emb_np, ids_np = _filter_out_padding_and_mask(feature_name, emb_np, ids_np, pad_idx=0, mask_idx=mask_idx)
    if emb_np is None or ids_np is None or len(ids_np) == 0:
        return

    X = normalize(emb_np, norm='l2')
    n = X.shape[0]
    safe_perpl = max(5, min(perplexity, (n - 1) // 3 if n > 10 else 5))

    tsne = TSNE(n_components=2, perplexity=safe_perpl, metric='cosine', random_state=42)
    reduced = tsne.fit_transform(X)

    if feature_name == "location":
        core_ids = ids_np - 1
        xy = id_to_xy(core_ids, grid_width=grid_width)
        x, y = xy[:, 0], xy[:, 1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax1, ax2, ax3 = axes

        sc1 = ax1.scatter(reduced[:, 0], reduced[:, 1], c=x, s=18, alpha=0.9, cmap='Reds')
        cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("x (grid, 1-based)")
        ax1.set_title("location t-SNE — color by x")

        sc2 = ax2.scatter(reduced[:, 0], reduced[:, 1], c=y, s=18, alpha=0.9, cmap='Greens')
        cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("y (grid, 1-based)")
        ax2.set_title("location t-SNE — color by y")

        sc3 = ax3.scatter(reduced[:, 0], reduced[:, 1], c=ids_np, s=18, alpha=0.9, cmap='viridis')
        cbar3 = plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("location id (1-based)")
        ax3.set_title("location t-SNE — color by id")

        plt.tight_layout()
        wandb.log({f"codebook/{feature_name}_tsne": wandb.Image(fig), "epoch": epoch + 1})
        plt.close(fig)

    elif feature_name == "day":
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c',
                   '#d62728', '#9467bd', '#8c564b', '#17becf']
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        weekdays = ((ids_np - 1) % 7).astype(int)
        weeks    = ((ids_np - 1) // 7 + 1).astype(int)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax1, ax2, ax3 = axes

        colors1 = [palette[w] for w in weekdays]
        ax1.scatter(reduced[:, 0], reduced[:, 1], c=colors1, s=22, alpha=0.95)
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=palette[i], markersize=7, label=weekday_names[i])
                   for i in range(7)]
        ax1.legend(handles=handles, title="weekday", loc='best', frameon=True)
        ax1.set_title("day t-SNE — color by weekday")

        sc2 = ax2.scatter(reduced[:, 0], reduced[:, 1], c=weeks, s=22, alpha=0.95, cmap='plasma')
        cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("week index")
        ax2.set_title("day t-SNE — color by week")

        colors3 = [palette[w] for w in weekdays]
        ax3.scatter(reduced[:, 0], reduced[:, 1], c=colors3, s=22, alpha=0.95)
        for i, (px, py) in enumerate(reduced):
            ax3.text(px, py, str(weeks[i]), fontsize=7, alpha=0.8,
                     ha='center', va='center')
        ax3.set_title("day t-SNE — original (weekday color + week text)")

        plt.tight_layout()
        wandb.log({f"codebook/{feature_name}_tsne": wandb.Image(fig), "epoch": epoch + 1})
        plt.close(fig)

    else:
        ids_one_based = _to_one_based(ids_np) if np.min(ids_np) == 0 else ids_np
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(reduced[:, 0], reduced[:, 1], c=ids_one_based, s=20, alpha=0.85, cmap='viridis')
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=f"{feature_name} (display)")
        ax.set_title(f"{feature_name} codebook t-SNE")
        wandb.log({f"codebook/{feature_name}_tsne": wandb.Image(fig), "epoch": epoch + 1})
        plt.close(fig)


def log_codebook_heatmap(feature_name, emb_np, epoch=0, ids_np=None, num_location_ids=None):
    """
    Plot cosine-similarity heatmap after filtering PAD/MASK if ids_np is provided.
    """
    if ids_np is not None:
        mask_idx = None
        if feature_name == "location" and num_location_ids is not None:
            mask_idx = int(num_location_ids) - 1
        emb_np, _ = _filter_out_padding_and_mask(feature_name, emb_np, ids_np, pad_idx=0, mask_idx=mask_idx)

    if emb_np is None or emb_np.shape[0] == 0:
        return

    if emb_np.shape[0] > 500:
        idx = np.random.RandomState(42).choice(emb_np.shape[0], 500, replace=False)
        E = emb_np[idx]
    else:
        E = emb_np

    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim = E @ E.T

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(sim, cmap='viridis', xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(f"{feature_name} codebook cosine similarity")
    wandb.log({f"codebook/{feature_name}_heatmap": wandb.Image(fig), "epoch": epoch + 1})
    plt.close(fig)