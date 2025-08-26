# --- Standard library ---
import os

# --- Third-party ---
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import seaborn as sns
import wandb

# Matplotlib: headless backend (safe for servers)
os.environ["MPLBACKEND"] = "Agg"     # 환경변수로 백엔드 지정 (우선 적용)
import matplotlib
matplotlib.use("Agg")                # 이중 안전망 (선택)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def id_to_xy(loc_ids, grid_width=200):
    """Convert location IDs to (x, y) coordinates."""
    loc_ids = np.asarray(loc_ids).astype(np.int64)
    x = (loc_ids // grid_width) + 1
    y = (loc_ids % grid_width) + 1
    return np.stack([x, y], axis=1)  # (N,2)


@torch.no_grad()
def compute_feature_codebook(model, feature_name, device,
                             n_loc_samples=2000,
                             delta_max=48*75, delta_steps=256):
    model.eval()
    if not hasattr(model, "embedding"):
        raise AttributeError("model.embedding 이 없습니다.")

    blocks = model.embedding.feature_blocks
    # === 추가된 가드 ===
    if not blocks or feature_name not in blocks:
        print(f"[Viz] feature '{feature_name}' not found in EmbeddingLayer; skipping codebook.")
        return None, None
    # ===================

    fblock = blocks[feature_name].to(device)

    if feature_name == "day":
        ids = torch.arange(75, device=device, dtype=torch.long).unsqueeze(0)
    elif feature_name == "time":
        ids = torch.arange(48, device=device, dtype=torch.long).unsqueeze(0)
    elif feature_name == "dow":
        ids = torch.arange(7, device=device, dtype=torch.long).unsqueeze(0)
    elif feature_name == "weekday":
        ids = torch.arange(2, device=device, dtype=torch.long).unsqueeze(0)
    elif feature_name == "location":
        V = blocks["location"].categorical_emb.num_embeddings
        if V > n_loc_samples:
            sel = torch.randperm(V, device=device)[:n_loc_samples]
            ids = sel.unsqueeze(0).to(torch.long)
        else:
            ids = torch.arange(V, device=device, dtype=torch.long).unsqueeze(0)
    elif feature_name == "delta":
        if (delta_steps is None) or (int(delta_steps) >= int(delta_max) + 1):
            vals = torch.arange(0, int(delta_max) + 1, device=device, dtype=torch.long)
        else:
            lin = torch.linspace(0, int(delta_max), steps=int(delta_steps), device=device)
            vals = torch.round(lin).to(torch.long).unique(sorted=True)
        ids = vals.unsqueeze(0)
    else:
        raise ValueError(f"알 수 없는 feature: {feature_name}")

    emb = fblock(ids)                                     # (1, N, D)
    emb_np = emb.squeeze(0).detach().cpu().numpy()        # (N, D)
    ids_np = ids.squeeze(0).detach().cpu().numpy()        # (N,)
    return emb_np, ids_np


def log_codebook_tsne(feature_name, emb_np, ids_np, epoch=0, perplexity=15, grid_width=200):
    """
    Codebook embedding t-SNE visualization.

    location → 3 subplots:
      (1) color by x only
      (2) color by y only
      (3) color by flattened location id (original style)

    day → 3 subplots:
      (1) color by weekday (Mon~Sun)
      (2) color by week index (continuous colormap; same week = same color)
      (3) original: weekday colors + week number text labels

    others → single plot (as before).
    """
    if emb_np is None or ids_np is None:
        return

    X = normalize(emb_np, norm='l2')
    n = X.shape[0]
    safe_perpl = max(5, min(perplexity, (n - 1) // 3 if n > 10 else 5))

    tsne = TSNE(n_components=2, perplexity=safe_perpl, metric='cosine', random_state=42)
    reduced = tsne.fit_transform(X)

    if feature_name == "location":
        # --- Prepare xy and normalization ---
        xy = id_to_xy(ids_np, grid_width=grid_width)  # (N,2), 1-based
        x, y = xy[:, 0], xy[:, 1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax1, ax2, ax3 = axes

        # (1) color by x only
        sc1 = ax1.scatter(reduced[:, 0], reduced[:, 1], c=x, s=18, alpha=0.9, cmap='Reds')
        cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("x (grid)")
        ax1.set_title("location t-SNE — color by x")

        # (2) color by y only
        sc2 = ax2.scatter(reduced[:, 0], reduced[:, 1], c=y, s=18, alpha=0.9, cmap='Greens')
        cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("y (grid)")
        ax2.set_title("location t-SNE — color by y")

        # (3) original: color by flattened id (ids_np)
        sc3 = ax3.scatter(reduced[:, 0], reduced[:, 1], c=ids_np, s=18, alpha=0.9, cmap='viridis')
        cbar3 = plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("location id (flattened)")
        ax3.set_title("location t-SNE — color by id")

        plt.tight_layout()
        wandb.log({f"codebook/{feature_name}_tsne": wandb.Image(fig), "epoch": epoch + 1})
        plt.close(fig)

    elif feature_name == "day":
        # Weekday palette & names
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c',
                   '#d62728', '#9467bd', '#8c564b', '#17becf']
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        weekdays = (ids_np % 7).astype(int)   # 0..6
        weeks = (ids_np // 7 + 1).astype(int) # 1..N

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax1, ax2, ax3 = axes

        # (1) color by weekday (categorical)
        colors1 = [palette[w] for w in weekdays]
        ax1.scatter(reduced[:, 0], reduced[:, 1], c=colors1, s=22, alpha=0.95)
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=palette[i], markersize=7, label=weekday_names[i])
                   for i in range(7)]
        ax1.legend(handles=handles, title="weekday", loc='best', frameon=True)
        ax1.set_title("day t-SNE — color by weekday")

        # (2) color by week index (continuous colormap)
        sc2 = ax2.scatter(reduced[:, 0], reduced[:, 1], c=weeks, s=22, alpha=0.95, cmap='plasma')
        cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("week index")
        ax2.set_title("day t-SNE — color by week")

        # (3) original: weekday colors + week number text
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
        # default single plot (unchanged)
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(reduced[:, 0], reduced[:, 1], c=ids_np, s=20, alpha=0.85, cmap='viridis')
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=f"{feature_name} value")
        ax.set_title(f"{feature_name} codebook t-SNE")
        wandb.log({f"codebook/{feature_name}_tsne": wandb.Image(fig), "epoch": epoch + 1})
        plt.close(fig)


def log_codebook_heatmap(feature_name, emb_np, epoch=0):
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