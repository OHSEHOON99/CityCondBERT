import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import seaborn as sns
import wandb

import os
os.environ["MPLBACKEND"] = "Agg"   # 가장 안전 (환경변수)

import matplotlib
matplotlib.use("Agg")              # 혹시를 위한 이중 안전망 (선택)

import matplotlib.pyplot as plt



@torch.no_grad()
def compute_feature_codebook(model, feature_name, device,
                             n_loc_samples=2000,
                             delta_max=2000, delta_steps=256):
    """
    특정 feature block에 대해 가능한 값의 '코드북 임베딩'을 계산.
    반환: emb_np (N, D), ids_np (N,)  — 여기서 ids는 시각화용 인덱스/값
    """
    model.eval()
    if not hasattr(model, "embedding"):
        raise AttributeError("model.embedding 이 없습니다.")

    blocks = model.embedding.feature_blocks
    if feature_name not in blocks:
        raise KeyError(f"feature '{feature_name}'가 EmbeddingLayer에 없습니다. 현재: {list(blocks.keys())}")

    fblock = blocks[feature_name]
    fblock = fblock.to(device)

    # --- 도메인 구성 ---
    if feature_name == "day":
        # 0..74
        ids = torch.arange(75, device=device).unsqueeze(0)   # (1,75)
    elif feature_name == "time":
        # 0..47
        ids = torch.arange(48, device=device).unsqueeze(0)   # (1,48)
    elif feature_name == "dow":
        # 0..6
        ids = torch.arange(7, device=device).unsqueeze(0)    # (1,7)
    elif feature_name == "weekday":
        # 0..1
        ids = torch.arange(2, device=device).unsqueeze(0)    # (1,2)
    elif feature_name == "location":
        V = model.embedding.feature_blocks["location"].categorical_emb.num_embeddings
        # 전체가 너무 크면 샘플링
        if V > n_loc_samples:
            sel = torch.randperm(V, device=device)[:n_loc_samples]
            ids = sel.unsqueeze(0)                           # (1,K)
        else:
            ids = torch.arange(V, device=device).unsqueeze(0) # (1,V)
    elif feature_name == "delta":
        # 연속값 → 균등 그리드(필요 시 로그그리드/분위수로 바꿔도 OK)
        vals = torch.linspace(0, float(delta_max), steps=delta_steps, device=device)  # (steps,)
        # FeatureBlock은 (B,T) 또는 (B,T,1) 허용. 내부 fourier가 (B,T,1) 처리도 가능.
        ids = vals.unsqueeze(0)                               # (1,steps)
    else:
        raise ValueError(f"알 수 없는 feature: {feature_name}")

    # --- FeatureBlock forward ---
    # 입력 shape은 (B,T). 여기선 B=1, T=N
    emb = fblock(ids)             # (1, N, D)
    emb_np = emb.squeeze(0).detach().cpu().numpy()   # (N, D)
    ids_np = ids.squeeze(0).detach().cpu().numpy()   # (N,)

    return emb_np, ids_np


def log_codebook_tsne(feature_name, emb_np, ids_np, epoch=0, perplexity=15):
    """
    코드북 임베딩의 t-SNE 시각화. ids를 색으로 표시(연속값은 그라데이션 느낌).
    """
    X = normalize(emb_np, norm='l2')
    perpl = min(perplexity, max(5, X.shape[0] // 5))
    tsne = TSNE(n_components=2, perplexity=perpl, metric='cosine', random_state=42)
    reduced = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(reduced[:, 0], reduced[:, 1], c=ids_np, s=20, alpha=0.85, cmap='viridis')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=f"{feature_name} value")
    ax.set_title(f"{feature_name} codebook t-SNE")
    wandb.log({f"codebook/{feature_name}_tsne": wandb.Image(fig), "epoch": epoch + 1})
    plt.close(fig)


def log_codebook_heatmap(feature_name, emb_np, epoch=0):
    """
    코드북 벡터 간 코사인 유사도 히트맵(값 개수가 너무 크면 주의).
    """
    if emb_np.shape[0] > 500:
        # 너무 크면 무작위 샘플로 제한
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
