import math
from collections import defaultdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------
# Geometry helpers
# ---------------------------------------
def make_local_offsets_physical(
    win=7, beta_km=0.5,
    cell_km_x=0.5, cell_km_y=0.5, distance_scale=2.0,
    *, device="cpu", dtype=torch.float32
):
    """
    Compute distance-based kernel weights: ω = exp(-beta_km * d_km)
      - win: odd integer; total neighbors K = win * win
      - d_km(dy, dx) = hypot(dy * cell_km_y, dx * cell_km_x) * distance_scale

    Returns:
        offsets  : (K, 2) int64 tensor — neighbor (Δr, Δc)
        kernel_w : (K,)  float tensor  — distance-decayed weights
    """
    assert win % 2 == 1, "win must be odd."
    r = win // 2

    dy = torch.arange(-r, r + 1, dtype=torch.int64, device=device)
    dx = torch.arange(-r, r + 1, dtype=torch.int64, device=device)
    DY, DX = torch.meshgrid(dy, dx, indexing='ij')

    offsets = torch.stack([DY, DX], dim=-1).reshape(-1, 2).contiguous()

    DYf = DY.to(dtype) * cell_km_y
    DXf = DX.to(dtype) * cell_km_x
    dist_km = torch.hypot(DYf, DXf) * distance_scale

    kernel_w = torch.exp(-beta_km * dist_km).reshape(-1).contiguous()
    return offsets, kernel_w


def id_to_rc(ids, W: int):
    r = ids // W
    c = ids % W
    return r, c


def rc_to_id(r, c, W: int):
    return r * W + c


# ---------------------------------------
# Precompute neighbor tables (for all cells)
# ---------------------------------------
@torch.no_grad()
def _precompute_neighbors_all(H, W, offsets, device="cpu"):
    """
    Precompute neighbor indices for all V=H*W cells.
    Returns:
        nb_ids_all : (V, K) long tensor
        valid_all  : (V, K) bool tensor
    """
    V = H * W
    ids = torch.arange(V, dtype=torch.long, device=device)  # (V,)
    r = ids // W
    c = ids % W

    dy = offsets[:, 0].view(1, -1)  # (1,K)
    dx = offsets[:, 1].view(1, -1)  # (1,K)

    r_nb_raw = r.unsqueeze(1) + dy  # (V,K)
    c_nb_raw = c.unsqueeze(1) + dx  # (V,K)

    valid = ((r_nb_raw >= 0) & (r_nb_raw < H) &
             (c_nb_raw >= 0) & (c_nb_raw < W))  # (V,K)

    r_nb = r_nb_raw.clamp(0, H - 1)
    c_nb = c_nb_raw.clamp(0, W - 1)
    nb_ids_all = (r_nb * W + c_nb).to(torch.long).contiguous()
    return nb_ids_all, valid.contiguous()


# ---------------------------------------
# 1-gram U-matrix via chunked gather (no sparse ops)
# ---------------------------------------
def u_1gram_probs_gather_chunked(
    p,                 # (B,T,V) softmax probabilities
    target_ids,        # (B,T)
    kernel_w,          # (K,)
    nb_ids_all,        # (V,K)
    valid_all,         # (V,K)
    H=200, W=200,
    chunk_j: int = 16  # column chunk size (trade-off)
):
    """
    Compute U[b,i,j] = sum_k p[b,i, nb_ids_all[target_ids[b,j], k]] * w[b,j,k]
      - Memory: O(T * K * chunk_j)
      - Loop: over column chunks only
    """
    B, T, V = p.shape
    assert V == H * W
    device = p.device
    dtype  = p.dtype
    K = kernel_w.numel()

    nb_ids = nb_ids_all[target_ids]  # (B,T,K)
    valid  = valid_all[target_ids]   # (B,T,K)

    w = valid.to(p.dtype) * kernel_w.to(p.dtype).view(1, 1, K)
    w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-12))

    U_list = []
    for b in range(B):
        P_b = p[b]     # (T,V)
        nb_b = nb_ids[b]
        w_b  = w[b]

        U_b = torch.empty((T, T), device=device, dtype=dtype)

        # column-chunked gather
        for j0 in range(0, T, chunk_j):
            j1 = min(j0 + chunk_j, T)
            Jc = j1 - j0

            nb_block = nb_b[j0:j1]     # (Jc,K)
            w_block  = w_b[j0:j1]      # (Jc,K)

            idx = nb_block.unsqueeze(0).expand(T, Jc, K)
            vals = P_b.unsqueeze(1).expand(T, Jc, V).gather(2, idx)
            U_block = (vals * w_block.unsqueeze(0)).sum(dim=-1)
            U_b[:, j0:j1] = U_block

        U_list.append(U_b)

    U = torch.stack(U_list, dim=0).clamp_min(1e-12)  # (B,T,T)
    return U


# ---------------------------------------
# n-gram similarity matrix S^{(n)}
# ---------------------------------------
def ngram_similarity_matrix(U, n):
    """
    S^{(n)}[b,i,j] = ∏_{k=0}^{n-1} U[b, i+k, j+k]
    U: (B,T,T) -> S: (B,I,I) with I=T-n+1
    """
    B, T, _ = U.shape
    I = T - n + 1
    if I <= 0:
        return None

    sB, sI, sJ = U.stride()
    U_view = U.as_strided(size=(B, I, I, n),
                          stride=(sB, sI, sJ, sI + sJ))
    S = U_view.prod(dim=-1)  # (B, I, I)
    return S.clamp_min(1e-12)


# ---------------------------------------
# Sinkhorn (Entropy-Regularized OT)
# ---------------------------------------
def sinkhorn_transport(S, eps=0.1, n_iters=30):
    """
    Approximate entropy-regularized OT:
      M = argmax_M <M,S> s.t. M1 = μ, Mᵀ1 = ν
    Args:
        eps: regularization strength
        n_iters: number of row/col scaling iterations
    """
    B, I, J = S.shape
    if I == 0 or J == 0:
        return torch.zeros_like(S)

    mu = torch.full((B, I), 1.0 / I, device=S.device, dtype=S.dtype)
    nu = torch.full((B, J), 1.0 / J, device=S.device, dtype=S.dtype)

    K = torch.exp(S / eps).clamp_min(1e-30)
    a = torch.ones_like(mu)
    b = torch.ones_like(nu)

    for _ in range(n_iters):
        Kb  = (K * b.unsqueeze(1)).sum(dim=-1).clamp_min(1e-30)
        a   = mu / Kb
        KTa = (K.transpose(1,2) * a.unsqueeze(1)).sum(dim=-1).clamp_min(1e-30)
        b   = nu / KTa

    M = a.unsqueeze(-1) * K * b.unsqueeze(-2)
    return M


# ---------------------------------------
# Distance-Decayed CrossEntropy (DDCE)
# ---------------------------------------
class DistanceDecayedCrossEntropy(nn.Module):
    def __init__(self, H, W, win=7, beta=0.5,
                 cell_km_x=0.2, cell_km_y=0.2, distance_scale=2.0,
                 ignore_index=None, reduction="mean"):
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.win = int(win)
        self.beta = float(beta)
        self.cell_km_x = float(cell_km_x)
        self.cell_km_y = float(cell_km_y)
        self.distance_scale = float(distance_scale)
        self.ignore_index = ignore_index
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

        offsets, kernel_w = make_local_offsets_physical(
            win=self.win, beta_km=self.beta,
            cell_km_x=self.cell_km_x, cell_km_y=self.cell_km_y,
            distance_scale=self.distance_scale,
            device="cpu", dtype=torch.float32
        )
        self.register_buffer("offsets_buf", offsets)
        self.register_buffer("kernelw_buf", kernel_w)

    def _ensure_buffers_on(self, device, dtype):
        if (not hasattr(self, "_offsets_dev")
            or self._offsets_dev.device != device):
            self._offsets_dev = self.offsets_buf.to(device=device, non_blocking=True)
        if (not hasattr(self, "_kernelw_dev")
            or self._kernelw_dev.device != device
            or self._kernelw_dev.dtype  != dtype):
            self._kernelw_dev = self.kernelw_buf.to(device=device, dtype=dtype, non_blocking=True)

    @torch.no_grad()
    def _neighbors_flat(self, ids, offsets, H, W):
        r = ids // W
        c = ids %  W
        dy = offsets[:, 0].view(1, -1)
        dx = offsets[:, 1].view(1, -1)
        r_nb_raw = r.unsqueeze(1) + dy
        c_nb_raw = c.unsqueeze(1) + dx
        valid = ((r_nb_raw >= 0) & (r_nb_raw < H) &
                 (c_nb_raw >= 0) & (c_nb_raw < W))
        r_nb = r_nb_raw.clamp(0, H - 1)
        c_nb = c_nb_raw.clamp(0, W - 1)
        nb_ids = (r_nb * W + c_nb).to(torch.long).contiguous()
        return nb_ids, valid

    def forward(self, logits, target_ids):
        EPS = 1e-12
        device, dtype = logits.device, logits.dtype
        H, W = self.H, self.W

        # flatten
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits_flat = logits.reshape(B * T, V)
            target_flat = target_ids.reshape(B * T)
        elif logits.dim() == 2:
            logits_flat = logits
            target_flat = target_ids
            V = logits.size(-1)
        else:
            raise ValueError("logits must be (B,T,V) or (N,V)")
        assert V == H * W

        if self.win == 1:
            return F.cross_entropy(
                logits_flat, target_flat,
                reduction=self.reduction,
                ignore_index=self.ignore_index if self.ignore_index is not None else -100000000
            )

        valid_mask = torch.ones_like(target_flat, dtype=torch.bool) if self.ignore_index is None \
                     else (target_flat != self.ignore_index)
        if not valid_mask.any():
            return logits_flat.new_tensor(0.0, requires_grad=True)

        t_ids = target_flat[valid_mask].long()
        logits_sel = logits_flat[valid_mask]

        # prepare cached buffers
        self._ensure_buffers_on(device, dtype)
        offsets = self._offsets_dev
        kernel_w = self._kernelw_dev

        nb_ids, valid = self._neighbors_flat(t_ids, offsets, H, W)
        w = (kernel_w.view(1, -1) * valid.to(dtype)).clamp_min(0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + EPS)

        logp = F.log_softmax(logits_sel, dim=-1)
        logp_nb = torch.gather(logp, dim=1, index=nb_ids)
        nll = -(w * logp_nb).sum(dim=-1)

        if self.reduction == "sum":  return nll.sum()
        if self.reduction == "none": return nll
        return nll.mean()


# =======================================
# GEO-BLEU Sinkhorn Loss (chunked gather + neighbor table)
# =======================================
class GeoBleuSinkhornLoss(nn.Module):
    """Differentiable GEO-BLEU loss using Sinkhorn Optimal Transport.

    Combines spatially tolerant unigram similarity (via local gather)
    with n-gram continuity and entropy-regularized OT alignment.
    """

    expects_masked_inputs = True  # supports (N,V)+mask_pos mode

    def __init__(
        self,
        H,
        W,
        n_list=(1, 2, 3, 4, 5),
        win=7,
        beta=0.5,
        cell_km_x=0.5,
        cell_km_y=0.5,
        distance_scale=2.0,
        eps=0.1,
        n_iters=30,
        weights=None,
        chunk_j=16,
    ):
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.n_list = tuple(n_list)
        self.win = int(win)
        self.beta = float(beta)
        self.cell_km_x = float(cell_km_x)
        self.cell_km_y = float(cell_km_y)
        self.distance_scale = float(distance_scale)
        self.eps = float(eps)
        self.n_iters = int(n_iters)
        self.chunk_j = int(chunk_j)

        # n-gram weights
        if weights is None:
            w = torch.full((len(self.n_list),), 1.0 / len(self.n_list), dtype=torch.float32)
        else:
            w = torch.as_tensor(weights, dtype=torch.float32)
            w = w / w.sum()
        self.register_buffer("weights_buf", w)

        # spatial kernel
        off, ker = make_local_offsets_physical(
            win=self.win,
            beta_km=self.beta,
            cell_km_x=self.cell_km_x,
            cell_km_y=self.cell_km_y,
            distance_scale=self.distance_scale,
            device="cpu",
            dtype=torch.float32,
        )
        self.register_buffer("offsets_buf", off)
        self.register_buffer("kernelw_buf", ker)

        # neighbor table
        nb_ids_all, valid_all = _precompute_neighbors_all(self.H, self.W, off, device="cpu")
        self.register_buffer("nb_ids_all", nb_ids_all)
        self.register_buffer("valid_all", valid_all)

    # -----------------------------
    # internal helpers
    # -----------------------------
    def _ensure_buffers_on(self, device, dtype):
        """Ensure all cached buffers are on correct device/dtype."""
        if (
            not hasattr(self, "_kernelw_dev")
            or self._kernelw_dev.device != device
            or self._kernelw_dev.dtype != dtype
        ):
            self._kernelw_dev = self.kernelw_buf.to(device=device, dtype=dtype, non_blocking=True)

        if not hasattr(self, "_nb_ids_all_dev") or self._nb_ids_all_dev.device != device:
            self._nb_ids_all_dev = self.nb_ids_all.to(device=device, non_blocking=True)

        if not hasattr(self, "_valid_all_dev") or self._valid_all_dev.device != device:
            self._valid_all_dev = self.valid_all.to(device=device, non_blocking=True)

        if (
            not hasattr(self, "_weights_dev")
            or self._weights_dev.device != device
            or self._weights_dev.dtype != dtype
        ):
            self._weights_dev = self.weights_buf.to(device=device, dtype=dtype, non_blocking=True)

    # -----------------------------
    # forward passes
    # -----------------------------
    def _forward_BT(self, logits_BT_V: torch.Tensor, target_BT: torch.Tensor):
        """Forward for (B,T,V) logits."""
        B, T, V = logits_BT_V.shape
        device, dtype = logits_BT_V.device, logits_BT_V.dtype
        assert V == self.H * self.W, f"V({V}) must equal H*W({self.H*self.W})."

        if target_BT.dtype != torch.long:
            target_BT = target_BT.long()

        # probability
        p = F.softmax(logits_BT_V, dim=-1)

        # ensure cached buffers
        self._ensure_buffers_on(device, dtype)

        # 1-gram similarity matrix
        U = u_1gram_probs_gather_chunked(
            p,
            target_BT,
            self._kernelw_dev,
            self._nb_ids_all_dev,
            self._valid_all_dev,
            self.H,
            self.W,
            chunk_j=self.chunk_j,
        )

        # accumulate n-gram Sinkhorn losses
        per_sample = None
        aux = {}
        for w_n, n in zip(self._weights_dev, self.n_list):
            S = ngram_similarity_matrix(U, n)
            if S is None:
                continue
            M = sinkhorn_transport(S, eps=self.eps, n_iters=self.n_iters)
            q_n = (M * S).sum(dim=(-1, -2)).clamp_min(1e-12)

            aux[f"q_{int(n)}"] = q_n.mean().item()
            term = -w_n * torch.log(q_n)
            per_sample = term if per_sample is None else per_sample + term

        if per_sample is None:
            return logits_BT_V.new_tensor(0.0, requires_grad=True), {
                f"q_{int(n)}": 0.0 for n in self.n_list
            }
        return per_sample.mean(), aux

    def forward_pre_split(self, buckets: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """Forward for grouped batches (different lengths)."""
        total_loss_sum = None
        total_B = 0
        aux_accum = defaultdict(float)
        first_logits = None

        for T, (P, Y) in buckets.items():
            if P.numel() == 0:
                continue
            if first_logits is None:
                first_logits = P
            loss_T, aux_T = self._forward_BT(P, Y)
            B_group = P.size(0)
            total_B += B_group

            total_loss_sum = (
                loss_T * B_group if total_loss_sum is None else total_loss_sum + loss_T * B_group
            )
            for k, v in aux_T.items():
                aux_accum[k] += v * B_group

        if total_B == 0:
            base = first_logits if first_logits is not None else torch.tensor(0.0, device="cpu")
            zero = base.new_tensor(0.0, requires_grad=True)
            return zero, {f"q_{int(n)}": 0.0 for n in self.n_list}

        loss = total_loss_sum / total_B
        for k in list(aux_accum.keys()):
            aux_accum[k] /= total_B
        return loss, dict(aux_accum)

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        mask_pos: torch.Tensor = None,
        buckets: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Forward with flexible input modes."""
        if buckets is not None:
            return self.forward_pre_split(buckets)
        if logits.dim() == 3:
            return self._forward_BT(logits, target_ids)

        if logits.dim() == 2:
            if mask_pos is None:
                raise ValueError(
                    "GeoBleuSinkhornLoss: (N,V) input requires mask_pos or use forward_pre_split(buckets)."
                )
            counts = mask_pos.sum(dim=1).tolist()
            nz = [c for c in counts if c > 0]
            if sum(nz) != logits.size(0):
                raise ValueError("GeoBleuSinkhornLoss: mismatch between mask_pos sum and N.")

            p_list = list(torch.split(logits, nz, dim=0))
            y_list = list(torch.split(target_ids, nz, dim=0))

            buck = defaultdict(list)
            for p_i, y_i in zip(p_list, y_list):
                buck[p_i.size(0)].append(
                    (p_i.view(1, p_i.size(0), -1), y_i.view(1, -1))
                )
            for T, items in list(buck.items()):
                P = torch.cat([p for p, _ in items], dim=0)
                Y = torch.cat([y for _, y in items], dim=0)
                buck[T] = (P, Y)
            return self.forward_pre_split(buck)

        raise ValueError(f"GeoBleuSinkhornLoss: unsupported logits shape {logits.shape}")


# =======================================
# CE + GEO-BLEU Combo Loss
# =======================================
class CE_GeoBLEU_Combo(nn.Module):
    """Joint CE + GEO-BLEU loss with learnable α weighting."""

    expects_masked_inputs = True

    def __init__(
        self,
        ce_loss: nn.Module,
        geobleu_loss: nn.Module,
        alpha_init: float = 1.0,
        track_mavg: bool = True,
        m: float = 0.99,
        skip_geobleu_when_alpha_ge: float = 0.999,
    ):
        super().__init__()
        self.ce_loss = ce_loss
        self.geobleu_loss = geobleu_loss
        self.track_mavg = bool(track_mavg)
        self.m = float(m)
        self.skip_geobleu_when_alpha_ge = float(skip_geobleu_when_alpha_ge)

        a0 = math.log(alpha_init / (1.0 - alpha_init + 1e-8) + 1e-8)
        self.a_raw = nn.Parameter(torch.tensor(a0, dtype=torch.float32), requires_grad=False)

        self.register_buffer("ce_ma", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("gb_ma", torch.tensor(1.0, dtype=torch.float32))

    # -----------------------------
    # utilities
    # -----------------------------
    @property
    def alpha(self) -> float:
        """Current α value (CE weight)."""
        return torch.sigmoid(self.a_raw).item()

    @torch.no_grad()
    def set_alpha(self, a: float):
        """Manually set α (CE weight)."""
        a = float(min(max(a, 1e-6), 1.0 - 1e-6))
        a0 = math.log(a / (1.0 - a))
        self.a_raw.copy_(torch.tensor(a0, dtype=torch.float32, device=self.a_raw.device))

    # -----------------------------
    # helpers
    # -----------------------------
    def _ce_from_buckets(self, buckets: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Compute CE over variable-length buckets."""
        ce_sum = None
        tok_sum = 0
        first_logits = None

        for T, (P, Y) in buckets.items():
            if P.numel() == 0:
                continue
            if first_logits is None:
                first_logits = P

            if isinstance(self.ce_loss, nn.CrossEntropyLoss):
                G, T_, V = P.shape
                ce_logits = P.reshape(G * T_, V)
                ce_targets = Y.reshape(G * T_).long()
                ce_val = self.ce_loss(ce_logits, ce_targets)
                tokens = G * T_
            else:
                ce_val = self.ce_loss(P, Y)
                tokens = P.size(0) * P.size(1)

            ce_sum = ce_val * tokens if ce_sum is None else ce_sum + ce_val * tokens
            tok_sum += tokens

        if tok_sum == 0:
            base = first_logits if first_logits is not None else torch.tensor(0.0, device="cpu")
            return base.new_tensor(0.0, requires_grad=True)
        return ce_sum / tok_sum

    # -----------------------------
    # main forward
    # -----------------------------
    def forward_pre_split(self, buckets: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """Forward with variable-length buckets."""
        ce = self._ce_from_buckets(buckets)
        alpha_now = torch.sigmoid(self.a_raw).detach().item()
        compute_gb = alpha_now < self.skip_geobleu_when_alpha_ge

        gb, aux = (self.geobleu_loss.forward_pre_split(buckets) if compute_gb else (ce.detach() * 0.0, {}))

        # EMA tracking
        if self.track_mavg and self.training:
            with torch.no_grad():
                self.ce_ma = self.m * self.ce_ma + (1.0 - self.m) * ce.detach()
                if compute_gb:
                    self.gb_ma = self.m * self.gb_ma + (1.0 - self.m) * gb.detach()

        # normalize
        ce_n = ce / (self.ce_ma + 1e-8)
        gb_n = gb / (self.gb_ma + 1e-8 if compute_gb else 1.0)

        alpha = torch.sigmoid(self.a_raw)
        loss = alpha * ce_n + (1.0 - alpha) * gb_n

        logs = {"alpha": alpha.item(), "ce_raw": float(ce.detach().item())}
        if compute_gb:
            logs["geobleu_raw"] = float(gb.detach().item())
            logs.update(aux)
        return loss, logs

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_pos: torch.Tensor = None,
        buckets: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Unified forward for (B,T,V), (N,V), or buckets."""
        if buckets is not None:
            return self.forward_pre_split(buckets)

        # ----- Case 1: (B,T,V) -----
        if logits.dim() == 3:
            if isinstance(self.ce_loss, nn.CrossEntropyLoss):
                B, T, V = logits.shape
                ce_logits = logits.reshape(B * T, V)
                ce_targets = targets.reshape(B * T).long()
                ce = self.ce_loss(ce_logits, ce_targets)
            else:
                ce = self.ce_loss(logits, targets)

            alpha_now = torch.sigmoid(self.a_raw).detach().item()
            compute_gb = alpha_now < self.skip_geobleu_when_alpha_ge
            gb, aux = (self.geobleu_loss(logits, targets) if compute_gb else (ce.detach() * 0.0, {}))

            # EMA
            if self.track_mavg and self.training:
                with torch.no_grad():
                    self.ce_ma = self.m * self.ce_ma + (1.0 - self.m) * ce.detach()
                    if compute_gb:
                        self.gb_ma = self.m * self.gb_ma + (1.0 - self.m) * gb.detach()

            # normalize & combine
            ce_n = ce / (self.ce_ma + 1e-8)
            gb_n = gb / (self.gb_ma + 1e-8 if compute_gb else 1.0)
            alpha = torch.sigmoid(self.a_raw)
            loss = alpha * ce_n + (1.0 - alpha) * gb_n

            logs = {"alpha": alpha.item(), "ce_raw": float(ce.detach().item())}
            if compute_gb:
                logs["geobleu_raw"] = float(gb.detach().item())
                logs.update(aux)
            return loss, logs

        # ----- Case 2: (N,V) -----
        if logits.dim() == 2:
            alpha_now = torch.sigmoid(self.a_raw).detach().item()
            compute_gb = alpha_now < self.skip_geobleu_when_alpha_ge

            # compute GeoBLEU if needed
            if compute_gb:
                if mask_pos is None:
                    raise ValueError(
                        "CE_GeoBLEU_Combo: (N,V) input requires mask_pos or use forward_pre_split(buckets)."
                    )
                counts = mask_pos.sum(dim=1).tolist()
                nz = [c for c in counts if c > 0]
                if sum(nz) != logits.size(0):
                    raise ValueError("CE_GeoBLEU_Combo: mismatch between mask_pos sum and N.")

                p_list = list(torch.split(logits, nz, dim=0))
                y_list = list(torch.split(targets, nz, dim=0))
                buck = defaultdict(list)
                for p_i, y_i in zip(p_list, y_list):
                    buck[p_i.size(0)].append((p_i.view(1, p_i.size(0), -1), y_i.view(1, -1)))
                for T, items in list(buck.items()):
                    P = torch.cat([p for p, _ in items], dim=0)
                    Y = torch.cat([y for _, y in items], dim=0)
                    buck[T] = (P, Y)
                return self.forward_pre_split(buck)

            # CE-only case
            if isinstance(self.ce_loss, nn.CrossEntropyLoss):
                ce = self.ce_loss(logits, targets.long())
            else:
                if mask_pos is None:
                    raise ValueError(
                        "Combo(DDCE): (N,V) input requires mask_pos or use forward_pre_split(buckets)."
                    )
                counts = mask_pos.sum(dim=1).tolist()
                nz = [c for c in counts if c > 0]
                p_list = list(torch.split(logits, nz, dim=0))
                y_list = list(torch.split(targets, nz, dim=0))
                buck = defaultdict(list)
                for p_i, y_i in zip(p_list, y_list):
                    buck[p_i.size(0)].append((p_i.view(1, p_i.size(0), -1), y_i.view(1, -1)))
                for T, items in list(buck.items()):
                    P = torch.cat([p for p, _ in items], dim=0)
                    Y = torch.cat([y for _, y in items], dim=0)
                    buck[T] = (P, Y)
                ce = self._ce_from_buckets(buck)

            if self.track_mavg and self.training:
                with torch.no_grad():
                    self.ce_ma = self.m * self.ce_ma + (1.0 - self.m) * ce.detach()

            ce_n = ce / (self.ce_ma + 1e-8)
            alpha = torch.sigmoid(self.a_raw)
            loss = alpha * ce_n
            logs = {"alpha": alpha.item(), "ce_raw": float(ce.detach().item())}
            return loss, logs

        raise ValueError(f"CE_GeoBLEU_Combo: unsupported logits shape {logits.shape}")