import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math



# ---------------------------------------
# build_criterion (수정본)
# ---------------------------------------
def build_criterion(name, **kwargs):
    """
    name ∈ {"ce", "ddce", "geobleu", "combo"}
    kwargs:
      - ce:     ignore_index, reduction 등 CrossEntropyLoss 인자
      - ddce:   H,W,win,beta,cell_km_x,cell_km_y,distance_scale,ignore_index,reduction
      - geobleu:H,W,n_list,win,beta,cell_km_x,cell_km_y,distance_scale,eps,n_iters,weights
      - combo:  {
                  "ce_name": "ce"|"ddce",
                  "ce_kwargs": {...},           # ce_name에 맞는 kwargs
                  "geobleu_kwargs": {...},      # GeoBleuSinkhornLoss kwargs
                  "alpha_init": 1.0,
                  "ema_m": 0.99,
                  "track_mavg": True,
                  "skip_geobleu_when_alpha_ge": 0.999
                }
    """
    name = name.lower()
    if name == "ce":
        # 🔧 sanitize: None이면 키 제거 (PyTorch CE는 None 허용 X)
        if "ignore_index" in kwargs and kwargs["ignore_index"] is None:
            kwargs = {k: v for k, v in kwargs.items() if k != "ignore_index"}
        return nn.CrossEntropyLoss(**kwargs)

    elif name == "ddce":
        return DistanceDecayedCrossEntropy(**kwargs)

    elif name == "geobleu":
        return GeoBleuSinkhornLoss(**kwargs)

    elif name == "combo":
        ce_name = kwargs.get("ce_name", "ce").lower()
        ce_kwargs = dict(kwargs.get("ce_kwargs", {}))
        geobleu_kwargs = kwargs.get("geobleu_kwargs", {})
        alpha_init = float(kwargs.get("alpha_init", 1.0))
        ema_m = float(kwargs.get("ema_m", 0.99))
        track_mavg = bool(kwargs.get("track_mavg", True))
        skip_thr = float(kwargs.get("skip_geobleu_when_alpha_ge", 0.999))

        # CE 파트 생성
        if ce_name == "ce":
            # 🔧 sanitize 여기서도
            if "ignore_index" in ce_kwargs and ce_kwargs["ignore_index"] is None:
                ce_kwargs.pop("ignore_index")
            ce_loss = nn.CrossEntropyLoss(**ce_kwargs)
        elif ce_name == "ddce":
            ce_loss = DistanceDecayedCrossEntropy(**ce_kwargs)
        else:
            raise ValueError(f"Unknown ce_name: {ce_name}")

        # GeoBLEU 파트 생성
        geobleu_loss = GeoBleuSinkhornLoss(**geobleu_kwargs)

        # 콤보 래퍼 반환
        return CE_GeoBLEU_Combo(
            ce_loss, geobleu_loss,
            alpha_init=alpha_init,
            track_mavg=track_mavg, m=ema_m,
            skip_geobleu_when_alpha_ge=skip_thr
        )

    else:
        raise ValueError(f"Unknown loss name: {name}")


def make_local_offsets_physical(
    win=7, beta_km=0.5,
    cell_km_x=0.5, cell_km_y=0.5, distance_scale=2.0,
    *, device="cpu", dtype=torch.float32
):
    """
    물리 거리 기반 커널 가중치 ω = exp(-beta_km * d_km).
    - win: 홀수, 총 이웃 수 K = win*win
    - d_km(dy,dx) = hypot(dy*cell_km_y, dx*cell_km_x) * distance_scale

    반환:
      offsets  : (K, 2) int64 on device   — 각 이웃의 (Δr, Δc)
      kernel_w : (K,)  dtype on device    — 각 이웃의 가중치
    """
    assert win % 2 == 1, "win must be odd."
    r = win // 2

    # 불필요한 캐스팅 제거: 처음부터 int64로 생성
    dy = torch.arange(-r, r + 1, dtype=torch.int64, device=device)  # (win,)
    dx = torch.arange(-r, r + 1, dtype=torch.int64, device=device)  # (win,)
    DY, DX = torch.meshgrid(dy, dx, indexing='ij')                  # (win, win)

    # offsets: (K, 2) int64 on device
    offsets = torch.stack([DY, DX], dim=-1).reshape(-1, 2).contiguous()

    # 거리(km) 계산: dtype/device 일관성 유지
    DYf = DY.to(dtype) * cell_km_y
    DXf = DX.to(dtype) * cell_km_x
    dist_km = torch.hypot(DYf, DXf) * distance_scale               # (win, win), dtype on device

    # 커널 가중치: (K,) dtype on device
    kernel_w = torch.exp(-beta_km * dist_km).reshape(-1).contiguous()

    return offsets, kernel_w


def id_to_rc(ids, W: int):
    """일렬 id -> (row, col). ids: int tensor, W: int"""
    # 불필요한 torch.div 옵션 제거, 정수 나눗셈으로 단순화
    r = ids // W
    c = ids % W
    return r, c


def rc_to_id(r, c, W: int):
    """(row, col) -> 일렬 id. r,c: int tensor, W: int"""
    return r * W + c


@torch.no_grad()
def _gather_neighbor_indices(target_ids, offsets, H, W):
    """
    각 참조 시점 u의 셀 id에서 offsets(Δr,Δc)로 이동한 이웃 셀 id (B,T,K)와
    격자 내 유효 마스크 (B,T,K)를 반환.
    """
    device = target_ids.device
    if offsets.device != device:
        offsets = offsets.to(device)  # 한 번만 이동

    r_u, c_u = id_to_rc(target_ids, W)   # (B,T)
    r_u = r_u.unsqueeze(-1)              # (B,T,1)
    c_u = c_u.unsqueeze(-1)

    dy = offsets[:, 0].view(1, 1, -1)    # (1,1,K)
    dx = offsets[:, 1].view(1, 1, -1)    # (1,1,K)

    r_nb_raw = r_u + dy
    c_nb_raw = c_u + dx

    valid = ((r_nb_raw >= 0) & (r_nb_raw < H) &
             (c_nb_raw >= 0) & (c_nb_raw < W)).to(torch.float32)  # (B,T,K)

    r_nb = r_nb_raw.clamp(0, H - 1)
    c_nb = c_nb_raw.clamp(0, W - 1)

    nb_ids = rc_to_id(r_nb, c_nb, W).contiguous()  # (B,T,K)

    return nb_ids, valid


# ---------- 1-그램 유사도 U (메모리-세이프) ----------
def u_1gram_probs_memsafe(
    p,                 # (B,T,V) softmax 확률
    target_ids,        # (B,T)
    offsets,           # (K,2)
    kernel_w,          # (K,)
    H=200, W=200
):
    """
    U[b,t,u] = sum_{k in N(u)} ω_k * p[b,t, v_k]
    반환: U ∈ (B,T,T)
    """
    device, dtype = p.device, p.dtype
    B, T, _ = p.shape
    K = offsets.shape[0]

    # (B,T,K), (B,T,K)
    nb_ids, valid = _gather_neighbor_indices(target_ids, offsets, H, W)
    nb_ids = nb_ids.contiguous()

    # 가중치 한 번만 준비: (B,T,K)
    kernel_w = kernel_w.to(device=device, dtype=dtype).view(1, 1, K)
    weights = (valid.to(dtype) * kernel_w)  # (B,T,K)

    # 출력 초기화
    U = p.new_zeros((B, T, T))

    # 인덱스 평탄화: (B, T*K)
    nb_ids_flat = nb_ids.view(B, -1)

    for t in range(T):
        # (B,V)
        probs_t = p[:, t, :]
        # (B, T*K) -> (B, T, K)
        probs_nb = probs_t.gather(1, nb_ids_flat).view(B, T, K)
        # (B, T)
        U[:, t, :] = (probs_nb * weights).sum(dim=-1)

    return U.clamp_min(1e-12)


# ---------- n-그램 유사도 S^{(n)} ----------
def ngram_similarity_matrix(U, n):
    """
    S^{(n)}[b,i,j] = prod_{k=0}^{n-1} U[b, i+k, j+k]
    U: (B,T,T) -> S: (B,I,I) with I=T-n+1
    """
    B, T, _ = U.shape
    I = T - n + 1
    if I <= 0:
        return None

    # U[b, i+k, j+k]를 (B, I, I, n) 뷰로 구성
    sB, sI, sJ = U.stride()  # strides in elements
    U_view = U.as_strided(size=(B, I, I, n),
                          stride=(sB, sI, sJ, sI + sJ))
    S = U_view.prod(dim=-1)  # (B, I, I)
    return S.clamp_min(1e-12)


# ---------- Sinkhorn (Entropy-OT) ----------
def sinkhorn_transport(S, eps=0.1, n_iters=30):
    """
    엔트로피 정규화된 OT로 M = argmax_M <M,S> (s.t. M 1 = mu, M^T 1 = nu) 근사
    - eps(ε): 정규화 강도 (작을수록 날카로운 매칭)
    - n_iters: 행/열 스케일링 반복 횟수
    """
    B, I, J = S.shape
    if I == 0 or J == 0:
        return torch.zeros_like(S)

    mu = torch.full((B, I), 1.0 / I, device=S.device, dtype=S.dtype)
    nu = torch.full((B, J), 1.0 / J, device=S.device, dtype=S.dtype)

    K = torch.exp(S / eps).clamp_min(1e-30)  # (B,I,J)
    a = torch.ones_like(mu)
    b = torch.ones_like(nu)

    for _ in range(n_iters):
        Kb  = (K * b.unsqueeze(1)).sum(dim=-1).clamp_min(1e-30)        # (B,I)
        a   = mu / Kb
        KTa = (K.transpose(1,2) * a.unsqueeze(1)).sum(dim=-1).clamp_min(1e-30)  # (B,J)
        b   = nu / KTa

    M = a.unsqueeze(-1) * K * b.unsqueeze(-2)  # (B,I,J)
    return M


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

        # offsets, kernel 미리 생성해서 buffer로 등록
        offsets, kernel_w = make_local_offsets_physical(
            win=self.win, beta_km=self.beta,
            cell_km_x=self.cell_km_x, cell_km_y=self.cell_km_y,
            distance_scale=self.distance_scale,
            device="cpu", dtype=torch.float32
        )
        self.register_buffer("offsets_buf", offsets)  # (K,2) int64
        self.register_buffer("kernelw_buf", kernel_w) # (K,)  float32

    @torch.no_grad()
    def _neighbors_flat(self, ids, offsets, H, W):
        """
        ids: (N,) long
        반환: nb_ids(N,K) long, valid(N,K) bool
        """
        device = ids.device
        r = ids // W                   # (N,)
        c = ids %  W                   # (N,)
        dy = offsets[:, 0].to(device).view(1, -1)  # (1,K)
        dx = offsets[:, 1].to(device).view(1, -1)  # (1,K)

        r_nb_raw = r.unsqueeze(1) + dy            # (N,K)
        c_nb_raw = c.unsqueeze(1) + dx            # (N,K)

        valid = ((r_nb_raw >= 0) & (r_nb_raw < H) &
                 (c_nb_raw >= 0) & (c_nb_raw < W))  # bool (N,K)

        # out-of-bound은 clamp로 안전 인덱스로 보정 (valid가 False라 가중치 0 됨)
        r_nb = r_nb_raw.clamp(0, H - 1)
        c_nb = c_nb_raw.clamp(0, W - 1)
        nb_ids = (r_nb * W + c_nb).to(torch.long).contiguous()
        return nb_ids, valid

    def forward(self, logits, target_ids):
        EPS = 1e-12
        device, dtype = logits.device, logits.dtype
        H, W = self.H, self.W

        # (B,T,V) 또는 (N,V) flatten
        if logits.dim() == 3:
            B, T, V = logits.shape
            N = B * T
            logits_flat = logits.reshape(N, V)
            target_flat = target_ids.reshape(N)
        elif logits.dim() == 2:
            N, V = logits.shape
            logits_flat = logits
            target_flat = target_ids
        else:
            raise ValueError("logits must be (B,T,V) or (N,V)")

        assert V == H * W, f"V({V}) must equal H*W({H*W})"

        # fast path: win==1 → 표준 CE
        if self.win == 1:
            if self.ignore_index is None:
                return F.cross_entropy(logits_flat, target_flat, reduction=self.reduction)
            else:
                return F.cross_entropy(logits_flat, target_flat,
                                       ignore_index=self.ignore_index, reduction=self.reduction)

        # 무시 라벨 마스크
        valid_mask = torch.ones_like(target_flat, dtype=torch.bool) \
            if self.ignore_index is None else (target_flat != self.ignore_index)

        if not valid_mask.any():
            return logits_flat.new_tensor(0.0, requires_grad=True)

        t_ids = target_flat[valid_mask].to(torch.long)  # (N_valid,)
        logits_sel = logits_flat[valid_mask]            # (N_valid, V)

        # offsets, kernel → device/dtype 맞추기
        offsets = self.offsets_buf.to(device)
        kernel_w = self.kernelw_buf.to(device=device, dtype=dtype)

        nb_ids, valid = self._neighbors_flat(t_ids, offsets, H, W)  # (N_valid,K), bool
        w = (kernel_w.view(1, -1) * valid.to(dtype)).clamp_min(0.0) # (N_valid,K)
        w = w / (w.sum(dim=-1, keepdim=True) + EPS)

        logp = F.log_softmax(logits_sel, dim=-1)        # (N_valid,V)
        logp_nb = torch.gather(logp, dim=1, index=nb_ids)  # (N_valid,K)
        nll = -(w * logp_nb).sum(dim=-1)                # (N_valid,)

        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "none":
            return nll
        else:
            return nll.mean()


class GeoBleuSinkhornLoss(nn.Module):
    def __init__(self, H, W, n_list=(1,2,3,4,5), win=7,
                 beta=0.5, cell_km_x=0.5, cell_km_y=0.5, distance_scale=2.0,
                 eps=0.1, n_iters=30, weights=None):
        super().__init__()
        # 고정 파라미터
        self.H, self.W = int(H), int(W)
        self.n_list = tuple(n_list)
        self.win = int(win)
        self.beta = float(beta)
        self.cell_km_x = float(cell_km_x)
        self.cell_km_y = float(cell_km_y)
        self.distance_scale = float(distance_scale)
        self.eps = float(eps)
        self.n_iters = int(n_iters)

        # weights: 한 번만 만들어 normalize 후 buffer로 보관
        if weights is None:
            w = torch.full((len(self.n_list),), 1.0 / len(self.n_list), dtype=torch.float32)
        else:
            w = torch.as_tensor(weights, dtype=torch.float32)
            w = w / w.sum()
        self.register_buffer("weights_buf", w)  # (Nn,)

        # offsets / kernel_w: CPU/float32로 생성해 buffer로 보관 (forward에서 device/dtype 맞춤)
        off, ker = make_local_offsets_physical(
            win=self.win, beta_km=self.beta,
            cell_km_x=self.cell_km_x, cell_km_y=self.cell_km_y,
            distance_scale=self.distance_scale,
            device="cpu", dtype=torch.float32
        )
        self.register_buffer("offsets_buf", off)   # (K,2) int64
        self.register_buffer("kernelw_buf", ker)   # (K,)  float32

    def forward(self, logits, target_ids):
        """
        logits: (B,T,V) — softmax 전 로짓
        target_ids: (B,T) — 정답 id (0..V-1)
        반환: loss(scalar), aux(dict: q_n 평균)
        """
        _, _, V = logits.shape
        device, dtype = logits.device, logits.dtype
        assert V == self.H * self.W, f"V({V}) must equal H*W({self.H*self.W})."

        # 1) 확률
        p = F.softmax(logits, dim=-1)

        # 2) 커널/오프셋을 현재 device/dtype으로
        offsets = self.offsets_buf.to(device)                               # (K,2) int64
        kernel_w = self.kernelw_buf.to(device=device, dtype=dtype)          # (K,)
        weights = self.weights_buf.to(device=device, dtype=dtype)           # (Nn,)

        # 3) 1-그램 유사도 U (B,T,T)
        U = u_1gram_probs_memsafe(p, target_ids, offsets, kernel_w, self.H, self.W)

        # 4) n-그램 유사도 + Sinkhorn + q_n
        loss_terms = []
        aux = {}
        for w_n, n in zip(weights, self.n_list):
            S = ngram_similarity_matrix(U, n)        # (B,I,I) or None
            if S is None:
                continue

            # sinkhorn 구현
            M = sinkhorn_transport(S, eps=self.eps, n_iters=self.n_iters)   # (B,I,I)

            I = S.shape[1]
            q_n = (M * S).sum(dim=(-1, -2)) / max(I, 1)                     # (B,)
            q_n = q_n.clamp_min(1e-12)

            aux[f"q_{n}"] = q_n.mean().item()
            loss_terms.append(- w_n * torch.log(q_n))                       # (B,)

        if not loss_terms:
            # 시퀀스가 너무 짧아 모든 n이 스킵된 경우
            return logits.new_tensor(0.0, requires_grad=True), {f"q_{n}": 0.0 for n in self.n_list}

        loss = torch.stack(loss_terms, dim=0).sum(dim=0).mean()             # 배치 평균
        return loss, aux



class CE_GeoBLEU_Combo(nn.Module):
    """
    loss = alpha * CE + (1-alpha) * GeoBLEU
    - EMA 정규화로 두 손실 스케일 자동 보정
    - alpha는 set_alpha(...)로 외부 스케줄에서 업데이트
    - alpha가 거의 1일 때 GeoBLEU 계산 스킵(옵션)
    """
    def __init__(self, ce_loss: nn.Module, geobleu_loss: nn.Module,
                 alpha_init: float = 1.0, track_mavg: bool = True, m: float = 0.99,
                 skip_geobleu_when_alpha_ge: float = 0.999):
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

    @property
    def alpha(self) -> float:
        return torch.sigmoid(self.a_raw).item()

    @torch.no_grad()
    def set_alpha(self, a: float):
        a = float(min(max(a, 1e-6), 1.0 - 1e-6))
        a0 = math.log(a / (1.0 - a))
        self.a_raw.copy_(torch.tensor(a0, dtype=torch.float32, device=self.a_raw.device))

    def forward(self, logits, targets):
        """
        logits: (B,T,V) or (N,V)
        targets: (B,T) or (N,)
        """
        # ---- CE ----
        if isinstance(self.ce_loss, nn.CrossEntropyLoss):
            if logits.dim() == 3:
                B, T, V = logits.shape
                ce_logits = logits.reshape(B * T, V)
                ce_targets = targets.reshape(B * T)
            else:
                ce_logits = logits
                ce_targets = targets
            if ce_targets.dtype != torch.long:
                ce_targets = ce_targets.long()
            ce = self.ce_loss(ce_logits, ce_targets)
        else:
            ce = self.ce_loss(logits, targets)

        # ---- GeoBLEU (alpha가 거의 1이면 스킵) ----
        alpha_now = torch.sigmoid(self.a_raw).detach().item()
        compute_gb = alpha_now < self.skip_geobleu_when_alpha_ge
        if compute_gb:
            gb, aux = self.geobleu_loss(logits, targets)
        else:
            gb, aux = ce.detach() * 0.0, {}

        # ---- EMA 정규화 (학습 모드에서만 갱신) ----
        # 🔸 CHANGED: self.training 조건 추가
        if self.track_mavg and self.training:
            with torch.no_grad():
                self.ce_ma = self.m * self.ce_ma + (1.0 - self.m) * ce.detach()
                if compute_gb:
                    self.gb_ma = self.m * self.gb_ma + (1.0 - self.m) * gb.detach()

        ce_n = ce / (self.ce_ma + 1e-8)
        gb_n = gb / (self.gb_ma + 1e-8 if compute_gb else 1.0)

        alpha = torch.sigmoid(self.a_raw)
        loss = alpha * ce_n + (1.0 - alpha) * gb_n

        logs = {"alpha": alpha.item(), "ce_raw": float(ce.detach().item())}
        if compute_gb:
            logs["geobleu_raw"] = float(gb.detach().item())
            logs.update(aux)
        return loss, logs