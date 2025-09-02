import torch
from torch import nn
from transformers import BertModel, BertConfig
from typing import Tuple, Optional, Dict, List

from .embedding import EmbeddingLayer
from config import IGNORE_INDEX



class CityFiLM(nn.Module):
    """
    Generate per-city FiLM parameters (gamma, beta) to modulate hidden states: h' = gamma ⊙ h + beta.
    Initialized so that gamma≈1, beta≈0 at the start (no harm to pretrained backbone).
    """
    def __init__(self, num_cities: int, city_dim: int, hidden_dim: int, zero_init: bool = True):
        super().__init__()
        self.city_emb = nn.Embedding(num_cities, city_dim)
        self.to_gamma_beta = nn.Linear(city_dim, 2 * hidden_dim)
        if zero_init:
            nn.init.zeros_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, city_ids: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        city_ids: (B,)
        returns gamma, beta of shape (B, T, H)
        """
        gb = self.to_gamma_beta(self.city_emb(city_ids))  # (B, 2H)
        gamma, beta = gb.chunk(2, dim=-1)                 # (B, H), (B, H)
        gamma = 1.0 + gamma
        gamma = gamma.unsqueeze(1).expand(-1, T, -1)
        beta  = beta.unsqueeze(1).expand(-1, T, -1)
        return gamma, beta


class Adapter(nn.Module):
    """
    Lightweight residual bottleneck: x + W_up(GELU(W_down(x))).
    Up projection is zero-initialized to start as an identity mapping.
    """
    def __init__(self, d_model: int, r: int = 16, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(d_model, r)
        self.up   = nn.Linear(r, d_model)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.up(self.act(self.down(x))))


class CityCondBERT(nn.Module):
    """
    Inputs:
      - input_seq_feature: (B, L, 6)  [day, time, dow, weekday, delta, loc_input_idx]
      - attention_mask   : (B, L)     1=keep, 0=pad
      - labels           : (B, L)     masked positions in [0..V-1], others = IGNORE_INDEX
      - city_ids         : (B,)       integer city ids (e.g., {1:A,2:B,3:C,4:D})

    Outputs:
      - logits_masked: (N_masked, V)  (V = num_location_ids - 2)
      - mask_pos     : (B, L) bool
    """
    def __init__(self,
                 num_location_ids,
                 transformer_cfg,                  # {hidden_size, hidden_layers, attention_heads, dropout, max_seq_length}
                 embedding_sizes,                  # {day, time, dow, weekday, location}
                 delta_embedding_dims,             # (f_dim, h_dim, d_dim)
                 feature_configs,                  # {"day": {...}, "time": {...}, ...}
                 embedding_combine_mode: str = "cat",
                 # ---- transfer knobs ----
                 num_cities: int = 5,              # 0: pad (unused), 1..4: A..D
                 city_emb_dim: int = 32,
                 use_film: bool = True,
                 apply_film_at: str = "post",      # "none" | "pre" | "post" | "both"
                 film_share: bool = True,          # True: share pre/post FiLM params; False: split
                 use_adapter: bool = True,
                 adapter_layers: int = 2,
                 adapter_r: int = 16,
                 adapter_dropout: float = 0.0):
        super().__init__()

        # ----- Embedding layer (inputs_embeds path) -----
        self.embedding = EmbeddingLayer(
            num_location_ids=num_location_ids,
            day_embedding_size=embedding_sizes["day"],
            time_embedding_size=embedding_sizes["time"],
            dow_embedding_size=embedding_sizes["dow"],
            weekday_embedding_size=embedding_sizes["weekday"],
            location_embedding_size=embedding_sizes["location"],
            delta_embedding_dims=tuple(delta_embedding_dims),
            feature_configs=feature_configs,
            combine_mode=embedding_combine_mode,
            dropout=transformer_cfg["dropout"]
        )
        emb_out_dim = self.embedding.output_dim

        # ----- BERT backbone -----
        self.config = BertConfig(
            vocab_size=1,  # we use inputs_embeds
            hidden_size=transformer_cfg["hidden_size"],
            num_hidden_layers=transformer_cfg["hidden_layers"],
            num_attention_heads=transformer_cfg["attention_heads"],
            intermediate_size=transformer_cfg["hidden_size"] * 4,
            max_position_embeddings=transformer_cfg["max_seq_length"],
            hidden_act='gelu',
            hidden_dropout_prob=transformer_cfg["dropout"],
            attention_probs_dropout_prob=transformer_cfg["dropout"],
            initializer_range=.02,
            layer_norm_eps=1e-12
        )
        self.bert = BertModel(self.config)

        # ----- Projections -----
        self.input_projection  = nn.Linear(emb_out_dim, self.config.hidden_size, bias=False)  # preserve zeros
        self.num_location_ids = int(num_location_ids)
        self.num_classes      = self.num_location_ids - 2
        self.output_projection = nn.Linear(self.config.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(transformer_cfg["dropout"])

        # ----- FiLM / Adapter for transfer -----
        self.use_film = bool(use_film)
        valid_modes = {"none", "pre", "post", "both"}
        apply_film_at = apply_film_at.lower() if isinstance(apply_film_at, str) else "post"
        if not self.use_film:
            apply_film_at = "none"
        assert apply_film_at in valid_modes, f"apply_film_at must be one of {valid_modes}"
        self.apply_film_at = apply_film_at

        self.film_share = bool(film_share)
        self.city_film_pre: Optional[nn.Module] = None
        self.city_film_post: Optional[nn.Module] = None
        self._film_ctor_kwargs = dict(
            num_cities=num_cities,
            city_dim=city_emb_dim,
            hidden_dim=self.config.hidden_size,
        )
        if self.use_film:
            # Pre FiLM module if needed
            if self.apply_film_at in ("pre", "both"):
                self.city_film_pre = CityFiLM(**self._film_ctor_kwargs)
            # Post FiLM module if needed
            if self.apply_film_at in ("post", "both"):
                if self.film_share and self.city_film_pre is not None and self.apply_film_at == "both":
                    self.city_film_post = self.city_film_pre   # share weights
                else:
                    self.city_film_post = CityFiLM(**self._film_ctor_kwargs)

        self.use_adapter = bool(use_adapter)
        if self.use_adapter:
            self.adapter_stack = nn.ModuleList([
                Adapter(self.config.hidden_size, r=adapter_r, dropout=adapter_dropout)
                for _ in range(adapter_layers)
            ])

    # ------------------------
    # Utilities: freeze/unfreeze
    # ------------------------
    def freeze_backbone(self):
        """Freeze embedding, input projection, and BERT backbone (keeps head/FiLM/Adapter trainable)."""
        for m in [self.embedding, self.input_projection, self.bert]:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for m in [self.embedding, self.input_projection, self.bert]:
            for p in m.parameters():
                p.requires_grad = True

    def freeze_for_transfer(self, freeze_head: bool = False):
        """Convenience: freeze backbone; optionally freeze output head too."""
        self.freeze_backbone()
        if freeze_head:
            for p in self.output_projection.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    # --------- FiLM helpers (state-preserving) ---------
    def _clone_film(self, src: nn.Module) -> nn.Module:
        dst = CityFiLM(**self._film_ctor_kwargs)
        dst.load_state_dict(src.state_dict())
        return dst

    def set_film_position(self, mode: str, film_share: Optional[bool] = None):
        """
        Dynamically switch FiLM position: 'none'|'pre'|'post'|'both'
        Optionally toggle weight sharing between pre/post FiLM.
        Keeps learned states where possible to avoid performance drops.
        """
        mode = mode.lower()
        assert mode in {"none", "pre", "post", "both"}

        if film_share is not None:
            self.film_share = bool(film_share)

        # keep old handles for state preservation
        old_pre, old_post = self.city_film_pre, self.city_film_post

        # apply mode
        self.apply_film_at = "none" if not self.use_film else mode
        if not self.use_film or self.apply_film_at == "none":
            self.city_film_pre = None
            self.city_film_post = None
            return

        # rebuild with state retention
        if self.apply_film_at in ("pre", "both"):
            # reuse old pre if exists, else create new
            self.city_film_pre = old_pre or CityFiLM(**self._film_ctor_kwargs)
        else:
            self.city_film_pre = None

        if self.apply_film_at in ("post", "both"):
            if self.film_share and self.apply_film_at == "both":
                # share params: prefer existing pre; else reuse old post; else new
                if self.city_film_pre is not None:
                    self.city_film_post = self.city_film_pre
                elif old_post is not None:
                    self.city_film_post = old_post
                else:
                    self.city_film_post = CityFiLM(**self._film_ctor_kwargs)
            else:
                # split params: keep old distinct post, else clone pre, else new
                if old_post is not None and (old_post is not old_pre):
                    self.city_film_post = old_post
                elif self.city_film_pre is not None:
                    self.city_film_post = self._clone_film(self.city_film_pre)
                else:
                    self.city_film_post = CityFiLM(**self._film_ctor_kwargs)
        else:
            self.city_film_post = None

    def trainable_parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def param_groups(self,
                    lr_backbone: float,
                    lr_transfer: float,
                    lr_head: Optional[float] = None,
                    freeze_head: bool = False,
                    lr_location: Optional[float] = None,
                    weight_decay: float = 0.01) -> List[Dict]:
        """
        Optimizer param groups with LR split + no-decay split + location-embedding split.
        - lr_head: if None -> lr_backbone
        - lr_location: if None -> lr_backbone
        - freeze_head: exclude head params when True
        - weight_decay: applied only to *decay* groups
        """
        groups: List[Dict] = []
        if lr_head is None:
            lr_head = lr_backbone
        if lr_location is None:
            lr_location = lr_backbone

        # -------- helpers --------
        def is_no_decay_param(name: str) -> bool:
            # bias & LayerNorm.* (HF BERT는 LayerNorm.weight 사용)
            return (
                name.endswith(".bias")
                or name.endswith("LayerNorm.weight")
                or ".layer_norm." in name  # 안전망
                or name.endswith(".ln.weight")
            )

        # ========== BACKBONE ==========
        bb_decay, bb_nodecay, loc_params = [], [], []

        # Embedding 모듈: location embedding 분리 (wd=0)
        for n, p in self.embedding.named_parameters():
            if not p.requires_grad:
                continue
            if "feature_blocks.location" in n:
                loc_params.append(p)  # 전용 그룹 (wd=0)
            elif is_no_decay_param(n) or (".embedding" in n and n.endswith(".weight")):
                # 임베딩 weight는 보통 wd=0이 유리
                bb_nodecay.append(p)
            else:
                bb_decay.append(p)

        # input_projection
        for n, p in self.input_projection.named_parameters():
            if not p.requires_grad:
                continue
            (bb_nodecay if is_no_decay_param(n) else bb_decay).append(p)

        # BERT backbone
        for n, p in self.bert.named_parameters():
            if not p.requires_grad:
                continue
            (bb_nodecay if is_no_decay_param(n) else bb_decay).append(p)

        if bb_decay:
            groups.append({"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay})
        if bb_nodecay:
            groups.append({"params": bb_nodecay, "lr": lr_backbone, "weight_decay": 0.0})
        if loc_params:
            groups.append({"params": loc_params, "lr": lr_location, "weight_decay": 0.0})

        # ========== HEAD ==========
        if not freeze_head:
            head_decay, head_nodecay = [], []
            for n, p in self.output_projection.named_parameters():
                if not p.requires_grad:
                    continue
                (head_nodecay if is_no_decay_param(n) else head_decay).append(p)
            if head_decay:
                groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
            if head_nodecay:
                groups.append({"params": head_nodecay, "lr": lr_head, "weight_decay": 0.0})

        # ========== TRANSFER (FiLM + Adapter) ==========
        transfer_modules = []
        if self.use_film:
            if self.city_film_pre is not None:
                transfer_modules.append(self.city_film_pre)
            if self.city_film_post is not None and self.city_film_post is not self.city_film_pre:
                transfer_modules.append(self.city_film_post)
        if self.use_adapter:
            transfer_modules.append(self.adapter_stack)

        tr_decay, tr_nodecay = [], []
        for m in transfer_modules:
            for n, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                (tr_nodecay if is_no_decay_param(n) else tr_decay).append(p)

        if tr_decay:
            groups.append({"params": tr_decay, "lr": lr_transfer, "weight_decay": weight_decay})
        if tr_nodecay:
            groups.append({"params": tr_nodecay, "lr": lr_transfer, "weight_decay": 0.0})

        return groups

    # ------------
    # Forward
    # ------------
    def forward(self,
                input_seq_feature: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor,
                city_ids: torch.Tensor):
        """
        input_seq_feature : (B, L, 6)
        attention_mask    : (B, L)
        labels            : (B, L)
        city_ids          : (B,)
        """
        # 1) Embedding
        seq5 = input_seq_feature[:, :, :5]  # (B, L, 5)
        loc  = input_seq_feature[:, :, 5]   # (B, L)
        emb  = self.embedding(seq5, loc)    # (B, L, E)
        x    = self.dropout(self.input_projection(emb))  # (B, L, H)

        # (optional) pre-FiLM
        if self.use_film and self.apply_film_at in ("pre", "both"):
            assert self.city_film_pre is not None
            gamma0, beta0 = self.city_film_pre(city_ids, T=x.size(1))  # (B, L, H)
            x = gamma0 * x + beta0

        # 2) BERT backbone
        h = self.bert(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state  # (B, L, H)

        # (optional) post-FiLM
        if self.use_film and self.apply_film_at in ("post", "both"):
            assert self.city_film_post is not None
            gamma, beta = self.city_film_post(city_ids, T=h.size(1))  # (B, L, H)
            h = gamma * h + beta

        # 3) Adapters
        if self.use_adapter:
            for adapter in self.adapter_stack:
                h = adapter(h)  # (B, L, H)

        # 4) Masked positions → logits
        mask_pos = (labels != IGNORE_INDEX)  # (B, L) bool
        if not mask_pos.any().item():
            empty = h.new_zeros((0, self.num_classes))
            return empty, mask_pos

        h_masked = h[mask_pos]                           # (N_masked, H)
        logits_masked = self.output_projection(h_masked) # (N_masked, V)
        return logits_masked, mask_pos