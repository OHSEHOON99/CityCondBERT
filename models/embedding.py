import torch
import torch.nn as nn
from .spatiotemporal_embedding import PeriodicEncoding, LearnableFourierFeatures


class FeatureBlock(nn.Module):
    """
    Single feature block supporting categorical / periodic / Fourier encodings.

    Args:
        categorical_dim: dimension of categorical embedding
        categorical_num: number of categories (including PAD)
        categorical_padding_idx: padding index
        periodic_dim: dimension of periodic encoding
        periodic_period: period value (e.g., 48 for time-of-day)
        periodic_assume_shifted: if True, assume +1-shifted integer input (0=PAD)
        fourier_dims: (f_dim, h_dim, d_dim)
        fourier_log1p: apply log1p before Fourier encoding
        use_modes: which encoders to use ("categorical", "periodic", "fourier")
        combine_mode_inner: "cat" | "sum" | "mlp"
    """
    def __init__(
        self,
        categorical_dim=None,
        categorical_num=None,
        categorical_padding_idx=None,
        periodic_dim=None,
        periodic_period=None,
        periodic_assume_shifted=True,
        fourier_dims=None,
        fourier_log1p=False,
        use_modes=("categorical",),
        combine_mode_inner="cat"
    ):
        super().__init__()
        self.use_modes = tuple(use_modes)
        self.combine_mode_inner = combine_mode_inner
        self.fourier_log1p = fourier_log1p
        self.periodic_assume_shifted = periodic_assume_shifted

        # --- categorical ---
        if "categorical" in self.use_modes:
            assert categorical_dim is not None and categorical_num is not None
            self.categorical_emb = nn.Embedding(
                num_embeddings=categorical_num,
                embedding_dim=categorical_dim,
                padding_idx=categorical_padding_idx
            )

        # --- periodic ---
        if "periodic" in self.use_modes:
            assert periodic_dim is not None and periodic_period is not None
            self.periodic_enc = PeriodicEncoding(model_dim=periodic_dim, period=periodic_period)

        # --- fourier ---
        if "fourier" in self.use_modes:
            assert fourier_dims is not None and len(fourier_dims) == 3
            f_dim, h_dim, d_dim = fourier_dims
            self.fourier_enc = LearnableFourierFeatures(pos_dim=1, f_dim=f_dim, h_dim=h_dim, d_dim=d_dim)

        # output dimension
        sizes = []
        if "categorical" in self.use_modes:
            sizes.append(categorical_dim)
        if "periodic" in self.use_modes:
            sizes.append(periodic_dim)
        if "fourier" in self.use_modes:
            sizes.append(fourier_dims[-1] if fourier_dims else 0)

        self.total_size = sum(sizes)
        self.max_size = max(sizes) if sizes else 0

        if combine_mode_inner == "cat":
            self.out_dim = self.total_size
            self._mlp = None
        elif combine_mode_inner == "sum":
            assert len(set(sizes)) <= 1, "sum mode requires equal dims"
            self.out_dim = sizes[0] if sizes else 0
            self._mlp = None
        elif combine_mode_inner == "mlp":
            self.out_dim = self.max_size
            self._mlp = nn.Sequential(
                nn.Linear(self.total_size, self.out_dim),
                nn.ReLU(),
                nn.Linear(self.out_dim, self.out_dim)
            )
        else:
            raise ValueError(f"Unknown combine_mode_inner: {combine_mode_inner}")

    def forward(self, x):
        """
        Args:
            x: (B, T)
        """
        outs = []

        # categorical
        if "categorical" in self.use_modes:
            outs.append(self.categorical_emb(x))

        # periodic
        if "periodic" in self.use_modes:
            if self.periodic_assume_shifted:
                core = torch.where(x > 0, x - 1, x)
            else:
                core = x
            outs.append(self.periodic_enc(core))

        # fourier
        if "fourier" in self.use_modes:
            x_in = x.unsqueeze(-1).float() if x.dim() == 2 else x.float()
            if self.fourier_log1p:
                x_in = torch.log1p(x_in)
            outs.append(self.fourier_enc(x_in))

        if len(outs) == 1:
            y = outs[0]
        elif self.combine_mode_inner == "cat":
            y = torch.cat(outs, dim=-1)
        elif self.combine_mode_inner == "sum":
            y = torch.stack(outs, dim=0).sum(dim=0)
        elif self.combine_mode_inner == "mlp":
            y = self._mlp(torch.cat(outs, dim=-1))
        else:
            raise ValueError(f"Unknown combine_mode_inner: {self.combine_mode_inner}")
        return y


class EmbeddingLayer(nn.Module):
    """
    Unified embedding layer for spatio-temporal features.

    Args:
        seq_feature: (B,T,5) [day(+1), time(+1), dow(+1), weekday(+1), delta(raw)]
        loc_feature: (B,T) {0=PAD, 1..V, V+1=[MASK]}
    """
    def __init__(
        self,
        num_location_ids,
        day_embedding_size,
        time_embedding_size,
        dow_embedding_size,
        weekday_embedding_size,
        location_embedding_size,
        delta_embedding_dims,
        feature_configs,
        combine_mode="cat",
        dropout=0.1
    ):
        super().__init__()
        self.combine_mode = combine_mode
        self.feature_blocks = nn.ModuleDict()

        # day
        if "day" in feature_configs:
            cfg = feature_configs["day"]
            self.feature_blocks["day"] = FeatureBlock(
                categorical_dim=day_embedding_size,
                categorical_num=75 + 1,
                categorical_padding_idx=0,
                periodic_dim=day_embedding_size if "periodic" in cfg["modes"] else None,
                periodic_period=75 if "periodic" in cfg["modes"] else None,
                periodic_assume_shifted=True,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"],
            )

        # time
        if "time" in feature_configs:
            cfg = feature_configs["time"]
            self.feature_blocks["time"] = FeatureBlock(
                categorical_dim=time_embedding_size,
                categorical_num=48 + 1,
                categorical_padding_idx=0,
                periodic_dim=time_embedding_size if "periodic" in cfg["modes"] else None,
                periodic_period=48 if "periodic" in cfg["modes"] else None,
                periodic_assume_shifted=True,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"],
            )

        # dow
        if "dow" in feature_configs:
            cfg = feature_configs["dow"]
            self.feature_blocks["dow"] = FeatureBlock(
                categorical_dim=dow_embedding_size,
                categorical_num=7 + 1,
                categorical_padding_idx=0,
                periodic_dim=dow_embedding_size if "periodic" in cfg["modes"] else None,
                periodic_period=7 if "periodic" in cfg["modes"] else None,
                periodic_assume_shifted=True,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"],
            )

        # weekday
        if "weekday" in feature_configs:
            cfg = feature_configs["weekday"]
            self.feature_blocks["weekday"] = FeatureBlock(
                categorical_dim=weekday_embedding_size,
                categorical_num=2 + 1,
                categorical_padding_idx=0,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"],
            )

        # location
        if "location" in feature_configs:
            cfg = feature_configs["location"]
            self.feature_blocks["location"] = FeatureBlock(
                categorical_dim=location_embedding_size,
                categorical_num=num_location_ids,
                categorical_padding_idx=0,
                use_modes=("categorical",),
                combine_mode_inner=cfg["combine_mode_inner"],
            )

        # delta
        if "delta" in feature_configs:
            cfg = feature_configs["delta"]
            self.feature_blocks["delta"] = FeatureBlock(
                categorical_dim=cfg.get("categorical_dim", None),
                categorical_num=cfg.get("categorical_num", None),
                periodic_dim=cfg.get("periodic_dim", None),
                periodic_period=cfg.get("periodic_period", None),
                periodic_assume_shifted=False,
                fourier_dims=delta_embedding_dims,
                fourier_log1p=True,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"],
            )

        block_out_dims = {k: fb.out_dim for k, fb in self.feature_blocks.items()}
        self.feature_order = list(self.feature_blocks.keys())
        self.cat_dim = sum(block_out_dims.values()) if block_out_dims else 0
        self.final_dim = (
            self.cat_dim if combine_mode == "cat" else max(block_out_dims.values()) if block_out_dims else 0
        )

        if combine_mode == "cat":
            self.pre_ln = None
            self._mlp = None
            self.post_ln = nn.LayerNorm(self.final_dim) if self.final_dim > 0 else nn.Identity()
        elif combine_mode == "mlp":
            self.pre_ln = nn.LayerNorm(self.cat_dim) if self.cat_dim > 0 else nn.Identity()
            self._mlp = nn.Sequential(
                nn.Linear(self.cat_dim, self.final_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.final_dim, self.final_dim, bias=True),
            )
            self.post_ln = nn.LayerNorm(self.final_dim) if self.final_dim > 0 else nn.Identity()
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

        self.dropout = nn.Dropout(dropout)
        self.output_dim = self.final_dim

    def _make_feature_list(self, seq_feature, loc_feature):
        """Apply feature-specific masking and embedding."""
        outs = []
        col = {"day": 0, "time": 1, "dow": 2, "weekday": 3, "delta": 4}

        for fname, fblock in self.feature_blocks.items():
            if fname == "location":
                feat = fblock(loc_feature)
                outs.append(feat)
                continue

            x_in = seq_feature[:, :, col[fname]]
            feat = fblock(x_in)

            if fname == "delta":
                pad_mask = seq_feature[:, :, col["day"]] == 0
            else:
                pad_mask = x_in == 0

            if pad_mask.any():
                feat = feat.masked_fill(pad_mask.unsqueeze(-1), 0.0)

            outs.append(feat)
        return outs

    def forward(self, seq_feature, loc_feature):
        """Forward pass."""
        feats = self._make_feature_list(seq_feature, loc_feature)
        cat = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

        if self.combine_mode == "cat":
            out = self.post_ln(cat)
        else:
            mid = self.pre_ln(cat)
            mid = self._mlp(mid)
            out = self.post_ln(mid)

        return self.dropout(out)