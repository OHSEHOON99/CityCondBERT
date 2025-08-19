import torch
import torch.nn as nn
from spatiotemporal_embedding import PeriodicEncoding, LearnableFourierFeatures


class FeatureBlock(nn.Module):
    """
    단일 feature(day, time, dow, weekday, location, delta)에 대해
    categorical / periodic / fourier 표현을 선택적으로 생성하고
    내부 결합(cat/sum/mlp)으로 최종 임베딩을 반환하는 블록.
    """
    def __init__(self,
                 # categorical
                 categorical_dim=None,
                 categorical_num=None,
                 # periodic
                 periodic_dim=None,
                 periodic_period=None,
                 # fourier
                 fourier_dims=None,            # (f_dim, h_dim, d_dim)
                 fourier_log1p=False,          # True면 log1p 입력(연속값 권장: delta), False면 원값
                 # 선택/결합
                 use_modes=("categorical",),   # ("categorical", "periodic", "fourier") 중 조합
                 combine_mode_inner="cat"):    # "cat" | "sum" | "mlp"
        super().__init__()
        self.use_modes = tuple(use_modes)
        self.combine_mode_inner = combine_mode_inner
        self.fourier_log1p = fourier_log1p

        # === categorical ===
        if "categorical" in self.use_modes:
            assert categorical_dim is not None and categorical_num is not None, \
                "[FeatureBlock] categorical requires categorical_dim & categorical_num"
            self.categorical_emb = nn.Embedding(categorical_num, categorical_dim)

        # === periodic ===
        if "periodic" in self.use_modes:
            assert periodic_dim is not None and periodic_period is not None, \
                "[FeatureBlock] periodic requires periodic_dim & periodic_period"
            self.periodic_enc = PeriodicEncoding(model_dim=periodic_dim,
                                                 period=periodic_period)

        # === fourier ===
        if "fourier" in self.use_modes:
            assert fourier_dims is not None and len(fourier_dims) == 3, \
                "[FeatureBlock] fourier requires (f_dim, h_dim, d_dim)"
            f_dim, h_dim, d_dim = fourier_dims
            self.fourier_enc = LearnableFourierFeatures(pos_dim=1,
                                                        f_dim=f_dim,
                                                        h_dim=h_dim,
                                                        d_dim=d_dim)

        # === 차원 산정 ===
        sizes = []
        if "categorical" in self.use_modes: sizes.append(categorical_dim)
        if "periodic"    in self.use_modes: sizes.append(periodic_dim)
        if "fourier"     in self.use_modes: sizes.append(fourier_dims[-1])

        # 내부 결합 방식에 따른 최종 out_dim 계산 및 검증
        self.total_size = sum(sizes)             # cat 기준 합산 차원
        self.max_size   = max(sizes) if sizes else 0  # sum/mlp 기준 최대 차원

        if self.combine_mode_inner == "cat":
            self.out_dim = self.total_size
            self._mlp = None
        elif self.combine_mode_inner == "sum":
            # sum은 모든 표현 차원이 같아야 함
            assert len(set(sizes)) <= 1, \
                f"[FeatureBlock] inner=sum requires equal dims, got {sizes}"
            self.out_dim = sizes[0] if sizes else 0
            self._mlp = None
        elif self.combine_mode_inner == "mlp":
            # concat → MLP → max_size로 압축
            self.out_dim = self.max_size
            self._mlp = nn.Sequential(
                nn.Linear(self.total_size, self.out_dim),
                nn.ReLU(),
                nn.Linear(self.out_dim, self.out_dim)
            )
        else:
            raise ValueError(f"[FeatureBlock] Unknown combine_mode_inner: {self.combine_mode_inner}")

    def forward(self, x):
        """
        x: (B, T)  정수 인덱스(카테고리/주기형) 또는 정수/실수(연속값)
           fourier의 경우 (B, T, 1)도 허용
        """
        outs = []

        # categorical
        if "categorical" in self.use_modes:
            outs.append(self.categorical_emb(x))  # (B,T,Dc)

        # periodic
        if "periodic" in self.use_modes:
            # PeriodicEncoding이 정수 index 또는 float를 받아 period 기반 사인/코사인을 생성한다고 가정
            outs.append(self.periodic_enc(x))     # (B,T,Dp)

        # fourier
        if "fourier" in self.use_modes:
            if x.dim() == 2:     # (B,T)
                x_in = x.unsqueeze(-1).float()
            else:                # (B,T,1)
                x_in = x.float()
            if self.fourier_log1p:
                x_in = torch.log1p(x_in)
            outs.append(self.fourier_enc(x_in))   # (B,T,Df)

        # 내부 결합
        if len(outs) == 1:
            return outs[0]

        if self.combine_mode_inner == "cat":
            return torch.cat(outs, dim=-1)                # (B,T,sum D)
        elif self.combine_mode_inner == "sum":
            return torch.stack(outs, dim=0).sum(dim=0)    # (B,T,D)
        elif self.combine_mode_inner == "mlp":
            return self._mlp(torch.cat(outs, dim=-1))     # (B,T,max D)
        else:
            raise ValueError(f"[FeatureBlock] Unknown combine_mode_inner: {self.combine_mode_inner}")


class EmbeddingLayer(nn.Module):
    """
    여러 FeatureBlock의 출력을 합쳐 최종 임베딩 시퀀스를 만드는 레이어.
    - 외부 결합: "cat" | "mlp"  (sum은 제외)
    - 미래 구간의 location은 zero-vector로 대체 + LN 이후 재마스킹으로 "진짜 0" 복구
    - 모델에서 쓰기 편하도록 output_dim 노출(과거 코드 호환용 total_size/max_size도 제공)
    """
    def __init__(self, num_location_ids,
                 day_embedding_size, time_embedding_size, dow_embedding_size,
                 weekday_embedding_size, location_embedding_size, delta_embedding_dims,
                 feature_configs,
                 combine_mode="cat", dropout=0.1):
        super().__init__()

        self.combine_mode = combine_mode
        self.feature_blocks = nn.ModuleDict()

        # === FeatureBlock 정의 (삽입 순서가 출력 결합 순서가 됨) ===
        if "day" in feature_configs:
            cfg = feature_configs["day"]
            self.feature_blocks["day"] = FeatureBlock(
                categorical_dim=day_embedding_size, categorical_num=75,
                periodic_dim=day_embedding_size,    periodic_period=75,
                fourier_dims=delta_embedding_dims,
                fourier_log1p=False,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"]
            )

        if "time" in feature_configs:
            cfg = feature_configs["time"]
            self.feature_blocks["time"] = FeatureBlock(
                categorical_dim=time_embedding_size, categorical_num=48,
                periodic_dim=time_embedding_size,    periodic_period=48,
                fourier_dims=delta_embedding_dims,
                fourier_log1p=False,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"]
            )

        if "dow" in feature_configs:
            cfg = feature_configs["dow"]
            self.feature_blocks["dow"] = FeatureBlock(
                categorical_dim=dow_embedding_size, categorical_num=7,
                periodic_dim=dow_embedding_size,    periodic_period=7,
                fourier_dims=delta_embedding_dims,
                fourier_log1p=False,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"]
            )

        if "weekday" in feature_configs:
            cfg = feature_configs["weekday"]
            self.feature_blocks["weekday"] = FeatureBlock(
                categorical_dim=weekday_embedding_size, categorical_num=2,
                fourier_dims=delta_embedding_dims,
                fourier_log1p=False,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"]
            )

        if "location" in feature_configs:
            cfg = feature_configs["location"]
            self.feature_blocks["location"] = FeatureBlock(
                categorical_dim=location_embedding_size, categorical_num=num_location_ids,
                fourier_dims=delta_embedding_dims,
                fourier_log1p=False,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"]
            )

        if "delta" in feature_configs:
            cfg = feature_configs["delta"]
            self.feature_blocks["delta"] = FeatureBlock(
                categorical_dim=cfg.get("categorical_dim", None),
                categorical_num=cfg.get("categororical_num", None),  # 주의: 오타 방지 필요시 수정
                periodic_dim=cfg.get("periodic_dim", None),
                periodic_period=cfg.get("periodic_period", None),
                fourier_dims=delta_embedding_dims,
                fourier_log1p=True,
                use_modes=cfg["modes"],
                combine_mode_inner=cfg["combine_mode_inner"]
            )

        # === 블록 출력 차원 수집 및 feature 순서/슬라이스 계산 ===
        self.block_out_dims = {name: fb.out_dim for name, fb in self.feature_blocks.items()}
        self.feature_order = list(self.feature_blocks.keys())  # 삽입 순서 보존
        # 각 feature가 concat 상에서 차지하는 구간(start, end) 미리 계산
        self.feature_slices_cat = {}
        offset = 0
        for name in self.feature_order:
            d = self.block_out_dims[name]
            self.feature_slices_cat[name] = (offset, offset + d)
            offset += d

        # === 외부 결합 정의 및 최종 차원 ===
        if self.combine_mode == "cat":
            self.final_dim = sum(self.block_out_dims.values()) if self.block_out_dims else 0
            # cat 경로: post-LN 용
            self.layer_norm = nn.LayerNorm(self.final_dim) if self.final_dim > 0 else nn.Identity()
            # mlp 관련 모듈은 사용하지 않음
            self._mlp = None
            self.pre_mlp_layer_norm = None

        elif self.combine_mode == "mlp":
            # concat → (pre) LN(cat_dim) → MLP(first linear bias=False) → (post) LN(final_dim)
            self.cat_dim = sum(self.block_out_dims.values()) if self.block_out_dims else 0
            self.final_dim = max(self.block_out_dims.values()) if self.block_out_dims else 0
            # pre-MLP LN (cat 공간에서)
            self.pre_mlp_layer_norm = nn.LayerNorm(self.cat_dim) if self.cat_dim > 0 else nn.Identity()
            # 첫 Linear는 bias=False로 "0 기여" 보존
            self._mlp = nn.Sequential(
                nn.Linear(self.cat_dim, self.final_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.final_dim, self.final_dim, bias=True)
            )
            # post-MLP LN
            self.layer_norm = nn.LayerNorm(self.final_dim) if self.final_dim > 0 else nn.Identity()
        else:
            raise ValueError(f"[EmbeddingLayer] Unknown combine_mode: {self.combine_mode}")

        # 모델에서 쓰기 편하도록 노출(과거 코드 호환까지)
        self.output_dim = self.final_dim
        self.total_size = sum(self.block_out_dims.values()) if self.block_out_dims else 0
        self.max_size   = max(self.block_out_dims.values()) if self.block_out_dims else 0

        self.dropout = nn.Dropout(dropout)

        # location 블록 차원/슬라이스 캐시
        if "location" in self.feature_slices_cat:
            self.loc_slice_cat = self.feature_slices_cat["location"]  # (start, end) in cat space
            self.loc_dim = self.block_out_dims["location"]
        else:
            self.loc_slice_cat = None
            self.loc_dim = 0

    def _make_feature_list(self, seq_feature, loc_feature, is_future):
        """Feature별 텐서를 리스트로 생성 (name, tensor) 튜플 리스트 반환"""
        outs = []
        for fname, fblock in self.feature_blocks.items():
            if fname == "location":
                if is_future:
                    # 미래 구간: location은 0 벡터 (fblock.out_dim 유지)
                    feat = torch.zeros(seq_feature.size(0), seq_feature.size(1),
                                       fblock.out_dim, device=seq_feature.device, dtype=torch.float32)
                else:
                    feat = fblock(loc_feature)  # (B,T) → (B,T,D_loc)
            elif fname == "day":
                feat = fblock(seq_feature[:, :, 0])
            elif fname == "time":
                feat = fblock(seq_feature[:, :, 1])
            elif fname == "dow":
                feat = fblock(seq_feature[:, :, 2])
            elif fname == "weekday":
                feat = fblock(seq_feature[:, :, 3])
            elif fname == "delta":
                feat = fblock(seq_feature[:, :, 4])
            else:
                raise ValueError(f"[EmbeddingLayer] Unknown feature name: {fname}")
            outs.append((fname, feat))
        return outs

    def embed_sequence(self, seq_feature, loc_feature=None, is_future=False):
        """
        seq_feature : (B, T, 5) → [day(0), time(1), dow(2), weekday(3), delta(4)]
        loc_feature : (B, T)    → 위치 id
        is_future   : True면 location은 zero-vector로 대체
        """
        outs = self._make_feature_list(seq_feature, loc_feature, is_future)

        if self.combine_mode == "cat":
            # 1) concat (cat 공간 E_cat == final_dim)
            cat = torch.cat([f for _, f in outs], dim=-1) if outs else torch.empty(0, device=seq_feature.device)

            # 2) LN (전 채널) → 3) 미래면 location 슬라이스만 0으로 재마스킹 → 4) Dropout
            cat = self.layer_norm(cat)
            if is_future and self.loc_slice_cat is not None:
                s, e = self.loc_slice_cat
                cat[..., s:e] = 0.0  # LN으로 변한 값 복구

            out = self.dropout(cat)
            return out

        elif self.combine_mode == "mlp":
            # 1) concat (cat 공간; 차원 = self.cat_dim)
            cat = torch.cat([f for _, f in outs], dim=-1) if outs else torch.empty(0, device=seq_feature.device)

            # 2) (pre) LN(cat_dim) → 3) 미래면 location 구간만 0으로 재마스킹
            if isinstance(self.pre_mlp_layer_norm, nn.Identity) or self.cat_dim == 0:
                cat_norm = cat
            else:
                cat_norm = self.pre_mlp_layer_norm(cat)

            if is_future and self.loc_slice_cat is not None:
                s, e = self.loc_slice_cat
                cat_norm[..., s:e] = 0.0  # 첫 Linear(bias=False) 전, "진짜 0" 보장

            # 4) MLP (첫 Linear bias=False) → 5) (post) LN → 6) Dropout
            mid = self._mlp(cat_norm) if self._mlp is not None else cat_norm
            out = self.layer_norm(mid)
            out = self.dropout(out)
            return out

        else:
            raise ValueError(f"[EmbeddingLayer] Unknown combine_mode: {self.combine_mode}")

    def forward(self, hist_seq, hist_loc, future_seq):
        """
        hist_seq   : (B, T_hist, 5) → [day, time, dow, weekday, delta]
        hist_loc   : (B, T_hist)
        future_seq : (B, T_future, 5)
        """
        historical_input = self.embed_sequence(hist_seq, hist_loc, is_future=False)  # (B, T_hist, E)
        future_input     = self.embed_sequence(future_seq, None, is_future=True)     # (B, T_fut , E)
        return historical_input, future_input