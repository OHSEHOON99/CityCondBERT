def default_feature_configs():
    return {
        "day":     {"modes": ["categorical"], "combine_mode_inner": "cat"},
        "time":    {"modes": ["categorical", "periodic"], "combine_mode_inner": "cat"},
        "dow":     {"modes": ["categorical", "periodic"], "combine_mode_inner": "cat"},
        "weekday": {"modes": ["categorical"],             "combine_mode_inner": "cat"},
        "location":{"modes": ["categorical"],             "combine_mode_inner": "cat"},
        "delta":   {"modes": ["fourier"],      "combine_mode_inner": "cat"}
    }

def base_defaults():
    return {
        # General ...
        "device": 0,
        "seed": 42,
        "city": "A",
        "base_path": "/home/sehoon/Desktop/0804",
        "mode": "train",
        "model_name": None,
        "wandb_api_key": "a40ff5a6f0abbaf9771e3512b265e4d7dba35e37",

        # Data ...
        "input_seq_length": 240,
        "predict_seq_length": 48,
        "look_back_len": 40,
        "split_ratio": (8, 1, 1),
        "batch_size": 128,
        "subsample": False,
        "subsample_number": 1000,

        # Model (flat for CLI) ...
        "hidden_size": 768,
        "hidden_layers": 12,
        "attention_heads": 16,
        "dropout": 0.1,
        "max_seq_length": 75*48,

        # Embedding sizes ...
        "day_embedding_size": 64,
        "time_embedding_size": 64,
        "day_of_week_embedding_size": 64,
        "weekday_embedding_size": 32,
        "location_embedding_size": 256,
        "delta_embedding_dims": (8, 8, 8),

        # 상위 결합(통일 이름)
        "feature_combine_mode": "mlp",  # cat | mlp

        # ✅ 단일 기본: 여기만 수정하면 전체 반영
        "feature_configs": default_feature_configs(),

        # Train ...
        "lr": 3e-4,
        "location_embedding_lr": None,
        "num_epochs": 100,

        # ---------------------------
        # 🔻 Loss-related defaults 🔻
        # ---------------------------
        "loss_name": "combo",  # ← combo 사용

        "H": 200, "W": 200, "cell_km_x": 0.5, "cell_km_y": 0.5,

        "ddce": {
            "win": 7, "beta": 0.5, "distance_scale": 2.0,
            "ignore_index": None, "reduction": "mean",
        },
        "geobleu": {
            "n_list": [1, 2, 3, 4, 5],
            "win": 7, "beta": 0.5, "distance_scale": 2.0,
            "eps": 0.1, "n_iters": 30, "weights": None,
        },

        # ✅ 새로 추가: combo 설정
        "combo": {
            "ce_name": "ce",         # 또는 "ddce"로 바꿔도 됨
            "ce_kwargs": { "ignore_index": None, "reduction": "mean" },
            "geobleu_kwargs": {      # GeoBLEU 인자에 H/W/셀 크기 포함
                "H": 200, "W": 200, "n_list": [1,2,3,4,5],
                "win": 7, "beta": 0.5, "cell_km_x": 0.5, "cell_km_y": 0.5,
                "distance_scale": 2.0, "eps": 0.1, "n_iters": 30, "weights": None
            },
            "alpha_init": 1.0,  # Epoch 0은 CE만
            "ema_m": 0.99,
            "track_mavg": True,

            # α 스케줄 파라미터 (train 루프에서 사용)
            "alpha_warmup_epochs": 10,   # CE-only 기간
            "alpha_transition_epochs": 20,
            "alpha_start": 0.9,         # 전환 시작시 α
            "alpha_end": 0.3,           # 전환 끝나면 α
        },
    }