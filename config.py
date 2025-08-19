def default_feature_configs():
    return {
        "day":     {"modes": ["categorical"], "combine_mode_inner": "cat"},
        "time":    {"modes": ["categorical"], "combine_mode_inner": "cat"},
        "dow":     {"modes": ["categorical"], "combine_mode_inner": "cat"},
        "weekday": {"modes": ["categorical"],             "combine_mode_inner": "cat"},
        "location":{"modes": ["categorical"],             "combine_mode_inner": "cat"},
        "delta":   {"modes": ["fourier"],                 "combine_mode_inner": "cat"},
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
        "hidden_size": 512,
        "hidden_layers": 12,
        "attention_heads": 16,
        "dropout": 0.2,
        "max_seq_length": 75*48,

        # Embedding sizes ...
        "day_embedding_size": 32,
        "time_embedding_size": 32,
        "day_of_week_embedding_size": 16,
        "weekday_embedding_size": 8,
        "location_embedding_size": 256,
        "delta_embedding_dims": (8, 8, 8),

        # 상위 결합(통일 이름)
        "feature_combine_mode": "cat",  # cat | sum | mlp

        # ✅ 단일 기본: 여기만 수정하면 전체 반영
        "feature_configs": default_feature_configs(),

        # Train ...
        "lr": 1e-4,
        "location_embedding_lr": None,
        "num_epochs": 50,
    }