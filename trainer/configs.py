# ----------------------------
# Canonical config builder
# ----------------------------
def to_canonical_config(cfg, num_location_ids):
    # Transformer (canonical or legacy)
    transformer_cfg = cfg.get("transformer", {
        "hidden_size":     cfg["hidden_size"],
        "hidden_layers":   cfg["hidden_layers"],
        "attention_heads": cfg["attention_heads"],
        "dropout":         cfg["dropout"],
        "max_seq_length":  cfg["max_seq_length"],
    })

    # Embedding sizes (canonical or legacy)
    embedding_sizes = cfg.get("embedding_sizes", {
        "day":      cfg["day_embedding_size"],
        "time":     cfg["time_embedding_size"],
        "dow":      cfg["day_of_week_embedding_size"],
        "weekday":  cfg["weekday_embedding_size"],
        "location": cfg["location_embedding_size"],
    })

    feature_configs = cfg["feature_configs"]
    feature_combine_mode = cfg.get(
        "feature_combine_mode",
        cfg.get("embedding_combine_mode", "cat")
    )

    delta_embedding_dims = tuple(cfg["delta_embedding_dims"])

    return {
        "schema_version": "1.0.0",
        "city": cfg["city"],
        "num_location_ids": int(num_location_ids),
        "transformer": transformer_cfg,
        "embedding_sizes": embedding_sizes,
        "delta_embedding_dims": list(delta_embedding_dims),
        "feature_configs": feature_configs,
        "feature_combine_mode": feature_combine_mode,
    }


def make_loss_kwargs_from_cfg(cfg):
    """Build loss function kwargs from configuration."""
    H = int(cfg.get("H", 200))
    W = int(cfg.get("W", 200))
    cell_km_x = float(cfg.get("cell_km_x", 0.5))
    cell_km_y = float(cfg.get("cell_km_y", 0.5))

    loss_name = cfg.get("loss_name", "ce").lower()

    if loss_name == "ce":
        return {}

    elif loss_name == "ddce":
        d = cfg.get("ddce", {})
        return {
            "H": H, "W": W,
            "win": int(d.get("win", 7)),
            "beta": float(d.get("beta", 0.5)),
            "cell_km_x": cell_km_x, "cell_km_y": cell_km_y,
            "distance_scale": float(d.get("distance_scale", 2.0)),
            "ignore_index": d.get("ignore_index", None),
            "reduction": d.get("reduction", "mean"),
        }

    elif loss_name == "geobleu":
        g = cfg.get("geobleu", {})
        return {
            "H": H, "W": W,
            "n_list": tuple(g.get("n_list", [1, 2, 3, 4, 5])),
            "win": int(g.get("win", 7)),
            "beta": float(g.get("beta", 0.5)),
            "cell_km_x": cell_km_x, "cell_km_y": cell_km_y,
            "distance_scale": float(g.get("distance_scale", 2.0)),
            "eps": float(g.get("eps", 0.1)),
            "n_iters": int(g.get("n_iters", 30)),
            "weights": g.get("weights", None),
        }

    elif loss_name == "combo":
        # Combo loss configuration
        c = cfg.get("combo", {})
        ce_name = c.get("ce_name", "ce").lower()

        # CE kwargs
        if ce_name == "ddce":
            base = cfg.get("ddce", {})
            ck = c.get("ce_kwargs", {})
            ce_kwargs = {
                "H": H, "W": W,
                "win": int(ck.get("win", base.get("win", 7))),
                "beta": float(ck.get("beta", base.get("beta", 0.5))),
                "cell_km_x": cell_km_x, "cell_km_y": cell_km_y,
                "distance_scale": float(ck.get("distance_scale", base.get("distance_scale", 2.0))),
                "ignore_index": ck.get("ignore_index", base.get("ignore_index", None)),
                "reduction": ck.get("reduction", base.get("reduction", "mean")),
            }
        else:
            ce_kwargs = c.get("ce_kwargs", {})

        # GeoBLEU kwargs
        gk = c.get("geobleu_kwargs", {})
        geobleu_kwargs = {
            "H": int(gk.get("H", H)),
            "W": int(gk.get("W", W)),
            "n_list": tuple(gk.get("n_list", cfg.get("geobleu", {}).get("n_list", [1, 2, 3, 4, 5]))),
            "win": int(gk.get("win", cfg.get("geobleu", {}).get("win", 7))),
            "beta": float(gk.get("beta", cfg.get("geobleu", {}).get("beta", 0.5))),
            "cell_km_x": float(gk.get("cell_km_x", cell_km_x)),
            "cell_km_y": float(gk.get("cell_km_y", cell_km_y)),
            "distance_scale": float(gk.get("distance_scale", cfg.get("geobleu", {}).get("distance_scale", 2.0))),
            "eps": float(gk.get("eps", cfg.get("geobleu", {}).get("eps", 0.1))),
            "n_iters": int(gk.get("n_iters", cfg.get("geobleu", {}).get("n_iters", 30))),
            "weights": gk.get("weights", cfg.get("geobleu", {}).get("weights", None)),
        }

        return {
            "ce_name": ce_name,
            "ce_kwargs": ce_kwargs,
            "geobleu_kwargs": geobleu_kwargs,
            "alpha_init": float(c.get("alpha_init", 1.0)),
            "ema_m": float(c.get("ema_m", 0.99)),
            "track_mavg": bool(c.get("track_mavg", True)),
            # Optionally skip GeoBLEU when alpha is high for faster training
            "skip_geobleu_when_alpha_ge": float(c.get("skip_geobleu_when_alpha_ge", 0.999)),
        }

    else:
        raise ValueError(f"[cfg] Unknown loss_name: {loss_name}")


def alpha_sched(epoch: int, combo_cfg: dict) -> float:
    """
    Compute alpha schedule based on combo_cfg.
    - α = 1.0 during warmup (CE only)
    - Linearly decays from a_start → a_end during transition
    """
    e_warm = int(combo_cfg.get("alpha_warmup_epochs", 5))
    e_trans = int(combo_cfg.get("alpha_transition_epochs", 3))
    a_start = float(combo_cfg.get("alpha_start", 0.9))
    a_end = float(combo_cfg.get("alpha_end", 0.3))

    if epoch < e_warm:
        return 1.0

    r = min(1.0, (epoch - e_warm) / max(1, e_trans))
    return a_start + (a_end - a_start) * r
