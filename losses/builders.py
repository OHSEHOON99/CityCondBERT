import torch
import torch.nn as nn

from .losses import DistanceDecayedCrossEntropy, GeoBleuSinkhornLoss, CE_GeoBLEU_Combo



# ---------------------------------------
# build_criterion (supports optional torch.compile)
# ---------------------------------------
def build_criterion(name, compile_mode=None, **kwargs):
    """
    Build loss criterion.

    Args:
        name (str): one of {"ce", "ddce", "geobleu", "combo"}.
        compile_mode (str or None): torch.compile mode; e.g., "default", "reduce-overhead", "max-autotune".
        **kwargs: arguments specific to each loss type.

    Supported kwargs:
      - ce:
          ignore_index, reduction, etc.
      - ddce:
          H, W, win, beta, cell_km_x, cell_km_y, distance_scale,
          ignore_index, reduction
      - geobleu:
          H, W, n_list, win, beta, cell_km_x, cell_km_y, distance_scale,
          eps, n_iters, weights
      - combo:
          {
            "ce_name": "ce" | "ddce",
            "ce_kwargs": {...},
            "geobleu_kwargs": {...},
            "alpha_init": 1.0,
            "ema_m": 0.99,
            "track_mavg": True,
            "skip_geobleu_when_alpha_ge": 0.999
          }

    Returns:
        nn.Module: loss function (optionally compiled)
    """
    name = str(name).lower()

    # CrossEntropy
    if name == "ce":
        ce_kwargs = dict(kwargs)
        if ce_kwargs.get("ignore_index") is None:
            ce_kwargs.pop("ignore_index", None)
        criterion = nn.CrossEntropyLoss(**ce_kwargs)

    # Distance-decayed CE
    elif name == "ddce":
        criterion = DistanceDecayedCrossEntropy(**kwargs)

    # GEO-BLEU Sinkhorn loss
    elif name == "geobleu":
        geobleu_loss = GeoBleuSinkhornLoss(**kwargs)
        setattr(geobleu_loss, "expects_masked_inputs", True)
        criterion = geobleu_loss

    # Combo: CE + GEO-BLEU (with α schedule)
    elif name == "combo":
        ce_name = kwargs.get("ce_name", "ce").lower()
        ce_kwargs = dict(kwargs.get("ce_kwargs", {}))
        geobleu_kwargs = dict(kwargs.get("geobleu_kwargs", {}))

        alpha_init = float(kwargs.get("alpha_init", 1.0))
        ema_m = float(kwargs.get("ema_m", 0.99))
        track_mavg = bool(kwargs.get("track_mavg", True))
        skip_thr = float(kwargs.get("skip_geobleu_when_alpha_ge", 0.999))

        # build CE component
        if ce_name == "ce":
            if ce_kwargs.get("ignore_index") is None:
                ce_kwargs.pop("ignore_index", None)
            ce_loss = nn.CrossEntropyLoss(**ce_kwargs)
        elif ce_name == "ddce":
            ce_loss = DistanceDecayedCrossEntropy(**ce_kwargs)
        else:
            raise ValueError(f"Unknown ce_name for combo: {ce_name}")

        # build GEO-BLEU component
        geobleu_loss = GeoBleuSinkhornLoss(**geobleu_kwargs)
        setattr(geobleu_loss, "expects_masked_inputs", True)

        combo = CE_GeoBLEU_Combo(
            ce_loss,
            geobleu_loss,
            alpha_init=alpha_init,
            track_mavg=track_mavg,
            m=ema_m,
            skip_geobleu_when_alpha_ge=skip_thr,
        )
        setattr(combo, "expects_masked_inputs", True)
        criterion = combo

    else:
        raise ValueError(f"Unknown loss name: {name}")

    # Optionally compile (PyTorch 2.1+)
    if compile_mode is not None and hasattr(torch, "compile"):
        try:
            criterion = torch.compile(criterion, mode=compile_mode)
        except Exception as e:
            print(f"[build_criterion] torch.compile failed ({e}); returning uncompiled criterion.")

    return criterion