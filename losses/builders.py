import torch
import torch.nn as nn

from .losses import DistanceDecayedCrossEntropy, GeoBleuSinkhornLoss, CE_GeoBLEU_Combo


# ---------------------------------------
# build_criterion (업데이트: compile_mode 지원)
# ---------------------------------------
def build_criterion(name, compile_mode=None, **kwargs):
    """
    name ∈ {"ce", "ddce", "geobleu", "combo"}
    kwargs:
      - ce:     ignore_index, reduction 등 CrossEntropyLoss 인자
      - ddce:   H,W,win,beta,cell_km_x,cell_km_y,distance_scale,ignore_index,reduction
      - geobleu:H,W,n_list,win,beta,cell_km_x,cell_km_y,distance_scale,eps,n_iters,weights
      - combo:  {
                  "ce_name": "ce"|"ddce",
                  "ce_kwargs": {...},
                  "geobleu_kwargs": {...},
                  "alpha_init": 1.0,
                  "ema_m": 0.99,
                  "track_mavg": True,
                  "skip_geobleu_when_alpha_ge": 0.999
                }

    compile_mode:
      - None(기본): torch.compile 미적용
      - "default" | "reduce-overhead" | "max-autotune" 등 (PyTorch 2.1+)
    """
    name = str(name).lower()

    if name == "ce":
        ce_kwargs = dict(kwargs)
        if "ignore_index" in ce_kwargs and ce_kwargs["ignore_index"] is None:
            ce_kwargs.pop("ignore_index")
        criterion = nn.CrossEntropyLoss(**ce_kwargs)

    elif name == "ddce":
        criterion = DistanceDecayedCrossEntropy(**kwargs)

    elif name == "geobleu":
        geobleu_loss = GeoBleuSinkhornLoss(**kwargs)
        setattr(geobleu_loss, "expects_masked_inputs", True)
        criterion = geobleu_loss

    elif name == "combo":
        ce_name = kwargs.get("ce_name", "ce").lower()
        ce_kwargs = dict(kwargs.get("ce_kwargs", {}))
        geobleu_kwargs = dict(kwargs.get("geobleu_kwargs", {}))

        alpha_init = float(kwargs.get("alpha_init", 1.0))
        ema_m = float(kwargs.get("ema_m", 0.99))
        track_mavg = bool(kwargs.get("track_mavg", True))
        skip_thr = float(kwargs.get("skip_geobleu_when_alpha_ge", 0.999))

        if ce_name == "ce":
            if "ignore_index" in ce_kwargs and ce_kwargs["ignore_index"] is None:
                ce_kwargs.pop("ignore_index")
            ce_loss = nn.CrossEntropyLoss(**ce_kwargs)
        elif ce_name == "ddce":
            ce_loss = DistanceDecayedCrossEntropy(**ce_kwargs)
        else:
            raise ValueError(f"Unknown ce_name for combo: {ce_name}")

        geobleu_loss = GeoBleuSinkhornLoss(**geobleu_kwargs)
        setattr(geobleu_loss, "expects_masked_inputs", True)

        combo = CE_GeoBLEU_Combo(
            ce_loss, geobleu_loss,
            alpha_init=alpha_init,
            track_mavg=track_mavg, m=ema_m,
            skip_geobleu_when_alpha_ge=skip_thr,
        )
        setattr(combo, "expects_masked_inputs", True)
        criterion = combo

    else:
        raise ValueError(f"Unknown loss name: {name}")

    if compile_mode is not None and hasattr(torch, "compile"):
        try:
            criterion = torch.compile(criterion, mode=compile_mode)
        except Exception as e:
            print(f"[build_criterion] torch.compile failed ({e}), returning uncompiled criterion.")
    return criterion