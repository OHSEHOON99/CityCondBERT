import torch



def configure_optimizer(model, base_lr, location_embedding_lr=None,
                        lr_transfer=None, lr_head=None, weight_decay=0.01):
    # sane defaults
    if lr_head is None:
        lr_head = base_lr
    if lr_transfer is None:
        lr_transfer = max(base_lr * 5, 1e-4)
    lr_transfer = max(lr_transfer, base_lr)
    if location_embedding_lr is None:
        location_embedding_lr = base_lr

    # Preferred: model-defined param groups
    if hasattr(model, "param_groups") and callable(getattr(model, "param_groups")):
        groups = model.param_groups(
            lr_backbone=base_lr,
            lr_transfer=lr_transfer,
            lr_head=lr_head,
            freeze_head=False,
            lr_location=location_embedding_lr,
            weight_decay=weight_decay,
        )
        opt = torch.optim.AdamW(groups, foreach=False)
        print("[Optimizer] Using model.param_groups: "
              f"backbone={base_lr:g}, transfer={lr_transfer:g}, head={lr_head:g}, "
              f"location={location_embedding_lr:g}, wd={weight_decay}")
        return opt

    # Fallback (이 경로는 이전과 동일)
    base_params, location_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embedding.feature_blocks.location" in name:
            location_params.append(p)
        else:
            base_params.append(p)

    opt = torch.optim.AdamW(
        [{"params": base_params,    "lr": base_lr,               "weight_decay": weight_decay},
         {"params": location_params,"lr": location_embedding_lr, "weight_decay": 0.0}],
        foreach=False
    )
    print(f"[Optimizer] (fallback) base={base_lr:g} (wd={weight_decay}), "
          f"location={location_embedding_lr:g} (wd=0)")
    return opt
