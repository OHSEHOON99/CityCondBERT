import argparse



def parse_tuple(s):
    try:
        return tuple(int(x.strip()) for x in s.strip('()').split(','))
    except:
        raise argparse.ArgumentTypeError("Expected format: '(32,64,128)'")


def get_pretrain_parser():
    p = argparse.ArgumentParser(description='Pretrain CityCondBERT with canonical config.')

    # Config
    p.add_argument('--config', type=str, default=None, help='Input config.json path (optional)')

    # General
    p.add_argument('--device', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--base_path', type=str, default=None)
    p.add_argument('--wandb_api_key', type=str, default=None)
    p.add_argument('--wandb_project', type=str, default=None)

    # Data
    p.add_argument('--split_ratio', type=parse_tuple, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--subsample', action='store_true')
    p.add_argument('--subsample_number', type=int, default=None)
    p.add_argument('--mask_days', type=int, default=None)

    # Model (BERT backbone)
    p.add_argument('--hidden_size', type=int, default=None)
    p.add_argument('--hidden_layers', type=int, default=None)
    p.add_argument('--attention_heads', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--max_seq_length', type=int, default=None)

    # Embedding sizes
    p.add_argument('--day_embedding_size', type=int, default=None)
    p.add_argument('--time_embedding_size', type=int, default=None)
    p.add_argument('--day_of_week_embedding_size', type=int, default=None)
    p.add_argument('--weekday_embedding_size', type=int, default=None)
    p.add_argument('--location_embedding_size', type=int, default=None)
    p.add_argument('--delta_embedding_dims', type=parse_tuple, default=None)

    # Embedding combine
    p.add_argument('--feature_combine_mode', type=str, default=None)

    # Transfer knobs
    p.add_argument('--apply_film_at', type=str, choices=['none', 'pre', 'post', 'both'], default=None)
    p.add_argument('--film_share', type=str, choices=['true', 'false'], default=None)
    p.add_argument('--num_cities', type=int, default=None)
    p.add_argument('--city_emb_dim', type=int, default=None)
    p.add_argument('--use_adapter', type=str, choices=['true', 'false'], default=None)
    p.add_argument('--adapter_layers', type=int, default=None)
    p.add_argument('--adapter_r', type=int, default=None)
    p.add_argument('--adapter_dropout', type=float, default=None)

    # Optimizer / training
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--location_embedding_lr', type=float, default=None)
    p.add_argument('--lr_transfer', type=float, default=None)
    p.add_argument('--lr_head', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--num_epochs', type=int, default=None)
    p.add_argument('--use_amp', type=str, choices=['true', 'false'], default=None)

    # Loss
    p.add_argument('--loss_name', type=str, choices=['ce', 'ddce', 'geobleu', 'combo'], default=None)

    # Freeze (for compatibility)
    p.add_argument('--freeze_backbone', type=str, choices=['true', 'false'], default=None)
    p.add_argument('--freeze_head', type=str, choices=['true', 'false'], default=None)

    return p


def get_transfer_parser():
    p = argparse.ArgumentParser(
        description="Transfer-train CityCondBERT from a pretrained checkpoint."
    )

    # Config
    p.add_argument("--config", type=str, default=None, help="Optional config.json path")

    # General
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--base_path", type=str, default=None)
    p.add_argument("--city", type=str, required=True, help="Target city for transfer (A|B|C|D)")
    p.add_argument("--model_name", type=str, required=True,
                   help="Pretrained checkpoint dir name under checkpoints/")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)

    # Data
    p.add_argument("--split_ratio", type=parse_tuple, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--subsample", action="store_true")
    p.add_argument("--subsample_number", type=int, default=None)
    p.add_argument("--mask_days", type=int, default=None)

    # Optimizer / training
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--location_embedding_lr", type=float, default=None)
    p.add_argument("--lr_transfer", type=float, default=None)
    p.add_argument("--lr_head", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--use_amp", type=str, choices=["true", "false"], default=None)

    # Loss
    p.add_argument("--loss_name", type=str,
                   choices=["ce", "ddce", "geobleu", "combo"], default=None)

    # Optional freezing (useful in transfer learning)
    p.add_argument("--freeze_backbone", type=str, choices=["true", "false"], default=None)
    p.add_argument("--freeze_head", type=str, choices=["true", "false"], default=None)

    return p


def get_predict_parser():
    p = argparse.ArgumentParser(
        description="Run masked-UID prediction (x==999) for a single city using a trained model."
    )
    # Config
    p.add_argument("--config", type=str, default=None, help="Optional config.json path")

    # General
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--base_path", type=str, default=None)
    p.add_argument("--city", type=str, help="Target city (A|B|C|D)")
    p.add_argument("--model_name", type=str, required=True,
                   help="Checkpoint dir name under checkpoints/")

    # Optional overrides
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size for mask loader")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output dir (default: <ckpt_dir>/results)")
    return p


def coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "t", "yes", "y")
    return bool(v)