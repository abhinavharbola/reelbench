"""
Re-runs export_embeddings against an *existing* sasrec.pt checkpoint, using
the fixed SASRec.forward (see build_attention_mask in src/models/sasrec.py).

Use this before retraining. The NaN bug that fixed forward() addresses is a
masking bug in the forward pass, not something baked into the trained
weights themselves -- so if your checkpoint's weights are otherwise healthy,
simply re-exporting with the patched code should be enough, no GPU time
needed.

This will NOT help if training itself already produced NaN parameters (the
old, buggy forward pass was also used during training, and produced NaN
loss/gradients on any batch with padding, which is nearly every batch --
see the fix's docstring). Check for that first:

    python -c "
    import torch
    ckpt = torch.load('data/processed/checkpoints/sasrec.pt', map_location='cpu')
    import itertools
    nan_params = [k for k, v in ckpt['model_state'].items() if torch.isnan(v).any()]
    print('NaN parameters in checkpoint:', nan_params or 'none')
    "

If that prints "none", run this script and re-check with
check_embeddings_for_nan.py, you likely don't need to retrain. If it lists
parameters, the weights are already corrupted and you'll need to retrain
from scratch with the fixed code (see scripts/train_sasrec.py).

Usage:
    python scripts/reexport_sasrec_embeddings.py \\
        --checkpoint-path data/processed/checkpoints/sasrec.pt \\
        --train-path data/processed/train.parquet \\
        --output-dir data/processed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.models.sasrec import build_user_sequences, export_embeddings, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    resumed = load_checkpoint(args.checkpoint_path)
    if resumed is None:
        raise FileNotFoundError(f"no checkpoint at {args.checkpoint_path}")
    model, _, epoch, id_maps, config = resumed
    print(f"loaded checkpoint from epoch {epoch}, max_seq_len={config['max_seq_len']}")

    train_df = pl.read_parquet(args.train_path)
    sequences = build_user_sequences(train_df, id_maps)

    export_embeddings(model, id_maps, sequences, args.output_dir, max_seq_len=config["max_seq_len"])
    print(f"\nre-exported with the fixed forward pass. Now run:\n"
          f"  python scripts/check_embeddings_for_nan.py {args.output_dir / 'sasrec_user_embeddings.parquet'}")


if __name__ == "__main__":
    main()
