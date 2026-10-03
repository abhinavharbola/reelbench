import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import torch

from src.config import DATA_DIR
from src.data.split import RANKER_SPLIT_SUFFIX, ranker_train_view
from src.models.sasrec import build_user_sequences, export_embeddings, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.parquet")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--ranker-split", action="store_true")
    args = parser.parse_args()

    resumed = load_checkpoint(args.checkpoint_path)
    if resumed is None:
        raise FileNotFoundError(f"no checkpoint at {args.checkpoint_path}")
    model, _, epoch, id_maps, config = resumed

    corrupted = [name for name, tensor in model.state_dict().items() if not torch.isfinite(tensor).all()]
    if corrupted:
        raise SystemExit(
            f"checkpoint at epoch {epoch} has non-finite parameters in {corrupted}; "
            "it cannot be recovered by re-exporting and must be retrained"
        )
    print(f"loaded checkpoint from epoch {epoch}, max_seq_len={config['max_seq_len']}, all parameters finite")

    train_df = pl.read_parquet(args.train_path)
    prefix = "sasrec"
    if args.ranker_split:
        train_df = ranker_train_view(train_df)
        prefix = f"{prefix}{RANKER_SPLIT_SUFFIX}"
    sequences = build_user_sequences(train_df, id_maps)

    export_embeddings(
        model, id_maps, sequences, args.output_dir, max_seq_len=config["max_seq_len"], prefix=prefix
    )
    print(f"\nre-exported. Now run:\n"
          f"  python scripts/check_embeddings_for_nan.py {args.output_dir / (prefix + '_user_embeddings.parquet')}")


if __name__ == "__main__":
    main()
