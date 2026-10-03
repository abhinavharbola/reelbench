import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.config import DATA_DIR
from src.data.split import RANKER_SPLIT_SUFFIX, ranker_train_view
from src.models.sasrec import build_user_sequences, export_embeddings, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.parquet")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ranker-split", action="store_true")
    args = parser.parse_args()

    train_df = pl.read_parquet(args.train_path)
    prefix = "sasrec"
    if args.ranker_split:
        train_df = ranker_train_view(train_df)
        prefix = f"{prefix}{RANKER_SPLIT_SUFFIX}"

    model, id_maps, config = train(
        train_df,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        embedding_dim=args.embedding_dim,
        run_name=prefix,
    )

    sequences = build_user_sequences(train_df, id_maps)
    export_embeddings(
        model, id_maps, sequences, args.output_dir, max_seq_len=config["max_seq_len"], prefix=prefix
    )


if __name__ == "__main__":
    main()
