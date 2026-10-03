import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.config import DATA_DIR
from src.data.split import RANKER_SPLIT_SUFFIX, ranker_train_view
from src.models.two_tower import export_embeddings, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.parquet")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ranker-split", action="store_true")
    args = parser.parse_args()

    train_df = pl.read_parquet(args.train_path)
    prefix = "two_tower"
    if args.ranker_split:
        train_df = ranker_train_view(train_df)
        prefix = f"{prefix}{RANKER_SPLIT_SUFFIX}"

    model, id_maps = train(
        train_df,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        run_name=prefix,
    )

    export_embeddings(model, id_maps, args.output_dir, prefix=prefix)


if __name__ == "__main__":
    main()
