"""
SASRec training entrypoint, meant to run on Colab/Kaggle GPU. Same
resume-on-restart pattern as train_two_tower.py — point --checkpoint-path
at persistent storage (Google Drive on Colab) so a disconnected runtime
doesn't lose progress.

Usage (Colab):
    !python scripts/train_sasrec.py \\
        --train-path data/processed/train.parquet \\
        --checkpoint-path /content/drive/MyDrive/recsys-checkpoints/sasrec.pt \\
        --output-dir data/processed \\
        --epochs 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.models.sasrec import build_user_sequences, export_embeddings, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()

    train_df = pl.read_parquet(args.train_path)

    model, id_maps, config = train(
        train_df,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        embedding_dim=args.embedding_dim,
    )

    # use config["max_seq_len"], not args.max_seq_len: if this run resumed
    # from a checkpoint trained with a different --max-seq-len, train()
    # silently uses the checkpoint's value internally, and exporting with
    # the CLI arg instead would crash on a position_embedding shape
    # mismatch (see src/models/sasrec.py's train() docstring/comment).
    sequences = build_user_sequences(train_df, id_maps)
    export_embeddings(model, id_maps, sequences, args.output_dir, max_seq_len=config["max_seq_len"])


if __name__ == "__main__":
    main()
