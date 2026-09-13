"""
Two-tower training entrypoint, meant to run on Colab/Kaggle GPU.

Free-tier sessions get cut off without warning, so this script always
resumes from --checkpoint-path if it exists rather than assuming one
uninterrupted run. Point --checkpoint-path at a path inside your mounted
Google Drive (Colab) or /kaggle/working (Kaggle, then download it) so the
checkpoint survives a disconnected runtime.

Usage (Colab):
    from google.colab import drive
    drive.mount('/content/drive')

    !python scripts/train_two_tower.py \\
        --train-path data/processed/train.parquet \\
        --checkpoint-path /content/drive/MyDrive/recsys-checkpoints/two_tower.pt \\
        --output-dir data/processed \\
        --epochs 10

Re-running the exact same command after a disconnect resumes automatically
from the last completed epoch saved in the checkpoint.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.models.two_tower import export_embeddings, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()

    train_df = pl.read_parquet(args.train_path)

    model, id_maps = train(
        train_df,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
    )

    export_embeddings(model, id_maps, args.output_dir)


if __name__ == "__main__":
    main()
