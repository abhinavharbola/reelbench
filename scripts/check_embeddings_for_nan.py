"""
Quick diagnostic: checks an embedding parquet file (as produced by
export_embeddings() in two_tower.py / sasrec.py) for NaN or all-zero rows,
without running the full retrieval/ranker pipeline.

A NaN embedding causes FAISS to return zero search results for that user
(NaN comparisons are always False, so nothing gets selected as "top-k"),
which downstream shows up as an empty candidate list -- this script finds
those rows directly and fast, so you don't have to wait through a full
ranker-training run to discover them.

Usage:
    python scripts/check_embeddings_for_nan.py data/processed/sasrec_user_embeddings.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_embeddings_for_nan.py <path-to-embeddings.parquet>")
        sys.exit(1)

    path = Path(sys.argv[1])
    df = pl.read_parquet(path)
    id_col = "userId" if "userId" in df.columns else "movieId"

    embeddings = np.array(df["embedding"].to_list())
    nan_mask = np.isnan(embeddings).any(axis=1)
    zero_mask = (embeddings == 0).all(axis=1)

    n_nan = nan_mask.sum()
    n_zero = zero_mask.sum()

    print(f"{path}: {df.height:,} rows, embedding dim {embeddings.shape[1]}")
    print(f"NaN rows: {n_nan:,} ({100 * n_nan / df.height:.2f}%)")
    print(f"all-zero rows: {n_zero:,} ({100 * n_zero / df.height:.2f}%)")

    if n_nan > 0:
        bad_ids = df[id_col].to_numpy()[nan_mask][:10]
        print(f"first {min(10, n_nan)} affected {id_col}s: {bad_ids.tolist()}")

    if n_nan == 0:
        print("clean -- no NaN embeddings found.")


if __name__ == "__main__":
    main()
