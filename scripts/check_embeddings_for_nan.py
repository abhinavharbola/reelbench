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
    nan_mask = (~np.isfinite(embeddings)).any(axis=1)
    zero_mask = (embeddings == 0).all(axis=1)

    n_nan = nan_mask.sum()
    n_zero = zero_mask.sum()

    print(f"{path}: {df.height:,} rows, embedding dim {embeddings.shape[1]}")
    print(f"non-finite rows (NaN or infinity): {n_nan:,} ({100 * n_nan / df.height:.2f}%)")
    print(f"all-zero rows: {n_zero:,} ({100 * n_zero / df.height:.2f}%)")

    if n_nan > 0:
        bad_ids = df[id_col].to_numpy()[nan_mask][:10]
        print(f"first {min(10, n_nan)} affected {id_col}s: {bad_ids.tolist()}")

    if n_nan == 0:
        print("clean: no NaN or infinite embeddings found.")


if __name__ == "__main__":
    main()
