"""
Curates named personas from the real trained data (not the demo dataset).
Run after scripts/run_phase1.py, since it reads data/processed/train.parquet.

Usage:
    python scripts/curate_personas.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.data.personas import curate_personas

DATA_DIR = Path("data/processed")


def main():
    train_path = DATA_DIR / "train.parquet"
    movies_path = DATA_DIR / "movies.parquet"
    if not train_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"{train_path} or {movies_path} not found. Run scripts/run_phase1.py first."
        )

    train = pl.read_parquet(train_path)
    movies = pl.read_parquet(movies_path)

    personas = curate_personas(train, movies)

    output_path = DATA_DIR / "personas.json"
    output_path.write_text(json.dumps(personas, indent=2))
    print(f"{len(personas)} personas written to {output_path}")
    for p in personas:
        print(f"  {p['name']}: user {p['user_id']}")


if __name__ == "__main__":
    main()
