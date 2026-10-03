import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR
from src.retrieval.faiss_index import build_and_save_index


def main():
    cold_start_path = DATA_DIR / "cold_start_embeddings.parquet"
    if not cold_start_path.exists():
        raise FileNotFoundError(f"{cold_start_path} not found. Run scripts/run_cold_start.py first.")
    build_and_save_index(cold_start_path, DATA_DIR / "faiss_index" / "cold_start_items.index", normalize=True)


if __name__ == "__main__":
    main()
