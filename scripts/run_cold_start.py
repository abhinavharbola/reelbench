import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.config import DATA_DIR

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE = 64
DEFAULT_FLUSH_EVERY = 1000


def _parts_dir(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_parts")


def load_existing_cache(output_path: Path) -> set[int]:
    done: set[int] = set()
    if output_path.exists():
        done |= set(pl.read_parquet(output_path, columns=["movieId"])["movieId"].to_list())
    parts_dir = _parts_dir(output_path)
    if parts_dir.exists():
        for part in sorted(parts_dir.glob("part-*.parquet")):
            done |= set(pl.read_parquet(part, columns=["movieId"])["movieId"].to_list())
    return done


def _write_part(output_path: Path, rows: list[dict]) -> None:
    parts_dir = _parts_dir(output_path)
    parts_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(parts_dir.glob("part-*.parquet")))
    pl.DataFrame(rows).write_parquet(parts_dir / f"part-{index:06d}.parquet")


def _merge_parts(output_path: Path) -> None:
    parts_dir = _parts_dir(output_path)
    part_files = sorted(parts_dir.glob("part-*.parquet")) if parts_dir.exists() else []
    if not part_files:
        return
    frames = [pl.read_parquet(f) for f in part_files]
    if output_path.exists():
        frames.insert(0, pl.read_parquet(output_path))
    combined = pl.concat(frames).unique(subset=["movieId"], keep="first", maintain_order=True)
    temp_path = output_path.with_suffix(".tmp")
    combined.write_parquet(temp_path)
    temp_path.replace(output_path)
    for part in part_files:
        part.unlink()
    parts_dir.rmdir()


def run_cold_start_job(
    movies: pl.DataFrame,
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> None:
    if SentenceTransformer is None:
        raise ImportError("pip install sentence-transformers")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _merge_parts(output_path)
    already_done = load_existing_cache(output_path)
    remaining = movies.filter(~pl.col("movieId").is_in(already_done))
    print(f"{len(already_done)} already cached, {remaining.height} remaining")
    if remaining.height == 0:
        print(f"nothing to do, {output_path} already complete")
        return

    print(f"loading {model_name} (first run downloads the weights, ~1.2GB, needs internet once)")
    model = SentenceTransformer(model_name)
    print(f"model loaded, running on device: {model.device}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    movie_ids = remaining["movieId"].to_list()
    texts = [f"{t} | genres: {g}" for t, g in zip(remaining["title"].to_list(), remaining["genres"].to_list())]
    total = len(texts)

    pending_rows = []
    done = 0
    for start in range(0, total, batch_size):
        batch_ids = movie_ids[start:start + batch_size]
        batch_texts = texts[start:start + batch_size]

        t0 = time.time()
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        elapsed = time.time() - t0

        if start == 0:
            rate = len(batch_texts) / elapsed if elapsed > 0 else float("inf")
            eta_minutes = (total - len(batch_texts)) / rate / 60 if rate > 0 else 0
            print(f"first batch: {len(batch_texts)} items in {elapsed:.1f}s ({rate:.1f} items/s), "
                  f"estimated ~{eta_minutes:.1f} min for the remaining {total - len(batch_texts)}")

        for movie_id, embedding in zip(batch_ids, batch_embeddings):
            pending_rows.append({"movieId": movie_id, "embedding": embedding.tolist()})
        done += len(batch_texts)

        if len(pending_rows) >= flush_every:
            _write_part(output_path, pending_rows)
            pending_rows = []
            print(f"{done}/{total} done")

    if pending_rows:
        _write_part(output_path, pending_rows)
    _merge_parts(output_path)

    print(f"cold-start embeddings written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies-path", type=Path, default=DATA_DIR / "movies.parquet")
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "cold_start_embeddings.parquet")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--flush-every", type=int, default=DEFAULT_FLUSH_EVERY)
    args = parser.parse_args()

    movies_df = pl.read_parquet(args.movies_path)
    run_cold_start_job(
        movies_df,
        args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        flush_every=args.flush_every,
    )
