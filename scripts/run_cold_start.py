"""
Cold-start batch job: embeds movie titles + genres locally with
Qwen3-Embedding-0.6B (via sentence-transformers), caches to parquet. Run
once, offline. Never imported by src/serving/ -- serving reads only the
cached parquet output of this script.

No API key, no rate limits, no daily quota, no external network calls
during the run itself (sentence-transformers only needs the internet
once, to download the model weights from Hugging Face on first use --
after that it's fully offline). Runs on CPU fine; if a CUDA or MPS device
is available sentence-transformers picks it up automatically, no config
needed either way.

Qwen3-Embedding-0.6B is a document/query embedding model: queries are
meant to get an instruction prefix, documents are not (see the model
card at https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). Every text
this script embeds is a catalog item going into the content-similarity
index, i.e. always the document side, so no prefix is added here.

Resumable: writes incrementally and skips movieIds already present in the
output file, so an interrupted run (Ctrl-C, closed laptop, anything)
doesn't lose progress. Prints a throughput estimate after the first batch
so you know roughly how long the full run will take before committing to
it.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE = 64
DEFAULT_FLUSH_EVERY = 1000


def load_existing_cache(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    return set(pl.read_parquet(output_path)["movieId"].to_list())


def _append_to_cache(output_path: Path, rows: list[dict]) -> None:
    new_df = pl.DataFrame(rows)
    if output_path.exists():
        existing = pl.read_parquet(output_path)
        combined = pl.concat([existing, new_df])
    else:
        combined = new_df
    combined.write_parquet(output_path)


def run_cold_start_job(
    movies: pl.DataFrame,
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    flush_every: int = DEFAULT_FLUSH_EVERY,
) -> None:
    if SentenceTransformer is None:
        raise ImportError("pip install sentence-transformers")

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
            _append_to_cache(output_path, pending_rows)
            pending_rows = []
            print(f"{done}/{total} done")

    if pending_rows:
        _append_to_cache(output_path, pending_rows)

    print(f"cold-start embeddings written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies-path", type=Path, default=Path("data/processed/movies.parquet"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/cold_start_embeddings.parquet"))
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
