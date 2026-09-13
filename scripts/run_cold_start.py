"""
Cold-start batch job: embeds movie titles + genres with Gemini's embedding
API, caches to parquet. Run once, offline. Never imported by src/serving/ --
serving reads only the cached parquet output of this script.

Before running: check current free-tier quota at
https://ai.google.dev/gemini-api/docs/rate-limits for whichever
--model-name you run this with (defaults to text-embedding-004) -- RPM/TPM/
RPD figures for Gemini's embedding models have changed more than once
through 2025-2026, so don't trust a hardcoded number here. Set
--requests-per-minute to match what you see on that page (this script
defaults conservatively to 10). Also worth a quick check: `google-
generativeai` (imported below) is Google's older Gemini SDK -- confirm it's
still the recommended package before a large run, in case it's since been
superseded by `google-genai`.

Resumable: writes incrementally and skips movieIds already present in the
output file, so a rate-limit or network interruption partway through a
60k-item run doesn't lose progress.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def load_existing_cache(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    return set(pl.read_parquet(output_path)["movieId"].to_list())


def embed_movie(text: str, model_name: str, max_retries: int = 5) -> list[float]:
    """Exponential backoff on 429s, since documented RPM ceilings are
    unreliable in practice (see module docstring)."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(model=model_name, content=text)
            return result["embedding"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"embed_content failed ({e}), retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def run_cold_start_job(
    movies: pl.DataFrame,
    output_path: Path,
    api_key: str,
    model_name: str = "models/text-embedding-004",
    requests_per_minute: int = 10,
) -> None:
    if genai is None:
        raise ImportError("pip install google-generativeai")

    genai.configure(api_key=api_key)

    already_done = load_existing_cache(output_path)
    remaining = movies.filter(~pl.col("movieId").is_in(already_done))
    print(f"{len(already_done)} already cached, {remaining.height} remaining")

    seconds_per_request = 60.0 / requests_per_minute
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in remaining.iter_rows(named=True):
        text = f"{row['title']} | genres: {row['genres']}"
        embedding = embed_movie(text, model_name)
        rows.append({"movieId": row["movieId"], "embedding": embedding})

        # flush every 100 items so a crash never loses more than that
        if len(rows) >= 100:
            _append_to_cache(output_path, rows)
            rows = []

        time.sleep(seconds_per_request)

    if rows:
        _append_to_cache(output_path, rows)

    print(f"cold-start embeddings written to {output_path}")


def _append_to_cache(output_path: Path, rows: list[dict]) -> None:
    new_df = pl.DataFrame(rows)
    if output_path.exists():
        existing = pl.read_parquet(output_path)
        combined = pl.concat([existing, new_df])
    else:
        combined = new_df
    combined.write_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies-path", type=Path, default=Path("data/processed/movies.parquet"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/cold_start_embeddings.parquet"))
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="models/text-embedding-004")
    parser.add_argument("--requests-per-minute", type=int, default=10,
                         help="check https://ai.google.dev/gemini-api/docs/rate-limits before setting this")
    args = parser.parse_args()

    movies_df = pl.read_parquet(args.movies_path)
    run_cold_start_job(movies_df, args.output_path, args.api_key, args.model_name, args.requests_per_minute)
