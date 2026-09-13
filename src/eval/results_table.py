"""
Shared read-modify-write logic for results/comparison_table.csv, used by
both run_phase1.py and evaluate_pipeline_models.py so the merge behavior
and row order can't drift apart between the two callers.
"""

from pathlib import Path

import polars as pl

CANONICAL_MODEL_ORDER = ["popularity", "item_item_cf", "als", "two_tower", "sasrec"]


def upsert_results(table_path: Path, new_rows: list[dict], owned_models: list[str]) -> pl.DataFrame:
    new_table = pl.DataFrame(new_rows)
    new_table = new_table.select(["model"] + [c for c in new_table.columns if c != "model"])

    if table_path.exists():
        existing = pl.read_csv(table_path).filter(~pl.col("model").is_in(owned_models))
        combined = pl.concat([existing.select(new_table.columns), new_table]) if existing.height > 0 else new_table
    else:
        combined = new_table

    rows_by_model = {row["model"]: row for row in combined.to_dicts()}
    known = [m for m in CANONICAL_MODEL_ORDER if m in rows_by_model]
    unknown = [m for m in rows_by_model if m not in CANONICAL_MODEL_ORDER]
    combined = pl.DataFrame([rows_by_model[m] for m in known + unknown])

    table_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(table_path)
    return combined
