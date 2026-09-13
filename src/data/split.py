"""
Temporal train/test split.

Design note (this fixes an inconsistency in the original spec, where one
section described a global timestamp cutoff and another described a plain
per-user leave-last-N-out split with no shared cutoff):

  A pure per-user leave-last-N-out split, with no global cutoff, does not
  fully prevent leakage: one user's "last N" interactions can be earlier in
  real time than another user's "training" interactions, so models can still
  absorb future signal (item popularity trends, release-driven spikes)
  across the user base even though each individual user's split looks
  temporally correct in isolation.

  Fix used here: a single global timestamp cutoff is the hard train/test
  boundary. No train interaction is ever later than the cutoff, and no test
  interaction is ever earlier than it. Within the post-cutoff pool, each
  user's test set is capped at N interactions (earliest-first) purely to
  keep per-user metric computation comparable across users, never to
  override the global boundary.
"""

from dataclasses import dataclass

import polars as pl


@dataclass
class SplitResult:
    train: pl.DataFrame
    test: pl.DataFrame
    cutoff_timestamp: int


def compute_global_cutoff(interactions: pl.DataFrame, test_quantile: float = 0.9) -> int:
    """
    Pick a single global timestamp cutoff such that roughly
    (1 - test_quantile) of interactions fall after it.
    """
    cutoff = interactions.select(pl.col("timestamp").quantile(test_quantile)).item()
    return int(cutoff)


def temporal_split(
    interactions: pl.DataFrame,
    cutoff_timestamp: int | None = None,
    test_quantile: float = 0.9,
    max_test_per_user: int = 10,
    min_train_interactions: int = 5,
) -> SplitResult:
    """
    Split interactions into train/test using a global cutoff, then cap each
    user's test set at max_test_per_user (earliest-first among their
    post-cutoff interactions).

    Users with fewer than min_train_interactions in train are dropped from
    both train and test — they don't have enough history for a fair
    per-user evaluation, and are handled separately by the cold-start path.
    """
    if cutoff_timestamp is None:
        cutoff_timestamp = compute_global_cutoff(interactions, test_quantile)

    train = interactions.filter(pl.col("timestamp") < cutoff_timestamp)
    test_pool = interactions.filter(pl.col("timestamp") >= cutoff_timestamp)

    train_counts = train.group_by("userId").agg(pl.len().alias("n_train"))
    eligible_users = train_counts.filter(pl.col("n_train") >= min_train_interactions)["userId"].to_list()

    train = train.filter(pl.col("userId").is_in(eligible_users))

    test = (
        test_pool.filter(pl.col("userId").is_in(eligible_users))
        .sort(["userId", "timestamp"])
        .group_by("userId", maintain_order=True)
        .head(max_test_per_user)
    )

    return SplitResult(train=train, test=test, cutoff_timestamp=cutoff_timestamp)


def build_user_seen_items(interactions: pl.DataFrame) -> dict[int, set]:
    """userId -> set of movieIds in `interactions`. Shared helper so every
    caller that needs to exclude already-seen items from a candidate list
    (ranker training, serving, the UI) builds that set the same way."""
    by_user = interactions.group_by("userId").agg(pl.col("movieId")).to_dict(as_series=False)
    return {uid: set(items) for uid, items in zip(by_user["userId"], by_user["movieId"])}


def carve_ranker_supervision_split(
    train: pl.DataFrame,
    val_quantile: float = 0.9,
    min_seen_interactions: int = 1,
) -> SplitResult:
    """
    Carves a *second*, purely-internal split out of the outer train set, used
    only to supervise the LightGBM ranker.

    Bug this fixes: build_serving_artifacts.py / build_ui_artifacts.py used
    to label ranker training candidates with `test.parquet` -- the exact
    same held-out set the harness later scores every model against. That
    means the ranker's own training data would already contain the labels
    the harness is meant to blind-check it on, and once the harness is
    extended to score the full retrieval+ranking pipeline (see
    scripts/evaluate_pipeline_models.py), the reported numbers would be
    inflated by that leak rather than reflecting real generalization.

    Fix: reuse temporal_split on `train` itself (never on test) to get a
    second, earlier cutoff strictly before the outer cutoff:
      - `.train`: each user's earlier in-train interactions -- the "seen"
        set to exclude from FAISS candidates when building the ranker's
        training table, so the ranker isn't trained to just re-rank items
        it's already been told the user watched.
      - `.test`: each user's later in-train interactions -- used as the
        positive labels the ranker learns to rank highly.

    The real held-out test.parquet is never read by anything that touches
    ranker training; it stays reserved for harness evaluation only.
    """
    return temporal_split(
        train,
        test_quantile=val_quantile,
        max_test_per_user=10_000,  # no per-user cap here -- unlike the
        # outer harness split, this isn't scored for cross-user metric
        # comparability, so keep every available label
        min_train_interactions=min_seen_interactions,
    )


def assert_no_leakage(split: SplitResult) -> None:
    """
    Hard invariant check, also exercised in tests/test_split.py:
      1. no train interaction timestamp >= cutoff
      2. no test interaction timestamp < cutoff
      3. every test user also appears in train (no cold users in eval set)
    """
    max_train_ts = split.train.select(pl.col("timestamp").max()).item()
    min_test_ts = split.test.select(pl.col("timestamp").min()).item()

    if max_train_ts is not None and max_train_ts >= split.cutoff_timestamp:
        raise AssertionError("train contains interactions at/after the cutoff")
    if min_test_ts is not None and min_test_ts < split.cutoff_timestamp:
        raise AssertionError("test contains interactions before the cutoff")

    train_users = set(split.train["userId"].unique().to_list())
    test_users = set(split.test["userId"].unique().to_list())
    if not test_users.issubset(train_users):
        raise AssertionError("test contains users absent from train")
