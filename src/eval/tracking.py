import logging
import os
from contextlib import contextmanager

import mlflow

EXPERIMENT_NAME = "movielens-recsys-benchmark"

logger = logging.getLogger("recsys.tracking")


def _ensure_experiment() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def _safe_end_run() -> None:
    try:
        mlflow.end_run()
    except Exception as exc:
        logger.warning("mlflow end_run failed: %s", exc)


@contextmanager
def log_model_run(run_name: str, params: dict, metrics: dict, extra_config: dict | None = None):
    try:
        _ensure_experiment()
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        if extra_config:
            mlflow.log_params({f"config_{k}": v for k, v in extra_config.items()})
        numeric_metrics = {
            k.replace("@", "_at_"): v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        mlflow.log_metrics(numeric_metrics)
    except Exception as exc:
        logger.warning("experiment tracking unavailable, continuing without it: %s", exc)
        _safe_end_run()
        yield None
        return

    try:
        yield run
    finally:
        _safe_end_run()
