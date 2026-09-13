"""
Experiment tracking: MLflow, optionally backed by a Dagshub-hosted MLflow
server (per project spec section 8). Logs params, metrics, and config once
per run -- only the final checkpoint per model, never intermediate epochs,
to keep logged artifacts small.

Dagshub setup (optional): set these environment variables before running
any training script, then every run logs to your Dagshub repo instead of
the local ./mlruns folder:

    export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
    export MLFLOW_TRACKING_USERNAME="<user>"
    export MLFLOW_TRACKING_PASSWORD="<dagshub-token>"

With none of these set, MLflow falls back to its own local default tracking
store (a ./mlflow.db SQLite file or ./mlruns folder, depending on MLflow
version) -- no external account required to run the pipeline.
"""

import os
from contextlib import contextmanager

import mlflow

EXPERIMENT_NAME = "movielens-recsys-benchmark"


def _ensure_experiment() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def log_model_run(run_name: str, params: dict, metrics: dict, extra_config: dict | None = None):
    """
    Usage:
        with log_model_run("popularity", params={"top_k": 50}, metrics=eval_metrics):
            pass  # metrics/params already captured on entry; body is for
                   # any additional artifact logging the caller wants to do
                   # inside the same run.

    metrics values that aren't int/float (e.g. the 'model' name string) are
    silently skipped -- MLflow's log_metrics requires numeric values.
    """
    _ensure_experiment()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        if extra_config:
            mlflow.log_params({f"config_{k}": v for k, v in extra_config.items()})
        # MLflow metric names only allow alphanumerics, _, -, ., space, :, /
        # -- our metric keys use "@" (recall@10, ndcg@20, ...), so sanitize.
        numeric_metrics = {
            k.replace("@", "_at_"): v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        mlflow.log_metrics(numeric_metrics)
        yield run
