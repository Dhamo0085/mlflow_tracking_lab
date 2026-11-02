# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
"""
Train an ElasticNet wine-quality model and track it with MLflow.

Usage (the original positional arguments still work):
    python linear_regression.py [alpha] [l1_ratio] [--data auto|local|url]
                                [--experiment NAME] [--no-register]
"""

import argparse
import logging
import math
import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
# Byte-identical copy of DATA_URL, so the lab also runs offline and gives the same numbers.
LOCAL_DATA = Path(__file__).resolve().parent / "data" / "winequality-red.csv"
TARGET = "quality"
RANDOM_SEED = 40
REGISTERED_MODEL_NAME = "ElasticnetWineModel"
DEFAULT_ALPHA = 0.5
DEFAULT_L1_RATIO = 0.5


class DataLoadError(RuntimeError):
    """Raised when the dataset can be read neither locally nor from DATA_URL."""


def load_data(source: str = "auto") -> pd.DataFrame:
    """
    Load the red wine-quality dataset.

    source: "local" (bundled CSV), "url" (download, as in the original lab), or
    "auto" (local copy if present, otherwise download).
    """
    if source not in {"auto", "local", "url"}:
        raise ValueError(f"source must be 'auto', 'local' or 'url', got {source!r}")

    attempts = []
    if source in {"auto", "local"}:
        attempts.append(("local file", LOCAL_DATA))
    if source in {"auto", "url"}:
        attempts.append(("URL", DATA_URL))

    errors = []
    for label, location in attempts:
        try:
            data = pd.read_csv(location, sep=";")
        except Exception as e:  # network errors, missing file, bad CSV
            errors.append(f"{label} {location}: {e}")
            continue
        if TARGET not in data.columns:
            errors.append(f"{label} {location}: no '{TARGET}' column")
            continue
        return data

    # The original logged the error and then crashed with NameError on `data`.
    raise DataLoadError("Unable to load the wine-quality data. Tried " + "; ".join(errors))


def split_data(data: pd.DataFrame, random_state: int = RANDOM_SEED):
    """
    75/25 train/test split (train_test_split's default).

    random_state=40 gives exactly the split the original produced through
    np.random.seed(40), but no longer depends on hidden global state.
    """
    train, test = train_test_split(data, random_state=random_state)
    train_x = train.drop([TARGET], axis=1)
    test_x = test.drop([TARGET], axis=1)
    train_y = train[[TARGET]]
    test_y = test[[TARGET]]
    return train_x, test_x, train_y, test_y


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def validate_hyperparams(alpha: float, l1_ratio: float) -> None:
    """Reject values ElasticNet cannot use, with a readable message."""
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a finite number >= 0 (got {alpha})")
    if not math.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
        raise ValueError(f"l1_ratio must be between 0 and 1 (got {l1_ratio})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and log an ElasticNet wine-quality model.")
    parser.add_argument("alpha", nargs="?", type=float, default=DEFAULT_ALPHA,
                        help=f"regularisation strength >= 0 (default {DEFAULT_ALPHA})")
    parser.add_argument("l1_ratio", nargs="?", type=float, default=DEFAULT_L1_RATIO,
                        help=f"L1 mix between 0 and 1 (default {DEFAULT_L1_RATIO})")
    parser.add_argument("--data", choices=["auto", "local", "url"], default="auto",
                        help="where to read the dataset from (default: auto)")
    parser.add_argument("--experiment", default=None,
                        help="MLflow experiment name (default: MLflow's Default experiment)")
    parser.add_argument("--no-register", action="store_true",
                        help=f"log the model without registering it as {REGISTERED_MODEL_NAME}")
    return parser


def train_and_log(alpha: float, l1_ratio: float, data: pd.DataFrame,
                  register: bool = True, nested: bool = False, extra_tags: dict | None = None) -> dict:
    """
    Fit ElasticNet, log params/metrics/model to MLflow and return a summary.

    Logged metric names (rmse, r2, mae) and param names (alpha, l1_ratio) are
    the original lab's.
    """
    validate_hyperparams(alpha, l1_ratio)
    train_x, test_x, train_y, test_y = split_data(data)

    with mlflow.start_run(nested=nested) as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.set_tags({"model_family": "ElasticNet", "dataset": "winequality-red", **(extra_tags or {})})

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if register and tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            model_info = mlflow.sklearn.log_model(
                lr, name="model", registered_model_name=REGISTERED_MODEL_NAME, signature=signature
            )
        else:
            model_info = mlflow.sklearn.log_model(lr, name="model", signature=signature)

    return {
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code instead of crashing."""
    # Only silence noisy third-party warnings, not everything (the original
    # ignored all warnings, which also hid MLflow deprecation notices).
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

    args = build_parser().parse_args(argv)
    try:
        validate_hyperparams(args.alpha, args.l1_ratio)
        data = load_data(args.data)
    except (ValueError, DataLoadError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.experiment:
        mlflow.set_experiment(args.experiment)
    train_and_log(args.alpha, args.l1_ratio, data, register=not args.no_register)
    return 0


if __name__ == "__main__":
    sys.exit(main())
