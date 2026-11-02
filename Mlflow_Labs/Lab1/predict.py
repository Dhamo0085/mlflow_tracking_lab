"""
Score wine samples with a model from the MLflow registry (or any model URI).

Usage:
    python predict.py data/sample_wines.csv
    python predict.py my_wines.csv --model-uri runs:/<run_id>/model --output preds.csv

By default it loads models:/ElasticnetWineModel@champion (set by sweep.py).
The input CSV may use ";" or "," as separator; a "quality" column, if present,
is used to report RMSE.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow.pyfunc

from linear_regression import REGISTERED_MODEL_NAME, TARGET, eval_metrics

DEFAULT_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@champion"


class InputError(ValueError):
    """The input CSV cannot be scored."""


def read_samples(path: Path | str) -> pd.DataFrame:
    """Read a CSV with ';' or ',' separators."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except FileNotFoundError:
        raise InputError(f"input file not found: {path}")
    except Exception as e:
        raise InputError(f"could not read {path}: {e}")


def expected_columns(model) -> list[str]:
    """Feature names from the model signature logged at training time."""
    return [col.name for col in model.metadata.get_input_schema().inputs]


def prepare_features(samples: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Check the CSV has every feature column, numeric and without gaps."""
    missing = [c for c in columns if c not in samples.columns]
    if missing:
        raise InputError(f"missing column(s): {', '.join(missing)}")
    if samples.empty:
        raise InputError("the input file has no rows")
    features = samples[columns].apply(pd.to_numeric, errors="coerce")
    bad_rows = features.index[features.isna().any(axis=1)].tolist()
    if bad_rows:
        raise InputError(f"non-numeric or empty values in row(s) {[r + 1 for r in bad_rows][:10]}")
    return features.astype(float)


def predict(samples: pd.DataFrame, model_uri: str = DEFAULT_MODEL_URI) -> pd.DataFrame:
    """Return the input rows with a `predicted_quality` column added."""
    model = mlflow.pyfunc.load_model(model_uri)
    features = prepare_features(samples, expected_columns(model))
    result = samples.copy()
    result["predicted_quality"] = np.ravel(model.predict(features))
    return result


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
    parser = argparse.ArgumentParser(description="Predict wine quality with an MLflow model.")
    parser.add_argument("input_csv")
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--output", help="write predictions to this CSV instead of printing them")
    args = parser.parse_args(argv)

    try:
        samples = read_samples(args.input_csv)
        result = predict(samples, args.model_uri)
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except mlflow.exceptions.MlflowException as e:
        print(f"error: could not load model {args.model_uri!r}: {e.message}\n"
              "hint: run `python sweep.py` (registers @champion) or pass --model-uri runs:/<run_id>/model",
              file=sys.stderr)
        return 3

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote {len(result)} predictions to {args.output}")
    else:
        print(result[[c for c in (TARGET, "predicted_quality") if c in result.columns]].to_string())
    if TARGET in result.columns:
        rmse, mae, r2 = eval_metrics(result[TARGET], result["predicted_quality"])
        print(f"RMSE vs the '{TARGET}' column: {rmse:.4f} (MAE {mae:.4f}, R2 {r2:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
