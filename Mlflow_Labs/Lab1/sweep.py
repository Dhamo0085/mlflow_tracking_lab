"""
Hyperparameter sweep over ElasticNet alpha x l1_ratio, tracked in MLflow.

Each combination is a nested child run under one parent run. The parent run
records the best combination, a sweep_results.csv artifact, and the best model
is registered as ElasticnetWineModel with the alias "champion".

Usage:
    python sweep.py [--alphas 0.01 0.1 0.5 1.0] [--l1-ratios 0.1 0.5 0.9]
                    [--experiment wine-quality-sweep] [--data auto|local|url]
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

import mlflow
from mlflow import MlflowClient

from linear_regression import (
    REGISTERED_MODEL_NAME,
    DataLoadError,
    load_data,
    train_and_log,
    validate_hyperparams,
)

DEFAULT_ALPHAS = [0.01, 0.1, 0.5, 1.0]
DEFAULT_L1_RATIOS = [0.1, 0.5, 0.9]
CHAMPION_ALIAS = "champion"


def run_sweep(alphas: list[float], l1_ratios: list[float], data: pd.DataFrame,
              register_best: bool = True) -> dict:
    """Run every (alpha, l1_ratio) pair as a child run; return the results and the best one."""
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            validate_hyperparams(alpha, l1_ratio)  # fail before starting any run

    with mlflow.start_run(run_name="elasticnet-sweep") as parent:
        mlflow.log_params({"alphas": str(alphas), "l1_ratios": str(l1_ratios),
                           "n_combinations": len(alphas) * len(l1_ratios)})
        results = [
            train_and_log(alpha, l1_ratio, data, register=False, nested=True,
                          extra_tags={"sweep_parent": parent.info.run_id})
            for alpha in alphas
            for l1_ratio in l1_ratios
        ]
        table = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
        best = table.iloc[0].to_dict()

        mlflow.log_params({"best_alpha": best["alpha"], "best_l1_ratio": best["l1_ratio"]})
        mlflow.log_metrics({"best_rmse": best["rmse"], "best_mae": best["mae"], "best_r2": best["r2"]})
        mlflow.set_tag("best_run_id", best["run_id"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep_results.csv"
            table.drop(columns=["model_uri"]).to_csv(path, index=False)
            mlflow.log_artifact(str(path))

    registered_version = None
    if register_best:
        version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
        MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
        registered_version = version.version

    return {"parent_run_id": parent.info.run_id, "table": table, "best": best,
            "registered_version": registered_version}


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
    parser = argparse.ArgumentParser(description="ElasticNet hyperparameter sweep with MLflow nested runs.")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--l1-ratios", nargs="+", type=float, default=DEFAULT_L1_RATIOS)
    parser.add_argument("--experiment", default="wine-quality-sweep")
    parser.add_argument("--data", choices=["auto", "local", "url"], default="auto")
    parser.add_argument("--no-register", action="store_true", help="do not register the best model")
    args = parser.parse_args(argv)

    try:
        for alpha in args.alphas:
            for l1_ratio in args.l1_ratios:
                validate_hyperparams(alpha, l1_ratio)
        data = load_data(args.data)
    except (ValueError, DataLoadError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    mlflow.set_experiment(args.experiment)
    result = run_sweep(args.alphas, args.l1_ratios, data, register_best=not args.no_register)

    print("\nSweep results (best first):")
    print(result["table"][["alpha", "l1_ratio", "rmse", "mae", "r2"]].to_string(index=False))
    best = result["best"]
    print(f"\nBest: alpha={best['alpha']}, l1_ratio={best['l1_ratio']}, RMSE={best['rmse']:.4f} "
          f"(baseline RMSE {best['baseline_rmse']:.4f})")
    if result["registered_version"]:
        print(f"Registered {REGISTERED_MODEL_NAME} version {result['registered_version']} "
              f"as @{CHAMPION_ALIAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
