"""sweep.py (nested runs + champion alias) and predict.py (registry scoring)."""
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

import linear_regression as lr
import predict
import sweep

SAMPLES = Path(lr.__file__).resolve().parent / "data" / "sample_wines.csv"


@pytest.fixture
def small_sweep(wine_data):
    mlflow.set_experiment("test-sweep")
    return sweep.run_sweep([0.01, 0.5], [0.1, 0.9], wine_data)


# ------------------------------------------------------------------------- sweep

def test_sweep_creates_one_child_run_per_combination(small_sweep):
    runs = mlflow.search_runs(experiment_names=["test-sweep"])
    children = runs[runs["tags.mlflow.parentRunId"] == small_sweep["parent_run_id"]]
    assert len(children) == 4
    assert sorted(zip(children["params.alpha"], children["params.l1_ratio"])) == [
        ("0.01", "0.1"), ("0.01", "0.9"), ("0.5", "0.1"), ("0.5", "0.9")]


def test_sweep_picks_lowest_rmse(small_sweep):
    table = small_sweep["table"]
    assert small_sweep["best"]["rmse"] == table["rmse"].min()
    assert table["rmse"].is_monotonic_increasing
    assert (small_sweep["best"]["alpha"], small_sweep["best"]["l1_ratio"]) == (0.01, 0.1)


def test_sweep_parent_run_records_best(small_sweep):
    parent = mlflow.get_run(small_sweep["parent_run_id"])
    assert parent.data.params["best_alpha"] == "0.01"
    assert parent.data.params["n_combinations"] == "4"
    assert parent.data.metrics["best_rmse"] == pytest.approx(small_sweep["best"]["rmse"])
    assert parent.data.tags["best_run_id"] == small_sweep["best"]["run_id"]
    artifacts = {f.path for f in MlflowClient().list_artifacts(small_sweep["parent_run_id"])}
    assert "sweep_results.csv" in artifacts


def test_sweep_registers_best_model_as_champion(small_sweep):
    version = MlflowClient().get_model_version_by_alias(lr.REGISTERED_MODEL_NAME, sweep.CHAMPION_ALIAS)
    assert str(version.version) == str(small_sweep["registered_version"])
    assert version.run_id == small_sweep["best"]["run_id"]


def test_sweep_rejects_bad_grid_before_any_run(wine_data):
    with pytest.raises(ValueError, match="l1_ratio"):
        sweep.run_sweep([0.1], [0.5, 2.0], wine_data)
    assert mlflow.search_runs(search_all_experiments=True).empty


def test_sweep_cli(capsys):
    assert sweep.main(["--alphas", "0.1", "--l1-ratios", "0.5", "--no-register"]) == 0
    assert "Best: alpha=0.1, l1_ratio=0.5" in capsys.readouterr().out
    assert sweep.main(["--alphas", "-3"]) == 2


# ------------------------------------------------------------------------ predict

def test_read_samples_semicolon_and_comma(tmp_path):
    semi = predict.read_samples(SAMPLES)
    comma_file = tmp_path / "comma.csv"
    semi.to_csv(comma_file, index=False)
    comma = predict.read_samples(comma_file)
    assert semi.shape == comma.shape == (8, 12)


def test_prepare_features_errors():
    columns = ["a", "b"]
    with pytest.raises(predict.InputError, match="missing column"):
        predict.prepare_features(pd.DataFrame({"a": [1]}), columns)
    with pytest.raises(predict.InputError, match="no rows"):
        predict.prepare_features(pd.DataFrame({"a": [], "b": []}), columns)
    with pytest.raises(predict.InputError, match=r"row\(s\) \[2\]"):
        predict.prepare_features(pd.DataFrame({"a": [1, "x"], "b": [2, 3]}), columns)


def test_predict_with_run_uri_matches_model(wine_data):
    run = lr.train_and_log(0.5, 0.5, wine_data, register=False)
    samples = predict.read_samples(SAMPLES)
    result = predict.predict(samples, run["model_uri"])
    assert len(result) == 8
    assert result["predicted_quality"].between(3, 8).all()


def test_predict_cli_with_champion(small_sweep, tmp_path, capsys):
    out_file = tmp_path / "preds.csv"
    assert predict.main([str(SAMPLES), "--output", str(out_file)]) == 0
    written = pd.read_csv(out_file)
    assert "predicted_quality" in written.columns and len(written) == 8
    assert "RMSE vs the 'quality' column" in capsys.readouterr().out


def test_predict_cli_errors(capsys, tmp_path):
    assert predict.main([str(tmp_path / "nope.csv")]) == 2
    assert predict.main([str(SAMPLES), "--model-uri", "models:/DoesNotExist@champion"]) == 3
    err = capsys.readouterr().err
    assert "input file not found" in err and "could not load model" in err
