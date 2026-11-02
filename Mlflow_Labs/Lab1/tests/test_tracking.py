"""What linear_regression.py logs to MLflow, and backward compatibility."""
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

import linear_regression as lr

LAB_DIR = Path(lr.__file__).resolve().parent

# Output of the ORIGINAL script (baseline commit, see docs/evidence/linear_regression_before.txt).
ORIGINAL_RESULTS = {
    (0.5, 0.5): (0.7931640229276851, 0.6271946374319587, 0.10862644997792614),
    (0.1, 0.9): (0.735314956888905, 0.566974647785579, 0.23390870203034675),
}


@pytest.fixture
def default_run(wine_data):
    return lr.train_and_log(0.5, 0.5, wine_data)


@pytest.mark.parametrize("params", list(ORIGINAL_RESULTS), ids=["default", "0.1-0.9"])
def test_metrics_identical_to_original_script(wine_data, params):
    result = lr.train_and_log(*params, wine_data, register=False)
    rmse, mae, r2 = ORIGINAL_RESULTS[params]
    assert result["rmse"] == pytest.approx(rmse, abs=1e-12)
    assert result["mae"] == pytest.approx(mae, abs=1e-12)
    assert result["r2"] == pytest.approx(r2, abs=1e-12)


def test_original_param_and_metric_names_are_logged(default_run):
    run = mlflow.get_run(default_run["run_id"])
    assert run.data.params["alpha"] == "0.5"
    assert run.data.params["l1_ratio"] == "0.5"
    assert {"rmse", "r2", "mae"} <= run.data.metrics.keys()
    assert run.data.metrics["rmse"] == pytest.approx(default_run["rmse"])


def test_new_params_metrics_and_tags(default_run):
    run = mlflow.get_run(default_run["run_id"])
    assert run.data.params["n_train"] == "1199"
    assert run.data.params["n_test"] == "400"
    assert run.data.params["random_state"] == "40"
    for key in ("baseline_rmse", "baseline_mae", "baseline_r2", "rmse_improvement_vs_baseline"):
        assert key in run.data.metrics
    assert run.data.metrics["rmse_improvement_vs_baseline"] > 0  # beats predicting the mean
    assert run.data.tags["model_family"] == "ElasticNet"
    assert run.data.tags["dataset"] == "winequality-red"


def test_dataset_input_is_logged(default_run):
    run = mlflow.get_run(default_run["run_id"])
    inputs = run.inputs.dataset_inputs
    assert len(inputs) == 1
    assert inputs[0].dataset.name == "winequality-red"


def test_diagnostic_artifacts(default_run, tmp_path):
    client = MlflowClient()
    names = {f.path for f in client.list_artifacts(default_run["run_id"], "diagnostics")}
    assert names == {"diagnostics/coefficients.json", "diagnostics/pred_vs_actual.png"}

    coef_file = mlflow.artifacts.download_artifacts(
        run_id=default_run["run_id"], artifact_path="diagnostics/coefficients.json", dst_path=str(tmp_path))
    coefficients = json.loads(Path(coef_file).read_text())
    assert len(coefficients["coefficients"]) == 11
    png = mlflow.artifacts.download_artifacts(
        run_id=default_run["run_id"], artifact_path="diagnostics/pred_vs_actual.png", dst_path=str(tmp_path))
    assert Path(png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_logged_model_reloads_and_reproduces_rmse(default_run, wine_data):
    model = mlflow.pyfunc.load_model(default_run["model_uri"])
    _, test_x, _, test_y = lr.split_data(wine_data)
    rmse, _, _ = lr.eval_metrics(test_y, model.predict(test_x))
    assert rmse == pytest.approx(default_run["rmse"])


def test_model_signature_has_the_11_features(default_run):
    model = mlflow.pyfunc.load_model(default_run["model_uri"])
    names = [c.name for c in model.metadata.get_input_schema().inputs]
    assert len(names) == 11 and "alcohol" in names and lr.TARGET not in names


def test_model_is_registered_like_the_original(default_run):
    versions = MlflowClient().search_model_versions(f"name='{lr.REGISTERED_MODEL_NAME}'")
    assert len(versions) == 1
    assert versions[0].run_id == default_run["run_id"]


def test_no_register_flag_skips_registry(wine_data):
    lr.train_and_log(0.5, 0.5, wine_data, register=False)
    assert MlflowClient().search_model_versions(f"name='{lr.REGISTERED_MODEL_NAME}'") == []


def test_invalid_hyperparams_do_not_create_a_run(wine_data):
    with pytest.raises(ValueError):
        lr.train_and_log(-1, 0.5, wine_data)
    assert mlflow.search_runs(search_all_experiments=True).empty


def test_main_with_experiment_name(capsys):
    assert lr.main(["0.2", "0.3", "--experiment", "my-exp", "--no-register"]) == 0
    runs = mlflow.search_runs(experiment_names=["my-exp"])
    assert len(runs) == 1
    assert runs.loc[0, "params.alpha"] == "0.2"
    out = capsys.readouterr().out
    assert "l1_ratio=0.300000" in out and "l2_ratio" not in out  # label typo fixed


@pytest.mark.slow
def test_script_runs_from_another_directory(isolated_mlflow):
    # Original usage: `python linear_regression.py <alpha> <l1_ratio>`, here from a temp dir.
    result = subprocess.run(
        [sys.executable, str(LAB_DIR / "linear_regression.py"), "0.5", "0.5", "--data", "local"],
        cwd=isolated_mlflow, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "RMSE: 0.7931640229276851" in result.stdout


@pytest.mark.slow
def test_original_serving_script_still_runs():
    """serving.py is unchanged from the original lab; check it still works."""
    try:
        import xgboost  # noqa: F401  (needs libomp on macOS)
    except Exception as e:  # pragma: no cover - depends on the machine
        pytest.skip(f"xgboost cannot be imported here: {e}")
    import serving

    serving.main()  # contains its own asserts about logged requirements
