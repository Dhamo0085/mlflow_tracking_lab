"""Shared fixtures: every test gets a private MLflow tracking store.

Tests never write to the developer's ./mlflow.db or ./mlruns. The tracking
URI points at a temporary SQLite file and the working directory is a temp
folder, so model artifacts land there too.
"""
import mlflow
import pytest

import linear_regression


@pytest.fixture(scope="session")
def wine_data():
    return linear_regression.load_data("local")


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(uri)
    # MLflow remembers the last experiment set in this process; start each
    # test in the fresh store's Default experiment (id 0).
    mlflow.set_experiment(experiment_id="0")
    yield tmp_path
    while mlflow.active_run():
        mlflow.end_run()
