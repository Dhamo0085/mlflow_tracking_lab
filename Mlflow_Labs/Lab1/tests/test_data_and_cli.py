"""Data loading, the train/test split and command-line validation."""
import hashlib
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import linear_regression as lr

# sha256 of the CSV served at DATA_URL (the file the original lab downloads).
URL_FILE_SHA256 = "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"


# --------------------------------------------------------------------------- data

def test_bundled_csv_is_byte_identical_to_original_download():
    assert hashlib.sha256(lr.LOCAL_DATA.read_bytes()).hexdigest() == URL_FILE_SHA256


def test_dataset_shape_and_target(wine_data):
    assert wine_data.shape == (1599, 12)
    assert lr.TARGET in wine_data.columns
    assert wine_data[lr.TARGET].between(3, 8).all()
    assert not wine_data.isna().any().any()


def test_split_sizes(wine_data):
    train_x, test_x, train_y, test_y = lr.split_data(wine_data)
    assert (len(train_x), len(test_x)) == (1199, 400)
    assert len(train_y) == 1199 and len(test_y) == 400
    assert lr.TARGET not in train_x.columns
    assert list(train_y.columns) == [lr.TARGET]


def test_split_matches_original_global_seed_split(wine_data):
    # The original did np.random.seed(40) then train_test_split(data).
    np.random.seed(40)
    orig_train, orig_test = train_test_split(wine_data)
    train_x, test_x, _, _ = lr.split_data(wine_data)
    assert train_x.index.tolist() == orig_train.index.tolist()
    assert test_x.index.tolist() == orig_test.index.tolist()


def test_split_is_reproducible(wine_data):
    a = lr.split_data(wine_data)[1].index.tolist()
    b = lr.split_data(wine_data)[1].index.tolist()
    assert a == b


def test_invalid_source_rejected():
    with pytest.raises(ValueError, match="source must be"):
        lr.load_data("ftp")


def test_auto_falls_back_to_url_when_local_missing(monkeypatch, tmp_path):
    # Point "URL" at the bundled file so the test needs no network.
    monkeypatch.setattr(lr, "DATA_URL", str(lr.LOCAL_DATA))
    monkeypatch.setattr(lr, "LOCAL_DATA", tmp_path / "missing.csv")
    assert lr.load_data("auto").shape == (1599, 12)


def test_offline_and_no_local_copy_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(lr, "LOCAL_DATA", tmp_path / "missing.csv")
    monkeypatch.setattr(lr, "DATA_URL", "http://127.0.0.1:9/none.csv")
    with pytest.raises(lr.DataLoadError, match="Unable to load"):
        lr.load_data("auto")


def test_csv_without_target_column_is_rejected(monkeypatch, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, sep=";", index=False)
    monkeypatch.setattr(lr, "LOCAL_DATA", bad)
    with pytest.raises(lr.DataLoadError, match="no 'quality' column"):
        lr.load_data("local")


# ------------------------------------------------------------------- validation

@pytest.mark.parametrize("alpha, l1_ratio", [(0, 0), (0.5, 0.5), (10, 1), (0.01, 0.1)])
def test_valid_hyperparams(alpha, l1_ratio):
    lr.validate_hyperparams(alpha, l1_ratio)


@pytest.mark.parametrize(
    "alpha, l1_ratio, message",
    [
        (-1, 0.5, "alpha"),
        (math.nan, 0.5, "alpha"),
        (math.inf, 0.5, "alpha"),
        (0.5, -0.1, "l1_ratio"),
        (0.5, 1.5, "l1_ratio"),
        (0.5, math.nan, "l1_ratio"),
    ],
)
def test_invalid_hyperparams(alpha, l1_ratio, message):
    with pytest.raises(ValueError, match=message):
        lr.validate_hyperparams(alpha, l1_ratio)


def test_parser_keeps_original_positional_arguments():
    args = lr.build_parser().parse_args(["0.1", "0.9"])
    assert (args.alpha, args.l1_ratio) == (0.1, 0.9)
    defaults = lr.build_parser().parse_args([])
    assert (defaults.alpha, defaults.l1_ratio) == (0.5, 0.5)
    assert defaults.data == "auto" and defaults.no_register is False


def test_parser_accepts_negative_number_as_value_then_validation_rejects_it(capsys):
    assert lr.main(["-1", "0.5"]) == 2
    assert "alpha must be a finite number >= 0" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["abc"], ["0.5", "xyz"], ["--data", "ftp"]])
def test_parser_rejects_garbage_with_exit_code_2(argv):
    with pytest.raises(SystemExit) as exc:
        lr.main(argv)
    assert exc.value.code == 2


def test_main_returns_2_when_data_unavailable(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lr, "LOCAL_DATA", tmp_path / "missing.csv")
    monkeypatch.setattr(lr, "DATA_URL", "http://127.0.0.1:9/none.csv")
    assert lr.main([]) == 2
    assert "Unable to load the wine-quality data" in capsys.readouterr().err


def test_eval_metrics_perfect_prediction():
    rmse, mae, r2 = lr.eval_metrics([3, 5, 7], [3, 5, 7])
    assert (rmse, mae, r2) == (0, 0, 1)


def test_baseline_metrics_predict_training_mean(wine_data):
    _, _, train_y, test_y = lr.split_data(wine_data)
    baseline = lr.baseline_metrics(train_y, test_y)
    assert baseline["baseline_r2"] <= 0.0  # a constant can't explain variance
    assert baseline["baseline_rmse"] > 0
