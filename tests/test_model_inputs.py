import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_eda import export_dataset, load_data, load_dataset, model_inputs


def write_table(path: Path) -> None:
    """
    分類, 回帰, 時系列の予測に使える表を書く関数

    Parameters
    ----------
    path: Path
        書き出す csv の場所

    Returns
    ----------
    None
    """
    # 日付, 2 つの数の列, カテゴリの列, 3 クラスの列, 値の列
    rng = np.random.default_rng(0)
    rows = 60
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=rows, freq="D")
        .strftime("%Y-%m-%d"),
        "a": rng.normal(size=rows),
        "b": rng.normal(size=rows),
        "city": ["A", "B", "C"] * (rows // 3),
        "label": ["x", "y", "z"] * (rows // 3),
        "sales": rng.normal(100, 10, size=rows),
    }).to_csv(path, index=False)


def test_model_inputs_table(tmp_path: Path) -> None:
    """
    表のデータセットで, 入力の形, タスク, クラス, 出す値の数を返すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 分類: 入力の次元は学習に使う列の数, 出す値の数はクラスの数
    write_table(tmp_path / "t.csv")
    data = load_data(tmp_path / "t.csv", kind="table")
    recipe = {
        "target": "label", "task": "classification",
        "steps": [
            {"op": "drop_columns", "columns": ["date", "sales"]},
            {"op": "one_hot", "columns": ["city"]},
        ],
        "split": {"method": "stratified", "valid": 0.2},
    }
    export_dataset(data, recipe, tmp_path / "cls")
    result = model_inputs(tmp_path / "cls")
    json.dumps(result)
    features = load_dataset(tmp_path / "cls").features
    assert result == {
        "inputs": [
            {"name": "x", "shape": [None, len(features)], "dtype": "float32"}
        ],
        "task": "classification", "target": "label",
        "classes": ["x", "y", "z"], "outputs": 3,
    }
    x, _ = load_dataset(tmp_path / "cls").arrays("train")
    assert x.dtype == np.float32 and x.shape[1] == len(features)

    # 回帰: 出す値は 1 つ, クラスは無い
    recipe = {
        "target": "sales", "task": "regression",
        "steps": [{"op": "drop_columns", "columns": ["date", "city",
                                                    "label"]}],
        "split": {"method": "random", "valid": 0.2},
    }
    export_dataset(data, recipe, tmp_path / "reg")
    result = model_inputs(str(tmp_path / "reg"))
    assert result["inputs"][0]["shape"] == [None, 2]
    assert result["task"] == "regression" and result["classes"] is None
    assert result["outputs"] == 1

    # タスクを決めていない場合は task も outputs も None
    recipe["task"] = "none"
    export_dataset(data, recipe, tmp_path / "none")
    result = model_inputs(tmp_path / "none")
    assert result["task"] is None and result["outputs"] is None


def test_model_inputs_forecasting(tmp_path: Path) -> None:
    """
    時系列の予測のデータセットで, 出す値の数が 1 になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 過去の値の列を足して時刻の順に分割
    write_table(tmp_path / "t.csv")
    data = load_data(tmp_path / "t.csv")
    assert data.kind == "timeseries"
    recipe = {
        "target": "sales", "task": "forecasting",
        "steps": [
            {"op": "drop_columns", "columns": ["city", "label"]},
            {"op": "lag", "columns": ["sales"], "lags": [1, 7]},
            {"op": "drop_missing", "columns": "auto"},
        ],
        "split": {"method": "time", "valid": 0.2},
    }
    export_dataset(data, recipe, tmp_path / "ds")
    result = model_inputs(tmp_path / "ds")
    features = load_dataset(tmp_path / "ds").features
    assert result["inputs"][0]["shape"] == [None, len(features)]
    assert result["task"] == "forecasting" and result["outputs"] == 1


def test_model_inputs_items(tmp_path: Path) -> None:
    """
    文と画像のデータセットで, 中身の配列の形と型を返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 文: 語の番号の並び (int64)
    for label, words in (("pos", "good great"), ("neg", "bad awful")):
        (tmp_path / "reviews" / label).mkdir(parents=True)
        for index in range(5):
            (tmp_path / "reviews" / label / f"{index}.txt").write_text(
                f"this is {words} {index}", encoding="utf-8"
            )
    data = load_data(tmp_path / "reviews")
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [{"op": "vocabulary", "length": 6}],
        "split": {"method": "stratified", "valid": 0.2},
    }
    export_dataset(data, recipe, tmp_path / "text")
    result = model_inputs(tmp_path / "text")
    assert result["inputs"] == [
        {"name": "x", "shape": [None, 6], "dtype": "int64"}
    ]
    assert result["classes"] == ["neg", "pos"] and result["outputs"] == 2

    # 画像: (チャンネル, 高さ, 幅) の float32
    image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    for label in ("cat", "dog"):
        (tmp_path / "pets" / label).mkdir(parents=True)
        for index in range(4):
            pixels = rng.integers(0, 255, (8, 8, 3)).astype(np.uint8)
            image.fromarray(pixels).save(
                tmp_path / "pets" / label / f"{index}.png"
            )
    data = load_data(tmp_path / "pets")
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [
            {"op": "resize", "width": 4, "height": 4}, {"op": "normalize"}
        ],
        "split": {"method": "stratified", "valid": 0.25},
    }
    export_dataset(data, recipe, tmp_path / "images")
    result = model_inputs(tmp_path / "images")
    assert result["inputs"] == [
        {"name": "x", "shape": [None, 3, 4, 4], "dtype": "float32"}
    ]


def test_model_inputs_versions(tmp_path: Path) -> None:
    """
    版 2 と知らない版のデータセットと, 説明の無い場所で止まることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 2 (ファイルの組) はまだ返さない
    (tmp_path / "v2").mkdir()
    (tmp_path / "v2" / "dataset.yaml").write_text(
        yaml.safe_dump({"format_version": 2, "entries": {}}),
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="まだ返せません"):
        model_inputs(tmp_path / "v2")

    # 新しい版は読めない
    (tmp_path / "v9").mkdir()
    (tmp_path / "v9" / "dataset.yaml").write_text(
        yaml.safe_dump({"format_version": 9}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="読めません"):
        model_inputs(tmp_path / "v9")

    # 説明の無い場所
    with pytest.raises(FileNotFoundError):
        model_inputs(tmp_path / "missing")
