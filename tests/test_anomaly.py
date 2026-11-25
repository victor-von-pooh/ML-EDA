from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    export_dataset, load_data, load_dataset, load_recipe, model_inputs,
    prepare
)
from ml_eda.data import Data
from ml_eda.pipeline import anomaly_counts


def make_readings(labels: Optional[list]) -> pd.DataFrame:
    """
    正常の行が多いセンサの値の表を作る関数

    Parameters
    ----------
    labels: Optional[list]
        行ごとの異常の正解 (None は正解の列を作らない)

    Returns
    ----------
    frame: pd.DataFrame
        行の id, 2 つのセンサの値, (あれば) 正解の列を持つ表
    """
    # 乱数を固定して列を作る
    count = 200 if labels is None else len(labels)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "reading_id": [f"r{index:04d}" for index in range(count)],
        "temperature": rng.normal(50.0, 2.0, count),
        "vibration": rng.normal(1.0, 0.1, count),
    })
    if labels is not None:
        frame["is_anomaly"] = labels

    return frame


def few_labels() -> list:
    """
    異常の正解が少しだけある並びを作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    labels: list
        200 行の正解 (異常 10 行, 正常 20 行, 残りは欠け)
    """
    # 先頭の 10 行が異常, 次の 20 行が正常, 残りは正解の無い行
    labels = [1.0] * 10 + [0.0] * 20 + [np.nan] * 170

    return labels


def anomaly_recipe(unlabeled: str, target: bool = True) -> dict:
    """
    異常の検知のレシピを作る関数

    Parameters
    ----------
    unlabeled: str
        正解の欠けた行の扱い
    target: bool = True
        正解の列を書くか

    Returns
    ----------
    recipe: dict
        標準化して無作為に分けるレシピ
    """
    # 行の id の列は落としても元の値が残る
    recipe = {
        "task": "anomaly",
        "id": "reading_id",
        "steps": [
            {"op": "drop_columns", "columns": ["reading_id"]},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {
            "method": "random", "valid": 0.2, "test": 0.2, "seed": 0,
            "unlabeled": unlabeled,
        },
    }
    if target:
        recipe["target"] = "is_anomaly"

    return recipe


def load_data_frame(frame: pd.DataFrame) -> Data:
    """
    表から Data を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        表

    Returns
    ----------
    data: Data
        種類が表の Data
    """
    # 読み込みを通さずに作る
    data = Data(kind="table", frame=frame, source={})

    return data


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """
    表を CSV に書いて場所を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    path: Path
        書くディレクトリ

    Returns
    ----------
    file: Path
        CSV の場所
    """
    # 行の番号を書かない
    file = path / "readings.csv"
    frame.to_csv(file, index=False)

    return file


def test_few_labels_normal(tmp_path: Path) -> None:
    """
    正解が少しある表で, 学習用が正常の行だけになり, 異常の行が検証用と
    テスト用に入り, 行の id が残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出しと読み直し
    frame = make_readings(few_labels())
    data = load_data(write_table(frame, tmp_path))
    info = export_dataset(data, anomaly_recipe("normal"), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")

    # 学習用は正常の行だけ, 異常の行は検証用とテスト用に分かれる
    train = dataset.frames["train"]
    assert set(train["is_anomaly"]) == {0}
    valid = dataset.frames["valid"]
    test = dataset.frames["test"]
    assert int(valid["is_anomaly"].sum()) == 5
    assert int(test["is_anomaly"].sum()) == 5
    assert sum(len(item) for item in dataset.frames.values()) == 200

    # 行の id は元の値で, 学習に使う列ではない
    ids = pd.concat([train, valid, test])["reading_id"]
    assert sorted(ids) == sorted(frame["reading_id"])
    assert "reading_id" not in dataset.features
    assert dataset.features == ["temperature", "vibration"]
    anomalies = set(frame.loc[frame["is_anomaly"] == 1, "reading_id"])
    flagged = pd.concat([valid, test])
    assert set(flagged.loc[flagged["is_anomaly"] == 1, "reading_id"]) == (
        anomalies
    )

    # 標準化は学習用 (正常の行) だけで当てはめる
    assert abs(train["temperature"].mean()) < 1e-9

    # 説明の行の数と, 正解の型
    summary = info["anomaly"]
    assert summary["id"] == "reading_id"
    assert summary["labeled"] is True
    assert summary["unlabeled"] == "normal"
    rows = summary["rows"]
    assert rows["train"]["anomaly"] == 0
    assert rows["valid"]["anomaly"] + rows["test"]["anomaly"] == 10
    assert sum(item["normal"] for item in rows.values()) == 20
    assert sum(item["presumed"] for item in rows.values()) == 170
    _, y = dataset.arrays("valid")
    assert y.dtype == np.int64
    assert info["classes"] == []
    assert not any("異常の行がありません" in note for note in info["notes"])


def test_few_labels_keep(tmp_path: Path) -> None:
    """
    正解の欠けた行を unlabeled に残す場合に, 学習用が正解の正常の行だけに
    なり, unlabeled にも行の id が残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出しと読み直し
    frame = make_readings(few_labels())
    data = load_data(write_table(frame, tmp_path))
    info = export_dataset(data, anomaly_recipe("keep"), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")

    # 正解の付いた 30 行だけを分け, 欠けた 170 行は unlabeled
    labeled = [dataset.frames[name] for name in ("train", "valid", "test")]
    assert sum(len(item) for item in labeled) == 30
    assert set(dataset.frames["train"]["is_anomaly"]) == {0}
    unlabeled = dataset.frames["unlabeled"]
    assert len(unlabeled) == 170
    assert sorted(unlabeled["reading_id"]) == sorted(
        frame.loc[frame["is_anomaly"].isna(), "reading_id"]
    )
    assert info["anomaly"]["rows"]["unlabeled"]["unlabeled"] == 170
    x, y = dataset.arrays("unlabeled")
    assert x.shape == (170, 2) and y is None


def test_without_labels(tmp_path: Path) -> None:
    """
    正解の無い表で, 全ての行を正常として分け, 行の位置の id が残ることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 正解の列も id の列も書かない
    frame = make_readings(None).drop(columns=["reading_id"])
    data = load_data(write_table(frame, tmp_path))
    recipe = anomaly_recipe("drop", target=False)
    del recipe["id"]
    recipe["steps"] = recipe["steps"][1:]
    info = export_dataset(data, recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")

    # 全ての行が分割に入り, row_id は読み込んだ表の行の位置
    ids = pd.concat(
        [dataset.frames[name] for name in ("train", "valid", "test")]
    )["row_id"]
    assert sorted(ids) == list(range(200))
    assert info["target"] is None
    assert info["anomaly"]["labeled"] is False
    assert info["anomaly"]["id"] == "row_id"
    assert sum(
        item["presumed"] for item in info["anomaly"]["rows"].values()
    ) == 200
    assert dataset.features == ["temperature", "vibration"]
    _, y = dataset.arrays("train")
    assert y is None

    # モデルの出す数は学習に使う列の数
    inputs = model_inputs(tmp_path / "ds")
    assert inputs["task"] == "anomaly"
    assert inputs["outputs"] == 2 and inputs["classes"] is None


def test_no_anomaly_note() -> None:
    """
    正解の列があり, 異常の行が無い場合に注意が出ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 全ての正解が正常
    frame = make_readings([0] * 50)
    data = load_data_frame(frame)
    prepared = prepare(data, anomaly_recipe("drop"))
    assert any("異常の行がありません" in note for note in prepared.notes)


def test_anomaly_counts() -> None:
    """
    異常の行の検証用とテスト用への配り方を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 割合の比と, どちらにも 1 件以上
    assert anomaly_counts(10, 0.2, 0.2) == (5, 5)
    assert anomaly_counts(1, 0.2, 0.2) == (1, 0)
    assert anomaly_counts(2, 0.1, 0.3) == (1, 1)
    assert anomaly_counts(3, 0.2, 0.0) == (3, 0)
    assert anomaly_counts(0, 0.0, 0.0) == (0, 0)
    with pytest.raises(ValueError, match="検証用かテスト用"):
        anomaly_counts(3, 0.0, 0.0)


@pytest.mark.parametrize(
    "change, message",
    [
        (
            {"task": "classification", "split": {"method": "random"}},
            "レシピの id"
        ),
        ({"split": {"method": "stratified"}}, "random だけ"),
        ({"task": "regression", "split": {"unlabeled": "normal"}}, "normal"),
    ]
)
def test_recipe_errors(change: dict, message: str) -> None:
    """
    異常の検知の決まりに合わないレシピが load_recipe で止まることを
    確かめる関数

    Parameters
    ----------
    change: dict
        レシピに上書きする値
    message: str
        誤りの文に含まれる言葉

    Returns
    ----------
    None
    """
    # 上書きしたレシピ
    recipe = {**anomaly_recipe("normal"), **change}
    with pytest.raises(ValueError, match=message):
        load_recipe(recipe)


@pytest.mark.parametrize(
    "frame, recipe_id, message",
    [
        (make_readings([0, 2, 0]), "reading_id", "1 か真"),
        (make_readings(["正常", "異常", "正常"]), "reading_id", "1 か真"),
        (
            make_readings([0, 1, 0]).assign(reading_id=["a", "a", "b"]),
            "reading_id", "同じ値"
        ),
        (make_readings([0, 1, 0]), "missing", "ありません"),
        (
            make_readings([0, 1, 0]).rename(columns={"reading_id": "row_id"}),
            None, "row_id"
        ),
    ]
)
def test_data_errors(
    frame: pd.DataFrame, recipe_id: Optional[str], message: str
) -> None:
    """
    正解の値と行の id の列の誤りを確かめる関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    recipe_id: Optional[str]
        レシピの id (None は書かない)
    message: str
        誤りの文に含まれる言葉

    Returns
    ----------
    None
    """
    # 前処理で止まる
    recipe = anomaly_recipe("drop")
    recipe["steps"] = recipe["steps"][1:]
    if recipe_id is None:
        del recipe["id"]
    else:
        recipe["id"] = recipe_id
    with pytest.raises(ValueError, match=message):
        prepare(load_data_frame(frame), recipe)
