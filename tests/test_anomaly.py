from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    export_dataset, load_data, load_dataset, load_recipe, model_inputs,
    prepare, profile
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
    # 乱数を固定した表の列
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


def anomaly_recipe(
    unlabeled: str, target: bool = True, drop_id: bool = False
) -> dict:
    """
    異常の検知のレシピを作る関数

    Parameters
    ----------
    unlabeled: str
        正解の欠けた行の扱い
    target: bool = True
        正解の列を書くか
    drop_id: bool = False
        行の id の列を手順で落とすか

    Returns
    ----------
    recipe: dict
        符号化と標準化をして無作為に分けるレシピ
    """
    # auto の符号化と標準化 (行の id の列は役割で選ばれない)
    steps = [
        {"op": "one_hot", "columns": "auto"},
        {"op": "standardize", "columns": "auto"},
    ]
    if drop_id:
        steps.insert(0, {"op": "drop_columns", "columns": ["reading_id"]})
    recipe = {
        "task": "anomaly",
        "id": "reading_id",
        "steps": steps,
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
    # 読み込みを通さない Data
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
    # 行の番号を書かない CSV
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

    # 学習用は正常の行だけ, 異常の行の検証用とテスト用
    train = dataset.frames["train"]
    assert set(train["is_anomaly"]) == {0}
    valid = dataset.frames["valid"]
    test = dataset.frames["test"]
    assert int(valid["is_anomaly"].sum()) == 5
    assert int(test["is_anomaly"].sum()) == 5
    assert sum(len(item) for item in dataset.frames.values()) == 200

    # 元の値の行の id (手順で落とさなくても, 役割で学習に使う列から除き,
    # auto の符号化と標準化も選ばない)
    ids = pd.concat([train, valid, test])["reading_id"]
    assert sorted(ids) == sorted(frame["reading_id"])
    assert dataset.features == ["temperature", "vibration"]
    assert not any(
        name.startswith("reading_id") for name in dataset.features
    )
    assert dataset.recipe["steps"][0]["state"]["columns"] == []
    assert dataset.recipe["steps"][1]["state"]["columns"] == [
        "temperature", "vibration"
    ]
    anomalies = set(frame.loc[frame["is_anomaly"] == 1, "reading_id"])
    flagged = pd.concat([valid, test])
    assert set(flagged.loc[flagged["is_anomaly"] == 1, "reading_id"]) == (
        anomalies
    )

    # 学習用 (正常の行) だけで当てはめる標準化
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
    なり, 手順で落とした行の id が unlabeled にも元の値で残ることを
    確かめる関数

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
    recipe = anomaly_recipe("keep", drop_id=True)
    info = export_dataset(data, recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")

    # 正解の付いた 30 行の分割と, 欠けた 170 行の unlabeled
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
    # 正解の列と id の列を書かない表
    frame = make_readings(None).drop(columns=["reading_id"])
    data = load_data(write_table(frame, tmp_path))
    recipe = anomaly_recipe("drop", target=False)
    del recipe["id"]
    info = export_dataset(data, recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")

    # 全ての行の分割と, 行の位置の row_id
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
    # 全ての正解が正常 (検証用とテスト用の両方の注意)
    frame = make_readings([0] * 50)
    prepared = prepare(load_data_frame(frame), anomaly_recipe("drop"))
    assert any(note.startswith("valid に異常") for note in prepared.notes)
    assert any(note.startswith("test に異常") for note in prepared.notes)

    # 異常の行が 1 件だけ (検証用に入り, テスト用だけの注意)
    frame = make_readings([1] + [0] * 49)
    prepared = prepare(load_data_frame(frame), anomaly_recipe("drop"))
    assert int(prepared.frames["valid"]["is_anomaly"].sum()) == 1
    assert prepared.anomaly["rows"]["test"]["anomaly"] == 0
    assert not any(note.startswith("valid に異常") for note in prepared.notes)
    assert any(note.startswith("test に異常") for note in prepared.notes)


@pytest.mark.parametrize(
    "labels",
    [[True, False] * 10, [1.0, 0.0] * 10, [True, None] * 10]
)
def test_label_types(labels: list) -> None:
    """
    真偽値と 1.0 と 0.0 の正解が 0 と 1 の整数になることを確かめる関数

    Parameters
    ----------
    labels: list
        正解の並び

    Returns
    ----------
    None
    """
    # 正解の欠けた行を正常と見なす前処理
    frame = make_readings(labels)
    prepared = prepare(load_data_frame(frame), anomaly_recipe("normal"))
    values = pd.concat(prepared.frames.values())["is_anomaly"]
    assert values.dtype == np.int64
    assert set(values) == {0, 1}
    assert int(values.sum()) == 10


def test_kind_error() -> None:
    """
    表と時系列でない種類で異常の検知が誤りになることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 画像の種類の Data
    frame = make_readings([0, 1, 0])
    data = Data(kind="image", frame=frame, source={})
    with pytest.raises(ValueError, match="表と時系列だけ"):
        prepare(data, anomaly_recipe("drop"))


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
            make_readings([0, 1, 0]).assign(reading_id=["a", None, "b"]),
            "reading_id", "欠け"
        ),
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
    # 前処理での誤り
    recipe = anomaly_recipe("drop")
    if recipe_id is None:
        del recipe["id"]
    else:
        recipe["id"] = recipe_id
    with pytest.raises(ValueError, match=message):
        prepare(load_data_frame(frame), recipe)


def codes(result: dict) -> set:
    """
    分析の結果の気付きの名前を集める関数

    Parameters
    ----------
    result: dict
        分析の結果

    Returns
    ----------
    names: set
        気付きの code の集まり
    """
    # 気付きの code
    names = {item["code"] for item in result["findings"]}

    return names


def test_profile_few_labels() -> None:
    """
    正解が少しある表の分析で, 数と割合が返ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 異常 10 行, 正常 20 行, 欠け 170 行
    frame = make_readings(few_labels())
    result = profile(load_data_frame(frame), "is_anomaly", task="anomaly")
    anomaly = result["anomaly"]
    assert anomaly["column"] == "is_anomaly"
    assert anomaly["labeled"] is True
    assert anomaly["rows"] == 200
    assert anomaly["labeled_rows"] == 30
    assert anomaly["missing"] == 170
    assert anomaly["presumed"] == 0
    assert anomaly["normal"] == 20
    assert anomaly["anomaly"] == 10
    assert anomaly["ratio"] == pytest.approx(10 / 30)
    assert anomaly["presumed_ratio"] == pytest.approx(10 / 200)
    assert anomaly["valid"] is True
    assert anomaly["invalid"] == 0
    assert anomaly["invalid_examples"] == []

    # 偏りと欠けは異常の検知の前提なので出さない
    assert not codes(result) & {
        "few_anomalies", "label_invalid", "class_imbalance",
        "target_missing",
    }

    # タスクを渡さない分析には数を返さない
    assert profile(load_data_frame(frame), "is_anomaly")["anomaly"] is None


@pytest.mark.parametrize("labels", [[0] * 50, [1] + [0] * 49, [False] * 50])
def test_profile_all_normal(labels: list) -> None:
    """
    異常の行が 2 行未満の表の分析で, 気付きが出ることを確かめる関数

    Parameters
    ----------
    labels: list
        正解の並び

    Returns
    ----------
    None
    """
    # 異常の行が少なすぎる
    frame = make_readings(labels)
    result = profile(load_data_frame(frame), "is_anomaly", task="anomaly")
    anomaly = result["anomaly"]
    assert anomaly["anomaly"] == sum(labels)
    assert anomaly["normal"] == 50 - sum(labels)
    assert anomaly["ratio"] == pytest.approx(sum(labels) / 50)
    assert anomaly["valid"] is True
    assert "few_anomalies" in codes(result)
    assert "label_invalid" not in codes(result)


@pytest.mark.parametrize(
    "labels, invalid, examples",
    [
        ([0, 2, 1, 1, 0, 2], 2, ["2"]),
        ([0, 1, "異常", 1, 2, None], 2, ["異常", "2"]),
        (["0", "1", "1"], 3, ["0", "1"]),
    ]
)
def test_profile_invalid_labels(
    labels: list, invalid: int, examples: list
) -> None:
    """
    正解の値に 2 や文字の混ざる表の分析で, 収まらない値と気付きが返る
    ことを確かめる関数

    Parameters
    ----------
    labels: list
        正解の並び
    invalid: int
        収まらない値の行の数
    examples: list
        収まらない値の例

    Returns
    ----------
    None
    """
    # 収まらない値は異常と正常のどちらにも数えない
    frame = make_readings(labels)
    result = profile(load_data_frame(frame), "is_anomaly", task="anomaly")
    anomaly = result["anomaly"]
    assert anomaly["valid"] is False
    assert anomaly["invalid"] == invalid
    assert anomaly["invalid_examples"] == examples
    assert anomaly["labeled_rows"] == anomaly["normal"] + anomaly[
        "anomaly"] + invalid
    assert "label_invalid" in codes(result)

    # 前処理も同じ決まりで誤りにする
    with pytest.raises(ValueError, match="1 か真"):
        prepare(load_data_frame(frame), anomaly_recipe("drop"))


def test_profile_without_labels() -> None:
    """
    正解の列の無い表の分析で, 全ての行を正常と見なすことが返ることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正解の列を渡さない
    frame = make_readings(None)
    result = profile(load_data_frame(frame), task="anomaly")
    anomaly = result["anomaly"]
    assert anomaly["column"] is None
    assert anomaly["labeled"] is False
    assert anomaly["presumed"] == 200
    assert anomaly["anomaly"] == 0
    assert anomaly["ratio"] is None
    assert anomaly["presumed_ratio"] == 0.0
    assert not codes(result) & {"few_anomalies", "label_invalid"}


def test_profile_task_errors() -> None:
    """
    分析で数えられないタスクとデータの種類の誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 知らないタスクと, 表と時系列でないデータ
    data = load_data_frame(make_readings([0, 1, 0]))
    with pytest.raises(ValueError, match="anomaly だけ"):
        profile(data, "is_anomaly", task="uplift")
    data.kind = "text"
    with pytest.raises(ValueError, match="表と時系列だけ"):
        profile(data, "is_anomaly", task="anomaly")
