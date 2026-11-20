import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_eda import (
    export_capture_clouds, export_dataset, export_image_masks, load_data,
    load_dataset, model_inputs
)
from ml_eda.synthetic import write_image_masks, write_shape_captures


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
        "classes": ["x", "y", "z"], "target_kind": None, "outputs": 3,
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
    # 版 2 (ファイルの組) の知らない組はまだ返さない
    (tmp_path / "v2").mkdir()
    (tmp_path / "v2" / "dataset.yaml").write_text(
        yaml.safe_dump({"format_version": 2, "entries": {
            "image": {"kind": "image", "role": "input"},
            "boxes": {"kind": "boxes", "role": "target"},
        }}),
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


def test_model_inputs_image_masks(tmp_path: Path) -> None:
    """
    画像と領域のデータセット (版 2 と 3) の入力の形とクラスの数を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pytest.importorskip("PIL.Image")

    # 大きさの混ざる画像. 手順なし (版 2) は高さと幅が決まらない
    made = write_image_masks(
        tmp_path / "raw", count=4, sizes=[(32, 32), (48, 40)]
    )
    export_image_masks(
        made["images"], made["masks"], tmp_path / "v2",
        classes=made["classes"]
    )
    result = model_inputs(tmp_path / "v2")
    assert result["inputs"] == [
        {"name": "image", "shape": [None, 3, None, None], "dtype": "float32"}
    ]
    assert result["task"] is None and result["target"] == "mask"
    assert result["target_kind"] == "mask"

    # 名前の無い値 0 は背景として数える (ML-Quest の骨組みと同じ)
    assert result["classes"] == ["background", "circle", "square"]
    assert result["outputs"] == 3
    assert json.loads(json.dumps(result)) == result

    # 中身の手順 (版 3) の resize と grayscale の後の形
    export_image_masks(
        made["images"], made["masks"], tmp_path / "v3",
        classes=made["classes"], item_steps={"image": [
            {"op": "resize", "width": 40, "height": 32}, "grayscale"
        ]}
    )
    result = model_inputs(tmp_path / "v3")
    assert result["inputs"][0]["shape"] == [None, 1, 32, 40]
    assert result["outputs"] == 3

    # 説明の形の決まらない次元は手順から (名前だけの手順も読む)
    described = tmp_path / "v3" / "dataset.yaml"
    info = yaml.safe_load(described.read_text(encoding="utf-8"))
    info["entries"]["image"]["shape"] = [None, None, None]
    info["entries"]["image"]["steps"] = [
        {"op": "resize", "width": 40, "height": 32}, "grayscale"
    ]
    described.write_text(yaml.safe_dump(info), encoding="utf-8")
    result = model_inputs(tmp_path / "v3")
    assert result["inputs"][0]["shape"] == [None, 1, 32, 40]

    # 名前を渡さない領域は値の一覧の全て (名前は値の文字)
    export_image_masks(made["images"], made["masks"], tmp_path / "plain")
    result = model_inputs(tmp_path / "plain")
    assert result["classes"] == ["0", "1", "2"]
    assert result["outputs"] == 3


def test_model_inputs_capture_clouds(tmp_path: Path) -> None:
    """
    撮影と点群のデータセット (版 2 と 3) の入力の点の数と出す点の数を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pytest.importorskip("PIL.Image")

    # 手順なし (版 2) の点の数のそろった正解はその数
    made = write_shape_captures(
        tmp_path / "raw", views=3, size=(12, 16), points=100
    )
    export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "v2",
        split={"method": "none"}
    )
    assert model_inputs(tmp_path / "v2")["outputs"] == 100

    # 点の数がそろわない正解 (形の最初の次元が null) は既定の数
    described = tmp_path / "v2" / "dataset.yaml"
    info = yaml.safe_load(described.read_text(encoding="utf-8"))
    info["entries"]["cloud"]["shape"] = [None, 3]
    described.write_text(yaml.safe_dump(info), encoding="utf-8")
    result = model_inputs(tmp_path / "v2")
    assert result["inputs"] == [
        {"name": "capture", "shape": [None, 512, 3], "dtype": "float32"}
    ]
    assert result["target"] == "cloud" and result["classes"] is None
    assert result["target_kind"] == "point_cloud"
    assert result["outputs"] == 1024

    # 正解の点の数をそろえた書き出し (版 3) はその数
    export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "v3",
        split={"method": "none"},
        item_steps={"cloud": [{"op": "sample_points", "points": 64}]}
    )
    result = model_inputs(tmp_path / "v3")
    assert result["inputs"][0]["shape"] == [None, 512, 3]
    assert result["outputs"] == 64
