import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_eda import (
    Data, export_dataset, load_data, load_dataset, prepare, preview, profile
)


def table(frame: pd.DataFrame) -> Data:
    """
    表から Data を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        表

    Returns
    ----------
    data: Data
        表のデータ
    """
    data = Data(kind="table", frame=frame)

    return data


def test_comma_numbers_and_dates(tmp_path: Path) -> None:
    """
    カンマ付きの数を数として読み, 日時と取り違えないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # カンマ付きの数の列は数, 表は時系列にならない
    (tmp_path / "price.csv").write_text(
        'item,price\nA,"1,980"\nB,"2,480"\nC,"1,200"\nD,"12,300"\n',
        encoding="utf-8"
    )
    data = load_data(tmp_path / "price.csv")
    assert data.kind == "table"
    assert data.frame["price"].tolist() == [1980, 2480, 1200, 12300]

    # 時差の違う日時が混ざっても時系列として読み, 分析できる
    (tmp_path / "tz.csv").write_text(
        "time,v\n2024-01-01T00:00:00+09:00,1\n2024-01-01T01:00:00+00:00,2\n"
        "2024-01-01T02:00:00+00:00,3\n2024-01-01T03:00:00+00:00,4\n",
        encoding="utf-8"
    )
    data = load_data(tmp_path / "tz.csv")
    assert data.kind == "timeseries"
    json.dumps(profile(data))


def test_encodings(tmp_path: Path) -> None:
    """
    EUC-JP の csv と, 先頭より後に別の文字コードの文字がある csv を読める
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # EUC-JP は文字化けせずに読む
    text = "名前,値\nとうきょう,1\nおおさか,2\n"
    (tmp_path / "euc.csv").write_bytes(text.encode("euc-jp"))
    assert "とうきょう" in load_data(tmp_path / "euc.csv").frame["名前"].tolist()

    # 先頭 64KB は ASCII で, 後ろに cp932 の文字
    body = "a,b\n" + "x,1\n" * 20000 + "あ,2\n"
    (tmp_path / "late.csv").write_bytes(body.encode("cp932"))
    assert load_data(tmp_path / "late.csv").frame.shape == (20001, 2)


def test_auto_columns_skip_roles() -> None:
    """
    columns: auto が目的の列と日時の列を変えないことと, 推論でも同じ列を
    使うことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 回帰の目的の列は対数を取らない
    frame = pd.DataFrame({"x": np.arange(1.0, 41.0), "y": np.arange(40.0)})
    recipe = {
        "target": "y", "task": "regression",
        "steps": [{"op": "log", "columns": "auto"}],
        "split": {"method": "random", "valid": 0.25},
    }
    prepared = prepare(table(frame), recipe)
    assert prepared.frames["train"]["y"].max() > 30
    assert prepared.recipe["steps"][0]["state"]["columns"] == ["x"]


def test_split_counts() -> None:
    """
    層化とグループの分割の件数が割合に近いことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 50 クラス × 2 行で検証用 10 %
    frame = pd.DataFrame({
        "x": np.arange(100.0), "c": np.repeat(np.arange(50), 2)
    })
    recipe = {
        "target": "c", "task": "classification", "steps": [],
        "split": {"method": "stratified", "valid": 0.1},
    }
    frames = prepare(table(frame), recipe).frames
    assert len(frames["valid"]) == 10 and len(frames["train"]) == 90

    # 小さなデータでもテスト用を空にしない
    small = pd.DataFrame({"x": np.arange(6.0), "c": [0, 0, 1, 1, 2, 2]})
    recipe["split"] = {"method": "stratified", "valid": 0.2, "test": 0.2}
    frames = prepare(table(small), recipe).frames
    assert len(frames["valid"]) >= 1 and len(frames["test"]) >= 1

    # グループの分割は行の割合に近づける
    groups = pd.DataFrame({
        "x": np.arange(100.0), "g": np.repeat(np.arange(20), 5)
    })
    recipe = {
        "steps": [], "split": {"method": "group", "column": "g",
                               "valid": 0.2},
    }
    frames = prepare(table(groups), recipe).frames
    assert len(frames["valid"]) == 20
    assert not set(frames["valid"]["x"] // 5) & set(frames["train"]["x"] // 5)


def test_duplicates_across_splits_and_filter() -> None:
    """
    重複の削除が分割をまたいで効くことと, 行の絞り込みの比較を確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 種類の値を 12 回ずつ並べた表
    frame = pd.DataFrame({"x": [1, 2] * 12})
    recipe = {
        "steps": [{"op": "drop_duplicates"}],
        "split": {"method": "random", "valid": 0.5},
    }
    frames = prepare(table(frame), recipe).frames
    assert sum(len(f) for f in frames.values()) == 2

    # 比較は compare で書く
    recipe = {
        "steps": [{"op": "filter_rows", "column": "x", "compare": ">",
                   "value": 1}],
        "split": {"method": "none"},
    }
    assert set(prepare(table(frame), recipe).frames["train"]["x"]) == {2}


def test_classes_and_targets() -> None:
    """
    数のクラスの順, ID らしい目的の列の分析, 目的の列の誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 数のクラスは数の順
    frame = pd.DataFrame({"x": np.arange(30.0), "y": [1, 2, 10] * 10})
    recipe = {"target": "y", "task": "classification", "steps": []}
    assert prepare(table(frame), recipe).classes == ["1", "2", "10"]

    # 全て異なる文字の目的の列も分析できる
    names = pd.DataFrame({"x": np.arange(50.0),
                          "t": [f"c{i}" for i in range(50)]})
    assert profile(table(names), target="t")["target"]["task"] == (
        "classification"
    )

    # 目的の列を落とす手順と, 文字の回帰の目的の列は誤り
    with pytest.raises(ValueError, match="消えました"):
        prepare(table(frame), {
            "target": "y", "task": "classification",
            "steps": [{"op": "drop_columns", "columns": ["y"]}],
        })
    with pytest.raises(ValueError, match="数値ではありません"):
        prepare(table(names), {"target": "t", "task": "regression"})


def test_transform_and_export_safety(tmp_path: Path) -> None:
    """
    推論の前処理が欠損で型の変わったカテゴリでも合うことと, 書き出しが
    データセットでないディレクトリを消さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 整数のカテゴリでワンホットにして書き出す
    from ml_eda import load_dataset

    frame = pd.DataFrame({
        "k": [1, 2, 3] * 10, "y": [0, 1] * 15, "z": np.arange(30.0)
    })
    recipe = {
        "target": "y", "task": "classification",
        "steps": [
            {"op": "one_hot", "columns": ["k"]},
            {"op": "drop_missing"},
        ],
        "split": {"method": "random", "valid": 0.2},
    }
    export_dataset(table(frame), recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")

    # 欠損が入って float になった入力でも同じ列が立ち, 行は減らない
    new = pd.DataFrame({"k": [1.0, np.nan, 3.0], "z": [0.0, 1.0, 2.0]})
    result = dataset.transform(new)
    assert len(result) == 3
    assert result["k=1"].tolist() == [1, 0, 0]
    assert result["k=3"].tolist() == [0, 0, 1]

    # データセットでないディレクトリは置き換えない
    other = tmp_path / "keep"
    other.mkdir()
    (other / "important.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_dataset(table(frame), recipe, other, overwrite=True)
    assert (other / "important.txt").exists()

    # 失敗するレシピで置き換えても, 前のデータセットは残る
    bad = {**recipe, "steps": [{"op": "drop_columns", "columns": ["nope"]}]}
    with pytest.raises(ValueError):
        export_dataset(table(frame), bad, tmp_path / "ds", overwrite=True)
    assert (tmp_path / "ds" / "dataset.yaml").exists()


def test_preview_and_profile_odd_columns() -> None:
    """
    リストの列と, 文字でない名前や重なる名前の列を扱えることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # リストの列の前と後
    frame = pd.DataFrame({"a": [[1, 2], [3]] * 5, "b": np.arange(10.0)})
    view = preview(table(frame), {"steps": [{"op": "log",
                                             "columns": ["b"]}]}, 1)
    assert view["after"]["head"][0][0] == "[1, 2]"

    # 文字でない名前と重なる名前
    odd = pd.DataFrame(np.arange(12.0).reshape(4, 3), columns=[0, 1, 1])
    result = profile(table(odd))
    assert set(result["columns"]) == {"0", "1", "1.1"}


def test_big_regular_timeseries() -> None:
    """
    見本を使う大きな時系列で, 間隔の抜けを誤って数えないことを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 分ごとの 12 万行
    frame = pd.DataFrame({
        "t": pd.date_range("2024-01-01", periods=120_000, freq="min"),
        "v": np.sin(np.arange(120_000) / 60),
    })
    data = Data(kind="timeseries", frame=frame, roles={"time": "t"})
    series = profile(data)["timeseries"]
    assert series["gaps"] == 0 and series["interval_seconds"] == 60


def test_transform_needs_columns(tmp_path: Path) -> None:
    """
    推論の前処理で学習に使う列が足りない場合と, 列を落とす手順に auto を
    書いた場合を誤りにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 つの列で書き出す
    from ml_eda import load_dataset

    frame = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0),
                          "y": [0, 1] * 10})
    recipe = {"target": "y", "task": "classification",
              "split": {"method": "random", "valid": 0.2}}
    export_dataset(table(frame), recipe, tmp_path / "ds")

    # b の無い新しいデータは誤り
    with pytest.raises(ValueError, match="足りません"):
        load_dataset(tmp_path / "ds").transform(frame[["a"]])

    # 列を落とす手順の auto は誤り
    with pytest.raises(ValueError, match="落とす列"):
        prepare(table(frame), {"steps": [{"op": "drop_columns",
                                          "columns": "auto"}]})


def test_group_overlap(tmp_path: Path) -> None:
    """
    グループの列の同じ値が分割をまたぐと気付きにし, グループの分割は
    手順で列を落としても使えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 20 人 × 5 行の無作為な分割 (同じ人が分割をまたぐ)
    frame = pd.DataFrame({
        "x": np.arange(100.0), "customer": np.repeat(np.arange(20), 5),
        "y": [0, 1] * 50,
    })
    recipe = {
        "target": "y", "task": "classification",
        "split": {"method": "random", "valid": 0.3, "column": "customer"},
    }
    prepared = prepare(table(frame), recipe)
    assert [item["code"] for item in prepared.findings] == ["group_overlap"]
    item = prepared.findings[0]
    assert item["severity"] == "warning" and item["columns"] == ["customer"]
    assert "group (column: customer)" in item["message"]
    assert item["message"] in prepared.notes
    json.dumps(prepared.findings)

    # 書き出した dataset.yaml の注意と, 変わらない約束の版
    export_dataset(table(frame), recipe, tmp_path / "ds")
    described = yaml.safe_load(
        (tmp_path / "ds" / "dataset.yaml").read_text(encoding="utf-8")
    )
    assert described["format_version"] == 1
    assert item["message"] in described["notes"]

    # 分割の前に列を落としても, 落とす前の値で分けるグループの分割
    recipe["split"]["method"] = "group"
    recipe["steps"] = [{"op": "drop_columns", "columns": ["customer"]}]
    prepared = prepare(table(frame), recipe)
    assert prepared.findings == [] and "customer" not in prepared.features
    valid = set(prepared.frames["valid"]["x"] // 5)
    assert valid and not valid & set(prepared.frames["train"]["x"] // 5)
    after = preview(table(frame), recipe, 1)["after"]["columns"]
    assert [column["name"] for column in after] == ["x", "y"]

    # 落とした列の欠損で, drop_missing (列の指定なし) が行を落とさないこと
    missing = frame.assign(customer=[None] * 4 + list(frame["customer"][4:]))
    recipe["steps"].append({"op": "drop_missing"})
    frames = prepare(table(missing), recipe).frames
    assert sum(len(part) for part in frames.values()) == 100

    # 時刻の写しの欠損は今までどおり落とす (時刻の列を落とした後も)
    stamped = Data(kind="timeseries", frame=frame.assign(
        t=[pd.NaT] + list(pd.date_range("2024-01-01", periods=99, freq="D"))
    ), roles={"time": "t"})
    timed = {"target": "y", "task": "classification",
             "steps": [{"op": "drop_columns", "columns": ["t", "customer"]},
                       {"op": "drop_missing"}],
             "split": {"method": "time", "valid": 0.2, "seed": 0}}
    frames = prepare(stamped, timed).frames
    assert sum(len(part) for part in frames.values()) == 99

    # 時刻の順の分割では, 同じものが過去と未来にまたがるのは普通 (info)
    frame["t"] = pd.date_range("2024-01-01", periods=100, freq="D")
    frame["customer"] = np.tile(np.arange(20), 5)
    series = Data(kind="timeseries", frame=frame, roles={"time": "t"})
    recipe = {
        "target": "y", "task": "classification",
        "split": {"method": "time", "valid": 0.3, "column": "customer"},
    }
    findings = prepare(series, recipe).findings
    assert [item["severity"] for item in findings] == ["info"]

    # 欠損はグループにせず, 空の列名は書いていないもの, 無い列は誤り
    frame["customer"] = [None] * 99 + [1]
    assert prepare(series, recipe).findings == []
    recipe["split"]["column"] = ""
    assert prepare(series, recipe).findings == []
    recipe["split"]["column"] = "patient"
    with pytest.raises(ValueError, match="split.column"):
        prepare(series, recipe)


def unlabeled_table() -> pd.DataFrame:
    """
    目的の列の欠けた行を持つ表を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    frame: pd.DataFrame
        数値の列 x (欠損あり), カテゴリの列 city, 目的の列 y (後ろの 20 行
        は欠損) の表
    """
    # 正解の付いた 40 行と, 値の大きく違う正解の無い 20 行
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "x": np.concatenate([rng.normal(0, 1, 40), np.full(20, 1000.0)]),
        "city": list("AB") * 30,
        "y": [0, 1] * 20 + [None] * 20,
    })
    frame.loc[[3, 45], "x"] = np.nan
    frame.loc[50, "city"] = None

    return frame


def test_unlabeled_split(tmp_path: Path) -> None:
    """
    split.unlabeled: keep で目的の列の欠けた行を unlabeled の分割に残し,
    正解の付いた分割と当てはめた値は drop と同じことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 当てはめる手順と, 行を絞る手順
    frame = unlabeled_table()
    recipe = {
        "target": "y", "task": "classification",
        "steps": [
            {"op": "drop_missing", "columns": ["city"]},
            {"op": "impute", "columns": ["x"], "strategy": "median"},
            {"op": "standardize", "columns": ["x"]},
            {"op": "one_hot", "columns": ["city"]},
        ],
        "split": {
            "method": "stratified", "valid": 0.25, "test": 0.25, "seed": 0,
        },
    }
    dropped = prepare(table(frame), recipe)
    kept = prepare(
        table(frame),
        {**recipe, "split": {**recipe["split"], "unlabeled": "keep"}}
    )

    # 正解の付いた分割と当てはめた値は drop と同じ
    assert set(dropped.frames) == {"train", "valid", "test"}
    assert set(kept.frames) == {"train", "valid", "test", "unlabeled"}
    for name in ("train", "valid", "test"):
        pd.testing.assert_frame_equal(kept.frames[name], dropped.frames[name])
    assert kept.recipe["steps"] == dropped.recipe["steps"]
    assert kept.features == dropped.features

    # 正解の無い行は学習用の値で前処理 (city の欠けた 1 行は落ちる)
    unlabeled = kept.frames["unlabeled"]
    assert len(unlabeled) == 19 and unlabeled["y"].isna().all()
    assert list(unlabeled.columns) == list(kept.frames["train"].columns)
    state = kept.recipe["steps"][2]["state"]
    mean, scale = state["values"]["x"]
    expected = (1000.0 - mean) / scale
    assert np.allclose(unlabeled["x"].drop(index=5), expected)
    assert not unlabeled[kept.features].isna().any().any()

    # 書き出しは版 1 のまま, unlabeled の分割は目的の値を返さない
    recipe["split"]["unlabeled"] = "keep"
    info = export_dataset(table(frame), recipe, tmp_path / "ds")
    assert info["format_version"] == 1
    assert info["splits"]["unlabeled"]["rows"] == 19
    assert (tmp_path / "ds" / info["splits"]["unlabeled"]["file"]).is_file()
    dataset = load_dataset(tmp_path / "ds")
    x, y = dataset.arrays("unlabeled")
    assert x.shape == (19, len(dataset.features)) and y is None
    _, y = dataset.arrays("train")
    assert y is not None and y.dtype == np.int64

    # 推論と同じ前処理の値
    rows = frame[frame["y"].isna() & frame["city"].notna()]
    applied = dataset.transform(rows.reset_index(drop=True))
    pd.testing.assert_frame_equal(
        applied.reset_index(drop=True),
        dataset.frames["unlabeled"][dataset.features], check_dtype=False
    )


def test_unlabeled_split_cases() -> None:
    """
    split.unlabeled の誤り, 欠けた行の無い場合, 目的の列の過去の値の列を
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 知らない値と, 目的の列の無いレシピの keep は誤り
    frame = unlabeled_table()
    with pytest.raises(ValueError, match="drop か keep"):
        prepare(table(frame), {"split": {"method": "none",
                                         "unlabeled": "yes"}})
    with pytest.raises(ValueError, match="目的の列が要ります"):
        prepare(table(frame), {"split": {"method": "none",
                                         "unlabeled": "keep"}})

    # 欠けた行が無ければ unlabeled の分割は作らない
    recipe = {
        "target": "y", "task": "classification",
        "split": {"method": "none", "unlabeled": "keep"},
    }
    prepared = prepare(table(frame.iloc[:40]), recipe)
    assert set(prepared.frames) == {"train"}

    # 目的の列の過去の値は, 推論と同じく分かる値を読む
    series = pd.DataFrame({
        "t": np.arange(10.0),
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, None, None],
    })
    recipe = {
        "target": "y", "task": "regression",
        "steps": [{"op": "lag", "columns": ["y"], "lags": [1]}],
        "split": {"method": "none", "unlabeled": "keep"},
    }
    prepared = prepare(table(series), recipe)
    train = prepared.frames["train"]
    unlabeled = prepared.frames["unlabeled"]
    assert len(train) == 8 and train["y_lag1"].iloc[1:].tolist() == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
    ]
    assert unlabeled["y_lag1"].iloc[0] == 8.0
    assert np.isnan(unlabeled["y_lag1"].iloc[1])
