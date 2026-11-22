import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_eda import (
    Data, export_dataset, load_data, load_dataset, load_recipe, prepare,
    preview, profile
)
from ml_eda.pipeline import assign_splits


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


def test_profile_near_constant_column() -> None:
    """
    値の幅が浮動小数の誤差ほどしかない列のヒストグラムを求められることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 45 と 45 + 1e-14 だけの列 (np.histogram は 20 個の幅を作れない)
    values = np.array([45.0, 45.0 + 1e-14, 45.0, 45.0 + 1e-14])
    result = profile(table(pd.DataFrame({"gap": values, "n": range(4)})))
    histogram = result["columns"]["gap"]["histogram"]
    assert sum(histogram["counts"]) == 4
    assert histogram["edges"][0] == pytest.approx(44.5)
    assert histogram["edges"][-1] == pytest.approx(45.5)

    # 幅のある列は今までどおり最小から最大まで
    result = profile(table(pd.DataFrame({"x": np.arange(10.0)})))
    edges = result["columns"]["x"]["histogram"]["edges"]
    assert (edges[0], edges[-1]) == (0.0, 9.0)


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


def test_group_time_split(tmp_path: Path) -> None:
    """
    グループと時刻の組の分割が, 同じ人を 1 つの分割にまとめ, 現れた時刻の
    遅い人を検証用とテスト用にすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 20 人が 1 日ずつずれて現れ, 各人が 5 日分の行を持つ (行の順は混ぜる)
    rng = np.random.default_rng(0)
    customer = np.repeat(np.arange(20), 5)
    days = customer + np.tile(np.arange(5), 20)
    frame = pd.DataFrame({
        "t": pd.Timestamp("2024-01-01") + pd.to_timedelta(days, unit="D"),
        "customer": customer, "x": np.arange(100.0), "y": [0, 1] * 50,
    }).iloc[rng.permutation(100)].reset_index(drop=True)
    series = Data(kind="timeseries", frame=frame, roles={"time": "t"})
    recipe = {
        "target": "y", "task": "classification",
        "steps": [{"op": "drop_columns", "columns": ["customer"]}],
        "split": {"method": "group_time", "column": "customer",
                  "valid": 0.2, "test": 0.1},
    }
    prepared = prepare(series, recipe)
    assert prepared.findings == []
    people = {
        name: set(frame.set_index("x").loc[part["x"], "customer"])
        for name, part in prepared.frames.items()
    }
    assert people == {
        "train": set(range(14)), "valid": set(range(14, 18)),
        "test": {18, 19},
    }

    # 境目をまたぐ人は後ろの分割 (6 行の人を挟むとテスト用が増える)
    labels = assign_splits(
        pd.DataFrame({"t": pd.date_range("2024-01-01", periods=10),
                      "g": [0, 0, 1, 1, 1, 1, 1, 1, 2, 3]}),
        {"method": "group_time", "column": "g", "valid": 0.0,
         "test": 0.3},
        {"time": "t"}
    )
    assert labels.tolist() == ["train"] * 2 + ["test"] * 8
    big = pd.DataFrame({
        "t": pd.date_range("2024-01-01", periods=10),
        "g": [0, 0, 1, 1, 1, 1, 1, 1, 2, 3], "y": [0, 1] * 5,
    })
    stamped = Data(kind="timeseries", frame=big, roles={"time": "t"})
    notes = prepare(stamped, {
        "target": "y", "task": "classification",
        "split": {"method": "group_time", "column": "g", "test": 0.3,
                  "valid": 0.0},
    }).notes
    assert any("test の行は 8 行" in note for note in notes)

    # 日時の列とグループの列が要る
    with pytest.raises(ValueError, match="日時の列"):
        prepare(table(frame.drop(columns=["t"])), recipe)
    recipe["split"]["column"] = "h"
    with pytest.raises(ValueError, match="split.column\\) がありません: h"):
        prepare(series, recipe)
    recipe["split"].pop("column")
    with pytest.raises(ValueError, match="を書いてください"):
        prepare(series, recipe)

    # 時刻の順の分割の重なりの気付きは group_time を勧める
    recipe["split"] = {"method": "time", "column": "customer", "valid": 0.3}
    message = prepare(series, recipe).findings[0]["message"]
    assert "group_time (column: customer)" in message

    # 書き出したレシピに分け方が残り, 約束の版は変わらない
    recipe["split"]["method"] = "group_time"
    export_dataset(series, recipe, tmp_path / "ds")
    described = yaml.safe_load(
        (tmp_path / "ds" / "dataset.yaml").read_text(encoding="utf-8")
    )
    written = load_recipe(tmp_path / "ds" / described["recipe"])
    assert described["format_version"] == 1
    assert written["split"]["method"] == "group_time"


def test_stratified_group_split() -> None:
    """
    グループの層化の分割が, 同じ人を 1 つの分割にまとめ, 目的の値の割合を
    分割の間で近くすることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 40 人 × 3 行. 8 人だけが正解 1 (少ないクラス)
    customer = np.repeat(np.arange(40), 3)
    frame = pd.DataFrame({
        "customer": customer, "x": np.arange(120.0),
        "y": (customer < 8).astype(int),
    })
    recipe = {
        "target": "y", "task": "classification",
        "split": {"method": "stratified_group", "column": "customer",
                  "valid": 0.25, "seed": 3},
    }
    prepared = prepare(table(frame), recipe)
    assert prepared.findings == []
    train, valid = prepared.frames["train"], prepared.frames["valid"]
    assert not set(train["x"] // 3) & set(valid["x"] // 3)
    assert len(valid) == 30 and valid["y"].sum() == 6

    # 同じ種と表なら同じ分割
    again = prepare(table(frame), recipe).frames["valid"]
    assert again["x"].tolist() == valid["x"].tolist()

    # 混ざった人は多い値の層 (同じ数なら文字の小さい値)
    small = pd.DataFrame({
        "g": list("aabbccdd"), "y": [1, 0, 1, 1, 0, 0, 1, 1],
    })
    for seed in range(5):
        labels = assign_splits(
            small, {"method": "stratified_group", "column": "g",
                    "valid": 0.5, "seed": seed},
            {"target": "y"}
        )
        chosen = set(small["g"][labels == "valid"])
        assert len(chosen & {"a", "c"}) == 1
        assert len(chosen & {"b", "d"}) == 1

    # 多くのクラス: 層ごとの件数がグループより小さくても, 全体で配る
    many = pd.DataFrame({
        "customer": np.repeat(np.arange(100), 10),
        "y": np.repeat(np.arange(50), 20), "x": np.arange(1000.0),
    })
    split = {"method": "stratified_group", "column": "customer",
             "valid": 0.1, "test": 0.1, "seed": 0}
    prepared = prepare(table(many), {"target": "y", "task": "classification",
                                     "split": split})
    sizes = {name: len(part) for name, part in prepared.frames.items()}
    assert sizes == {"train": 800, "valid": 100, "test": 100}
    assert prepared.notes == []

    # 少ないグループ: 割合があれば 1 つは入れ, 外れた件数は注意にする
    few = pd.DataFrame({"customer": np.repeat([0, 1, 2], 10),
                        "y": [0] * 20 + [1] * 10, "x": np.arange(30.0)})
    prepared = prepare(table(few), {"target": "y", "task": "classification",
                                    "split": {**split, "valid": 0.05,
                                              "test": 0.0}})
    assert len(prepared.frames["valid"]) == 10
    assert any("大きく外れました" in note for note in prepared.notes)
    alone = few.assign(customer=0)
    prepared = prepare(table(alone), {"target": "y", "task": "classification",
                                      "split": split})
    assert set(prepared.frames) == {"train"}
    assert any("0 件になりました" in note for note in prepared.notes)

    # 回帰の値は層を作れない
    with pytest.raises(ValueError, match="regression"):
        load_recipe({"target": "x", "task": "regression", "split": split})
    with pytest.raises(ValueError, match="ほぼ全て違う"):
        prepare(table(many), {"target": "x", "split": split})

    # 層化の分割の重なりの気付きは stratified_group を勧める
    recipe["split"]["method"] = "stratified"
    message = prepare(table(frame), recipe).findings[0]["message"]
    assert "stratified_group (column: customer)" in message

    # 目的の列が要り, 知らない分け方は読んだ時点で誤り
    with pytest.raises(ValueError, match="目的の列"):
        prepare(table(frame), {"split": recipe["split"]})
    with pytest.raises(ValueError, match="知らない分割"):
        load_recipe({"split": {"method": "group_stratified"}})


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

    # 時刻の写しの欠損の行は削除 (時刻の列を落とした後も)
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

    # 版 1 のままの書き出しと, 目的の値の無い unlabeled の分割
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


def test_unlabeled_split_cases(tmp_path: Path) -> None:
    """
    split.unlabeled の誤り, 欠けた行の無い場合, 目的の列の過去の値の列を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

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

    # 読み込みの時点で止まる (保存したレシピのファイルも)
    no_target = {"split": {"method": "none", "unlabeled": "keep"}}
    with pytest.raises(ValueError, match="target"):
        load_recipe(no_target)
    (tmp_path / "recipe.yaml").write_text(
        yaml.safe_dump(no_target), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="target"):
        load_recipe(tmp_path / "recipe.yaml")
    assert load_recipe({**no_target, "target": "y"})["target"] == "y"

    # 欠けた行が無い場合の, unlabeled の分割の無い書き出し
    recipe = {
        "target": "y", "task": "classification",
        "split": {"method": "none", "unlabeled": "keep"},
    }
    prepared = prepare(table(frame.iloc[:40]), recipe)
    assert set(prepared.frames) == {"train"}

    # 推論と同じく分かる値を読む, 目的の列の過去の値
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


def test_unlabeled_target_indicator() -> None:
    """
    目的の列だけを補完して欠損の印を足す手順でも, keep の書き出しと
    欠損の残る列の注意ができることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 目的の列だけの欠損の印 (推論では 0)
    frame = pd.DataFrame({
        "x": np.arange(10.0),
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, None, None, 10.0],
    })
    recipe = {
        "target": "y", "task": "regression",
        "steps": [{"op": "impute", "columns": ["y"], "indicator": True}],
        "split": {"method": "none", "unlabeled": "keep"},
    }
    prepared = prepare(table(frame), recipe)
    assert prepared.frames["unlabeled"]["y_missing"].tolist() == [0, 0]

    # unlabeled の分割にだけ残る欠損の注意
    frame.loc[8, "x"] = np.nan
    recipe["steps"] = []
    prepared = prepare(table(frame), recipe)
    assert any("unlabeled の分割にだけ" in note for note in prepared.notes)


def test_unknown_step_params() -> None:
    """
    手順の知らない引数のあるレシピを, 読んだ時点で止めることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 知っている引数と当てはめた値 (state) は通る
    recipe = {"steps": [
        {"op": "lag", "columns": ["sales"], "lags": [1], "gap": 2},
        {"op": "standardize", "columns": ["x"], "state": {"x": [0.0, 1.0]}},
    ]}
    assert len(load_recipe(recipe)["steps"]) == 2

    # 知らない引数は, 手順と引数の名前を挙げて止まる
    recipe = {"steps": [{"op": "lag", "columns": ["sales"], "lagz": [1]}]}
    with pytest.raises(ValueError, match="lagz"):
        load_recipe(recipe)


def test_empty_step_params() -> None:
    """
    表の手順と中身の手順の空 (None) の引数のあるレシピを, 読んだ時点で
    止めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 空の当てはめた値 (state) は引数でないので通る
    recipe = {"steps": [{"op": "standardize", "columns": ["x"],
                         "state": None}]}
    assert len(load_recipe(recipe)["steps"]) == 1

    # 表の手順の空の引数は, 手順の位置と名前, 引数の名前を挙げて止まる
    # (省いた場合と同じ意味になる columns と by も)
    recipe = {"steps": [
        {"op": "standardize", "columns": ["x"]},
        {"op": "lag", "columns": None, "by": None, "lags": [1]},
    ]}
    with pytest.raises(ValueError, match=r"手順 2 \(lag\) の引数 by, "
                                         r"columns が空"):
        load_recipe(recipe)

    # YAML の空も同じく止まる
    text = "steps:\n  - {op: filter_rows, column: x, compare: isna, value:}\n"
    with pytest.raises(ValueError, match=r"手順 1 \(filter_rows\).*value"):
        load_recipe(yaml.safe_load(text))

    # 中身の手順の空の引数も, 中身の手順の位置と名前, 引数の名前を挙げる
    recipe = {"item_steps": [
        {"op": "resize", "width": 4, "height": 4},
        {"op": "vocabulary", "length": 8, "max_tokens": None},
    ]}
    with pytest.raises(ValueError, match=r"中身の手順 2 \(vocabulary\) "
                                         r"の引数 max_tokens が空"):
        load_recipe(recipe)


def test_unknown_item_step_params() -> None:
    """
    中身の手順の知らない引数のあるレシピを, 読んだ時点で止めることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 知っている引数と当てはめた値 (state) は通る
    recipe = {"item_steps": [
        {"op": "vocabulary", "length": 8, "unit": "char",
         "state": {"vocabulary": ["<pad>", "<unk>", "a"]}},
        {"op": "resize", "width": 4, "height": 4},
    ]}
    assert len(load_recipe(recipe)["item_steps"]) == 2

    # 知らない引数は, 中身の手順の位置と名前, 引数の名前を挙げて止まる
    recipe = {"item_steps": [
        {"op": "resize", "width": 4, "height": 4},
        {"op": "vocabulary", "length": 8, "max_token": 100},
    ]}
    with pytest.raises(ValueError, match=r"中身の手順 2 \(vocabulary\).*"
                                         r"max_token"):
        load_recipe(recipe)

    # prepare と前と後の様子も同じ所で止まり, 並びでない item_steps も誤り
    frame = pd.DataFrame({"text": ["a b", "c d"], "y": [0, 1]})
    data = Data(kind="text", frame=frame, roles={"text": "text"})
    with pytest.raises(ValueError, match="max_token"):
        prepare(data, recipe, read_items=False)
    with pytest.raises(ValueError, match="リスト"):
        load_recipe({"item_steps": {"op": "resize"}})
