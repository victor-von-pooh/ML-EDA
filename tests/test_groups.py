import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
import yaml

import ml_eda
from ml_eda import (
    Data, describe_groups, export_group_rows, load_items, model_inputs
)
from ml_eda import groups
from ml_eda.groups import MAX_WIDTH
from ml_eda.items import csv_values


def table(frame: pd.DataFrame, source: Optional[dict] = None) -> Data:
    """
    表から読んだデータを作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    source: Optional[dict] = None
        読み込んだ場所と読み方

    Returns
    ----------
    data: Data
        表のデータ
    """
    # 手元で作った表
    data = Data(kind="table", frame=frame, source=dict(source or {}))

    return data


def customers(path: Path) -> Path:
    """
    はっきり違う 3 つの群の顧客の表の CSV を書く関数

    Parameters
    ----------
    path: Path
        書くファイル

    Returns
    ----------
    path: Path
        書いたファイル (cid, spend, visits, region, code の 60 行. spend
        の 1 つと region の 1 つが欠け)
    """
    # 使う額, 来た回数, 地域が群ごとに違う
    rng = np.random.default_rng(0)
    rows = []
    for group, (spend, visits, region) in enumerate(
        [(10, 1, "east"), (100, 5, "west"), (1000, 20, "north")]
    ):
        for index in range(20):
            rows.append({
                "cid": f"c{group}_{index}",
                "spend": spend * (1 + 0.05 * rng.standard_normal()),
                "visits": visits + int(rng.integers(0, 2)),
                "region": region, "code": f"0{group}",
            })
    frame = pd.DataFrame(rows)
    frame.loc[3, "spend"] = np.nan
    frame.loc[45, "region"] = None
    frame.to_csv(path, index=False)

    return path


def test_public_names() -> None:
    """
    群分けの関数が公開の名前にあるかを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ml_eda から読める
    for name in ("describe_groups", "export_group_rows"):
        assert name in ml_eda.__all__


def test_summary_rows_like_the_skeleton(tmp_path: Path) -> None:
    """
    群の要約の行が ML-Quest の骨組みの群の要約と同じ形かを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 列の特徴をそのまま k-means で 3 つに分ける
    path = customers(tmp_path / "rows.csv")
    summary = describe_groups(path, ["spend", "visits", "region"], k=3)
    json.dumps(summary)
    assert summary["method"] == "kmeans"
    assert summary["k"] == 3 and summary["rows"] == 60
    assert summary["columns"] == {
        "numeric": ["spend", "visits"], "categorical": ["region"]
    }
    assert summary["width"] == 2 + 4

    # 群の番号の順の行と, 最後に全ての行 (鍵の順も骨組みと同じ)
    groups = summary["groups"]
    assert [row["group"] for row in groups] == [0, 1, 2, "全体"]
    for row in groups:
        assert list(row) == [
            "group", "rows", "share", "stability", "spend", "visits",
            "region",
        ]
    assert [row["rows"] for row in groups] == [20, 20, 20, 60]
    assert groups[-1]["share"] == 1.0 and groups[-1]["stability"] is None
    assert all(row["stability"] == 1.0 for row in groups[:-1])
    regions = sorted(row["region"] for row in groups[:-1])
    assert regions[0] == "east (100 %)"
    assert any(text.startswith("north") for text in regions)
    assert groups[-1]["region"].endswith("(33 %)")
    means = sorted(row["spend"] for row in groups[:-1])
    assert means[0] < 20 and 50 < means[1] < 200 and means[2] > 500

    # 安定性とまとまり, 欠けの気付き
    assert summary["stability"] == {
        "method": "seed", "repeats": 5, "fraction": None, "ari": 1.0,
        "ari_min": 1.0
    }
    assert summary["silhouette"] > 0.5
    assert any("欠けた値のある行が 2 行" in note
               for note in summary["notes"])


def test_given_labels(tmp_path: Path) -> None:
    """
    渡した群の割り当てで要約し, 安定性を数えないかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表の群の列と, 行ごとの並び
    frame = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 10.0, 11.0], "g": ["b", "b", "b", "a", "a"]
    })
    data = table(frame)
    by_column = describe_groups(data, ["x"], labels="g")
    by_list = describe_groups(data, ["x"], labels=[1, 1, 1, 0, 0])
    assert by_column["method"] == "labels" and by_column["seed"] is None
    assert by_column["stability"] is None
    assert [row["group"] for row in by_column["groups"]] == [
        "a", "b", "全体"
    ]
    assert by_column["groups"][0]["x"] == 10.5
    assert by_column["groups"][0]["stability"] is None
    assert [row["group"] for row in by_list["groups"]] == [0, 1, "全体"]
    assert by_list["groups"][1]["rows"] == 3

    # 割り当ての誤り
    with pytest.raises(ValueError, match="行の数"):
        describe_groups(data, ["x"], labels=[0, 1])
    with pytest.raises(ValueError, match="欠けた値"):
        describe_groups(data, ["x"], labels=[0, 1, None, 0, 1])
    with pytest.raises(ValueError, match="群の列"):
        describe_groups(data, ["x"], labels="nothing")
    with pytest.raises(ValueError, match="k を渡さないで"):
        describe_groups(data, ["x"], labels="g", k=2)


def test_describe_errors() -> None:
    """
    群の要約の引数の誤りで止まるかを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 列, 群の数, 幅の誤り
    data = table(pd.DataFrame({
        "x": [1.0, 2.0, 3.0], "rows": [1, 2, 3], "name": ["a", "b", "c"]
    }))
    with pytest.raises(ValueError, match="列の名前を渡して"):
        describe_groups(data, k=2)
    with pytest.raises(ValueError, match="群の要約か 1 件の鍵"):
        describe_groups(data, ["rows"], k=2)
    with pytest.raises(ValueError, match="列がありません"):
        describe_groups(data, ["y"], k=2)
    with pytest.raises(ValueError, match="2 回"):
        describe_groups(data, ["x", "x"], k=2)
    for k in (None, 1, 2.0, True):
        with pytest.raises(ValueError, match="群の数 k"):
            describe_groups(data, ["x"], k=k)
    with pytest.raises(ValueError, match="行の数 3 より多い"):
        describe_groups(data, ["x"], k=4)
    with pytest.raises(ValueError, match="repeats"):
        describe_groups(data, ["x"], k=2, repeats=0)
    for method in ("bootstrap", "Seed", None):
        with pytest.raises(ValueError, match="数え方 stability"):
            describe_groups(data, ["x"], k=2, stability=method)
    for fraction in (0, 1.5, -0.1, True, "0.5", float("nan")):
        with pytest.raises(ValueError, match="fraction"):
            describe_groups(data, ["x"], k=2, stability="sample",
                            fraction=fraction)

    # seed では割合を確かめない. labels でも数え方と割合を確かめる
    assert describe_groups(
        data, ["x"], k=2, fraction=None
    )["stability"]["fraction"] is None
    with pytest.raises(ValueError, match="数え方 stability"):
        describe_groups(data, ["x"], labels=[0, 1, 1], stability="bogus")
    with pytest.raises(ValueError, match="fraction"):
        describe_groups(data, ["x"], labels=[0, 1, 1], stability="sample",
                        fraction=0)
    given = describe_groups(data, ["x"], labels=[0, 1, 1],
                            stability="sample", fraction=0.5)
    assert given["stability"] is None
    wide = table(pd.DataFrame({"name": [str(i) for i in range(
        MAX_WIDTH + 1
    )]}))
    with pytest.raises(ValueError, match="上限"):
        describe_groups(wide, ["name"], k=2)


def test_empty_groups_note() -> None:
    """
    異なる行が群の数より少ないとき, 行の無い群を気付きに書くかを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同じ行だけ (群は 1 つにしか分かれない)
    data = table(pd.DataFrame({"x": [1.0, 1.0, 1.0]}))
    summary = describe_groups(data, ["x"], k=2)
    assert sorted(row["rows"] for row in summary["groups"][:-1]) == [0, 3]
    empty = [row for row in summary["groups"] if row["rows"] == 0][0]
    assert empty["x"] is None and empty["stability"] is None
    assert any("行の無い群" in note for note in summary["notes"])


@pytest.mark.parametrize("parquet", [True, False])
def test_export_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parquet: bool
) -> None:
    """
    群分けの表を版 2 で書き出し, load_items で同じ値に読めるかを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        parquet が無いときの書き出しにする
    parquet: bool
        parquet で書くか (False は csv)

    Returns
    ----------
    None
    """
    # parquet が無い場合の csv と, ある場合の parquet
    if parquet:
        pytest.importorskip("pyarrow")
    else:
        def no_parquet(*args: object, **kwargs: object) -> None:
            raise ImportError("pyarrow が無い")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", no_parquet)
    path = customers(tmp_path / "rows.csv")
    data = table(pd.read_csv(path, dtype={"code": str}),
                 {"path": str(path), "loader": "csv", "options": {}})
    info = export_group_rows(
        data, tmp_path / "ds", ["spend", "visits", "region", "code"],
        id="cid"
    )

    # 説明 (受け持ちは group で正解は無い. 場所は名前だけ)
    written = yaml.safe_load((tmp_path / "ds" / "dataset.yaml").read_text(
        encoding="utf-8"
    ))
    assert written == info
    assert info["format_version"] == 2 and info["role"] == "group"
    assert info["target"] == {"kind": "none", "entries": []}
    assert info["items"] == ("items.parquet" if parquet else "items.csv")
    assert info["entries"]["spend"] == {
        "kind": "value", "role": "input", "placement": "column",
        "dtype": "float64",
    }
    assert info["entries"]["visits"]["dtype"] == "int64"
    assert info["entries"]["region"]["kind"] == "class"
    assert info["entries"]["code"]["dtype"] == "str"
    assert info["splits"] == {"train": {"items": 48}, "valid": {"items": 12}}
    assert info["source"]["name"] == "rows.csv"
    assert "path" not in info["source"] and info["source"]["id"] == "cid"

    # 読んだ値 (欠けは欠けのまま, 数のような種類は文字のまま)
    dataset = load_items(tmp_path / "ds")
    items = dataset.items.set_index("id")
    assert np.isnan(items.loc["c0_3", "spend"])
    assert pd.isna(items.loc["c2_5", "region"])
    assert items.loc["c0_0", "code"] == "00"
    assert items.loc["c1_0", "visits"] in (5, 6)
    first = dataset.split("valid")[0]
    assert set(first) == {
        "id", "split", "group", "spend", "visits", "region", "code"
    }

    # データセットから求めた要約は, 表から求めたものと同じ群
    from_dataset = describe_groups(dataset, k=3)
    from_table = describe_groups(
        data, ["spend", "visits", "region", "code"], k=3
    )
    assert from_dataset["groups"] == from_table["groups"]
    assert from_dataset["source"]["name"] == "rows.csv"
    narrow = describe_groups(dataset, ["spend"], k=3)
    assert narrow["columns"]["numeric"] == ["spend"]


def test_export_without_id_and_split(tmp_path: Path) -> None:
    """
    行を見分ける列の無い書き出しと, 分割のしかたを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 行の位置が id で, 分けない場合は全て学習用
    data = table(pd.DataFrame({
        "x": pd.array([1, None, 3], dtype="Int64"), "flag": [True, False, True]
    }))
    info = export_group_rows(
        data, tmp_path / "ds", ["x", "flag"], split={"method": "none"}
    )
    assert info["splits"] == {"train": {"items": 3}}
    assert info["entries"]["x"]["dtype"] == "float64"
    assert info["entries"]["flag"]["kind"] == "class"
    items = load_items(tmp_path / "ds").items
    assert list(items["id"]) == ["0", "1", "2"]
    assert list(items["flag"]) == ["True", "False", "True"]

    # 置き換えは overwrite を渡したときだけ
    with pytest.raises(FileExistsError):
        export_group_rows(data, tmp_path / "ds", ["x"])
    export_group_rows(data, tmp_path / "ds", ["x"], overwrite=True)
    assert list(load_items(tmp_path / "ds").entries) == ["x"]


def test_export_errors(tmp_path: Path) -> None:
    """
    書き出しの誤りで止まり, 何も書かないかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 行を見分ける列の欠けと同じ値, 分割の誤り, 群の要約と同じ名前の列
    data = table(pd.DataFrame({
        "key": ["a", "a", None], "x": [1.0, 2.0, 3.0],
        "share": [1, 2, 3], "split": [1, 2, 3],
    }))
    target = tmp_path / "ds"
    with pytest.raises(ValueError, match="欠けた値"):
        export_group_rows(data, target, ["x"], id="key")
    data.frame.loc[2, "key"] = "b"
    with pytest.raises(ValueError, match="同じ値"):
        export_group_rows(data, target, ["x"], id="key")
    with pytest.raises(ValueError, match="群を作る列に入れないで"):
        export_group_rows(data, target, ["x", "key"], id="key")
    with pytest.raises(ValueError, match="random か none"):
        export_group_rows(data, target, ["x"], split={"method": "group"})
    for name in ("share", "split"):
        with pytest.raises(ValueError, match="群の要約か 1 件の鍵"):
            export_group_rows(data, target, [name])
    with pytest.raises(ValueError, match="読み込みの名前と設定"):
        export_group_rows(data, target, ["x"], loader="csv")
    assert not target.exists()


def test_not_group_dataset(tmp_path: Path) -> None:
    """
    モデルの入力と, 受け持ちが group でないデータセットの要約で止まるかを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # モデルの入力は返さない (知らない組の誤り)
    data = table(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    export_group_rows(data, tmp_path / "ds", ["x"])
    with pytest.raises(ValueError, match="入力 value と正解 None"):
        model_inputs(tmp_path / "ds")

    # 書き出した説明の受け持ちを書き換える
    place = tmp_path / "ds" / "dataset.yaml"
    info = yaml.safe_load(place.read_text(encoding="utf-8"))
    info["role"] = "predict"
    place.write_text(yaml.safe_dump(info, allow_unicode=True),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="受け持ちが predict"):
        describe_groups(load_items(tmp_path / "ds"), k=2)


def test_same_values_as_the_skeleton() -> None:
    """
    完全には分かれない小さな表で, ML-Quest の骨組みの群の要約, 安定性,
    シルエット係数と同じ値になるかを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 期待値は骨組みの summary_rows, stability (seed で 5 回), silhouette
    # (種 0) に同じ表を渡して求めた値
    frame = pd.DataFrame({
        "x": [6.4, 2.7, 0.4, 0.2, 8.1, 9.1, 6.1, 7.3, 5.4, 9.4, 8.2, 0.0],
        "y": [8.6, 0.3, 7.3, 1.8, 8.6, 5.4, 3.0, 4.2, 0.3, 1.2, 6.7, 6.5],
        "c": ["a", "b", None, "b", "b", None, None, None, "b", None, None,
              "b"],
    })
    summary = describe_groups(table(frame), ["x", "y", "c"], k=3)
    expected = [
        {"group": 0, "rows": 3, "share": 0.25, "stability": 0.92,
         "x": 2.766666666666667, "y": 0.8, "c": "b (100 %)"},
        {"group": 1, "rows": 4, "share": 1 / 3, "stability": 0.65,
         "x": 3.725, "y": 7.75, "c": "b (50 %)"},
        {"group": 2, "rows": 5, "share": 5 / 12,
         "stability": 0.8523809523809524, "x": 8.02, "y": 4.1,
         "c": "(欠損) (100 %)"},
        {"group": "全体", "rows": 12, "share": 1.0, "stability": None,
         "x": 5.275, "y": 4.491666666666667, "c": "(欠損) (50 %)"},
    ]
    assert summary["groups"] == [
        {key: pytest.approx(value) if isinstance(value, float) else value
         for key, value in row.items()} for row in expected
    ]
    assert summary["stability"]["ari"] == pytest.approx(0.6818593113933166)
    assert summary["stability"]["ari_min"] == pytest.approx(
        0.5271238485158649
    )
    assert summary["silhouette"] == pytest.approx(0.36390241512137544)


def test_sample_stability_like_the_skeleton() -> None:
    """
    標本を変える数え方 (sample) で, ML-Quest の骨組みの安定性と同じ値に
    なるかを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 期待値は骨組みの stability (group.stability: sample, 5 回, 学習用の
    # 行は全ての行, 種 0) に同じ表を渡して求めた値
    frame = pd.DataFrame({
        "x": [6.4, 2.7, 0.4, 0.2, 8.1, 9.1, 6.1, 7.3, 5.4, 9.4, 8.2, 0.0],
        "y": [8.6, 0.3, 7.3, 1.8, 8.6, 5.4, 3.0, 4.2, 0.3, 1.2, 6.7, 6.5],
        "c": ["a", "b", None, "b", "b", None, None, None, "b", None, None,
              "b"],
    })
    expected = {
        0.8: (0.47111697896895705, 0.11310190369540872,
              [0.7666666666666667, 0.58, 0.6464285714285715]),
        0.5: (0.33987565850181384, 0.11310190369540872,
              [0.55, 0.48, 0.6409523809523809]),
    }
    for fraction, (ari, ari_min, jaccard) in expected.items():
        summary = describe_groups(table(frame), ["x", "y", "c"], k=3,
                                  stability="sample", fraction=fraction)
        assert summary["stability"] == {
            "method": "sample", "repeats": 5, "fraction": fraction,
            "ari": pytest.approx(ari), "ari_min": pytest.approx(ari_min),
        }
        assert [row["stability"] for row in summary["groups"]] == [
            *map(pytest.approx, jaccard), None
        ]

    # 群と行の要約, まとまりは数え方で変わらない
    seed = describe_groups(table(frame), ["x", "y", "c"], k=3)
    sample = describe_groups(table(frame), ["x", "y", "c"], k=3,
                             stability="sample")
    assert ([{**row, "stability": None} for row in seed["groups"]]
            == [{**row, "stability": None} for row in sample["groups"]])
    assert seed["silhouette"] == sample["silhouette"]


def test_sample_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    標本を変える数え方で, 引く行の数が群の数より少なくならず, 割合 1 と
    行の数が群の数と同じ場合も数えられるかを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        k-means に渡す行の数を記録するための差し替え

    Returns
    ----------
    None
    """
    # k-means に渡した点の数を記録する
    sizes: list = []
    original = groups.kmeans

    def record(points: np.ndarray, k: int, rng: np.random.Generator,
               *args: Any) -> np.ndarray:
        sizes.append(len(points))
        return original(points, k, rng, *args)

    monkeypatch.setattr(groups, "kmeans", record)
    frame = pd.DataFrame({"x": [float(i % 4) for i in range(12)],
                          "y": [float(i) for i in range(12)]})

    # int(12 × 0.1) = 1 は群の数 3 に上げる (1 回目は元の群で全ての行)
    summary = describe_groups(table(frame), ["x", "y"], k=3,
                              stability="sample", fraction=0.1, repeats=4)
    assert sizes == [12, 3, 3, 3, 3]
    assert summary["stability"]["fraction"] == 0.1
    assert -1 <= summary["stability"]["ari_min"] <= 1

    # 割合 1 は全ての行を引く (並びが変わるだけ)
    sizes.clear()
    whole = describe_groups(table(frame), ["x", "y"], k=3,
                            stability="sample", fraction=1.0, repeats=2)
    assert sizes == [12, 12, 12]
    assert whole["stability"]["fraction"] == 1.0

    # 行の数と群の数が同じ (どの行も 1 つの群で, 作り直しても同じ)
    sizes.clear()
    small = describe_groups(table(frame.head(3)), ["x", "y"], k=3,
                            stability="sample")
    assert sizes == [3] * 6
    assert small["stability"]["ari"] == 1.0
    assert all(row["stability"] == 1.0 for row in small["groups"][:-1])


def test_width_limit_with_labels() -> None:
    """
    群を渡した場合も, 符号化した行の幅の上限で止まるかを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 種類の値が上限より多い列
    count = MAX_WIDTH + 1
    wide = table(pd.DataFrame({
        "name": [str(index) for index in range(count)],
        "g": [index % 2 for index in range(count)],
    }))
    with pytest.raises(ValueError, match="上限"):
        describe_groups(wide, ["name"], labels="g")


def test_silhouette_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    2000 行を超える表で, 種から作り直した乱数で選んだ行のシルエット係数
    を返すかを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        渡された乱数を記録する

    Returns
    ----------
    None
    """
    # 2 つの群の 2500 行
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": np.concatenate([
        rng.normal(0, 1, 1250), rng.normal(8, 1, 1250)
    ])})
    seen = []
    original = groups.silhouette

    def record(points: np.ndarray, labels: np.ndarray, size: int,
               pick: np.random.Generator) -> float:
        # 渡された乱数の初めの状態
        seen.append(pick.bit_generator.state)
        score = original(points, labels, size, pick)

        return score

    monkeypatch.setattr(groups, "silhouette", record)
    summary = describe_groups(table(frame), ["x"], k=2, seed=3)

    # 乱数は種 3 から作り直したもの, 値は同じ種で数え直した値
    assert seen == [np.random.default_rng(3).bit_generator.state]
    encoding = groups.make_encoding(frame)
    points = groups.encode_rows(frame, encoding)
    labels = np.array([0 if row["x"] < 4 else 1 for _, row in
                       frame.iterrows()])
    if summary["groups"][0]["x"] > 4:
        labels = 1 - labels
    again = original(points, labels, groups.SAMPLE,
                     np.random.default_rng(3))
    assert summary["silhouette"] == pytest.approx(again)
    assert 0.5 < summary["silhouette"] <= 1


def test_csv_values() -> None:
    """
    csv で読んだ列の値の項目の, 空の文字と数にならない文字の読み方を
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 数の項目は数に (数にならない文字は欠け), 他は文字 (空は欠け)
    entries = {
        "x": {"placement": "column", "dtype": "float64"},
        "c": {"placement": "column", "dtype": "str"},
        "image": {"placement": "file"},
    }
    items = pd.DataFrame({
        "x": ["1.5", "", "many"], "c": ["01", "", "a"],
        "image": ["", "b.png", ""],
    })
    read = csv_values(items, entries)
    assert read["x"].iloc[0] == 1.5 and read["x"].iloc[1:].isna().all()
    assert read["c"].iloc[0] == "01" and pd.isna(read["c"].iloc[1])
    assert list(read["image"]) == ["", "b.png", ""]
