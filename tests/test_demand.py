from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    describe_demand, export_dataset, load_data, load_dataset, load_recipe,
    model_inputs, prepare
)
from ml_eda.data import Data


def make_sales(
    days: int = 120, products: int = 4, stock: bool = True
) -> pd.DataFrame:
    """
    商品ごとに曜日と水準を変えた日ごとの売れた数の記録の表を作る関数

    Parameters
    ----------
    days: int = 120
        日数
    products: int = 4
        商品の数
    stock: bool = True
        在庫の数の列を足すか

    Returns
    ----------
    frame: pd.DataFrame
        商品の id, 日付, 特徴 (大きさ), 売れた数 (と在庫の数) の表
    """
    # 商品ごとの水準と, 商品ごとに違う曜日の山
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for product in range(products):
        level = 5.0 + 3.0 * product
        peak = product % 7
        for date in dates:
            lift = 2.0 if date.weekday() == peak else 1.0
            row = {
                "product_id": f"p{product}",
                "date": date,
                "size": float(product),
                "units": int(rng.poisson(level * lift)),
            }
            if stock:
                row["stock"] = float(rng.integers(1, 30))
            rows.append(row)
    frame = pd.DataFrame(rows)

    return frame


def demand_recipe(**changes: object) -> dict:
    """
    需要の予測のレシピを作る関数

    Parameters
    ----------
    **changes: object
        上書きするレシピの項目

    Returns
    ----------
    recipe: dict
        過去の売れた数のずらしと移動平均, 曜日の分解をするレシピ
    """
    # 時刻の順の分割 (発注を決める日の順)
    recipe = {
        "format_version": 1,
        "task": "demand",
        "target": "units",
        "product": "product_id",
        "date": "date",
        "stock": "stock",
        "horizon": 7,
        "steps": [
            {"op": "lag", "columns": ["units"], "lags": [1, 7],
             "windows": [7], "by": ["product_id"], "time": "date",
             "freq": "1D"},
            {"op": "datetime_parts", "columns": ["date"],
             "parts": ["weekday"]},
        ],
        "split": {"method": "time", "valid": 0.2, "test": 0.2},
    }
    recipe.update(changes)

    return recipe


def table_data(frame: pd.DataFrame) -> Data:
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


def daily_sum(
    frame: pd.DataFrame, product: str, start: pd.Timestamp, days: int
) -> float:
    """
    商品の start から days 日の売れた数の合計を求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        元の記録の表
    product: str
        商品の id
    start: pd.Timestamp
        最初の日
    days: int
        日数

    Returns
    ----------
    total: float
        合計
    """
    # 期間の行の合計
    end = start + pd.Timedelta(days=days)
    rows = frame[
        (frame["product_id"] == product) & (frame["date"] >= start)
        & (frame["date"] < end)
    ]
    assert len(rows) == days
    total = float(rows["units"].sum())

    return total


def test_export_with_stock(tmp_path: Path) -> None:
    """
    在庫の数のある記録を書き出して読み, 目的の値が先の期間の合計, 比べる
    相手が前の期間の合計, 在庫の数が発注を決める日の値で, 学習に使う列と
    モデルの入力が合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出しと読み込み
    frame = make_sales()
    export_dataset(table_data(frame), demand_recipe(), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    info = dataset.info["demand"]
    assert dataset.target == "demand"
    assert info["target"] == "demand" and info["previous"] == (
        "demand_previous"
    )
    assert info["horizon"] == 7 and info["interval"] == 7
    assert info["stock"] == "stock" and info["units"] == "units"
    assert dataset.features == [
        "size", "units_lag1", "units_lag7", "units_mean7", "date_weekday"
    ]

    # 学習のスクリプトが読む目的の値, 前の期間の合計, 在庫の数
    for split in ("train", "valid", "test"):
        part = dataset.frames[split]
        x, y = dataset.arrays(split)
        previous = dataset.previous(split)
        stock = dataset.stock(split)
        assert x.shape == (len(part), 5) and y.dtype == np.float32
        for row in range(len(part)):
            product = part["product_id"].iloc[row]
            day = pd.Timestamp(part["date"].iloc[row])
            assert y[row] == daily_sum(frame, product, day, 7)
            assert previous[row] == daily_sum(
                frame, product, day - pd.Timedelta(days=7), 7
            )
            original = frame[
                (frame["product_id"] == product) & (frame["date"] == day)
            ]
            assert stock[row] == original["stock"].iloc[0]
            assert x[row, 1] == frame[
                (frame["product_id"] == product)
                & (frame["date"] == day - pd.Timedelta(days=1))
            ]["units"].iloc[0]
        assert info["splits"][split]["rows"] == len(part)
        assert info["splits"][split]["products"] == 4

    # 発注を決める日は 7 日ごとで, 最初の 7 日 (前の期間が無い) は除く
    days = pd.to_datetime(dataset.frames["train"]["date"])
    assert days.min() == pd.Timestamp("2024-01-08")
    assert ((days - pd.Timestamp("2024-01-01")).dt.days % 7 == 0).all()
    assert info["skipped"]["previous"] == 4

    # モデルの入力の幅
    inputs = model_inputs(tmp_path / "ds")
    assert inputs["inputs"] == [
        {"name": "x", "shape": [None, 5], "dtype": "float32"}
    ]
    assert inputs["task"] == "demand" and inputs["outputs"] == 1
    assert inputs["target"] == "demand" and inputs["classes"] is None


def test_export_without_stock(tmp_path: Path) -> None:
    """
    在庫の数の無い記録を書き出すと, 在庫の数が None で, 説明の stock と
    stockout が null になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 在庫の数の列を書かないレシピ
    recipe = demand_recipe()
    del recipe["stock"]
    export_dataset(
        table_data(make_sales(stock=False)), recipe, tmp_path / "ds"
    )
    dataset = load_dataset(tmp_path / "ds")
    assert dataset.stock("train") is None
    assert dataset.info["demand"]["stock"] is None
    assert dataset.info["demand"]["splits"]["train"]["stockout"] is None
    assert len(dataset.previous("valid")) == len(dataset.frames["valid"])


def test_export_from_file(tmp_path: Path) -> None:
    """
    CSV を load_data で読んだ時系列の記録を書き出して読み, 要約もファイル
    から読めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 時系列として読んだ記録
    file = tmp_path / "daily_sales.csv"
    make_sales().to_csv(file, index=False)
    data = load_data(file)
    assert data.kind == "timeseries"
    export_dataset(data, demand_recipe(), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    x, y = dataset.arrays("valid")
    assert x.shape[1] == 5 and len(y) == len(dataset.previous("valid"))
    summary = describe_demand(file, "product_id", "date", "units")
    assert summary["rows"] == 480 and summary["source"]["loader"]


@pytest.mark.parametrize("interval", [1, 3, 7])
def test_periods_inside_splits(interval: int) -> None:
    """
    先の期間が分割の境目をまたがず, またぐ行を落とした数が説明と注意に
    残ることを確かめる関数

    Parameters
    ----------
    interval: int
        発注を決める日の間隔

    Returns
    ----------
    None
    """
    # 分割ごとの発注を決める日と先の期間の終わり
    prepared = prepare(
        table_data(make_sales()), demand_recipe(interval=interval)
    )
    frames = prepared.frames
    first = {
        name: pd.to_datetime(frames[name]["date"]).min()
        for name in ("valid", "test")
    }
    for name, later in (("train", "valid"), ("valid", "test")):
        reach = pd.to_datetime(frames[name]["date"]) + pd.Timedelta(days=6)
        assert (reach < first[later]).all()
    boundary = prepared.demand["skipped"]["boundary"]
    if interval < 7:
        assert boundary > 0
        assert any("境目をまたぐ" in note for note in prepared.notes)
    else:
        assert boundary == 0


def test_date_gaps(tmp_path: Path) -> None:
    """
    記録の無い日を含む期間の行を作らず, 要約が日の抜けの気付きを出すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # p0 の 2024-02-10 の行を落とす
    frame = make_sales()
    gap = pd.Timestamp("2024-02-10")
    frame = frame[
        ~((frame["product_id"] == "p0") & (frame["date"] == gap))
    ].reset_index(drop=True)
    prepared = prepare(table_data(frame), demand_recipe())
    for part in prepared.frames.values():
        rows = part[part["product_id"] == "p0"]
        days = pd.to_datetime(rows["date"])
        touches = (days <= gap + pd.Timedelta(days=7)) & (
            days + pd.Timedelta(days=6) >= gap
        )
        assert not touches.any()

    # 日の抜けの気付き
    summary = describe_demand(
        table_data(frame), "product_id", "date", "units", stock="stock"
    )
    codes = [item["code"] for item in summary["findings"]]
    assert "date_gaps" in codes
    assert summary["products"]["gaps"] == 1
    row = next(
        item for item in summary["products"]["rows"]
        if item["product"] == "p0"
    )
    assert row["gaps"] == 1 and row["days"] == 119


def test_stockout_days() -> None:
    """
    在庫が 0 の日の気付きと, 先の期間に在庫が 0 の日を含む行の数が
    出ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # p1 の 2024-02-14 から 3 日を在庫 0 に
    frame = make_sales()
    empty = (frame["product_id"] == "p1") & frame["date"].between(
        "2024-02-14", "2024-02-16"
    )
    frame.loc[empty, "stock"] = 0.0
    frame.loc[empty, "units"] = 0
    summary = describe_demand(
        table_data(frame), "product_id", "date", "units", stock="stock"
    )
    codes = [item["code"] for item in summary["findings"]]
    assert "stockout_days" in codes
    assert summary["stockout"]["rows"] == 3
    assert summary["stockout"]["products"] == 1

    # 2024-02-12 の行の先の期間 (2024-02-12 から 7 日) が在庫 0 の日を含む
    prepared = prepare(table_data(frame), demand_recipe())
    total = sum(
        item["stockout"] for item in prepared.demand["splits"].values()
    )
    assert total == 1


def test_keep_unlabeled(tmp_path: Path) -> None:
    """
    split.unlabeled: keep で, 先の期間が記録の終わりを越える発注を決める
    日の行が unlabeled の分割に入り, 目的の値が無いことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 120 日の記録の最後の発注を決める日 (2024-04-29) は 2 日しか無い
    recipe = demand_recipe(
        split={"method": "time", "valid": 0.2, "test": 0.2,
               "unlabeled": "keep"}
    )
    export_dataset(table_data(make_sales()), recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    part = dataset.frames["unlabeled"]
    assert len(part) == 4
    assert (pd.to_datetime(part["date"]) == "2024-04-29").all()
    x, y = dataset.arrays("unlabeled")
    assert y is None and x.shape == (4, 5)
    assert not np.isnan(dataset.previous("unlabeled")).any()
    assert dataset.info["demand"]["splits"]["unlabeled"]["demand_mean"] is (
        None
    )


@pytest.mark.parametrize(("change", "message"), [
    ({"units": [-1]}, "負か無限大"),
    ({"stock": [np.nan]}, "在庫の数の列 stock に欠け"),
    ({"stock": [-2.0]}, "在庫の数の列 stock に負か無限大"),
    ({"product_id": [None]}, "商品の id の列 product_id に欠け"),
    ({"date": ["2024-01-01 10:00"]}, "時刻のある行"),
    ({"date": ["not a date"]}, "読めない行"),
])
def test_value_errors(change: dict, message: str) -> None:
    """
    役割の列の値の誤りで書き出しが止まることを確かめる関数

    Parameters
    ----------
    change: dict
        最初の行を書き換える列と値
    message: str
        誤りの文の一部

    Returns
    ----------
    None
    """
    # 最初の行だけを書き換える (文字と欠けを入れる列は object に)
    frame = make_sales()
    for name, values in change.items():
        if name == "date":
            frame[name] = frame[name].astype(str)
        if name in ("date", "product_id"):
            frame[name] = frame[name].astype(object)
        frame.loc[0, name] = values[0]
    with pytest.raises(ValueError, match=message):
        prepare(table_data(frame), demand_recipe())


def test_duplicate_days() -> None:
    """
    同じ商品の同じ日の行があると書き出しが止まり, 要約が気付きを出すことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 最初の行を重ねる
    frame = make_sales()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="同じ商品の同じ日の行"):
        prepare(table_data(frame), demand_recipe())
    summary = describe_demand(
        table_data(frame), "product_id", "date", "units"
    )
    assert summary["duplicates"] == 1
    assert "duplicate_days" in [item["code"] for item in summary["findings"]]


@pytest.mark.parametrize(("change", "message"), [
    ({"split": {"method": "random"}}, "time だけ"),
    ({"horizon": 0}, "horizon は 1 以上の整数"),
    ({"interval": 1.5}, "interval は 1 以上の整数"),
    ({"date": None}, "日付の列"),
    ({"product": "units"}, "違う列"),
    ({"stock": "date"}, "違う列"),
])
def test_recipe_errors(change: dict, message: str) -> None:
    """
    需要の予測のレシピの誤りを load_recipe が止めることを確かめる関数

    Parameters
    ----------
    change: dict
        上書きするレシピの項目
    message: str
        誤りの文の一部

    Returns
    ----------
    None
    """
    # 読んだ時点の確かめ
    with pytest.raises(ValueError, match=message):
        load_recipe(demand_recipe(**change))


@pytest.mark.parametrize("key", ["date", "stock", "horizon", "interval"])
def test_demand_keys_other_task(key: str) -> None:
    """
    需要の予測だけの鍵を他のタスクで書くと誤りになることを確かめる関数

    Parameters
    ----------
    key: str
        需要の予測だけの鍵

    Returns
    ----------
    None
    """
    # 回帰のレシピに足す
    recipe = {"format_version": 1, "task": "regression", "target": "units",
              key: "date" if key in ("date", "stock") else 7}
    with pytest.raises(ValueError, match="需要の予測"):
        load_recipe(recipe)


def test_lag_after_fit() -> None:
    """
    値を学ぶ手順より後の lag で書き出しが止まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 標準化の後に lag
    recipe = demand_recipe(steps=[
        {"op": "standardize", "columns": ["size"]},
        {"op": "lag", "columns": ["units"], "lags": [1],
         "by": ["product_id"], "time": "date"},
    ])
    with pytest.raises(ValueError, match="lag は, 値を学ぶ最初の手順より前"):
        prepare(table_data(make_sales()), recipe)


def test_reserved_and_fixed_columns() -> None:
    """
    表に期間の合計の列と同じ名前の列があると誤りで, 在庫の数の列は手順で
    変えられず, auto の手順が役割の列を選ばないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 期間の合計の列と重なる名前
    frame = make_sales()
    with pytest.raises(ValueError, match="demand_previous の列"):
        prepare(
            table_data(frame.assign(demand_previous=1.0)), demand_recipe()
        )

    # 在庫の数の列を手順で変える
    recipe = demand_recipe()
    recipe["steps"] = recipe["steps"] + [
        {"op": "standardize", "columns": ["stock"]}
    ]
    with pytest.raises(ValueError, match="在庫の数の列 stock は手順で"):
        prepare(table_data(frame), recipe)

    # auto の標準化は売れた数, 合計, 在庫の数を選ばない
    recipe = demand_recipe()
    recipe["steps"] = recipe["steps"] + [
        {"op": "standardize", "columns": "auto"}
    ]
    prepared = prepare(table_data(frame), recipe)
    chosen = prepared.recipe["steps"][-1]["state"]["columns"]
    assert "units" not in chosen and "stock" not in chosen
    assert "units_lag1" in chosen
    train = prepared.frames["train"]
    assert (train["demand"] >= 0).all() and (train["stock"] >= 1).all()


def test_describe_clean() -> None:
    """
    揃った記録の要約の日付の範囲, 曜日ごとの売れた数, 商品ごとの記録の
    日数と, 気付きが出ないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 4 つの商品の 120 日
    frame = make_sales()
    summary = describe_demand(
        table_data(frame), "product_id", "date", "units", stock="stock"
    )
    assert summary["rows"] == 480 and summary["duplicates"] == 0
    assert summary["dates"] == {
        "min": "2024-01-01", "max": "2024-04-29", "days": 120
    }
    assert summary["products"]["count"] == 4
    assert summary["products"]["short"] == 0
    assert len(summary["weekdays"]) == 7
    assert sum(item["rows"] for item in summary["weekdays"]) == 480
    assert summary["stockout"]["rows"] == 0
    assert summary["findings"] == []

    # p0 の山の曜日 (月曜) は他の曜日より多い
    p0 = describe_demand(
        table_data(frame[frame["product_id"] == "p0"]), "product_id",
        "date", "units"
    )
    means = [item["units_mean"] for item in p0["weekdays"]]
    assert means[0] == max(means)


def test_describe_short_and_missing() -> None:
    """
    記録の短い商品, 売れた数の欠け, 日付の欠けの気付きを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # p3 を 20 日 (28 日未満) に縮め, 売れた数と日付を 1 つずつ欠く
    frame = make_sales()
    frame = frame[
        (frame["product_id"] != "p3")
        | (frame["date"] < pd.Timestamp("2024-01-21"))
    ].reset_index(drop=True)
    frame["units"] = frame["units"].astype(float)
    frame.loc[5, "units"] = np.nan
    frame["date"] = frame["date"].astype(object)
    frame.loc[6, "date"] = None
    summary = describe_demand(
        table_data(frame), "product_id", "date", "units"
    )
    codes = [item["code"] for item in summary["findings"]]
    assert {"short_history", "units_missing", "date_missing"} <= set(codes)
    assert summary["missing"]["units"] == 1
    assert summary["missing"]["date"] == 1
    assert summary["products"]["short"] == 1
    assert summary["products"]["rows"][0]["product"] == "p3"
    assert summary["stock"] is None and summary["stockout"] is None


def test_describe_errors() -> None:
    """
    要約の引数の誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同じ列, 無い列, 数の日付, 先の期間の日数
    data = table_data(make_sales())
    with pytest.raises(ValueError, match="違う列"):
        describe_demand(data, "product_id", "date", "date")
    with pytest.raises(ValueError, match="ありません"):
        describe_demand(data, "product_id", "day", "units")
    with pytest.raises(ValueError, match="日付ではありません"):
        describe_demand(data, "product_id", "size", "units")
    with pytest.raises(ValueError, match="horizon"):
        describe_demand(data, "product_id", "date", "units", horizon=0)
