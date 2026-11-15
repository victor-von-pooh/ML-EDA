import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    describe_loaders, describe_steps, detect, export_dataset, load_data,
    load_dataset, load_recipe, prepare, preview, profile
)


def make_table(rows: int = 300) -> pd.DataFrame:
    """
    気付きが出る列を持つ表を作る関数

    Parameters
    ----------
    rows: int = 300
        行の数

    Returns
    ----------
    frame: pd.DataFrame
        ID, 欠損の多い数値, 偏った数値, カテゴリ, 定数, 偏ったラベル,
        ラベルから作った列を持つ表
    """
    # 乱数を固定して列を作る
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "id": range(rows),
        "age": rng.normal(40, 10, rows).round(),
        "income": np.exp(rng.normal(10, 1, rows)),
        "city": rng.choice(list("ABCDE"), rows),
        "const": 1,
        "label": rng.choice(["no", "yes"], rows, p=[0.95, 0.05]),
    })
    frame.loc[:100, "age"] = np.nan
    frame["leak"] = (frame["label"] == "yes").astype(int) * 3

    return frame


def test_detect_and_load_formats(tmp_path: Path) -> None:
    """
    csv, tsv, json, jsonl, parquet, Excel, npy を判定して読めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ表を形式を変えて保存
    frame = make_table(50)
    frame.to_csv(tmp_path / "a.csv", index=False)
    frame.to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    frame.to_csv(tmp_path / "a.txt", sep=";", index=False)
    frame.to_json(tmp_path / "a.json", orient="records")
    frame.to_json(tmp_path / "a.jsonl", orient="records", lines=True)
    np.save(tmp_path / "a.npy", np.arange(12.0).reshape(4, 3))
    names = ["a.csv", "a.tsv", "a.txt", "a.json", "a.jsonl"]

    # 任意の依存 (pyarrow, openpyxl) がある場合は parquet と Excel も
    try:
        frame.to_parquet(tmp_path / "a.parquet")
        names.append("a.parquet")
    except ImportError:
        pass
    try:
        frame.to_excel(tmp_path / "a.xlsx", index=False)
        names.append("a.xlsx")
    except ImportError:
        pass

    # どれも表として読め, 行と列が揃う
    for name in names:
        data = load_data(tmp_path / name)
        assert data.kind == "table", name
        assert data.frame.shape == frame.shape, name
        assert detect(tmp_path / name)[0]["loader"] == data.source["loader"]
    assert load_data(tmp_path / "a.npy").frame.shape == (4, 3)

    # 入れ子の JSON と, shift_jis の csv
    nested = {"meta": 1, "rows": [{"a": {"b": 1}, "c": 2}] * 3}
    (tmp_path / "n.json").write_text(json.dumps(nested), encoding="utf-8")
    assert set(load_data(tmp_path / "n.json").frame.columns) == {"a.b", "c"}
    frame[["city", "age"]].rename(columns={"city": "都市"}).to_csv(
        tmp_path / "sjis.csv", index=False, encoding="cp932"
    )
    assert "都市" in load_data(tmp_path / "sjis.csv").frame.columns
    assert {item["name"] for item in describe_loaders()} >= {
        "csv", "json", "parquet", "excel", "npy"
    }


def test_profile_findings(tmp_path: Path) -> None:
    """
    列の型の推定と, 気付き (ID, 欠損, 定数, 偏り, リーク, 偏った分布) を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 分析して JSON にできることの確認
    make_table().to_csv(tmp_path / "t.csv", index=False)
    data = load_data(tmp_path / "t.csv")
    result = profile(data, target="label")
    json.dumps(result)

    # 列の型
    types = {name: info["type"] for name, info in result["columns"].items()}
    assert types["id"] == "id"
    assert types["age"] == "numeric"
    assert types["city"] == "categorical"
    assert types["const"] == "constant"
    assert result["target"]["task"] == "classification"

    # 気付きの識別子
    codes = {item["code"] for item in result["findings"]}
    assert {
        "id_like", "high_missing", "constant", "class_imbalance",
        "target_leak", "skewed"
    } <= codes
    assert result["findings"][0]["severity"] == "warning"


def test_timeseries(tmp_path: Path) -> None:
    """
    日時の列から時系列と推定し, 間隔, 抜け, 周期を求めることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 7 日周期の系列から 2 日を抜き, 順番を崩して保存
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=200, freq="D")
        .astype(str),
        "value": np.sin(np.arange(200) * 2 * np.pi / 7)
        + rng.normal(0, 0.1, 200),
    }).drop(index=[50, 51]).sample(frac=1, random_state=0)
    frame.to_csv(tmp_path / "ts.csv", index=False)

    # 時刻の順に並び, 1 日の間隔, 抜け 1 か所, 周期 7
    data = load_data(tmp_path / "ts.csv")
    assert data.kind == "timeseries"
    assert data.frame["date"].is_monotonic_increasing
    series = profile(data)["timeseries"]
    assert series["interval_seconds"] == 86400
    assert series["gaps"] == 1
    assert series["columns"]["value"]["period"] == 7


def test_recipe_export_and_load(tmp_path: Path) -> None:
    """
    レシピで前処理と層化の分割をし, 書き出して読み込み直せることと,
    当てはめが学習用のデータだけで行われることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 前処理のレシピ
    frame = make_table()
    frame.to_csv(tmp_path / "t.csv", index=False)
    data = load_data(tmp_path / "t.csv")
    recipe = {
        "target": "label", "task": "classification",
        "steps": [
            {"op": "drop_columns", "columns": ["id", "leak", "const"]},
            {"op": "impute", "columns": ["age"], "strategy": "median",
             "indicator": True},
            {"op": "log", "columns": ["income"]},
            {"op": "one_hot", "columns": "auto"},
            {"op": "standardize", "columns": ["age", "income"]},
        ],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.1},
    }

    # 1 段ずつの前と後
    view = preview(data, recipe, 4)
    assert "city=A" in view["added"] and "city" in view["removed"]

    # 標準化は学習用のデータだけで当てはめる (学習用の平均は 0)
    prepared = prepare(data, recipe)
    train = prepared.frames["train"]
    assert abs(train["age"].mean()) < 1e-9
    assert abs(prepared.frames["valid"]["age"].mean()) > 1e-6
    assert set(prepared.frames["valid"]["label"]) == {0, 1}

    # 書き出しと読み込み
    info = export_dataset(data, recipe, tmp_path / "ds")
    assert info["classes"] == ["no", "yes"]
    assert sum(item["rows"] for item in info["splits"].values()) == 300
    dataset = load_dataset(tmp_path / "ds")
    x, y = dataset.arrays("train")
    assert x.shape == (info["splits"]["train"]["rows"], len(dataset.features))
    assert y.dtype == np.int64

    # 新しいデータに同じ前処理 (当てはめ直さない)
    new = dataset.transform(frame.head(5))
    assert list(new.columns) == dataset.features
    # 欠損だった age は, 学習用の中央値で埋めてから学習用の平均と幅で変換
    steps = dataset.recipe["steps"]
    median = steps[1]["state"]["values"]["age"]
    center, scale = steps[4]["state"]["values"]["age"]
    assert np.isclose(new["age"].iloc[0], (median - center) / scale)

    # 既にあるディレクトリは置き換えない
    with pytest.raises(FileExistsError):
        export_dataset(data, recipe, tmp_path / "ds")


def test_time_split_and_lag(tmp_path: Path) -> None:
    """
    時系列の時刻の順の分割と, 過去の値の列を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 時刻の順に並べた系列
    frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=100, freq="h"),
        "y": np.arange(100.0),
    })
    frame.to_csv(tmp_path / "ts.csv", index=False)
    data = load_data(tmp_path / "ts.csv")
    recipe = {
        "target": "y", "task": "forecasting",
        "steps": [
            {"op": "lag", "columns": ["y"], "lags": [1, 2], "windows": [3]},
            {"op": "drop_missing"},
            {"op": "datetime_parts", "columns": ["time"],
             "parts": ["hour", "weekday"]},
        ],
        "split": {"method": "time", "valid": 0.2},
    }

    # 検証用は学習用より後の時刻で, ずらした値は 1 つ前の値
    prepared = prepare(data, recipe)
    train, valid = prepared.frames["train"], prepared.frames["valid"]
    assert train["y"].max() < valid["y"].min()
    assert (train["y_lag1"] == train["y"] - 1).all()
    assert "time_hour" in prepared.features
    assert "y" not in prepared.features

    # lag の auto は目的の列も選ぶ (目的の列そのものは学習に使わない)
    recipe["steps"][0] = {"op": "lag", "columns": "auto", "lags": [1]}
    prepared = prepare(data, recipe)
    assert "y_lag1" in prepared.features and "y" not in prepared.features


def test_recipe_errors() -> None:
    """
    レシピの誤り (新しい版, 知らない手順, 知らない分割) を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # どれも理由付きの ValueError
    with pytest.raises(ValueError, match="format_version"):
        load_recipe({"format_version": 99})
    with pytest.raises(ValueError, match="op"):
        load_recipe({"steps": [{"op": "magic"}]})
    with pytest.raises(ValueError, match="分割"):
        load_recipe({"split": {"method": "magic"}})
    assert {step["op"] for step in describe_steps()} >= {
        "impute", "standardize", "one_hot", "lag"
    }


def test_units_dates_and_series(tmp_path: Path) -> None:
    """
    単位の付いた数, 書き方の混ざった日付, 店ごとの系列 (系列ごとのずらし)
    を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 店の 30 日分. 売上は円付き, 日付は 3 つの書き方
    rows = []
    for store, base in (("A店", 100), ("B店", 1000)):
        for day in range(30):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
            text = (
                f"{date.year}年{date.month}月{date.day}日" if day % 3 == 0
                else date.strftime("%Y/%m/%d") if day % 3 == 1
                else date.strftime("%Y-%m-%d")
            )
            rows.append({
                "date": text, "store": store,
                "sales": f"{base + day:,}円", "rate": f"{day % 5}%",
            })
    pd.DataFrame(rows).sample(frac=1, random_state=0).to_csv(
        tmp_path / "s.csv", index=False
    )
    data = load_data(tmp_path / "s.csv")
    assert data.kind == "timeseries" and data.roles == {"time": "date"}
    assert data.frame["sales"].dtype == float
    assert data.frame["rate"].max() == 4

    # 店ごとの系列の気付きと, 間隔は 1 つの系列で
    result = profile(data, target="sales")
    codes = {item["code"] for item in result["findings"]}
    assert "panel_series" in codes and "irregular_interval" not in codes
    assert result["timeseries"]["series_by"] == "store"

    # 系列ごとにずらすと, 1 つ前の値は同じ店の前の日
    recipe = {
        "target": "sales", "task": "forecasting",
        "steps": [{"op": "lag", "columns": ["sales"], "lags": [1],
                   "windows": [2], "by": ["store"]}],
        "split": {"method": "none"},
    }
    frame = prepare(data, recipe).frames["train"]
    b = frame[frame["store"] == "B店"]
    assert (b["sales"] - b["sales_lag1"]).dropna().eq(1).all()
    assert b["sales_mean2"].dropna().min() >= 1000
