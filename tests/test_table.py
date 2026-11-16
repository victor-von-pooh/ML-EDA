import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    describe_loaders, describe_steps, detect, export_dataset, load_data,
    load_dataset, load_recipe, prepare, preview, profile
)
from ml_eda.loaders import table_kind, to_datetime, to_numbers
from ml_eda.pipeline import assign_splits
from ml_eda.recipe import apply_lag


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
    assert data.kind == "timeseries"
    assert data.roles == {"time": "date", "series": "store"}
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


def test_review_fixes_units_panel() -> None:
    """
    単位とマイナスの書き方, 日本語の日時, 注文の表を時系列にしないこと,
    系列の列の欠損を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 単位の違う値が混ざれば数にしない. 0 で始まる値も. マイナスは前でも
    assert to_numbers(pd.Series(["1kg", "500g", "2kg"])) is None
    assert to_numbers(pd.Series(["001件", "002件"])) is None
    assert to_numbers(pd.Series(["-¥100", "¥-200", "¥300"])).tolist() == [
        -100, -200, 300
    ]

    # 日と時の間に空白の無い日時, 時だけ, 年月だけ
    stamps = to_datetime(pd.Series(["2024年1月3日12時30分", "2024年1月4日9時"]))
    assert stamps.dt.hour.tolist() == [12, 9]
    assert to_datetime(pd.Series(["2024年1月", "2024年2月"])).notna().all()

    # 注文の表 (顧客ごとの日がまばら) は時系列にしない
    rng = np.random.default_rng(0)
    orders = pd.DataFrame({
        "date": pd.Timestamp("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 365, 2000), unit="D"),
        "customer": rng.integers(0, 150, 2000).astype(str),
    })
    assert table_kind(orders)[0] == "table"

    # 系列の列の欠損は, 1 つの系列として扱う (ずらした値を欠損にしない)
    frame = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0], "s": ["a", None, "a", None]
    })
    result = apply_lag(
        frame, {"columns": ["y"], "lags": [1], "by": ["s"]}, {}
    )
    assert result["y_lag1"].tolist()[2:] == [1.0, 2.0]



def test_time_split_keeps_same_time_together() -> None:
    """
    時刻の順の分割で, 同じ時刻の行が学習用と検証用にまたがらないことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 3 店の 10 日分 (30 行) を 2 割で分けると, 境目が日の途中になる
    frame = pd.DataFrame({
        "date": np.repeat(pd.date_range("2024-01-01", periods=10), 3),
        "store": ["a", "b", "c"] * 10,
    })
    labels = assign_splits(
        frame, {"method": "time", "valid": 0.25}, {"time": "date"}
    )
    per_day = frame.assign(split=labels).groupby("date")["split"].nunique()
    assert (per_day == 1).all()
    last_train = frame["date"][labels == "train"].max()
    assert last_train < frame["date"][labels == "valid"].min()


def test_panel_needs_shared_times(tmp_path: Path) -> None:
    """
    時刻が行ごとに違う表の 2 値の列を, 系列を分ける列にしないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 投稿の時刻は行ごとに違い, 重複した行だけで時刻が重なるレビュー
    times = pd.date_range("2025-01-01", periods=400, freq="h")
    frame = pd.DataFrame({
        "posted": times.strftime("%Y/%m/%d %H:%M"),
        "stars": [1, 5] * 200,
        "label": ["悪い", "良い"] * 200,
    })
    frame = pd.concat([frame, frame.head(20)])
    path = tmp_path / "reviews.csv"
    frame.to_csv(path, index=False)
    data = load_data(path)
    assert data.kind == "timeseries" and "series" not in data.roles

    # 同じ時刻に 2 つの店が並べば系列
    shops = pd.DataFrame({
        "date": list(times.strftime("%Y-%m-%d %H:%M")) * 2,
        "shop": ["A"] * 400 + ["B"] * 400,
        "sales": range(800),
    })
    shops.to_csv(path, index=False)
    assert load_data(path).roles.get("series") == "shop"

    # 片方の店が後から始まっても (並ぶ時刻が 4 割) 系列
    late = shops[(shops["shop"] == "A") | (shops.index >= 640)]
    late.to_csv(path, index=False)
    assert load_data(path).roles.get("series") == "shop"


def test_excel_sheets(tmp_path: Path) -> None:
    """
    Excel の全てのシートの名前を記録し, 他のシートを気付きにすることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 枚のシート (読み込みの既定は最初のシート)
    pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1, 2, 3]}).to_excel(
            writer, sheet_name="表紙", index=False
        )
        pd.DataFrame({"b": [4, 5, 6]}).to_excel(
            writer, sheet_name="売上", index=False
        )
    data = load_data(path)
    assert data.source["sheets"] == ["表紙", "売上"]
    assert data.source["sheet"] == "表紙"
    result = profile(data)
    assert result["overview"]["sheets"] == ["表紙", "売上"]
    codes = {item["code"]: item for item in result["findings"]}
    assert "売上" in codes["multiple_sheets"]["message"]

    # 名前で選んだシート
    chosen = load_data(path, sheet="売上")
    assert list(chosen.frame) == ["b"] and chosen.source["sheet"] == "売上"
    assert load_data(path, sheet=1).source["sheet"] == "売上"
    assert load_data(path, sheet=np.int64(-1)).source["sheet"] == "売上"
    with pytest.raises(ValueError, match="シートの番号 5"):
        load_data(path, sheet=5)
    with pytest.raises(ValueError, match="シート 無い"):
        load_data(path, sheet="無い")

    # 1 枚だけなら気付きにしない
    single = tmp_path / "one.xlsx"
    pd.DataFrame({"a": [1, 2]}).to_excel(single, index=False)
    codes = [item["code"] for item in profile(load_data(single))["findings"]]
    assert "multiple_sheets" not in codes


def test_time_lag() -> None:
    """
    時刻でずらす lag (抜けを挟んでも暦でずらす) を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 店 A は 1 月 3 日が休み. 並びは時刻の順でなくてよい
    frame = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-05", "2024-01-01", "2024-01-02", "2024-01-04",
            "2024-01-01", "2024-01-02", "2024-01-03",
        ]),
        "shop": ["A", "A", "A", "A", "B", "B", "B"],
        "v": [5.0, 1.0, 2.0, 4.0, 10.0, 20.0, 30.0],
    })
    params = {
        "columns": ["v"], "lags": [1, 2], "windows": [2], "by": ["shop"],
        "time": "date", "freq": "1D",
    }
    result = apply_lag(frame, params, {})
    a = result[result["shop"] == "A"].set_index("date")
    # 休みの翌日の 1 つ前は, 休みより前の最も近い値 (2 日の 2)
    assert a.loc["2024-01-04", "v_lag1"] == 2.0
    assert a.loc["2024-01-05", "v_lag2"] == 2.0
    # 2 日の移動平均は, その日を含めない直前の 2 日にある値だけ
    assert a.loc["2024-01-05", "v_mean2"] == 4.0
    assert np.isnan(a.loc["2024-01-01", "v_lag1"])
    # 系列をまたがない
    b = result[result["shop"] == "B"].set_index("date")
    assert b.loc["2024-01-03", "v_mean2"] == 15.0
    assert list(result.index) == list(frame.index)

    # exact は, その時刻が無ければ欠損
    exact = apply_lag(frame, {**params, "match": "exact"}, {})
    assert np.isnan(exact.loc[3, "v_lag1"])

    # 同じ時刻の行は移動平均に全て入れる (1 と 10 の平均)
    twice = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "v": [1.0, 10.0, 3.0],
    })
    doubled = apply_lag(twice, {
        "columns": ["v"], "lags": [1], "windows": [2], "time": "date",
    }, {})
    assert doubled.loc[2, "v_mean2"] == 5.5 and doubled.loc[2, "v_lag1"] == 10

    # 書き方の混ざった文字の日付も読む
    mixed = twice.assign(date=["2024/01/01", "2024-01-01", "2024-01-02 00:00"])
    assert apply_lag(mixed, {
        "columns": ["v"], "lags": [1], "time": "date",
    }, {}).loc[2, "v_lag1"] == 10

    # 読めない間隔, 空の間隔, 数値でない列, 時刻の列が無い場合は誤り
    with pytest.raises(ValueError, match="freq"):
        apply_lag(frame, {**params, "freq": "毎日"}, {})
    with pytest.raises(ValueError, match="freq"):
        apply_lag(frame, {**params, "freq": None}, {})
    with pytest.raises(ValueError, match="数値"):
        apply_lag(frame, {**params, "columns": ["shop", "v"], "by": []}, {})
    with pytest.raises(ValueError, match="時刻の列"):
        apply_lag(frame, {**params, "time": "when"}, {})


def test_transform_with_lag(tmp_path: Path) -> None:
    """
    目的の列の過去の値の列を使うデータセットで, 新しいデータに前処理を
    かけられることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 目的の列も過去の値にする (columns: auto)
    days = pd.date_range("2024-01-01", periods=60, freq="D")
    frame = pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "sales": np.arange(60, dtype=float) * 2 + 1,
        "temp": np.linspace(0, 10, 60),
    })
    path = tmp_path / "s.csv"
    frame.to_csv(path, index=False)
    data = load_data(path)
    recipe = {
        "target": "sales", "task": "forecasting",
        "steps": [
            {"op": "lag", "columns": "auto", "lags": [1]},
            {"op": "drop_missing"},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": "time", "valid": 0.2},
    }
    export_dataset(data, recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    assert "sales_lag1" in dataset.features

    # 過去の値を含めて渡すと, 学習のときと同じ値になる (最初の行は過去が
    # 無いので欠損. 学習用は欠損の行を落とした後の並び)
    out = dataset.transform(data.frame)
    train = dataset.frames["train"]
    assert np.isnan(out["sales_lag1"].iloc[0])
    assert np.allclose(
        out["sales_lag1"].iloc[1:1 + len(train)], train["sales_lag1"]
    )

    # 目的の列が無ければ, 足りない列を挙げた誤り
    with pytest.raises(ValueError, match="sales の過去の値"):
        dataset.transform(data.frame.drop(columns=["sales"]))

    # 学習のときに落とした列は, 新しいデータに無くてよい
    recipe["steps"] = [
        {"op": "drop_columns", "columns": ["temp"]}, *recipe["steps"]
    ]
    export_dataset(data, recipe, tmp_path / "ds2")
    dropped = load_dataset(tmp_path / "ds2")
    assert list(dropped.transform(data.frame.drop(columns=["temp"]))) == [
        "sales_lag1"
    ]
