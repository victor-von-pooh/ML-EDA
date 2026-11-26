from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    describe_price, export_dataset, load_data, load_dataset, load_recipe,
    model_inputs, prepare
)
from ml_eda.data import Data
from ml_eda.price import price_layers


def make_sales(count: int = 600, products: int = 30) -> pd.DataFrame:
    """
    商品ごとに値段を無作為に変えた売れ行きの記録の表を作る関数

    Parameters
    ----------
    count: int = 600
        行の数
    products: int = 30
        商品の数

    Returns
    ----------
    frame: pd.DataFrame
        記録の id, 商品の id, 特徴 (大きさ, 分類), 値段, 売れた数, 原価の表
    """
    # 商品ごとの特徴と元の値段, 無作為に変えた値段
    rng = np.random.default_rng(0)
    product = rng.integers(0, products, count)
    size = rng.normal(0.0, 1.0, products)[product]
    group = np.array(["food", "tool", "toy"])[product % 3]
    base = 100.0 + 20.0 * size
    price = base * rng.choice([0.8, 0.9, 1.0, 1.1, 1.2], count)

    # 値段を上げると減る売れた数 (弾力性 -1.5)
    units = rng.poisson(np.exp(3.0 - 1.5 * np.log(price / base)))
    frame = pd.DataFrame({
        "record_id": [f"r{index:05d}" for index in range(count)],
        "product_id": [f"p{index:03d}" for index in product],
        "size": size,
        "group": group,
        "price": price,
        "units": units,
        "unit_cost": base * 0.6,
    })

    return frame


def price_recipe(**changes: object) -> dict:
    """
    値段を変える判断のレシピを作る関数

    Parameters
    ----------
    **changes: object
        上書きするレシピの項目

    Returns
    ----------
    recipe: dict
        値段の対数を取り, 分類を符号化し, 数の列を標準化するレシピ
    """
    # 層化の分割 (値段の段で層を作る)
    recipe = {
        "format_version": 1,
        "task": "price",
        "target": "units",
        "price": "price",
        "cost": "unit_cost",
        "product": "product_id",
        "id": "record_id",
        "steps": [
            {"op": "log", "columns": ["price"]},
            {"op": "one_hot", "columns": "auto"},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.2,
                  "seed": 0},
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
    file = path / "sales.csv"
    frame.to_csv(file, index=False)

    return file


def test_export_with_cost(tmp_path: Path) -> None:
    """
    原価のある記録を書き出して読み, 値段の列が学習に使う列の最後に入り,
    元の値段, 原価, 商品の id, 行の id が残り, 説明が分割と合うことを
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
    frame = make_sales()
    data = load_data(write_table(frame, tmp_path))
    info = export_dataset(data, price_recipe(), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    assert set(dataset.frames) == {"train", "valid", "test"}
    assert sum(len(item) for item in dataset.frames.values()) == 600
    assert info["task"] == "price"
    assert info["classes"] == []

    # 学習に使う列 (値段の列は最後. 役割の列は除く)
    assert dataset.features[-1] == "price"
    for name in ("units", "unit_cost", "product_id", "record_id",
                 "price_raw"):
        assert name not in dataset.features
    assert "size" in dataset.features
    assert any(name.startswith("group") for name in dataset.features)

    # 説明の price
    summary = info["price"]
    assert summary["id"] == "record_id"
    assert summary["price"] == "price"
    assert summary["raw"] == "price_raw"
    assert summary["cost"] == "unit_cost"
    assert summary["product"] == "product_id"
    assert summary["unlabeled"] == "drop"
    assert summary["steps"] == ["log", "standardize"]
    for name, item in dataset.frames.items():
        entry = summary["splits"][name]
        assert entry["rows"] == len(item)
        assert entry["products"] == item["product_id"].nunique()
        assert entry["price_min"] == pytest.approx(item["price_raw"].min())
        assert entry["price_max"] == pytest.approx(item["price_raw"].max())
        assert entry["prices"] == item["price_raw"].nunique()
        assert entry["units_mean"] == pytest.approx(item["units"].mean())
        assert entry["cost_mean"] == pytest.approx(item["unit_cost"].mean())

    # 元の値の値段, 原価, 商品の id
    merged = pd.concat(dataset.frames.values())
    lookup = frame.set_index("record_id")
    rows = lookup.loc[merged["record_id"]]
    assert np.allclose(merged["price_raw"], rows["price"])
    assert np.allclose(merged["unit_cost"], rows["unit_cost"])
    assert (merged["product_id"].to_numpy() == rows["product_id"]).all()

    # 正解の型, 元の値段と原価の配列
    x, y = dataset.arrays("valid")
    assert y.dtype == np.float32
    assert x.shape[1] == len(dataset.features)
    prices = dataset.price("valid")
    costs = dataset.cost("valid")
    assert prices.dtype == np.float64 and len(prices) == len(y)
    assert costs.dtype == np.float64 and len(costs) == len(y)

    # モデルの入力 (値段の列を含む幅) と出す値の数
    inputs = model_inputs(tmp_path / "ds")
    assert inputs["task"] == "price"
    assert inputs["target"] == "units"
    assert inputs["classes"] is None
    assert inputs["outputs"] == 1
    assert inputs["inputs"][0]["shape"] == [None, len(dataset.features)]


def test_export_without_cost(tmp_path: Path) -> None:
    """
    原価と商品の id の無い記録を書き出して読み, 原価が None になり, 行の
    id が行の位置になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 原価, 商品の id, 行の id を書かないレシピ
    frame = make_sales().drop(columns=["unit_cost", "record_id"])
    recipe = price_recipe()
    for key in ("cost", "product", "id"):
        recipe.pop(key)
    recipe["steps"] = [
        {"op": "drop_columns", "columns": ["product_id"]},
        {"op": "one_hot", "columns": ["group"]},
        {"op": "standardize", "columns": "auto"},
    ]
    info = export_dataset(table_data(frame), recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    summary = info["price"]
    assert summary["cost"] is None
    assert summary["product"] is None
    assert summary["id"] == "row_id"
    assert summary["steps"] == ["standardize"]
    for entry in summary["splits"].values():
        assert entry["products"] is None
        assert entry["cost_mean"] is None
    assert dataset.cost("test") is None
    assert len(dataset.price("test")) == len(dataset.frames["test"])
    assert dataset.features[-1] == "price"
    assert "row_id" not in dataset.features


def test_price_inputs(tmp_path: Path) -> None:
    """
    候補の値段での入力が, 同じ値段なら分割の入力と同じで, 変えた値段なら
    値段を変えた元の表を transform にかけた値と同じになることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出し
    frame = make_sales()
    export_dataset(table_data(frame), price_recipe(), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    x, _ = dataset.arrays("test")
    prices = dataset.price("test")

    # 同じ値段と, 1 割上げた値段
    same = dataset.price_inputs("test", prices)
    assert np.allclose(same, x, atol=1e-6)
    raised = dataset.price_inputs("test", prices * 1.1)
    assert np.allclose(raised[:, :-1], x[:, :-1])
    assert (raised[:, -1] > x[:, -1]).all()
    lookup = frame.set_index("record_id").loc[
        dataset.frames["test"]["record_id"]
    ].reset_index()
    lookup["price"] = lookup["price"] * 1.1
    applied = dataset.transform(lookup)
    assert np.allclose(
        raised[:, -1], applied["price"].to_numpy(dtype=np.float32), atol=1e-6
    )

    # 1 つの数の候補は全ての行
    single = dataset.price_inputs("test", 90.0)
    assert np.allclose(single[:, -1], single[0, -1])

    # 候補の誤り
    with pytest.raises(ValueError, match="行の数"):
        dataset.price_inputs("test", prices[:3])
    with pytest.raises(ValueError, match="0 より大きい"):
        dataset.price_inputs("test", 0.0)
    with pytest.raises(ValueError, match="0 より大きい"):
        dataset.price_inputs("test", np.full(len(prices), np.inf))


def test_price_layers_in_splits() -> None:
    """
    値段の段で層を作り, どの分割にも全ての値段の段の行が比例して入る
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 分割ごとの値段の段の行の割合
    frame = make_sales()
    prepared = prepare(table_data(frame), price_recipe())
    layers = price_layers(frame["price"].to_numpy())
    lookup = dict(zip(frame["record_id"], layers))
    for name, item in prepared.frames.items():
        counts = np.bincount(
            [lookup[key] for key in item["record_id"]], minlength=5
        )
        share = counts / counts.sum()
        assert np.allclose(share, 0.2, atol=0.02), name


def test_price_layers() -> None:
    """
    値段の段が順位で 5 つに等しく分かれ, 同じ値段が同じ段になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 10 行の順位と, 同じ値段の平均の順位
    layers = price_layers(np.arange(1.0, 11.0))
    assert layers.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    same = price_layers(np.array([5.0, 5.0, 5.0, 1.0]))
    assert same.tolist() == [2, 2, 2, 0]
    assert price_layers(np.zeros(0)).tolist() == []


@pytest.mark.parametrize("count, sizes", [
    (1, {"train": 1}),
    (2, {"train": 1, "valid": 1}),
    (3, {"train": 1, "valid": 1, "test": 1}),
    (5, {"train": 3, "valid": 1, "test": 1}),
])
def test_few_rows(tmp_path: Path, count: int, sizes: dict) -> None:
    """
    行が少ない記録でも書き出して読め, 分割の行の数が割合のとおりで,
    説明が分割と合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    count: int
        行の数
    sizes: dict
        分割ごとの行の数

    Returns
    ----------
    None
    """
    # 少ない行の書き出しと読み直し
    frame = make_sales(count, 2)
    info = export_dataset(table_data(frame), price_recipe(), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    assert {k: len(v) for k, v in dataset.frames.items()} == sizes
    for name, entry in info["price"]["splits"].items():
        assert entry["rows"] == sizes[name]
        x = dataset.price_inputs(name, dataset.price(name))
        assert x.shape == (sizes[name], len(dataset.features))


def test_same_price(tmp_path: Path) -> None:
    """
    全ての行が同じ値段の記録を書き出して読め, 要約で値段の種類と幅の
    気付きが出て, 相関と弾力性が None になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ値段の書き出し (標準化の幅は 1 にそろう)
    frame = make_sales()
    frame["price"] = 100.0
    info = export_dataset(table_data(frame), price_recipe(), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    for entry in info["price"]["splits"].values():
        assert entry["price_min"] == entry["price_max"] == 100.0
        assert entry["prices"] == 1
    x, _ = dataset.arrays("test")
    same = dataset.price_inputs("test", 100.0)
    assert np.allclose(same, x, atol=1e-6)

    # 商品の id を渡さない要約と, 渡した要約
    summary = describe_price(table_data(frame), "price", "units")
    codes = {item["code"] for item in summary["findings"]}
    assert {"few_prices", "narrow_price_range"} <= codes
    assert summary["price_stats"]["width"] == 0.0
    assert summary["relation"]["correlation"] is None
    assert summary["relation"]["elasticity"] is None
    assert len(summary["relation"]["bins"]) == 1
    summary = describe_price(
        table_data(frame), "price", "units", product="product_id"
    )
    codes = {item["code"] for item in summary["findings"]}
    assert {"few_prices", "single_price_products"} <= codes
    assert summary["products"]["single_price"] == summary["products"]["count"]
    assert summary["products"]["width"] is None


def test_narrow_message_by_product() -> None:
    """
    商品の id を渡すと, 値段の幅の気付きの文が商品ごとの幅の中央値を
    指すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 商品ごとに 2 % だけ変えた記録
    frame = make_sales()
    base = frame.groupby("product_id")["price"].transform("median")
    flags = np.arange(len(frame)) % 2 == 0
    frame["price"] = np.where(flags, base, base * 1.02)
    summary = describe_price(
        table_data(frame), "price", "units", product="product_id"
    )
    codes = {item["code"]: item for item in summary["findings"]}
    assert "商品ごと" in codes["narrow_price_range"]["message"]
    plain = describe_price(table_data(frame), "price", "units")
    for item in plain["findings"]:
        assert "商品ごと" not in item["message"]


def test_keep_unlabeled(tmp_path: Path) -> None:
    """
    売れた数の欠けた行を unlabeled の分割に残すと, 元の値段, 原価, 商品の
    id が置かれ, 説明の売れた数の平均が None になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 売れた数の欠けた行
    frame = make_sales()
    frame["units"] = frame["units"].astype(float)
    frame.loc[:9, "units"] = np.nan
    recipe = price_recipe(split={
        "method": "stratified", "valid": 0.2, "test": 0.2, "seed": 0,
        "unlabeled": "keep",
    })
    info = export_dataset(table_data(frame), recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    extra = dataset.frames["unlabeled"]
    assert len(extra) == 10
    assert np.allclose(extra["price_raw"], frame.loc[:9, "price"])
    assert np.allclose(extra["unit_cost"], frame.loc[:9, "unit_cost"])
    assert extra["product_id"].tolist() == frame.loc[:9, "product_id"].tolist()
    assert info["price"]["splits"]["unlabeled"]["units_mean"] is None
    x = dataset.price_inputs("unlabeled", dataset.price("unlabeled"))
    assert x.shape == (10, len(dataset.features))


@pytest.mark.parametrize("change, message", [
    ({"price": 0.0}, "値段の列 price に 0 以下"),
    ({"price": -5.0}, "値段の列 price に 0 以下"),
    ({"price": np.nan}, "値段の列 price に欠け"),
    ({"price": np.inf}, "値段の列 price に 0 以下か無限大"),
    ({"units": -1}, "売れた数の列 units に負"),
    ({"units": np.inf}, "売れた数の列 units に負か無限大"),
    ({"unit_cost": -1.0}, "原価の列 unit_cost に負"),
    ({"unit_cost": np.inf}, "原価の列 unit_cost に負か無限大"),
    ({"unit_cost": np.nan}, "原価の列 unit_cost に欠け"),
    ({"product_id": None}, "商品の id の列 product_id に欠け"),
])
def test_value_errors(change: dict, message: str) -> None:
    """
    値段, 売れた数, 原価, 商品の id の値の誤りで止まることを確かめる関数

    Parameters
    ----------
    change: dict
        1 行目に入れる値
    message: str
        誤りの文

    Returns
    ----------
    None
    """
    # 1 行目の値の誤り (売れた数は無限大を入れられる float)
    frame = make_sales()
    frame["units"] = frame["units"].astype(float)
    for name, value in change.items():
        frame.loc[0, name] = value
    with pytest.raises(ValueError, match=message):
        prepare(table_data(frame), price_recipe())


@pytest.mark.parametrize("name, values, message", [
    ("price", ["a"] * 600, "値段の列 price が数ではありません"),
    ("units", ["a"] * 600, "売れた数の列 units が数ではありません"),
    ("unit_cost", [True] * 600, "原価の列 unit_cost が数ではありません"),
    ("price_raw", [1.0] * 600, "price_raw の列があります"),
])
def test_column_errors(name: str, values: list, message: str) -> None:
    """
    数でない値段, 売れた数, 原価の列と, 元の値段の列と同じ名前の列で
    止まることを確かめる関数

    Parameters
    ----------
    name: str
        書き換える列
    values: list
        列の値
    message: str
        誤りの文

    Returns
    ----------
    None
    """
    # 列の書き換え
    frame = make_sales()
    frame[name] = values
    with pytest.raises(ValueError, match=message):
        prepare(table_data(frame), price_recipe())


@pytest.mark.parametrize("step, message", [
    ({"op": "one_hot", "columns": ["price"]}, "値段の列 price を変え"),
    ({"op": "drop_columns", "columns": ["price"]}, "値段の列 price を変え"),
    ({"op": "cast", "columns": ["price"], "to": "str"},
     "値段の列 price を変え"),
    ({"op": "standardize", "columns": ["size", "unit_cost"]},
     "原価の列 unit_cost は手順で"),
    ({"op": "label_encode", "columns": ["product_id"]},
     "商品の id の列 product_id は手順で"),
])
def test_role_columns_in_steps(step: dict, message: str) -> None:
    """
    値段の列を値段の手順の他で変えるか, 原価と商品の id の列を手順に
    名前で書くと誤りにすることを確かめる関数

    Parameters
    ----------
    step: dict
        役割の列を書いた手順
    message: str
        誤りの文

    Returns
    ----------
    None
    """
    # 名前で書いた役割の列
    recipe = price_recipe()
    recipe["steps"] = [step] + recipe["steps"][1:]
    with pytest.raises(ValueError, match=message):
        prepare(table_data(make_sales()), recipe)


def test_auto_columns() -> None:
    """
    値段の手順の auto は値段の列を選び, 他の手順の auto は選ばず, 原価と
    商品の id の列はどちらも選ばず, 落とす手順は原価を落とせることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # auto の列と落とした原価
    recipe = price_recipe()
    recipe["steps"] = [
        {"op": "drop_columns", "columns": ["unit_cost"]},
        {"op": "cast", "columns": "auto", "to": "float"},
        {"op": "one_hot", "columns": "auto"},
        {"op": "clip", "columns": "auto"},
        {"op": "minmax", "columns": "auto"},
    ]
    prepared = prepare(table_data(make_sales()), recipe)
    steps = prepared.recipe["steps"]
    assert "price" not in steps[1]["state"]["columns"]
    assert "price" in steps[3]["state"]["columns"]
    assert "price" in steps[4]["state"]["columns"]
    for step in steps[1:]:
        assert "unit_cost" not in step["state"].get("columns", [])
        assert "product_id" not in step["state"].get("columns", [])
    assert prepared.price["steps"] == ["clip", "minmax"]
    assert prepared.features[-1] == "price"
    frame = prepared.frames["train"]
    assert "unit_cost" in frame.columns
    assert prepared.price["splits"]["train"]["cost_mean"] is not None


@pytest.mark.parametrize("change, message", [
    ({"price": None}, "値段の列 \\(price\\)"),
    ({"target": None}, "売れた数の列 \\(target\\)"),
    ({"price": "units"}, "違う列"),
    ({"cost": "price"}, "違う列"),
    ({"product": "record_id"}, "違う列"),
    ({"cost": 3}, "列の名前 \\(文字\\)"),
    ({"split": {"method": "random"}}, "stratified だけ"),
    ({"split": {"method": "stratified", "unlabeled": "normal"}},
     "normal は異常の検知"),
])
def test_recipe_errors(change: dict, message: str) -> None:
    """
    値段を変える判断のレシピの誤りを load_recipe で止めることを確かめる
    関数

    Parameters
    ----------
    change: dict
        上書きするレシピの項目
    message: str
        誤りの文

    Returns
    ----------
    None
    """
    # 読んだ時点の誤り
    with pytest.raises(ValueError, match=message):
        load_recipe(price_recipe(**change))


@pytest.mark.parametrize("key", ["price", "cost", "product"])
def test_price_keys_other_task(key: str) -> None:
    """
    値段を変える判断だけの鍵を他のタスクに書くと誤りにすることを確かめる
    関数

    Parameters
    ----------
    key: str
        値段を変える判断だけの鍵

    Returns
    ----------
    None
    """
    # 回帰のレシピ
    recipe = {"task": "regression", "target": "units", key: "price"}
    with pytest.raises(ValueError, match="task: price"):
        load_recipe(recipe)


def test_dataset_without_price(tmp_path: Path) -> None:
    """
    値段を変える判断でないデータセットの値段と候補の入力は誤りにする
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 回帰のデータセット
    frame = make_sales().drop(columns=["record_id", "product_id", "group"])
    recipe = {"task": "regression", "target": "units",
              "split": {"method": "random", "valid": 0.2, "seed": 0}}
    export_dataset(table_data(frame), recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    for call in (dataset.price, dataset.cost):
        with pytest.raises(ValueError, match="task: price"):
            call("train")
    with pytest.raises(ValueError, match="task: price"):
        dataset.price_inputs("train", 1.0)


def test_describe_clean() -> None:
    """
    値段を無作為に変えた記録の要約で, 値段と売れた数の関わりと商品ごとの
    要約が返り, 誤りの気付きが出ないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 要約の数
    frame = make_sales()
    summary = describe_price(
        table_data(frame), "price", "units", cost="unit_cost",
        product="product_id", id="record_id"
    )
    assert summary["rows"] == 600
    assert summary["missing"] == {
        "price": 0, "units": 0, "cost": 0, "product": 0
    }
    assert summary["invalid"] == {"price": 0, "units": 0, "cost": 0}
    assert summary["price_stats"]["min"] == pytest.approx(
        frame["price"].min()
    )
    assert summary["price_stats"]["width"] == pytest.approx(0.4)
    assert summary["unit_stats"]["mean"] == pytest.approx(
        frame["units"].mean()
    )
    assert summary["relation"]["correlation"] < 0
    assert summary["relation"]["elasticity"] == pytest.approx(-1.5, abs=0.2)
    bins = summary["relation"]["bins"]
    assert [item["bin"] for item in bins] == [0, 1, 2, 3, 4]
    assert sum(item["rows"] for item in bins) == 600
    assert bins[0]["units_mean"] > bins[-1]["units_mean"]

    # 商品ごとの要約と, 原価の要約
    products = summary["products"]
    assert products["count"] == frame["product_id"].nunique()
    assert products["single_price"] == 0
    assert len(products["rows"]) == products["count"]
    counts = [item["prices"] for item in products["rows"]]
    assert counts == sorted(counts)
    costs = summary["cost_stats"]
    assert costs["rows"] == 600
    assert costs["margin"] == pytest.approx(
        ((frame["price"] - frame["unit_cost"]) / frame["price"]).mean()
    )
    assert costs["below"] == 0

    # 商品ごとの値段に対する関わり (商品の特徴は関わらない)
    columns = [item["column"] for item in summary["confounding"]]
    assert set(columns) == {"size", "group"}
    assert all(item["strength"] < 0.3 for item in summary["confounding"])
    assert summary["findings"] == []


def test_describe_findings(tmp_path: Path) -> None:
    """
    値段の誤り, 試した値段が 1 つの商品, 売れた数の欠け, 原価より低い値段,
    値段と関わる列の気付きが出ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 値段の誤り (0 と負と欠け), 売れた数の欠け, 原価より低い値段
    frame = make_sales()
    frame["units"] = frame["units"].astype(float)
    frame.loc[0, "price"] = 0.0
    frame.loc[1, "price"] = -3.0
    frame.loc[2, "price"] = np.nan
    frame.loc[3, "units"] = np.nan
    frame.loc[4, "unit_cost"] = frame.loc[4, "price"] * 2

    # 1 つの値段しか試していない商品と, 値段と関わる列
    single = frame["product_id"] == "p000"
    frame.loc[single, "price"] = 120.0
    rng = np.random.default_rng(1)
    frame["promo"] = frame["price"] * 0.01 + rng.normal(0, 0.01, len(frame))
    file = write_table(frame, tmp_path)
    summary = describe_price(
        file, "price", "units", cost="unit_cost", product="product_id"
    )
    codes = {item["code"]: item for item in summary["findings"]}
    assert summary["missing"]["price"] == 1
    assert summary["missing"]["units"] == 1
    assert summary["invalid"]["price"] == 2
    assert summary["rows"] == 600 - 4
    assert "price_missing" in codes
    assert "price_invalid" in codes
    assert codes["units_missing"]["severity"] == "info"
    assert codes["price_below_cost"]["severity"] == "info"
    assert "p000" in codes["single_price_products"]["message"]
    assert summary["products"]["single_price"] == 1
    assert summary["products"]["rows"][0]["product"] == "p000"
    assert "promo" in codes["price_confounded"]["columns"]
    severities = [item["severity"] for item in summary["findings"]]
    assert severities == sorted(severities, key=lambda s: s != "warning")


def test_describe_narrow_and_few() -> None:
    """
    値段の種類が少ない記録と, 値段の幅の狭い記録の気付きが出ることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 つの値段だけで, 幅が 4 % ほどの記録 (商品の id を渡さない)
    rng = np.random.default_rng(0)
    price = rng.choice([100.0, 104.0], 200)
    frame = pd.DataFrame({
        "price": price, "units": rng.poisson(10, 200),
        "size": rng.normal(0, 1, 200),
    })
    summary = describe_price(table_data(frame), "price", "units")
    codes = {item["code"] for item in summary["findings"]}
    assert {"few_prices", "narrow_price_range"} <= codes
    assert summary["price_stats"]["distinct"] == 2
    assert summary["price_stats"]["width"] == pytest.approx(
        4.0 / np.median(price)
    )
    assert summary["products"] is None
    assert summary["cost_stats"] is None
    assert summary["missing"]["cost"] is None
    assert summary["invalid"]["cost"] is None


def test_describe_errors() -> None:
    """
    要約の列の誤りで止まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 無い列, 同じ列, 数でない列, 除く列を関わりの列に渡す, 読み方の重ね
    data = table_data(make_sales())
    with pytest.raises(ValueError, match="値段の列 none"):
        describe_price(data, "none", "units")
    with pytest.raises(ValueError, match="違う列"):
        describe_price(data, "price", "price")
    with pytest.raises(ValueError, match="数ではありません"):
        describe_price(data, "group", "units")
    with pytest.raises(ValueError, match="渡せません"):
        describe_price(data, "price", "units", columns=["units"])
    with pytest.raises(ValueError, match="読み込みの名前と設定"):
        describe_price(data, "price", "units", loader="csv")
