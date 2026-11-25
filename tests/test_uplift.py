from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    describe_uplift, export_dataset, load_data, load_dataset, load_recipe,
    model_inputs, prepare
)
from ml_eda.data import Data


def make_coupons(count: int = 400, treated_rate: float = 0.5) -> pd.DataFrame:
    """
    クーポンを無作為に配った記録の表を作る関数

    Parameters
    ----------
    count: int = 400
        行の数
    treated_rate: float = 0.5
        配った割合

    Returns
    ----------
    frame: pd.DataFrame
        顧客の id, 特徴 (年齢, 前の月の額, 地域), 配ったか, 買ったか,
        買った額の表
    """
    # 乱数を固定した特徴と, 無作為の処置
    rng = np.random.default_rng(0)
    age = rng.normal(40.0, 10.0, count)
    spend = rng.gamma(2.0, 50.0, count)
    region = rng.choice(["east", "west", "north"], count)
    coupon = rng.random(count) < treated_rate

    # 若い顧客ほど配ると買いやすい結果と, 買った額
    base = 0.2 + 0.1 * (spend > 100)
    lift = np.where(age < 40, 0.3, 0.05)
    bought = rng.random(count) < base + lift * coupon
    amount = np.where(bought, rng.gamma(3.0, 20.0, count), 0.0)
    frame = pd.DataFrame({
        "customer_id": [f"c{index:05d}" for index in range(count)],
        "age": age,
        "spend": spend,
        "region": region,
        "coupon": coupon.astype(int),
        "bought": bought.astype(int),
        "amount": amount,
    })

    return frame


def uplift_recipe(target: str = "bought", **changes: object) -> dict:
    """
    効果を見積もるのレシピを作る関数

    Parameters
    ----------
    target: str = "bought"
        結果の列
    **changes: object
        上書きするレシピの項目

    Returns
    ----------
    recipe: dict
        数の列を標準化し, 地域を符号化するレシピ
    """
    # 層化の分割 (処置で層を作る)
    recipe = {
        "format_version": 1,
        "task": "uplift",
        "target": target,
        "treatment": "coupon",
        "id": "customer_id",
        "steps": [
            {"op": "drop_columns",
             "columns": ["amount" if target == "bought" else "bought"]},
            {"op": "one_hot", "columns": ["region"]},
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
    file = path / "coupons.csv"
    frame.to_csv(file, index=False)

    return file


def check_groups(dataset: object, info: dict, target: str) -> None:
    """
    全ての分割に処置の群と対照の群があり, 説明の数と平均が分割の表と
    合うことを確かめる関数

    Parameters
    ----------
    dataset: object
        読み直したデータセット
    info: dict
        書き出しの説明
    target: str
        結果の列

    Returns
    ----------
    None
    """
    # 分割ごとの群の行の数と結果の平均
    for name, frame in dataset.frames.items():
        treated = frame["coupon"] == 1
        entry = info["uplift"]["splits"][name]
        assert entry["treated"] == int(treated.sum()) > 0
        assert entry["control"] == int((~treated).sum()) > 0
        assert entry["treated_mean"] == pytest.approx(
            frame.loc[treated, target].mean()
        )
        assert entry["control_mean"] == pytest.approx(
            frame.loc[~treated, target].mean()
        )
        assert entry["effect"] == pytest.approx(
            entry["treated_mean"] - entry["control_mean"]
        )


def test_binary_outcome(tmp_path: Path) -> None:
    """
    買ったかの結果の表を書き出して読み, 全ての分割に両方の群が入り,
    行の id と処置が残り, 説明の数と平均が合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出しと読み直し
    frame = make_coupons()
    data = load_data(write_table(frame, tmp_path))
    info = export_dataset(data, uplift_recipe(), tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    assert set(dataset.frames) == {"train", "valid", "test"}
    assert sum(len(item) for item in dataset.frames.values()) == 400
    check_groups(dataset, info, "bought")

    # 説明の uplift と, 結果の種類を値から決めたこと
    summary = info["uplift"]
    assert summary["id"] == "customer_id"
    assert summary["treatment"] == "coupon"
    assert summary["outcome"] == "binary"
    assert summary["unlabeled"] == "drop"
    assert info["task"] == "uplift"
    assert info["classes"] == []

    # 行の id と処置は元の値で, 学習に使う列から除く (auto の標準化も)
    merged = pd.concat(dataset.frames.values())
    assert sorted(merged["customer_id"]) == sorted(frame["customer_id"])
    lookup = frame.set_index("customer_id")["coupon"]
    assert (
        merged["coupon"].to_numpy()
        == lookup[merged["customer_id"]].to_numpy()
    ).all()
    assert "coupon" not in dataset.features
    assert "customer_id" not in dataset.features
    assert "bought" not in dataset.features
    assert "age" in dataset.features
    standardize = dataset.recipe["steps"][2]["state"]["columns"]
    assert "coupon" not in standardize

    # 処置の割合を保つ分割と, 正解と処置の型
    rates = [
        float((item["coupon"] == 1).mean())
        for item in dataset.frames.values()
    ]
    assert max(rates) - min(rates) < 0.05
    x, y = dataset.arrays("valid")
    assert y.dtype == np.int64
    assert x.shape[1] == len(dataset.features)
    treatment = dataset.treatment("valid")
    assert treatment.dtype == np.int64
    assert set(treatment.tolist()) == {0, 1}
    assert len(treatment) == len(y)

    # モデルの入力 (処置の列を除く幅) と出す値の数
    inputs = model_inputs(tmp_path / "ds")
    assert inputs["task"] == "uplift"
    assert inputs["target"] == "bought"
    assert inputs["classes"] is None
    assert inputs["outputs"] == 1
    assert inputs["inputs"][0]["shape"] == [None, len(dataset.features)]

    # 推論のときは処置と結果の列が無くても前処理をかけられる
    fresh = frame.drop(columns=["coupon", "bought"]).head(5)
    applied = dataset.transform(fresh)
    assert list(applied[dataset.features].columns) == dataset.features


def test_amount_outcome(tmp_path: Path) -> None:
    """
    買った額の結果の表で, 結果の種類が amount になり, 正解が float32 で
    返ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出しと読み直し
    frame = make_coupons()
    info = export_dataset(
        table_data(frame), uplift_recipe("amount"), tmp_path / "ds"
    )
    dataset = load_dataset(tmp_path / "ds")
    assert info["uplift"]["outcome"] == "amount"
    check_groups(dataset, info, "amount")
    _, y = dataset.arrays("train")
    assert y.dtype == np.float32

    # 書いた種類は値から決めた種類より先
    prepared = prepare(
        table_data(frame), uplift_recipe("bought", outcome="amount")
    )
    assert prepared.uplift["outcome"] == "amount"


def test_small_group_in_all_splits() -> None:
    """
    処置の群が少ない表でも, 全ての分割に処置の群が入り, 分割が種で
    決まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 200 行のうち処置の群は 3 行
    frame = make_coupons(200, 0.0)
    frame.loc[[5, 50, 150], "coupon"] = 1
    prepared = prepare(table_data(frame), uplift_recipe())
    for name, split in prepared.frames.items():
        assert int((split["coupon"] == 1).sum()) == 1, name
        assert int((split["coupon"] == 0).sum()) > 0, name
    again = prepare(table_data(frame), uplift_recipe())
    for name in prepared.frames:
        assert list(prepared.frames[name]["customer_id"]) == list(
            again.frames[name]["customer_id"]
        )

    # 置く分割より群の行が少なければ誤り
    frame.loc[[150], "coupon"] = 0
    with pytest.raises(ValueError, match="処置の群の行が 2 行"):
        prepare(table_data(frame), uplift_recipe())

    # 割合が 0 の分割には置かない
    recipe = uplift_recipe(split={
        "method": "stratified", "valid": 0.2, "test": 0.0, "seed": 0
    })
    prepared = prepare(table_data(frame), recipe)
    assert set(prepared.frames) == {"train", "valid"}


def test_keep_and_filtered_treatment(tmp_path: Path) -> None:
    """
    結果の欠けた行を unlabeled の分割に残し, 処置の欠けた行を分割の前の
    手順で落とせ, 処置の列を手順で落としても元の値が残ることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 結果の欠けた 20 行と, 処置の欠けた 5 行
    frame = make_coupons()
    frame["bought"] = frame["bought"].astype(float)
    frame.loc[:19, "bought"] = np.nan
    frame["coupon"] = frame["coupon"].astype(float)
    frame.loc[100:104, "coupon"] = np.nan
    recipe = uplift_recipe(
        split={"method": "stratified", "valid": 0.2, "test": 0.2,
               "seed": 0, "unlabeled": "keep"}
    )
    recipe["steps"] = [
        {"op": "filter_rows", "column": "coupon", "compare": "notna"},
        {"op": "drop_columns", "columns": ["amount", "coupon"]},
    ] + recipe["steps"][1:]
    info = export_dataset(table_data(frame), recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")

    # unlabeled の分割の処置は元の値で, 平均は None
    assert len(dataset.frames["unlabeled"]) == 20
    assert set(dataset.frames["unlabeled"]["coupon"]) <= {0, 1}
    entry = info["uplift"]["splits"]["unlabeled"]
    assert entry["treated"] + entry["control"] == 20
    assert entry["treated_mean"] is None and entry["effect"] is None
    labeled = sum(
        len(item) for name, item in dataset.frames.items()
        if name != "unlabeled"
    )
    assert labeled == 375
    assert dataset.frames["train"]["coupon"].dtype == np.int64
    assert info["uplift"]["unlabeled"] == "keep"

    # 処置の欠けた行を落とさなければ誤り
    recipe["steps"] = recipe["steps"][1:]
    with pytest.raises(ValueError, match="処置の列 coupon に欠け"):
        prepare(table_data(frame), recipe)


@pytest.mark.parametrize("change, message", [
    ({"treatment": None}, "処置の列 \\(treatment\\)"),
    ({"target": None}, "結果の列 \\(target\\)"),
    ({"outcome": "count"}, "binary か amount"),
    ({"treatment": "bought"}, "違う列"),
    ({"treatment": "customer_id"}, "違う列"),
    ({"split": {"method": "random"}}, "stratified だけ"),
    ({"split": {"method": "stratified", "unlabeled": "normal"}},
     "normal は異常の検知"),
])
def test_recipe_errors(change: dict, message: str) -> None:
    """
    効果を見積もるのレシピの誤りを, 読んだ時点で止めることを確かめる
    関数

    Parameters
    ----------
    change: dict
        誤りにするレシピの項目
    message: str
        誤りの文に含まれる言葉

    Returns
    ----------
    None
    """
    # 項目の上書き (None は項目を消す)
    recipe = uplift_recipe()
    for key, value in change.items():
        if value is None:
            recipe.pop(key)
        else:
            recipe[key] = value
    with pytest.raises(ValueError, match=message):
        load_recipe(recipe)


@pytest.mark.parametrize("key", ["treatment", "outcome"])
def test_uplift_keys_other_task(key: str) -> None:
    """
    処置の列と結果の種類を, 効果を見積もるの他のタスクで書くと誤りに
    することを確かめる関数

    Parameters
    ----------
    key: str
        書く項目

    Returns
    ----------
    None
    """
    # 分類のレシピに効果を見積もるだけの項目
    recipe = {"task": "classification", "target": "bought", key: "coupon"}
    with pytest.raises(ValueError, match="task: uplift"):
        load_recipe(recipe)
    with pytest.raises(ValueError, match="id"):
        load_recipe({"task": "regression", "id": "customer_id"})


@pytest.mark.parametrize("column, values, change, message", [
    ("coupon", 2, {}, "処置の列 coupon は"),
    ("coupon", "yes", {}, "処置の列 coupon は"),
    ("bought", 2, {"outcome": "binary"}, "binary では"),
    ("bought", "many", {}, "数ではありません"),
    ("missing", 0, {"treatment": "missing"}, "処置の列 missing がありません"),
])
def test_data_errors(
    column: str, values: object, change: dict, message: str
) -> None:
    """
    処置と結果の値が読み方に合わない表を, 理由付きの誤りにすることを
    確かめる関数

    Parameters
    ----------
    column: str
        書き換える列
    values: object
        先頭の行に入れる値
    change: dict
        上書きするレシピの項目
    message: str
        誤りの文に含まれる言葉

    Returns
    ----------
    None
    """
    # 先頭の行の値を書き換えた表
    frame = make_coupons()
    if column in frame.columns:
        frame[column] = frame[column].astype(object)
        frame.loc[0, column] = values
    with pytest.raises(ValueError, match=message):
        prepare(table_data(frame), uplift_recipe(**change))


def test_describe_balanced() -> None:
    """
    無作為に配った記録の要約で, 群の数, 結果の差, 釣り合いが返り, 気付きが
    出ないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 群の数と結果の平均と差
    frame = make_coupons(2000).drop(columns=["amount"])
    summary = describe_uplift(
        table_data(frame), "coupon", "bought", id="customer_id"
    )
    treated = frame["coupon"] == 1
    assert summary["outcome_kind"] == "binary"
    assert summary["rows"] == 2000
    assert summary["groups"]["treated"]["rows"] == int(treated.sum())
    assert summary["groups"]["control"]["rows"] == int((~treated).sum())
    assert summary["treated_rate"] == pytest.approx(treated.mean())
    means = frame.groupby("coupon")["bought"].mean()
    assert summary["effect"] == pytest.approx(means[1] - means[0])
    assert summary["effect"] > 0.05

    # 釣り合いの列 (処置, 結果, 行の id を除く) と SMD
    columns = {item["column"]: item for item in summary["balance"]}
    assert set(columns) == {"age", "spend", "region"}
    age = frame["age"]
    expected = (age[treated].mean() - age[~treated].mean()) / np.sqrt(
        (age[treated].var() + age[~treated].var()) / 2
    )
    assert columns["age"]["smd"] == pytest.approx(expected)
    assert columns["region"]["value"] in {"east", "west", "north"}
    smds = [abs(item["smd"]) for item in summary["balance"]]
    assert smds == sorted(smds, reverse=True)

    # 気付きは無し
    assert summary["findings"] == []


def test_describe_findings(tmp_path: Path) -> None:
    """
    少ない群, 崩れた釣り合い, 処置の後に決まる列, 欠けた行を気付きに
    することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 年齢の高い顧客に偏って配り, 使ったかの列を足した記録
    frame = make_coupons(400)
    frame["coupon"] = (frame["age"] > 45).astype(int)
    frame.loc[frame.index[:3], "coupon"] = 1
    frame["used"] = np.where(
        frame["coupon"] == 1, frame["bought"], 0
    )
    frame["copied"] = frame["bought"] * 2
    frame["coupon"] = frame["coupon"].astype(float)
    frame.loc[10:11, "coupon"] = np.nan
    frame["bought"] = frame["bought"].astype(float)
    frame.loc[20:22, "bought"] = np.nan
    summary = describe_uplift(
        write_table(frame, tmp_path), "coupon", "bought",
        id="customer_id"
    )
    found = {}
    for item in summary["findings"]:
        found.setdefault(item["code"], []).append(item)

    # 処置の後に決まる列 (片方の群でだけ値が変わる, 結果の写し, 処置の
    # 写しの年齢は崩れた釣り合い)
    leaked = {item["columns"][0] for item in found["post_treatment"]}
    assert {"used", "copied"} <= leaked
    assert "age" in found["treatment_imbalance"][0]["columns"]
    assert "used" not in found["treatment_imbalance"][0]["columns"]

    # 欠けた行の数
    assert summary["missing"] == {"treatment": 2, "outcome": 3}
    assert found["treatment_missing"][0]["severity"] == "warning"
    assert found["outcome_missing"][0]["severity"] == "info"
    assert summary["rows"] == 395
    severities = [item["severity"] for item in summary["findings"]]
    assert severities == sorted(severities, key=lambda s: s != "warning")

    # 少ない群
    small = make_coupons(400, 0.03)
    summary = describe_uplift(table_data(small), "coupon", "amount")
    assert summary["outcome_kind"] == "amount"
    assert "small_group" in {item["code"] for item in summary["findings"]}


def test_describe_errors() -> None:
    """
    要約の列と値の誤りを理由付きで止めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 無い列, 同じ列, 除く列, 読み込みの設定, 処置の値
    data = table_data(make_coupons())
    with pytest.raises(ValueError, match="処置の列 none"):
        describe_uplift(data, "none", "bought")
    with pytest.raises(ValueError, match="違う列"):
        describe_uplift(data, "coupon", "coupon")
    with pytest.raises(ValueError, match="渡せません"):
        describe_uplift(data, "coupon", "bought", columns=["coupon"])
    with pytest.raises(ValueError, match="読み込みの名前と設定"):
        describe_uplift(data, "coupon", "bought", loader="csv")
    with pytest.raises(ValueError, match="binary では"):
        describe_uplift(data, "coupon", "amount", outcome_kind="binary")
    summary = describe_uplift(data, "coupon", "bought", columns=["age"])
    assert [item["column"] for item in summary["balance"]] == ["age"]


def test_groups_after_steps() -> None:
    """
    分割の後の手順で群の行が無くなる分割を, 理由付きの誤りにすることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 標準化の後に対照の群だけを残す手順
    recipe = uplift_recipe()
    recipe["steps"] = recipe["steps"] + [
        {"op": "filter_rows", "column": "coupon", "compare": "==",
         "value": 0},
    ]
    with pytest.raises(ValueError, match="処置の群の行がありません"):
        prepare(table_data(make_coupons()), recipe)
