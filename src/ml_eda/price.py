from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.profile import association, finding, infer_type, to_json
from ml_eda.uplift import column_of, uplift_table

# 値段を変えた記録の表として読む種類
PRICE_KINDS = ("table", "timeseries")

# 書き出しの分割に置く, 手順をかける前の値段の列
RAW_PRICE = "price_raw"

# 値段の段の数 (分割の層と, 値段と売れた数の関わりの要約)
PRICE_LAYERS = 5

# 値段の種類の下限 (これ未満は曲がり方を学べない)
MIN_PRICES = 3

# 狭すぎると見なす値段の幅 ((最大 − 最小) / 中央値)
NARROW_WIDTH = 0.1

# 値段と強く関わると見なす関わりの強さ
CONFOUNDED = 0.3

# 関わりを求める分析の型
CONFOUNDING_TYPES = ("numeric", "boolean", "categorical")

# 要約に並べる商品の数の上限
PRODUCT_ROWS = 50

# 気付きの文に挙げる列と商品の数
FINDING_EXAMPLES = 5


def numeric_values(values: pd.Series, name: str, label: str) -> pd.Series:
    """
    数 (真偽値を除く) の列であることを確かめ, float64 にする関数

    Parameters
    ----------
    values: pd.Series
        列の値
    name: str
        列の名前 (誤りの文に使う)
    label: str
        列の呼び名 (誤りの文に使う)

    Returns
    ----------
    numbers: pd.Series
        float64 の値 (欠けは nan)

    Raises
    ----------
    ValueError
        数でない列と, 真偽値の列の場合
    """
    # 真偽値と数でない列は誤り
    numeric = pd.api.types.is_numeric_dtype(values) and not (
        pd.api.types.is_bool_dtype(values)
    )
    if not numeric:
        examples = values.dropna().astype(str).unique().tolist()[:3]
        raise ValueError(
            f"{label} {name} が数ではありません ({values.dtype}. 例: "
            f"{', '.join(examples)})"
        )
    numbers = values.astype("float64")

    return numbers


def price_values(values: pd.Series, name: str) -> pd.Series:
    """
    値段の列の値を確かめ, float64 にする関数

    Parameters
    ----------
    values: pd.Series
        値段の列の値 (全ての行)
    name: str
        値段の列の名前

    Returns
    ----------
    prices: pd.Series
        float64 の値段

    Raises
    ----------
    ValueError
        数でない場合, 欠けがある場合, 0 以下か無限大の値がある場合
    """
    # 欠けと, 0 以下か無限大の値
    prices = numeric_values(values, name, "値段の列")
    missing = int(prices.isna().sum())
    if missing:
        raise ValueError(
            f"値段の列 {name} に欠けがあります ({missing} 行). 読み込む前の"
            "表から落としてください"
        )
    wrong = ~np.isfinite(prices.to_numpy()) | (prices.to_numpy() <= 0)
    if wrong.any():
        examples = prices[wrong].astype(str).unique().tolist()[:3]
        raise ValueError(
            f"値段の列 {name} に 0 以下か無限大の値があります "
            f"({int(wrong.sum())} 行. 例: {', '.join(examples)}). 読み込む"
            "前の表で直してください"
        )

    return prices


def cost_values(values: pd.Series, name: str) -> pd.Series:
    """
    原価の列の値を確かめ, float64 にする関数

    Parameters
    ----------
    values: pd.Series
        原価の列の値 (全ての行)
    name: str
        原価の列の名前

    Returns
    ----------
    costs: pd.Series
        float64 の原価

    Raises
    ----------
    ValueError
        数でない場合, 欠けがある場合, 負か無限大の値がある場合
    """
    # 欠けと, 負か無限大の値
    costs = numeric_values(values, name, "原価の列")
    missing = int(costs.isna().sum())
    if missing:
        raise ValueError(
            f"原価の列 {name} に欠けがあります ({missing} 行). 読み込む前の"
            "表で埋めるか落としてください"
        )
    wrong = ~np.isfinite(costs.to_numpy()) | (costs.to_numpy() < 0)
    if wrong.any():
        examples = costs[wrong].astype(str).unique().tolist()[:3]
        raise ValueError(
            f"原価の列 {name} に負か無限大の値があります ({int(wrong.sum())}"
            f" 行. 例: {', '.join(examples)}). 読み込む前の表で直して"
            "ください"
        )

    return costs


def units_values(values: pd.Series, name: str) -> pd.Series:
    """
    売れた数の列の欠けの無い値を確かめる関数

    Parameters
    ----------
    values: pd.Series
        売れた数の列の欠けの無い値
    name: str
        売れた数の列の名前

    Returns
    ----------
    units: pd.Series
        数のままの値 (元の型)

    Raises
    ----------
    ValueError
        数でない場合と, 負か無限大の値がある場合
    """
    # 負か無限大の値
    numbers = numeric_values(values, name, "売れた数の列").to_numpy()
    wrong = ~np.isfinite(numbers) | (numbers < 0)
    if wrong.any():
        examples = values[wrong].astype(str).unique().tolist()[:3]
        raise ValueError(
            f"売れた数の列 {name} に負か無限大の値があります "
            f"({int(wrong.sum())} 行. 例: {', '.join(examples)}). 読み込む"
            "前の表で直してください"
        )
    units = values

    return units


def product_values(values: pd.Series, name: str) -> pd.Series:
    """
    商品の id の列に欠けが無いことを確かめる関数

    Parameters
    ----------
    values: pd.Series
        商品の id の列の値 (全ての行)
    name: str
        商品の id の列の名前

    Returns
    ----------
    products: pd.Series
        元の値

    Raises
    ----------
    ValueError
        欠けがある場合
    """
    # 欠け (同じ値は何行あってもよい)
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(
            f"商品の id の列 {name} に欠けがあります ({missing} 行). 読み込む"
            "前の表で埋めるか落としてください"
        )
    products = values

    return products


def price_layers(prices: np.ndarray) -> np.ndarray:
    """
    行を値段の順位で 5 つに等しく分けた値段の段を返す関数

    Parameters
    ----------
    prices: np.ndarray
        行ごとの値段

    Returns
    ----------
    layers: np.ndarray
        行ごとの段 (0 から 4 の int64. 同じ値段は平均の順位で同じ段)
    """
    # 平均の順位 r (1 から) の段は min(4, ⌊(r − 1) × 5 / 行の数⌋)
    count = len(prices)
    if not count:
        layers = np.zeros(0, dtype=np.int64)
        return layers
    ranks = pd.Series(prices).rank(method="average").to_numpy()
    layers = np.minimum(
        PRICE_LAYERS - 1, np.floor((ranks - 1) * PRICE_LAYERS / count)
    ).astype(np.int64)

    return layers


def width_of(prices: np.ndarray) -> Optional[float]:
    """
    値段の幅 ((最大 − 最小) / 中央値) を求める関数

    Parameters
    ----------
    prices: np.ndarray
        0 より大きい値段

    Returns
    ----------
    width: Optional[float]
        値段の幅 (値段が無ければ None)
    """
    # 中央値に対する範囲
    if not len(prices):
        return None
    width = float((prices.max() - prices.min()) / np.median(prices))

    return width


def elasticity_of(
    prices: np.ndarray, units: np.ndarray, products: Optional[np.ndarray]
) -> Optional[float]:
    """
    log(売れた数) を log(値段) に当てた直線の傾きを求める関数

    Parameters
    ----------
    prices: np.ndarray
        値段
    units: np.ndarray
        売れた数
    products: Optional[np.ndarray]
        行ごとの商品の id (渡すと商品ごとに平均を引く)

    Returns
    ----------
    slope: Optional[float]
        傾き. 売れた数が 0 より大きい行が 3 つ未満か, 値段が変わらない
        場合は None
    """
    # 売れた数が 0 より大きい行の対数
    keep = units > 0
    if int(keep.sum()) < 3:
        return None
    x = pd.Series(np.log(prices[keep]))
    y = pd.Series(np.log(units[keep]))

    # 平均の差し引き (商品ごとか全体)
    if products is None:
        x = x - x.mean()
        y = y - y.mean()
    else:
        keys = pd.Series(products[keep]).astype(str)
        x = x - x.groupby(keys).transform("mean")
        y = y - y.groupby(keys).transform("mean")
    denominator = float((x * x).sum())
    if denominator <= 1e-12:
        return None
    slope = float((x * y).sum() / denominator)

    return slope


def relation_bins(prices: np.ndarray, units: np.ndarray) -> list[dict]:
    """
    値段の段ごとの行の数, 値段の範囲と平均, 売れた数の平均を求める関数

    Parameters
    ----------
    prices: np.ndarray
        値段
    units: np.ndarray
        売れた数

    Returns
    ----------
    bins: list[dict]
        段ごとの bin, rows, price_min, price_max, price_mean, units_mean
        (行の無い段は除く)
    """
    # 段ごとの集計
    layers = price_layers(prices)
    bins = []
    for layer in range(PRICE_LAYERS):
        flags = layers == layer
        if not flags.any():
            continue
        bins.append({
            "bin": layer,
            "rows": int(flags.sum()),
            "price_min": float(prices[flags].min()),
            "price_max": float(prices[flags].max()),
            "price_mean": float(prices[flags].mean()),
            "units_mean": float(units[flags].mean()),
        })

    return bins


def product_summary(
    prices: np.ndarray, units: np.ndarray, products: np.ndarray
) -> dict:
    """
    商品ごとの試した値段の数と範囲, 売れた数の平均を求める関数

    Parameters
    ----------
    prices: np.ndarray
        値段
    units: np.ndarray
        売れた数
    products: np.ndarray
        行ごとの商品の id

    Returns
    ----------
    summary: dict
        count, single_price, width, rows (試した値段の数の少ない順で
        最大 50 件), singles (試した値段が 1 つの商品の文字の並び)
    """
    # 商品ごとの集計 (商品の id は文字)
    frame = pd.DataFrame({
        "product": pd.Series(products).astype(str),
        "price": prices, "units": units,
    })
    grouped = frame.groupby("product", sort=True)
    table = pd.DataFrame({
        "rows": grouped.size(),
        "prices": grouped["price"].nunique(),
        "price_min": grouped["price"].min(),
        "price_max": grouped["price"].max(),
        "units_mean": grouped["units"].mean(),
    }).reset_index()
    table = table.sort_values(["prices", "product"], kind="stable")

    # 試した値段が 2 つ以上の商品ごとの幅の中央値
    widths = [
        width_of(group.to_numpy())
        for _, group in grouped["price"] if group.nunique() >= 2
    ]
    singles = table.loc[table["prices"] == 1, "product"].tolist()
    summary = {
        "count": int(len(table)),
        "single_price": int(len(singles)),
        "width": float(np.median(widths)) if widths else None,
        "rows": table.head(PRODUCT_ROWS).to_dict(orient="records"),
        "singles": singles,
    }

    return summary


def confounding_columns(
    frame: pd.DataFrame, columns: Any, reserved: list[str], roles: dict
) -> list[tuple[str, str]]:
    """
    値段との関わりを求める列と, その分析の型を決める関数

    Parameters
    ----------
    frame: pd.DataFrame
        列の名前を文字にそろえた表
    columns: Any
        列の名前の並び, None の場合は数, 真偽, カテゴリの型の全ての列
    reserved: list[str]
        除く列 (値段, 売れた数, 原価, 商品の id, 行の id)
    roles: dict
        列の役割 (型の推定に使う)

    Returns
    ----------
    chosen: list[tuple[str, str]]
        (列の名前, 分析の型) の並び

    Raises
    ----------
    ValueError
        渡した列が無いか, 除く列を渡した場合
    """
    # 省いた場合の, 除く列の他の数, 真偽, カテゴリの列
    if columns is None:
        chosen = []
        for name in frame.columns:
            kind = infer_type(frame[name], roles)
            if name not in reserved and kind in CONFOUNDING_TYPES:
                chosen.append((name, kind))
        return chosen

    # 渡した列の確かめ
    names = [columns] if isinstance(columns, str) else list(columns)
    chosen = []
    for name in names:
        column = column_of(frame, name, "関わりの列")
        if str(name) in reserved:
            raise ValueError(
                "関わりの列に値段, 売れた数, 原価, 商品の id, 行の id の列は"
                f"渡せません: {name}"
            )
        chosen.append((str(name), infer_type(column, roles)))

    return chosen


def listed(names: list[str]) -> str:
    """
    気付きの文に挙げる名前を, 上限まで並べた文字にする関数

    Parameters
    ----------
    names: list[str]
        名前の並び

    Returns
    ----------
    text: str
        上限までの名前と, 残りの数
    """
    # 上限を超えた分は数だけ
    text = ", ".join(names[:FINDING_EXAMPLES])
    if len(names) > FINDING_EXAMPLES:
        text += f" ほか {len(names) - FINDING_EXAMPLES} 件"

    return text


def value_findings(
    missing: dict, invalid: dict, names: dict
) -> list[dict]:
    """
    値段, 売れた数, 原価, 商品の id の欠けと値の誤りを気付きにする関数

    Parameters
    ----------
    missing: dict
        欠けた行の数 (price, units, cost, product)
    invalid: dict
        値の誤りの行の数 (price, units, cost)
    names: dict
        列の名前 (price, units, cost, product)

    Returns
    ----------
    findings: list[dict]
        price_missing, price_invalid, units_invalid, cost_invalid,
        product_missing, units_missing の気付き
    """
    # 書き出しでは誤りになるもの
    fix = "書き出しでは誤りになるので, 読み込む前の表で直してください"
    findings = []
    checks = [
        ("price_missing", missing["price"], "price", "値段の列", "欠けた行"),
        ("price_invalid", invalid["price"], "price", "値段の列",
         "0 以下か無限大の行"),
        ("units_invalid", invalid["units"], "units", "売れた数の列",
         "負か無限大の行"),
        ("cost_invalid", (missing["cost"] or 0) + (invalid["cost"] or 0),
         "cost", "原価の列", "欠けたか, 負か無限大の行"),
        ("product_missing", missing["product"], "product", "商品の id の列",
         "欠けた行"),
    ]
    for code, count, key, label, kind in checks:
        if count:
            findings.append(finding(
                "warning", code,
                f"{label} {names[key]} の{kind}が {count} 行あります. {fix}",
                [names[key]]
            ))

    # 売れた数の欠け (落とすか残す)
    if missing["units"]:
        findings.append(finding(
            "info", "units_missing",
            f"売れた数の列 {names['units']} の欠けた行が {missing['units']} "
            "行あります. 書き出しでは split.unlabeled で落とすか unlabeled "
            "の分割に残します", [names["units"]]
        ))

    return findings


def range_findings(
    distinct: int, width: Optional[float], products: Optional[dict],
    names: dict
) -> list[dict]:
    """
    値段の種類と幅の不足と, 試した値段が 1 つの商品を気付きにする関数

    Parameters
    ----------
    distinct: int
        使う行の値段の種類の数
    width: Optional[float]
        値段の幅
    products: Optional[dict]
        商品ごとの要約 (商品の id の列を渡さなければ None)
    names: dict
        列の名前 (price, product)

    Returns
    ----------
    findings: list[dict]
        few_prices, single_price_products, narrow_price_range の気付き
    """
    # 値段の種類
    findings = []
    price = names["price"]
    if distinct < MIN_PRICES:
        findings.append(finding(
            "warning", "few_prices",
            f"値段の種類が {distinct} つです ({MIN_PRICES} つ未満). 値段を"
            "変えたときの売れた数の曲がり方を学べないので, 試す値段を"
            "増やしてください", [price]
        ))

    # 試した値段が 1 つの商品
    if products is not None and products["singles"]:
        singles = products["singles"]
        findings.append(finding(
            "warning", "single_price_products",
            f"試した値段が 1 つの商品が {len(singles)} 件あります "
            f"({listed(singles)}). その商品の値段を変えたときの売れ行きは"
            "他の商品からしか分からないので, 値段を変えて試してください",
            [names["product"], price]
        ))

    # 値段の幅
    if width is not None and width < NARROW_WIDTH:
        findings.append(finding(
            "warning", "narrow_price_range",
            f"値段を変えた幅が {width:.1%} です ({NARROW_WIDTH:.0%} 未満. "
            "幅は (最大 − 最小) / 中央値). 候補の値段の多くが記録の外になり"
            "予測が当てにならないので, 値段を変える幅を広げてください",
            [price]
        ))

    return findings


def confounded_findings(
    confounding: list[dict], price: str
) -> list[dict]:
    """
    値段と強く関わる列を気付きにする関数

    Parameters
    ----------
    confounding: list[dict]
        列ごとの関わりの強さ
    price: str
        値段の列の名前

    Returns
    ----------
    findings: list[dict]
        price_confounded の気付き (無ければ空)
    """
    # 関わりの強さが大きい列
    names = [
        item["column"] for item in confounding
        if item["strength"] is not None and item["strength"] >= CONFOUNDED
    ]
    if not names:
        return []
    findings = [finding(
        "warning", "price_confounded",
        f"値段と {listed(names)} が強く関わっています (関わりの強さが "
        f"{CONFOUNDED} 以上). 値段を無作為に近く変えた記録に見えないので, "
        "変え方を確かめてください (偏りのある値段の記録は今の約束の外です)",
        names + [price]
    )]

    return findings


def cost_summary(
    prices: np.ndarray, costs: np.ndarray, name: str
) -> tuple[dict, list[dict]]:
    """
    原価の分かる使う行の原価の平均, 値段に対する割合と, 値段が原価より
    低い行を求める関数

    Parameters
    ----------
    prices: np.ndarray
        使う行の値段
    costs: np.ndarray
        使う行の原価 (欠けと誤りの値は nan)
    name: str
        原価の列の名前

    Returns
    ----------
    summary: dict
        rows, mean, margin, below
    findings: list[dict]
        price_below_cost の気付き (無ければ空)
    """
    # 原価の分かる行
    known = np.isfinite(costs)
    rows = int(known.sum())
    below = int((prices[known] < costs[known]).sum())
    summary = {
        "rows": rows,
        "mean": float(costs[known].mean()) if rows else None,
        "margin": float(
            ((prices[known] - costs[known]) / prices[known]).mean()
        ) if rows else None,
        "below": below,
    }
    findings = []
    if below:
        findings.append(finding(
            "info", "price_below_cost",
            f"値段が原価 ({name}) より低い行が {below} 行あります (売るほど"
            "損になる値段の記録)", [name]
        ))

    return summary, findings


def describe_price(
    source: Union[str, Path, Data], price: str, units: str,
    cost: Optional[str] = None, product: Optional[str] = None,
    columns: Any = None, id: Optional[str] = None,
    loader: Optional[str] = None, **options: Any
) -> dict:
    """
    値段を変えた記録の表の値段の幅と数, 売れた数, 値段と売れた数の
    関わり, 商品ごとの試した値段, 値段と他の列の関わりと気付きを返す関数

    Parameters
    ----------
    source: Union[str, Path, Data]
        表のファイルか, load_data で読んだデータ
    price: str
        値段の列
    units: str
        売れた数の列
    cost: Optional[str] = None
        原価の列
    product: Optional[str] = None
        商品の id の列
    columns: Any = None
        値段との関わりを求める列, None の場合は役割の列を除く数, 真偽,
        カテゴリの列
    id: Optional[str] = None
        行の id の列 (関わりの列から除く)
    loader: Optional[str] = None
        読み込みの名前, None の場合は推定 (load_data と同じ)
    **options: Any
        読み込みの設定 (load_data と同じ)

    Returns
    ----------
    summary: dict
        JSON にできる値段を変えた記録の要約 (price_stats, relation,
        products, confounding, findings など)

    Raises
    ----------
    ValueError
        表でない場合, 列が無い場合, 役割の列が重なる場合, 値段, 売れた数,
        原価の列が数でない場合
    """
    # 表と役割の列
    data = uplift_table(source, loader, options, "値段を変えた記録")
    frame = data.frame.reset_index(drop=True).copy(deep=False)
    frame.columns = [str(name) for name in frame.columns]
    names = {"price": price, "units": units, "cost": cost,
             "product": product, "id": id}
    given = {key: str(name) for key, name in names.items() if name is not None}
    if len(set(given.values())) != len(given):
        raise ValueError(
            "値段, 売れた数, 原価, 商品の id, 行の id の列は違う列にして"
            "ください"
        )
    labels = {"price": "値段の列", "units": "売れた数の列", "cost": "原価の列",
              "product": "商品の id の列", "id": "行の id の列"}
    picked = {
        key: column_of(frame, name, labels[key])
        for key, name in given.items()
    }
    chosen = confounding_columns(
        frame, columns, list(given.values()), data.roles
    )
    price_all = numeric_values(picked["price"], given["price"], "値段の列")
    units_all = numeric_values(picked["units"], given["units"], "売れた数の列")
    cost_all = numeric_values(
        picked["cost"], given["cost"], "原価の列"
    ) if cost is not None else None

    # 欠けと値の誤りの行の数と除外 (原価は数えるだけ)
    price_known = price_all.notna().to_numpy()
    price_array = price_all.to_numpy()
    units_array = units_all.to_numpy()
    price_ok = price_known & np.isfinite(price_array) & (price_array > 0)
    units_known = units_all.notna().to_numpy()
    units_ok = units_known & np.isfinite(units_array) & (units_array >= 0)
    product_known = (
        picked["product"].notna().to_numpy() if product is not None
        else np.ones(len(frame), dtype=bool)
    )
    missing = {
        "price": int((~price_known).sum()),
        "units": int((price_known & ~units_known).sum()),
        "cost": int(cost_all.isna().sum()) if cost_all is not None else None,
        "product": int((price_known & ~product_known).sum())
        if product is not None else None,
    }
    cost_array = cost_all.to_numpy() if cost_all is not None else None
    invalid = {
        "price": int((price_known & ~price_ok).sum()),
        "units": int((units_known & ~units_ok).sum()),
        "cost": int((
            ~np.isnan(cost_array)
            & (~np.isfinite(cost_array) | (cost_array < 0))
        ).sum()) if cost_array is not None else None,
    }
    used = price_ok & units_ok & product_known
    prices = price_array[used]
    sold = units_array[used]
    keys = (
        picked["product"].to_numpy()[used] if product is not None else None
    )

    # 値段と売れた数の要約と関わり
    rows = int(used.sum())
    products = (
        product_summary(prices, sold, keys) if keys is not None else None
    )
    width = products["width"] if products is not None else width_of(prices)
    price_stats = {
        "min": float(prices.min()) if rows else None,
        "max": float(prices.max()) if rows else None,
        "median": float(np.median(prices)) if rows else None,
        "mean": float(prices.mean()) if rows else None,
        "distinct": int(len(np.unique(prices))),
        "width": width,
    }
    unit_stats = {
        "min": float(sold.min()) if rows else None,
        "max": float(sold.max()) if rows else None,
        "mean": float(sold.mean()) if rows else None,
        "zero_rate": float((sold == 0).mean()) if rows else None,
    }
    correlation = (
        float(pd.Series(prices).rank().corr(pd.Series(sold).rank()))
        if rows >= 3 else None
    )
    relation = {
        "correlation": correlation,
        "elasticity": elasticity_of(prices, sold, keys),
        "bins": relation_bins(prices, sold),
    }

    # 値段と他の列の関わり (商品の id を渡すと商品ごとの中央値に対する値段)
    reference = pd.Series(prices)
    if keys is not None:
        groups = pd.Series(keys).astype(str)
        reference = reference / reference.groupby(groups).transform("median")
    table = frame[used].reset_index(drop=True)
    confounding = []
    for name, column_type in chosen:
        strength = association(
            table[name], column_type, reference, "regression"
        )
        confounding.append(
            {"column": name, "type": column_type, "strength": strength}
        )
    confounding.sort(key=lambda item: (
        item["strength"] is None, -(item["strength"] or 0.0)
    ))

    # 原価の要約と気付き (warning を先に)
    findings = value_findings(missing, invalid, given | {
        key: None for key in ("cost", "product") if key not in given
    })
    findings += range_findings(
        price_stats["distinct"], width, products, given
    )
    findings += confounded_findings(confounding, given["price"])
    costs = None
    if cost_array is not None:
        costs, below = cost_summary(
            prices, np.where(
                np.isfinite(cost_array) & (cost_array >= 0), cost_array,
                np.nan
            )[used], given["cost"]
        )
        findings += below
    findings.sort(key=lambda item: item["severity"] != "warning")
    if products is not None:
        products = {k: v for k, v in products.items() if k != "singles"}
    summary = to_json({
        "price": given["price"],
        "units": given["units"],
        "cost": given.get("cost"),
        "product": given.get("product"),
        "rows": rows,
        "missing": missing,
        "invalid": invalid,
        "price_stats": price_stats,
        "unit_stats": unit_stats,
        "relation": relation,
        "products": products,
        "cost_stats": costs,
        "confounding": confounding,
        "findings": findings,
        "source": dict(data.source),
    })

    return summary
