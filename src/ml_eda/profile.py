import math
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.loaders import unique_names

# 分析に使う行の上限 (超える場合は無作為に選んだ見本を使う)
SAMPLE_ROWS = 100_000

# ヒストグラムの区間の数と, カテゴリの頻度を返す数
BINS = 20
TOP_VALUES = 20

# 相関を計算する列の数の上限
MAX_CORRELATION_COLUMNS = 40

# 時系列のグラフに返す点の数の上限
SERIES_POINTS = 500

# 整数の列をカテゴリとみなす異なる値の数の上限
CATEGORY_LIMIT = 10

# 文とみなす平均の文字数
TEXT_LENGTH = 40


def profile(
    data: Data, target: Optional[str] = None, seed: int = 0
) -> dict:
    """
    データを分析し, 概要, 列ごとの結果, 列の間の関係, 気付きを返す関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    target: Optional[str] = None
        目的の列, None の場合は data.roles の target (無ければ目的無し)
    seed: int = 0
        見本を選ぶ乱数のシード

    Returns
    ----------
    result: dict
        JSON にできる分析の結果 (overview, columns, correlations,
        associations, target, timeseries, findings)
    """
    # 列の名前を重ならない文字にそろえる
    frame = data.frame.copy(deep=False)
    frame.columns = unique_names(list(frame.columns))

    # 大きなデータは見本で分析 (時系列は間隔を保つように等間隔に間引く)
    full_time = (
        frame[data.roles["time"]] if data.kind == "timeseries" else None
    )
    sampled = len(frame) > SAMPLE_ROWS
    if sampled and data.kind == "timeseries":
        frame = frame.iloc[::math.ceil(len(frame) / SAMPLE_ROWS)]
    elif sampled:
        frame = frame.sample(SAMPLE_ROWS, random_state=seed)
    target = target or data.roles.get("target")
    if target is not None and target not in frame.columns:
        raise ValueError(f"目的の列 {target} がありません")

    # 列ごとの型と統計
    columns = {
        str(name): describe_column(frame[name], data.roles)
        for name in frame.columns
    }

    # 列の間の関係と, 目的の列との関係
    correlations = numeric_correlations(frame, columns)
    associations = categorical_associations(frame, columns)
    target_result = (
        describe_target(frame, target, columns) if target else None
    )
    series = (
        describe_timeseries(frame, data.roles["time"], columns, full_time)
        if data.kind == "timeseries" else None
    )

    # 概要と気付き
    overview = describe_overview(data, frame, sampled)
    result = {
        "overview": overview,
        "columns": columns,
        "correlations": correlations,
        "associations": associations,
        "target": target_result,
        "timeseries": series,
    }
    result["findings"] = find_issues(result, target)
    if data.item_loader is not None:
        from ml_eda.media import describe_items

        result["items"] = describe_items(data, seed)
        # 中身の様子の列 (幅, 色の形式など) が揃っているのは良いことなので,
        # 定数の気付きは出さない
        quiet = {"constant", "near_constant"}
        result["findings"] = result["items"].pop("findings") + [
            item for item in result["findings"] if item["code"] not in quiet
        ]
    result = to_json(result)

    return result


def describe_overview(
    data: Data, frame: pd.DataFrame, sampled: bool
) -> dict:
    """
    件数, 列数, メモリ, 重複した行などの概要を求める関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    frame: pd.DataFrame
        分析に使う表 (見本の場合あり)
    sampled: bool
        見本を使ったか

    Returns
    ----------
    overview: dict
        概要
    """
    # 重複は値の比べられる列だけで数える (リストなどを含む列は除く)
    hashable = [
        name for name in frame.columns
        if not frame[name].map(lambda v: isinstance(v, (list, dict))).any()
    ]
    duplicates = int(frame[hashable].duplicated().sum()) if hashable else 0
    overview = {
        "kind": data.kind,
        "rows": int(len(data.frame)),
        "columns": int(data.frame.shape[1]),
        "memory_bytes": int(data.frame.memory_usage(deep=True).sum()),
        "duplicate_rows": duplicates,
        "missing_cells": int(frame.isna().sum().sum()),
        "sampled": sampled,
        "sample_rows": int(len(frame)),
        "roles": dict(data.roles),
    }

    return overview


def infer_type(column: pd.Series, roles: dict) -> str:
    """
    列の分析上の型を推定する関数

    Parameters
    ----------
    column: pd.Series
        列
    roles: dict
        列の役割 (日時の列などの指定)

    Returns
    ----------
    kind: str
        numeric, categorical, boolean, datetime, text, id, constant,
        empty, other のどれか
    """
    # 値の無い列と, 値が 1 つの列
    values = column.dropna()
    name = str(column.name)
    if values.empty:
        return "empty"
    if name == roles.get("time") or pd.api.types.is_datetime64_any_dtype(
        column
    ):
        return "datetime"
    if pd.api.types.is_bool_dtype(column):
        return "boolean"
    if values.map(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
        return "other"
    unique = values.nunique()
    if unique == 1:
        return "constant"

    # 数値の列 (整数で値の少ないものはカテゴリ, 連番は ID)
    if pd.api.types.is_numeric_dtype(column):
        integer = pd.api.types.is_integer_dtype(column) or bool(
            np.all(np.mod(values.to_numpy(dtype=float), 1) == 0)
        )
        if integer and unique == 2:
            return "boolean"
        if integer and unique <= CATEGORY_LIMIT and len(values) > 2 * unique:
            return "categorical"
        id_name = re.search(r"(^|_)id$|Id$|ID$", name) is not None
        if integer and unique == len(values) and len(values) > 20 and (
            values.is_monotonic_increasing or id_name
        ):
            return "id"
        return "numeric"

    # 文字の列 (長いものは文, 全て異なる短いものは ID)
    text = values.astype(str)
    if text.str.len().mean() >= TEXT_LENGTH:
        return "text"
    if unique == 2:
        return "boolean"
    if unique == len(values) and len(values) > 20:
        return "id"
    kind = "categorical"

    return kind


def describe_column(column: pd.Series, roles: dict) -> dict:
    """
    列の型, 欠損, 異なる値の数と, 型ごとの統計を求める関数

    Parameters
    ----------
    column: pd.Series
        列
    roles: dict
        列の役割

    Returns
    ----------
    result: dict
        列の分析の結果
    """
    # 型と欠損と異なる値の数
    kind = infer_type(column, roles)
    missing = int(column.isna().sum())
    try:
        unique = int(column.nunique())
    except TypeError:
        unique = int(column.astype(str).nunique())
    result = {
        "type": kind,
        "dtype": str(column.dtype),
        "count": int(len(column)),
        "missing": missing,
        "missing_ratio": missing / len(column) if len(column) else 0.0,
        "unique": unique,
    }

    # 型ごとの統計
    values = column.dropna()
    if kind == "numeric":
        result.update(numeric_stats(values.astype(float)))
    elif kind in ("categorical", "boolean", "id", "constant"):
        result.update(category_stats(values))
    elif kind == "text":
        result.update(text_stats(values.astype(str)))
    elif kind == "datetime":
        result.update(datetime_stats(values))

    return result


def numeric_stats(values: pd.Series) -> dict:
    """
    数値の列の統計とヒストグラムを求める関数

    Parameters
    ----------
    values: pd.Series
        欠損を除いた値

    Returns
    ----------
    stats: dict
        平均, 標準偏差, 分位点, 歪度, 0 の数, 外れ値の数, ヒストグラム
    """
    # 分位点と四分位範囲の外の値 (外れ値)
    finite = values[np.isfinite(values)]
    q1, median, q3 = (float(v) for v in finite.quantile([0.25, 0.5, 0.75]))
    iqr = q3 - q1
    outliers = int(
        ((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).sum()
    ) if iqr > 0 else 0

    # ヒストグラム
    counts, edges = np.histogram(finite, bins=BINS)
    stats = {
        "mean": float(finite.mean()),
        "std": float(finite.std()) if len(finite) > 1 else 0.0,
        "min": float(finite.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(finite.max()),
        "skew": float(finite.skew()) if len(finite) > 2 else 0.0,
        "zeros": int((finite == 0).sum()),
        "negatives": int((finite < 0).sum()),
        "infinite": int(len(values) - len(finite)),
        "outliers": outliers,
        "histogram": {
            "counts": counts.tolist(), "edges": edges.tolist()
        },
    }

    return stats


def category_stats(values: pd.Series) -> dict:
    """
    カテゴリの列の頻度の上位を求める関数

    Parameters
    ----------
    values: pd.Series
        欠損を除いた値

    Returns
    ----------
    stats: dict
        頻度の上位 (値と件数), 最も多い値の割合
    """
    # 頻度の上位と, それ以外の件数
    counts = values.astype(str).value_counts()
    top = [
        {"value": str(value), "count": int(count)}
        for value, count in counts.head(TOP_VALUES).items()
    ]
    stats = {
        "top": top,
        "other_count": int(counts.iloc[TOP_VALUES:].sum()),
        "top_ratio": float(counts.iloc[0] / len(values)) if len(values) else 0,
    }

    return stats


def text_stats(values: pd.Series) -> dict:
    """
    文の列の長さの統計と頻出語を求める関数

    Parameters
    ----------
    values: pd.Series
        欠損を除いた文

    Returns
    ----------
    stats: dict
        文字数と語の数の統計, 長さのヒストグラム, 頻出語
    """
    # 文字数と, 空白で分けた語の数
    lengths = values.str.len()
    words = values.str.split()
    word_counts = words.map(len)
    counts, edges = np.histogram(lengths, bins=BINS)

    # 頻出語 (小文字にし, 記号を除いた語)
    tokens = (
        values.str.lower().str.findall(r"\w+").explode().dropna()
    )
    top_words = [
        {"value": str(word), "count": int(count)}
        for word, count in tokens.value_counts().head(TOP_VALUES).items()
    ]
    stats = {
        "length_mean": float(lengths.mean()),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "words_mean": float(word_counts.mean()),
        "vocabulary": int(tokens.nunique()),
        "length_histogram": {
            "counts": counts.tolist(), "edges": edges.tolist()
        },
        "top_words": top_words,
    }

    return stats


def datetime_stats(values: pd.Series) -> dict:
    """
    日時の列の範囲と間隔を求める関数

    Parameters
    ----------
    values: pd.Series
        欠損を除いた日時

    Returns
    ----------
    stats: dict
        最初と最後の時刻, 最も多い間隔 (秒)
    """
    # 並べた時刻の差で間隔を求める
    ordered = as_datetime(values).sort_values()
    diffs = ordered.diff().dropna().dt.total_seconds()
    positive = diffs[diffs > 0]
    stats = {
        "min": ordered.iloc[0].isoformat(),
        "max": ordered.iloc[-1].isoformat(),
        "interval_seconds": (
            float(positive.mode().iloc[0]) if len(positive) else None
        ),
    }

    return stats


def numeric_correlations(frame: pd.DataFrame, columns: dict) -> dict:
    """
    数値の列の間の相関 (Pearson と Spearman) を求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    columns: dict
        列ごとの分析の結果

    Returns
    ----------
    result: dict
        列の名前の並びと, 2 つの相関の行列
    """
    # 数値の列 (多い場合は先頭から上限まで)
    names = [
        name for name, info in columns.items() if info["type"] == "numeric"
    ][:MAX_CORRELATION_COLUMNS]
    if len(names) < 2:
        return {"columns": names, "pearson": [], "spearman": []}
    numeric = frame[names].astype(float)
    result = {
        "columns": names,
        "pearson": numeric.corr(method="pearson").to_numpy().tolist(),
        "spearman": numeric.rank().corr(method="pearson").to_numpy().tolist(),
    }

    return result


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    """
    2 つのカテゴリの列の関連の強さ (Cramér の V) を求める関数

    Parameters
    ----------
    left: pd.Series
        カテゴリの列
    right: pd.Series
        カテゴリの列

    Returns
    ----------
    value: float
        0 (関連なし) から 1 (一方で他方が決まる)
    """
    # 分割表から χ² を求め, 値の種類が多いと大きく出る偏りを補正して
    # 正規化 (Bergsma の補正)
    table = pd.crosstab(left, right).to_numpy(dtype=float)
    total = table.sum()
    if total <= 1 or min(table.shape) < 2:
        return 0.0
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True)
    expected = expected / total
    chi2 = float(((table - expected) ** 2 / expected).sum())
    rows, cols = table.shape
    phi2 = max(0.0, chi2 / total - (rows - 1) * (cols - 1) / (total - 1))
    rows_c = rows - (rows - 1) ** 2 / (total - 1)
    cols_c = cols - (cols - 1) ** 2 / (total - 1)
    denominator = min(rows_c - 1, cols_c - 1)
    if denominator <= 0:
        return 0.0
    value = min(math.sqrt(phi2 / denominator), 1.0)

    return value


def as_datetime(values: pd.Series) -> pd.Series:
    """
    日時の列に変換する関数 (時差の違う値が混ざる場合だけ UTC にそろえる)

    Parameters
    ----------
    values: pd.Series
        日時か日時の文字の列

    Returns
    ----------
    converted: pd.Series
        日時の列
    """
    # 時差の無い列はそのまま, 混ざる列は UTC
    try:
        converted = pd.to_datetime(values)
        if pd.api.types.is_datetime64_any_dtype(converted):
            return converted
    except (ValueError, TypeError):
        pass
    converted = pd.to_datetime(values, utc=True)

    return converted


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    カテゴリと数値の関連の強さ (相関比) を求める関数

    Parameters
    ----------
    categories: pd.Series
        カテゴリの列
    values: pd.Series
        数値の列

    Returns
    ----------
    eta: float
        0 (関連なし) から 1 (カテゴリで値が決まる)
    """
    # カテゴリごとの平均のばらつきと, 全体のばらつきの比
    frame = pd.DataFrame({"c": categories.astype(str), "v": values})
    frame = frame.dropna()
    if frame.empty:
        return 0.0
    total = ((frame["v"] - frame["v"].mean()) ** 2).sum()
    if total == 0:
        return 0.0
    groups = frame.groupby("c")["v"]
    between = (groups.count() * (groups.mean() - frame["v"].mean()) ** 2)
    eta = min(math.sqrt(float(between.sum()) / float(total)), 1.0)

    return eta


def categorical_associations(frame: pd.DataFrame, columns: dict) -> dict:
    """
    カテゴリの列の間の関連 (Cramér の V) を求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    columns: dict
        列ごとの分析の結果

    Returns
    ----------
    result: dict
        列の名前の並びと, Cramér の V の行列
    """
    # 値の種類が多すぎない カテゴリの列
    names = [
        name for name, info in columns.items()
        if info["type"] in ("categorical", "boolean") and info["unique"] <= 50
    ][:MAX_CORRELATION_COLUMNS // 2]
    matrix = [
        [
            1.0 if left == right
            else cramers_v(frame[left].astype(str), frame[right].astype(str))
            for right in names
        ]
        for left in names
    ]
    result = {"columns": names, "cramers_v": matrix}

    return result


def guess_task(info: dict) -> str:
    """
    目的の列の型からタスクの種類を推定する関数

    Parameters
    ----------
    info: dict
        目的の列の分析の結果

    Returns
    ----------
    task: str
        classification か regression
    """
    # 数値は回帰, それ以外 (カテゴリ, 2 値, 文, ID など) は分類
    task = "regression" if info["type"] == "numeric" else "classification"

    return task


def association(
    feature: pd.Series, feature_type: str, target: pd.Series, task: str
) -> Optional[float]:
    """
    1 つの列と目的の列の関連の強さを求める関数

    Parameters
    ----------
    feature: pd.Series
        列
    feature_type: str
        列の分析上の型
    target: pd.Series
        目的の列
    task: str
        classification か regression

    Returns
    ----------
    value: Optional[float]
        0 から 1 の強さ, 求められない型の場合は None
    """
    # 型の組み合わせで指標を選ぶ
    numeric = feature_type == "numeric"
    category = feature_type in ("categorical", "boolean")
    pair = pd.DataFrame({"f": feature, "t": target}).dropna()
    if len(pair) < 3:
        return None
    if numeric and task == "regression":
        value = abs(float(pair["f"].astype(float).rank().corr(
            pair["t"].astype(float).rank()
        )))
    elif numeric:
        value = correlation_ratio(pair["t"], pair["f"].astype(float))
    elif category and task == "regression":
        value = correlation_ratio(pair["f"], pair["t"].astype(float))
    elif category:
        value = cramers_v(pair["f"].astype(str), pair["t"].astype(str))
    else:
        return None
    if math.isnan(value):
        return None

    return value


def describe_target(
    frame: pd.DataFrame, target: str, columns: dict
) -> dict:
    """
    目的の列の分布と, 各列との関連の強さの順位を求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    target: str
        目的の列
    columns: dict
        列ごとの分析の結果

    Returns
    ----------
    result: dict
        目的の列の名前, タスクの推定, 分布, 関連の強さの順位
    """
    # タスクの推定と目的の列の分布
    info = columns[target]
    task = guess_task(info)
    result: dict = {"column": target, "task": task, "type": info["type"]}
    if task == "classification":
        counts = frame[target].dropna().astype(str).value_counts()
        result["classes"] = [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ]

    # 各列と目的の列の関連の強さを強い順に並べる (クラスが多すぎる場合は
    # 分割表が大きくなるので求めない)
    ranking = []
    for name, feature in columns.items():
        many = task == "classification" and info["unique"] > 500
        if name == target or many:
            continue
        value = association(frame[name], feature["type"], frame[target], task)
        if value is not None:
            ranking.append({"column": name, "strength": value})
    ranking.sort(key=lambda item: -item["strength"])
    result["associations"] = ranking

    return result


def autocorrelation(values: np.ndarray, max_lag: int) -> list[float]:
    """
    系列の自己相関をずらし幅ごとに求める関数

    Parameters
    ----------
    values: np.ndarray
        欠損を除いた系列
    max_lag: int
        最大のずらし幅

    Returns
    ----------
    acf: list[float]
        ずらし幅 1 から max_lag までの自己相関
    """
    # 平均を引いた系列の内積で求める
    centered = values - values.mean()
    denominator = float((centered ** 2).sum())
    if denominator == 0:
        return [0.0] * max_lag
    acf = [
        float((centered[:-lag] * centered[lag:]).sum() / denominator)
        for lag in range(1, max_lag + 1)
    ]

    return acf


def find_period(acf: list[float]) -> Optional[int]:
    """
    自己相関の最初の山から周期の候補を求める関数

    Parameters
    ----------
    acf: list[float]
        ずらし幅 1 からの自己相関

    Returns
    ----------
    period: Optional[int]
        周期の候補のずらし幅 (はっきりした山が無ければ None)
    """
    # 一度下がってから上がる最初の極大で, 十分に高いもの
    for lag in range(2, len(acf) - 1):
        peak = acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1]
        if peak and acf[lag] > 0.3 and min(acf[:lag]) < acf[lag] - 0.1:
            return lag + 1

    return None


def describe_timeseries(
    frame: pd.DataFrame, time: str, columns: dict,
    full_time: Optional[pd.Series] = None
) -> dict:
    """
    時系列の間隔, 抜け, 周期の候補, グラフ用の系列を求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        時刻の順に並んだ表
    time: str
        日時の列
    columns: dict
        列ごとの分析の結果
    full_time: Optional[pd.Series] = None
        間引く前の日時の列 (間隔と抜けは全体で求める)

    Returns
    ----------
    result: dict
        間隔 (秒), 抜けと重なりの数, 列ごとの周期の候補と自己相関,
        間引いた系列
    """
    # 間隔と, 抜け (最も多い間隔の 1.5 倍より長い) と重なり
    source = full_time if full_time is not None else frame[time]
    times = as_datetime(source).dropna().sort_values()
    diffs = times.diff().dropna().dt.total_seconds()
    positive = diffs[diffs > 0]
    interval = float(positive.mode().iloc[0]) if len(positive) else None
    gaps = int((diffs > 1.5 * interval).sum()) if interval else 0
    duplicates = int((diffs == 0).sum())

    # 数値の列ごとの自己相関と周期の候補
    names = [n for n, i in columns.items() if i["type"] == "numeric"][:10]
    max_lag = max(1, min(len(frame) // 2, 200))
    per_column = {}
    for name in names:
        values = frame[name].astype(float).dropna().to_numpy()
        if len(values) < 10:
            continue
        acf = autocorrelation(values, min(max_lag, len(values) - 1))
        per_column[name] = {"acf": acf, "period": find_period(acf)}

    # グラフ用に間引いた系列
    step = max(1, len(frame) // SERIES_POINTS)
    thinned = frame.iloc[::step]
    series = {
        "time": [
            t.isoformat() if pd.notna(t) else None
            for t in as_datetime(thinned[time])
        ],
        "values": {
            name: thinned[name].astype(float).tolist() for name in names
        },
    }
    result = {
        "time_column": time,
        "interval_seconds": interval,
        "gaps": gaps,
        "duplicates": duplicates,
        "start": times.iloc[0].isoformat() if len(times) else None,
        "end": times.iloc[-1].isoformat() if len(times) else None,
        "columns": per_column,
        "series": series,
    }

    return result


def finding(
    severity: str, code: str, message: str, columns: list[str]
) -> dict:
    """
    気付きを 1 つの辞書にする関数

    Parameters
    ----------
    severity: str
        warning か info
    code: str
        決まった英字の識別子
    message: str
        理由と次にできることの文
    columns: list[str]
        関係する列

    Returns
    ----------
    item: dict
        気付き
    """
    # 決まった形の辞書
    item = {
        "severity": severity, "code": code, "message": message,
        "columns": columns,
    }

    return item


def find_issues(result: dict, target: Optional[str]) -> list[dict]:
    """
    分析の結果から, 学習の前に確かめた方がよいことを集める関数

    Parameters
    ----------
    result: dict
        分析の結果
    target: Optional[str]
        目的の列

    Returns
    ----------
    findings: list[dict]
        気付きの並び (warning を先に)
    """
    # 列ごとの気付き
    findings = []
    for name, info in result["columns"].items():
        findings.extend(column_issues(name, info, target))

    # 重複した行
    duplicates = result["overview"]["duplicate_rows"]
    if duplicates:
        where = "見本の中に" if result["overview"]["sampled"] else ""
        findings.append(finding(
            "warning", "duplicate_rows",
            f"{where}全く同じ行が {duplicates} 行あります. 学習用と検証用に同じ行が"
            "入ると検証の結果が良く見えすぎるので, 重複を除く手順を検討して"
            "ください", []
        ))

    # 強く相関する数値の列の組
    findings.extend(correlation_issues(result["correlations"], target))

    # 目的の列
    if result["target"]:
        findings.extend(target_issues(result, target))

    # 時系列の間隔
    series = result["timeseries"]
    if series and (series["gaps"] or series["duplicates"]):
        findings.append(finding(
            "warning", "irregular_interval",
            f"時刻の間隔が一定ではありません (抜け {series['gaps']} か所, "
            f"同じ時刻 {series['duplicates']} か所). ずらしや移動平均の前に, "
            "一定の間隔に並べ直すか補完することを検討してください",
            [series["time_column"]]
        ))
    findings.sort(key=lambda item: item["severity"] != "warning")

    return findings


def column_issues(name: str, info: dict, target: Optional[str]) -> list:
    """
    1 つの列の気付きを集める関数

    Parameters
    ----------
    name: str
        列の名前
    info: dict
        列の分析の結果
    target: Optional[str]
        目的の列

    Returns
    ----------
    findings: list
        気付きの並び
    """
    # 目的の列は目的の列の気付きで扱う
    findings: list = []
    if name == target:
        return findings
    kind = info["type"]
    if info["missing_ratio"] >= 0.3:
        findings.append(finding(
            "warning", "high_missing",
            f"{name} は {info['missing_ratio']:.0%} が欠損しています. "
            "列を落とすか, 補完するか (欠損したことを表す列も足す) を"
            "検討してください", [name]
        ))
    if kind in ("constant", "empty"):
        findings.append(finding(
            "warning", "constant",
            f"{name} は値が 1 つ (か無い) ので, 学習に役立ちません. "
            "列を落とす候補です", [name]
        ))
    elif kind in ("categorical", "boolean") and (
        info.get("top_ratio", 0) >= 0.95
    ):
        findings.append(finding(
            "info", "near_constant",
            f"{name} は {info['top_ratio']:.0%} が同じ値です. "
            "情報が少ないので, 落とす候補です", [name]
        ))
    if kind == "id":
        findings.append(finding(
            "warning", "id_like",
            f"{name} は全ての値が異なり, ID らしい列です. 学習に使うと"
            "覚えるだけになるので, 落とす候補です", [name]
        ))
    if kind == "categorical" and info["unique"] > 50:
        findings.append(finding(
            "info", "high_cardinality",
            f"{name} は異なる値が {info['unique']} 個あります. ワンホットに"
            "すると列が増えすぎるので, 頻度の低い値をまとめるか番号付けを"
            "検討してください", [name]
        ))
    if kind == "numeric" and abs(info.get("skew", 0)) >= 2:
        findings.append(finding(
            "info", "skewed",
            f"{name} は分布が大きく偏っています (歪度 {info['skew']:.1f}). "
            "値が全て 0 以上なら, 対数を取る手順の候補です", [name]
        ))
    if kind == "numeric" and info.get("outliers", 0) > 0.05 * info["count"]:
        findings.append(finding(
            "info", "outliers",
            f"{name} は外れ値 (四分位範囲の 1.5 倍の外) が "
            f"{info['outliers']} 件あります. 切り詰める手順の候補です",
            [name]
        ))

    return findings


def correlation_issues(correlations: dict, target: Optional[str]) -> list:
    """
    とても強く相関する数値の列の組を気付きにする関数

    Parameters
    ----------
    correlations: dict
        数値の列の相関
    target: Optional[str]
        目的の列 (目的の列との相関はリークの気付きで扱う)

    Returns
    ----------
    findings: list
        気付きの並び
    """
    # Pearson の絶対値が 0.95 以上の組
    findings = []
    names = correlations["columns"]
    matrix = correlations["pearson"]
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            value = matrix[i][j]
            if target in (left, right) or value is None or math.isnan(value):
                continue
            if abs(value) >= 0.95:
                findings.append(finding(
                    "info", "high_correlation",
                    f"{left} と {right} はとても強く相関しています "
                    f"(相関係数 {value:.2f}). どちらか一方で足りる可能性が"
                    "あります", [left, right]
                ))

    return findings


def target_issues(result: dict, target: str) -> list:
    """
    目的の列の気付き (欠損, クラスの偏り, リークの疑い) を集める関数

    Parameters
    ----------
    result: dict
        分析の結果
    target: str
        目的の列

    Returns
    ----------
    findings: list
        気付きの並び
    """
    # 目的の列の欠損
    findings = []
    info = result["columns"][target]
    if info["missing"]:
        findings.append(finding(
            "warning", "target_missing",
            f"目的の列 {target} に欠損が {info['missing']} 件あります. "
            "学習の前にその行を落としてください", [target]
        ))

    # 分類のクラスの偏り (最も少ないクラスが最も多いクラスの 1/10 未満)
    described = result["target"]
    classes = described.get("classes") or []
    if len(classes) >= 2:
        counts = [item["count"] for item in classes]
        if min(counts) < 0.1 * max(counts):
            findings.append(finding(
                "warning", "class_imbalance",
                f"目的の列 {target} のクラスが偏っています (最多 "
                f"{max(counts)} 件, 最少 {min(counts)} 件). 層化して分割し, "
                "評価には正解率ではなく F1 や AUC を使うことを検討して"
                "ください", [target]
            ))

    # 目的の列とほぼ同じ情報を持つ列 (リークの疑い)
    for item in described["associations"]:
        if item["strength"] >= 0.98:
            findings.append(finding(
                "warning", "target_leak",
                f"{item['column']} は目的の列 {target} とほぼ同じ情報を"
                f"持っています (関連の強さ {item['strength']:.2f}). 予測の"
                "ときに手に入らない情報 (答えから作った列など) なら, "
                "落としてください", [item["column"], target]
            ))

    return findings


def to_json(value: Any) -> Any:
    """
    numpy の値と NaN を, JSON にできる値に変換する関数

    Parameters
    ----------
    value: Any
        変換する値 (辞書とリストは中も変換)

    Returns
    ----------
    converted: Any
        JSON にできる値 (NaN と無限大は None)
    """
    # 入れ物は中を順に変換
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    converted = value

    return converted
