import copy
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import yaml

# レシピの形式の版
FORMAT_VERSION = 1

# タスクの種類
TASKS = ("classification", "regression", "forecasting", "none")

# 分割のしかた
SPLIT_METHODS = ("random", "stratified", "time", "group", "none")

# 目的の列の欠けた行の扱い (落とすか, unlabeled の分割に残すか)
UNLABELED_MODES = ("drop", "keep")


@dataclass
class Step:
    # 手順の名前 (レシピの op)
    name: str
    # 当てはめ (学習用のデータから値を学ぶ. 学ばない手順は None)
    fit: Optional[Callable[[pd.DataFrame, dict, dict], dict]]
    # 適用 (当てはめた値を使って表を変える)
    apply: Callable[[pd.DataFrame, dict, dict], pd.DataFrame]
    # 引数の説明 (書いた鍵だけを手順の引数として受け付ける)
    params: dict = field(default_factory=dict)
    # 説明
    description: str = ""
    # 行を増やしたり減らしたりするか (目的の列と並べ直す必要)
    changes_rows: bool = False


# 手順の登録簿
STEPS: dict[str, Step] = {}

# 手順の引数の他に手順が持つ鍵 (手順の名前と, 当てはめた値)
STEP_KEYS = {"op", "state"}

# 処理の間だけ使う列 (分割, 日時の写し, 元の行の位置, グループの写し)
HIDDEN = ("__split__", "__time__", "__row__", "__group__")

# columns が auto のときに選ぶ列の型 (書いていない手順は数値の列)
WANT = {
    "one_hot": "categorical", "label_encode": "categorical",
    "normalize_text": "categorical",
    "datetime_parts": "datetime",
}


def category_key(value: Any) -> str:
    """
    カテゴリの値を, 型に依らない文字の鍵にする関数

    整数の列に欠損が入ると float になり 1 が 1.0 になるので, 整数の値の
    float は整数の文字にそろえる

    Parameters
    ----------
    value: Any
        値 (欠損でないもの)

    Returns
    ----------
    key: str
        鍵
    """
    # 整数の値の float は整数の文字
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    key = str(value)

    return key


def register_step(step: Step) -> None:
    """
    前処理の手順を登録簿に加える関数

    Parameters
    ----------
    step: Step
        加える手順

    Returns
    ----------
    None
    """
    # 名前で登録
    STEPS[step.name] = step


def describe_steps() -> list[dict]:
    """
    登録した前処理の手順の一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    steps: list[dict]
        名前, 引数の説明, 説明, 当てはめるか の辞書の並び
    """
    # JSON にできる形に変換
    steps = [
        {
            "op": step.name, "params": dict(step.params),
            "description": step.description, "fits": step.fit is not None,
        }
        for step in STEPS.values()
    ]

    return steps


def resolve_columns(
    frame: pd.DataFrame, spec: Any, roles: dict, want: str = "numeric"
) -> list[str]:
    """
    手順の columns (列の名前の並びか auto) を列の名前の並びにする関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    spec: Any
        列の名前の並び, 1 つの名前, auto (目的の列などを除く, want の型の列)
    roles: dict
        列の役割
    want: str = "numeric"
        auto で選ぶ列の型 (numeric, categorical, datetime)

    Returns
    ----------
    names: list[str]
        列の名前の並び

    Raises
    ----------
    ValueError
        無い列を指定した場合
    """
    # 名前の指定はそのまま (無い列は誤り)
    reserved = {
        roles.get("target"), roles.get("time"), roles.get("id"),
        roles.get("text")
    }
    if spec not in (None, "auto"):
        names = [spec] if isinstance(spec, str) else list(spec)
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise ValueError(f"列がありません: {', '.join(missing)}")
        return names

    # auto は型で選ぶ
    names = []
    for name in frame.columns:
        # 役割のある列と, 処理の間だけ使う列は除く
        if name in reserved or name in HIDDEN:
            continue
        column = frame[name]
        numeric = pd.api.types.is_numeric_dtype(column) and not (
            pd.api.types.is_bool_dtype(column)
        )
        datetime = pd.api.types.is_datetime64_any_dtype(column)
        if want == "numeric" and numeric:
            names.append(name)
        elif want == "categorical" and not numeric and not datetime:
            names.append(name)
        elif want == "datetime" and datetime:
            names.append(name)

    return names


def fit_impute(frame: pd.DataFrame, params: dict, roles: dict) -> dict:
    """
    欠損を埋める値を学習用のデータから求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        学習用の表
    params: dict
        columns, strategy (mean, median, most_frequent, constant), value
    roles: dict
        列の役割

    Returns
    ----------
    state: dict
        列ごとの埋める値
    """
    # 戦略に合わせて列ごとの値を求める
    strategy = params.get("strategy", "median")
    want = "categorical" if strategy == "most_frequent" else "numeric"
    names = resolve_columns(frame, params.get("columns"), roles, want)
    values = {}
    for name in names:
        column = frame[name]
        if strategy == "constant":
            values[name] = params.get("value", 0)
        elif strategy == "mean":
            values[name] = float(column.mean())
        elif strategy == "median":
            values[name] = float(column.median())
        elif strategy == "most_frequent":
            mode = column.mode()
            values[name] = mode.iloc[0] if len(mode) else None
        else:
            raise ValueError(f"知らない補完のしかたです: {strategy}")
    state = {"values": to_plain(values)}

    return state


def apply_impute(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    欠損を当てはめた値で埋める関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        indicator (欠損したことを表す列を足すか)
    state: dict
        列ごとの埋める値

    Returns
    ----------
    result: pd.DataFrame
        埋めた表
    """
    # 欠損の印の列を足してから埋める
    result = frame.copy()
    for name, value in state["values"].items():
        if params.get("indicator"):
            result[f"{name}_missing"] = result[name].isna().astype(int)
        result[name] = result[name].fillna(value)

    return result


def fit_scale(frame: pd.DataFrame, params: dict, roles: dict) -> dict:
    """
    標準化か最小と最大の正規化の値を学習用のデータから求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        学習用の表
    params: dict
        columns, method (standard か minmax)
    roles: dict
        列の役割

    Returns
    ----------
    state: dict
        列ごとの中心と幅
    """
    # 列ごとに引く値と割る値
    names = resolve_columns(frame, params.get("columns"), roles)
    method = params.get("method", "standard")
    values = {}
    for name in names:
        column = frame[name].astype(float)
        if method == "standard":
            center, scale = float(column.mean()), float(column.std())
        else:
            center = float(column.min())
            scale = float(column.max()) - center
        values[name] = [center, scale if scale and scale > 0 else 1.0]
    state = {"values": values}

    return state


def apply_scale(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    当てはめた中心と幅で列を変換する関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        使わない
    state: dict
        列ごとの中心と幅

    Returns
    ----------
    result: pd.DataFrame
        変換した表
    """
    # 中心を引いて幅で割る
    result = frame.copy()
    for name, (center, scale) in state["values"].items():
        result[name] = (result[name].astype(float) - center) / scale

    return result


def fit_one_hot(frame: pd.DataFrame, params: dict, roles: dict) -> dict:
    """
    ワンホットにするカテゴリを学習用のデータから求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        学習用の表
    params: dict
        columns, max_categories (多い場合は頻度の上位だけ, 残りは other)
    roles: dict
        列の役割

    Returns
    ----------
    state: dict
        列ごとのカテゴリの並び
    """
    # 列ごとに頻度の上位のカテゴリ (欠損はカテゴリにしない)
    names = resolve_columns(frame, params.get("columns"), roles, "categorical")
    limit = int(params.get("max_categories", 20))
    categories = {
        name: [
            str(value) for value in frame[name].dropna().map(category_key)
            .value_counts().index[:limit]
        ]
        for name in names
    }
    state = {"categories": categories}

    return state


def apply_one_hot(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    カテゴリの列を 0 と 1 の列に置き換える関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        使わない
    state: dict
        列ごとのカテゴリの並び

    Returns
    ----------
    result: pd.DataFrame
        置き換えた表 (学習用に無かった値は全て 0)
    """
    # カテゴリごとの列を作り, 元の列を落とす (欠損は全て 0)
    result = frame.copy()
    for name, categories in state["categories"].items():
        values = result[name].map(
            lambda v: category_key(v) if pd.notna(v) else None
        )
        for category in categories:
            result[f"{name}={category}"] = (values == category).astype(int)
        result = result.drop(columns=[name])

    return result


def fit_label_encode(frame: pd.DataFrame, params: dict, roles: dict) -> dict:
    """
    カテゴリに付ける番号を学習用のデータから求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        学習用の表
    params: dict
        columns
    roles: dict
        列の役割

    Returns
    ----------
    state: dict
        列ごとのカテゴリの並び (並びの位置が番号)
    """
    # 値の並びを決める
    names = resolve_columns(frame, params.get("columns"), roles, "categorical")
    categories = {
        name: sorted(frame[name].dropna().map(category_key).unique().tolist())
        for name in names
    }
    state = {"categories": categories}

    return state


def apply_label_encode(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    カテゴリの列を番号に置き換える関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        使わない
    state: dict
        列ごとのカテゴリの並び

    Returns
    ----------
    result: pd.DataFrame
        置き換えた表 (学習用に無かった値と欠損は -1)
    """
    # 並びの位置を番号にする
    result = frame.copy()
    for name, categories in state["categories"].items():
        index = {value: number for number, value in enumerate(categories)}
        result[name] = result[name].map(
            lambda v: index.get(category_key(v), -1) if pd.notna(v) else -1
        ).astype(int)

    return result


def fit_clip(frame: pd.DataFrame, params: dict, roles: dict) -> dict:
    """
    外れ値を切り詰める境界を学習用のデータから求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        学習用の表
    params: dict
        columns, factor (四分位範囲の何倍の外を切るか, 既定 1.5)
    roles: dict
        列の役割

    Returns
    ----------
    state: dict
        列ごとの下限と上限
    """
    # 四分位範囲から境界を求める
    names = resolve_columns(frame, params.get("columns"), roles)
    factor = float(params.get("factor", 1.5))
    bounds = {}
    for name in names:
        q1, q3 = frame[name].astype(float).quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        bounds[name] = [float(q1) - factor * iqr, float(q3) + factor * iqr]
    state = {"bounds": bounds}

    return state


def apply_clip(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    当てはめた境界で値を切り詰める関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        使わない
    state: dict
        列ごとの下限と上限

    Returns
    ----------
    result: pd.DataFrame
        切り詰めた表
    """
    # 下限と上限の間に収める
    result = frame.copy()
    for name, (low, high) in state["bounds"].items():
        result[name] = result[name].astype(float).clip(low, high)

    return result


def apply_drop_columns(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    列を落とす関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        columns (落とす列の名前の並び)
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        列を落とした表
    """
    # 無い列は誤り
    names = resolve_columns(frame, params.get("columns", []), {})
    result = frame.drop(columns=names)

    return result


def apply_drop_missing(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    欠損のある行を落とす関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        columns (見る列, 省略すると全ての列)
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        行を落とした表
    """
    # 指定の列 (省略するとグループの写しを除く全ての列) に欠損がある行の
    # 削除 (グループの写しは落とした列の欠損を残すので見ない. 時刻の写しは
    # 見る)
    columns = params.get("columns")
    subset = [c for c in frame.columns if c != "__group__"] if columns in (
        None, "auto"
    ) else (
        [columns] if isinstance(columns, str) else list(columns)
    )
    result = frame.dropna(subset=subset)

    return result


def apply_drop_duplicates(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    全く同じ行を落とす関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        columns (比べる列の名前の並び. 省略すると全ての列)
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        重複を落とした表

    Raises
    ----------
    ValueError
        無い列を指定した場合
    """
    # 処理の間だけ使う列を除いて比べ, 最初の行だけを残す
    columns = params.get("columns")
    if columns in (None, "auto", []):
        subset = [c for c in frame.columns if c not in HIDDEN]
    else:
        subset = [columns] if isinstance(columns, str) else list(columns)
        missing = [c for c in subset if c not in frame.columns]
        if missing:
            raise ValueError(f"列がありません: {', '.join(missing)}")
    result = frame.drop_duplicates(subset=subset)

    return result


def normal_text(value: str, lowercase: bool = False) -> str:
    """
    文字の表記のゆれをそろえる関数

    Parameters
    ----------
    value: str
        文字
    lowercase: bool = False
        小文字にするか

    Returns
    ----------
    text: str
        NFKC で全角の英数字と半角のかなをそろえ, 前後の空白を除いた文字
    """
    text = unicodedata.normalize("NFKC", value).strip()
    if lowercase:
        text = text.lower()

    return text


def apply_normalize_text(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    文字の列の表記のゆれ (全角と半角, 前後の空白, 大文字と小文字) を
    そろえる関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        columns, lowercase (小文字にするか, 既定 False)
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        そろえた表 (欠損はそのまま. 文字の型は保つ)

    Raises
    ----------
    ValueError
        数値か日時の列を指定した場合
    """
    # 欠損以外の値の表記をそろえる
    result = frame.copy()
    lower = bool(params.get("lowercase", False))
    for name in params["columns"]:
        column = result[name]
        if pd.api.types.is_numeric_dtype(column) or (
            pd.api.types.is_datetime64_any_dtype(column)
        ):
            raise ValueError(f"{name} は文字の列ではありません")
        result[name] = column.map(
            lambda v: v if pd.isna(v) else normal_text(str(v), lower)
        ).astype(column.dtype if pd.api.types.is_string_dtype(column)
                 and not pd.api.types.is_object_dtype(column) else object)

    return result


def apply_filter_rows(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    条件に合う行だけを残す関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        column, compare (==, !=, <, <=, >, >=, in, notna, isna), value
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        絞った表
    """
    # 任意の式は評価せず, 決まった比較だけを使う
    column = frame[params["column"]]
    op = params.get("compare", "==")
    value = params.get("value")
    comparisons = {
        "==": lambda: column == value, "!=": lambda: column != value,
        "<": lambda: column < value, "<=": lambda: column <= value,
        ">": lambda: column > value, ">=": lambda: column >= value,
        "in": lambda: column.isin(value or []),
        "notna": lambda: column.notna(), "isna": lambda: column.isna(),
    }
    if op not in comparisons:
        raise ValueError(f"知らない比較です: {op}")
    result = frame[comparisons[op]()]

    return result


def apply_log(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    列の値を log(1 + x) にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        columns
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        変換した表

    Raises
    ----------
    ValueError
        負の値がある場合
    """
    # 負の値は log1p にできない
    result = frame.copy()
    for name in resolve_columns(frame, params.get("columns"), {}):
        values = result[name].astype(float)
        if (values < 0).any():
            raise ValueError(f"{name} に負の値があるので対数を取れません")
        result[name] = np.log1p(values)

    return result


def apply_cast(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    列の型を変える関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        columns, to (float, int, str, category, datetime)
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        変換した表
    """
    # 決まった型だけに変える
    result = frame.copy()
    to = params.get("to", "float")
    for name in resolve_columns(frame, params.get("columns"), {}):
        if to == "datetime":
            result[name] = pd.to_datetime(result[name], errors="coerce")
        elif to == "int" and result[name].isna().any():
            # 欠損を持てる整数の型
            result[name] = result[name].astype("Int64")
        elif to in ("float", "int", "str", "category"):
            result[name] = result[name].astype(to)
        else:
            raise ValueError(f"知らない型です: {to}")

    return result


def apply_datetime_parts(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    日時の列を年, 月, 日, 曜日, 時の列に分ける関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    params: dict
        columns, parts (year, month, day, weekday, hour の並び),
        drop (元の列を落とすか, 既定 true)
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        列を足した表
    """
    # 日時の列ごとに部分の列を足す
    result = frame.copy()
    parts = params.get("parts", ["year", "month", "day", "weekday", "hour"])
    names = resolve_columns(frame, params.get("columns"), {}, "datetime")
    for name in names:
        values = pd.to_datetime(result[name])
        for part in parts:
            attribute = "day_of_week" if part == "weekday" else part
            result[f"{name}_{part}"] = getattr(values.dt, attribute)
        if params.get("drop", True):
            result = result.drop(columns=[name])

    return result


def apply_lag(
    frame: pd.DataFrame, params: dict, state: dict
) -> pd.DataFrame:
    """
    時系列の列を過去にずらした列と移動平均の列を足す関数

    Parameters
    ----------
    frame: pd.DataFrame
        時刻の順に並んだ表
    params: dict
        columns, lags (ずらし幅の並び), windows (移動平均の幅の並び),
        gap (移動平均をいくつ前から始めるか. 既定 1),
        by (系列を分ける列の並び. 店ごとの売上のように, 複数の系列が同じ
        時刻に並ぶ場合は, 系列ごとにずらす), time と freq と match (時刻で
        ずらす場合. apply_time_lag)
    state: dict
        使わない

    Returns
    ----------
    result: pd.DataFrame
        列を足した表 (始めの行は欠損になる. 時刻でずらす場合の移動平均は,
        期間にある値だけの平均で, 値が 1 つでもあれば欠損にならない)

    Raises
    ----------
    ValueError
        系列を分ける列が無い場合
    """
    # 未来の値を使わないように, 過去だけを見る (系列ごと)
    result = frame.copy()
    by = params.get("by") or []
    by = [by] if isinstance(by, str) else list(by)
    missing = [name for name in by if name not in frame.columns]
    if missing:
        raise ValueError(f"系列を分ける列がありません: {', '.join(missing)}")
    gap = window_gap(params)
    if params.get("time"):
        return apply_time_lag(result, params, by)
    for name in resolve_columns(frame, params.get("columns"), {}):
        if name in by:
            continue
        column = result[name]
        groups = column.groupby(
            [result[b] for b in by], dropna=False
        ) if by else None
        for lag in params.get("lags", [1]):
            result[f"{name}_lag{lag}"] = (
                groups.shift(int(lag)) if groups is not None
                else column.shift(int(lag))
            )
        for window in params.get("windows", []):
            mean = (
                groups.transform(
                    lambda s, w=int(window), g=gap: s.shift(g).rolling(
                        w
                    ).mean()
                ) if groups is not None
                else column.shift(gap).rolling(int(window)).mean()
            )
            result[mean_name(name, window, gap)] = mean

    return result


def window_gap(params: dict) -> int:
    """
    移動平均をいくつ前から始めるか (gap) を確かめて返す関数

    Parameters
    ----------
    params: dict
        lag の手順の引数

    Returns
    ----------
    gap: int
        1 以上の整数 (既定 1. 1 なら 1 つ前から)

    Raises
    ----------
    ValueError
        1 以上の整数でない場合
    """
    # 1 以上の整数の確認 (真偽値は除く)
    value = params.get("gap", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"gap は 1 以上の整数にしてください: {value}")
    gap = value

    return gap


def mean_name(name: str, window: Any, gap: int) -> str:
    """
    移動平均の列の名前を作る関数

    Parameters
    ----------
    name: str
        元の列の名前
    window: Any
        移動平均の幅
    gap: int
        いくつ前から始めるか

    Returns
    ----------
    column: str
        <列>_mean<幅> (gap が 1 でなければ <列>_mean<幅>_gap<gap>)
    """
    # gap が 1 なら gap の付かない名前
    column = f"{name}_mean{window}" + (f"_gap{gap}" if gap != 1 else "")

    return column


def window_means(
    rows: pd.DataFrame, base: pd.DataFrame, keys: list, time_key: str,
    columns: list, shift: pd.Timedelta, span: pd.Timedelta
) -> dict[str, np.ndarray]:
    """
    行ごとに, [時刻 - shift - span, 時刻 - shift) にある値の平均を求める
    関数

    Parameters
    ----------
    rows: pd.DataFrame
        時刻の順に並べた, 系列の鍵と時刻と値の表
    base: pd.DataFrame
        求める行の系列の鍵と時刻 (元の表の並び)
    keys: list
        系列の鍵の列 (無ければ空)
    time_key: str
        時刻の列
    columns: list
        平均を求める列
    shift: pd.Timedelta
        窓の終わりを前にずらす長さ
    span: pd.Timedelta
        窓の長さ

    Returns
    ----------
    means: dict[str, np.ndarray]
        列ごとの, 元の表の並びの平均 (窓に値が無ければ欠損)
    """
    # 系列ごとの累積和と値の数 (同じ時刻の行は全て入る)
    ordered = rows.reset_index(drop=True)
    totals = ordered[keys + [time_key]].copy()
    grouped = ordered.groupby(keys, dropna=False) if keys else None
    for name in columns:
        filled = ordered[name].fillna(0.0).astype(float)
        counted = ordered[name].notna().astype(float)
        totals[f"{name}__sum"] = (
            filled.groupby([ordered[k] for k in keys]).cumsum()
            if grouped is not None else filled.cumsum()
        )
        totals[f"{name}__count"] = (
            counted.groupby([ordered[k] for k in keys]).cumsum()
            if grouped is not None else counted.cumsum()
        )

    # 窓の両端より前の累積 (端の時刻ちょうどは含めない)
    edges = {}
    for edge, offset in (("hi", shift), ("lo", shift + span)):
        query = base[keys].copy()
        query["__at"] = base[time_key] - offset
        query["__row"] = np.arange(len(base))
        found = pd.merge_asof(
            query.dropna(subset=["__at"]).sort_values("__at"),
            totals, left_on="__at", right_on=time_key, by=keys or None,
            direction="backward", allow_exact_matches=False
        ).set_index("__row")
        edges[edge] = found.reindex(np.arange(len(base)))
    means = {}
    for name in columns:
        total = (edges["hi"][f"{name}__sum"].fillna(0.0)
                 - edges["lo"][f"{name}__sum"].fillna(0.0))
        count = (edges["hi"][f"{name}__count"].fillna(0.0)
                 - edges["lo"][f"{name}__count"].fillna(0.0))
        valid = (count > 0) & base[time_key].notna().to_numpy()
        means[name] = np.where(valid, total / count.where(count > 0), np.nan)

    return means


def apply_time_lag(
    frame: pd.DataFrame, params: dict, by: list
) -> pd.DataFrame:
    """
    時系列の列を, 行ではなく時刻でずらした列と移動平均の列を足す関数

    Parameters
    ----------
    frame: pd.DataFrame
        表 (並びは問わない. 返す表は元の並び)
    params: dict
        columns, lags, windows, gap, time (時刻の列), freq (1 つ分の間隔.
        1D, 1h など), match (previous はその時刻が無ければそれより前で
        最も近い値, exact は無ければ欠損. 既定 previous)
    by: list
        系列を分ける列の並び

    Returns
    ----------
    result: pd.DataFrame
        列を足した表

    Raises
    ----------
    ValueError
        時刻の列が無い, 日時でない, freq が読めない場合
    """
    # 時刻の列と間隔の確認
    time = str(params["time"])
    if time not in frame.columns:
        raise ValueError(f"時刻の列がありません: {time}")
    column = frame[time]
    times = column if pd.api.types.is_datetime64_any_dtype(column) else (
        pd.to_datetime(column, errors="coerce", format="mixed")
    )
    if times.notna().sum() == 0:
        raise ValueError(f"{time} は日時の列ではありません")
    try:
        step = pd.to_timedelta(params.get("freq", "1D"))
    except ValueError as error:
        raise ValueError(
            f"freq を間隔として読めません: {params.get('freq')} "
            "(1D, 1h, 30min など. 月や年のような長さの変わる間隔は不可)"
        ) from error
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("freq は正の間隔 (1D, 1h など) にしてください")
    match = params.get("match", "previous")
    if match not in ("previous", "exact"):
        raise ValueError(f"match は previous か exact にしてください: {match}")

    # 過去の値の表 (系列と時刻ごとに 1 つ. 重なれば後の行) と, 探す側の表
    result = frame.copy()
    # 内部の列の名前 (表の列と重ならない名前)
    taken = {str(name) for name in frame.columns}

    def fresh(name: str) -> str:
        while name in taken:
            name = f"_{name}"
        taken.add(name)
        return name

    time_key, row_key = fresh("__time"), fresh("__row")
    keys = [fresh(f"__by{k}") for k in range(len(by))]
    base = pd.DataFrame(
        {key: frame[name].astype(str).to_numpy() for key, name in zip(
            keys, by
        )}, index=frame.index
    )
    base[time_key] = times
    columns = [
        name for name in resolve_columns(frame, params.get("columns"), {})
        if name not in by and name != time
    ]
    text = [
        name for name in columns
        if not pd.api.types.is_numeric_dtype(frame[name])
    ]
    if text:
        raise ValueError(
            f"時刻でずらす列は数値にしてください: {', '.join(text)}"
        )
    rows = pd.concat([base, frame[columns]], axis=1).dropna(
        subset=[time_key]
    ).sort_values(time_key, kind="stable")
    known = rows.drop_duplicates(keys + [time_key], keep="last")
    for lag in params.get("lags", [1]):
        wanted = base.copy()
        wanted[time_key] = base[time_key] - step * int(lag)
        wanted[row_key] = np.arange(len(base))
        found = pd.merge_asof(
            wanted.dropna(subset=[time_key]).sort_values(time_key),
            known, on=time_key, by=keys or None, direction="backward",
            tolerance=pd.Timedelta(0) if match == "exact" else None
        ).set_index(row_key)
        for name in columns:
            values = np.full(len(base), np.nan)
            values[found.index.to_numpy()] = found[name].to_numpy(
                dtype=float, na_value=np.nan
            )
            result[f"{name}_lag{lag}"] = values

    # 移動平均は w × freq 前からその時刻の前まで (その時刻は含めない. 同じ
    # 時刻の行は全て平均に入れ, 期間にある値だけの平均). gap が g なら, 窓を
    # (g - 1) × freq だけ前にずらし, 累積和で窓の平均を直接求める (ずらした
    # 時刻に行が無い休みの日も, 窓に値があれば平均を出す)
    gap = window_gap(params)
    positions = base.assign(**{row_key: np.arange(len(base))})
    for window in params.get("windows", []):
        span = step * int(window)
        if gap > 1:
            means = window_means(rows, base, keys, time_key, columns,
                                 step * (gap - 1), span)
            for name in columns:
                result[mean_name(name, window, gap)] = means[name]
            continue
        indexed = rows.set_index(time_key)
        rolled = (
            indexed.groupby(keys, dropna=False)[columns] if keys
            else indexed[columns]
        ).rolling(span, closed="left").mean().reset_index()
        rolled = rolled.drop_duplicates(keys + [time_key], keep="last")
        merged = positions.merge(rolled, on=keys + [time_key], how="inner")
        for name in columns:
            values = np.full(len(base), np.nan)
            values[merged[row_key].to_numpy()] = merged[name].to_numpy(
                dtype=float, na_value=np.nan
            )
            result[mean_name(name, window, gap)] = values

    return result


def register_builtin_steps() -> None:
    """
    組み込みの前処理の手順を登録する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 列と行の手順
    columns = "列の名前の並び (auto で型から選ぶ)"
    register_step(Step(
        "drop_columns", None, apply_drop_columns,
        {"columns": "落とす列の名前の並び"}, "列を落とす"
    ))
    register_step(Step(
        "drop_missing", None, apply_drop_missing,
        {"columns": "見る列 (省略すると全て)"}, "欠損のある行を落とす",
        changes_rows=True
    ))
    register_step(Step(
        "drop_duplicates", None, apply_drop_duplicates,
        {"columns": "比べる列の名前の並び (省略すると全ての列)"},
        "同じ行 (columns の値が同じ行) を落とす", changes_rows=True
    ))
    register_step(Step(
        "normalize_text", None, apply_normalize_text,
        {"columns": columns, "lowercase": "小文字にするか (既定 false)"},
        "表記のゆれ (全角と半角, 前後の空白) をそろえる (NFKC)"
    ))
    register_step(Step(
        "filter_rows", None, apply_filter_rows,
        {"column": "列",
         "compare": "==, !=, <, <=, >, >=, in, notna, isna",
         "value": "比べる値"}, "条件に合う行だけを残す", changes_rows=True
    ))
    register_step(Step(
        "cast", None, apply_cast,
        {"columns": columns, "to": "float, int, str, category, datetime"},
        "列の型を変える"
    ))

    # 値を変える手順 (学習用のデータで当てはめるもの)
    register_step(Step(
        "impute", fit_impute, apply_impute,
        {"columns": columns,
         "strategy": "mean, median, most_frequent, constant",
         "value": "constant の値", "indicator": "欠損の印の列を足すか"},
        "欠損を埋める"
    ))
    register_step(Step(
        "clip", fit_clip, apply_clip,
        {"columns": columns, "factor": "四分位範囲の何倍の外を切るか"},
        "外れ値を切り詰める"
    ))
    register_step(Step(
        "log", None, apply_log, {"columns": columns},
        "log(1 + x) にする (偏った分布を整える)"
    ))
    register_step(Step(
        "standardize", fit_scale, apply_scale,
        {"columns": columns}, "平均 0, 標準偏差 1 にする"
    ))
    register_step(Step(
        "minmax", lambda f, p, r: fit_scale(f, {**p, "method": "minmax"}, r),
        apply_scale, {"columns": columns}, "最小 0, 最大 1 にする"
    ))
    register_step(Step(
        "one_hot", fit_one_hot, apply_one_hot,
        {"columns": columns, "max_categories": "列ごとのカテゴリの上限"},
        "カテゴリを 0 と 1 の列にする"
    ))
    register_step(Step(
        "label_encode", fit_label_encode, apply_label_encode,
        {"columns": columns}, "カテゴリを番号にする"
    ))
    register_step(Step(
        "datetime_parts", None, apply_datetime_parts,
        {"columns": columns, "parts": "year, month, day, weekday, hour",
         "drop": "元の列を落とすか"}, "日時を年, 月, 曜日などに分ける"
    ))
    register_step(Step(
        "lag", None, apply_lag,
        {"columns": columns, "lags": "ずらし幅の並び",
         "windows": "移動平均の幅の並び",
         "gap": "移動平均をいくつ前から始めるか (既定 1. 決める時点で 1 つ前が"
                "分からないなら 2)",
         "by": "系列を分ける列の並び (店ごとなど. 省略すると 1 つの系列)",
         "time": "時刻でずらす場合の時刻の列 (省略すると行でずらす)",
         "freq": "時刻でずらす 1 つ分の間隔 (1D, 1h など. 既定 1D)",
         "match": "previous (無ければ前で最も近い値) か exact"},
        "時系列の過去の値と移動平均の列を足す (by で系列ごと. time で時刻で)"
    ))


def to_plain(value: Any) -> Any:
    """
    numpy の値を YAML にできる Python の値に変換する関数

    Parameters
    ----------
    value: Any
        変換する値 (辞書とリストは中も変換)

    Returns
    ----------
    plain: Any
        Python の値
    """
    # 入れ物は中を順に変換
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    plain = value

    return plain


def check_step_params(index: int, step: dict) -> None:
    """
    手順の引数が, その手順の知っている引数だけかを確かめる関数

    知らない引数を黙って無視すると, 新しい版の ML-EDA で書いたレシピを
    古い版で読んだときに, 別の列を作って後の手順で分かりにくい誤りに
    なるため, 読んだ時点で止める

    Parameters
    ----------
    index: int
        手順の位置 (0 から)
    step: dict
        手順 (op と引数, 当てはめた値の state)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        知らない引数がある場合
    """
    # op と当てはめた値 (state) の他は, 登録した引数だけ
    known = set(STEPS[step["op"]].params) | STEP_KEYS
    unknown = sorted(str(key) for key in step if key not in known)
    if unknown:
        raise ValueError(
            f"手順 {index + 1} ({step['op']}) に知らない引数があります: "
            f"{', '.join(unknown)} (この版の ML-EDA が知る引数は "
            f"{', '.join(sorted(STEPS[step['op']].params)) or '無し'}. "
            "新しい版の ML-EDA で書いたレシピなら, ML-EDA を上げて"
            "ください)"
        )


def load_recipe(recipe: Union[str, Path, dict]) -> dict:
    """
    レシピを読み込み, 形を確かめる関数

    Parameters
    ----------
    recipe: Union[str, Path, dict]
        YAML のファイルか辞書

    Returns
    ----------
    loaded: dict
        版を確かめたレシピの写し

    Raises
    ----------
    ValueError
        新しい版, 知らない手順, 手順の知らない引数, 形の誤りの場合と, 目的
        の列の無いレシピで split.unlabeled が keep の場合
    """
    # ファイルなら読み込み
    if isinstance(recipe, (str, Path)):
        loaded = yaml.safe_load(Path(recipe).read_text(encoding="utf-8"))
    else:
        loaded = copy.deepcopy(recipe)
    if not isinstance(loaded, dict):
        raise ValueError("レシピは辞書 (YAML のマップ) にしてください")

    # 版 (書かなければ版 1), 手順, 分割の確認
    version = loaded.setdefault("format_version", 1)
    if loaded.get("steps") is None:
        loaded["steps"] = []
    if loaded.get("split") is None:
        loaded["split"] = {"method": "none"}
    if loaded.get("item_steps") is None:
        loaded["item_steps"] = []
    valid = isinstance(version, int) and not isinstance(version, bool)
    if not valid or not 1 <= version <= FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のレシピは, このライブラリ (版 "
            f"{FORMAT_VERSION} まで) では読めません"
        )
    loaded.setdefault("steps", [])
    loaded.setdefault("split", {"method": "none"})
    task = loaded.setdefault("task", "none")
    if task not in TASKS:
        raise ValueError(f"知らないタスクです: {task}")
    for index, step in enumerate(loaded["steps"]):
        if not isinstance(step, dict) or step.get("op") not in STEPS:
            raise ValueError(
                f"手順 {index + 1} の op が知らないものです: "
                f"{step.get('op') if isinstance(step, dict) else step}"
            )
        check_step_params(index, step)
    method = loaded["split"].get("method", "random")
    if method not in SPLIT_METHODS:
        raise ValueError(f"知らない分割のしかたです: {method}")
    unlabeled = loaded["split"].get("unlabeled", "drop")
    if unlabeled not in UNLABELED_MODES:
        raise ValueError(
            f"split.unlabeled は drop か keep にしてください: {unlabeled}"
        )

    # keep は目的の列を書いたレシピだけ (読み込んだ時点で分かるように.
    # 書かないと前と後の様子と書き出しの全てで誤りになるため)
    if unlabeled == "keep" and not loaded.get("target"):
        raise ValueError(
            "split.unlabeled: keep には目的の列が要ります (レシピに target "
            "を書く. 目的の列の欠けた行を残すため)"
        )

    return loaded


register_builtin_steps()
