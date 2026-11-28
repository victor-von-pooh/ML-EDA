import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.loaders import load_data
from ml_eda.profile import to_json
from ml_eda.trajectory import number_summary

# 列の役割 (要る 3 つと任意の反応の強さと見せただけの印)
REQUIRED_ROLES = ("user", "item", "time")
STRENGTH_ROLE = "reaction"
SHOWN_ROLE = "shown"

# 見せただけの印として読む文字 (前後の空白と大文字小文字を問わない)
SHOWN_TEXT = {"true": True, "1": True, "false": False, "0": False}

# 反応の数の分布の区切り (下の端. 最後は上の端が無い)
COUNT_BINS = (1, 2, 3, 6, 11, 21, 51, 101)

# 人気の偏りで上位とみなす品物の割合
TOP_ITEM_SHARE = 0.1

# 要約に並べる人気の品物の数
MAX_POPULAR_ITEMS = 10


@dataclass
class Reactions:
    # 反応の表 (user, item, time と, 強さがあれば reaction. 時刻の順)
    frame: pd.DataFrame
    # 役割から元の列の名前 (user, item, time, reaction, shown. 強さが
    # 無ければ reaction, 印が無ければ shown は None)
    columns: dict
    # 時刻の型 (number か datetime)
    time_kind: str
    # 読んだ行の数 (除く前)
    rows: int
    # 利用者, 品物か時刻の欠けで除いた行の数 (日時として読めない時刻も)
    dropped_missing: int
    # 反応の強さが 0 以下か数でないので除いた行の数
    dropped_not_positive: int
    # 欠けでない値のうち, 日時として読めなかった時刻の数
    unreadable_time: int = 0
    # 見せただけの記録の表 (user, item, time. 時刻の順. 印が無ければ None)
    shown: Optional[pd.DataFrame] = None
    # 見せただけの印を読めないので除いた行の数
    dropped_shown: int = 0
    # 選んだ反応と重なるので除いた見せただけの行の数
    dropped_conflict: int = 0
    # 読み込んだ場所と読み方 (load_data の Data.source)
    source: dict = field(default_factory=dict)
    # 読むときに気付いたこと (日本語の文)
    notes: list = field(default_factory=list)


def role_columns(
    user: Hashable, item: Hashable, time: Hashable,
    reaction: Optional[Hashable], shown: Optional[Hashable] = None
) -> dict:
    """
    役割と列の名前の対応を作り, 同じ列が 2 つの役割に無いかを確かめる関数

    Parameters
    ----------
    user: Hashable
        利用者の列
    item: Hashable
        品物の列
    time: Hashable
        時刻の列
    reaction: Optional[Hashable]
        反応の強さの列 (None は無し)
    shown: Optional[Hashable] = None
        見せただけの印の列 (None は無し)

    Returns
    ----------
    columns: dict
        役割から列の名前 (reaction と shown は None もある)

    Raises
    ----------
    ValueError
        列の名前が None の場合と, 同じ列を 2 つの役割に渡した場合
    """
    # 要る 3 つの列は名前が要る
    columns = {"user": user, "item": item, "time": time,
               STRENGTH_ROLE: reaction, SHOWN_ROLE: shown}
    absent = [role for role in REQUIRED_ROLES if columns[role] is None]
    if absent:
        raise ValueError(
            f"反応の表の {', '.join(absent)} の列の名前を渡してください"
        )

    # 同じ列は 1 つの役割だけ
    named = [name for name in columns.values() if name is not None]
    if len(set(named)) < len(named):
        raise ValueError(
            "反応の表の利用者, 品物, 時刻, 反応の強さ, 見せただけの印には, "
            "別々の列を渡してください"
        )

    return columns


def read_frame(
    source: Union[str, Path, Data], loader: Optional[str], options: dict
) -> tuple[pd.DataFrame, dict]:
    """
    反応の表を load_data で読むか, 読んだデータから表を取る関数

    Parameters
    ----------
    source: Union[str, Path, Data]
        反応の表のファイルか, load_data で読んだデータ
    loader: Optional[str]
        読み込みの名前 (None は推定)
    options: dict
        読み込みの設定

    Returns
    ----------
    frame: pd.DataFrame
        読んだ表
    place: dict
        読み込んだ場所と読み方 (Data.source)

    Raises
    ----------
    ValueError
        読んだデータに読み込みの名前か設定を渡した場合
    """
    # 読んだデータはそのまま使う (読み方は読んだときのもの)
    if isinstance(source, Data):
        if loader is not None or options:
            raise ValueError(
                "読んだデータ (Data) を渡す場合は, 読み込みの名前と設定を"
                "渡さないでください"
            )
        return source.frame, dict(source.source)
    data = load_data(source, loader=loader, **options)

    return data.frame, dict(data.source)


def read_time(series: pd.Series, name: Hashable) -> tuple[pd.Series, str]:
    """
    時刻の列を数か日時にする関数

    Parameters
    ----------
    series: pd.Series
        時刻の列
    name: Hashable
        元の列の名前 (誤りの文に使う)

    Returns
    ----------
    time: pd.Series
        数か日時の列 (日時として読めない値は欠け)
    kind: str
        number か datetime

    Raises
    ----------
    ValueError
        欠けでない値があり, どれも日時として読めない場合
    """
    # 数の列はそのまま
    if pd.api.types.is_numeric_dtype(series):
        return series, "number"

    # 他は日時として読み, 読めない値は欠け (書き方を推定できない警告は
    # 出さない)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0 and series.notna().any():
        raise ValueError(f"時刻の列 {name} を数か日時として読めません")

    return parsed, "datetime"


def shown_value(value: Any) -> Optional[bool]:
    """
    見せただけの印の値 1 つを真偽にする関数

    Parameters
    ----------
    value: Any
        印の列の値

    Returns
    ----------
    mark: Optional[bool]
        真 (見せただけ), 偽 (選んだ反応), 読めなければ None
    """
    # 真偽, 数の 1 と 0, 文字の true, false, 1, 0 (他は読めない)
    mark = None
    if isinstance(value, (bool, np.bool_)):
        mark = bool(value)
    elif isinstance(value, (int, np.integer, float, np.floating)):
        mark = {1: True, 0: False}.get(value) if value == value else None
    elif isinstance(value, str):
        mark = SHOWN_TEXT.get(value.strip().lower())

    return mark


def read_shown(series: pd.Series, name: Hashable) -> pd.Series:
    """
    見せただけの印の列を真偽にする関数

    Parameters
    ----------
    series: pd.Series
        印の列
    name: Hashable
        元の列の名前 (誤りの文に使う)

    Returns
    ----------
    marks: pd.Series
        真, 偽か, 読めない値は None の列 (object)

    Raises
    ----------
    ValueError
        欠けでない値があり, どれも印として読めない場合
    """
    # 値ごとに読み, 読めない値は None
    marks = series.map(shown_value).astype(object)
    if marks.notna().sum() == 0 and series.notna().any():
        raise ValueError(
            f"見せただけの印の列 {name} を true, false, 1, 0 として読めません"
        )

    return marks


def split_shown(
    table: pd.DataFrame, name: Hashable, notes: list
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    見せただけの印で, 反応の行と見せただけの行に分ける関数

    Parameters
    ----------
    table: pd.DataFrame
        欠けた行を除いた表 (shown の列を持つ)
    name: Hashable
        印の元の列の名前 (気付きの文に使う)
    notes: list
        気付きの並び (印を読めない行があれば足す)

    Returns
    ----------
    chosen: pd.DataFrame
        印が偽の行 (shown の列は除く)
    shown: pd.DataFrame
        印が真の行の user, item, time
    dropped: int
        印を読めないので除いた行の数
    """
    # 印を読めない行の除外と, 印の真偽での分け方
    marks = table[SHOWN_ROLE]
    readable = marks.notna()
    dropped = int((~readable).sum())
    if dropped:
        notes.append(
            f"見せただけの印 ({name}) を読めない {dropped} 行を除きました"
        )
    marked = np.array([mark is True for mark in marks], dtype=bool)
    chosen = table[readable.to_numpy() & ~marked].drop(columns=SHOWN_ROLE)
    shown = table[marked][list(REQUIRED_ROLES)]

    return chosen, shown, dropped


def drop_conflicts(
    shown: pd.DataFrame, chosen: pd.DataFrame, notes: list
) -> tuple[pd.DataFrame, int]:
    """
    選んだ反応と利用者, 品物, 時刻の全てが同じ見せただけの行を除く関数

    Parameters
    ----------
    shown: pd.DataFrame
        見せただけの行
    chosen: pd.DataFrame
        残した反応の表
    notes: list
        気付きの並び (除いた行があれば足す)

    Returns
    ----------
    kept: pd.DataFrame
        残した見せただけの行 (時刻の順)
    dropped: int
        除いた行の数
    """
    # 反応の利用者, 品物, 時刻の組に入る行の除外
    keys = list(REQUIRED_ROLES)
    taken = pd.MultiIndex.from_frame(chosen[keys].drop_duplicates())
    inside = pd.MultiIndex.from_frame(shown[keys]).isin(taken)
    dropped = int(inside.sum())
    if dropped:
        notes.append(
            f"選んだ反応と利用者, 品物, 時刻が同じ見せただけの {dropped} 行を"
            "除きました"
        )
    kept = shown[~inside].sort_values("time", kind="stable")
    kept = kept.reset_index(drop=True)

    return kept, dropped


def load_reactions(
    source: Union[str, Path, Data], user: Hashable, item: Hashable,
    time: Hashable, reaction: Optional[Hashable] = None,
    shown: Optional[Hashable] = None, loader: Optional[str] = None,
    **options: Any
) -> Reactions:
    """
    1 行が 1 つの反応の表を読み, 並べるの反応の表の約束を確かめる関数

    Parameters
    ----------
    source: Union[str, Path, Data]
        反応の表のファイルか, load_data で読んだデータ
    user: Hashable
        利用者の列
    item: Hashable
        品物の列
    time: Hashable
        時刻の列 (数か日時)
    reaction: Optional[Hashable] = None
        反応の強さの列, None の場合は全ての行を反応とみなす
    shown: Optional[Hashable] = None
        見せただけの印の列 (真は見せただけ, 偽は選んだ反応), None の
        場合は全ての行を選んだ反応とみなす
    loader: Optional[str] = None
        読み込みの名前, None の場合は推定 (load_data と同じ)
    **options: Any
        読み込みの設定 (load_data と同じ)

    Returns
    ----------
    reactions: Reactions
        残した反応の表 (時刻の順), 見せただけの記録の表 (印があれば),
        列の名前, 時刻の型, 除いた行の数, 気付き

    Raises
    ----------
    ValueError
        列の名前の誤り, 表に列が無い場合, 時刻が数でも日時でもない場合と,
        見せただけの印をどれも読めない場合
    """
    # 列の名前と表
    columns = role_columns(user, item, time, reaction, shown)
    frame, place = read_frame(source, loader, options)
    roles = [role for role, name in columns.items() if name is not None]
    missing = [
        str(columns[role]) for role in roles
        if columns[role] not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"反応の表に {', '.join(missing)} の列がありません. 利用者, "
            "品物, 時刻, 反応の強さ, 見せただけの印の列の名前を直して"
            "ください"
        )
    repeated = [
        str(columns[role]) for role in roles
        if (frame.columns == columns[role]).sum() > 1
    ]
    if repeated:
        raise ValueError(
            f"反応の表に同じ名前の列 {', '.join(repeated)} が 2 つ以上あります"
        )
    table = pd.DataFrame(
        {role: frame[columns[role]].reset_index(drop=True) for role in roles}
    )

    # 時刻は数か日時 (日時として読めない値は欠け)
    raw_time = table["time"]
    table["time"], time_kind = read_time(raw_time, columns["time"])
    unreadable = int((raw_time.notna() & table["time"].isna()).sum())

    # 見せただけの印は真偽 (読めない値は None)
    if SHOWN_ROLE in table:
        table[SHOWN_ROLE] = read_shown(table[SHOWN_ROLE], columns[SHOWN_ROLE])

    # 欠けた行, 印を読めない行と, 反応の強さが 0 以下か数でない行の除外.
    # 印が真の行は見せただけの記録へ
    notes = []
    rows = len(table)
    table = table.dropna(subset=list(REQUIRED_ROLES))
    dropped_missing = rows - len(table)
    if dropped_missing:
        notes.append(
            f"利用者, 品物か時刻の欠けた {dropped_missing} 行を除きました"
        )
    if unreadable:
        notes.append(
            f"時刻の列 {columns['time']} の {unreadable} 個の値を日時として"
            "読めず, 欠けとみなしました"
        )
    shown_table, dropped_shown = None, 0
    if SHOWN_ROLE in table:
        table, shown_table, dropped_shown = split_shown(
            table, columns[SHOWN_ROLE], notes
        )
    dropped_not_positive = 0
    if STRENGTH_ROLE in table:
        strength = pd.to_numeric(table[STRENGTH_ROLE], errors="coerce")
        kept = strength > 0
        dropped_not_positive = int((~kept).sum())
        if dropped_not_positive:
            notes.append(
                f"反応の強さ ({columns[STRENGTH_ROLE]}) が 0 以下か数でない "
                f"{dropped_not_positive} 行を除きました"
            )
        table = table[kept].assign(
            **{STRENGTH_ROLE: strength[kept].astype(float)}
        )
    table = table.sort_values("time", kind="stable").reset_index(drop=True)
    if table.empty:
        notes.append("反応として残る行がありません")

    # 残した反応と重なる見せただけの行の除外
    dropped_conflict = 0
    if shown_table is not None:
        shown_table, dropped_conflict = drop_conflicts(
            shown_table, table, notes
        )
    reactions = Reactions(
        frame=table, columns=columns, time_kind=time_kind, rows=rows,
        dropped_missing=dropped_missing,
        dropped_not_positive=dropped_not_positive,
        unreadable_time=unreadable, shown=shown_table,
        dropped_shown=dropped_shown, dropped_conflict=dropped_conflict,
        source=place, notes=notes
    )

    return reactions


def count_distribution(counts: np.ndarray) -> dict:
    """
    利用者か品物ごとの反応の数の分布をまとめる関数

    Parameters
    ----------
    counts: np.ndarray
        利用者か品物ごとの反応の数

    Returns
    ----------
    distribution: dict
        min, max, mean, std, median, once (1 回だけの数) と, bins
        (区切りごとの from, to (含む. 最後は None), count の並び)
    """
    # 数の要約と 1 回だけの数
    values = counts.astype(float)
    distribution = {**number_summary(values), "once": int((counts == 1).sum())}

    # 区切りごとの数
    bins = []
    for index, lower in enumerate(COUNT_BINS):
        upper = (
            COUNT_BINS[index + 1] - 1 if index + 1 < len(COUNT_BINS) else None
        )
        inside = counts >= lower
        if upper is not None:
            inside &= counts <= upper
        bins.append({"from": lower, "to": upper, "count": int(inside.sum())})
    distribution["bins"] = bins

    return distribution


def gini(counts: np.ndarray) -> Optional[float]:
    """
    品物ごとの反応の数の偏り (ジニ係数) を求める関数

    Parameters
    ----------
    counts: np.ndarray
        品物ごとの反応の数

    Returns
    ----------
    value: Optional[float]
        0 (どの品物も同じ数) から 1 に近いほど偏る. 品物が無ければ None
    """
    # 小さい順に並べた数の累積から
    if counts.size == 0:
        return None
    ordered = np.sort(counts.astype(float))
    ranks = np.arange(1, ordered.size + 1)
    value = float(
        (2 * ranks - ordered.size - 1).dot(ordered)
        / (ordered.size * ordered.sum())
    )

    return value


def json_value(value: Any) -> Any:
    """
    利用者か品物の値を JSON にできる値にする関数

    Parameters
    ----------
    value: Any
        表の値

    Returns
    ----------
    converted: Any
        int, 有限の float, bool, str のどれか (他は文字)
    """
    # 数と真偽と文字はそのまま, 他は文字
    if isinstance(value, (bool, np.bool_, int, np.integer, str)):
        return to_json(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(value):
        return float(value)
    converted = str(value)

    return converted


def describe_popularity(item_counts: pd.Series, total: int) -> dict:
    """
    品物の人気の偏りをまとめる関数

    Parameters
    ----------
    item_counts: pd.Series
        品物ごとの反応の数 (多い順)
    total: int
        反応の数

    Returns
    ----------
    popularity: dict
        top_items (上位の品物の数), top_share (上位の品物の反応の割合),
        gini, popular (多い順の item と reactions の並び)
    """
    # 上位の割合の品物 (少なくとも 1 つ) の反応の割合
    top = math.ceil(len(item_counts) * TOP_ITEM_SHARE)
    share = (
        float(item_counts.iloc[:top].sum() / total) if total else None
    )
    popularity = {
        "top_items": int(top),
        "top_share": share,
        "gini": gini(item_counts.to_numpy()),
        "popular": [
            {"item": json_value(name), "reactions": int(count)}
            for name, count in item_counts.iloc[:MAX_POPULAR_ITEMS].items()
        ],
    }

    return popularity


def describe_time(time: pd.Series, kind: str) -> dict:
    """
    反応の時刻の範囲をまとめる関数

    Parameters
    ----------
    time: pd.Series
        時刻の順の時刻の列
    kind: str
        number か datetime

    Returns
    ----------
    summary: dict
        kind, min, max (日時は ISO 8601 の文字), distinct (異なる時刻の数)
    """
    # 値の無い場合は範囲も None
    summary = {"kind": kind, "min": None, "max": None,
               "distinct": int(time.nunique())}
    if time.empty:
        return summary
    low, high = time.min(), time.max()
    if kind == "datetime":
        summary["min"], summary["max"] = low.isoformat(), high.isoformat()
    else:
        summary["min"], summary["max"] = float(low), float(high)

    return summary


def describe_shown(reactions: Reactions) -> Optional[dict]:
    """
    見せただけの記録の数と, 反応との結び付きをまとめる関数

    Parameters
    ----------
    reactions: Reactions
        load_reactions で読んだ反応

    Returns
    ----------
    summary: Optional[dict]
        negatives, users, items, presentations, chosen_presentations,
        unseen_items. 印の列が無ければ None
    """
    # 印の列が無い場合は要約無し
    shown, frame = reactions.shown, reactions.frame
    if shown is None:
        return None

    # 1 回の提示は利用者と時刻の組
    keys = ["user", "time"]
    chosen = frame[keys].drop_duplicates()
    every = pd.concat([chosen, shown[keys]]).drop_duplicates()
    summary = {
        "negatives": int(len(shown)),
        "users": int(shown["user"].nunique()),
        "items": int(shown["item"].nunique()),
        "presentations": int(len(every)),
        "chosen_presentations": int(len(chosen)),
        "unseen_items": int((~shown["item"].isin(frame["item"])).sum()),
    }

    return summary


def describe_reactions(reactions: Reactions) -> dict:
    """
    反応の表の利用者, 品物, 反応の数, 偏り, 時刻の範囲, 除いた行を
    まとめる関数

    Parameters
    ----------
    reactions: Reactions
        load_reactions で読んだ反応

    Returns
    ----------
    summary: dict
        JSON にできる辞書. 鍵は docs/DESIGN.md の「並べるの反応の表」
    """
    # 利用者と品物ごとの反応の数
    frame = reactions.frame
    total = int(len(frame))
    user_counts = frame["user"].value_counts(sort=False)
    item_counts = frame["item"].value_counts()
    users, items = int(len(user_counts)), int(len(item_counts))
    pairs = int(len(frame.drop_duplicates(["user", "item"])))

    # 要約
    strength = (
        number_summary(frame[STRENGTH_ROLE].to_numpy())
        if STRENGTH_ROLE in frame else None
    )
    summary = {
        "columns": {role: json_value(name) if name is not None else None
                    for role, name in reactions.columns.items()},
        "source": dict(reactions.source),
        "rows": int(reactions.rows),
        "reactions": total,
        "dropped": {
            "missing": int(reactions.dropped_missing),
            "unreadable_time": int(reactions.unreadable_time),
            "not_positive": int(reactions.dropped_not_positive),
            "unreadable_shown": int(reactions.dropped_shown),
            "conflict": int(reactions.dropped_conflict),
            "total": int(
                reactions.dropped_missing + reactions.dropped_not_positive
                + reactions.dropped_shown + reactions.dropped_conflict
            ),
        },
        "users": users,
        "items": items,
        "pairs": pairs,
        "repeated": total - pairs,
        "density": pairs / (users * items) if users and items else None,
        "per_user": count_distribution(user_counts.to_numpy()),
        "per_item": count_distribution(item_counts.to_numpy()),
        "popularity": describe_popularity(item_counts, total),
        "time": describe_time(frame["time"], reactions.time_kind),
        "reaction": strength,
        "shown": describe_shown(reactions),
        "notes": list(reactions.notes),
    }

    # 並べて確かめるには, 利用者と品物が 2 つ以上要る
    if total and (users < 2 or items < 2):
        summary["notes"].append(
            "利用者か品物が 1 つだけなので, 並べて確かめられません"
        )

    # 反応の表に無い品物の見せただけの行は, 骨組みが使えないので気付きに
    unseen = summary["shown"]["unseen_items"] if summary["shown"] else 0
    if unseen:
        summary["notes"].append(
            f"反応の表に無い品物の見せただけの行が {unseen} 行あり, 品物の"
            "番号が無いので負の例と評価に使えません"
        )
    summary = to_json(summary)

    return summary
