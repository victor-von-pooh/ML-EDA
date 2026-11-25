import datetime as dt
import os
import re
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ml_eda.data import Data
from ml_eda.itemprep import (
    apply_steps, check_item_steps, fit_item_steps, item_notes,
    run_item_steps, to_array
)
from ml_eda.loaders import within
from ml_eda.pinned import (
    drop_holder, entry_name, entry_state, make_holder, move_into,
    open_under
)
from ml_eda.profile import finding, to_json
from ml_eda.recipe import (
    HIDDEN, STEPS, WANT, category_key, check_recipe_steps, load_recipe,
    resolve_columns, to_plain
)

# データセットの説明の形式の版 (load_dataset が読む, 1 件 = 1 行の約束.
# ML-Inference が写しの確かめに使うので, 版 2 を足しても 1 のまま)
DATASET_FORMAT_VERSION = 1

# 入力と正解がファイルの組の約束の版 (items.load_items が読む)
ITEMS_FORMAT_VERSION = 2

# 版 2 に項目ごとの中身の手順 (entries.<名前>.steps) を足した版 (中身の
# 手順を渡した書き出しだけが書く)
ITEMS_STEPS_FORMAT_VERSION = 3

# load_items が読む版の組
ITEMS_FORMAT_VERSIONS = (ITEMS_FORMAT_VERSION, ITEMS_STEPS_FORMAT_VERSION)

# 分割を表す列と, 分割に使う日時とグループの写し (処理の間だけ使う)
SPLIT_COLUMN = "__split__"
TIME_COLUMN = "__time__"
ROW_COLUMN = "__row__"
GROUP_COLUMN = "__group__"
HIDDEN_COLUMNS = (SPLIT_COLUMN, TIME_COLUMN, ROW_COLUMN, GROUP_COLUMN)

# 重なりの例として挙げる値の数
OVERLAP_EXAMPLES = 3

# グループの層化で, 目的の値の種類がこれより多く, 行の半分より多ければ
# 層を作れない (回帰の値など) とする
STRATA_LIMIT = 20

# グループでまとめる分割のしかた (分割の件数が割合から外れうる)
GROUP_METHODS = ("group", "group_time", "stratified_group")

# 異常の検知で, レシピに id が無い場合の行の id の列 (読み込んだ表の
# 行の位置)
ANOMALY_ID = "row_id"

# 異常の検知で書き出せる種類
ANOMALY_KINDS = ("table", "timeseries")

# 異常の検知で, 検証用とテスト用に異常の行が無い場合の注意
NO_ANOMALY_NOTE = (
    "検証用とテスト用に異常の行がありません (上位 K 件の適合率と PR-AUC "
    "を数えられない. 異常の行を増やすか, valid か test の割合を見直す)"
)

# 目的の列があるのにタスクを決めていない場合の注意
NO_TASK_NOTE = (
    "タスクが決まっていません. 目的の列はそのままの値で書き出します"
)

# 分割の名前
SPLITS = ("train", "valid", "test")

# モデルの入力の名前 (版 1 の入力は tensors の特徴量 1 つ)
MODEL_INPUT = "x"

# ファイルの組の点群のモデルの入力と出す点の数の既定 (点の数がそろわない
# 項目の場合. ML-Quest の骨組みの config の data.input_points と
# data.target_points の既定と同じ. 学習で各件から選ぶ点の数)
ITEM_INPUT_POINTS = 512
ITEM_TARGET_POINTS = 1024

# 目的の列の欠けた行を残す分割の名前 (split.unlabeled: keep の場合)
UNLABELED = "unlabeled"

# 書き出しの source に添える, 元の場所の書き方
SOURCE_NOTE = (
    "元の場所は名前だけを残す (手元のディレクトリの構成を渡さないため)"
)

# 版 1 の書き出しの source から除く, 手元の場所の欄
LOCAL_PLACES = ("path", "root")

# 版 1 の書き出しの source に列の名前の並びを残す種類
TABLE_KINDS = ("table", "timeseries")


@dataclass
class Prepared:
    # 分割ごとの前処理した表 (目的の列を含む)
    frames: dict
    # 当てはめた値を含むレシピ
    recipe: dict
    # 学習に使う列 (目的の列を除く)
    features: list
    # 目的の列 (無ければ None)
    target: Optional[str]
    # 分類のクラス (番号の順)
    classes: list = field(default_factory=list)
    # 前処理の後の注意 (数値でない列が残っているなど)
    notes: list = field(default_factory=list)
    # 画像, 文, 点群, 音声の中身の手順をかけた, 分割ごとの配列
    arrays: dict = field(default_factory=dict)
    # 分割の後に分かる気付き (分割をまたぐグループや画像の重なり)
    findings: list = field(default_factory=list)
    # 異常の検知の説明 (行の id の列, 正解の有る無し, 分割ごとの行の数.
    # 異常の検知でなければ None)
    anomaly: Optional[dict] = None


def split_counts(count: int, valid: float, test: float) -> tuple[int, int]:
    """
    全体の件数から, 検証用とテスト用の件数を求める関数

    Parameters
    ----------
    count: int
        件数
    valid: float
        検証用の割合
    test: float
        テスト用の割合

    Returns
    ----------
    counts: tuple[int, int]
        検証用とテスト用の件数 (割合が 0 でなければ, 学習用を 1 件以上
        残せる限り 1 件以上)
    """
    # 四捨五入し, 割合があるのに 0 件になる場合は 1 件にする
    n_valid = int(round(count * valid))
    n_test = int(round(count * test))
    if valid > 0 and n_valid == 0 and count - n_test >= 2:
        n_valid = 1
    if test > 0 and n_test == 0 and count - n_valid >= 2:
        n_test = 1
    counts = (n_valid, n_test)

    return counts


def allocate(sizes: list[int], total: int) -> list[int]:
    """
    全体の件数を, グループの大きさに比例して配る関数 (最大剰余法)

    Parameters
    ----------
    sizes: list[int]
        グループごとの件数
    total: int
        配る件数

    Returns
    ----------
    shares: list[int]
        グループごとに配った件数 (合計は total, グループの件数を超えない)
    """
    # 比例した値の整数部を配り, 残りを端数の大きい順に配る
    whole = sum(sizes)
    if whole == 0 or total == 0:
        return [0] * len(sizes)
    exact = [size * total / whole for size in sizes]
    shares = [min(int(value), size) for value, size in zip(exact, sizes)]
    order = sorted(
        range(len(sizes)), key=lambda k: exact[k] - int(exact[k]),
        reverse=True
    )
    rest = total - sum(shares)
    for k in order * 2:
        if rest <= 0:
            break
        if shares[k] < sizes[k]:
            shares[k] += 1
            rest -= 1

    return shares


def assign_splits(
    frame: pd.DataFrame, split: dict, roles: dict
) -> pd.Series:
    """
    各行を学習用, 検証用, テスト用に割り当てる関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    split: dict
        method (random, stratified, time, group, group_time,
        stratified_group, none), valid, test (割合), seed, column
        (グループの列)
    roles: dict
        列の役割 (target と time)

    Returns
    ----------
    assigned: pd.Series
        行ごとの分割の名前

    Raises
    ----------
    ValueError
        分割に要る列が無い場合と, 割合の誤りの場合
    """
    # 割合の確認
    method = split.get("method", "random")
    valid = float(split.get("valid", 0.2))
    test = float(split.get("test", 0.0))
    if method == "none":
        return pd.Series("train", index=frame.index)
    if valid < 0 or test < 0 or valid + test >= 1:
        raise ValueError("valid と test の割合は 0 以上で, 合計を 1 未満に")
    rng = np.random.default_rng(int(split.get("seed", 0)))
    n_valid, n_test = split_counts(len(frame), valid, test)

    # 時刻の順は, 最後の部分を検証用とテスト用にする
    if method == "time":
        time = roles.get("time")
        if not time:
            raise ValueError("時刻の順の分割には日時の列が要ります")
        times = pd.to_datetime(frame[time], utc=True)
        order = np.argsort(times.to_numpy(), kind="stable")
        labels = label_by_position(frame.index, order, n_valid, n_test)
        return align_to_time(labels, times)

    # グループと時刻の組, グループの層化
    if method == "group_time":
        assigned = group_time_splits(frame, split, roles, n_valid, n_test)
        return assigned
    if method == "stratified_group":
        assigned = stratified_group_splits(
            frame, split, roles, n_valid, n_test, rng
        )
        return assigned

    # グループごとに同じ分割にする (行の件数が割合に近くなるまで加える)
    if method == "group":
        column = group_name(frame, split)
        keys = frame[column].astype(str).where(frame[column].notna(), None)
        sizes = keys.value_counts(dropna=False)
        groups = list(sizes.index)
        rng.shuffle(groups)
        labels = fill_groups(groups, sizes, n_valid, n_test)
        return keys.map(lambda key: labels.get(key, "train"))

    # 層化は, 全体の件数を目的の列の値ごとに比例して配る
    if method == "stratified":
        target = roles.get("target")
        if not target:
            raise ValueError("層化の分割には目的の列が要ります")
        groups = [
            group.index
            for _, group in frame.groupby(frame[target].astype(str))
        ]
        sizes = [len(index) for index in groups]
        valid_shares = allocate(sizes, n_valid)
        test_shares = allocate(
            [size - share for size, share in zip(sizes, valid_shares)],
            n_test
        )
        assigned = pd.Series("train", index=frame.index, dtype=object)
        for index, v_count, t_count in zip(groups, valid_shares, test_shares):
            order = rng.permutation(len(index))
            assigned.loc[index] = label_by_position(
                index, order, v_count, t_count
            )
        return assigned
    order = rng.permutation(len(frame))
    assigned = label_by_position(frame.index, order, n_valid, n_test)

    return assigned


def group_name(frame: pd.DataFrame, split: dict) -> str:
    """
    分割に使うグループの列の名前を, 表にあるかを確かめて返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    split: dict
        レシピの分割 (column がグループの列)

    Returns
    ----------
    column: str
        グループの列の名前

    Raises
    ----------
    ValueError
        グループの列を書いていない場合か, 表に無い場合
    """
    # 書いていない場合は書く案内, 表に無い場合は列の名前を挙げた誤り
    column = split.get("column")
    if not column:
        raise ValueError(
            "グループを使う分割には, グループの列 (split.column) を"
            "書いてください"
        )
    if column not in frame.columns:
        raise ValueError(f"グループの列 (split.column) がありません: {column}")

    return column


def group_codes(frame: pd.DataFrame, split: dict) -> np.ndarray:
    """
    グループの列の値を, グループの番号にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    split: dict
        レシピの分割 (column がグループの列)

    Returns
    ----------
    codes: np.ndarray
        行ごとのグループの番号 (欠損の行は 1 つのグループ)

    Raises
    ----------
    ValueError
        グループの列が無い場合
    """
    # 値は型に依らない文字にそろえ, 欠損も 1 つの番号にする
    column = group_name(frame, split)
    keys = frame[column].astype(str).where(frame[column].notna(), None)
    codes, _ = pd.factorize(keys, use_na_sentinel=False)

    return codes


def fill_groups(
    groups: list, sizes: pd.Series, n_valid: int, n_test: int
) -> dict:
    """
    並べたグループを, 割合の件数に収まる限りテスト用と検証用に入れる
    関数

    Parameters
    ----------
    groups: list
        グループの鍵 (この順に入れる)
    sizes: pd.Series
        グループの鍵ごとの行の数
    n_valid: int
        検証用の行の数
    n_test: int
        テスト用の行の数

    Returns
    ----------
    labels: dict
        グループの鍵から分割の名前 (収まらないグループは学習用)
    """
    # テスト用, 検証用の順に, 合うかを確かめて入れる
    labels: dict = {}
    wanted = {"test": n_test, "valid": n_valid}
    filled = {"test": 0, "valid": 0}
    for group in groups:
        size = int(sizes[group])
        labels[group] = "train"
        for name in ("test", "valid"):
            after = filled[name] + size
            if wanted[name] > 0 and after <= wanted[name]:
                labels[group] = name
                filled[name] = after
                break

    return labels


def group_time_splits(
    frame: pd.DataFrame, split: dict, roles: dict, n_valid: int,
    n_test: int
) -> pd.Series:
    """
    同じグループを 1 つの分割にまとめ, 現れた時刻の遅いグループを検証用
    とテスト用にする関数

    グループを最初の時刻 (同じなら値の文字) の順に並べ, 行を並べた位置で
    件数を切る. 境目をまたぐグループは, 時刻の順の分割の境目の時刻と
    同じく後ろの分割にそろえる

    Parameters
    ----------
    frame: pd.DataFrame
        表
    split: dict
        レシピの分割 (column がグループの列)
    roles: dict
        列の役割 (time)
    n_valid: int
        検証用の行の数
    n_test: int
        テスト用の行の数

    Returns
    ----------
    assigned: pd.Series
        行ごとの分割の名前

    Raises
    ----------
    ValueError
        グループの列か日時の列が無い場合
    """
    # グループの番号と, グループごとの最初の時刻と行の数
    time = roles.get("time")
    if not time or time not in frame.columns:
        raise ValueError(
            "グループと時刻の組の分割 (group_time) には日時の列が要ります"
        )
    codes = group_codes(frame, split)
    column = frame[split["column"]]
    times = pd.to_datetime(frame[time], utc=True).reset_index(drop=True)
    starts = times.groupby(codes).min()
    sizes = pd.Series(codes).value_counts()
    first = pd.Series(codes).drop_duplicates()
    names = column.iloc[first.index].astype(str).to_numpy()
    order = pd.DataFrame({
        "start": starts.reindex(first.to_numpy()).to_numpy(),
        "name": names, "code": first.to_numpy(),
    }).sort_values(["start", "name"], na_position="first", kind="stable")

    # 並べた位置の最後の行が入る分割 (時刻の読めないグループは先頭)
    count = len(frame)
    labels: dict = {}
    end = 0
    for code in order["code"]:
        end += int(sizes[code])
        if end > count - n_test:
            labels[code] = "test"
        elif end > count - n_test - n_valid:
            labels[code] = "valid"
        else:
            labels[code] = "train"
    assigned = pd.Series(
        [labels[code] for code in codes], index=frame.index, dtype=object
    )

    return assigned


def stratified_group_splits(
    frame: pd.DataFrame, split: dict, roles: dict, n_valid: int,
    n_test: int, rng: np.random.Generator
) -> pd.Series:
    """
    同じグループを 1 つの分割にまとめ, 目的の列の値の割合を分割の間で
    近くする関数

    グループの層は, そのグループの行で最も多い目的の値 (同じ数なら文字の
    小さい値). 検証用とテスト用の件数を層の行の数に比例して配り, 層ごと
    に無作為に並べたグループを, 入れると配った件数に近づく限り入れる
    (少し超えてもよい). 層で配り残した件数と超えた件数は次の層に回し,
    全体で足りなければ学習用のグループを足す

    Parameters
    ----------
    frame: pd.DataFrame
        表
    split: dict
        レシピの分割 (column がグループの列)
    roles: dict
        列の役割 (target)
    n_valid: int
        検証用の行の数
    n_test: int
        テスト用の行の数
    rng: np.random.Generator
        グループを並べる乱数

    Returns
    ----------
    assigned: pd.Series
        行ごとの分割の名前

    Raises
    ----------
    ValueError
        グループの列か目的の列が無い場合と, 目的の値がほぼ全て違う場合
    """
    # グループごとの層 (最も多い目的の値)
    target = roles.get("target")
    if not target:
        raise ValueError(
            "グループの層化の分割 (stratified_group) には目的の列が要ります"
        )
    codes = group_codes(frame, split)
    values = frame[target].astype(str).to_numpy()
    distinct = len(set(values))
    if distinct > STRATA_LIMIT and distinct > len(values) * 0.5:
        raise ValueError(
            f"目的の列 {target} の値がほぼ全て違う ({distinct} 種類) ので, "
            "グループの層を作れません (回帰の値なら group にしてください)"
        )
    table = pd.crosstab(codes, values)
    strata = table.idxmax(axis=1)
    sizes = pd.Series(codes).value_counts()

    # 層の行の数に比例して配り, 層ごとに入れる
    layers = sorted(strata.unique())
    members = [sorted(strata.index[strata == layer]) for layer in layers]
    rows = [int(sizes[group].sum()) for group in members]
    valid_shares = allocate(rows, n_valid)
    test_shares = allocate(
        [size - share for size, share in zip(rows, valid_shares)], n_test
    )
    labels = {group: "train" for group in strata.index}
    carry = {"test": 0, "valid": 0}
    for groups, v_count, t_count in zip(members, valid_shares, test_shares):
        groups = list(groups)
        rng.shuffle(groups)
        wanted = {"test": carry["test"] + t_count,
                  "valid": carry["valid"] + v_count}
        got = {"test": 0, "valid": 0}
        for group in groups:
            size = int(sizes[group])
            for name in ("test", "valid"):
                after = got[name] + size
                if abs(after - wanted[name]) < wanted[name] - got[name]:
                    labels[group] = name
                    got[name] = after
                    break
        carry = {name: wanted[name] - got[name] for name in carry}

    # 全体で足りなければ, 学習用のグループを全体の件数に近い順に足す
    for name, count in (("test", n_test), ("valid", n_valid)):
        top_up(labels, sizes, name, count)
    assigned = pd.Series(
        [labels[code] for code in codes], index=frame.index, dtype=object
    )

    return assigned


def top_up(labels: dict, sizes: pd.Series, name: str, count: int) -> None:
    """
    分割の行の数が全体の件数に届かない場合に, 学習用のグループを足す関数

    入れると件数に近づくグループを近い順に足す. 割合があるのに 0 件の
    場合は, 学習用にグループが残る限り, 件数に最も近いグループを 1 つ
    足す (split_counts の 1 件以上の決まりと同じ)

    Parameters
    ----------
    labels: dict
        グループの鍵から分割の名前 (書き換える)
    sizes: pd.Series
        グループの鍵ごとの行の数
    name: str
        足す分割の名前 (valid か test)
    count: int
        その分割の行の数

    Returns
    ----------
    None
    """
    # 学習用のグループを 1 つずつ, 件数に近づく限り
    while count > 0:
        filled = sum(int(sizes[g]) for g, n in labels.items() if n == name)
        rest = [g for g, n in labels.items() if n == "train"]
        if len(rest) < 2 or filled >= count:
            return
        best = min(rest, key=lambda g: (abs(filled + int(sizes[g]) - count),
                                        str(g)))
        after = filled + int(sizes[best])
        if filled and abs(after - count) >= count - filled:
            return
        labels[best] = name


def align_to_time(labels: pd.Series, times: pd.Series) -> pd.Series:
    """
    同じ時刻の行が分割をまたがないように, 分割の境目を時刻にそろえる関数

    店ごとの売上のように同じ時刻の行が複数ある場合, 件数で切ると境目の
    時刻が学習用と検証用の両方に入る (同じ日の他の店の値を学習に使う).
    境目の時刻の行は, 後ろの分割にそろえる

    Parameters
    ----------
    labels: pd.Series
        件数で決めた行ごとの分割の名前
    times: pd.Series
        行ごとの時刻 (欠損は件数で決めた分割のまま)

    Returns
    ----------
    aligned: pd.Series
        時刻ごとに 1 つの分割にした名前
    """
    # 時刻ごとに, その時刻の行の中で最も後ろの分割にそろえる
    rank = {"train": 0, "valid": 1, "test": 2}
    names = ["train", "valid", "test"]
    known = times.notna()
    level = labels.map(rank)
    latest = level[known].groupby(times[known]).transform("max")
    aligned = labels.copy()
    aligned[known] = latest.map(lambda k: names[int(k)])

    return aligned


def label_by_position(
    index: pd.Index, order: np.ndarray, n_valid: int, n_test: int
) -> pd.Series:
    """
    並べた順の最後の部分を検証用とテスト用にする関数

    Parameters
    ----------
    index: pd.Index
        行の名前
    order: np.ndarray
        行の位置の並べ方
    n_valid: int
        検証用の件数
    n_test: int
        テスト用の件数

    Returns
    ----------
    labels: pd.Series
        行ごとの分割の名前
    """
    # 並べた順の後ろからテスト用, その前を検証用に割り当てる
    count = len(order)
    names = np.array(["train"] * count, dtype=object)
    start = max(count - n_test - n_valid, 0)
    names[order[start:count - n_test]] = "valid"
    if n_test:
        names[order[count - n_test:]] = "test"
    labels = pd.Series(names, index=index)

    return labels


def anomaly_labels(values: pd.Series, target: str) -> pd.Series:
    """
    異常の検知の正解を, 異常は 1, 正常は 0 の整数にする関数

    Parameters
    ----------
    values: pd.Series
        欠けの無い正解の列の値
    target: str
        正解の列の名前 (誤りの文に使う)

    Returns
    ----------
    labels: pd.Series
        int64 の 0 と 1

    Raises
    ----------
    ValueError
        真偽値でも, 0 と 1 だけの数でもない場合
    """
    # 真偽値はそのまま, 数は 0 と 1 だけ
    if pd.api.types.is_bool_dtype(values):
        labels = values.astype("int64")
        return labels
    numeric = (
        pd.api.types.is_numeric_dtype(values)
        or values.map(lambda value: isinstance(value, bool)).all()
    )
    others = [] if not numeric else [
        value for value in values.unique().tolist()
        if value not in (0, 1)
    ]
    if not numeric or others:
        examples = (
            [str(value) for value in others[:3]] if numeric
            else values.astype(str).unique().tolist()[:3]
        )
        raise ValueError(
            f"異常の検知の正解の列 {target} は, 異常を 1 か真, 正常を 0 か偽"
            f"にしてください (他の値: {', '.join(examples)})"
        )
    labels = values.astype("int64")

    return labels


def anomaly_target(
    frame: pd.DataFrame, target: str, unlabeled: str
) -> pd.DataFrame:
    """
    異常の検知の正解を 0 と 1 にし, 欠けた行を落とすか正常と見なす関数

    Parameters
    ----------
    frame: pd.DataFrame
        読み込んだ表
    target: str
        正解の列
    unlabeled: str
        欠けた行の扱い (drop と keep は落とし, normal は正常と見なす)

    Returns
    ----------
    result: pd.DataFrame
        正解を int64 の 0 と 1 にした表

    Raises
    ----------
    ValueError
        正解の値が 0 と 1 (か真偽値) でない場合
    """
    # 欠けの無い値の確認
    labeled = frame[target].notna()
    labels = anomaly_labels(frame.loc[labeled, target], target)

    # 欠けた行は正常と見なすか, 落とす (keep の行は後で unlabeled に)
    if unlabeled == "normal":
        values = pd.Series(0, index=frame.index, dtype="int64")
        values.loc[labeled] = labels
        result = frame.copy()
    else:
        values = labels
        result = frame[labeled].copy()
    result[target] = values

    return result


def anomaly_counts(count: int, valid: float, test: float) -> tuple[int, int]:
    """
    異常の行の数を, 検証用とテスト用に割合の比で配る関数

    Parameters
    ----------
    count: int
        異常の行の数
    valid: float
        検証用の割合
    test: float
        テスト用の割合

    Returns
    ----------
    counts: tuple[int, int]
        検証用とテスト用の数 (合計は count. 両方の割合があり 2 件以上なら
        どちらも 1 件以上)

    Raises
    ----------
    ValueError
        異常の行があるのに, 検証用とテスト用の割合が共に 0 の場合
    """
    # 割合の比で四捨五入 (0.5 は切り上げ)
    if count and valid + test <= 0:
        raise ValueError(
            "異常の行を置く検証用かテスト用が要ります (valid か test の割合を"
            "0 より大きくする. 異常の行は学習用に入れないため)"
        )
    if not count:
        return 0, 0
    n_valid = int(np.floor(count * valid / (valid + test) + 0.5))
    if valid > 0 and test > 0 and count >= 2:
        n_valid = min(max(n_valid, 1), count - 1)
    counts = (n_valid, count - n_valid)

    return counts


def anomaly_splits(
    frame: pd.DataFrame, split: dict, target: Optional[str]
) -> pd.Series:
    """
    異常の検知の分割 (正常の行は無作為に分け, 異常の行は検証用とテスト用)
    をする関数

    Parameters
    ----------
    frame: pd.DataFrame
        正解を 0 と 1 にした表
    split: dict
        valid, test (割合), seed
    target: Optional[str]
        正解の列 (正解の無い表は None で, 全ての行を正常とする)

    Returns
    ----------
    assigned: pd.Series
        行ごとの分割の名前

    Raises
    ----------
    ValueError
        割合の誤りの場合と, 異常の行を置く分割が無い場合
    """
    # 割合の確認
    valid = float(split.get("valid", 0.2))
    test = float(split.get("test", 0.0))
    if valid < 0 or test < 0 or valid + test >= 1:
        raise ValueError("valid と test の割合は 0 以上で, 合計を 1 未満に")
    rng = np.random.default_rng(int(split.get("seed", 0)))
    anomalous = (
        (frame[target] == 1).to_numpy() if target
        else np.zeros(len(frame), dtype=bool)
    )

    # 正常の行は random と同じ分け方
    normal = frame.index[~anomalous]
    n_valid, n_test = split_counts(len(normal), valid, test)
    order = rng.permutation(len(normal))
    assigned = pd.Series("train", index=frame.index, dtype=object)
    assigned.loc[normal] = label_by_position(normal, order, n_valid, n_test)

    # 異常の行は検証用とテスト用だけ
    anomaly = frame.index[anomalous]
    a_valid, a_test = anomaly_counts(len(anomaly), valid, test)
    order = rng.permutation(len(anomaly))
    assigned.loc[anomaly] = label_by_position(anomaly, order, a_valid, a_test)

    return assigned


def anomaly_id(data: Data, recipe: dict) -> str:
    """
    異常の検知の行の id の列を確かめて, その名前を返す関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: dict
        読み込んだレシピ

    Returns
    ----------
    name: str
        行の id の列 (レシピに無ければ行の位置の row_id)

    Raises
    ----------
    ValueError
        書いた列が無いか, 欠けか同じ値がある場合と, id を書かずに表に
        row_id の列がある場合
    """
    # レシピに無ければ行の位置 (同じ名前の列があれば取り違えるので誤り)
    columns = [str(name) for name in data.frame.columns]
    name = recipe.get("id")
    if not name:
        if ANOMALY_ID in columns:
            raise ValueError(
                f"表に {ANOMALY_ID} の列があります. 行の id の列をレシピの id "
                "に書いてください"
            )
        return ANOMALY_ID

    # 書いた列は 1 つだけあり, 欠けも同じ値も無いこと
    if columns.count(name) != 1:
        raise ValueError(f"行の id の列 {name} がありません (か 2 つ以上)")
    values = data.frame.iloc[:, columns.index(name)]
    if values.isna().any():
        raise ValueError(
            f"行の id の列 {name} に欠けがあります ({int(values.isna().sum())}"
            " 行)"
        )
    if values.duplicated().any():
        repeated = values[values.duplicated()].astype(str).unique()[:3]
        raise ValueError(
            f"行の id の列 {name} に同じ値があります (例: "
            f"{', '.join(repeated)})"
        )

    return name


def id_values(data: Data, name: str, rows: list[int]) -> np.ndarray:
    """
    行の id の列の, 元の表での値を返す関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    name: str
        行の id の列 (row_id で表に無ければ行の位置)
    rows: list[int]
        元のデータでの行の位置

    Returns
    ----------
    values: np.ndarray
        行ごとの元の値
    """
    # 表の列の元の値か, 行の位置
    columns = [str(column) for column in data.frame.columns]
    if name not in columns:
        values = np.asarray(rows, dtype=np.int64)
        return values
    values = data.frame.iloc[list(rows), columns.index(name)].to_numpy()

    return values


def anomaly_summary(
    data: Data, recipe: dict, target: Optional[str], frames: dict,
    positions: dict
) -> dict:
    """
    異常の検知の説明 (行の id の列, 正解の有る無し, 分割ごとの行の数) を
    作る関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: dict
        読み込んだレシピ
    target: Optional[str]
        正解の列 (正解の無い表は None)
    frames: dict
        分割ごとの表
    positions: dict
        分割ごとの, 元のデータでの行の位置

    Returns
    ----------
    summary: dict
        id, labeled, unlabeled, rows (分割ごとの normal, presumed,
        anomaly, unlabeled)
    """
    # 元の表で正解の欠けた行 (正解の無い表は全ての行)
    columns = [str(name) for name in data.frame.columns]
    missing = (
        data.frame.iloc[:, columns.index(target)].isna().to_numpy()
        if target else np.ones(len(data.frame), dtype=bool)
    )

    # 分割ごとの行の数
    rows = {}
    for split, frame in frames.items():
        counts = {"normal": 0, "presumed": 0, "anomaly": 0, "unlabeled": 0}
        if split == UNLABELED:
            counts["unlabeled"] = int(len(frame))
        else:
            presumed = int(missing[positions[split]].sum())
            anomaly = int((frame[target] == 1).sum()) if target else 0
            counts["presumed"] = presumed
            counts["anomaly"] = anomaly
            counts["normal"] = int(len(frame)) - presumed - anomaly
        rows[split] = counts
    summary = {
        "id": str(recipe.get("id") or ANOMALY_ID),
        "labeled": bool(target),
        "unlabeled": str(recipe["split"].get("unlabeled", "drop")),
        "rows": rows,
    }

    return summary


def first_fit(steps: list) -> int:
    """
    値を学ぶ最初の手順の位置を返す関数

    Parameters
    ----------
    steps: list
        レシピの手順

    Returns
    ----------
    position: int
        最初の値を学ぶ手順の位置 (無ければ手順の数)
    """
    # それより前の手順は, 分割の前に全体にかけられる
    position = next(
        (k for k, spec in enumerate(steps) if STEPS[spec["op"]].fit),
        len(steps)
    )

    return position


def group_column(frame: pd.DataFrame, split: dict) -> Optional[str]:
    """
    レシピの分割に書いたグループの列の名前を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        手順をかける前の表 (列の名前は文字)
    split: dict
        レシピの分割 (column がグループの列)

    Returns
    ----------
    name: Optional[str]
        グループの列の名前 (書いていなければ None)

    Raises
    ----------
    ValueError
        書いた列が表に無い場合
    """
    # 空の文字は書いていないものと同じ扱い (画面で選び直した場合など)
    column = split.get("column")
    name = None if column is None or column == "" else str(column)
    if name is not None and name not in frame.columns:
        raise ValueError(f"グループの列 (split.column) がありません: {name}")

    return name


def count_overlap(keys: pd.Series, splits: pd.Series) -> dict:
    """
    同じ値が複数の分割にまたがる様子を数える関数

    Parameters
    ----------
    keys: pd.Series
        行ごとの値 (グループや画像のハッシュ. 欠損は数えない)
    splits: pd.Series
        行ごとの分割の名前 (keys と同じ並び)

    Returns
    ----------
    overlap: dict
        values (またがる値の数), rows (その値を持つ行の数), examples
        (またがる値の例)
    """
    # 値ごとの分割の数が 2 以上のものの集計
    known = keys.notna().to_numpy()
    frame = pd.DataFrame({
        "key": keys[known].astype(str).to_numpy(),
        "split": splits[known].to_numpy(),
    })
    spans = frame.groupby("key", sort=True)["split"].nunique()
    shared = spans[spans > 1].index
    overlap = {
        "values": int(len(shared)),
        "rows": int(frame["key"].isin(shared).sum()),
        "examples": [str(key) for key in shared[:OVERLAP_EXAMPLES]],
    }

    return overlap


def split_notes(split: dict, result: pd.DataFrame) -> list[str]:
    """
    グループでまとめた分割の件数が, 割合から大きく外れたことを注意にする
    関数

    Parameters
    ----------
    split: dict
        レシピの分割 (method, valid, test)
    result: pd.DataFrame
        分割の列を持つ, 手順をかけた表 (正解の付いた行)

    Returns
    ----------
    notes: list[str]
        0 件になった分割と, 割合の件数の半分より多く外れた分割の注意
    """
    # 割合を書いた分割だけ (割合の件数は split_counts と同じ決まり)
    notes: list[str] = []
    method = split.get("method", "random")
    if method not in GROUP_METHODS:
        return notes
    ratios = {"valid": float(split.get("valid", 0.2)),
              "test": float(split.get("test", 0.0))}
    wanted = dict(zip(
        ("valid", "test"),
        split_counts(len(result), ratios["valid"], ratios["test"])
    ))
    for name in ("valid", "test"):
        got = int((result[SPLIT_COLUMN] == name).sum())
        if ratios[name] <= 0:
            continue
        if got == 0:
            notes.append(
                f"{name} の割合を {ratios[name]} にしましたが, 0 件になりました "
                f"({method} の分割で, グループが大きく割合に収まらないか, "
                "件が少ないため). グループを細かくするか割合を上げてください"
            )
        elif abs(got - wanted[name]) > wanted[name] / 2:
            notes.append(
                f"{name} の行は {got} 行で, 割合の {wanted[name]} 行から大きく"
                f"外れました ({method} の分割で, 大きなグループが境目にある"
                "ため). グループを細かくするか割合を見直してください"
            )

    return notes


def overlap_findings(
    data: Data, recipe: dict, result: pd.DataFrame
) -> list[dict]:
    """
    分割をまたぐグループと画像の重なりを気付きにする関数

    確かめるのは元の行の値 (分割の前の手順で変える前). 同じ人や同じ
    画像が学習用と検証用の両方にあると, 新しい人や画像に当てはめる力を
    甘く測るため

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: dict
        読み込んだレシピ
    result: pd.DataFrame
        分割の列と元の行の位置の列を持つ, 手順をかけた表

    Returns
    ----------
    findings: list[dict]
        group_overlap と item_overlap の気付き
    """
    # 元の表の, 残った行の値と分割
    source = data.frame.copy(deep=False)
    source.columns = [str(name) for name in source.columns]
    rows = result[ROW_COLUMN].to_numpy()
    splits = result[SPLIT_COLUMN].reset_index(drop=True)
    method = recipe["split"].get("method", "random")
    findings = []
    if method == "none":
        return findings

    # グループの列の同じ値
    group = group_column(source, recipe["split"])
    if group:
        keys = source[group].iloc[rows].reset_index(drop=True)
        overlap = count_overlap(keys, splits)
        if overlap["values"]:
            usual = (
                "時刻の順の分割では, 同じものが過去と未来にまたがるのは普通"
                "です. " if method == "time" else ""
            )
            advice = {
                "time": "group_time", "stratified": "stratified_group"
            }.get(method, "group")
            findings.append(finding(
                "info" if method == "time" else "warning", "group_overlap",
                f"{group} の同じ値が {overlap['values']} 個 (計 "
                f"{overlap['rows']} 行) 複数の分割にまたがります (例: "
                f"{', '.join(overlap['examples'])}). 学習用にいる人や物を"
                "検証用で当てると, 新しい人や物に当てはめる力より甘く"
                f"測ります. {usual}新しい {group} に当てはめるなら, 分割を "
                f"{advice} (column: {group}) にしてください",
                [group]
            ))

    # 同じ画像 (ファイルの中身が同じ)
    if data.kind == "image" and "digest" in source.columns:
        keys = source["digest"].iloc[rows].reset_index(drop=True)
        overlap = count_overlap(keys, splits)
        if overlap["values"]:
            findings.append(finding(
                "warning", "item_overlap",
                f"同じ画像 (ファイルの中身が同じ) が {overlap['values']} "
                f"種類 (計 {overlap['rows']} 枚) 複数の分割にまたがります. "
                "検証用に学習用と同じ画像があると正解率が甘くなるので, "
                "分割の前に drop_duplicates (columns: [digest]) で落として"
                "ください",
                ["digest"]
            ))

    return findings


def run_step(
    frame: pd.DataFrame, spec: dict, roles: dict, number: int,
    fit_rows: Optional[pd.Series]
) -> pd.DataFrame:
    """
    1 つの手順の列を決め, 当てはめてからかける関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    spec: dict
        手順 (決めた列と当てはめた値を state に書き込む)
    roles: dict
        列の役割 (auto の列から目的の列などを除く)
    number: int
        手順の番号 (誤りの文に使う)
    fit_rows: Optional[pd.Series]
        当てはめに使う行 (学習用の行の印). None は全ての行

    Returns
    ----------
    result: pd.DataFrame
        手順をかけた表

    Raises
    ----------
    ValueError
        手順が失敗した場合 (番号と手順の名前を添える)
    """
    # auto の列を役割を除いて 1 度だけ決め, 推論のときも同じ列を使う
    step = STEPS[spec["op"]]
    params = {k: v for k, v in spec.items() if k not in ("op", "state")}
    try:
        if spec["op"] == "drop_columns" and params.get("columns") in (
            None, "auto"
        ):
            raise ValueError("落とす列の名前を columns に書いてください")
        if "columns" in step.params and spec["op"] not in (
            "drop_missing", "drop_duplicates"
        ):
            # 時系列の予測では目的の列の過去の値が最も大事な手がかりなので,
            # lag の auto は目的の列も選ぶ (ずらすので未来の値は使わない)
            chosen = {
                k: v for k, v in roles.items() if k != "target"
            } if spec["op"] == "lag" else roles
            params["columns"] = resolve_columns(
                frame, params.get("columns"), chosen,
                WANT.get(spec["op"], "numeric")
            )
            # 文の列は中身として読むので, 表の手順で書き換えても配列に
            # 届かない (中身の手順で変える)
            text = roles.get("text")
            if text and text in params["columns"] and not step.changes_rows:
                raise ValueError(
                    f"文の列 {text} は表の手順では変えられません. 中身の"
                    "手順 (item_steps の normalize_text など) を使ってください"
                )
        state: dict = {}
        if step.fit is not None:
            rows = frame if fit_rows is None else frame[fit_rows]
            if rows.empty:
                raise ValueError("当てはめる学習用の行がありません")
            state = to_plain(step.fit(rows, params, roles))
        if "columns" in params:
            state["columns"] = params["columns"]
        spec["state"] = state
        result = step.apply(frame, params, state)
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(
            f"手順 {number} ({spec['op']}) で失敗しました: {error}"
        ) from error

    return result


def apply_recipe(
    data: Data, recipe: dict, upto: Optional[int] = None
) -> tuple[pd.DataFrame, dict]:
    """
    分割と手順を順にかける関数

    値を学ぶ最初の手順より前の手順 (列や行を落とす, 過去の値の列など) は
    分割の前に全体にかける (分割をまたぐ重複も落とせる). それ以降は,
    値を学ぶ手順を学習用の行だけで当てはめる

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: dict
        読み込んだレシピ (当てはめた値を各手順の state に書き込む)
    upto: Optional[int] = None
        この数の手順までかける (None は全て)

    Returns
    ----------
    result: pd.DataFrame
        分割の列 (SPLIT_COLUMN) を持つ, 手順をかけた表
    roles: dict
        目的の列を含む列の役割
    """
    # 目的の列の欠損の行を落とす
    target = recipe.get("target") or data.roles.get("target")
    roles = {**data.roles, "target": target} if target else dict(data.roles)
    frame = data.frame.copy()
    frame.columns = [str(name) for name in frame.columns]
    frame[ROW_COLUMN] = np.arange(len(frame))
    anomaly = recipe["task"] == "anomaly"
    if target:
        if target not in frame.columns:
            raise ValueError(f"目的の列 {target} がありません")
        frame = (
            anomaly_target(
                frame, target, recipe["split"].get("unlabeled", "drop")
            )
            if anomaly else frame[frame[target].notna()]
        )
    if anomaly:
        roles["id"] = anomaly_id(data, recipe)
    steps = recipe["steps"] if upto is None else recipe["steps"][:upto]

    # 分割の前の手順, 分割, 分割の後の手順. 分割に使う日時とグループの
    # 列は, 手順で落としても使えるように別の列への写し
    time = roles.get("time")
    if time and time in frame.columns:
        frame[TIME_COLUMN] = frame[time]
    group = group_column(frame, recipe["split"])
    if group:
        frame[GROUP_COLUMN] = frame[group]
    head = first_fit(steps)
    for number, spec in enumerate(steps[:head], 1):
        frame = run_step(frame, spec, roles, number, None)
    if frame.empty:
        raise ValueError("手順の後に行が残りませんでした")
    split_roles = {**roles, "time": TIME_COLUMN} if time else roles
    split = recipe["split"]
    if group:
        # 手順で値を変えた (表記をそろえたなど) 列が残っていれば, その値
        if group in frame.columns:
            frame[GROUP_COLUMN] = frame[group]
        split = {**split, "column": GROUP_COLUMN}
    frame[SPLIT_COLUMN] = (
        anomaly_splits(frame, split, target) if anomaly
        else assign_splits(frame, split, split_roles)
    )
    frame = frame.drop(columns=[TIME_COLUMN, GROUP_COLUMN], errors="ignore")
    for number, spec in enumerate(steps[head:], head + 1):
        frame = run_step(
            frame, spec, roles, number, frame[SPLIT_COLUMN] == "train"
        )
    result = frame

    return result, roles


def sorted_classes(values: pd.Series) -> list[str]:
    """
    分類のクラスを, 元の値の順 (数なら数の順) に並べる関数

    Parameters
    ----------
    values: pd.Series
        目的の列

    Returns
    ----------
    classes: list[str]
        クラスの文字 (category_key の形)
    """
    # 数として並べられれば数の順, そうでなければ文字の順
    unique = values.dropna().unique().tolist()
    try:
        unique = sorted(unique, key=float)
    except (TypeError, ValueError):
        unique = sorted(unique, key=str)
    classes = list(dict.fromkeys(category_key(value) for value in unique))

    return classes


def unlabeled_split(
    data: Data, recipe: dict, target: str, features: list[str],
    columns: list[str]
) -> tuple[pd.DataFrame, list[int]]:
    """
    目的の列の欠けた行に, 学習用で当てはめた前処理をかける関数

    推論のとき (Dataset.transform) と同じかけ方で, 当てはめ直さない.
    行を絞る手順で落ちた行は除く

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: dict
        当てはめた値を含むレシピ
    target: str
        目的の列
    features: list[str]
        学習に使う列 (画像などは空)
    columns: list[str]
        学習用の分割の表の列 (同じ並びにそろえる)

    Returns
    ----------
    frame: pd.DataFrame
        学習に使う列は前処理の後の値, 他の列は元の値, 目的の列は欠損の表
    rows: list[int]
        残した行の元のデータでの位置

    Raises
    ----------
    ValueError
        前処理をかけられない場合
    """
    # 目的の列の欠けた行の位置
    source = data.frame.copy(deep=False)
    source.columns = [str(name) for name in source.columns]
    missing = source[target].isna().to_numpy()
    if not missing.any():
        frame = pd.DataFrame(columns=columns)
        rows: list[int] = []
        return frame, rows

    # 推論と同じかけ方 (過去の値の列の前後関係も推論と同じ)
    reader = Dataset(
        path=Path("."), recipe=recipe, frames={},
        info={
            "kind": data.kind, "task": recipe["task"], "target": target,
            "features": [{"name": name} for name in features],
        }
    )
    try:
        applied = reader.transform(source)
    except ValueError as error:
        raise ValueError(
            f"目的の列の欠けた行 ({UNLABELED}) の前処理で失敗しました: "
            f"{error}"
        ) from error
    dropped = set(applied.attrs.get("dropped", {}))
    rows = [
        int(row) for row in np.flatnonzero(missing) if int(row) not in dropped
    ]

    # 学習用の表と同じ列 (前処理の値, 元の値, 欠損の順に選ぶ)
    values = {}
    for name in columns:
        if name == target:
            values[name] = np.full(len(rows), np.nan)
        elif name in applied.columns:
            values[name] = applied[name].iloc[rows].reset_index(drop=True)
        elif name in source.columns:
            values[name] = source[name].iloc[rows].reset_index(drop=True)
        else:
            values[name] = np.full(len(rows), np.nan)
    frame = pd.DataFrame(values, index=range(len(rows)), columns=columns)

    return frame, rows


def prepare(
    data: Data, recipe: Union[str, Path, dict],
    item_dir: Optional[Path] = None, read_items: bool = True
) -> Prepared:
    """
    レシピで分割と前処理をして, 学習用の表にする関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: Union[str, Path, dict]
        レシピ
    item_dir: Optional[Path] = None
        画像などの配列を書くディレクトリ (指定しなければメモリに置く)
    read_items: bool = True
        画像などの中身を読んで配列にするか. False なら中身を読まずに,
        手順の確認と注意だけをする (画面の前と後の様子に使う)

    Returns
    ----------
    prepared: Prepared
        分割ごとの表, 当てはめたレシピ, 学習に使う列, クラス, 分割を
        またぐ重なりの気付き

    Raises
    ----------
    ValueError
        手順の失敗, 目的の列が手順で消えた場合, 学習用の行が無い場合,
        回帰の目的の列が数値でない場合
    """
    # 多視点の撮影は, 今は読み込みと分析だけ
    if data.kind == "capture":
        raise ValueError(
            "多視点の撮影は, 今は読み込みと分析だけです (前処理と書き出し"
            "は, 多視点の学習の正解の形が決まってから)"
        )

    # 分割と手順 (異常の検知は表と時系列だけ)
    recipe = load_recipe(recipe)
    anomaly = recipe["task"] == "anomaly"
    if anomaly and data.kind not in ANOMALY_KINDS:
        raise ValueError(
            f"異常の検知 (task: anomaly) は表と時系列だけです: {data.kind}"
        )
    result, roles = apply_recipe(data, recipe)
    target = roles.get("target")
    if target and target not in result.columns:
        raise ValueError(f"目的の列 {target} が手順で消えました")
    if not (result[SPLIT_COLUMN] == "train").any():
        raise ValueError("学習用の行がありません (割合か絞り込みを見直す)")
    keep = recipe["split"].get("unlabeled", "drop") == "keep"
    if keep and not target:
        raise ValueError(
            "split.unlabeled: keep には目的の列が要ります (目的の列の欠けた"
            "行を残すため)"
        )

    # 目的の列の形 (分類は番号, 回帰は数値)
    classes: list = []
    if target and recipe["task"] == "classification":
        classes = sorted_classes(result[target])
        index = {value: number for number, value in enumerate(classes)}
        result[target] = result[target].map(category_key).map(index)
        result[target] = result[target].astype(int)
    elif target and recipe["task"] in ("regression", "forecasting"):
        if not pd.api.types.is_numeric_dtype(result[target]):
            raise ValueError(
                f"回帰の目的の列 {target} が数値ではありません "
                f"({result[target].dtype}). タスクを分類にするか, 型を変える"
            )

    # 異常の検知の行の id (手順で落としても変えても元の値)
    if anomaly:
        result[roles["id"]] = id_values(
            data, roles["id"], result[ROW_COLUMN].tolist()
        )
    names = [n for n in SPLITS if (result[SPLIT_COLUMN] == n).any()]
    positions = {
        n: result.loc[result[SPLIT_COLUMN] == n, ROW_COLUMN].tolist()
        for n in names
    }
    frames = {
        n: result[result[SPLIT_COLUMN] == n].drop(
            columns=[SPLIT_COLUMN, ROW_COLUMN]
        ).reset_index(drop=True)
        for n in names
    }

    # 分割をまたぐグループと画像の重なり (文は注意にも足す)
    findings = overlap_findings(data, recipe, result)
    overlaps = [item["message"] for item in findings]
    overlaps += split_notes(recipe["split"], result)

    # 学習に使う列 (画像などは中身の配列を使うので無し. 役割の列を除く)
    reserved = {target, roles.get("id"), roles.get("time"), SPLIT_COLUMN}
    features = [] if data.item_loader is not None else [
        name for name in result.columns
        if name not in reserved and name not in HIDDEN_COLUMNS
    ]

    # 目的の列の欠けた行の, 学習用で当てはめた値での unlabeled の分割
    if keep:
        extra, extra_rows = unlabeled_split(
            data, recipe, target, features, list(frames["train"].columns)
        )
        if extra_rows and anomaly:
            extra[roles["id"]] = id_values(data, roles["id"], extra_rows)
        if extra_rows:
            frames[UNLABELED] = extra
            positions[UNLABELED] = extra_rows

    # 画像, 文, 点群, 音声は中身の手順で配列にする (表の列は使わない)
    arrays: dict = {}
    if data.item_loader is not None:
        check_item_steps(recipe["item_steps"], data.kind)
        rows = data.frame.iloc[result[ROW_COLUMN].tolist()]
        notes = item_notes(rows, recipe["item_steps"], data.kind)
        notes.extend(overlaps)
        if target and recipe["task"] == "none":
            notes.append(NO_TASK_NOTE)
        if not read_items:
            return Prepared(
                frames=frames, recipe=recipe, features=[], target=target,
                classes=classes, notes=notes, findings=findings
            )
        arrays = run_item_steps(
            item_reader(data), positions.get("train", []), positions,
            recipe["item_steps"], data.kind, item_dir,
            int(recipe["split"].get("seed", 0))
        )
        prepared = Prepared(
            frames=frames, recipe=recipe, features=[], target=target,
            classes=classes, notes=notes, arrays=arrays, findings=findings
        )
        return prepared

    # 数値でない列の注意
    notes = []
    for name in features:
        if not pd.api.types.is_numeric_dtype(result[name]):
            notes.append(
                f"{name} は数値ではありません ({result[name].dtype}). "
                "学習の前に符号化 (one_hot, label_encode) か日時の分解を"
                "してください"
            )
        elif result[name].isna().any():
            notes.append(f"{name} に欠損が残っています (impute で埋める)")
        elif UNLABELED in frames and frames[UNLABELED][name].isna().any():
            notes.append(
                f"{name} の欠損が {UNLABELED} の分割にだけ残っています "
                "(impute で埋める)"
            )
    notes.extend(overlaps)
    if target and recipe["task"] == "none":
        notes.append(NO_TASK_NOTE)

    # 異常の検知の説明と, 検証用とテスト用に異常の行が無い注意
    summary = None
    if anomaly:
        summary = anomaly_summary(data, recipe, target, frames, positions)
        found = sum(
            summary["rows"].get(name, {}).get("anomaly", 0)
            for name in ("valid", "test")
        )
        if target and not found:
            notes.append(NO_ANOMALY_NOTE)
    prepared = Prepared(
        frames=frames, recipe=recipe, features=features, target=target,
        classes=classes, notes=notes, findings=findings, anomaly=summary
    )

    return prepared


def preview_items(
    data: Data, recipe: Union[str, Path, dict], count: int = 8,
    fit_limit: int = 100, seed: Optional[int] = None
) -> dict:
    """
    画像, 文, 点群, 音声の中身の手順を, 学習用の見本で当てはめて, 数件に
    かけた前と後を返す関数 (画面で手順の効き方を見るため)

    Parameters
    ----------
    data: Data
        中身を持つデータ
    recipe: Union[str, Path, dict]
        レシピ
    count: int = 8
        前と後を返す件数 (学習用から無作為に選ぶ)
    fit_limit: int = 100
        当てはめに使う学習用の件数の上限 (書き出しより少なくして速くする)
    seed: Optional[int] = None
        当てはめの見本と件を選ぶ乱数のシード (None なら分割の seed.
        書き出しと同じ見本の選び方)

    Returns
    ----------
    result: dict
        items (index, label, before (元の中身), after (配列)), steps
        (当てはめた中身の手順), fit_rows (当てはめに使った件数)

    Raises
    ----------
    ValueError
        中身を持たない種類, 手順の誤り, 目的の列が手順で消えた場合,
        学習用の件が無い場合, 件数が負か当てはめの件数が 1 未満の場合
    """
    # 引数と手順の確認, 表の手順と分割による学習用の位置の決定
    if data.item_loader is None:
        raise ValueError("中身を持たない種類です (画像, 文, 点群, 音声で使う)")
    if count < 0 or fit_limit < 1:
        raise ValueError("count は 0 以上, fit_limit は 1 以上にしてください")
    loaded = load_recipe(recipe)
    check_item_steps(loaded["item_steps"], data.kind)
    frame, roles = apply_recipe(data, loaded)
    target = roles.get("target")
    if target and target not in frame.columns:
        raise ValueError(f"目的の列 {target} が手順で消えました")
    if seed is None:
        seed = int(loaded["split"].get("seed", 0))
    train = frame.loc[
        frame[SPLIT_COLUMN] == "train", ROW_COLUMN
    ].astype(int).tolist()
    if not train:
        raise ValueError("学習用の件がありません (割合か絞り込みを見直す)")

    # 少ない見本での当てはめと, 無作為に選んだ件への適用
    read = item_reader(data)
    steps = loaded["item_steps"]
    fit_item_steps(read, train, steps, seed, fit_limit, fit_limit)
    rng = np.random.default_rng(seed)
    chosen = sorted(
        rng.choice(train, min(count, len(train)), replace=False).tolist()
    )
    target = target or roles.get("label")
    items = []
    for index in chosen:
        before = read(index)
        label = data.frame.iloc[index].get(target) if target else None
        items.append({
            "index": int(index),
            "label": None if label is None else str(label),
            "before": before,
            "after": to_array(apply_steps(before, steps), data.kind),
        })
    result = {
        "items": items, "steps": steps,
        "fit_rows": min(len(train), fit_limit),
    }

    return result


def item_reader(data: Data) -> Any:
    """
    中身を読む関数を, 読めない件の場所と直し方を添えた誤りにする関数

    Parameters
    ----------
    data: Data
        中身を持つデータ

    Returns
    ----------
    read: Any
        位置から中身を読む関数
    """
    def read(index: int) -> Any:
        """
        1 件の中身を読む関数

        Parameters
        ----------
        index: int
            元のデータの位置

        Returns
        ----------
        item: Any
            中身
        """
        # 読めない件はファイルの場所を添えて, 除き方を案内
        try:
            item = data.load_item(index)
        except Exception as error:
            row = data.frame.iloc[index]
            where = row.get("path", row.get("file", index))
            raise ValueError(
                f"{where} を読めません ({error}). 表の手順 filter_rows で, "
                "読めない件 (画像は width が欠損, 点群と音声は error がある"
                "件) を除いてください"
            ) from error

        return item

    return read


def preview(
    data: Data, recipe: Union[str, Path, dict], upto: int, rows: int = 20
) -> dict:
    """
    手順を 1 つずつかけた前と後の表の様子を返す関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: Union[str, Path, dict]
        レシピ
    upto: int
        この数の手順までかけた後を見る (前は upto - 1 まで)
    rows: int = 20
        返す行の数

    Returns
    ----------
    result: dict
        前と後の行数, 列, 先頭の行, 足した列と消えた列

    Raises
    ----------
    ValueError
        upto が手順の数の範囲の外の場合
    """
    # 手順を upto - 1 と upto までかける
    loaded = load_recipe(recipe)
    if not 0 <= upto <= len(loaded["steps"]):
        raise ValueError(f"upto は 0 から {len(loaded['steps'])} にして")
    before, _ = apply_recipe(data, load_recipe(recipe), max(upto - 1, 0))
    after, _ = apply_recipe(data, load_recipe(recipe), upto)

    # 列の差と先頭の行
    before = before.drop(columns=[SPLIT_COLUMN, ROW_COLUMN])
    after = after.drop(columns=[SPLIT_COLUMN, ROW_COLUMN])
    result = {
        "before": frame_view(before, rows),
        "after": frame_view(after, rows),
        "added": [c for c in after.columns if c not in before.columns],
        "removed": [c for c in before.columns if c not in after.columns],
    }
    result = to_json(result)

    return result


def frame_view(frame: pd.DataFrame, rows: int) -> dict:
    """
    表の行数, 列と型, 先頭の行を JSON にできる形にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    rows: int
        返す行の数

    Returns
    ----------
    view: dict
        行数 (rows), 列の名前と型 (columns の name と dtype), 先頭の行
        (head. 値は int, 有限の float, bool, str か None)
    """
    # 先頭の行を JSON にできる値にする (入れ物は文字, 欠損と無限大は None)
    head = frame.head(rows)
    records = [
        [
            str(value) if isinstance(value, (list, dict, np.ndarray))
            else None if pd.isna(value) else (
                value if isinstance(
                    value, (int, float, bool, np.number, np.bool_)
                )
                else str(value)
            )
            for value in row
        ]
        for row in head.itertuples(index=False)
    ]
    view = {
        "rows": int(len(frame)),
        "columns": [
            {"name": str(name), "dtype": str(dtype)}
            for name, dtype in frame.dtypes.items()
        ],
        "head": records,
    }
    view = to_json(view)

    return view


def write_frame(frame: pd.DataFrame, path: Path) -> str:
    """
    表を parquet (無ければ csv) で保存する関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    path: Path
        拡張子を除いた保存先

    Returns
    ----------
    name: str
        保存したファイルの名前
    """
    # pyarrow があれば型を保てる parquet
    try:
        target = path.with_suffix(".parquet")
        frame.to_parquet(target, index=False)
    except ImportError:
        target = path.with_suffix(".csv")
        frame.to_csv(target, index=False)
    name = target.name

    return name


def start_writing(
    path: Union[str, Path], overwrite: bool, parent: Optional[int] = None
) -> tuple[Path, Path]:
    """
    データセットを書き出す一時ディレクトリを作る関数

    版 1 と版 2 の書き出しで共通. 一時ディレクトリに書いてから
    finish_writing で入れ替える. parent を渡すと, 一時ディレクトリは
    途中の場所を含まない手元の一時の場所に作り, 既にある場所は開いた親
    に対してリンクをたどらずに確かめる

    Parameters
    ----------
    path: Union[str, Path]
        書き出すディレクトリ (parent を渡すと最後の部分が親の中の名前.
        途中は使わないので, 開いた親と合うことは使う側が確かめる)
    overwrite: bool
        既にあるディレクトリを置き換えるか
    parent: Optional[int] = None
        使う側が開いた, 書き出す場所の親のディレクトリの記述子

    Returns
    ----------
    places: tuple[Path, Path]
        書き出す場所と, 一時ディレクトリ

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合と, データセット
        ではない (dataset.yaml の無い) ディレクトリを置き換えようとした場合
    ValueError
        parent を渡し, 最後の部分が親の中の名前にできない場合
    """
    # 開いた親に書く場合 (途中の場所は使わない)
    final = Path(path).expanduser()
    if parent is not None:
        name = entry_name(final)
        state = entry_state(parent, name)
        if state is not None and not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if state == "other":
            raise FileExistsError(
                f"{final} はデータセットではないので置き換えません"
            )
        places = (final, make_holder(name))
        return places

    # 置き換えられるのは, 前に書き出したデータセットだけ
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} は既にあります")
        if not (final / "dataset.yaml").is_file():
            raise FileExistsError(
                f"{final} はデータセットではないので置き換えません"
            )

    # 前の失敗で残った一時ディレクトリは消して作り直す
    target = final.with_name(f".{final.name}.writing")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    places = (final, target)

    return places


def drop_writing(target: Path, parent: Optional[int] = None) -> None:
    """
    書き出しに失敗した一時ディレクトリを消す関数

    Parameters
    ----------
    target: Path
        start_writing が返した一時ディレクトリ
    parent: Optional[int] = None
        start_writing に渡した親の記述子 (渡していれば手元の一時の場所
        ごと消す)

    Returns
    ----------
    None
    """
    # 開いた親に書く場合は手元の一時の場所ごとの消去
    if parent is not None:
        drop_holder(target)
        return
    shutil.rmtree(target, ignore_errors=True)


def finish_writing(
    target: Path, final: Path, info: dict, parent: Optional[int] = None
) -> None:
    """
    説明を書き, 一時ディレクトリを書き出す場所に入れ替える関数

    Parameters
    ----------
    target: Path
        一時ディレクトリ
    final: Path
        書き出す場所
    info: dict
        dataset.yaml の中身
    parent: Optional[int] = None
        start_writing に渡した親の記述子 (渡していれば, 親に対してリンク
        をたどらずに入れ替える)

    Returns
    ----------
    None

    Raises
    ----------
    FileExistsError
        parent を渡し, 書き出す場所がデータセットではないものに替わった
        場合
    """
    # 説明は最後に書き, 前のデータセットと入れ替える
    (target / "dataset.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )
    if parent is not None:
        move_into(target, parent, entry_name(final), final)
        return
    if final.exists():
        shutil.rmtree(final)
    target.rename(final)


def source_name(place: Union[str, Path]) -> str:
    """
    説明の source に書く, 元の場所の名前を返す関数

    手元のディレクトリの構成を残さないため, 最後の部分だけにする

    Parameters
    ----------
    place: Union[str, Path]
        元の場所 (ディレクトリかファイル)

    Returns
    ----------
    name: str
        場所の名前 (. や .. は畳んだ先の名前. ~ だけと根は空)
    """
    # . と .. だけを畳んで最後の部分 (シンボリックリンクはたどらず, 渡した
    # 名前のまま. ~ だけや根は手元の名前になるので空)
    text = str(place)
    if text.strip().rstrip("/\\") in ("~", ""):
        name = ""
        return name
    name = Path(os.path.abspath(os.path.expanduser(text))).name

    return name


def shared_source(data: Data) -> dict:
    """
    版 1 の書き出しの source に書く, 手元の場所を除いた写しを返す関数

    データセットは人に渡すことがあるので, 読み込んだ場所 (path) と 1 件の
    場所の基 (root) は名前 (name) だけにする. 表と時系列は, 読み込んだ表の
    列の名前の並び (columns) を添える

    Parameters
    ----------
    data: Data
        読み込んだデータ

    Returns
    ----------
    shared: dict
        path と root を除き, 場所があれば name, paths (name_only), note を,
        表と時系列なら columns を添えた写し
    """
    # 場所の名前は読み込んだ場所 (無ければ 1 件の場所の基) の最後の部分
    source = data.source
    shared = {
        key: value for key, value in source.items()
        if key not in LOCAL_PLACES
    }
    place = next(
        (source[key] for key in LOCAL_PLACES if source.get(key)), None
    )
    if place is not None:
        shared = {
            "name": source_name(place), **shared,
            "paths": "name_only", "note": SOURCE_NOTE,
        }

    # 読む側が元のデータの列を知るための, 列の名前だけ (値は残さない)
    if data.kind in TABLE_KINDS:
        shared["columns"] = [str(name) for name in data.frame.columns]

    return shared


def export_dataset(
    data: Data, recipe: Union[str, Path, dict], path: Union[str, Path],
    name: Optional[str] = None, overwrite: bool = False,
    parent: Optional[int] = None
) -> dict:
    """
    前処理して分割したデータを, 学習用のディレクトリに書き出す関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: Union[str, Path, dict]
        レシピ
    path: Union[str, Path]
        書き出すディレクトリ
    name: Optional[str] = None
        データセットの名前 (省略するとディレクトリの名前)
    overwrite: bool = False
        既にあるディレクトリを置き換えるか
    parent: Optional[int] = None
        使う側が開いた, 書き出す場所の親のディレクトリの記述子 (渡すと
        path の最後の部分を親の中の名前とし, 途中の場所をたどらずに書く.
        path の途中は使わないので, 開いた親と合うことは使う側が確かめる.
        閉じるのは使う側)

    Returns
    ----------
    info: dict
        書き出した dataset.yaml の中身

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合と, データセット
        ではない (dataset.yaml の無い) ディレクトリを置き換えようとした場合
    ValueError
        parent を渡し, path の最後の部分が親の中の名前にできない場合
    """
    # 一時ディレクトリで前処理 (中身の配列も 1 件ずつここに書く) を
    # 済ませてから入れ替える. 失敗したら一時ディレクトリを消す
    final, target = start_writing(path, overwrite, parent)
    try:
        prepared = prepare(data, recipe, item_dir=target)
        splits = {}
        for split, frame in prepared.frames.items():
            splits[split] = {
                "file": write_frame(frame, target / split),
                "rows": int(len(frame)),
            }
            if split in prepared.arrays:
                splits[split]["items"] = f"{split}_items.npy"
                splits[split]["shape"] = list(prepared.arrays[split].shape)
        (target / "recipe.yaml").write_text(
            yaml.safe_dump(to_plain(prepared.recipe), allow_unicode=True,
                           sort_keys=False),
            encoding="utf-8"
        )

        # 説明
        first = next(iter(prepared.frames.values()))
        reference = prepared.frames.get("train", first)
        info = {
            "format_version": DATASET_FORMAT_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "kind": data.kind,
            "task": prepared.recipe["task"],
            "target": prepared.target,
            "features": [
                {"name": str(c), "dtype": str(reference[c].dtype)}
                for c in prepared.features
            ],
            "classes": prepared.classes,
            "text_column": (
                data.roles.get("text") if data.kind == "text" else None
            ),
            "splits": splits,
            "source": to_plain(shared_source(data)),
            "recipe": "recipe.yaml",
            "notes": prepared.notes,
        }
        if prepared.anomaly is not None:
            info["anomaly"] = to_plain(prepared.anomaly)
        finish_writing(target, final, info, parent)
    except Exception:
        drop_writing(target, parent)
        raise

    return info


def restore_rows(
    frame: pd.DataFrame, count: int, copies: dict[int, int],
    dropped: dict[int, str]
) -> pd.DataFrame:
    """
    行を落とした表を, 入力の行の並びに戻す関数

    Parameters
    ----------
    frame: pd.DataFrame
        行の名前が入力の行の位置の表
    count: int
        入力の行の数
    copies: dict[int, int]
        重複として落とした行から残した行
    dropped: dict[int, str]
        落とした行から手順の名前

    Returns
    ----------
    result: pd.DataFrame
        入力と同じ行の数の表 (重複は残した行の値, 落とした行は欠損).
        attrs["dropped"] に落とした行の位置から手順の名前
    """
    # 重複した行は残した行の値 (残した行が後で落ちたら同じく落とす)
    source = np.arange(count)
    lost = dict(dropped)
    for row in sorted(copies):
        kept = copies[row]
        while kept in copies:
            kept = copies[kept]
        source[row] = kept
        if kept in lost:
            lost[row] = lost[kept]
    result = frame.reindex(source).reset_index(drop=True)
    result.attrs["dropped"] = lost

    return result


def without_column(state: dict, name: str) -> dict:
    """
    当てはめた値から 1 つの列を除く関数

    Parameters
    ----------
    state: dict
        手順の当てはめた値 (columns と, 列ごとの値の辞書)
    name: str
        除く列の名前

    Returns
    ----------
    result: dict
        name を除いた写し
    """
    # 列の並びと, 列の名前を鍵にした辞書 (values, categories, bounds)
    result = {}
    for key, value in state.items():
        if key == "columns":
            result[key] = [c for c in value if c != name]
        elif isinstance(value, dict):
            result[key] = {k: v for k, v in value.items() if k != name}
        else:
            result[key] = value

    return result


@dataclass
class Dataset:
    # 書き出したディレクトリ
    path: Path
    # dataset.yaml の中身
    info: dict
    # 分割ごとの表
    frames: dict
    # 当てはめた値を含むレシピ
    recipe: dict

    @property
    def features(self) -> list[str]:
        """
        学習に使う列の名前の並びを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        names: list[str]
            列の名前
        """
        # 説明の features の名前
        names = [item["name"] for item in self.info["features"]]

        return names

    @property
    def target(self) -> Optional[str]:
        """
        目的の列の名前を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        target: Optional[str]
            目的の列 (無ければ None)
        """
        # 説明の target
        target = self.info.get("target")

        return target

    def arrays(self, split: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        分割の特徴量と目的の値を numpy の配列で返す関数

        Parameters
        ----------
        split: str
            train, valid, test, unlabeled のどれか

        Returns
        ----------
        x: np.ndarray
            (件数, 列の数) の float32 の配列
        y: Optional[np.ndarray]
            目的の値 (分類は int64 の番号, 回帰は float32). 目的の列が
            無い場合と, 目的の値が全て欠けた分割 (unlabeled) は None
        """
        # 特徴量 (画像などは中身の配列) と目的の値を取り出す
        frame = self.frames[split]
        items = self.info["splits"][split].get("items")
        x = (
            np.load(self.path / items, allow_pickle=False, mmap_mode="r")
            if items
            else frame[self.features].to_numpy(dtype=np.float32, copy=True)
        )
        y = None
        labeled = (
            bool(self.target) and self.target in frame.columns
            and bool(frame[self.target].notna().any())
        )
        if labeled:
            dtype = (
                np.int64
                if self.info["task"] in ("classification", "anomaly")
                else np.float32
            )
            # 書き込める配列にする (PyTorch のテンソルにそのまま渡せるように)
            y = frame[self.target].to_numpy(dtype=dtype, copy=True)

        return x, y

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        新しいデータ (推論のときなど) に, 当てはめた前処理をかける関数

        Parameters
        ----------
        frame: pd.DataFrame
            元のデータと同じ列を持つ表

        Returns
        ----------
        result: pd.DataFrame
            学習に使う列だけの表 (当てはめた値をそのまま使う). 行は入力と
            同じ並び (行の名前は 0 からの位置) で, 行を落とす手順で落ちた
            行は欠損. 落ちた行と手順の名前は result.attrs["dropped"] (行の
            位置から手順の名前)

        Raises
        ----------
        ValueError
            同じ名前の列がある場合と, 前処理に要る列が無い場合
        """
        # 同じ名前の列は, どちらを使うか決められない
        names = [str(name) for name in frame.columns]
        twins = sorted({name for name in names if names.count(name) > 1})
        if twins:
            raise ValueError(
                f"同じ名前の列があります: {', '.join(twins)}. 列の名前を"
                "変えてから渡してください"
            )
        frame = frame.copy()
        frame.columns = names
        frame = frame.reset_index(drop=True)

        # 過去の値の列が目的の列を読む場合, 学習のときは目的の列が分かる
        # 行だけで前後関係を作った. 分かる行は分かる行だけでかけ, 分からない
        # 行 (先の予測など) は全ての行でかけた値を使う
        target = self.target
        reads = bool(target) and target in frame.columns and any(
            spec["op"] == "lag"
            and target in spec.get("state", {}).get("columns", [])
            for spec in self.recipe["steps"]
        )
        known = frame[target].notna().to_numpy() if reads else None
        if known is None or known.all() or not known.any():
            result = self.apply_steps(frame)
            return result
        everything = self.apply_steps(frame)
        part = self.apply_steps(frame[known].reset_index(drop=True))
        rows = np.flatnonzero(known)
        result = everything.copy()
        result.iloc[rows] = part.to_numpy()
        lost = {
            row: op for row, op in everything.attrs["dropped"].items()
            if not known[row]
        }
        lost.update({
            int(rows[row]): op for row, op in part.attrs["dropped"].items()
        })
        result.attrs["dropped"] = lost

        return result

    def apply_steps(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        行の名前が 0 からの位置の表に, 当てはめた前処理をかける関数

        Parameters
        ----------
        frame: pd.DataFrame
            元のデータと同じ列を持つ表

        Returns
        ----------
        result: pd.DataFrame
            学習に使う列だけの表 (当てはめた値をそのまま使う). 行は入力と
            同じ並びで, 行を落とす手順で落ちた行は欠損. 落ちた行と手順の
            名前は result.attrs["dropped"] (行の位置から手順の名前)
        """
        # 当てはめ直さずに, 保存した値と列での手順の適用. 行を落とす手順は,
        # 過去の値の列の前後関係を学習のときとそろえるためにかけ, 最後に
        # 入力の行の並びに戻す (重複した行は残した行と同じ値, 落とした行は
        # 欠損). 目的の列を書き換える手順は,
        # 過去の値の列 (lag) が目的の列を読み, 目的の列 (過去の値) を
        # 渡された場合は学習のときと同じく目的の列にもかけ (lag が同じ
        # 値を読むように), それ以外は目的の列だけを除いてかける (同じ
        # 手順で変えた特徴量は学習のときと同じ値に)
        result = frame.copy()
        count = len(result)
        copies: dict[int, int] = {}
        dropped: dict[int, str] = {}
        target = self.target
        keep_target = bool(target) and target in result.columns and any(
            spec["op"] == "lag"
            and target in spec.get("state", {}).get("columns", [])
            for spec in self.recipe["steps"]
        )
        for spec in self.recipe["steps"]:
            step = STEPS[spec["op"]]
            state = spec.get("state", {})
            columns = state.get("columns", [])
            reads_only = spec["op"] == "lag"
            if step.changes_rows:
                result = self.drop_rows(result, spec, copies, dropped)
                continue
            dropped_target = False
            if target and target in columns and not (
                reads_only or keep_target
            ):
                state = without_column(state, target)
                columns = state["columns"]
                dropped_target = True
                if not columns:
                    # 目的の列だけの手順も, 欠損の印の列は学習のときと同じ 0
                    if spec.get("indicator"):
                        result[f"{target}_missing"] = 0
                    continue
            params = {
                k: v for k, v in spec.items() if k not in ("op", "state")
            }
            if "columns" in state:
                params["columns"] = [c for c in columns if c in result]
                # 当てはめた列が無ければ, 途中で落ちる前に分かる誤りに
                # (目的の列の過去の値の列は, 目的の列 (過去の値) が要る)
                # (落とす列は入力に無くてよい)
                absent = [c for c in columns if c not in result]
                if absent and not reads_only and spec["op"] != "drop_columns":
                    lagged = re.compile(
                        rf"{re.escape(str(self.target))}_(lag|mean)\d+"
                    )
                    hint = (
                        f". 目的の列 {self.target} の過去の値から作る列は, "
                        f"{self.target} の過去の値を含めて渡してください"
                        if self.target and any(
                            lagged.fullmatch(c) for c in absent
                        ) else ""
                    )
                    raise ValueError(
                        f"{spec['op']} にかける列がありません: "
                        f"{', '.join(absent)}{hint}"
                    )
            result = step.apply(result, params, state)
            # 除いた目的の列の欠損の印は, 学習のときと同じ 0 (目的の列が
            # 欠損した行は学習に使わない)
            if dropped_target and spec.get("indicator"):
                result[f"{target}_missing"] = 0
        # 学習に使う列が足りない場合は, 黙って落とさずに誤りにする
        missing = [c for c in self.features if c not in result.columns]
        if missing:
            raise ValueError(
                f"前処理の後に, 学習に使う列が足りません: {', '.join(missing)}"
                " (元のデータと同じ列を渡してください. カンマ付きの数などは "
                "load_data で読んだ後の形にそろえてください)"
            )
        result = restore_rows(result[self.features], count, copies, dropped)

        return result

    def drop_rows(
        self, frame: pd.DataFrame, spec: dict, copies: dict[int, int],
        dropped: dict[int, str]
    ) -> pd.DataFrame:
        """
        推論のときに行を落とす手順をかけ, 落とした行を覚える関数

        Parameters
        ----------
        frame: pd.DataFrame
            表 (行の名前は入力の行の位置)
        spec: dict
            行を落とす手順
        copies: dict[int, int]
            重複として落とした行から残した行 (書き足す)
        dropped: dict[int, str]
            落とした行から手順の名前 (書き足す)

        Returns
        ----------
        result: pd.DataFrame
            行を落とした表

        Raises
        ----------
        ValueError
            絞る列が入力に無い場合
        """
        # 重複は残した行を覚える. 欠損と条件は目的の列を見ない (推論の
        # ときは目的の列が分からない行を予測するため)
        op = spec["op"]
        params = {k: v for k, v in spec.items() if k not in ("op", "state")}
        target = self.target
        if op == "drop_duplicates":
            columns = params.get("columns")
            subset = [c for c in frame.columns if c not in HIDDEN] if (
                columns in (None, "auto", [])
            ) else [c for c in ([columns] if isinstance(columns, str)
                                else columns) if c in frame.columns]
            if not subset:
                return frame
            codes = frame.groupby(subset, dropna=False, sort=False).ngroup()
            first = pd.Series(frame.index, index=frame.index).groupby(
                codes
            ).transform("first")
            twin = frame.duplicated(subset=subset, keep="first")
            copies.update({int(r): int(first[r]) for r in frame.index[twin]})
            result = frame[~twin]
            return result
        if op == "drop_missing":
            columns = params.get("columns")
            names = [c for c in frame.columns if c not in HIDDEN] if (
                columns in (None, "auto")
            ) else ([columns] if isinstance(columns, str) else list(columns))
            subset = [c for c in names if c in frame.columns and c != target]
            lost = frame[subset].isna().any(axis=1)
        else:
            column = params.get("column")
            if column not in frame.columns:
                if column == target:
                    return frame
                raise ValueError(f"{op} にかける列がありません: {column}")
            kept = STEPS[op].apply(frame, params, spec.get("state", {}))
            lost = pd.Series(~frame.index.isin(kept.index), index=frame.index)
            # 目的の列で絞る場合, 目的の列が分からない行は残す
            if column == target:
                lost &= frame[column].notna()
        dropped.update({int(r): op for r in frame.index[lost]})
        result = frame[~lost]

        return result

    def transform_item(self, item: Any) -> np.ndarray:
        """
        新しい画像, 文, 点群, 音声に, 当てはめた中身の手順をかける関数

        Parameters
        ----------
        item: Any
            中身 (画像は配列, 文は文字, 点群は points を持つ辞書, 音声は
            read_audio の辞書. 画像と音声はファイルの場所も)

        Returns
        ----------
        array: np.ndarray
            学習のときと同じ形の配列
        """
        # 画像のファイルの場所は学習のときと同じ読み方で読み, 保存した値で
        # 手順をかけて配列にする
        if isinstance(item, (str, Path)) and self.info["kind"] == "image":
            from ml_eda.media import read_image

            item = read_image(Path(item))
        if isinstance(item, (str, Path)) and self.info["kind"] == "audio":
            from ml_eda.audio import read_audio

            item = read_audio(Path(item))
        result = apply_steps(item, self.recipe.get("item_steps", []))
        array = to_array(result, self.info["kind"])

        return array

    def tensors(self, split: str) -> tuple:
        """
        分割の特徴量と目的の値を PyTorch のテンソルで返す関数

        Parameters
        ----------
        split: str
            train, valid, test, unlabeled のどれか

        Returns
        ----------
        tensors: tuple
            (特徴量, 目的の値) のテンソル (目的の値が無ければ特徴量だけ.
            unlabeled の分割も)
        """
        # PyTorch は使うときだけ読み込む
        import torch

        x, y = self.arrays(split)
        # ファイルから読んだ配列は書き込めないので写してから渡す
        x = np.array(x)
        tensors = (torch.from_numpy(x),) if y is None else (
            torch.from_numpy(x), torch.from_numpy(y)
        )

        return tensors


def load_dataset(
    path: Union[str, Path], follow_links: bool = False
) -> Dataset:
    """
    書き出したデータセットを読み込む関数

    Parameters
    ----------
    path: Union[str, Path]
        export_dataset で書き出したディレクトリ
    follow_links: bool = False
        説明, 分割のデータ, 中身の配列, レシピのうち, リンクをたどった後に
        データセットの外を指すものも読むか (False では誤り)

    Returns
    ----------
    dataset: Dataset
        説明, 分割ごとの表, 当てはめたレシピ

    Raises
    ----------
    ValueError
        版 2 と 3 (ファイルの組. load_items で読む) と, 新しい版
        の場合と, レシピに知らない手順, 手順と中身の手順の知らない引数が
        ある場合, 説明に書かれた場所がデータセットの外 (絶対パスや ..,
        リンクの先) を指す場合
    FileNotFoundError
        説明と, 説明に書かれたファイルが無い場合
    """
    # 説明と版の確認. 読む場所は全てデータセットの中に限る
    root = Path(path).expanduser().absolute()
    place = partial(
        within, root, directory=False, follow_links=follow_links,
        missing=FileNotFoundError
    )
    info = yaml.safe_load(place("dataset.yaml").read_text(encoding="utf-8"))
    version = info.get("format_version") if isinstance(info, dict) else None
    if version in ITEMS_FORMAT_VERSIONS:
        raise ValueError(
            f"format_version {version} のデータセットは入力と正解がファイル"
            "の組のもので, load_dataset (版 1 の表) では読めません. "
            "load_items で読んでください"
        )
    if not isinstance(version, int) or version > DATASET_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のデータセットは, このライブラリ (版 "
            f"{DATASET_FORMAT_VERSION} まで) では読めません"
        )

    # 分割ごとの表とレシピ
    frames = {}
    for split, item in info["splits"].items():
        file = place(item["file"])
        frames[split] = (
            pd.read_parquet(file) if file.suffix == ".parquet"
            else pd.read_csv(file)
        )
        if item.get("items"):
            place(item["items"])
    recipe = yaml.safe_load(
        place(info.get("recipe", "recipe.yaml")).read_text(encoding="utf-8")
    )

    # 当てはめたレシピの手順と引数 (知らない引数を黙って無視しない. 空の
    # 引数は書き出した時の当てはめと同じ読まれ方をするので確かめない)
    if not isinstance(recipe, dict):
        raise ValueError(
            "データセットのレシピが辞書 (YAML のマップ) ではありません"
        )
    check_recipe_steps(
        recipe.get("steps") or [], recipe.get("item_steps") or [], False
    )
    dataset = Dataset(path=root, info=info, frames=frames, recipe=recipe)

    return dataset


def item_input(place: "DatasetPlace", splits: dict) -> Optional[dict]:
    """
    画像, 文, 点群, 音声の中身の配列から, モデルの入力の形と型を求める関数

    Parameters
    ----------
    place: DatasetPlace
        データセットの場所 (読み方)
    splits: dict
        説明の splits

    Returns
    ----------
    result: Optional[dict]
        {name, shape, dtype} (中身の配列の無いデータセットは None)

    Raises
    ----------
    ValueError
        配列の場所がデータセットの外を指す場合
    FileNotFoundError
        配列が無い場合
    """
    # 学習用を先に, 無ければ配列を持つ最初の分割. 型はファイルの頭だけ
    names = [name for name in splits if name == "train"] + [
        name for name in splits if name != "train"
    ]
    chosen = [name for name in names if splits[name].get("items")]
    if not chosen:
        result = None
        return result
    item = splits[chosen[0]]
    found, dtype = place.array_header(item["items"])
    shape = list(item.get("shape") or found)
    result = {
        "name": MODEL_INPUT,
        "shape": [None] + [int(size) for size in shape[1:]],
        "dtype": str(dtype),
    }

    return result


def text_vocabulary(place: "DatasetPlace", info: dict) -> Optional[int]:
    """
    文のデータセットの当てはめた語彙から, 語彙の大きさを求める関数

    Parameters
    ----------
    place: DatasetPlace
        データセットの場所 (読み方)
    info: dict
        説明 (dataset.yaml)

    Returns
    ----------
    size: Optional[int]
        語彙の大きさ (語の番号の上限 + 1. <pad> と <unk> を含む). 文でない
        か, 当てはめた vocabulary の手順が無い場合は None

    Raises
    ----------
    ValueError
        レシピの場所がデータセットの外を指す場合
    FileNotFoundError
        レシピが無い場合
    """
    # 当てはめたレシピの中身の手順の最後の vocabulary
    size: Optional[int] = None
    if info.get("kind") != "text":
        return size
    recipe = yaml.safe_load(
        place.read(info.get("recipe", "recipe.yaml")).decode("utf-8")
    ) or {}
    for step in recipe.get("item_steps") or []:
        if not isinstance(step, dict) or step.get("op") != "vocabulary":
            continue
        state = step.get("state") or {}
        if state.get("vocabulary"):
            size = len(state["vocabulary"])

    return size


def model_outputs(
    task: Optional[str], classes: list, width: int
) -> Optional[int]:
    """
    モデルが 1 件に出す値の数を, タスクから求める関数

    Parameters
    ----------
    task: Optional[str]
        タスク (決めていなければ None)
    classes: list
        分類のクラス
    width: int
        学習に使う列の数

    Returns
    ----------
    outputs: Optional[int]
        分類はクラスの数, 回帰と時系列の予測は 1, 異常の検知は学習に使う
        列の数 (自己符号化器が入力を作り直す). 分からなければ None
    """
    # 版 1 の説明は何期先までを持たないので, 時系列の予測も 1 行に 1 つ
    outputs = None
    if task == "classification" and classes:
        outputs = len(classes)
    elif task in ("regression", "forecasting"):
        outputs = 1
    elif task == "anomaly":
        outputs = width

    return outputs


def mask_class_names(entry: dict) -> list[str]:
    """
    領域の項目の値の名前を, 値の小さい順に返す関数

    Parameters
    ----------
    entry: dict
        説明の領域の項目 (values と classes)

    Returns
    ----------
    names: list[str]
        クラスの名前 (classes が無ければ値の文字. 名前の無い値 0 は
        background)
    """
    # 名前の付いた値 (無ければ値の一覧の全て). ML-Quest の骨組みの数え方
    values = [int(value) for value in entry.get("values") or []]
    listed = entry.get("classes") or [
        {"value": value, "name": str(value)} for value in values
    ]
    named = {int(item["value"]): str(item["name"]) for item in listed}

    # 名前の無い値 0 は背景のクラス (骨組みが背景も 1 つのクラスで学ぶため)
    if 0 in values and 0 not in named:
        named[0] = "background"
    names = [named[value] for value in sorted(named)]

    return names


def image_input(name: str, entry: dict) -> dict:
    """
    画像の項目から, モデルの入力 (バッチ, チャンネル, 高さ, 幅) を作る関数

    Parameters
    ----------
    name: str
        項目の名前
    entry: dict
        説明の画像の項目 (shape は手順の後の (高さ, 幅, チャンネル))

    Returns
    ----------
    single: dict
        {name, shape, dtype} (そろわない大きさは None)
    """
    # 説明の形 (書き出しが手順の後の大きさを書く). 無ければ手順の resize,
    # grayscale, rgb から
    shape = list(entry.get("shape") or [None, None, None])[:3]
    shape += [None] * (3 - len(shape))
    # 手順は名前だけの文字も {op: 名前} と同じ
    for listed in entry.get("steps") or []:
        step = {"op": listed} if isinstance(listed, str) else dict(listed)
        if step.get("op") == "resize":
            shape[0] = shape[0] or step.get("height")
            shape[1] = shape[1] or step.get("width")
        elif step.get("op") in ("grayscale", "rgb"):
            shape[2] = shape[2] or (1 if step["op"] == "grayscale" else 3)
    sizes = [None if size is None else int(size) for size in shape]
    single = {
        "name": name,
        "shape": [None, sizes[2], sizes[0], sizes[1]],
        "dtype": "float32",
    }

    return single


def cloud_count(entry: dict, default: int) -> int:
    """
    点群の項目の点の数を返す関数

    Parameters
    ----------
    entry: dict
        説明の点群の項目 (shape は (点の数, 3))
    default: int
        点の数がそろわない場合の数

    Returns
    ----------
    count: int
        そろった点の数 (sample_points の後など). そろわなければ default
    """
    # 形の最初の次元 (そろわなければ null)
    shape = entry.get("shape") or [None]
    count = int(shape[0]) if shape[0] is not None else default

    return count


def items_model_inputs(info: dict) -> dict:
    """
    版 2 と 3 (ファイルの組) の説明から, モデルの入力と出す値の数を
    求める関数

    Parameters
    ----------
    info: dict
        dataset.yaml の中身

    Returns
    ----------
    result: dict
        inputs, task (None), target (正解の項目の名前), classes (領域の
        クラスの名前), target_kind, outputs の辞書

    Raises
    ----------
    ValueError
        入力と正解の種類の組を知らない場合
    """
    # 入力と正解の項目 (説明の順の最初のもの)
    entries = info.get("entries") or {}
    inputs = [key for key, value in entries.items()
              if value.get("role") == "input"]
    targets = [key for key, value in entries.items()
               if value.get("role") == "target"]
    kinds = (
        entries[inputs[0]].get("kind") if inputs else None,
        entries[targets[0]].get("kind") if targets else None,
    )

    # 画像と領域: (バッチ, チャンネル, 高さ, 幅) と, 背景を含むクラスの数
    if kinds == ("image", "mask"):
        names = mask_class_names(entries[targets[0]])
        result = {
            "inputs": [image_input(inputs[0], entries[inputs[0]])],
            "task": None, "target": targets[0], "classes": names,
            "target_kind": "mask", "outputs": len(names) or None,
        }
        return result

    # 撮影か点群と点群: (バッチ, 入力の点の数, 3) と, 出す点の数 (撮影は
    # 学習のコードが深度から点に戻して選ぶので既定の数)
    if kinds[0] in ("capture", "point_cloud") and kinds[1] == "point_cloud":
        points = ITEM_INPUT_POINTS if kinds[0] == "capture" else cloud_count(
            entries[inputs[0]], ITEM_INPUT_POINTS
        )
        result = {
            "inputs": [{"name": inputs[0], "shape": [None, points, 3],
                        "dtype": "float32"}],
            "task": None, "target": targets[0], "classes": None,
            "target_kind": "point_cloud",
            "outputs": cloud_count(entries[targets[0]], ITEM_TARGET_POINTS),
        }
        return result

    raise ValueError(
        f"入力 {kinds[0]} と正解 {kinds[1]} の組のデータセットのモデルの"
        "入力はまだ返せません. 項目の形は dataset.yaml の entries を"
        "見てください"
    )


@dataclass(frozen=True)
class DatasetPlace:
    # 書き出したデータセットのディレクトリ
    root: Path
    # リンクをたどった後にデータセットの外を指す場所も読むか
    follow_links: bool = False
    # 親のディレクトリの記述子 (渡すと, root の最後の部分を親の中の名前
    # とし, 親に対する相対でリンクをたどらずに読む)
    parent: Optional[int] = None

    def read(self, name: Any) -> bytes:
        """
        データセットの中のファイルを読む関数

        Parameters
        ----------
        name: Any
            書かれた場所 (データセットからの相対)

        Returns
        ----------
        data: bytes
            ファイルの中身
        """
        # 開いた親に対するか, パスでの読み込み (どちらも中に限る)
        if self.parent is not None:
            with open_under(
                self.parent, entry_name(self.root), name
            ) as file:
                data = file.read()
            return data
        data = within(
            self.root, name, False, self.follow_links, FileNotFoundError
        ).read_bytes()

        return data

    def array_header(self, name: Any) -> tuple[tuple, np.dtype]:
        """
        中身の配列のファイルの頭から, 形と型を読む関数

        Parameters
        ----------
        name: Any
            書かれた場所 (データセットからの相対)

        Returns
        ----------
        header: tuple[tuple, np.dtype]
            配列の形と型
        """
        # パスでは配列の写像, 開いた親ではファイルの頭だけの読み込み
        if self.parent is None:
            array = np.load(
                within(
                    self.root, name, False, self.follow_links,
                    FileNotFoundError
                ),
                allow_pickle=False, mmap_mode="r"
            )
            header = (tuple(array.shape), array.dtype)
            return header
        with open_under(self.parent, entry_name(self.root), name) as file:
            version = np.lib.format.read_magic(file)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(file)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(file)
            else:
                raise ValueError(
                    f"{name} の .npy の版 {version} は頭だけでは読めません"
                )

        # 物体型の配列の断り (パスで読む場合 (allow_pickle=False) と同じ)
        if dtype.hasobject:
            raise ValueError(
                f"{name} は Python の物体を持つ配列なので読みません "
                "(pickle を使うため)"
            )
        header = (tuple(shape), dtype)

        return header


def model_inputs(
    path: Union[str, Path], follow_links: bool = False,
    parent: Optional[int] = None
) -> dict:
    """
    書き出したデータセットから, モデルの入力と出す値の数を求める関数

    Parameters
    ----------
    path: Union[str, Path]
        export_dataset で書き出したディレクトリ
    follow_links: bool = False
        説明, 中身の配列, レシピのうち, リンクをたどった後にデータセットの
        外を指すものも読むか (False では誤り. load_dataset と同じ)
    parent: Optional[int] = None
        書き出した場所の親のディレクトリの記述子 (渡すと path の最後の
        部分を親の中の名前とし, 親に対する相対でリンクをたどらずに読む.
        path の途中は使わない. follow_links とは一緒に渡せない)

    Returns
    ----------
    result: dict
        inputs ({name, shape, dtype} の並び. 文の語の番号の入力は語彙の
        大きさ vocabulary も), task, target, classes, target_kind, outputs
        の JSON にできる辞書

    Raises
    ----------
    ValueError
        知らない版のデータセットと, 版 2 と 3 (ファイルの組) で入力と
        正解の種類の組を知らない場合, 読む場所がデータセットの外 (絶対
        パスや .., リンクの先) を指す場合
    FileNotFoundError
        dataset.yaml が無い場合と, 中身の配列か文のデータセットで書き出した
        レシピが無い場合
    """
    # 説明と版の確認 (分割のデータは読まない. ファイルの組は説明だけ).
    # 読む場所はデータセットの中に限る
    if parent is not None and follow_links:
        raise ValueError(
            "開いた親 (parent) を渡す場合は, リンクをたどれません "
            "(follow_links=True は渡せません)"
        )
    root = Path(path).expanduser().absolute()
    place = DatasetPlace(root, follow_links, parent)
    info = yaml.safe_load(place.read("dataset.yaml").decode("utf-8"))
    version = info.get("format_version") if isinstance(info, dict) else None
    if version in ITEMS_FORMAT_VERSIONS:
        result = items_model_inputs(info)
        return result
    if not isinstance(version, int) or version > DATASET_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のデータセットは, このライブラリ (版 "
            f"{DATASET_FORMAT_VERSION} まで) では読めません"
        )

    # 入力 (画像などは中身の配列, 表は学習に使う列を並べた 1 つ)
    single = item_input(place, info.get("splits") or {})
    if single is not None:
        # 文の語の番号は, 埋め込みに要る語彙の大きさも (書き出したレシピから)
        vocabulary = text_vocabulary(place, info)
        if vocabulary is not None:
            single["vocabulary"] = vocabulary
    if single is None:
        single = {
            "name": MODEL_INPUT,
            "shape": [None, len(info.get("features") or [])],
            "dtype": "float32",
        }

    # タスクと出す値の数 (none は決めていないこと)
    task = info.get("task")
    task = None if task in (None, "none") else str(task)
    classes = list(info.get("classes") or [])
    result = {
        "inputs": [single],
        "task": task,
        "target": info.get("target"),
        "classes": classes if task == "classification" else None,
        "target_kind": None,
        "outputs": model_outputs(
            task, classes, len(info.get("features") or [])
        ),
    }

    return result
