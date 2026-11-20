import datetime as dt
import re
import shutil
from dataclasses import dataclass, field
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
from ml_eda.profile import finding, to_json
from ml_eda.recipe import (
    HIDDEN, STEPS, WANT, category_key, load_recipe, resolve_columns,
    to_plain
)

# データセットの説明の形式の版 (load_dataset が読む, 1 件 = 1 行の約束.
# ML-Inference が写しの確かめに使うので, 版 2 を足しても 1 のまま)
DATASET_FORMAT_VERSION = 1

# 入力と正解がファイルの組の約束の版 (items.load_items が読む)
ITEMS_FORMAT_VERSION = 2

# 分割を表す列と, 分割に使う日時とグループの写し (処理の間だけ使う)
SPLIT_COLUMN = "__split__"
TIME_COLUMN = "__time__"
ROW_COLUMN = "__row__"
GROUP_COLUMN = "__group__"
HIDDEN_COLUMNS = (SPLIT_COLUMN, TIME_COLUMN, ROW_COLUMN, GROUP_COLUMN)

# 重なりの例として挙げる値の数
OVERLAP_EXAMPLES = 3

# 目的の列があるのにタスクを決めていない場合の注意
NO_TASK_NOTE = (
    "タスクが決まっていません. 目的の列はそのままの値で書き出します"
)

# 分割の名前
SPLITS = ("train", "valid", "test")

# モデルの入力の名前 (版 1 の入力は tensors の特徴量 1 つ)
MODEL_INPUT = "x"

# 目的の列の欠けた行を残す分割の名前 (split.unlabeled: keep の場合)
UNLABELED = "unlabeled"


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
    # 画像, 文, 点群の中身の手順をかけた, 分割ごとの配列
    arrays: dict = field(default_factory=dict)
    # 分割の後に分かる気付き (分割をまたぐグループや画像の重なり)
    findings: list = field(default_factory=list)


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
        method (random, stratified, time, group, none), valid, test
        (割合), seed, column (group の列)
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

    # グループごとに同じ分割にする (行の件数が割合に近くなるまで加える)
    if method == "group":
        column = split.get("column")
        if not column or column not in frame.columns:
            raise ValueError(
                f"グループの分割には列 (split.column) が要ります: {column}"
            )
        keys = frame[column].astype(str).where(frame[column].notna(), None)
        sizes = keys.value_counts(dropna=False)
        groups = list(sizes.index)
        rng.shuffle(groups)
        labels: dict = {}
        filled = {"test": 0, "valid": 0}
        for group in groups:
            size = int(sizes[group])
            if filled["test"] + size <= max(n_test, 0) and n_test:
                labels[group] = "test"
                filled["test"] += size
            elif filled["valid"] + size <= max(n_valid, 0) and n_valid:
                labels[group] = "valid"
                filled["valid"] += size
            else:
                labels[group] = "train"
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
            findings.append(finding(
                "info" if method == "time" else "warning", "group_overlap",
                f"{group} の同じ値が {overlap['values']} 個 (計 "
                f"{overlap['rows']} 行) 複数の分割にまたがります (例: "
                f"{', '.join(overlap['examples'])}). 学習用にいる人や物を"
                "検証用で当てると, 新しい人や物に当てはめる力より甘く"
                f"測ります. {usual}新しい {group} に当てはめるなら, 分割を "
                f"group (column: {group}) にしてください",
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
    if target:
        if target not in frame.columns:
            raise ValueError(f"目的の列 {target} がありません")
        frame = frame[frame[target].notna()]
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
    frame[SPLIT_COLUMN] = assign_splits(frame, split, split_roles)
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

    # 分割と手順
    recipe = load_recipe(recipe)
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
        if extra_rows:
            frames[UNLABELED] = extra
            positions[UNLABELED] = extra_rows

    # 画像, 文, 点群は中身の手順で配列にする (表の列は学習に使わない)
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
    prepared = Prepared(
        frames=frames, recipe=recipe, features=features, target=target,
        classes=classes, notes=notes, findings=findings
    )

    return prepared


def preview_items(
    data: Data, recipe: Union[str, Path, dict], count: int = 8,
    fit_limit: int = 100, seed: Optional[int] = None
) -> dict:
    """
    画像, 文, 点群の中身の手順を, 学習用の見本で当てはめて, 数件に
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
        raise ValueError("中身を持たない種類です (画像, 文, 点群で使う)")
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
                "読めない件 (画像は width が欠損, 点群は error がある件) を"
                "除いてください"
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
        行数, 列の名前と型, 先頭の行 (値は文字か数)
    """
    # 先頭の行を JSON にできる値にする
    head = frame.head(rows)
    records = [
        [
            str(value) if isinstance(value, (list, dict, np.ndarray))
            else None if pd.isna(value) else (
                value if isinstance(value, (int, float, bool, np.number))
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
    path: Union[str, Path], overwrite: bool
) -> tuple[Path, Path]:
    """
    データセットを書き出す一時ディレクトリを作る関数

    版 1 と版 2 の書き出しで共通. 隣の一時ディレクトリに書いてから
    finish_writing で入れ替える

    Parameters
    ----------
    path: Union[str, Path]
        書き出すディレクトリ
    overwrite: bool
        既にあるディレクトリを置き換えるか

    Returns
    ----------
    places: tuple[Path, Path]
        書き出す場所と, 一時ディレクトリ

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合と, データセット
        ではない (dataset.yaml の無い) ディレクトリを置き換えようとした場合
    """
    # 置き換えられるのは, 前に書き出したデータセットだけ
    final = Path(path).expanduser()
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


def finish_writing(target: Path, final: Path, info: dict) -> None:
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

    Returns
    ----------
    None
    """
    # 説明は最後に書き, 前のデータセットと入れ替える
    (target / "dataset.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )
    if final.exists():
        shutil.rmtree(final)
    target.rename(final)


def export_dataset(
    data: Data, recipe: Union[str, Path, dict], path: Union[str, Path],
    name: Optional[str] = None, overwrite: bool = False
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

    Returns
    ----------
    info: dict
        書き出した dataset.yaml の中身

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合と, データセット
        ではない (dataset.yaml の無い) ディレクトリを置き換えようとした場合
    """
    # 隣の一時ディレクトリで前処理 (中身の配列も 1 件ずつここに書く) を
    # 済ませてから入れ替える. 失敗したら一時ディレクトリを消す
    final, target = start_writing(path, overwrite)
    try:
        prepared = prepare(data, recipe, item_dir=target)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
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
        "text_column": data.roles.get("text") if data.kind == "text" else None,
        "splits": splits,
        "source": to_plain(data.source),
        "recipe": "recipe.yaml",
        "notes": prepared.notes,
    }
    finish_writing(target, final, info)

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
                np.int64 if self.info["task"] == "classification"
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
        新しい画像, 文, 点群に, 当てはめた中身の手順をかける関数

        Parameters
        ----------
        item: Any
            中身 (画像は配列, 文は文字, 点群は points を持つ辞書)

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


def load_dataset(path: Union[str, Path]) -> Dataset:
    """
    書き出したデータセットを読み込む関数

    Parameters
    ----------
    path: Union[str, Path]
        export_dataset で書き出したディレクトリ

    Returns
    ----------
    dataset: Dataset
        説明, 分割ごとの表, 当てはめたレシピ

    Raises
    ----------
    ValueError
        版 2 (入力と正解がファイルの組. load_items で読む) と, 新しい版
        の場合
    """
    # 説明と版の確認
    root = Path(path).expanduser()
    info = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    version = info.get("format_version") if isinstance(info, dict) else None
    if version == ITEMS_FORMAT_VERSION:
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
        file = root / item["file"]
        frames[split] = (
            pd.read_parquet(file) if file.suffix == ".parquet"
            else pd.read_csv(file)
        )
    recipe = yaml.safe_load(
        (root / info.get("recipe", "recipe.yaml")).read_text(encoding="utf-8")
    )
    dataset = Dataset(path=root, info=info, frames=frames, recipe=recipe)

    return dataset


def item_input(root: Path, splits: dict) -> Optional[dict]:
    """
    画像, 文, 点群の中身の配列から, モデルの入力の形と型を求める関数

    Parameters
    ----------
    root: Path
        データセットのディレクトリ
    splits: dict
        説明の splits

    Returns
    ----------
    result: Optional[dict]
        {name, shape, dtype} (中身の配列の無いデータセットは None)
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
    array = np.load(root / item["items"], allow_pickle=False, mmap_mode="r")
    shape = list(item.get("shape") or array.shape)
    result = {
        "name": MODEL_INPUT,
        "shape": [None] + [int(size) for size in shape[1:]],
        "dtype": str(array.dtype),
    }

    return result


def model_outputs(task: Optional[str], classes: list) -> Optional[int]:
    """
    モデルが 1 件に出す値の数を, タスクから求める関数

    Parameters
    ----------
    task: Optional[str]
        タスク (決めていなければ None)
    classes: list
        分類のクラス

    Returns
    ----------
    outputs: Optional[int]
        分類はクラスの数, 回帰と時系列の予測は 1. 分からなければ None
    """
    # 版 1 の説明は何期先までを持たないので, 時系列の予測も 1 行に 1 つ
    outputs = None
    if task == "classification" and classes:
        outputs = len(classes)
    elif task in ("regression", "forecasting"):
        outputs = 1

    return outputs


def model_inputs(path: Union[str, Path]) -> dict:
    """
    書き出したデータセットから, モデルの入力と出す値の数を求める関数

    Parameters
    ----------
    path: Union[str, Path]
        export_dataset で書き出したディレクトリ

    Returns
    ----------
    result: dict
        inputs ({name, shape, dtype} の並び), task, target, classes,
        outputs の JSON にできる辞書

    Raises
    ----------
    ValueError
        版 2 (ファイルの組) と知らない版のデータセットの場合
    FileNotFoundError
        dataset.yaml が無い場合
    """
    # 説明と版の確認 (分割のデータは読まない)
    root = Path(path).expanduser()
    info = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    version = info.get("format_version") if isinstance(info, dict) else None
    if version == ITEMS_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のデータセット (ファイルの組) の"
            "モデルの入力はまだ返せません. 項目の形は dataset.yaml の "
            "entries を見てください"
        )
    if not isinstance(version, int) or version > DATASET_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のデータセットは, このライブラリ (版 "
            f"{DATASET_FORMAT_VERSION} まで) では読めません"
        )

    # 入力 (画像などは中身の配列, 表は学習に使う列を並べた 1 つ)
    single = item_input(root, info.get("splits") or {})
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
        "outputs": model_outputs(task, classes),
    }

    return result
