import datetime as dt
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
from ml_eda.profile import to_json
from ml_eda.recipe import (
    STEPS, WANT, category_key, load_recipe, resolve_columns, to_plain
)

# データセットの説明の形式の版
DATASET_FORMAT_VERSION = 1

# 分割を表す列と, 分割に使う日時の写し (処理の間だけ使う)
SPLIT_COLUMN = "__split__"
TIME_COLUMN = "__time__"
ROW_COLUMN = "__row__"
HIDDEN_COLUMNS = (SPLIT_COLUMN, TIME_COLUMN, ROW_COLUMN)

# 目的の列があるのにタスクを決めていない場合の注意
NO_TASK_NOTE = (
    "タスクが決まっていません. 目的の列はそのままの値で書き出します"
)

# 分割の名前
SPLITS = ("train", "valid", "test")


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
        times = pd.to_datetime(frame[time], utc=True).to_numpy()
        order = np.argsort(times, kind="stable")
        return label_by_position(frame.index, order, n_valid, n_test)

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

    # 分割の前の手順, 分割, 分割の後の手順. 時刻の順の分割に使う日時は,
    # 手順で日時の列を落としても使えるように別の列に写しておく
    time = roles.get("time")
    if time and time in frame.columns:
        frame[TIME_COLUMN] = frame[time]
    head = first_fit(steps)
    for number, spec in enumerate(steps[:head], 1):
        frame = run_step(frame, spec, roles, number, None)
    if frame.empty:
        raise ValueError("手順の後に行が残りませんでした")
    split_roles = {**roles, "time": TIME_COLUMN} if time else roles
    frame[SPLIT_COLUMN] = assign_splits(frame, recipe["split"], split_roles)
    frame = frame.drop(columns=[TIME_COLUMN], errors="ignore")
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
        分割ごとの表, 当てはめたレシピ, 学習に使う列, クラス

    Raises
    ----------
    ValueError
        手順の失敗, 目的の列が手順で消えた場合, 学習用の行が無い場合,
        回帰の目的の列が数値でない場合
    """
    # 分割と手順
    recipe = load_recipe(recipe)
    result, roles = apply_recipe(data, recipe)
    target = roles.get("target")
    if target and target not in result.columns:
        raise ValueError(f"目的の列 {target} が手順で消えました")
    if not (result[SPLIT_COLUMN] == "train").any():
        raise ValueError("学習用の行がありません (割合か絞り込みを見直す)")

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

    # 画像, 文, 点群は中身の手順で配列にする (表の列は学習に使わない)
    arrays: dict = {}
    if data.item_loader is not None:
        check_item_steps(recipe["item_steps"], data.kind)
        rows = data.frame.iloc[result[ROW_COLUMN].tolist()]
        notes = item_notes(rows, recipe["item_steps"], data.kind)
        if target and recipe["task"] == "none":
            notes.append(NO_TASK_NOTE)
        if not read_items:
            return Prepared(
                frames=frames, recipe=recipe, features=[], target=target,
                classes=classes, notes=notes
            )
        arrays = run_item_steps(
            item_reader(data), positions.get("train", []), positions,
            recipe["item_steps"], data.kind, item_dir,
            int(recipe["split"].get("seed", 0))
        )
        prepared = Prepared(
            frames=frames, recipe=recipe, features=[], target=target,
            classes=classes, notes=notes, arrays=arrays
        )
        return prepared

    # 学習に使う列 (役割の列を除く) と, 数値でない列の注意
    reserved = {target, roles.get("id"), roles.get("time"), SPLIT_COLUMN}
    features = [
        name for name in result.columns
        if name not in reserved and name not in HIDDEN_COLUMNS
    ]
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
    if target and recipe["task"] == "none":
        notes.append(NO_TASK_NOTE)
    prepared = Prepared(
        frames=frames, recipe=recipe, features=features, target=target,
        classes=classes, notes=notes
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
    # 置き換えられるのは, 前に書き出したデータセットだけ
    target = Path(path).expanduser()
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} は既にあります")
        if not (target / "dataset.yaml").is_file():
            raise FileExistsError(
                f"{target} はデータセットではないので置き換えません"
            )

    # 隣の一時ディレクトリで前処理 (中身の配列も 1 件ずつここに書く) を
    # 済ませてから入れ替える. 失敗したら一時ディレクトリを消す
    final = target
    target = final.with_name(f".{final.name}.writing")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
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
        "splits": splits,
        "source": to_plain(data.source),
        "recipe": "recipe.yaml",
        "notes": prepared.notes,
    }
    (target / "dataset.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )
    if final.exists():
        shutil.rmtree(final)
    target.rename(final)

    return info


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
            train, valid, test のどれか

        Returns
        ----------
        x: np.ndarray
            (件数, 列の数) の float32 の配列
        y: Optional[np.ndarray]
            目的の値 (分類は int64 の番号, 回帰は float32), 無ければ None
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
        if self.target:
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
            学習に使う列だけの表 (当てはめた値をそのまま使う)
        """
        # 当てはめ直さずに, 保存した値と列で手順を順にかける. 行を増減
        # する手順と, 目的の列にかける手順は飛ばす (入力と行をそろえる)
        result = frame.copy()
        result.columns = [str(name) for name in result.columns]
        for spec in self.recipe["steps"]:
            step = STEPS[spec["op"]]
            state = spec.get("state", {})
            columns = state.get("columns", [])
            if step.changes_rows or (self.target and self.target in columns):
                continue
            params = {
                k: v for k, v in spec.items() if k not in ("op", "state")
            }
            if "columns" in state:
                params["columns"] = [c for c in columns if c in result]
            result = step.apply(result, params, state)
        # 学習に使う列が足りない場合は, 黙って落とさずに誤りにする
        missing = [c for c in self.features if c not in result.columns]
        if missing:
            raise ValueError(
                f"前処理の後に, 学習に使う列が足りません: {', '.join(missing)}"
                " (元のデータと同じ列を渡してください. カンマ付きの数などは "
                "load_data で読んだ後の形にそろえてください)"
            )
        result = result[self.features]

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
            train, valid, test のどれか

        Returns
        ----------
        tensors: tuple
            (特徴量, 目的の値) のテンソル (目的の列が無ければ特徴量だけ)
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
        新しい版の場合
    """
    # 説明と版の確認
    root = Path(path).expanduser()
    info = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    version = info.get("format_version")
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

