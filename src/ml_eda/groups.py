import datetime as dt
import shutil
from pathlib import Path
from typing import Any, Hashable, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.items import (
    ItemDataset, RESERVED, check_entry_names, split_items, write_items
)
from ml_eda.loaders import load_data
from ml_eda.pipeline import (
    ITEMS_FORMAT_VERSION, TABLE_KINDS, shared_source, start_writing
)
from ml_eda.profile import to_json

# 群の要約の行の, 群を作る列と同じ名前にできない鍵 (ML-Quest のまとめるの
# 骨組みの群の要約と同じ)
SUMMARY_COLUMNS = ("group", "rows", "share", "stability")

# 群の要約の最後の, 全ての行の行の group
ALL_ROWS = "全体"

# 種類の列の欠けた値を 1 つの種類にする文字
MISSING = "(欠損)"

# 群を作る列の種類の項目の kind (数の列と他の列)
NUMBER_KIND = "value"
CATEGORY_KIND = "class"

# 群の作り方の既定 (骨組みの config の group と同じ値)
REPEATS = 5
INITS = 5
ITERATIONS = 100
SAMPLE = 2000
FRACTION = 0.8

# 群の安定性の数え方 (ML-Quest の問題の設定の答え group.stability の値)
STABILITY_METHODS = ("seed", "sample")

# 符号化した行の幅の上限 (種類の値の多い列で表が大きくなりすぎないため)
MAX_WIDTH = 1000

# 値の多い種類の列とみなす, 行の数に対する異なる値の数の割合
MANY_VALUES = 0.5


def table_of(
    source: Union[str, Path, Data], loader: Optional[str], options: dict
) -> Data:
    """
    表を load_data で読むか, 読んだデータを使う関数

    Parameters
    ----------
    source: Union[str, Path, Data]
        表のファイルか, load_data で読んだデータ
    loader: Optional[str]
        読み込みの名前 (None は推定)
    options: dict
        読み込みの設定

    Returns
    ----------
    data: Data
        読んだ表のデータ

    Raises
    ----------
    ValueError
        読んだデータに読み込みの名前か設定を渡した場合と, 表でない場合
    """
    # 読んだデータはそのまま使う (読み方は読んだときのもの)
    if isinstance(source, Data):
        if loader is not None or options:
            raise ValueError(
                "読んだデータ (Data) を渡す場合は, 読み込みの名前と設定を"
                "渡さないでください"
            )
        data = source
    else:
        data = load_data(source, loader=loader, **options)
    if data.kind not in TABLE_KINDS:
        raise ValueError(
            f"群を作るデータは表にしてください ({data.kind} のデータです)"
        )

    return data


def group_columns(
    frame: pd.DataFrame, columns: Any, id: Optional[Hashable] = None
) -> list:
    """
    群を作る列と行を見分ける列を確かめる関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    columns: Any
        群を作る列の名前の並び
    id: Optional[Hashable] = None
        行を見分ける列 (None は無し)

    Returns
    ----------
    names: list
        群を作る列の名前

    Raises
    ----------
    ValueError
        列が無い場合, 同じ列を 2 回渡した場合, 群の要約の鍵と同じ名前の
        場合, 表に同じ名前の列が 2 つ以上ある場合
    """
    # 1 つ以上の, 別々の列
    if columns is None or isinstance(columns, (str, bytes)):
        names = [] if columns is None else [columns]
    else:
        names = list(columns)
    if not names:
        raise ValueError("群を作る列の名前を渡してください")
    texts = [str(name) for name in names]
    if len(set(texts)) < len(texts):
        raise ValueError("群を作る列に同じ列を 2 回渡さないでください")
    if id is not None and str(id) in texts:
        raise ValueError("行を見分ける列は群を作る列に入れないでください")

    # 表にある列で, 群の要約と 1 件の辞書の鍵と重ならない名前
    wanted = names + ([id] if id is not None else [])
    missing = [str(name) for name in wanted if name not in frame.columns]
    if missing:
        raise ValueError(
            f"表に {', '.join(missing)} の列がありません. 列の名前を直して"
            "ください"
        )
    repeated = [str(name) for name in wanted
                if (frame.columns == name).sum() > 1]
    if repeated:
        raise ValueError(
            f"表に同じ名前の列 {', '.join(repeated)} が 2 つ以上あります"
        )
    taken = [text for text in texts
             if text in SUMMARY_COLUMNS or text in RESERVED]
    if taken:
        raise ValueError(
            f"群を作る列 {', '.join(taken)} は群の要約か 1 件の鍵 ("
            f"{', '.join(dict.fromkeys(SUMMARY_COLUMNS + RESERVED))}) と"
            "同じ名前です. 名前を変えた列を作ってから選んでください"
        )

    return names


def is_number(column: pd.Series) -> bool:
    """
    列を数の列として扱うかを返す関数

    Parameters
    ----------
    column: pd.Series
        列

    Returns
    ----------
    number: bool
        数の列 (真偽値の列は種類の列にする)
    """
    # 真偽値は 2 つの種類 (ML-Quest の骨組みと同じ)
    number = (pd.api.types.is_numeric_dtype(column)
              and not pd.api.types.is_bool_dtype(column))

    return number


def category_text(value: Any) -> str:
    """
    種類の列の値を, 符号化と要約に使う文字にする関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    text: str
        文字 (欠けた値は MISSING)
    """
    # 欠けた値も 1 つの種類にする
    text = MISSING if pd.isna(value) else str(value)

    return text


def column_notes(frame: pd.DataFrame) -> list[str]:
    """
    群を作る列の, 欠けた値と値の多い種類の列の気付きを作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列

    Returns
    ----------
    notes: list[str]
        気付きの日本語の文
    """
    # 欠けた値のある行と, 行を見分ける列のように値の多い種類の列
    notes = []
    lacking = int(frame.isna().any(axis=1).sum())
    if lacking:
        notes.append(
            f"群を作る列に欠けた値のある行が {lacking} 行あります (数の列"
            f"は平均に, 種類の列は {MISSING} という 1 つの値にして群を"
            "作ります)"
        )
    for name in frame.columns:
        column = frame[name]
        distinct = column.nunique(dropna=False)
        if not is_number(column) and len(frame) > 1 and (
            distinct > len(frame) * MANY_VALUES
        ):
            notes.append(
                f"種類の列 {name} は値が {distinct} 種類あり, 行を見分ける"
                "列のように多いです. 群を作る列から外すか, 値をまとめて"
                "ください"
            )

    return notes


def export_group_rows(
    source: Union[str, Path, Data], path: Union[str, Path], columns: Any,
    id: Optional[Hashable] = None, split: Optional[dict] = None,
    name: Optional[str] = None, overwrite: bool = False,
    loader: Optional[str] = None, **options: Any
) -> dict:
    """
    群分け (まとめる) の表を, 版 2 のデータセットに書き出す関数

    Parameters
    ----------
    source: Union[str, Path, Data]
        1 行が 1 つの対象の表のファイルか, load_data で読んだデータ
    path: Union[str, Path]
        書き出すディレクトリ
    columns: Any
        群を作る列の名前の並び
    id: Optional[Hashable] = None
        行を見分ける列, None の場合は行の位置 (0 から) を id にする
    split: Optional[dict] = None
        分割のしかた (method は random か none, valid, test, seed. 既定は
        random で検証用 2 割)
    name: Optional[str] = None
        データセットの名前 (省略するとディレクトリの名前)
    overwrite: bool = False
        既にあるデータセットを置き換えるか
    loader: Optional[str] = None
        読み込みの名前, None の場合は推定 (load_data と同じ)
    **options: Any
        読み込みの設定 (load_data と同じ)

    Returns
    ----------
    info: dict
        書き出した dataset.yaml の中身

    Raises
    ----------
    ValueError
        表でない場合, 列の誤り, 行が無い場合, 行を見分ける列に欠けか同じ
        値がある場合と, 分割の誤り
    """
    # 表と列
    data = table_of(source, loader, options)
    names = group_columns(data.frame, columns, id)
    frame = data.frame[names].reset_index(drop=True).copy()
    frame.columns = [str(column) for column in names]
    if frame.empty:
        raise ValueError("群を作る表に行がありません")

    # 件の id (行を見分ける列か行の位置)
    if id is None:
        ids = [str(index) for index in range(len(frame))]
    else:
        keys = data.frame[id].reset_index(drop=True)
        if keys.isna().any():
            raise ValueError(f"行を見分ける列 {id} に欠けた値があります")
        ids = [str(key) for key in keys]
        if len(set(ids)) < len(ids):
            raise ValueError(f"行を見分ける列 {id} に同じ値があります")
    if (split or {}).get("method") == "group":
        raise ValueError(
            "群分けの表の分割は random か none にしてください (グループの"
            "列を持たないため)"
        )
    labels, split_info, notes = split_items(ids, None, split)
    split_info.pop("unlabeled", None)
    notes = column_notes(frame) + notes

    # 項目 (群を作る列ごとに, 数は value, 他は文字にした class)
    entries, values = {}, {}
    for column in frame.columns:
        if is_number(frame[column]):
            # pandas の欠けを持てる型は float64 (numpy の型の名前にする)
            series = frame[column]
            if not isinstance(series.dtype, np.dtype):
                series = series.astype("float64")
            values[column] = series
            entries[column] = {
                "kind": NUMBER_KIND, "role": "input", "placement": "column",
                "dtype": str(series.dtype),
            }
        else:
            values[column] = frame[column].map(
                lambda value: None if pd.isna(value) else str(value)
            ).astype(object)
            entries[column] = {
                "kind": CATEGORY_KIND, "role": "input",
                "placement": "column", "dtype": "str",
            }
    check_entry_names(entries)
    rows = [
        {"id": key, "split": label,
         **{column: values[column].iloc[index] for column in frame.columns}}
        for index, (key, label) in enumerate(zip(ids, labels))
    ]

    # 説明と表 (失敗したら一時ディレクトリを削除)
    final, target = start_writing(path, overwrite)
    try:
        info = {
            "format_version": ITEMS_FORMAT_VERSION,
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "group",
            "target": {"kind": "none", "entries": []},
            "entries": entries,
            "items": None,
            "split": split_info,
            "splits": None,
            "source": {
                **shared_source(data),
                "id": None if id is None else str(id),
            },
            "notes": notes,
        }
        info = write_items(target, final, rows, info)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise

    return info


def dataset_frame(dataset: ItemDataset) -> tuple[pd.DataFrame, list]:
    """
    群分けの版 2 のデータセットから, 群を作る列の表を取る関数

    Parameters
    ----------
    dataset: ItemDataset
        load_items で読んだデータセット

    Returns
    ----------
    frame: pd.DataFrame
        全ての件の表 (id, split と, 入力の項目の列. 件の表の順)
    names: list
        入力の項目の名前

    Raises
    ----------
    ValueError
        受け持ちが group でない場合
    """
    # 群分けのデータセットの入力の項目だけ
    if dataset.info.get("role") != "group":
        raise ValueError(
            f"受け持ちが {dataset.info.get('role')} のデータセットです. "
            "群の要約は受け持ちが group のデータセットで求めます"
        )
    names = [key for key, spec in dataset.entries.items()
             if spec.get("role") == "input"]
    frame = dataset.items.reset_index(drop=True)

    return frame, names


def make_encoding(frame: pd.DataFrame) -> dict:
    """
    行から, 列の符号化を決める関数

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列

    Returns
    ----------
    encoding: dict
        numeric (数の列ごとの mean と std. 標準化する) と categories
        (種類の列ごとの値の並び. 初めて出た順. 値ごとに 0 と 1 の列に
        する)
    """
    # 散らばりの無い数の列は 1 で割る (ML-Quest の骨組みと同じ)
    numeric: dict = {}
    categories: dict = {}
    for name in frame.columns:
        column = frame[name]
        if is_number(column):
            values = column.astype(float)
            std = float(values.std(ddof=0)) if values.notna().any() else 0.0
            numeric[name] = {
                "mean": float(values.mean()) if values.notna().any() else 0.0,
                "std": std if std > 0 else 1.0,
            }
        else:
            categories[name] = list(dict.fromkeys(column.map(category_text)))
    encoding = {"numeric": numeric, "categories": categories}

    return encoding


def encode_rows(frame: pd.DataFrame, encoding: dict) -> np.ndarray:
    """
    行を, 列の符号化で数の並びにする関数

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列
    encoding: dict
        make_encoding の符号化

    Returns
    ----------
    features: np.ndarray
        (行の数, 符号化した行の幅) の float64 (float32 に丸めた値. 数の
        列は標準化して欠けた値は 0, 種類の列は値ごとの 0 と 1)
    """
    # 数の列を前に, 種類の列を後ろに並べる
    parts = []
    for name, stats in encoding["numeric"].items():
        values = (frame[name].astype(float) - stats["mean"]) / stats["std"]
        parts.append(values.fillna(0.0).to_numpy()[:, None])
    for name, values in encoding["categories"].items():
        text = frame[name].map(category_text).to_numpy()
        parts.append(np.stack([text == value for value in values], 1))
    # 骨組みと同じ float32 の値にしてから float64 で数える
    features = np.concatenate(parts, 1).astype(np.float32).astype(np.float64)

    return features


def assign(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    点を最も近い群の中心に割り当てる関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    centers: np.ndarray
        (群の数, 幅) の群の中心

    Returns
    ----------
    labels: np.ndarray
        (件数,) の群の番号 (同じ距離は番号の小さい群)
    """
    # 二乗距離の最も小さい中心
    distance = (
        (points ** 2).sum(1)[:, None] - 2 * points @ centers.T
        + (centers ** 2).sum(1)[None, :]
    )
    labels = distance.argmin(1)

    return labels


def kmeans_once(
    points: np.ndarray, k: int, rng: np.random.Generator, iterations: int
) -> np.ndarray:
    """
    k-means++ で中心を置き, Lloyd 法で群の中心を求める関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    k: int
        群の数
    rng: np.random.Generator
        中心の初めの置き方の乱数
    iterations: int
        Lloyd 法の繰り返しの上限

    Returns
    ----------
    centers: np.ndarray
        (群の数, 幅) の群の中心 (点の無い群の中心は動かさない)
    """
    # k-means++ (近い中心から遠い点ほど選ばれやすい)
    first = int(rng.integers(len(points)))
    chosen = [points[first]]
    closest = ((points - points[first]) ** 2).sum(1)
    for _ in range(1, k):
        total = closest.sum()
        index = (int(rng.choice(len(points), p=closest / total))
                 if total > 0 else int(rng.integers(len(points))))
        chosen.append(points[index])
        closest = np.minimum(closest, ((points - points[index]) ** 2).sum(1))
    centers = np.stack(chosen)

    # 割り当てと群の平均を, 中心が動かなくなるまで
    for _ in range(iterations):
        labels = assign(points, centers)
        moved = centers.copy()
        for group in range(k):
            members = points[labels == group]
            if len(members):
                moved[group] = members.mean(0)
        if np.allclose(moved, centers):
            break
        centers = moved

    return centers


def kmeans(
    points: np.ndarray, k: int, rng: np.random.Generator,
    iterations: int = ITERATIONS, inits: int = INITS
) -> np.ndarray:
    """
    中心の置き方を変えて k-means を繰り返し, 群の中心までの二乗距離の和が
    最も小さい群の中心を返す関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    k: int
        群の数
    rng: np.random.Generator
        中心の初めの置き方の乱数
    iterations: int = ITERATIONS
        Lloyd 法の繰り返しの上限
    inits: int = INITS
        中心の置き方を変える回数

    Returns
    ----------
    centers: np.ndarray
        (群の数, 幅) の群の中心
    """
    # 悪い置き方から始めた局所解を避ける
    best, smallest = None, np.inf
    for _ in range(max(int(inits), 1)):
        centers = kmeans_once(points, k, rng, iterations)
        inertia = ((points - centers[assign(points, centers)]) ** 2).sum()
        if inertia < smallest:
            best, smallest = centers, inertia

    return best


def silhouette(
    points: np.ndarray, labels: np.ndarray, size: int,
    rng: np.random.Generator
) -> float:
    """
    群のシルエット係数 (群のまとまり) を数える関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    labels: np.ndarray
        (件数,) の群の番号
    size: int
        数える点の数の上限 (多ければ無作為に選ぶ)
    rng: np.random.Generator
        点を選ぶ乱数

    Returns
    ----------
    score: float
        -1 から 1 (群が 1 つなら 0. 1 点だけの群の点は 0)
    """
    # 点を選び, 全ての組の距離
    if len(points) > size:
        picked = rng.choice(len(points), size, replace=False)
        points, labels = points[picked], labels[picked]
    groups = np.unique(labels)
    if len(groups) < 2:
        return 0.0
    squared = (points ** 2).sum(1)
    distance = np.sqrt(np.maximum(
        squared[:, None] - 2 * points @ points.T + squared[None, :], 0.0
    ))

    # 自分の群の他の点までの平均 a と, 最も近い他の群の点までの平均 b
    member = labels[:, None] == groups[None, :]
    sums = distance @ member
    counts = member.sum(0)
    own = member.argmax(1)
    rows = np.arange(len(points))
    a = sums[rows, own] / np.maximum(counts[own] - 1, 1)
    other = sums / counts
    other[rows, own] = np.inf
    b = other.min(1)
    scores = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    scores[counts[own] == 1] = 0.0
    score = float(scores.mean())

    return score


def adjusted_rand(first: np.ndarray, second: np.ndarray) -> float:
    """
    2 つの群の割り当ての一致 (調整ランド指数) を数える関数

    Parameters
    ----------
    first: np.ndarray
        (件数,) の群の番号
    second: np.ndarray
        (件数,) の群の番号 (番号の付け方は first と違ってよい)

    Returns
    ----------
    score: float
        1 は同じ群, 0 は偶然と同じ (負もある)
    """
    # 同じ群に入る組の数を, 群の組の表から数える
    first = np.unique(first, return_inverse=True)[1]
    second = np.unique(second, return_inverse=True)[1]
    table = np.zeros((first.max() + 1, second.max() + 1))
    np.add.at(table, (first, second), 1)
    pairs = (table * (table - 1) / 2).sum()
    rows = (table.sum(1) * (table.sum(1) - 1) / 2).sum()
    columns = (table.sum(0) * (table.sum(0) - 1) / 2).sum()
    total = len(first) * (len(first) - 1) / 2
    expected = rows * columns / total if total else 0.0
    top = (rows + columns) / 2
    score = 1.0 if top == expected else float(
        (pairs - expected) / (top - expected)
    )

    return score


def group_jaccard(
    labels: np.ndarray, again: np.ndarray, k: int
) -> np.ndarray:
    """
    群ごとに, 作り直した群のうち最も重なる群との Jaccard 係数を数える関数

    Parameters
    ----------
    labels: np.ndarray
        (件数,) の元の群の番号
    again: np.ndarray
        (件数,) の作り直した群の番号
    k: int
        群の数

    Returns
    ----------
    scores: np.ndarray
        (群の数,) の 0 から 1 (行の無い群は NaN)
    """
    # 元の群と作り直した群の, 共通の行と合わせた行の数の比の最も大きい値
    scores = np.full(k, np.nan)
    for group in range(k):
        members = labels == group
        if not members.any():
            continue
        scores[group] = max(
            (members & (again == other)).sum()
            / (members | (again == other)).sum()
            for other in np.unique(again)
        )

    return scores


def measure_stability(
    points: np.ndarray, labels: np.ndarray, k: int, seed: int, repeats: int,
    method: str, fraction: Optional[float]
) -> dict:
    """
    群を作り直し, 元の群との一致を数える関数

    Parameters
    ----------
    points: np.ndarray
        (行の数, 幅) の点
    labels: np.ndarray
        (行の数,) の元の群の番号
    k: int
        群の数
    seed: int
        元の群の乱数の種 (作り直しは seed + 1 + 回の番号)
    repeats: int
        作り直す回数
    method: str
        数え方 (seed は種を変えて全ての行で, sample は行の fraction の割合
        を引き直して作り直す)
    fraction: Optional[float]
        sample で引く行の割合 (seed では None)

    Returns
    ----------
    found: dict
        ari (調整ランド指数の平均), ari_min (最も低い値), jaccard (群ごと
        の Jaccard 係数の平均. 行の無い群は NaN)
    """
    # 作り直すたびに別の種. 標本を変えるなら行の一部を引き直す (ML-Quest の
    # 骨組みの stability と同じ. 骨組みの学習用の行はここでは全ての行)
    scores, overlaps = [], []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 1 + repeat)
        rows = np.arange(len(points))
        if method == "sample":
            size = max(k, int(len(points) * fraction))
            rows = rng.choice(rows, size, replace=False)
        again = assign(points, kmeans(points[rows], k, rng))
        scores.append(adjusted_rand(labels, again))
        overlaps.append(group_jaccard(labels, again, k))
    overlaps = np.array(overlaps)
    jaccard = np.array([
        column[~np.isnan(column)].mean() if (~np.isnan(column)).any()
        else np.nan for column in overlaps.T
    ])
    found = {"ari": float(np.mean(scores)), "ari_min": float(np.min(scores)),
             "jaccard": jaccard}

    return found


def summary_rows(
    frame: pd.DataFrame, labels: np.ndarray, keys: list,
    jaccard: Optional[np.ndarray]
) -> list[dict]:
    """
    群ごとの行の数, 割合, 安定性と, 群を作った列の要約を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        全ての行の, 群を作る列 (元の値)
    labels: np.ndarray
        (全ての行の数,) の群の位置 (keys の番号)
    keys: list
        群の位置から群の値 (行の group)
    jaccard: Optional[np.ndarray]
        (群の数,) の群ごとの Jaccard 係数の平均 (None は数えていない)

    Returns
    ----------
    rows: list[dict]
        群の順の行と, 最後に全ての行 (group は「全体」)
    """
    # 数の列は平均, 種類の列は最も多い値と割合 (行の無い群は None)
    rows = []
    for position in [*range(len(keys)), None]:
        part = frame if position is None else frame[labels == position]
        overlap = (None if position is None or jaccard is None
                   else jaccard[position])
        row: dict = {
            "group": ALL_ROWS if position is None else keys[position],
            "rows": len(part), "share": len(part) / len(frame),
            "stability": (None if overlap is None or np.isnan(overlap)
                          else float(overlap)),
        }
        for name in frame.columns:
            column = part[name]
            if is_number(frame[name]):
                row[name] = (float(column.astype(float).mean())
                             if column.notna().any() else None)
                continue
            counts = column.map(category_text).value_counts()
            row[name] = (f"{counts.index[0]} "
                         f"({counts.iloc[0] / len(part) * 100:.0f} %)"
                         if len(part) else None)
        rows.append(row)

    return rows


def given_labels(
    frame: pd.DataFrame, labels: Any
) -> tuple[np.ndarray, list]:
    """
    渡した群の割り当てを, 群の位置と群の値にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        全ての行の表
    labels: Any
        行ごとの群の並びか, 表の群の列の名前

    Returns
    ----------
    result: tuple[np.ndarray, list]
        (行の数,) の群の位置と, 群の位置から群の値 (小さい順)

    Raises
    ----------
    ValueError
        列が無い場合, 長さが行の数と違う場合, 欠けた群がある場合
    """
    # 並びはそのまま, 他は表の列の名前
    if isinstance(labels, (list, tuple, np.ndarray, pd.Series)):
        values = pd.Series(list(labels))
    elif labels in frame.columns:
        values = frame[labels].reset_index(drop=True)
    else:
        raise ValueError(f"表に群の列 {labels} がありません")
    if len(values) != len(frame):
        raise ValueError(
            f"群の割り当ての数 {len(values)} が行の数 {len(frame)} と"
            "違います"
        )
    if values.isna().any():
        raise ValueError("群の割り当てに欠けた値があります")

    # 群の値の小さい順 (数と文字が混ざれば文字の順)
    distinct = list(dict.fromkeys(values))
    try:
        keys = sorted(distinct)
    except TypeError:
        keys = sorted(distinct, key=str)
    position = {key: index for index, key in enumerate(keys)}
    result = (np.array([position[value] for value in values]), keys)

    return result


def check_count(value: Any, label: str, lowest: int) -> int:
    """
    群の数と作り直しの回数が, 決まった値以上の整数かを確かめる関数

    Parameters
    ----------
    value: Any
        値
    label: str
        引数の名前 (誤りの文に使う)
    lowest: int
        いちばん小さい値

    Returns
    ----------
    count: int
        値

    Raises
    ----------
    ValueError
        整数でないか小さい場合
    """
    # 真偽値は整数とみなさない
    whole = isinstance(value, (int, np.integer)) and not isinstance(
        value, bool
    )
    if not whole or value < lowest:
        raise ValueError(
            f"{label} は {lowest} 以上の整数にしてください (今は {value})"
        )
    count = int(value)

    return count


def check_method(method: Any, fraction: Any) -> Optional[float]:
    """
    群の安定性の数え方と, 引く行の割合を確かめる関数

    Parameters
    ----------
    method: Any
        数え方 (STABILITY_METHODS のどれか)
    fraction: Any
        sample で引く行の割合 (seed では確かめない)

    Returns
    ----------
    share: Optional[float]
        引く行の割合 (seed では None)

    Raises
    ----------
    ValueError
        知らない数え方の場合と, 割合が 0 より大きく 1 以下の数でない場合
    """
    # 割合は sample だけで確かめる (真偽値は数とみなさない)
    if method not in STABILITY_METHODS:
        raise ValueError(
            f"群の安定性の数え方 stability は {', '.join(STABILITY_METHODS)}"
            f" のどれかにしてください (今は {method})"
        )
    if method != "sample":
        return None
    number = isinstance(fraction, (int, float, np.integer, np.floating))
    if (not number or isinstance(fraction, bool)
            or not 0 < fraction <= 1):
        raise ValueError(
            "引く行の割合 fraction は 0 より大きく 1 以下にしてください "
            f"(今は {fraction})"
        )
    share = float(fraction)

    return share


def describe_groups(
    source: Union[str, Path, Data, ItemDataset], columns: Any = None,
    k: Optional[int] = None, labels: Any = None, seed: int = 0,
    repeats: int = REPEATS, stability: str = "seed",
    fraction: Optional[float] = FRACTION, loader: Optional[str] = None,
    **options: Any
) -> dict:
    """
    学習に通さず, 表の列から群を作るか渡した群を使い, 群の要約を返す関数

    群は ML-Quest のまとめるの骨組みと同じ符号化 (数の列は標準化,
    他の列は値ごとの 0 と 1) をかけた列の特徴を, そのまま k-means で分けた
    もの (骨組みの比べる相手と同じ作り方). 群の要約の行は骨組みの
    evaluations/group_summary.json の行と同じ形

    Parameters
    ----------
    source: Union[str, Path, Data, ItemDataset]
        表のファイル, load_data で読んだデータか, export_group_rows で
        書き出して load_items で読んだデータセット
    columns: Any = None
        群を作る列の名前の並び, None の場合はデータセットの入力の項目
        (表とデータでは要る)
    k: Optional[int] = None
        群の数 (2 以上. labels を渡さなければ要る)
    labels: Any = None
        行ごとの群の並びか表の群の列の名前, None の場合は k-means で作る
    seed: int = 0
        k-means の乱数の種
    repeats: int = REPEATS
        群の安定性を数える作り直しの回数 (1 以上)
    stability: str = "seed"
        群の安定性の数え方 (ML-Quest の問題の設定の答え group.stability と
        同じ値. seed は種を変えて, sample は行の fraction の割合を引き直して
        作り直す)
    fraction: Optional[float] = FRACTION
        sample で引く行の割合 (0 より大きく 1 以下. seed では使わない)
    loader: Optional[str] = None
        読み込みの名前, None の場合は推定 (load_data と同じ)
    **options: Any
        読み込みの設定 (load_data と同じ)

    Returns
    ----------
    summary: dict
        JSON にできる群の要約 (groups, stability, silhouette など)

    Raises
    ----------
    ValueError
        列の誤り, 群の数の誤り, 群の数が行の数より多い場合, 符号化した行の
        幅が大きすぎる場合, 群の割り当ての誤り, 安定性の数え方の誤り
    """
    # 表と群を作る列
    if isinstance(source, ItemDataset):
        if loader is not None or options:
            raise ValueError(
                "データセットを渡す場合は, 読み込みの名前と設定を渡さないで"
                "ください"
            )
        table, inputs = dataset_frame(source)
        names = group_columns(table, inputs if columns is None else columns)
        place = dict(source.info.get("source") or {})
    else:
        data = table_of(source, loader, options)
        table = data.frame.reset_index(drop=True)
        names = group_columns(table, columns)
        place = dict(data.source)
    frame = table[names].copy()
    frame.columns = [str(name) for name in names]
    if frame.empty:
        raise ValueError("群を作る表に行がありません")
    notes = column_notes(frame)

    # 符号化 (全ての行で決める. 幅は表を作る前に確かめる)
    encoding = make_encoding(frame)
    width = len(encoding["numeric"]) + sum(
        len(values) for values in encoding["categories"].values()
    )
    if width > MAX_WIDTH:
        raise ValueError(
            f"符号化した行の幅が {width} で, 上限 {MAX_WIDTH} を超えます. "
            "値の多い種類の列を群を作る列から外してください"
        )
    features = encode_rows(frame, encoding)

    # 群 (渡した割り当てか, 列の特徴をそのまま k-means で)
    # 数え方は labels でも確かめる (labels では使わない)
    rng = np.random.default_rng(seed)
    found = None
    fraction = check_method(stability, fraction)
    if labels is None:
        count = check_count(k, "群の数 k", 2)
        repeats = check_count(repeats, "作り直しの回数 repeats", 1)
        if count > len(frame):
            raise ValueError(
                f"群の数 {count} が行の数 {len(frame)} より多いです"
            )
        positions = assign(features, kmeans(features, count, rng))
        keys = list(range(count))
        found = measure_stability(features, positions, count, seed,
                                  repeats, stability, fraction)
    else:
        if k is not None:
            raise ValueError(
                "群の割り当て (labels) を渡す場合は, 群の数 k を渡さないで"
                "ください"
            )
        positions, keys = given_labels(table, labels)
    rows = summary_rows(
        frame, positions, keys, None if found is None else found["jaccard"]
    )
    empty = [row["group"] for row in rows[:-1] if row["rows"] == 0]
    if empty:
        notes.append(f"行の無い群が {len(empty)} 個あります")
    if len(keys) < 2:
        notes.append("群が 1 つだけで, 群を比べられません")

    # 群の安定性とまとまり (点を選ぶ乱数は種から作り直す. ML-Quest の
    # 骨組みと同じ)
    summary = {
        "method": "kmeans" if labels is None else "labels",
        "k": len(keys),
        "rows": len(frame),
        "columns": {
            "numeric": list(encoding["numeric"]),
            "categorical": list(encoding["categories"]),
        },
        "width": int(features.shape[1]),
        "seed": seed if labels is None else None,
        "groups": rows,
        "stability": None if found is None else {
            "method": stability, "repeats": repeats,
            "fraction": fraction,
            "ari": found["ari"], "ari_min": found["ari_min"],
        },
        "silhouette": silhouette(
            features, positions, SAMPLE, np.random.default_rng(seed)
        ),
        "source": place,
        "notes": notes,
    }
    summary = to_json(summary)

    return summary
