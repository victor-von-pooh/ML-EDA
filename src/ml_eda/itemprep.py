import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# 当てはめに使う学習用の件数の上限 (画像の平均など)
FIT_SAMPLE = 500


@dataclass
class ItemStep:
    # 手順の名前 (レシピの item_steps の op)
    name: str
    # 使える種類
    kind: str
    # 当てはめ (学習用の中身から値を学ぶ. 学ばない手順は None)
    fit: Optional[Callable[[list, dict], dict]]
    # 1 件の中身を変える
    apply: Callable[[Any, dict, dict], Any]
    # 引数の説明
    params: dict = field(default_factory=dict)
    # 説明
    description: str = ""
    # 当てはめに全ての学習用の件を使うか (語を数えるだけの軽いもの)
    fit_all: bool = False
    # 省けない引数
    required: tuple = ()


# 中身の手順の登録簿
ITEM_STEPS: dict[str, ItemStep] = {}


def register_item_step(step: ItemStep) -> None:
    """
    中身の手順を登録簿に加える関数

    Parameters
    ----------
    step: ItemStep
        加える手順

    Returns
    ----------
    None
    """
    # 名前で登録
    ITEM_STEPS[step.name] = step


def describe_item_steps() -> list[dict]:
    """
    中身の手順の一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    steps: list[dict]
        名前, 種類, 引数の説明, 説明, 当てはめるか の辞書の並び
    """
    # JSON にできる形に変換
    steps = [
        {
            "op": step.name, "kind": step.kind, "params": dict(step.params),
            "description": step.description, "fits": step.fit is not None,
        }
        for step in ITEM_STEPS.values()
    ]

    return steps


def resize(image: np.ndarray, params: dict, state: dict) -> np.ndarray:
    """
    画像の大きさを変える関数

    Parameters
    ----------
    image: np.ndarray
        (高さ, 幅) か (高さ, 幅, チャンネル) の配列
    params: dict
        width, height
    state: dict
        使わない

    Returns
    ----------
    resized: np.ndarray
        大きさを変えた配列
    """
    # Pillow で縮小と拡大 (uint8 にそろえてから)
    from PIL import Image

    width, height = int(params["width"]), int(params["height"])
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    resized = np.asarray(
        Image.fromarray(array).resize((width, height), Image.BILINEAR)
    )

    return resized


def grayscale(image: np.ndarray, params: dict, state: dict) -> np.ndarray:
    """
    画像を灰色 (1 チャンネル) にする関数

    Parameters
    ----------
    image: np.ndarray
        配列
    params: dict
        使わない
    state: dict
        使わない

    Returns
    ----------
    gray: np.ndarray
        (高さ, 幅) の配列
    """
    # 色の重みで混ぜる (透明度は除く). 他の手順と同じく 0 から 255 の uint8
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        rgb = array[:, :, :3]
        weights = np.array([0.299, 0.587, 0.114][:rgb.shape[2]])
        array = (rgb * weights).sum(axis=2) / weights.sum()
    gray = np.clip(np.round(array), 0, 255).astype(np.uint8)

    return gray


def rgb(image: np.ndarray, params: dict, state: dict) -> np.ndarray:
    """
    画像を 3 チャンネル (RGB) にする関数 (灰色とカラーが混ざる場合)

    Parameters
    ----------
    image: np.ndarray
        配列
    params: dict
        使わない
    state: dict
        使わない

    Returns
    ----------
    color: np.ndarray
        (高さ, 幅, 3) の uint8 の配列
    """
    # 灰色は 3 つ重ね, 透明度は落とす
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    color = array[:, :, :3].astype(np.uint8)

    return color


def as_channels(image: np.ndarray) -> np.ndarray:
    """
    画像を (チャンネル, 高さ, 幅) の float32 にする関数

    Parameters
    ----------
    image: np.ndarray
        (高さ, 幅) か (高さ, 幅, チャンネル) の配列

    Returns
    ----------
    array: np.ndarray
        (チャンネル, 高さ, 幅) の配列
    """
    # PyTorch の画像の並びにそろえる
    array = np.asarray(image, dtype=np.float32)
    array = array[None] if array.ndim == 2 else array.transpose(2, 0, 1)

    return array


def fit_normalize_image(items: list, params: dict) -> dict:
    """
    画素の平均と標準偏差をチャンネルごとに求める関数

    Parameters
    ----------
    items: list
        学習用の画像 (前の手順をかけたもの)
    params: dict
        使わない

    Returns
    ----------
    state: dict
        チャンネルごとの mean と std (0 から 1 の値で)
    """
    # チャンネルごとの和と二乗和を 1 件ずつ足す (全てを並べない)
    total = squares = None
    count = 0
    for item in items:
        array = as_channels(item).astype(np.float64) / 255.0
        flat = array.reshape(array.shape[0], -1)
        if total is None:
            total = np.zeros(flat.shape[0])
            squares = np.zeros(flat.shape[0])
        if flat.shape[0] != len(total):
            raise ValueError(
                "チャンネル数がそろいません. 先に grayscale か rgb で"
                "そろえてください"
            )
        total += flat.sum(axis=1)
        squares += (flat ** 2).sum(axis=1)
        count += flat.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    state = {
        "mean": mean.round(6).tolist(),
        "std": np.where(std > 0, std, 1.0).round(6).tolist(),
    }

    return state


def normalize_image(
    image: np.ndarray, params: dict, state: dict
) -> np.ndarray:
    """
    画像を (チャンネル, 高さ, 幅) にし, 平均と標準偏差で正規化する関数

    Parameters
    ----------
    image: np.ndarray
        画像
    params: dict
        使わない
    state: dict
        チャンネルごとの mean と std

    Returns
    ----------
    normalized: np.ndarray
        (チャンネル, 高さ, 幅) の float32 の配列
    """
    # 0 から 1 にしてから平均を引き, 標準偏差で割る (チャンネル数を確認)
    array = as_channels(image) / 255.0
    if array.shape[0] != len(state["mean"]):
        raise ValueError(
            f"チャンネル数 {array.shape[0]} が当てはめたとき "
            f"({len(state['mean'])}) と違います. grayscale か rgb で"
            "そろえてください"
        )
    mean = np.array(state["mean"], dtype=np.float32)[:, None, None]
    std = np.array(state["std"], dtype=np.float32)[:, None, None]
    normalized = ((array - mean) / std).astype(np.float32)

    return normalized


def normalize_item_text(text: Any, params: dict, state: dict) -> str:
    """
    文の表記のゆれ (全角と半角, 前後の空白, 大文字と小文字) をそろえる関数

    Parameters
    ----------
    text: Any
        文
    params: dict
        lowercase (小文字にするか)
    state: dict
        使わない

    Returns
    ----------
    normalized: str
        そろえた文
    """
    from ml_eda.recipe import normal_text

    normalized = normal_text(
        str(text), bool(params.get("lowercase", False))
    )

    return normalized


def tokenize(text: str, params: dict) -> list[str]:
    """
    文を語に分ける関数

    Parameters
    ----------
    text: str
        文
    params: dict
        lowercase (小文字にするか, 既定 true), unit (word か char)

    Returns
    ----------
    tokens: list[str]
        語 (unit が char なら文字) の並び
    """
    # 語は文字と数字の並び (漢字, かなは 1 文字ずつ), 記号も 1 つずつ.
    # 文字は空白以外の 1 文字ずつ
    value = str(text)
    if params.get("lowercase", True):
        value = value.lower()
    if params.get("unit", "word") == "char":
        return [c for c in value if not c.isspace()]
    tokens = re.findall(
        r"[\u3040-\u30ff\u3400-\u9fff]|[^\W_\u3040-\u30ff\u3400-\u9fff]+"
        r"|[^\w\s]", value
    )

    return tokens


def fit_vocabulary(items: list, params: dict) -> dict:
    """
    学習用の文から語彙を作る関数

    Parameters
    ----------
    items: list
        学習用の文
    params: dict
        max_tokens (語彙の上限, 既定 20000. 部分語は語彙の大きさで既定
        8000), min_count (最小の出現数, 既定 1. 部分語では使わない),
        lowercase, unit (word, char, subword)

    Returns
    ----------
    state: dict
        語彙 (0 は埋め草, 1 は語彙に無い語. 部分語は区切りの JSON の
        tokenizer も)
    """
    # 部分語は学習用の文での BPE の区切りの作成
    if params.get("unit", "word") == "subword":
        from ml_eda.textunits import fit_subword, subword_vocabulary

        lower = params.get("lowercase", True)
        serialized = fit_subword(
            [str(item).lower() if lower else str(item) for item in items],
            int(params.get("max_tokens", 8000))
        )
        # 語彙の並びは区切りの JSON と重なるが, 番号を語に戻す使い手の
        # ため (前と後の表示, 語彙の大きさ) の保存
        return {
            "vocabulary": subword_vocabulary(serialized),
            "tokenizer": serialized,
        }

    # 頻度の上位を語彙にする
    counts: Counter = Counter()
    for item in items:
        counts.update(tokenize(item, params))
    limit = int(params.get("max_tokens", 20000))
    minimum = int(params.get("min_count", 1))
    words = [w for w, c in counts.most_common(limit) if c >= minimum]
    state = {"vocabulary": ["<pad>", "<unk>"] + words}

    return state


# 語彙から番号への辞書 (語彙ごとに 1 度だけ作る)
INDEX_CACHE: dict[int, dict[str, int]] = {}


def vocabulary_index(vocabulary: list) -> dict[str, int]:
    """
    語彙の語から番号への辞書を返す関数 (同じ語彙なら作り直さない)

    Parameters
    ----------
    vocabulary: list
        語彙

    Returns
    ----------
    index: dict[str, int]
        語から番号
    """
    # 語彙の並びの id で覚える (多くなったら捨てる)
    key = id(vocabulary)
    index = INDEX_CACHE.get(key)
    if index is None or len(index) != len(vocabulary):
        if len(INDEX_CACHE) > 8:
            INDEX_CACHE.clear()
        index = {word: k for k, word in enumerate(vocabulary)}
        INDEX_CACHE[key] = index

    return index


def encode_text(text: str, params: dict, state: dict) -> np.ndarray:
    """
    文を語の番号の並び (決まった長さ) にする関数

    Parameters
    ----------
    text: str
        文
    params: dict
        length (並びの長さ, 既定 128), lowercase, unit
    state: dict
        語彙

    Returns
    ----------
    ids: np.ndarray
        (length,) の int64 の配列 (短い場合は 0 で埋め, 長い場合は切る)
    """
    # 長さで切り, 短い場合は 0 で埋めた並び
    length = int(params.get("length", 128))
    tokens = encode_tokens(text, params, state)[:length]
    ids = np.zeros(length, dtype=np.int64)
    ids[:len(tokens)] = tokens

    return ids


def encode_tokens(text: str, params: dict, state: dict) -> list[int]:
    """
    文を語の番号の並び (長さをそろえない) にする関数

    Parameters
    ----------
    text: str
        文
    params: dict
        lowercase, unit
    state: dict
        語彙 (部分語は区切りの JSON の tokenizer も)

    Returns
    ----------
    ids: list[int]
        語の番号 (語彙に無い語は 1)
    """
    # 部分語は保存した区切りでの分割
    if params.get("unit", "word") == "subword":
        from ml_eda.textunits import subword_tokenizer

        if "tokenizer" not in state:
            raise ValueError(
                "部分語の区切りがありません. vocabulary を当てはめ直して"
                "ください"
            )
        value = str(text)
        if params.get("lowercase", True):
            value = value.lower()
        ids = list(subword_tokenizer(state["tokenizer"]).encode(value).ids)
        return ids
    index = vocabulary_index(state["vocabulary"])
    ids = [index.get(token, 1) for token in tokenize(text, params)]

    return ids


def sample_points(cloud: Any, params: dict, state: dict) -> dict:
    """
    点群の点の数をそろえる関数 (多い場合は間引き, 少ない場合は重ねる)

    Parameters
    ----------
    cloud: Any
        points を持つ辞書
    params: dict
        points (そろえる点の数, 既定 1024), seed
    state: dict
        使わない

    Returns
    ----------
    sampled: dict
        点の数をそろえた points (と colors)
    """
    # 同じ点群からは毎回同じ点の選択 (有効な点が無い点群は誤り)
    points = np.asarray(cloud["points"], dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) == 0:
        raise ValueError("有効な点が無い点群があります")
    count = int(params.get("points", 1024))
    rng = np.random.default_rng(int(params.get("seed", 0)))
    replace = len(points) < count
    chosen = rng.choice(len(points), count, replace=replace)
    sampled = {"points": points[chosen]}
    colors = cloud.get("colors")
    if colors is not None:
        sampled["colors"] = np.asarray(colors)[valid][chosen]

    return sampled


def normalize_cloud(cloud: Any, params: dict, state: dict) -> dict:
    """
    点群の中心を原点にし, 最も遠い点までの距離を 1 にする関数

    Parameters
    ----------
    cloud: Any
        points を持つ辞書
    params: dict
        使わない
    state: dict
        使わない

    Returns
    ----------
    normalized: dict
        変換した points (色はそのまま)
    """
    # 点群ごとに中心と大きさをそろえる
    points = np.asarray(cloud["points"], dtype=np.float64)
    centered = points - points.mean(axis=0)
    radius = float(np.linalg.norm(centered, axis=1).max(initial=0.0))
    normalized = {
        **cloud, "points": centered / radius if radius > 0 else centered
    }

    return normalized


def to_array(item: Any, kind: str) -> np.ndarray:
    """
    手順の後の中身を, 学習に渡す配列にする関数

    Parameters
    ----------
    item: Any
        中身
    kind: str
        種類

    Returns
    ----------
    array: np.ndarray
        画像は (チャンネル, 高さ, 幅) の float32 (0 から 1, 正規化した
        場合はその値), 点群は (点の数, 3) の float32, 文は番号の並び
    """
    # 種類ごとの形にそろえる
    if kind == "pointcloud" and isinstance(item, dict):
        return np.asarray(item["points"], dtype=np.float32)
    if kind == "image":
        image = np.asarray(item)
        if image.dtype == np.bool_:
            image = image.astype(np.uint8) * 255
        if image.dtype == np.uint8:
            return as_channels(image) / 255.0
        return image.astype(np.float32)
    array = np.asarray(item)

    return array


def check_item_steps(steps: Any, kind: str) -> None:
    """
    中身の手順の並びを, 中身を読まずに確かめる関数

    Parameters
    ----------
    steps: Any
        item_steps
    kind: str
        種類

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        並びでない, 知らない手順, 種類に合わない手順, 引数が足りない,
        文に vocabulary が無い場合
    """
    # 1 つずつ, 名前と種類と省けない引数を確かめる
    if not isinstance(steps, list):
        raise ValueError("item_steps は手順の並び (リスト) にしてください")
    for spec in steps:
        op = spec.get("op") if isinstance(spec, dict) else spec
        step = ITEM_STEPS.get(op) if isinstance(op, str) else None
        if step is None or step.kind != kind:
            raise ValueError(f"{kind} に使えない中身の手順です: {op}")
        missing = [name for name in step.required if spec.get(name) is None]
        if missing:
            raise ValueError(
                f"{op} の引数 {', '.join(missing)} を決めてください"
            )

        # 文の区切り方 (部分語は tokenizers が要る)
        if op == "vocabulary":
            unit = spec.get("unit", "word")
            if unit not in ("word", "char", "subword"):
                raise ValueError(
                    f"vocabulary の unit は word, char, subword のどれかに"
                    f"してください: {unit}"
                )
            if unit == "subword" and "state" not in spec:
                from ml_eda.textunits import has_subword

                if not has_subword():
                    raise ValueError(
                        "部分語 (unit: subword) には tokenizers が要ります "
                        "(pip install tokenizers)"
                    )

    # 文は語彙で番号にしないと学習に渡せない
    if kind == "text" and not any(s["op"] == "vocabulary" for s in steps):
        raise ValueError("文は vocabulary の手順で番号の並びにしてください")


def item_notes(frame: pd.DataFrame, steps: list, kind: str) -> list[str]:
    """
    読み込みの表 (大きさ, 色の形式, 点の数, 読めない件) から, 書き出しで
    失敗しそうなことを中身を読まずに挙げる関数

    Parameters
    ----------
    frame: pd.DataFrame
        書き出す件の, 読み込みの表の行
    steps: list
        item_steps
    kind: str
        種類

    Returns
    ----------
    notes: list[str]
        注意
    """
    # 読めない件 (画像は大きさが欠損, 点群は error がある件)
    ops = {spec["op"] for spec in steps}
    notes = []
    unreadable = 0
    if kind == "image" and "width" in frame.columns:
        unreadable = int(frame["width"].isna().sum())
    elif "error" in frame.columns:
        unreadable = int(frame["error"].notna().sum())
    if unreadable:
        how = "width が notna" if kind == "image" else "error が isna"
        notes.append(
            f"読めない件が {unreadable} 件あります. 書き出しで失敗するので, "
            f"表の手順 filter_rows ({how}) で除いてください"
        )

    # 形がそろわない (大きさ, 色の形式, 点の数)
    readable = frame.dropna(subset=["width"]) if (
        kind == "image" and "width" in frame.columns
    ) else frame
    if kind == "image" and "resize" not in ops and len(
        readable.groupby(["width", "height"])
    ) > 1:
        notes.append("画像の大きさがそろっていません. resize を足してください")
    if kind == "image" and not ops & {"rgb", "grayscale"} and (
        "channels" in readable.columns and readable["channels"].nunique() > 1
    ):
        notes.append(
            "灰色とカラーの画像が混ざっています. rgb か grayscale を"
            "足してください"
        )
    if kind == "pointcloud" and "sample_points" not in ops and (
        "points" in readable.columns and readable["points"].nunique() > 1
    ):
        notes.append(
            "点の数がそろっていません. sample_points を足してください"
        )

    return notes


def fit_item_steps(
    load: Callable[[int], Any], train: list[int], steps: list,
    seed: int = 0, limit: int = FIT_SAMPLE, limit_all: Optional[int] = None
) -> None:
    """
    中身の手順を, 前から順に学習用の件で当てはめる関数

    Parameters
    ----------
    load: Callable[[int], Any]
        元のデータの位置から中身を読む関数
    train: list[int]
        学習用の位置
    steps: list
        item_steps (当てはめた値を state に書き込む)
    seed: int = 0
        当てはめの見本を選ぶ乱数のシード
    limit: int = FIT_SAMPLE
        当てはめに使う件数の上限 (無作為に選ぶ)
    limit_all: Optional[int] = None
        全ての学習用の件で当てはめる手順 (語彙) の件数の上限. None なら
        全て

    Returns
    ----------
    None
    """
    # 学習用の無作為な見本 (語彙は全て, または limit_all まで)
    rng = np.random.default_rng(seed)
    sample = (
        rng.choice(train, limit, replace=False).tolist()
        if len(train) > limit else list(train)
    )
    if limit_all is None or len(train) <= limit_all:
        every = list(train)
    elif limit_all <= len(sample):
        every = sample[:limit_all]
    else:
        every = rng.choice(train, limit_all, replace=False).tolist()
    for number, spec in enumerate(steps):
        step = ITEM_STEPS[spec["op"]]
        if step.fit is None:
            continue
        chosen = every if step.fit_all else sample
        items = (apply_steps(load(i), steps[:number]) for i in chosen)
        params = {k: v for k, v in spec.items() if k not in ("op", "state")}
        spec["state"] = step.fit(list(items), params)


def run_item_steps(
    load: Callable[[int], Any], train: list[int], splits: dict,
    steps: list, kind: str, out_dir: Optional[Any] = None, seed: int = 0
) -> dict:
    """
    中身の手順を当てはめてかけ, 分割ごとの配列を作る関数

    Parameters
    ----------
    load: Callable[[int], Any]
        元のデータの位置から中身を読む関数
    train: list[int]
        学習用の位置
    splits: dict
        分割の名前から元のデータの位置の並び
    steps: list
        item_steps (当てはめた値を state に書き込む)
    kind: str
        種類
    out_dir: Optional[Any] = None
        配列を書くディレクトリ (Path). 指定すると 1 件ずつファイルに
        書き (全てをメモリに並べない), 読み込み専用の配列を返す
    seed: int = 0
        当てはめの見本を選ぶ乱数のシード

    Returns
    ----------
    arrays: dict
        分割の名前から (件数, ...) の配列

    Raises
    ----------
    ValueError
        知らない手順, 種類に合わない手順, 形のそろわない中身の場合
    """
    # 手順の確認と当てはめ
    check_item_steps(steps, kind)
    fit_item_steps(load, train, steps, seed)

    # 分割ごとに全ての手順をかけ, 最初の件の形にそろうかを確かめて積む
    arrays = {}
    for name, positions in splits.items():
        arrays[name] = stack_items(
            (to_array(apply_steps(load(i), steps), kind) for i in positions),
            len(positions), out_dir, name
        )

    return arrays


def stack_items(
    values: Any, count: int, out_dir: Optional[Any], name: str
) -> np.ndarray:
    """
    1 件ずつの配列を, 形を確かめながら 1 つの配列に積む関数

    Parameters
    ----------
    values: Any
        1 件ずつの配列を返すイテレータ
    count: int
        件数
    out_dir: Optional[Any]
        書くディレクトリ (Path). None ならメモリに積む
    name: str
        分割の名前 (ファイルの名前に使う)

    Returns
    ----------
    stacked: np.ndarray
        (件数, ...) の配列

    Raises
    ----------
    ValueError
        形がそろわない場合
    """
    # 最初の件で形を決め, 入れ物を用意してから 1 件ずつ入れる
    stacked: Optional[np.ndarray] = None
    for number, value in enumerate(values):
        if stacked is None:
            shape = (count, *value.shape)
            stacked = (
                np.lib.format.open_memmap(
                    out_dir / f"{name}_items.npy", mode="w+",
                    dtype=value.dtype, shape=shape
                )
                if out_dir is not None
                else np.empty(shape, dtype=value.dtype)
            )
        if value.shape != stacked.shape[1:]:
            raise ValueError(
                f"中身の形がそろいません ({stacked.shape[1:]} と "
                f"{value.shape}). 画像は resize (と grayscale か rgb), "
                "点群は sample_points で形をそろえてください"
            )
        stacked[number] = value
    if stacked is None:
        stacked = np.zeros((0,))
    if isinstance(stacked, np.memmap):
        stacked.flush()

    return stacked


def apply_steps(item: Any, steps: list) -> Any:
    """
    1 件の中身に手順を順にかける関数

    Parameters
    ----------
    item: Any
        中身
    steps: list
        item_steps (当てはめた値は state)

    Returns
    ----------
    result: Any
        手順をかけた中身
    """
    # 当てはめた値を使って順にかける
    result = item
    for spec in steps:
        step = ITEM_STEPS[spec["op"]]
        params = {k: v for k, v in spec.items() if k not in ("op", "state")}
        result = step.apply(result, params, spec.get("state", {}))

    return result


def register_builtin_item_steps() -> None:
    """
    組み込みの中身の手順を登録する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 画像
    register_item_step(ItemStep(
        "resize", "image", None, resize,
        {"width": "幅", "height": "高さ"}, "画像の大きさをそろえる",
        required=("width", "height")
    ))
    register_item_step(ItemStep(
        "grayscale", "image", None, grayscale, {}, "灰色 (1 チャンネル) にする"
    ))
    register_item_step(ItemStep(
        "rgb", "image", None, rgb, {},
        "3 チャンネル (RGB) にする (灰色とカラーが混ざる場合)"
    ))
    register_item_step(ItemStep(
        "normalize", "image", fit_normalize_image, normalize_image, {},
        "画素を学習用の平均と標準偏差で正規化する ((チャンネル, 高さ, 幅) に)"
    ))

    # 文
    register_item_step(ItemStep(
        "normalize_text", "text", None, normalize_item_text,
        {"lowercase": "小文字にするか (既定 false)"},
        "表記のゆれ (全角と半角, 前後の空白) をそろえる (NFKC)"
    ))
    register_item_step(ItemStep(
        "vocabulary", "text", fit_vocabulary, encode_text,
        {"max_tokens": "語彙の上限", "min_count": "語彙に入れる最小の出現数",
         "length": "並びの長さ", "lowercase": "小文字にするか",
         "unit": "word (語), char (文字), subword (部分語)"},
        "学習用の文で語彙を作り, 語の番号の並び (決まった長さ) にする",
        fit_all=True
    ))

    # 点群
    register_item_step(ItemStep(
        "sample_points", "pointcloud", None, sample_points,
        {"points": "そろえる点の数", "seed": "乱数のシード"},
        "点の数をそろえる (間引きと重ね)", required=("points",)
    ))
    register_item_step(ItemStep(
        "normalize_cloud", "pointcloud", None, normalize_cloud, {},
        "中心を原点にし, 最も遠い点までを 1 にする"
    ))


register_builtin_item_steps()
