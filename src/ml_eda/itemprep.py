import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

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
    # 色の重みで混ぜる (透明度は除く)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        rgb = array[:, :, :3]
        weights = np.array([0.299, 0.587, 0.114][:rgb.shape[2]])
        array = (rgb * weights).sum(axis=2) / weights.sum()
    gray = array

    return gray


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
    # 0 から 1 にした画素で集計
    stacked = np.concatenate(
        [as_channels(item).reshape(as_channels(item).shape[0], -1) / 255.0
         for item in items], axis=1
    )
    std = stacked.std(axis=1)
    state = {
        "mean": stacked.mean(axis=1).round(6).tolist(),
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
    # 0 から 1 にしてから平均を引き, 標準偏差で割る
    array = as_channels(image) / 255.0
    mean = np.array(state["mean"], dtype=np.float32)[:, None, None]
    std = np.array(state["std"], dtype=np.float32)[:, None, None]
    normalized = ((array - mean) / std).astype(np.float32)

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
    # 語は英数字の並び (日本語などは 1 文字ずつ), 文字は 1 文字ずつ
    value = str(text)
    if params.get("lowercase", True):
        value = value.lower()
    if params.get("unit", "word") == "char":
        return [c for c in value if not c.isspace()]
    tokens = re.findall(r"[a-z0-9]+|[^\sa-z0-9]", value)

    return tokens


def fit_vocabulary(items: list, params: dict) -> dict:
    """
    学習用の文から語彙を作る関数

    Parameters
    ----------
    items: list
        学習用の文
    params: dict
        max_tokens (語彙の上限, 既定 20000), min_count (最小の出現数,
        既定 1), lowercase, unit

    Returns
    ----------
    state: dict
        語彙 (0 は埋め草, 1 は語彙に無い語)
    """
    # 頻度の上位を語彙にする
    counts: Counter = Counter()
    for item in items:
        counts.update(tokenize(item, params))
    limit = int(params.get("max_tokens", 20000))
    minimum = int(params.get("min_count", 1))
    words = [w for w, c in counts.most_common(limit) if c >= minimum]
    state = {"vocabulary": ["<pad>", "<unk>"] + words}

    return state


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
    # 語彙に無い語は 1
    index = {word: k for k, word in enumerate(state["vocabulary"])}
    length = int(params.get("length", 128))
    tokens = tokenize(text, params)[:length]
    ids = np.zeros(length, dtype=np.int64)
    ids[:len(tokens)] = [index.get(token, 1) for token in tokens]

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
    # 同じ点群からは毎回同じ点を選ぶ
    points = np.asarray(cloud["points"], dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    count = int(params.get("points", 1024))
    rng = np.random.default_rng(int(params.get("seed", 0)))
    replace = len(points) < count
    chosen = rng.choice(len(points), count, replace=replace)
    sampled = {"points": points[chosen]}
    colors = cloud.get("colors")
    if colors is not None:
        sampled["colors"] = np.asarray(colors)[
            np.isfinite(np.asarray(cloud["points"])).all(axis=1)
        ][chosen]

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
    points = np.asarray(cloud["points"], dtype=np.float32)
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
        画像は (チャンネル, 高さ, 幅) の float32, 点群は (点の数, 3) の
        float32, 文は番号の並び
    """
    # 種類ごとの形にそろえる
    if kind == "pointcloud" and isinstance(item, dict):
        return np.asarray(item["points"], dtype=np.float32)
    if kind == "image" and np.asarray(item).dtype == np.uint8:
        return as_channels(item) / 255.0
    array = np.asarray(item)

    return array


def run_item_steps(
    load: Callable[[int], Any], train: list[int], splits: dict,
    steps: list, kind: str
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

    Returns
    ----------
    arrays: dict
        分割の名前から (件数, ...) の配列

    Raises
    ----------
    ValueError
        知らない手順, 種類に合わない手順, 形のそろわない中身の場合
    """
    # 手順の確認
    for spec in steps:
        step = ITEM_STEPS.get(spec.get("op"))
        if step is None or step.kind != kind:
            raise ValueError(
                f"{kind} に使えない中身の手順です: {spec.get('op')}"
            )

    # 前から順に, 学習用の見本で当てはめる
    sample = train[:FIT_SAMPLE]
    for number, spec in enumerate(steps):
        step = ITEM_STEPS[spec["op"]]
        if step.fit is None:
            continue
        items = [apply_steps(load(i), steps[:number]) for i in sample]
        params = {k: v for k, v in spec.items() if k not in ("op", "state")}
        spec["state"] = step.fit(items, params)

    # 分割ごとに全ての手順をかけて積む
    arrays = {}
    for name, positions in splits.items():
        values = [
            to_array(apply_steps(load(i), steps), kind) for i in positions
        ]
        shapes = {value.shape for value in values}
        if len(shapes) > 1:
            raise ValueError(
                f"中身の形がそろいません ({len(shapes)} 種類. 例: "
                f"{sorted(shapes)[:3]}). 画像は resize, 点群は "
                "sample_points で形をそろえてください"
            )
        arrays[name] = np.stack(values) if values else np.zeros((0,))

    return arrays


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
        {"width": "幅", "height": "高さ"}, "画像の大きさをそろえる"
    ))
    register_item_step(ItemStep(
        "grayscale", "image", None, grayscale, {}, "灰色 (1 チャンネル) にする"
    ))
    register_item_step(ItemStep(
        "normalize", "image", fit_normalize_image, normalize_image, {},
        "画素を学習用の平均と標準偏差で正規化する ((チャンネル, 高さ, 幅) に)"
    ))

    # 文
    register_item_step(ItemStep(
        "vocabulary", "text", fit_vocabulary, encode_text,
        {"max_tokens": "語彙の上限", "min_count": "語彙に入れる最小の出現数",
         "length": "並びの長さ", "lowercase": "小文字にするか",
         "unit": "word (語) か char (文字)"},
        "学習用の文で語彙を作り, 語の番号の並び (決まった長さ) にする"
    ))

    # 点群
    register_item_step(ItemStep(
        "sample_points", "pointcloud", None, sample_points,
        {"points": "そろえる点の数", "seed": "乱数のシード"},
        "点の数をそろえる (間引きと重ね)"
    ))
    register_item_step(ItemStep(
        "normalize_cloud", "pointcloud", None, normalize_cloud, {},
        "中心を原点にし, 最も遠い点までを 1 にする"
    ))


register_builtin_item_steps()
