import datetime as dt
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ml_eda.capture import (
    depth_scale, inside, is_tracked, pose_of, read_transforms, view_center
)
from ml_eda.itemprep import (
    apply_steps, check_entry_steps, mask_steps, nearest
)
from ml_eda.loaders import (
    ARGUMENT_HINT, load_data, option_flag, points_outside, within
)
from ml_eda.media import (
    IMAGE_EXTENSIONS, POINT_EXTENSIONS, image_header, list_files,
    read_cloud, read_image
)
from ml_eda.pipeline import (
    DATASET_FORMAT_VERSION, ITEMS_FORMAT_VERSION, ITEMS_FORMAT_VERSIONS,
    ITEMS_STEPS_FORMAT_VERSION, SOURCE_NOTE, SPLITS, UNLABELED,
    assign_splits, count_overlap, drop_writing, finish_writing,
    source_name, start_writing, write_frame
)

# 正解の出どころの種類
ORIGIN_TYPES = ("process", "annotation", "measurement", "synthetic", "unknown")

# 1 件の辞書で項目の名前に使えない鍵 (読み込みが入れる)
RESERVED = ("id", "split", "group")

# 項目の置き方 (読み込みが知っているもの)
PLACEMENTS = ("file", "directory", "column")

# 版 2 の書き出しで使える分割のしかた
SPLIT_METHODS = ("random", "group", "none")

# 正解の無い件の扱い (除くか, unlabeled の分割に書くか)
UNLABELED_MODES = ("drop", "keep")

# 1 件のファイルを置くディレクトリ
ITEMS_DIR = "items"

# 領域として読む画像のモード (1 チャンネルの整数)
MASK_MODES = ("1", "L", "P", "I;16", "I;16L", "I;16B", "I")

# 説明に並べる領域の値の数の上限
MAX_VALUES = 256

# 点群の単位からメートルへの倍率
CLOUD_UNITS = {"meter": 1.0, "centimeter": 0.01, "millimeter": 0.001}

# 注意に挙げる例の数
EXAMPLES = 3

# 撮影の 1 枚が指すファイルの欄
FRAME_FILES = (
    "file_path", "raw_depth_file_path", "depth_file_path",
    "confidence_file_path",
)

# 撮影の座標の決まりが無い場合のカメラの軸
DEFAULT_AXES = "OpenGL (+X right, +Y up, -Z forward)"

# 撮影の座標の決まりが無い場合の深度の測り方 (z はカメラの前の軸に沿った
# 距離) と画素の中心 (行 r, 列 c の中心を (c + 0.5, r + 0.5))
DEFAULT_DEPTH = {"depth": "z", "pixel_center": 0.5}

# 深度の測り方 (z はカメラの前の軸に沿った距離, ray は光線に沿った距離)
DEPTH_KINDS = ("z", "ray")

# 撮影の 1 枚が指す, 深度と信頼度のファイルの欄 (resize で最も近い画素で
# 大きさを変えるもの)
DEPTH_FILES = (
    "raw_depth_file_path", "depth_file_path", "confidence_file_path",
)

# 撮影の内部パラメータの欄
INTRINSICS = ("fl_x", "fl_y", "cx", "cy", "w", "h")

# 16 bit の領域の値の上限
MASK_MAX = 65535


def check_origin(origin: Optional[dict]) -> dict:
    """
    正解の出どころを確かめ, 決まった形にする関数

    Parameters
    ----------
    origin: Optional[dict]
        type (ORIGIN_TYPES のどれか), name (作った処理の名前), settings
        (設定の辞書). None なら unknown

    Returns
    ----------
    result: dict
        type, name, settings

    Raises
    ----------
    ValueError
        type が知らない値の場合と, settings が辞書でない場合
    """
    # 渡されなければ不明
    origin = dict(origin or {})
    kind = origin.get("type", "unknown")
    if kind not in ORIGIN_TYPES:
        raise ValueError(
            f"正解の出どころの type は {', '.join(ORIGIN_TYPES)} のどれかに"
            f"してください: {kind}"
        )
    settings = origin.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("正解の出どころの settings は辞書にしてください")
    name = origin.get("name")
    result = {
        "type": kind, "name": None if name is None else str(name),
        "settings": settings,
    }

    return result


def check_classes(classes: Any) -> Optional[list[dict]]:
    """
    領域の値の名前を, 値と名前の組の並びにする関数

    Parameters
    ----------
    classes: Any
        値から名前の辞書 (値は整数か, 整数の文字. 画面の JSON のため) か,
        名前の並び (番号が値). None なら無し

    Returns
    ----------
    named: Optional[list[dict]]
        value と name の辞書の並び (値の順)

    Raises
    ----------
    ValueError
        辞書でも並びでもない場合と, 値が整数でない場合
    """
    # 並びは番号を値にする
    named: Optional[list[dict]] = None
    if classes is None:
        return named
    if isinstance(classes, (list, tuple)):
        classes = dict(enumerate(classes))
    if not isinstance(classes, dict):
        raise ValueError("classes は値から名前の辞書か名前の並びにしてください")
    named = []
    for value, label in classes.items():
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"classes の値は整数にしてください: {value}")
        named.append({"value": int(value), "name": str(label)})
    named.sort(key=value_of)

    return named


def value_of(entry: dict) -> int:
    """
    値と名前の組から値を取り出す関数

    並べ替えの鍵に使う

    Parameters
    ----------
    entry: dict
        value と name の辞書

    Returns
    ----------
    value: int
        値
    """
    # 並べ替えの鍵
    value = entry["value"]

    return value


def unlabeled_mode(split: Optional[dict]) -> bool:
    """
    分割のしかたから, 正解の無い件を unlabeled の分割に書くかを返す関数

    Parameters
    ----------
    split: Optional[dict]
        分割のしかた (unlabeled は drop か keep. 既定は drop)

    Returns
    ----------
    keep: bool
        正解の無い件を書くか

    Raises
    ----------
    ValueError
        unlabeled が drop と keep のどちらでもない場合
    """
    # 書かなければ今までと同じく除く
    mode = (split or {}).get("unlabeled", "drop")
    if mode not in UNLABELED_MODES:
        raise ValueError(
            f"split.unlabeled は drop か keep にしてください: {mode}"
        )
    keep = mode == "keep"

    return keep


def split_items(
    ids: list[str], groups: Optional[dict], split: Optional[dict]
) -> tuple[list[str], dict, list[str]]:
    """
    件を学習用, 検証用, テスト用に割り当てる関数

    Parameters
    ----------
    ids: list[str]
        件の id
    groups: Optional[dict]
        件の id からグループ (同じ人や物, 現場). None なら無し
    split: Optional[dict]
        method (random, group, none), valid, test, seed, unlabeled

    Returns
    ----------
    result: tuple[list[str], dict, list[str]]
        件ごとの分割の名前, 説明に書く分割のしかた, 注意

    Raises
    ----------
    ValueError
        知らない分け方, グループの無い group, groups に無い id, 知らない
        unlabeled の場合
    """
    # 分割のしかたの既定 (グループがあればグループごと)
    conf = {
        "method": "group" if groups else "random", "valid": 0.2,
        "test": 0.0, "seed": 0,
    }
    conf.update(split or {})
    unlabeled_mode(conf)
    method = conf["method"]
    if method == "none":
        conf.update(valid=0.0, test=0.0)
    if method not in SPLIT_METHODS:
        raise ValueError(
            f"版 2 の分割は {', '.join(SPLIT_METHODS)} のどれかにして"
            f"ください: {method}"
        )
    if method == "group" and not groups:
        raise ValueError("group の分割には groups (件の id からグループ) が要ります")
    missing = [key for key in ids if groups and key not in groups]
    if missing:
        raise ValueError(
            f"groups に無い件が {len(missing)} 件あります (例: "
            f"{', '.join(missing[:EXAMPLES])})"
        )

    # 1 件ごとの割り当て (グループは列 group にして版 1 と同じ仕組み)
    frame = pd.DataFrame({"id": ids})
    if groups:
        frame["group"] = [str(groups[key]) for key in ids]
    if method == "group":
        conf["column"] = "group"
    labels = assign_splits(frame, conf, {})
    notes = []
    if groups and method != "group":
        overlap = count_overlap(frame["group"], labels.reset_index(drop=True))
        if overlap["values"]:
            notes.append(
                f"同じグループが {overlap['values']} 個 (計 "
                f"{overlap['rows']} 件) 複数の分割にまたがります (例: "
                f"{', '.join(overlap['examples'])}). 新しい物や現場に"
                "当てはめる力を甘く測るので, 分割を group にしてください"
            )
    for name in ("valid", "test"):
        if float(conf[name]) > 0 and not (labels == name).any():
            notes.append(
                f"{name} の割合を {conf[name]} にしましたが, 0 件になりました"
                " (グループが大きく割合に収まらないか, 件が少ないため). "
                "グループを細かくするか割合を上げてください"
            )
    result = ([str(label) for label in labels], conf, notes)

    return result


def links_followed(follow_links: Any) -> bool:
    """
    引数 follow_links を真偽にする関数

    Parameters
    ----------
    follow_links: Any
        引数の値 (画面からの文字の "false" や "0" も読む)

    Returns
    ----------
    followed: bool
        外を指すリンクの先も読むか

    Raises
    ----------
    ValueError
        知らない文字の場合
    """
    # 読み込みの設定と同じ読み方 (文字の "false" を真にしない)
    followed = option_flag(
        {"follow_links": follow_links}, "follow_links", False
    )

    return followed


def drop_outside(
    files: list[Path], base: Path, label: str, follow_links: bool
) -> tuple[list[Path], list[str]]:
    """
    読み込んだディレクトリの外を指すリンクのファイルを除く関数

    Parameters
    ----------
    files: list[Path]
        ファイル
    base: Path
        読み込んだディレクトリ (それ自体のリンクもたどって比べる)
    label: str
        注意に使う, ファイルの呼び名
    follow_links: bool
        外を指すリンクの先も読むか (True なら除かない)

    Returns
    ----------
    result: tuple[list[Path], list[str]]
        残したファイルと, 除いたものの注意
    """
    # たどった後の場所で比べる. 除いた件は数と例と, 読む案内を注意に
    notes: list[str] = []
    if links_followed(follow_links):
        result = (files, notes)
        return result
    outside = [file for file in files if points_outside(base, file)]
    if outside:
        names = [file.relative_to(base).as_posix() for file in outside]
        notes.append(
            f"{label}のディレクトリの外を指すリンクが {len(outside)} 件あり, "
            f"除きました (例: {', '.join(names[:EXAMPLES])}. 読むには "
            f"{ARGUMENT_HINT} にしてください)"
        )
    result = ([file for file in files if file not in outside], notes)

    return result


def index_by_stem(
    files: list[Path], label: str, suffix: str = ""
) -> dict[str, Path]:
    """
    ファイルを名前の stem で引けるようにする関数

    Parameters
    ----------
    files: list[Path]
        ファイル
    label: str
        誤りの文に使う, ファイルの呼び名
    suffix: str = ""
        stem の終わりから除く文字 (_mask など. 無い stem はそのまま)

    Returns
    ----------
    index: dict[str, Path]
        stem からファイル

    Raises
    ----------
    ValueError
        同じ stem のファイルが 2 つ以上ある場合 (どれを組にするか
        決まらない)
    """
    # 同じ stem は組が決まらないので誤り
    index: dict[str, Path] = {}
    for path in files:
        stem = path.stem
        if suffix and stem.endswith(suffix) and len(stem) > len(suffix):
            stem = stem[:-len(suffix)]
        if stem in index:
            raise ValueError(
                f"{label}に同じ名前 ({stem}) のファイルが 2 つあります: "
                f"{index[stem].name}, {path.name}"
            )
        index[stem] = path

    return index


def pair_files(
    left: dict[str, Path], right: dict[str, Path], labels: tuple[str, str],
    keep: bool = False
) -> tuple[list[tuple[str, Path, Optional[Path]]], list[str]]:
    """
    2 つの stem の索引を組にする関数

    Parameters
    ----------
    left: dict[str, Path]
        入力の stem からファイル
    right: dict[str, Path]
        正解の stem からファイル
    labels: tuple[str, str]
        入力と正解の呼び名
    keep: bool = False
        正解の無い入力を, 正解を None にして組の後ろに足すか

    Returns
    ----------
    result: tuple[list[tuple[str, Path, Optional[Path]]], list[str]]
        stem の順の組 (stem, 入力, 正解. keep なら正解の無い入力が
        stem の順で後ろに続く) と, 片方にしか無いものの注意
    """
    # 両方にある stem の組. 片方だけのものは数と例を注意に
    pairs: list = [(stem, left[stem], right[stem])
                   for stem in sorted(left) if stem in right]
    notes = []
    for own, other, label, partner in (
        (left, right, labels[0], labels[1]),
        (right, left, labels[1], labels[0]),
    ):
        alone = sorted(stem for stem in own if stem not in other)
        if not alone:
            continue
        if keep and own is left:
            pairs += [(stem, left[stem], None) for stem in alone]
            done = f"{UNLABELED} の分割に書きました"
        else:
            done = "除きました"
        notes.append(
            f"{partner}の無い{label}が {len(alone)} 件あり, {done} "
            f"(例: {', '.join(alone[:EXAMPLES])})"
        )
    result = (pairs, notes)

    return result


def read_mask(path: Path) -> np.ndarray:
    """
    領域の画像を, 値をそのまま保った整数の配列として読む関数

    Parameters
    ----------
    path: Path
        領域の画像

    Returns
    ----------
    array: np.ndarray
        (高さ, 幅) の整数の配列 (パレットは番号, 2 値は 0 と 1)

    Raises
    ----------
    ValueError
        1 チャンネルの整数の画像でない場合 (カラーの領域など)
    """
    # パレットの画像は色でなく番号のまま
    from PIL import Image

    with Image.open(path) as image:
        mode = image.mode
        if mode not in MASK_MODES:
            raise ValueError(
                f"{path.name} の領域は 1 チャンネルの整数の画像 (L, P, 1, "
                f"I;16, I) にしてください (今は {mode}. 色とクラスの対応は"
                "決めていません)"
            )
        array = np.array(image)
    if mode == "1":
        array = array.astype(np.uint8)

    return array


def image_channels(mode: Optional[str]) -> Optional[int]:
    """
    画像のモードから, read_image で読んだ配列のチャンネル数を返す関数

    Parameters
    ----------
    mode: Optional[str]
        Pillow の画像のモード

    Returns
    ----------
    channels: Optional[int]
        read_image と同じ決まりのチャンネル数 (灰色と 16 ビットは 1, 透明
        のあるものは 4, 他の色は 3. 分からなければ None)
    """
    # read_image の変換 (パレットと他の色の形式は RGB か RGBA)
    channels: Optional[int] = None
    if mode is None:
        return channels
    if mode in ("1", "L", "F") or mode.startswith("I"):
        channels = 1
    elif mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        channels = 4
    else:
        channels = 3

    return channels


def orientation_of(path: Path) -> int:
    """
    画像の EXIF の向きを返す関数

    Parameters
    ----------
    path: Path
        画像

    Returns
    ----------
    turn: int
        EXIF の向き (1 から 8. 無いか読めなければ 1)
    """
    # 向きの欄 (0x0112) だけ
    from PIL import Image

    turn = 1
    try:
        with Image.open(path) as image:
            turn = int(image.getexif().get(0x0112, 1) or 1)
    except Exception:
        turn = 1

    return turn


def check_image(stem: str, image: Path) -> tuple[int, int, Optional[int]]:
    """
    画像を読めるかと向きを確かめ, 形を返す関数

    Parameters
    ----------
    stem: str
        件の名前
    image: Path
        画像

    Returns
    ----------
    shape: tuple[int, int, Optional[int]]
        (高さ, 幅, チャンネル) (チャンネルは read_image と同じ決まり)

    Raises
    ----------
    ValueError
        画像が読めない場合と, EXIF の向きの付いた画像の場合
    """
    # ヘッダーだけを読む
    header = image_header(image)
    if header["width"] is None:
        raise ValueError(f"{image.name} は画像として読めません")
    turn = orientation_of(image)
    if turn != 1:
        raise ValueError(
            f"{stem} の画像に EXIF の向き ({turn}) があります. 画像は読む"
            "ときに向きを直しますが, 領域は直さないので画素がずれます. "
            "向きを直して保存した画像 (EXIF の向きが 1 か無いもの) と, "
            "それに合わせた領域を渡してください"
        )
    shape = (
        int(header["height"]), int(header["width"]),
        image_channels(header["mode"])
    )

    return shape


def check_pair(stem: str, image: Path, mask: Path) -> dict:
    """
    画像と領域の組の大きさと, 領域の値を確かめる関数

    Parameters
    ----------
    stem: str
        組の名前
    image: Path
        画像
    mask: Path
        領域

    Returns
    ----------
    check: dict
        image ((高さ, 幅, チャンネル)), mask ((高さ, 幅)), values (領域の
        値), dtype (領域の型)

    Raises
    ----------
    ValueError
        画像が読めない場合, EXIF の向きの付いた画像の場合, 画像と領域の
        大きさが違う場合
    """
    # 画像はヘッダーだけ, 領域は値まで読む
    shape = check_image(stem, image)
    size = shape[:2]
    array = read_mask(mask)
    if array.shape != size:
        raise ValueError(
            f"{stem} の画像 ({size[0]} x {size[1]}) と領域 ("
            f"{' x '.join(str(n) for n in array.shape)}) の大きさが違います"
        )
    check = {
        "image": shape, "mask": array.shape,
        "values": np.unique(array), "dtype": array.dtype,
    }

    return check


def common_shape(shapes: list[tuple]) -> list[Optional[int]]:
    """
    件ごとの形を, そろわない次元を None にした 1 つの形にする関数

    Parameters
    ----------
    shapes: list[tuple]
        件ごとの形 (次元の数は同じ)

    Returns
    ----------
    shape: list[Optional[int]]
        次元ごとの大きさ (件ごとに違うか分からなければ None)
    """
    # 次元ごとに全ての件で同じ値なら残す
    shape = [
        int(sizes[0]) if len(set(sizes)) == 1 and sizes[0] is not None
        else None
        for sizes in zip(*shapes)
    ]

    return shape


def item_dir(index: int) -> str:
    """
    1 件のファイルを置くディレクトリの相対の場所を返す関数

    Parameters
    ----------
    index: int
        表の行の位置

    Returns
    ----------
    place: str
        items/000000 の形
    """
    # 元の名前は表の id に残し, 場所は番号 (名前の文字に依らない)
    place = f"{ITEMS_DIR}/{index:06d}"

    return place


def write_ply(path: Path, cloud: dict) -> None:
    """
    点群を 2 進の PLY (リトルエンディアン) で書く関数

    Parameters
    ----------
    path: Path
        書く場所
    cloud: dict
        points ((N, 3)), colors ((N, 3) の 0 から 255 か None), normals
        ((N, 3) か None)

    Returns
    ----------
    None
    """
    # 座標と法線は float32, 色は uchar
    points = np.asarray(cloud["points"], dtype=np.float32)
    fields = [(axis, "<f4") for axis in ("x", "y", "z")]
    columns = [points[:, 0], points[:, 1], points[:, 2]]
    if cloud.get("normals") is not None:
        normals = np.asarray(cloud["normals"], dtype=np.float32)
        fields += [(axis, "<f4") for axis in ("nx", "ny", "nz")]
        columns += [normals[:, 0], normals[:, 1], normals[:, 2]]
    if cloud.get("colors") is not None:
        colors = np.clip(np.asarray(cloud["colors"]), 0, 255).astype(np.uint8)
        fields += [(axis, "u1") for axis in ("red", "green", "blue")]
        columns += [colors[:, 0], colors[:, 1], colors[:, 2]]
    table = np.empty(len(points), dtype=np.dtype(fields))
    for (field_name, _), column in zip(fields, columns):
        table[field_name] = column

    # ヘッダーと中身
    kinds = {"<f4": "float", "u1": "uchar"}
    lines = [
        "ply", "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        *(f"property {kinds[kind]} {field_name}"
          for field_name, kind in fields),
        "end_header",
    ]
    path.write_bytes(
        ("\n".join(lines) + "\n").encode("ascii") + table.tobytes()
    )


def check_entry_names(entries: dict) -> None:
    """
    項目の名前が, 1 件の辞書の決まった鍵 (id, split, group) と重ならない
    ことを確かめる関数

    Parameters
    ----------
    entries: dict
        項目の名前から説明

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        決まった鍵と同じ名前の項目がある場合
    """
    # 重なると 1 件の辞書で項目の中身が鍵を上書きするため
    taken = [name for name in entries if name in RESERVED]
    if taken:
        raise ValueError(
            f"項目の名前に {', '.join(taken)} は使えません ("
            f"{', '.join(RESERVED)} は 1 件の辞書の決まった鍵)"
        )


def write_items(
    target: Path, final: Path, rows: list[dict], info: dict,
    parent: Optional[int] = None
) -> dict:
    """
    1 行 1 件の表と分割の件数と説明を書き, 書き出しを終える関数

    Parameters
    ----------
    target: Path
        一時ディレクトリ
    final: Path
        書き出す場所
    rows: list[dict]
        件ごとの id, split, (group), 項目の場所か値
    info: dict
        dataset.yaml の中身 (items と splits はここで入れる)
    parent: Optional[int] = None
        start_writing に渡した親の記述子

    Returns
    ----------
    info: dict
        書き出した dataset.yaml の中身
    """
    # 項目の名前は 1 件の辞書の決まった鍵と重ならないもの
    check_entry_names(info["entries"])

    # 表 (parquet が無ければ csv) と, 件のある分割の件数
    frame = pd.DataFrame(rows)
    info["items"] = write_frame(frame, target / "items")
    counts = frame["split"].value_counts()
    info["splits"] = {
        name: {"items": int(counts[name])}
        for name in (*SPLITS, UNLABELED) if counts.get(name, 0)
    }
    finish_writing(target, final, info, parent)

    return info


def entry_steps(item_steps: Any, kinds: dict) -> dict[str, list]:
    """
    書き出しに渡した中身の手順を, 項目ごとに確かめる関数

    Parameters
    ----------
    item_steps: Any
        項目の名前から手順の並びの辞書 (None なら手順無し)
    kinds: dict
        手順を渡せる項目の名前から項目の種類

    Returns
    ----------
    steps: dict[str, list]
        手順のある項目の名前から, そろえた手順の並び

    Raises
    ----------
    ValueError
        辞書でない場合, 手順を渡せない項目がある場合, 手順の誤り
    """
    # 渡せる項目だけ (空の並びは手順無し)
    steps: dict[str, list] = {}
    if item_steps is None:
        return steps
    if not isinstance(item_steps, dict):
        raise ValueError(
            "item_steps は項目の名前から手順の並びの辞書にしてください"
        )
    for name, listed in item_steps.items():
        if name not in kinds:
            raise ValueError(
                f"項目 {name} には中身の手順を渡せません (渡せるもの: "
                f"{', '.join(kinds)})"
            )
        checked = check_entry_steps(listed, kinds[name])
        if checked:
            steps[name] = checked

    return steps


def write_image(path: Path, array: np.ndarray) -> None:
    """
    手順をかけた画像を PNG で書く関数

    Parameters
    ----------
    path: Path
        書く場所
    array: np.ndarray
        (高さ, 幅) か (高さ, 幅, チャンネル) の uint8 の配列

    Returns
    ----------
    None
    """
    # 1 チャンネルの軸を落とした灰色
    from PIL import Image

    array = np.asarray(array, dtype=np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    Image.fromarray(array).save(path, format="PNG")


def write_mask(path: Path, mask: np.ndarray) -> np.ndarray:
    """
    手順をかけた領域を, 値をそのまま保って PNG で書く関数

    Parameters
    ----------
    path: Path
        書く場所
    mask: np.ndarray
        (高さ, 幅) の整数の配列

    Returns
    ----------
    written: np.ndarray
        書いた配列 (uint8 は uint8, 他は uint16. 読み直すとこの型)

    Raises
    ----------
    ValueError
        値が 0 から 65535 に収まらない場合
    """
    # 8 bit はそのまま, 他は 16 bit の灰色 (値を変えない)
    from PIL import Image

    mask = np.asarray(mask)
    if mask.size and (mask.min() < 0 or mask.max() > MASK_MAX):
        raise ValueError(
            f"{path.parent.name} の領域の値が 0 から {MASK_MAX} に収まらず, "
            "手順をかけた領域を PNG で書けません"
        )
    written = mask.astype(np.uint8 if mask.dtype == np.uint8 else np.uint16)
    Image.fromarray(written).save(path, format="PNG")

    return written


def export_image_masks(
    images: Union[str, Path], masks: Union[str, Path],
    path: Union[str, Path], split: Optional[dict] = None,
    groups: Optional[dict] = None, classes: Any = None,
    origin: Optional[dict] = None, name: Optional[str] = None,
    overwrite: bool = False, mask_suffix: str = "",
    item_steps: Optional[dict] = None, follow_links: bool = False,
    parent: Optional[int] = None
) -> dict:
    """
    画像と領域の対を, 版 2 のデータセットに書き出す関数

    Parameters
    ----------
    images: Union[str, Path]
        画像のディレクトリ
    masks: Union[str, Path]
        領域のディレクトリ (画像とファイルの名前の stem で組にする)
    path: Union[str, Path]
        書き出すディレクトリ
    split: Optional[dict] = None
        分割のしかた (method, valid, test, seed. 既定は random で検証用
        2 割. groups があれば group). unlabeled: keep で, 領域の無い画像を
        unlabeled の分割に書く (既定 drop は除く)
    groups: Optional[dict] = None
        件の id (stem) からグループ
    classes: Any = None
        領域の値の名前 (値から名前の辞書か, 名前の並び)
    origin: Optional[dict] = None
        正解の出どころ (type, name, settings)
    name: Optional[str] = None
        データセットの名前 (省略するとディレクトリの名前)
    overwrite: bool = False
        既にあるデータセットを置き換えるか
    mask_suffix: str = ""
        領域のファイルの stem の終わりから除いて組にする文字 (a.png と
        a_mask.png を組にするなら _mask)
    item_steps: Optional[dict] = None
        項目 image にかける中身の手順 ({"image": [{"op": "resize", ...}]}.
        resize, grayscale, rgb). 領域には画像の resize と同じ大きさの
        resize_mask をかける. 渡すと版 3 で書く
    follow_links: bool = False
        画像と領域のディレクトリの外を指すリンクの先も写すか (False では
        除いて notes に書く)
    parent: Optional[int] = None
        使う側が開いた, 書き出す場所の親のディレクトリの記述子 (渡すと
        path の最後の部分を親の中の名前とし, 途中の場所をたどらずに書く.
        閉じるのは使う側)

    Returns
    ----------
    info: dict
        書き出した dataset.yaml の中身

    Raises
    ----------
    ValueError
        組が無い場合, 同じ stem が 2 つある場合, 大きさの違う組やカラーの
        領域, EXIF の向きの付いた画像がある場合, 分割と中身の手順の誤りの
        場合
    """
    # 中身の手順 (領域の手順は画像の手順から決める)
    if isinstance(item_steps, dict) and "mask" in item_steps:
        raise ValueError(
            "領域の手順は渡せません (画像の resize と同じ大きさの "
            "resize_mask を書き出しがかけます)"
        )
    steps = entry_steps(item_steps, {"image": "image"})
    if steps:
        steps["mask"] = mask_steps(steps["image"])
        steps = {key: value for key, value in steps.items() if value}

    # 組 (stem で) と, 正解の出どころ, 値の名前, 分割
    image_dir = Path(images).expanduser()
    mask_dir = Path(masks).expanduser()
    for place in (image_dir, mask_dir):
        if not place.is_dir():
            raise ValueError(f"{place} はディレクトリではありません")
    image_files, image_notes = drop_outside(
        list_files(image_dir, IMAGE_EXTENSIONS), image_dir, "画像",
        follow_links
    )
    mask_files, mask_notes = drop_outside(
        list_files(mask_dir, IMAGE_EXTENSIONS), mask_dir, "領域",
        follow_links
    )
    pairs, notes = pair_files(
        index_by_stem(image_files, "画像"),
        index_by_stem(mask_files, "領域", mask_suffix),
        ("画像", "領域"), unlabeled_mode(split)
    )
    notes = image_notes + mask_notes + notes
    alone = [(stem, image) for stem, image, mask in pairs if mask is None]
    pairs = [pair for pair in pairs if pair[2] is not None]
    if not pairs:
        raise ValueError(
            "画像と領域の組がありません (ファイルの名前の stem で組にします)"
        )
    origin_info = check_origin(origin)
    named = check_classes(classes)
    ids = [stem for stem, _, _ in pairs]
    labels, split_info, split_notes = split_items(ids, groups, split)
    notes += split_notes

    # 書く前に全ての組の大きさと領域の値, 正解の無い画像を確かめる
    checks = [check_pair(stem, image, mask) for stem, image, mask in pairs]
    lone_shapes = [check_image(stem, image) for stem, image in alone]
    values = np.unique(np.concatenate([check["values"] for check in checks]))
    listed: Optional[list[int]] = [int(value) for value in values]
    if len(values) > MAX_VALUES:
        listed = None
        notes.append(
            f"領域の値が {len(values)} 種類あり, 説明の values を省きました"
        )
    if named is not None:
        unnamed = sorted(
            set(int(v) for v in values) - {entry["value"] for entry in named}
        )
        if unnamed:
            notes.append(
                f"classes に名前の無い領域の値があります: "
                f"{', '.join(str(v) for v in unnamed[:EXAMPLES])}"
            )
    if any(mask.suffix.lower() in (".jpg", ".jpeg") for _, _, mask in pairs):
        notes.append(
            "JPEG の領域があります. 圧縮で境目の値が崩れるので, PNG を"
            "勧めます"
        )

    # 1 件ずつの写しか, 手順をかけた書き込み (失敗したら一時ディレクトリを削除)
    final, target = start_writing(path, overwrite, parent)
    try:
        rows, written = [], []
        for index, (stem, image, mask) in enumerate(pairs):
            place = item_dir(index)
            (target / place).mkdir(parents=True)
            row = {"id": stem, "split": labels[index]}
            if groups:
                row["group"] = str(groups[stem])
            if steps:
                row.update(write_pair(target, place, image, mask, steps))
                written.append(read_pair(target, row))
            else:
                for key, source in (("image", image), ("mask", mask)):
                    row[key] = f"{place}/{key}{source.suffix.lower()}"
                    shutil.copyfile(source, target / row[key])
            rows.append(row)

        # 正解の無い画像は, 正解の付いた件の後ろに unlabeled で書く
        for index, (stem, image) in enumerate(alone, len(pairs)):
            place = item_dir(index)
            (target / place).mkdir(parents=True)
            row = {"id": stem, "split": UNLABELED}
            if groups:
                row["group"] = lone_group(groups, stem)
            if steps:
                row["image"] = f"{place}/image.png"
                write_image(
                    target / row["image"],
                    apply_steps(read_image(image), steps["image"])
                )
                lone_shapes[index - len(pairs)] = with_channels(
                    read_image(target / row["image"])
                ).shape
            else:
                row["image"] = f"{place}/image{image.suffix.lower()}"
                shutil.copyfile(image, target / row["image"])
            row["mask"] = None
            rows.append(row)

        # 手順をかけた場合は, 書いたファイルの形と値
        if steps:
            checks = written
            values = np.unique(
                np.concatenate([check["values"] for check in checks])
            )
            listed = [int(value) for value in values] if (
                len(values) <= MAX_VALUES
            ) else None
        image_formats = ["png"] if steps else sorted({
            image.suffix.lower().lstrip(".")
            for image in [pair[1] for pair in pairs] + [i for _, i in alone]
        })
        mask_formats = ["png"] if steps else sorted({
            mask.suffix.lower().lstrip(".") for _, _, mask in pairs
        })
        info = {
            "format_version": (
                ITEMS_STEPS_FORMAT_VERSION if steps else ITEMS_FORMAT_VERSION
            ),
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "predict",
            "target": {
                "kind": "mask", "entries": ["mask"], "origin": origin_info,
            },
            "entries": {
                "image": {
                    "kind": "image", "role": "input", "placement": "file",
                    "pattern": f"{ITEMS_DIR}/{{index:06d}}/image{{suffix}}",
                    "formats": image_formats,
                    "shape": common_shape(
                        [c["image"] for c in checks] + lone_shapes
                    ),
                    "dtype": "uint8", "units": "pixel",
                    "convention": {
                        "origin": "top_left", "axes": ["row", "column"],
                    },
                },
                "mask": {
                    "kind": "mask", "role": "target", "placement": "file",
                    "pattern": f"{ITEMS_DIR}/{{index:06d}}/mask{{suffix}}",
                    "formats": mask_formats,
                    "shape": common_shape([c["mask"] for c in checks]),
                    "dtype": np.result_type(
                        *[c["dtype"] for c in checks]
                    ).name,
                    "units": "pixel",
                    "convention": {
                        "origin": "top_left", "axes": ["row", "column"],
                    },
                    "values": listed,
                    "classes": named,
                },
            },
            "items": None,
            "split": split_info,
            "splits": None,
            "source": {
                "images": source_name(image_dir),
                "masks": source_name(mask_dir), "mask_suffix": mask_suffix,
                "paths": "name_only", "note": SOURCE_NOTE,
            },
            "notes": notes,
        }
        for key, listed_steps in steps.items():
            info["entries"][key]["steps"] = listed_steps
        info = write_items(target, final, rows, info, parent)
    except Exception:
        drop_writing(target, parent)
        raise

    return info


def lone_group(groups: dict, stem: str) -> Optional[str]:
    """
    正解の無い件のグループを返す関数

    Parameters
    ----------
    groups: dict
        件の id からグループ
    stem: str
        件の id

    Returns
    ----------
    group: Optional[str]
        グループ (groups に無ければ None. 正解の無い件は分割に使わない
        ので, groups に無くてもよい)
    """
    # 正解の付いた件と同じく文字にそろえる
    group = str(groups[stem]) if stem in groups else None

    return group


def write_pair(
    target: Path, place: str, image: Path, mask: Path, steps: dict
) -> dict:
    """
    画像と領域の組に中身の手順をかけて PNG で書く関数

    Parameters
    ----------
    target: Path
        一時ディレクトリ
    place: str
        1 件のディレクトリの相対の場所
    image: Path
        元の画像
    mask: Path
        元の領域
    steps: dict
        項目の名前 (image, mask) から手順の並び

    Returns
    ----------
    columns: dict
        表の image と mask の列 (相対の場所)
    """
    # 画像は read_image, 領域は read_mask と同じ読み方で読み, 手順の適用
    columns = {"image": f"{place}/image.png", "mask": f"{place}/mask.png"}
    write_image(
        target / columns["image"],
        apply_steps(read_image(image), steps.get("image", []))
    )
    write_mask(
        target / columns["mask"],
        apply_steps(read_mask(mask), steps.get("mask", []))
    )

    return columns


def read_pair(target: Path, row: dict) -> dict:
    """
    書いた画像と領域の組の形と値を, 読み直して求める関数

    Parameters
    ----------
    target: Path
        一時ディレクトリ
    row: dict
        表の行 (image と mask の相対の場所)

    Returns
    ----------
    check: dict
        image ((高さ, 幅, チャンネル)), mask ((高さ, 幅)), values (領域の
        値), dtype (領域の型). check_pair と同じ形
    """
    # load_items と同じ読み方での読み直し
    image = with_channels(read_image(target / row["image"]))
    mask = read_mask(target / row["mask"])
    check = {
        "image": image.shape, "mask": mask.shape,
        "values": np.unique(mask), "dtype": mask.dtype,
    }

    return check


def kept_frames(
    source: Path, transforms: dict, base: Optional[Path] = None
) -> tuple[list[dict], int]:
    """
    撮影の枚のうち, 画像のある枚を選ぶ関数

    Parameters
    ----------
    source: Path
        撮影のディレクトリ
    transforms: dict
        transforms.json の中身
    base: Optional[Path] = None
        外かを比べるディレクトリ (リンクをたどった後にこの外を指す画像の
        枚は除く. None ならリンクの先をたどる)

    Returns
    ----------
    result: tuple[list[dict], int]
        画像のある枚 (transforms.json の frames の写し) と, 除いた枚数

    Raises
    ----------
    ValueError
        画像のある枚が無い場合
    """
    # 画像が撮影の中にある枚だけ
    frames = [dict(entry) for entry in transforms["frames"]
              if inside(source, entry.get("file_path"), base) is not None]
    if not frames:
        raise ValueError(f"{source.name} の撮影に画像のある枚がありません")
    result = (frames, len(transforms["frames"]) - len(frames))

    return result


def frames_shift(frames: list[dict], label: str) -> np.ndarray:
    """
    枚の姿勢から, 撮った物の中心を原点にする平行移動を求める関数

    Parameters
    ----------
    frames: list[dict]
        画像のある枚 (transforms.json の frames)
    label: str
        誤りの文に使う撮影の名前

    Returns
    ----------
    shift: np.ndarray
        世界の座標に足す量 (中心の符号を変えたもの. メートル)

    Raises
    ----------
    ValueError
        姿勢が 1 つも無い場合
    """
    # 追跡できた枚の姿勢を優先し, 無ければ全ての姿勢
    tracked = [pose_of(entry) for entry in frames if is_tracked(entry)]
    poses = [pose for pose in tracked if pose is not None] or [
        pose for pose in map(pose_of, frames) if pose is not None
    ]
    if not poses:
        raise ValueError(f"{label} の撮影に姿勢がありません")

    # 方位の隙間と同じ中心. 決まらなければカメラの位置の平均
    center = view_center(poses) if len(poses) >= 2 else None
    if center is None:
        center = np.mean([pose[:3, 3] for pose in poses], axis=0)
    shift = -center

    return shift


def capture_shift(
    capture: Union[str, Path], follow_links: bool = False,
    base: Optional[Path] = None
) -> np.ndarray:
    """
    撮影の正規化の平行移動 (書き出しと同じ決まり) を求める関数

    推論のときに, 新しい撮影に学習のときと同じ正規化をかけ, 出力を元の
    座標に戻す (出力 - shift) ため

    Parameters
    ----------
    capture: Union[str, Path]
        撮影のディレクトリ (transforms.json を持つ)
    follow_links: bool = False
        リンクをたどった後に外を指すファイルも読むか (False では無い
        ものとし, transforms.json が外なら誤り)
    base: Optional[Path] = None
        外かを比べるディレクトリ (既定は撮影のディレクトリ.
        export_capture_clouds は渡した撮影の場所を渡す)

    Returns
    ----------
    shift: np.ndarray
        世界の座標に足す量 (正規化した座標 = 元の座標 + shift. メートル)

    Raises
    ----------
    ValueError
        画像のある枚が無い場合, その枚に姿勢が無い場合と, transforms.json
        が外を指すリンクの場合
    """
    # 画像のある枚の姿勢だけから (書き出しで残す枚と同じ)
    source = Path(capture).expanduser()
    limit = link_limit(source, follow_links, base)
    frames, _ = kept_frames(
        source, read_transforms(source, limit, ARGUMENT_HINT), limit
    )
    shift = frames_shift(frames, source.name)

    return shift


def link_limit(
    source: Path, follow_links: bool, base: Optional[Path]
) -> Optional[Path]:
    """
    撮影の外を指すリンクを比べるディレクトリを返す関数

    Parameters
    ----------
    source: Path
        撮影のディレクトリ
    follow_links: bool
        外を指すリンクの先も読むか
    base: Optional[Path]
        外かを比べるディレクトリ (None なら撮影のディレクトリ)

    Returns
    ----------
    limit: Optional[Path]
        比べるディレクトリ (たどるなら None)
    """
    # read_capture と同じ決まり
    limit = None if links_followed(follow_links) else (
        source if base is None else base
    )

    return limit


def normalize_capture(
    capture: Union[str, Path], target: Union[str, Path],
    steps: Optional[list] = None, follow_links: bool = False,
    base: Optional[Path] = None
) -> dict:
    """
    撮影を写し, 姿勢と撮影が持つ点群に正規化の平行移動をかける関数

    書き出し (export_capture_clouds) と同じ正規化. 推論のときに新しい
    撮影にも同じものをかけられるように公開する

    Parameters
    ----------
    capture: Union[str, Path]
        元の撮影のディレクトリ
    target: Union[str, Path]
        写す先のディレクトリ (無いもの)
    steps: Optional[list] = None
        各枚の画像にかける中身の手順 (版 3 のデータセットの撮影の項目の
        steps. resize, grayscale, rgb). resize では深度と信頼度と内部
        パラメータも同じ倍率に直す
    follow_links: bool = False
        リンクをたどった後に外を指すファイルも読むか (False では無い
        ものとし, transforms.json が外なら誤り)
    base: Optional[Path] = None
        外かを比べるディレクトリ (既定は撮影のディレクトリ.
        export_capture_clouds は渡した撮影の場所を渡す)

    Returns
    ----------
    summary: dict
        shift (世界の座標に足した量), frames (枚数), height, width,
        convention (座標の決まり), notes (除いた枚と点群の注意)

    Raises
    ----------
    ValueError
        画像のある枚が無い場合, その枚に姿勢が無い場合, 中身の手順の
        誤り, 手順をかける撮影の画像の大きさがそろわない場合と,
        transforms.json が外を指すリンクの場合
    """
    # 画像のある枚だけを残し, その姿勢から平行移動を決める
    checked = check_entry_steps(steps or [], "capture")
    source = Path(capture).expanduser()
    target = Path(target).expanduser()
    limit = link_limit(source, follow_links, base)
    transforms = read_transforms(source, limit, ARGUMENT_HINT)
    frames, dropped = kept_frames(source, transforms, limit)
    shift = frames_shift(frames, source.name)
    notes = []
    if dropped:
        notes.append(
            f"{source.name} の撮影の, 画像の無い (か外を指すリンクの) "
            f"{dropped} 枚を除きました"
        )

    # 各ファイルを同じ相対の場所に写す. 撮影の外を指す場所 (リンクを
    # たどった後に外のものも) と無いファイルの欄は除く
    target.mkdir(parents=True)
    for entry in frames:
        for key in FRAME_FILES:
            place = inside(source, entry.get(key), limit)
            if place is None:
                entry.pop(key, None)
                continue
            relative = Path(os.path.normpath(entry[key])).as_posix()
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(place, target / relative)
            entry[key] = relative
        pose = pose_of(entry)
        if pose is not None:
            pose[:3, 3] += shift
            entry["transform_matrix"] = pose.tolist()

    # 撮影が持つ点群にも同じ平行移動 (無いか読めなければ除いて注意)
    copied = {key: value for key, value in transforms.items()
              if key not in ("frames", "point_cloud")}
    copied["frames"] = frames
    notes += copy_capture_cloud(
        source, target, transforms.get("point_cloud"), shift, copied, limit
    )

    # 中身の手順 (各枚の画像と, 大きさに合わせて深度と内部パラメータ)
    if checked:
        resize_frames(target, copied, checked)
    (target / "transforms.json").write_text(
        json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "shift": shift, "frames": len(frames),
        "height": copied.get("h"), "width": copied.get("w"),
        "convention": transforms.get("coordinate_convention") or {},
        "notes": notes,
    }

    return summary


def resize_frames(target: Path, transforms: dict, steps: list) -> None:
    """
    写した撮影の各枚の画像に中身の手順をかけ, 大きさが変われば深度,
    信頼度, 内部パラメータを同じ倍率に直す関数

    Parameters
    ----------
    target: Path
        写した撮影のディレクトリ (ファイルを書き換える)
    transforms: dict
        写した transforms.json の中身 (場所と内部パラメータを書き換える)
    steps: list
        画像の手順 (check_entry_steps でそろえたもの)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        画像の大きさがそろわない場合と, PNG に直した名前が他の画像,
        深度, 信頼度と重なる場合
    """
    # 各枚の画像に手順をかけた PNG (同じファイルは 1 度だけ). 重なりは
    # 画像と深度と信頼度の全ての場所で見る (深度を上書きしないため)
    sizes: Optional[tuple] = None
    done: dict[str, str] = {}
    resized: set[str] = set()
    taken = taken_files(transforms)
    for entry in transforms["frames"]:
        relative = entry["file_path"]
        if relative in done:
            # 同じ画像を指す枚も, 深度と内部パラメータは枚ごとに直す
            entry["file_path"] = done[relative]
            if sizes is not None and sizes[0] != sizes[1]:
                scale_depths(target, entry, sizes, resized)
                scale_intrinsics(entry, transforms, sizes)
            continue
        original = read_image(target / relative)
        changed = apply_steps(original, steps)
        pair = (original.shape[:2], np.asarray(changed).shape[:2])
        if sizes is not None and pair != sizes:
            raise ValueError(
                f"{target.name} の撮影の画像の大きさがそろわないので, 中身の"
                "手順をかけられません"
            )
        sizes = pair
        png = Path(relative).with_suffix(".png").as_posix()
        if png != relative and (png in taken or png in done.values()):
            raise ValueError(
                f"{target.name} の撮影の {relative} を PNG に直した名前が, "
                "他の画像か深度か信頼度と重なります"
            )
        (target / relative).unlink()
        write_image(target / png, changed)
        done[relative] = png
        entry["file_path"] = png

        # 大きさが変われば, 深度と信頼度も同じ倍率に (最も近い画素)
        if pair[0] != pair[1]:
            scale_depths(target, entry, pair, resized)
            scale_intrinsics(entry, transforms, pair)

    # 撮影全体の内部パラメータ
    if sizes is not None and sizes[0] != sizes[1]:
        scale_intrinsics(transforms, transforms, sizes)


def taken_files(transforms: dict) -> set[str]:
    """
    撮影の全ての枚の画像, 深度, 信頼度の場所を集める関数

    Parameters
    ----------
    transforms: dict
        transforms.json の中身

    Returns
    ----------
    taken: set[str]
        撮影の中の場所 (transforms.json に書いた形)
    """
    # 画像と DEPTH_FILES の鍵の場所
    taken = set()
    for entry in transforms["frames"]:
        for key in ("file_path", *DEPTH_FILES):
            place = entry.get(key)
            if isinstance(place, str) and place:
                taken.add(place)

    return taken


def scale_depths(
    target: Path, entry: dict, sizes: tuple, resized: set[str]
) -> None:
    """
    1 枚の深度と信頼度の画像を, 画像と同じ倍率に変える関数 (同じファイル
    は 1 度だけ)

    Parameters
    ----------
    target: Path
        写した撮影のディレクトリ
    entry: dict
        transforms.json の 1 枚
    sizes: tuple
        画像の元の (高さ, 幅) と変えた後の (高さ, 幅)
    resized: set[str]
        変えた場所 (書き足す)

    Returns
    ----------
    None
    """
    # 最も近い画素で, 値を混ぜずに
    for key in DEPTH_FILES:
        place = entry.get(key)
        if place and place not in resized:
            scale_nearest(target / place, sizes)
            resized.add(place)


def scale_nearest(path: Path, sizes: tuple) -> None:
    """
    深度か信頼度の画像の大きさを, 画像と同じ倍率で最も近い画素に変える
    関数 (値を混ぜない)

    Parameters
    ----------
    path: Path
        深度か信頼度の画像 (書き換える)
    sizes: tuple
        画像の元の (高さ, 幅) と変えた後の (高さ, 幅)

    Returns
    ----------
    None
    """
    # 画像との大きさの比の維持 (深度の画像の方が小さい撮影もある)
    from PIL import Image

    (height, width), (new_height, new_width) = sizes
    with Image.open(path) as opened:
        image_format = opened.format
        values = np.array(opened)
    size = (
        max(1, round(values.shape[1] * new_width / width)),
        max(1, round(values.shape[0] * new_height / height)),
    )
    Image.fromarray(nearest(values, *size)).save(path, format=image_format)


def scale_intrinsics(holder: dict, transforms: dict, sizes: tuple) -> None:
    """
    内部パラメータを, 画像の大きさの倍率に合わせて直す関数

    Parameters
    ----------
    holder: dict
        内部パラメータを持つ辞書 (撮影全体か 1 枚. 書き換える)
    transforms: dict
        撮影全体の transforms.json の中身 (画素の中心の決まり)
    sizes: tuple
        画像の元の (高さ, 幅) と変えた後の (高さ, 幅)

    Returns
    ----------
    None
    """
    # 画素の中心の維持: 端からの座標 (u - 中心 + 0.5) に倍率
    (height, width), (new_height, new_width) = sizes
    center = float(with_depth_convention(
        transforms.get("coordinate_convention") or {}
    )["pixel_center"])
    ratios = {"x": new_width / width, "y": new_height / height}
    for key in INTRINSICS:
        value = holder.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key == "w":
            holder[key] = int(new_width)
        elif key == "h":
            holder[key] = int(new_height)
        elif key.startswith("fl_"):
            holder[key] = float(value) * ratios[key[-1]]
        else:
            holder[key] = (
                (float(value) - center + 0.5) * ratios[key[-1]] + center - 0.5
            )


def copy_capture_cloud(
    source: Path, target: Path, described: Any, shift: np.ndarray,
    copied: dict, base: Optional[Path] = None
) -> list[str]:
    """
    撮影が持つ点群に平行移動をかけて写す関数

    Parameters
    ----------
    source: Path
        元の撮影のディレクトリ
    target: Path
        写す先のディレクトリ
    described: Any
        transforms.json の point_cloud
    shift: np.ndarray
        世界の座標に足す量
    copied: dict
        写す transforms.json (写せたら point_cloud を入れる)
    base: Optional[Path] = None
        外かを比べるディレクトリ (None ならリンクの先をたどる)

    Returns
    ----------
    notes: list[str]
        除いた点群の注意
    """
    # 書かれていなければ何もしない
    notes: list[str] = []
    if not isinstance(described, dict) or not described.get("file_path"):
        return notes
    cloud = inside(source, described.get("file_path"), base)
    if cloud is None:
        notes.append(
            f"{source.name} の撮影の点群 ({described.get('file_path')}) が"
            "無いか撮影の外を指す (リンクの先が外のものも) ので, 除きました"
        )
        return notes

    # 読めた点群だけ PLY で写す
    try:
        points = read_cloud(cloud)
        points["points"] = points["points"] + shift
        write_ply(target / "point_cloud.ply", points)
        copied["point_cloud"] = {**described, "file_path": "point_cloud.ply"}
    except (OSError, ValueError, KeyError, IndexError) as error:
        notes.append(
            f"{source.name} の撮影の点群を読めず, 除きました ({error})"
        )

    return notes


def find_clouds(
    path: Path, follow_links: bool = False
) -> tuple[dict[str, Path], list[str]]:
    """
    点群のファイルを stem で引けるようにする関数

    点群のディレクトリの直下のファイルだけを見る

    Parameters
    ----------
    path: Path
        点群のディレクトリ
    follow_links: bool = False
        点群のディレクトリの外を指すリンクの先も読むか (False では除く)

    Returns
    ----------
    result: tuple[dict[str, Path], list[str]]
        stem から点群のファイルと, 外を指すリンクを除いた注意

    Raises
    ----------
    ValueError
        ディレクトリでない場合と, 同じ stem が 2 つある場合
    """
    # 直下の, 隠しでない点群のファイル
    if not path.is_dir():
        raise ValueError(f"{path} はディレクトリではありません")
    files = sorted(
        child for child in path.iterdir()
        if child.is_file() and not child.name.startswith(".")
        and child.suffix.lower() in POINT_EXTENSIONS
    )
    files, notes = drop_outside(files, path, "点群", follow_links)
    result = (index_by_stem(files, "点群"), notes)

    return result


def capture_pairs(
    captures: Union[str, Path], clouds: Union[str, Path], keep: bool = False,
    follow_links: bool = False
) -> tuple[Any, list[tuple[str, int, Optional[Path]]], list[str]]:
    """
    撮影と点群を, 撮影のディレクトリの名前と点群の stem で組にする関数

    Parameters
    ----------
    captures: Union[str, Path]
        撮影のディレクトリか, それを並べたディレクトリ
    clouds: Union[str, Path]
        点群のディレクトリ
    keep: bool = False
        点群の無い撮影を, 点群を None にして組の後ろに足すか
    follow_links: bool = False
        撮影と点群のディレクトリの外を指すリンクの先も読むか (False では
        外を指す撮影は読めない撮影, 点群は除いて注意に)

    Returns
    ----------
    result: tuple[Any, list[tuple[str, int, Optional[Path]]], list[str]]
        読み込んだ撮影のデータ, 組 (名前, 撮影の行の位置, 点群), 注意

    Raises
    ----------
    ValueError
        撮影が無い場合と, 同じ stem の点群が 2 つある場合
    """
    # 読めない撮影 (外を指すリンクも) は除いて注意に
    data = load_data(
        Path(captures).expanduser(), loader="capture",
        follow_links=follow_links
    )
    named: dict[str, Path] = {}
    positions: dict[str, int] = {}
    unreadable = []
    for position, (place, error) in enumerate(
        zip(data.frame["path"], data.frame["error"])
    ):
        stem = Path(str(place)).name
        if isinstance(error, str):
            unreadable.append(stem)
            continue
        named[stem] = Path(str(place))
        positions[stem] = position
    cloud_files, cloud_notes = find_clouds(
        Path(clouds).expanduser(), follow_links
    )
    pairs, notes = pair_files(named, cloud_files, ("撮影", "点群"), keep)
    notes = cloud_notes + notes
    if data.source.get("capped"):
        notes.insert(0, "撮影が上限を超えたので, 先頭だけを読みました")
    if unreadable:
        notes.insert(0, (
            f"読めない撮影が {len(unreadable)} 件あり, 除きました (例: "
            f"{', '.join(unreadable[:EXAMPLES])})"
        ))
    result = (
        data, [(stem, positions[stem], cloud) for stem, _, cloud in pairs],
        notes
    )

    return result


def export_capture_clouds(
    captures: Union[str, Path], clouds: Union[str, Path],
    path: Union[str, Path], split: Optional[dict] = None,
    groups: Optional[dict] = None, origin: Optional[dict] = None,
    cloud_units: str = "meter", name: Optional[str] = None,
    overwrite: bool = False, item_steps: Optional[dict] = None,
    follow_links: bool = False, parent: Optional[int] = None
) -> dict:
    """
    多視点の撮影と正解の点群の対を, 版 2 のデータセットに書き出す関数

    1 件ごとに, 姿勢だけから求めた物の中心を原点にする平行移動を, 撮影の
    姿勢と点群の両方にかける (回転と拡大縮小はかけず, 実寸を残す)

    Parameters
    ----------
    captures: Union[str, Path]
        撮影のディレクトリか, それを並べたディレクトリ
    clouds: Union[str, Path]
        点群のディレクトリ (直下の <撮影のディレクトリの名前>.<拡張子>)
    path: Union[str, Path]
        書き出すディレクトリ
    split: Optional[dict] = None
        分割のしかた (method, valid, test, seed. 既定は random で検証用
        2 割. groups があれば group). unlabeled: keep で, 点群の無い撮影を
        unlabeled の分割に書く (既定 drop は除く)
    groups: Optional[dict] = None
        件の id (撮影のディレクトリの名前) からグループ (物や現場)
    origin: Optional[dict] = None
        正解の出どころ (type, name, settings)
    cloud_units: str = "meter"
        点群の単位 (meter, centimeter, millimeter)
    name: Optional[str] = None
        データセットの名前 (省略するとディレクトリの名前)
    overwrite: bool = False
        既にあるデータセットを置き換えるか
    item_steps: Optional[dict] = None
        項目にかける中身の手順 (capture は各枚の画像に resize, grayscale,
        rgb. cloud は sample_points). 渡すと版 3 で書く
    follow_links: bool = False
        撮影 (中の各枚と点群も) と点群のディレクトリの外を指すリンクの先も
        写すか (False では外を指す撮影, 枚, 点群を除いて notes に書く)
    parent: Optional[int] = None
        使う側が開いた, 書き出す場所の親のディレクトリの記述子 (渡すと
        path の最後の部分を親の中の名前とし, 途中の場所をたどらずに書く.
        閉じるのは使う側)

    Returns
    ----------
    info: dict
        書き出した dataset.yaml の中身

    Raises
    ----------
    ValueError
        組が無い場合, 姿勢の無い撮影がある場合, 単位, 分割と中身の手順の
        誤りの場合
    """
    # 中身の手順, 組, 単位, 正解の出どころ, 分割
    steps = entry_steps(
        item_steps, {"capture": "capture", "cloud": "point_cloud"}
    )
    if cloud_units not in CLOUD_UNITS:
        raise ValueError(
            f"cloud_units は {', '.join(CLOUD_UNITS)} のどれかにしてください"
        )
    scale = CLOUD_UNITS[cloud_units]
    data, pairs, notes = capture_pairs(
        captures, clouds, unlabeled_mode(split), follow_links
    )
    base = Path(captures).expanduser().absolute()
    alone = [pair for pair in pairs if pair[2] is None]
    pairs = [pair for pair in pairs if pair[2] is not None]
    if not pairs:
        raise ValueError(
            "撮影と点群の組がありません (撮影のディレクトリの名前と点群の"
            "ファイルの stem で組にします)"
        )
    origin_info = check_origin(origin)
    ids = [stem for stem, _, _ in pairs]
    labels, split_info, split_notes = split_items(ids, groups, split)
    notes += split_notes

    # 1 件ずつ, 正規化して写す (失敗したら一時ディレクトリを消す)
    final, target = start_writing(path, overwrite, parent)
    try:
        rows, frame_shapes, cloud_shapes, conventions = [], [], [], []
        # 正解の無い撮影は, 正解の付いた件の後ろに unlabeled で書く
        for index, (stem, position, cloud_path) in enumerate(pairs + alone):
            place = item_dir(index)
            summary = normalize_capture(
                data.load_item(position)["path"], target / place / "capture",
                steps.get("capture"), follow_links, base
            )
            shift = summary["shift"]
            notes += summary["notes"]
            lone = cloud_path is None
            row = {"id": stem, "split": UNLABELED if lone else labels[index]}
            if groups:
                row["group"] = (
                    lone_group(groups, stem) if lone else str(groups[stem])
                )
            row.update({"capture": f"{place}/capture", "cloud": None})
            if not lone:
                cloud = read_cloud(cloud_path)
                cloud["points"] = cloud["points"] * scale + shift
                cloud = apply_steps(cloud, steps.get("cloud", []))
                write_ply(target / place / "cloud.ply", cloud)
                row["cloud"] = f"{place}/cloud.ply"
                cloud_shapes.append((len(cloud["points"]), 3))
            row.update({
                "shift_x": float(shift[0]), "shift_y": float(shift[1]),
                "shift_z": float(shift[2]),
            })
            rows.append(row)
            frame_shapes.append((
                summary["frames"], summary["height"], summary["width"]
            ))
            conventions.append(summary["convention"])

        # 座標の決まり (深度の既定を足してから比べ, 撮影ごとに違えば先頭
        # のものを書いて注意. 深度の決まりが無ければ既定として注意)
        filled = [with_depth_convention(other) for other in conventions]
        convention, full = conventions[0], filled[0]
        if any(other != full for other in filled):
            notes.append(
                "撮影ごとに座標の決まり (coordinate_convention) が違います. "
                "説明には先頭の撮影のものを書きました"
            )
        lacking = [
            stem for (stem, _, _), other in zip(pairs + alone, conventions)
            if any(other.get(key) is None for key in DEFAULT_DEPTH)
        ]
        if lacking:
            notes.append(
                f"{len(lacking)} 件の撮影 ({', '.join(lacking[:EXAMPLES])}) "
                "の座標の決まりに深度の測り方 (depth) か画素の中心 "
                "(pixel_center) が無いので, depth: z (カメラの前の軸に沿った"
                "距離), pixel_center: 0.5 としました"
            )
        info = {
            "format_version": (
                ITEMS_STEPS_FORMAT_VERSION if steps else ITEMS_FORMAT_VERSION
            ),
            "name": name or final.name,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "role": "predict",
            "target": {
                "kind": "point_cloud", "entries": ["cloud"],
                "origin": origin_info,
            },
            "entries": {
                "capture": {
                    "kind": "capture", "role": "input",
                    "placement": "directory",
                    "pattern": f"{ITEMS_DIR}/{{index:06d}}/capture",
                    "formats": ["transforms.json"],
                    "shape": common_shape(frame_shapes),
                    "dtype": None, "units": "meter",
                    "convention": {
                        "pose": "camera_to_world",
                        "camera_axes": convention.get(
                            "camera_axes", DEFAULT_AXES
                        ),
                        "world_up": convention.get("world_up"),
                        "origin": "normalization",
                        "depth": full["depth"],
                        "pixel_center": full["pixel_center"],
                    },
                },
                "cloud": {
                    "kind": "point_cloud", "role": "target",
                    "placement": "file",
                    "pattern": f"{ITEMS_DIR}/{{index:06d}}/cloud.ply",
                    "formats": ["ply"],
                    "shape": common_shape(cloud_shapes),
                    "dtype": "float32", "units": "meter",
                    "convention": {
                        "frame": "capture", "origin": "normalization",
                    },
                },
            },
            "items": None,
            "split": split_info,
            "splits": None,
            "normalization": {
                "method": "view_center", "translation_only": True,
                "scale": 1.0, "units": "meter",
                "entries": ["capture", "cloud"],
                "columns": ["shift_x", "shift_y", "shift_z"],
                "formula": "normalized = original + shift",
            },
            "source": {
                "captures": source_name(captures),
                "clouds": source_name(clouds), "cloud_units": cloud_units,
                "paths": "name_only", "note": SOURCE_NOTE,
            },
            "notes": notes,
        }
        for key, listed_steps in steps.items():
            info["entries"][key]["steps"] = listed_steps
        info = write_items(target, final, rows, info, parent)
    except Exception:
        drop_writing(target, parent)
        raise

    return info


def with_channels(array: np.ndarray) -> np.ndarray:
    """
    画像の配列を (高さ, 幅, チャンネル) にそろえる関数

    Parameters
    ----------
    array: np.ndarray
        (高さ, 幅) か (高さ, 幅, チャンネル)

    Returns
    ----------
    result: np.ndarray
        (高さ, 幅, チャンネル)
    """
    # 灰色の画像にチャンネルの軸を足す
    result = array[:, :, None] if array.ndim == 2 else array

    return result


def read_depth(path: Path, scale: float, invalid: float) -> np.ndarray:
    """
    深度の画像をメートルの配列として読む関数

    Parameters
    ----------
    path: Path
        深度の画像
    scale: float
        画像の値をメートルにする倍率
    invalid: float
        欠けを表す値

    Returns
    ----------
    depth: np.ndarray
        (高さ, 幅) の float32 (欠けは 0)
    """
    # 欠けは 0 にそろえる
    from PIL import Image

    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float64)
    depth = np.where(values == invalid, 0.0, values * scale).astype(
        np.float32
    )

    return depth


def stack_or_none(arrays: list, count: int, label: str) -> Optional[Any]:
    """
    枚ごとの配列を重ねる関数

    全ての枚にある場合だけ重ねる

    Parameters
    ----------
    arrays: list
        枚ごとの配列 (無い枚は含まない)
    count: int
        枚数
    label: str
        誤りの文に使う呼び名

    Returns
    ----------
    stacked: Optional[Any]
        (枚数, ...) の配列 (無い枚があれば None)

    Raises
    ----------
    ValueError
        枚ごとの形が違う場合
    """
    # 全ての枚にあり, 形がそろう場合だけ重ねる
    stacked: Optional[Any] = None
    if len(arrays) != count or not arrays:
        return stacked
    if len({array.shape for array in arrays}) > 1:
        raise ValueError(f"撮影の{label}の大きさがそろいません")
    stacked = np.stack(arrays)

    return stacked


def with_depth_convention(convention: dict) -> dict:
    """
    撮影の座標の決まりに, 深度の測り方と画素の中心の既定を足す関数

    Parameters
    ----------
    convention: dict
        transforms.json の coordinate_convention

    Returns
    ----------
    full: dict
        depth (z か ray) と pixel_center を持つ写し (書かれていれば
        そのまま. null は書かれていないものとする)

    Raises
    ----------
    ValueError
        depth が z と ray のどちらでもない場合と, pixel_center が数で
        ない場合 (読み違えないため)
    """
    # 書かれていない欄 (null も) だけ既定
    full = dict(convention)
    for key, value in DEFAULT_DEPTH.items():
        if full.get(key) is None:
            full[key] = value
    center = full["pixel_center"]
    if full["depth"] not in DEPTH_KINDS:
        raise ValueError(
            f"撮影の coordinate_convention.depth は {', '.join(DEPTH_KINDS)}"
            f" のどちらかにしてください (今は {full['depth']})"
        )
    if isinstance(center, bool) or not isinstance(center, (int, float)):
        raise ValueError(
            "撮影の coordinate_convention.pixel_center は数にしてください "
            f"(今は {center})"
        )

    return full


def read_capture(
    path: Union[str, Path], follow_links: bool = False,
    base: Optional[Path] = None
) -> dict:
    """
    撮影のディレクトリの各枚を配列として読む関数

    load_items の capture の項目と同じ形で返す. 推論のときは
    normalize_capture で写した新しい撮影を読む

    Parameters
    ----------
    path: Union[str, Path]
        撮影のディレクトリ (transforms.json を持つ)
    follow_links: bool = False
        リンクをたどった後に外を指すファイルも読むか (False では無い
        ものとする)
    base: Optional[Path] = None
        外かを比べるディレクトリ (既定は撮影のディレクトリ. load_items は
        データセットのディレクトリを渡す)

    Returns
    ----------
    capture: dict
        intrinsics, poses ((枚数, 4, 4). 無い姿勢は NaN), images ((枚数,
        高さ, 幅, チャンネル) の uint8), depth ((枚数, 高さ, 幅) の
        float32, メートル, 欠けは 0. 無い枚があれば None), confidence
        (同じく uint8 か None), tracking, convention (深度の測り方
        depth と画素の中心 pixel_center を含む). 配列は全て書き込める

    Raises
    ----------
    ValueError
        画像の無い枚がある場合と, 大きさがそろわない場合
    """
    # 撮影の説明と, 外かを比べるディレクトリ
    path = Path(path).expanduser()
    limit = None if follow_links else (path if base is None else base)
    transforms = read_transforms(path, limit, ARGUMENT_HINT)

    # 深度は生の深度を優先 (信頼度と組)
    depth_info = transforms.get("depth") or {}
    scale = depth_scale(depth_info)
    invalid = depth_info.get("invalid_value")
    invalid = invalid if isinstance(invalid, (int, float)) else 0
    images, depths, confidences, poses, tracking = [], [], [], [], []
    for entry in transforms["frames"]:
        image = inside(path, entry.get("file_path"), limit)
        if image is None:
            raise ValueError(f"{path.name} の撮影に画像の無い枚があります")
        images.append(with_channels(read_image(image)))
        depth = inside(
            path, entry.get("raw_depth_file_path"), limit
        ) or inside(path, entry.get("depth_file_path"), limit)
        if depth is not None:
            depths.append(read_depth(depth, scale, invalid))
        confidence = inside(path, entry.get("confidence_file_path"), limit)
        if confidence is not None:
            from PIL import Image

            with Image.open(confidence) as opened:
                confidences.append(np.array(opened, dtype=np.uint8))
        pose = pose_of(entry)
        poses.append(np.full((4, 4), np.nan) if pose is None else pose)
        tracking.append(entry.get("tracking_state", "TRACKING"))
    count = len(images)
    capture = {
        "intrinsics": {key: transforms.get(key)
                       for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")},
        "poses": np.stack(poses) if poses else np.zeros((0, 4, 4)),
        "images": stack_or_none(images, count, "画像"),
        "depth": stack_or_none(depths, count, "深度"),
        "confidence": stack_or_none(confidences, count, "信頼度"),
        "tracking": tracking,
        "convention": with_depth_convention(
            transforms.get("coordinate_convention") or {}
        ),
    }

    return capture


def missing_place(spec: dict, value: Any) -> bool:
    """
    ファイルかディレクトリの項目の, 表の場所が空かを返す関数

    Parameters
    ----------
    spec: dict
        項目の説明 (placement)
    value: Any
        表の値 (parquet では None, csv では空の文字)

    Returns
    ----------
    missing: bool
        場所が空か (表の列の値の項目は常に False)
    """
    # 表の列の値は, 空でもその値
    if spec.get("placement") == "column":
        return False
    missing = value is None or value == "" or (
        isinstance(value, float) and np.isnan(value)
    )

    return missing


def read_entry(
    root: Path, spec: dict, value: Any, read: bool, follow_links: bool
) -> Any:
    """
    1 件の 1 つの項目を読む関数

    Parameters
    ----------
    root: Path
        データセットのディレクトリ
    spec: dict
        項目の説明 (kind, placement など)
    value: Any
        表の値 (場所か値)
    read: bool
        中身を配列として読むか (False なら場所)
    follow_links: bool
        リンクをたどった後にデータセットの外を指す場所も読むか

    Returns
    ----------
    content: Any
        中身 (配列か辞書), 場所 (絶対パスの文字), 表の値のどれか
    """
    # 表の列の値はそのまま, 読まない場合は場所
    content: Any = value
    if spec["placement"] == "column":
        return content
    path = within(
        root, value, spec["placement"] == "directory", follow_links
    )
    kind = spec.get("kind")
    content = str(path)
    if not read:
        return content

    # 種類ごとの読み方 (知らない種類は場所)
    if kind == "image":
        content: Any = with_channels(read_image(path))
    elif kind == "mask":
        content = read_mask(path).astype(spec.get("dtype") or "int64")
    elif kind == "point_cloud":
        content = read_cloud(path)["points"].astype(np.float32)
    elif kind == "capture":
        content = read_capture(path, follow_links, root)
    else:
        content = str(path)

    return content


@dataclass
class ItemSplit:
    # データセットのディレクトリ
    root: Path
    # 項目の説明
    entries: dict
    # この分割の件の表
    rows: pd.DataFrame
    # 中身を配列として読むか (False なら場所)
    read: bool = True
    # リンクをたどった後にデータセットの外を指す場所も読むか
    follow_links: bool = False

    def __len__(self) -> int:
        """
        分割の件数を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        count: int
            件数
        """
        # 表の行の数
        count = len(self.rows)

        return count

    def __getitem__(self, index: int) -> dict:
        """
        1 件を読む関数

        Parameters
        ----------
        index: int
            分割の中の位置 (負の数は後ろから)

        Returns
        ----------
        item: dict
            id, split, group (無ければ None) と, 項目の名前から中身

        Raises
        ----------
        IndexError
            位置が範囲の外の場合
        """
        # 位置の確かめ (範囲の外で for 文が止まる)
        if index < 0:
            index += len(self.rows)
        if not 0 <= index < len(self.rows):
            raise IndexError(f"{index} 番目の件はありません")
        row = self.rows.iloc[index]
        group = row.get("group")
        item = {
            "id": str(row["id"]), "split": str(row["split"]),
            "group": None if group is None or pd.isna(group) or group == ""
            else str(group),
        }
        for name, spec in self.entries.items():
            if not missing_place(spec, row[name]):
                item[name] = read_entry(
                    self.root, spec, row[name], self.read, self.follow_links
                )
            elif item["split"] == UNLABELED and spec.get("role") == "target":
                item[name] = None
            else:
                raise ValueError(
                    f"件 {item['id']} の項目 {name} の場所がありません ("
                    f"空にできるのは {UNLABELED} の分割の正解だけ)"
                )

        return item


@dataclass
class ItemDataset:
    # 書き出したディレクトリ
    path: Path
    # dataset.yaml の中身
    info: dict
    # 1 行 1 件の表
    items: pd.DataFrame
    # リンクをたどった後にデータセットの外を指す場所も読むか
    follow_links: bool = False

    @property
    def entries(self) -> dict:
        """
        項目の説明を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        entries: dict
            項目の名前から説明
        """
        # 説明の entries
        entries = self.info["entries"]

        return entries

    @property
    def splits(self) -> list[str]:
        """
        件のある分割の名前を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        names: list[str]
            分割の名前
        """
        # 説明の splits の名前
        names = list(self.info["splits"])

        return names

    def split(self, name: str, read: bool = True) -> ItemSplit:
        """
        分割の件の並びを返す関数

        Parameters
        ----------
        name: str
            分割の名前 (train, valid, test)
        read: bool = True
            中身を配列として読むか (False なら場所)

        Returns
        ----------
        items: ItemSplit
            len と番号で 1 件を返す並び

        Raises
        ----------
        ValueError
            その分割が無い場合
        """
        # 名前で選ぶ (件の無い分割は無い)
        if name not in self.info["splits"]:
            raise ValueError(
                f"分割 {name} はありません (あるもの: "
                f"{', '.join(self.splits)})"
            )
        rows = self.items[self.items["split"] == name].reset_index(drop=True)
        items = ItemSplit(
            root=self.path, entries=self.entries, rows=rows, read=read,
            follow_links=self.follow_links
        )

        return items


def number_entry(spec: dict) -> bool:
    """
    表の列の値の項目が, 数の型の値を持つかを返す関数

    Parameters
    ----------
    spec: dict
        項目の説明 (dtype)

    Returns
    ----------
    number: bool
        説明の dtype が数の型 (真偽値を除く) か
    """
    # 型の名前を numpy の型として読む (読めない名前は数でない)
    try:
        kind = np.dtype(str(spec.get("dtype"))).kind
    except TypeError:
        kind = ""
    number = kind != "" and kind in "iuf"

    return number


def csv_values(items: pd.DataFrame, entries: dict) -> pd.DataFrame:
    """
    csv で読んだ表の, 列の値の項目の空の値を欠けにする関数

    csv では欠けた値が空の文字になるので, 数の項目は数に (数にならない
    文字は欠け), 他の項目は文字のまま読み, 空の文字は欠けにする (csv では
    欠けと空の文字を見分けられないため)

    Parameters
    ----------
    items: pd.DataFrame
        csv で読んだ 1 行 1 件の表
    entries: dict
        項目の名前から説明

    Returns
    ----------
    items: pd.DataFrame
        列の値の項目を直した表
    """
    # 列の値の項目だけ (ファイルの場所の列は空の文字のまま)
    for name, spec in entries.items():
        if spec["placement"] != "column" or name not in items.columns:
            continue
        column = items[name]
        if number_entry(spec):
            items[name] = pd.to_numeric(
                column.replace("", np.nan), errors="coerce"
            )
        else:
            items[name] = column.map(
                lambda value: None if value == "" else value
            ).astype(object)

    return items


def load_items(
    path: Union[str, Path], follow_links: bool = False
) -> ItemDataset:
    """
    版 2 と 3 (入力と正解がファイルの組) のデータセットを読み込む関数

    Parameters
    ----------
    path: Union[str, Path]
        export_image_masks などで書き出したディレクトリ
    follow_links: bool = False
        表の場所のうち, リンクをたどった後にデータセットの外を指すものも
        読むか (False では 1 件を読むときに理由付きの誤り)

    Returns
    ----------
    dataset: ItemDataset
        説明と 1 行 1 件の表

    Raises
    ----------
    ValueError
        版 1 (load_dataset で読む) と知らない版の場合, 知らない置き方の
        項目がある場合, 表に項目の列が無い場合
    """
    # 説明と版の確認
    root = Path(path).expanduser().absolute()
    info = yaml.safe_load(
        within(root, "dataset.yaml", False, follow_links).read_text(
            encoding="utf-8"
        )
    )
    version = info.get("format_version") if isinstance(info, dict) else None
    if version == DATASET_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のデータセットは 1 件 = 1 行の表の"
            "もので, load_items では読めません. load_dataset で読んで"
            "ください"
        )
    if version not in ITEMS_FORMAT_VERSIONS:
        raise ValueError(
            f"format_version {version} のデータセットは, load_items (版 "
            f"{' と '.join(str(v) for v in ITEMS_FORMAT_VERSIONS)}) では"
            "読めません"
        )

    # 項目の置き方 (知らない置き方は読み違えないよう止まる)
    entries = info.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("dataset.yaml に entries がありません")
    check_entry_names(entries)
    for name, spec in entries.items():
        placement = spec.get("placement") if isinstance(spec, dict) else None
        if placement not in PLACEMENTS:
            raise ValueError(
                f"項目 {name} の置き方 {placement} は, このライブラリでは"
                f"読めません ({', '.join(PLACEMENTS)} だけ)"
            )

    # 1 行 1 件の表 (id とグループは文字のまま)
    file = within(root, str(info.get("items", "")), False, follow_links)
    if file.suffix == ".parquet":
        items = pd.read_parquet(file)
    else:
        texts = {"id": str, "split": str, "group": str}
        texts.update({
            name: str for name, spec in entries.items()
            if spec["placement"] == "column" and not number_entry(spec)
        })
        items = pd.read_csv(file, dtype=texts, keep_default_na=False)
        items = csv_values(items, entries)
    lacking = [name for name in ("id", "split", *entries)
               if name not in items.columns]
    if lacking:
        raise ValueError(
            f"{file.name} に列がありません: {', '.join(lacking)}"
        )
    dataset = ItemDataset(
        path=root, info=info, items=items, follow_links=follow_links
    )

    return dataset


def items_info(dataset: Union[ItemDataset, dict, str, Path]) -> dict:
    """
    版 2 か 3 のデータセットの説明を返す関数

    Parameters
    ----------
    dataset: Union[ItemDataset, dict, str, Path]
        load_items の結果, dataset.yaml の中身の辞書, データセットの
        ディレクトリのどれか

    Returns
    ----------
    info: dict
        dataset.yaml の中身

    Raises
    ----------
    ValueError
        版 2 と 3 でない場合
    """
    # 読んだ結果はその説明, ディレクトリは dataset.yaml の中身
    if isinstance(dataset, ItemDataset):
        info = dataset.info
        return info
    info = dataset if isinstance(dataset, dict) else yaml.safe_load(
        (Path(dataset).expanduser() / "dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    version = info.get("format_version") if isinstance(info, dict) else None
    if version not in ITEMS_FORMAT_VERSIONS:
        raise ValueError(
            f"format_version {version} のデータセットの項目には, 手順を"
            f"かけられません (版 "
            f"{' と '.join(str(v) for v in ITEMS_FORMAT_VERSIONS)} だけ)"
        )

    return info


def transform_entry(
    dataset: Union[ItemDataset, dict, str, Path], name: str, value: Any
) -> Any:
    """
    データセットの項目に残した中身の手順を, 新しい 1 件にかける関数

    推論のときに, 書き出しと同じ手順をかけるため. load_items がその項目
    に返すのと同じ形で返す

    Parameters
    ----------
    dataset: Union[ItemDataset, dict, str, Path]
        load_items の結果, dataset.yaml の中身の辞書, データセットの
        ディレクトリのどれか
    name: str
        項目の名前
    value: Any
        新しい 1 件 (画像と領域はファイルの場所か配列, 点群はファイルの
        場所, (点の数, 3) の配列か points を持つ辞書)

    Returns
    ----------
    content: Any
        画像は (高さ, 幅, チャンネル) の uint8, 領域は説明の dtype の
        (高さ, 幅), 点群は (点の数, 3) の float32

    Raises
    ----------
    ValueError
        版 2 と 3 でない場合, 項目が無い場合, 撮影と手順をかけられない
        種類の項目の場合, 知らない手順の場合
    """
    # 項目と手順 (読む側の知らない手順は止まる)
    info = items_info(dataset)
    spec = (info.get("entries") or {}).get(name)
    if not isinstance(spec, dict):
        raise ValueError(f"データセットに項目 {name} がありません")
    kind = spec.get("kind")
    if kind == "capture":
        raise ValueError(
            "撮影は normalize_capture(撮影, 写す先, steps=項目の steps) で"
            "写してから read_capture で読んでください"
        )
    if kind not in ("image", "mask", "point_cloud"):
        raise ValueError(f"項目 {name} の種類 {kind} には手順をかけられません")
    steps = check_entry_steps(spec.get("steps") or [], kind)

    # 種類ごとに load_items と同じ読み方で読み, 手順をかけた中身
    if kind == "image":
        image = value if isinstance(value, np.ndarray) else read_image(
            Path(value).expanduser()
        )
        content: Any = with_channels(
            np.asarray(apply_steps(image, steps), dtype=np.uint8)
        )
    elif kind == "mask":
        mask = value if isinstance(value, np.ndarray) else read_mask(
            Path(value).expanduser()
        )
        content = np.asarray(apply_steps(mask, steps)).astype(
            spec.get("dtype") or "int64"
        )
    else:
        cloud = value if isinstance(value, dict) else (
            {"points": np.asarray(value)} if isinstance(value, np.ndarray)
            else read_cloud(Path(value).expanduser())
        )
        content = np.asarray(
            apply_steps(cloud, steps)["points"], dtype=np.float32
        )

    return content
