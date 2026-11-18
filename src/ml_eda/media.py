import hashlib
import re
import struct
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.loaders import (
    Loader, guess_encoding, read_head, register_loader
)

# 画像の拡張子
IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"
)

# 点群の拡張子
POINT_EXTENSIONS = (".ply", ".pcd", ".xyz", ".pts", ".npy")

# 1 つのディレクトリから読むファイルの上限
MAX_FILES = 100_000

# 文として読む 1 ファイルの上限 (バイト)
MAX_TEXT_BYTES = 1 << 20


def list_files(root: Path, extensions: tuple) -> list[Path]:
    """
    ディレクトリの下 (2 階層まで) の, 拡張子の合うファイルを並べる関数

    Parameters
    ----------
    root: Path
        ディレクトリ
    extensions: tuple
        拡張子 (小文字, 点付き)

    Returns
    ----------
    files: list[Path]
        名前の順のファイル (隠しファイルを除く, 上限まで)
    """
    # 直下と, 1 つ下のディレクトリ (クラスごとのフォルダ) を探す. 隠しの
    # 判定は root より下の名前だけで行う
    files = []
    for depth in ("*", "*/*"):
        for path in root.glob(depth):
            hidden = any(
                part.startswith(".") for part in path.relative_to(root).parts
            )
            if path.is_file() and not hidden and (
                path.suffix.lower() in extensions
            ):
                files.append(path)
    files = sorted(files)[:MAX_FILES]

    return files


def labels_of(root: Path, files: list[Path]) -> Optional[list[str]]:
    """
    <クラス>/<ファイル> の形なら, フォルダの名前をラベルにする関数

    Parameters
    ----------
    root: Path
        ディレクトリ
    files: list[Path]
        ファイル

    Returns
    ----------
    labels: Optional[list[str]]
        ファイルごとのラベル, 全てが 1 つ下のフォルダに無ければ None
    """
    # 全てのファイルがクラスのフォルダの中にあり, クラスが 2 つ以上の場合
    # だけラベル付き (images/ だけのフォルダはラベルにしない)
    parents = [path.parent for path in files]
    if not files or any(parent == root for parent in parents):
        return None
    labels = [parent.name for parent in parents]
    if len(set(labels)) < 2:
        return None

    return labels


def sniff_image(path: Path) -> float:
    """
    画像 (1 枚か, 画像を集めたディレクトリ) として読めるかの確からしさを
    返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # 1 枚の画像か, 画像が多いディレクトリ
    if path.is_file():
        return 0.7 if path.suffix.lower() in IMAGE_EXTENSIONS else 0.0
    files = list_files(path, IMAGE_EXTENSIONS)
    every = [p for _, p in zip(range(2000), path.rglob("*")) if p.is_file()]
    if not files:
        return 0.0
    score = 0.9 if len(files) >= 0.5 * max(len(every), 1) else 0.4

    return score


def image_header(path: Path) -> dict:
    """
    画像のヘッダーだけを読み, 幅, 高さ, モードを返す関数

    Parameters
    ----------
    path: Path
        画像のファイル

    Returns
    ----------
    info: dict
        width, height, channels, mode, format (読めなければ None)
    """
    # 画素は読まずにヘッダーだけ. EXIF の向きが縦横を入れ替える場合は,
    # 見えるとおりの幅と高さにする
    from PIL import Image

    try:
        with Image.open(path) as image:
            mode = image.mode
            channels = len(image.getbands())
            width, height = image.width, image.height
            try:
                if image.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                    width, height = height, width
            except Exception:
                pass
            info = {
                "width": width, "height": height,
                "channels": 3 if mode == "P" else channels, "mode": mode,
                "format": image.format,
            }
    except Exception:
        info = {
            "width": None, "height": None, "channels": None, "mode": None,
            "format": None,
        }

    return info


def file_digest(path: Path) -> Optional[str]:
    """
    ファイルの中身の SHA-256 を求める関数

    同じ画像を見つけるため

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    digest: Optional[str]
        16 進の文字 (読めなければ None)
    """
    # 大きなファイルでもメモリに載せないよう, 1 MiB ずつの読み込み
    hasher = hashlib.sha256()
    digest: Optional[str] = None
    try:
        with open(path, "rb") as stream:
            for chunk in iter(partial(stream.read, 1 << 20), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    except OSError:
        digest = None

    return digest


def read_image(path: Path) -> np.ndarray:
    """
    画像を numpy の配列として読む関数

    Parameters
    ----------
    path: Path
        画像のファイル

    Returns
    ----------
    array: np.ndarray
        (高さ, 幅) か (高さ, 幅, チャンネル) の uint8 の配列
    """
    # EXIF の向きを直し, パレットは RGB, 16 ビットと浮動小数は値の範囲を
    # 0 から 255 に縮めた 1 チャンネル, 2 値は 0 と 255 の uint8 にそろえる
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode.startswith("I") or image.mode == "F":
            values = np.asarray(image, dtype=np.float64)
            top = 65535.0 if image.mode.startswith("I;16") else float(
                max(values.max(initial=0), 1)
            )
            array = np.clip(values / top * 255, 0, 255).astype(np.uint8)
            return array
        if image.mode == "1":
            return np.asarray(image, dtype=np.uint8) * 255
        if image.mode in ("P", "PA", "CMYK", "LA", "YCbCr", "LAB", "HSV"):
            image = image.convert("RGBA" if "A" in image.mode else "RGB")
        array = np.asarray(image)

    return array


def load_images(path: Path, **options: Any) -> Data:
    """
    画像のディレクトリ (か 1 枚の画像) を読み込む関数

    Parameters
    ----------
    path: Path
        ディレクトリか画像のファイル
    **options: Any
        digest (既定 True. False なら中身の hash を求めない. 大きな
        フォルダで全てのファイルを読む時間を省くため)

    Returns
    ----------
    data: Data
        1 行が 1 枚の画像のデータ (path, label, width, height, channels,
        mode, format, bytes, digest). 画素は load_item で読む
    """
    # ファイルの一覧とラベル (無い場合は置き方を案内)
    root = path if path.is_dir() else path.parent
    files = list_files(path, IMAGE_EXTENSIONS) if path.is_dir() else [path]
    labels = labels_of(root, files) if path.is_dir() else None
    if not files:
        raise ValueError(
            f"{path.name} の直下か 1 つ下に画像がありません (train/cat/x.png "
            "のような形なら, train のディレクトリを選んでください)"
        )

    # ヘッダーから大きさとモード
    rows = []
    for index, file in enumerate(files):
        row = {"path": file.relative_to(root).as_posix()}
        if labels:
            row["label"] = labels[index]
        row.update(image_header(file))
        row["bytes"] = file.stat().st_size
        if options.get("digest", True):
            row["digest"] = file_digest(file)
        rows.append(row)
    frame = pd.DataFrame(rows)
    roles = {"path": "path"}
    if labels:
        roles["label"] = "label"
    data = Data(
        kind="image", frame=frame, roles=roles,
        item_loader=lambda i: read_image(root / frame["path"].iloc[i])
    )
    data.source = {"root": str(root)}

    return data


def sniff_text(path: Path) -> float:
    """
    文 (1 行 1 文のファイルか, 文のファイルを集めたディレクトリ) として
    読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # 区切りの揃わない .txt か, .txt の多いディレクトリ
    if path.is_dir():
        files = list_files(path, (".txt",))
        return 0.85 if files and labels_of(path, files) else (
            0.5 if files else 0.0
        )
    if path.suffix.lower() not in (".txt", ".text", ".md", ""):
        return 0.0
    head = read_head(path)
    if b"\x00" in head[:4096]:
        return 0.0
    lines = [
        line
        for line in head.decode(guess_encoding(head), errors="replace")
        .splitlines()
        if line.strip()
    ]
    # 先頭だけを読んだ場合, 最後の行は途中で切れているので除く
    if len(head) >= 64 * 1024:
        lines = lines[:-1]
    lines = lines[:50]
    if not lines:
        return 0.0
    # 全ての行が数の並びなら, 文ではなく表か点群
    numeric = all(
        re.fullmatch(r"[\s\d.,eE+-]+", line) is not None for line in lines
    )
    if numeric:
        return 0.1
    # どれかの区切り文字が全ての行で同じ数 (1 以上) なら表らしい
    tabular = any(
        len(counts) == 1 and counts != {0}
        for counts in (
            {line.count(mark) for line in lines}
            for mark in (",", "\t", ";", "|")
        )
    )
    score = 0.4 if tabular else 0.7

    return score


def load_text(path: Path, **options: Any) -> Data:
    """
    文を読み込む関数 (1 行 1 文のファイルか, <クラス>/<文>.txt)

    Parameters
    ----------
    path: Path
        ファイルかディレクトリ
    **options: Any
        encoding (文字コード, 既定 utf-8)

    Returns
    ----------
    data: Data
        1 行が 1 つの文のデータ (text, label, length, words)
    """
    # ディレクトリはファイルごとに 1 つの文, ファイルは 1 行ごとに 1 つの文.
    # 文字コードは指定が無ければファイルごとに推定
    encoding = options.get("encoding")
    rows = []
    if path.is_dir():
        files = list_files(path, (".txt",))
        labels = labels_of(path, files)
        for index, file in enumerate(files):
            raw = file.read_bytes()[:MAX_TEXT_BYTES]
            text = raw.decode(
                encoding or guess_encoding(raw), errors="replace"
            )
            row = {"text": text, "file": file.relative_to(path).as_posix()}
            if labels:
                row["label"] = labels[index]
            rows.append(row)
    else:
        raw = path.read_bytes()
        content = raw.decode(
            encoding or guess_encoding(raw[:65536]), errors="replace"
        )
        rows = [
            {"text": line} for line in content.splitlines() if line.strip()
        ]
    if not rows:
        raise ValueError(f"{path.name} に文がありません")
    frame = pd.DataFrame(rows)

    # 文の長さと語の数 (分析と絞り込みのため)
    frame["length"] = frame["text"].str.len()
    frame["words"] = frame["text"].str.split().map(len)
    roles = {"text": "text"}
    if "label" in frame.columns:
        roles["label"] = "label"
    data = Data(
        kind="text", frame=frame, roles=roles,
        item_loader=lambda i: frame["text"].iloc[i]
    )

    return data


def parse_ply(path: Path) -> dict:
    """
    PLY (ASCII かバイナリ) の頂点を読む関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    cloud: dict
        points ((N, 3) の float32), colors ((N, 3) の uint8 か None),
        normals ((N, 3) か None)

    Raises
    ----------
    ValueError
        PLY でない場合と, 頂点の座標が無い場合
    """
    # ヘッダーの読み込み
    raw = path.read_bytes()
    found = re.search(rb"^end_header\r?$", raw, re.M)
    if not raw.startswith(b"ply") or found is None:
        raise ValueError(f"{path.name} は PLY ではありません")
    end = found.start()
    header = raw[:end].decode("ascii", errors="replace").splitlines()
    body = raw[raw.index(b"\n", end) + 1:]
    form = next(
        line.split()[1] for line in header if line.startswith("format")
    )
    elements = []
    for line in header:
        parts = line.split()
        if parts[:1] == ["element"]:
            elements.append({"name": parts[1], "count": int(parts[2]),
                             "props": []})
        elif parts[:1] == ["property"] and elements:
            elements[-1]["props"].append(parts[1:])

    # 頂点の要素の前にある要素の大きさは分からないので, 頂点が最初の場合だけ
    vertex = elements[0] if elements else None
    if not vertex or vertex["name"] != "vertex":
        raise ValueError("PLY の最初の要素が頂点ではありません")
    if any(prop[0] == "list" for prop in vertex["props"]):
        raise ValueError("頂点にリストの属性がある PLY は読めません")
    types = {
        "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
        "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
        "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
        "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
    }
    names = [prop[1] for prop in vertex["props"]]
    count = vertex["count"]
    if form == "ascii":
        values = np.loadtxt(
            body.decode("ascii", errors="replace").splitlines()[:count],
            ndmin=2
        )
        table = {name: values[:, k] for k, name in enumerate(names)}
    else:
        order = "<" if form == "binary_little_endian" else ">"
        dtype = np.dtype([
            (prop[1], order + types[prop[0]]) for prop in vertex["props"]
        ])
        array = np.frombuffer(body, dtype=dtype, count=count)
        table = {name: array[name] for name in names}
    cloud = cloud_from_columns(table)

    return cloud


def cloud_from_columns(table: dict) -> dict:
    """
    列の名前から, 座標, 色, 法線を取り出す関数

    Parameters
    ----------
    table: dict
        列の名前から値の配列

    Returns
    ----------
    cloud: dict
        points, colors, normals

    Raises
    ----------
    ValueError
        x, y, z が無い場合
    """
    # 名前の決まりで取り出す
    # 座標は精度を保つ float64 (地理座標など). 色は浮動小数で 1 以下なら
    # 0 から 1 の値として 255 倍する
    if not all(axis in table for axis in "xyz"):
        raise ValueError("点群に x, y, z の座標がありません")
    points = np.stack([table[a] for a in "xyz"], axis=1).astype(np.float64)
    if len(points) == 0:
        raise ValueError("点が 1 つもありません")
    colors = None
    for keys in (("red", "green", "blue"), ("r", "g", "b")):
        if all(key in table for key in keys):
            colors = np.stack([table[k] for k in keys], axis=1)
            floating = np.issubdtype(colors.dtype, np.floating)
            if floating and colors.max(initial=0) <= 1.0:
                colors = colors * 255
            colors = colors.clip(0, 255).astype(np.uint8)
    normals = None
    for keys in (("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")):
        if all(key in table for key in keys):
            normals = np.stack([table[k] for k in keys], axis=1)
    cloud = {"points": points, "colors": colors, "normals": normals}

    return cloud


def parse_pcd(path: Path) -> dict:
    """
    PCD (ASCII) の点を読む関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    cloud: dict
        points, colors, normals

    Raises
    ----------
    ValueError
        ASCII でない場合
    """
    # ヘッダーの FIELDS, TYPE, COUNT と DATA を読む (COUNT が 2 以上の
    # フィールドは 名前_番号 の列に分ける)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    fields: list[str] = []
    types: list[str] = []
    counts: list[int] = []
    start = 0
    for number, line in enumerate(lines):
        parts = line.split()
        if parts[:1] == ["FIELDS"]:
            fields = parts[1:]
        if parts[:1] == ["TYPE"]:
            types = parts[1:]
        if parts[:1] == ["COUNT"]:
            counts = [int(value) for value in parts[1:]]
        if parts[:1] == ["DATA"]:
            if parts[1] != "ascii":
                raise ValueError("バイナリの PCD は読めません (ASCII だけ)")
            start = number + 1
            break
    values = np.loadtxt(lines[start:], ndmin=2)
    counts = counts or [1] * len(fields)
    names = [
        name if count == 1 else f"{name}_{k}"
        for name, count in zip(fields, counts) for k in range(count)
    ]
    table = {name: values[:, k] for k, name in enumerate(names)}
    kinds = dict(zip(fields, types or ["F"] * len(fields)))
    for key in ("rgb", "rgba"):
        if key not in table:
            continue
        # F は float の中に詰めた値, U と I は整数のまま
        packed = (
            table[key].astype(np.float32).view(np.uint32)
            if kinds.get(key, "F") == "F"
            else table[key].astype(np.uint64).astype(np.uint32)
        )
        table["red"] = ((packed >> 16) & 255).astype(np.uint8)
        table["green"] = ((packed >> 8) & 255).astype(np.uint8)
        table["blue"] = (packed & 255).astype(np.uint8)
    cloud = cloud_from_columns(table)

    return cloud


def parse_xyz(path: Path) -> dict:
    """
    空白区切りの点 (.xyz, .pts) と numpy の配列 (.npy) を読む関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    cloud: dict
        points, colors (6 列以上なら 4 から 6 列目), normals (None)
    """
    # .npy は pickle を許さずに, それ以外は空白かカンマ区切りの表として
    # 読む. 先頭の行が数でなければ列の名前にする (1 語なら点の数)
    header: list[str] = []
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        rows = [line.replace(",", " ") for line in lines if line.strip()]
        if rows and not re.fullmatch(r"[\s\d.eE+-]+", rows[0]):
            header = rows.pop(0).split()
        elif rows and len(rows[0].split()) == 1:
            rows = rows[1:]
        values = np.loadtxt(rows, ndmin=2)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"点群は (点の数, 3 以上) の形にしてください {values.shape}")
    if len(header) == values.shape[1]:
        table = {name.lower(): values[:, k] for k, name in enumerate(header)}
        return cloud_from_columns(table)
    table = {axis: values[:, k] for k, axis in enumerate("xyz")}

    # .pts は x y z 強度 r g b. 他は 6 列目まで, 0 から 255 の整数なら色,
    # 長さ 1 のベクトルなら法線
    extra = values[:, 4:7] if (
        path.suffix.lower() == ".pts" and values.shape[1] >= 7
    ) else values[:, 3:6] if values.shape[1] >= 6 else None
    if extra is not None:
        whole = np.all(np.mod(extra, 1) == 0) and extra.min() >= 0 and (
            extra.max() <= 255
        )
        unit = np.allclose(np.linalg.norm(extra, axis=1), 1, atol=1e-3)
        if whole:
            table.update({"red": extra[:, 0], "green": extra[:, 1],
                          "blue": extra[:, 2]})
        elif unit:
            table.update({"nx": extra[:, 0], "ny": extra[:, 1],
                          "nz": extra[:, 2]})
    cloud = cloud_from_columns(table)

    return cloud


def read_cloud(path: Path) -> dict:
    """
    拡張子に合わせて点群を読む関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    cloud: dict
        points, colors, normals
    """
    # 形式ごとの読み込み
    suffix = path.suffix.lower()
    readers: dict[str, Callable[[Path], dict]] = {
        ".ply": parse_ply, ".pcd": parse_pcd,
    }
    cloud = readers.get(suffix, parse_xyz)(path)

    return cloud


def sniff_points(path: Path) -> float:
    """
    点群 (1 つのファイルか, 点群のファイルを集めたディレクトリ) として
    読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # PLY と PCD は印, .xyz などは拡張子, .npy は (N, 3 以上) の形
    if path.is_dir():
        files = list_files(path, (".ply", ".pcd", ".xyz", ".pts"))
        return 0.85 if files else 0.0
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return 0.95 if read_head(path, 3) == b"ply" else 0.0
    if suffix == ".pcd":
        return 0.9
    if suffix in (".xyz", ".pts"):
        return 0.8
    if suffix == ".npy" and read_head(path, 6) == b"\x93NUMPY":
        try:
            shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
        except ValueError:
            return 0.0
        named = re.search(
            r"(^|[_\-.])(points?|cloud|pcd|xyz|lidar)([_\-.]|$)",
            path.name, re.I
        )
        if len(shape) == 2 and 3 <= shape[1] <= 9 and shape[0] > 50:
            return 0.85 if named else 0.5
    score = 0.0

    return score


def describe_cloud(cloud: dict) -> dict:
    """
    点群の点の数, 範囲, 属性を求める関数

    Parameters
    ----------
    cloud: dict
        points, colors, normals

    Returns
    ----------
    info: dict
        points (点の数), 各軸の最小と最大と幅, 中心, 色と法線の有無
    """
    # 軸ごとの範囲
    points = cloud["points"]
    finite = points[np.isfinite(points).all(axis=1)]
    low = finite.min(axis=0) if len(finite) else np.zeros(3)
    high = finite.max(axis=0) if len(finite) else np.zeros(3)
    info: dict = {"points": int(len(points))}
    for k, axis in enumerate("xyz"):
        info[f"{axis}_min"] = float(low[k])
        info[f"{axis}_max"] = float(high[k])
        info[f"{axis}_extent"] = float(high[k] - low[k])
    center = finite.mean(axis=0) if len(finite) else np.zeros(3)
    info.update({
        "center_x": float(center[0]), "center_y": float(center[1]),
        "center_z": float(center[2]),
        "has_color": cloud.get("colors") is not None,
        "has_normal": cloud.get("normals") is not None,
        "invalid_points": int(len(points) - len(finite)),
    })

    return info


def load_points(path: Path, **options: Any) -> Data:
    """
    点群 (1 つのファイルか, 点群のファイルを集めたディレクトリ) を読む関数

    Parameters
    ----------
    path: Path
        ファイルかディレクトリ
    **options: Any
        使わない

    Returns
    ----------
    data: Data
        1 行が 1 つの点群のデータ (path, label, 点の数, 範囲, 属性).
        点は load_item で読む
    """
    # ファイルの一覧とラベル (無い場合は誤り)
    root = path if path.is_dir() else path.parent
    files = (
        list_files(path, (".ply", ".pcd", ".xyz", ".pts"))
        if path.is_dir() else [path]
    )
    if not files:
        raise ValueError(f"{path.name} に点群のファイルがありません")
    labels = labels_of(root, files) if path.is_dir() else None

    # ファイルごとの点の数と範囲
    rows = []
    for index, file in enumerate(files):
        row: dict = {"path": file.relative_to(root).as_posix()}
        if labels:
            row["label"] = labels[index]
        try:
            row.update(describe_cloud(read_cloud(file)))
        except (ValueError, OSError, struct.error, IndexError) as error:
            row["error"] = str(error) or type(error).__name__
        rows.append(row)
    frame = pd.DataFrame(rows)
    roles = {"path": "path"}
    if labels:
        roles["label"] = "label"
    data = Data(
        kind="pointcloud", frame=frame, roles=roles,
        item_loader=lambda i: read_cloud(root / frame["path"].iloc[i])
    )
    data.source = {"root": str(root)}

    return data


def describe_items(data: Data, seed: int = 0, sample: int = 200) -> dict:
    """
    画像, 文, 点群の種類ごとの分析と気付きを求める関数

    Parameters
    ----------
    data: Data
        中身を持つデータ
    seed: int = 0
        見本を選ぶ乱数のシード
    sample: int = 200
        中身を読む見本の件数 (画像の明るさなど)

    Returns
    ----------
    result: dict
        種類ごとの集計 (sizes, channels, brightness など) と, 気付き
        (findings)
    """
    # 種類ごとの集計 (ファイルが上限を超えて切った場合は気付きに)
    frame = data.frame
    result: dict = {"kind": data.kind, "count": int(len(frame))}
    findings = []
    if len(frame) >= MAX_FILES:
        findings.append({
            "severity": "warning", "code": "too_many_files",
            "message": (
                f"ファイルが多いので, 名前の順で先頭の {MAX_FILES} 件だけを"
                "読みました (後ろのクラスが欠けている場合があります)"
            ),
            "columns": [],
        })
    if data.kind == "image":
        valid = frame.dropna(subset=["width"])
        unreadable = int(len(frame) - len(valid))
        sizes = valid.groupby(["width", "height"]).size().sort_values(
            ascending=False
        )
        result["sizes"] = [
            {"width": int(w), "height": int(h), "count": int(c)}
            for (w, h), c in sizes.head(20).items()
        ]
        result["modes"] = {
            str(k): int(v) for k, v in valid["mode"].value_counts().items()
        }
        result["unreadable"] = unreadable

        # 見本の画像の明るさ (チャンネルごとの平均)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            valid.index.to_numpy(), min(sample, len(valid)), replace=False
        ) if len(valid) else []
        means = []
        for index in chosen:
            try:
                array = np.asarray(
                    data.load_item(int(frame.index.get_loc(index))),
                    dtype=np.float32
                )
            except Exception:
                continue
            array = array[:, :, None] if array.ndim == 2 else array
            means.append(array[:, :, :3].mean(axis=(0, 1)))
        if means:
            result["brightness"] = np.mean(
                [m.mean() for m in means]
            ).round(2).item()
        if len(sizes) > 1:
            findings.append({
                "severity": "warning", "code": "mixed_sizes",
                "message": (
                    f"画像の大きさが {len(sizes)} 種類あります (最も多いのは "
                    f"{result['sizes'][0]['width']}x"
                    f"{result['sizes'][0]['height']}). 学習の前に resize で"
                    "そろえてください"
                ),
                "columns": ["width", "height"],
            })
        if "digest" in frame.columns:
            # 同じ画像 (ファイルの中身が同じ. 分割をまたぐと検証が甘くなる)
            digests = frame["digest"].dropna()
            repeated = int(digests.duplicated().sum())
            result["duplicates"] = repeated
            if repeated:
                findings.append({
                    "severity": "warning", "code": "duplicate_items",
                    "message": (
                        f"同じ画像 (ファイルの中身が同じ) が {repeated} 件"
                        "あります (2 回目以降の数). 学習用と検証用に分かれる"
                        "と正解率が甘くなるので, drop_duplicates (columns: "
                        "[digest]) で落としてください"
                    ),
                    "columns": ["digest"],
                })
        if len(result["modes"]) > 1:
            findings.append({
                "severity": "info", "code": "mixed_modes",
                "message": (
                    "色の形式がそろいません ("
                    + ", ".join(result["modes"]) + "). 灰色にそろえるなら "
                    "grayscale を使ってください"
                ),
                "columns": ["mode"],
            })
    elif data.kind == "pointcloud":
        unreadable = int(frame["error"].notna().sum()) if (
            "error" in frame.columns
        ) else 0
        result["unreadable"] = unreadable
        if "points" in frame.columns and frame["points"].nunique() > 1:
            findings.append({
                "severity": "info", "code": "mixed_points",
                "message": (
                    "点の数が点群ごとに違います (最少 "
                    f"{int(frame['points'].min())}, 最多 "
                    f"{int(frame['points'].max())}). 学習の前に "
                    "sample_points でそろえてください"
                ),
                "columns": ["points"],
            })
        invalid = frame.get("invalid_points")
        if invalid is not None and invalid.fillna(0).sum() > 0:
            # 距離のセンサーの欠けなど. sample_points は有限の点だけを使う
            clouds = int((invalid.fillna(0) > 0).sum())
            findings.append({
                "severity": "info", "code": "invalid_points",
                "message": (
                    f"座標が数でない点 (NaN, inf) が {clouds} 件の点群に計 "
                    f"{int(invalid.fillna(0).sum())} 点あります. "
                    "sample_points は有限の点だけを使います"
                ),
                "columns": ["invalid_points"],
            })
    else:
        unreadable = 0
    if unreadable:
        findings.append({
            "severity": "warning", "code": "unreadable_items",
            "message": f"読めないファイルが {unreadable} 件あります",
            "columns": [],
        })
    if data.kind == "text":
        # 同じ文 (分割をまたぐと検証が甘くなる) と空の文
        text = data.roles.get("text", "text")
        values = frame[text].fillna("").astype(str).str.strip()
        empty = int((values == "").sum())
        repeated = int(values[values != ""].duplicated().sum())
        result["empty"] = empty
        result["duplicates"] = repeated

        # 文を区切る単位 (文字, 語, 部分語) の比較と vocabulary の勧め
        from ml_eda.textunits import compare_tokenizers

        result["tokenizers"] = compare_tokenizers(
            values[values != ""].tolist(), seed
        )
        if repeated:
            findings.append({
                "severity": "warning", "code": "duplicate_items",
                "message": (
                    f"同じ文が {repeated} 件あります (2 回目以降の数). "
                    "学習用と検証用に分かれると正解率が甘くなるので, "
                    f"drop_duplicates (columns: [{text}]) で落としてください"
                ),
                "columns": [text],
            })
        if empty:
            findings.append({
                "severity": "info", "code": "empty_items",
                "message": (
                    f"空の文が {empty} 件あります. 全てが詰め物になるので, "
                    f"filter_rows ({data.roles.get('length', 'length')} が 0 "
                    "より大きい) で除くことを検討してください"
                ),
                "columns": [text],
            })
    result["findings"] = findings

    return result


def register_collections() -> None:
    """
    画像, 文, 点群の読み込みを登録する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 件ずつの中身を持つ種類
    register_loader(Loader(
        "image_folder", "image", IMAGE_EXTENSIONS, sniff_image, load_images,
        {}, "画像 (1 枚か, <クラス>/<画像> のディレクトリ. Pillow が要る)"
    ))
    register_loader(Loader(
        "text_lines", "text", (".txt",), sniff_text, load_text,
        {"encoding": "文字コード (既定 utf-8)"},
        "文 (1 行 1 文の .txt か, <クラス>/<文>.txt のディレクトリ)"
    ))
    register_loader(Loader(
        "pointcloud", "pointcloud", POINT_EXTENSIONS, sniff_points,
        load_points, {},
        "点群 (.ply, .pcd (ASCII), .xyz, .pts, .npy. 1 つかディレクトリ)"
    ))


register_collections()
