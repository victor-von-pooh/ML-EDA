import json
import math
import struct
import zlib
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ml_eda.items import DEFAULT_AXES, DEFAULT_DEPTH, write_ply

# 領域の値の名前 (write_image_masks)
MASK_CLASSES = {0: "background", 1: "circle", 2: "square"}

# 既定の形 (1 つの撮影に 1 つの物. 位置は世界の原点から離しておく)
DEFAULT_SHAPES = (
    {"name": "sphere_a", "type": "sphere", "center": [0.4, 0.1, -1.2],
     "radius": 0.15, "color": [220, 80, 60]},
    {"name": "sphere_b", "type": "sphere", "center": [-0.6, 0.2, 0.5],
     "radius": 0.1, "color": [60, 160, 220]},
    {"name": "box_a", "type": "box", "center": [-0.5, 0.15, 0.7],
     "size": [0.3, 0.2, 0.25], "color": [80, 200, 90]},
    {"name": "box_b", "type": "box", "center": [0.8, 0.1, 0.3],
     "size": [0.15, 0.15, 0.3], "color": [230, 200, 60]},
)

# 乱数で作る形の中心の, 原点からの水平の距離の範囲 (メートル)
RANDOM_DISTANCE = (0.5, 1.5)

# 乱数で作る形の中心の高さの範囲 (メートル)
RANDOM_HEIGHT = (0.05, 0.3)

# 乱数で作る形の色の範囲 (RGB の各値)
RANDOM_COLOR = (60, 240)

# 乱数で作る形の大きさの範囲 (球の半径と箱の各辺. メートル)
RANDOM_RADIUS = (0.08, 0.18)
RANDOM_SIZE = (0.1, 0.3)

# 撮影の周回 (物の中心からの水平の距離と高さ. メートル)
ORBIT_RADIUS = 0.6
ORBIT_HEIGHT = 0.25


def png_chunk(kind: bytes, data: bytes) -> bytes:
    """
    PNG の 1 つのチャンクを作る関数

    Parameters
    ----------
    kind: bytes
        チャンクの種類 (IHDR, IDAT, IEND)
    data: bytes
        中身

    Returns
    ----------
    chunk: bytes
        長さ, 種類, 中身, CRC
    """
    # 長さと CRC は大きいエンディアン
    chunk = (
        struct.pack(">I", len(data)) + kind + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )

    return chunk


def write_png(path: Path, array: np.ndarray) -> None:
    """
    配列を PNG で書く関数

    標準ライブラリだけで書く (Pillow が無くても見本を作れるように)

    Parameters
    ----------
    path: Path
        書く場所
    array: np.ndarray
        (高さ, 幅) の uint8 か uint16 (灰色), (高さ, 幅, 3) の uint8 (RGB)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        形か型が上のどれでもない場合
    """
    # 色の形式とビットの深さ
    array = np.asarray(array)
    if array.ndim == 2 and array.dtype == np.uint16:
        depth, color, raw = 16, 0, array.astype(">u2")
    elif array.ndim == 2 and array.dtype == np.uint8:
        depth, color, raw = 8, 0, array
    elif array.ndim == 3 and array.shape[2] == 3 and array.dtype == np.uint8:
        depth, color, raw = 8, 2, array
    else:
        raise ValueError(
            f"PNG に書けない配列です ({array.shape}, {array.dtype})"
        )

    # 行ごとに絞り込みの種類 0 を付けて圧縮
    height, width = array.shape[:2]
    rows = b"".join(b"\x00" + row.tobytes() for row in raw)
    header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(rows)) + png_chunk(b"IEND", b"")
    )


def write_image_masks(
    root: Union[str, Path], count: int = 12,
    size: tuple[int, int] = (24, 32), seed: int = 0
) -> dict:
    """
    円か四角を 1 つ置いた画像と, その領域の見本を書く関数

    Parameters
    ----------
    root: Union[str, Path]
        書くディレクトリ (images と masks を作る)
    count: int = 12
        組の数
    size: tuple[int, int] = (24, 32)
        画像の (高さ, 幅)
    seed: int = 0
        乱数の種

    Returns
    ----------
    result: dict
        images, masks (ディレクトリ), classes (値から名前), names (stem)
    """
    # 雑音の背景に, 円 (値 1) と四角 (値 2) を交互に置く
    root = Path(root).expanduser()
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    height, width = size
    rows, columns = np.mgrid[0:height, 0:width]
    names = []
    for index in range(count):
        radius = rng.uniform(0.15, 0.3) * min(height, width)
        center_row = rng.uniform(radius, height - radius)
        center_column = rng.uniform(radius, width - radius)
        if index % 2 == 0:
            inside = (rows - center_row) ** 2 + (
                columns - center_column
            ) ** 2 <= radius ** 2
            value = 1
        else:
            inside = (np.abs(rows - center_row) <= radius) & (
                np.abs(columns - center_column) <= radius
            )
            value = 2
        image = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
        image[inside] = rng.integers(120, 256, size=3, dtype=np.uint8)
        mask = np.where(inside, value, 0).astype(np.uint8)
        name = f"sample_{index:03d}"
        write_png(images / f"{name}.png", image)
        write_png(masks / f"{name}.png", mask)
        names.append(name)
    result = {
        "images": images, "masks": masks, "classes": dict(MASK_CLASSES),
        "names": names,
    }

    return result


def sample_surface(
    shape: dict, count: int, rng: np.random.Generator
) -> np.ndarray:
    """
    形の面の上に一様に点を置く関数

    置いた点は全て面の上にある (正確な点群)

    Parameters
    ----------
    shape: dict
        type (sphere か box), center, radius (球), size (箱の各辺)
    count: int
        点の数
    rng: np.random.Generator
        乱数

    Returns
    ----------
    points: np.ndarray
        (count, 3) の世界の座標 (メートル)

    Raises
    ----------
    ValueError
        知らない形の場合
    """
    # 球は正規分布の向きを半径に, 箱は面積に比例して面を選ぶ
    center = np.asarray(shape["center"], dtype=np.float64)
    if shape["type"] == "sphere":
        normal = rng.normal(size=(count, 3))
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        points = center + shape["radius"] * normal
        return points
    if shape["type"] != "box":
        raise ValueError(f"知らない形です: {shape['type']}")
    half = np.asarray(shape["size"], dtype=np.float64) / 2
    areas = np.array([half[1] * half[2], half[0] * half[2],
                      half[0] * half[1]])
    axes = rng.choice(3, size=count, p=areas / areas.sum())
    points = rng.uniform(-half, half, size=(count, 3))
    signs = rng.choice([-1.0, 1.0], size=count)
    points[np.arange(count), axes] = signs * half[axes]
    points += center

    return points


def intersect(
    shape: dict, origin: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    光線と形の最も近い交わりを求める関数

    球と箱の式から解析的に求める

    Parameters
    ----------
    shape: dict
        type (sphere か box), center, radius, size
    origin: np.ndarray
        光線の始まり (カメラの位置. 形の外)
    directions: np.ndarray
        (光線の数, 3) の向き (長さは 1 でなくてよい)

    Returns
    ----------
    result: tuple[np.ndarray, np.ndarray]
        向きの何倍で当たるか (当たらなければ inf) と, 当たった面の法線
    """
    # 球は 2 次方程式の小さい方の根
    center = np.asarray(shape["center"], dtype=np.float64)
    offset = origin - center
    if shape["type"] == "sphere":
        a = np.sum(directions ** 2, axis=1)
        b = 2 * directions @ offset
        c = offset @ offset - shape["radius"] ** 2
        disc = b ** 2 - 4 * a * c
        root = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        hit = (disc >= 0) & (root > 0)
        t = np.where(hit, root, np.inf)
        normals = (origin + t[:, None] * directions - center) / (
            shape["radius"]
        )
        result = (t, np.where(hit[:, None], normals, 0.0))
        return result

    # 箱は軸ごとの板 (slab) の入りと出
    half = np.asarray(shape["size"], dtype=np.float64) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / directions
        near = (-half - offset) * inverse
        far = (half - offset) * inverse
    enter = np.nan_to_num(np.minimum(near, far), nan=-np.inf)
    leave = np.nan_to_num(np.maximum(near, far), nan=np.inf)
    t_in, t_out = enter.max(axis=1), leave.min(axis=1)
    hit = (t_in <= t_out) & (t_in > 0)
    t = np.where(hit, t_in, np.inf)
    axis = enter.argmax(axis=1)
    normals = np.zeros_like(directions)
    normals[np.arange(len(axis)), axis] = -np.sign(
        directions[np.arange(len(axis)), axis]
    )
    result = (t, np.where(hit[:, None], normals, 0.0))

    return result


def look_at(position: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    位置から中心を向くカメラの姿勢を作る関数

    OpenGL の決まり (カメラの +X が右, +Y が上, -Z が前). 世界の上は +Y

    Parameters
    ----------
    position: np.ndarray
        カメラの位置
    center: np.ndarray
        見る点

    Returns
    ----------
    pose: np.ndarray
        カメラから世界への 4 x 4
    """
    # 前 (-Z) が中心を向き, カメラの上が世界の上に近い
    forward = center - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = right, up, -forward
    pose[:3, 3] = position

    return pose


def render_view(
    shape: dict, pose: np.ndarray, intrinsics: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1 つの姿勢から形に光線を当て, 画像, 深度, 信頼度を求める関数

    Parameters
    ----------
    shape: dict
        形 (color も)
    pose: np.ndarray
        カメラから世界への 4 x 4 (OpenGL の決まり)
    intrinsics: dict
        fl_x, fl_y, cx, cy, w, h

    Returns
    ----------
    result: tuple[np.ndarray, np.ndarray, np.ndarray]
        (高さ, 幅, 3) の uint8 の画像, (高さ, 幅) のミリの uint16 の深度
        (カメラの前の軸に沿った距離 (z). 光線に沿った距離ではない.
        当たらない画素は 0), (高さ, 幅) の uint8 の信頼度 (当たれば 255)
    """
    # 画素の中心を通る光線 (カメラの前 -Z に 1 進むと深度が 1)
    width, height = int(intrinsics["w"]), int(intrinsics["h"])
    rows, columns = np.mgrid[0:height, 0:width]
    camera = np.stack([
        (columns + 0.5 - intrinsics["cx"]) / intrinsics["fl_x"],
        -(rows + 0.5 - intrinsics["cy"]) / intrinsics["fl_y"],
        -np.ones((height, width)),
    ], axis=-1).reshape(-1, 3)
    directions = camera @ pose[:3, :3].T
    t, normals = intersect(shape, pose[:3, 3], directions)
    hit = np.isfinite(t)

    # 光はカメラから. 明るさは法線と視線の角度
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    shade = 0.3 + 0.7 * np.clip(-np.sum(normals * unit, axis=1), 0, 1)
    color = np.asarray(shape.get("color", [200, 200, 200]), dtype=np.float64)
    image = np.where(hit[:, None], shade[:, None] * color, 0.0)
    depth = np.where(hit, np.round(np.where(hit, t, 0) * 1000), 0)
    result = (
        image.reshape(height, width, 3).astype(np.uint8),
        depth.reshape(height, width).astype(np.uint16),
        np.where(hit, 255, 0).reshape(height, width).astype(np.uint8),
    )

    return result


def random_shapes(count: int, rng: np.random.Generator) -> list[dict]:
    """
    球か箱を乱数でばらつかせて並べる関数

    Parameters
    ----------
    count: int
        形の数
    rng: np.random.Generator
        乱数

    Returns
    ----------
    shapes: list[dict]
        形の並び (名前は shape_000 から. 中心は原点から水平に離し, 高さも
        乱数)
    """
    # 種類, 中心 (方位と距離), 大きさ, 色
    shapes = []
    for index in range(count):
        angle = rng.uniform(0, 2 * math.pi)
        distance = rng.uniform(*RANDOM_DISTANCE)
        shape: dict = {
            "name": f"shape_{index:03d}",
            "type": str(rng.choice(["sphere", "box"])),
            "center": [
                distance * math.cos(angle),
                float(rng.uniform(*RANDOM_HEIGHT)),
                distance * math.sin(angle),
            ],
            "color": [
                int(value) for value in rng.integers(*RANDOM_COLOR, 3)
            ],
        }
        if shape["type"] == "sphere":
            shape["radius"] = float(rng.uniform(*RANDOM_RADIUS))
        else:
            shape["size"] = [
                float(value) for value in rng.uniform(*RANDOM_SIZE, 3)
            ]
        shapes.append(shape)

    return shapes


def write_shape_captures(
    root: Union[str, Path], shapes: Optional[list[dict]] = None,
    views: int = 8, size: tuple[int, int] = (24, 32), points: int = 2000,
    seed: int = 0, count: Optional[int] = None
) -> dict:
    """
    単純な形の正確な点群と, 決まった姿勢から撮った多視点の撮影を書く関数

    Parameters
    ----------
    root: Union[str, Path]
        書くディレクトリ (captures/<名前>/ と clouds/<名前>.ply を作る)
    shapes: Optional[list[dict]] = None
        形の並び (name, type (sphere, box), center, radius か size,
        color). 省略すると球 2 つと箱 2 つ (count を渡すと乱数の形)
    views: int = 8
        1 つの撮影の枚数 (物の周りに等間隔)
    size: tuple[int, int] = (24, 32)
        画像の (高さ, 幅)
    points: int = 2000
        点群の点の数
    seed: int = 0
        乱数の種 (点群の点と, count の形)
    count: Optional[int] = None
        乱数で作る形の数 (球か箱. 種類, 位置, 大きさ, 色を seed から)

    Returns
    ----------
    result: dict
        captures, clouds (ディレクトリ), shapes (形の並び)

    Raises
    ----------
    ValueError
        shapes と count を両方渡した場合と, count が 1 以上の整数でない
        場合
    """
    # 形の並び (渡したもの, 乱数, 既定のどれか)
    root = Path(root).expanduser()
    rng = np.random.default_rng(seed)
    if count is not None:
        if shapes is not None:
            raise ValueError("shapes と count は同時に渡せません")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"count は整数にしてください (今は {count})")
        if count < 1:
            raise ValueError(f"count は 1 以上にしてください (今は {count})")
        shapes = random_shapes(count, rng)
    shapes = [dict(shape) for shape in (shapes or DEFAULT_SHAPES)]

    # 内部パラメータ (横の画角はおよそ 60 度)
    height, width = size
    focal = width * 0.9
    intrinsics = {
        "fl_x": focal, "fl_y": focal, "cx": width / 2, "cy": height / 2,
        "w": width, "h": height,
    }
    (root / "clouds").mkdir(parents=True, exist_ok=True)

    # 形ごとに, 点群と周回の撮影
    for shape in shapes:
        cloud = sample_surface(shape, points, rng)
        write_ply(root / "clouds" / f"{shape['name']}.ply", {"points": cloud})
        place = root / "captures" / shape["name"]
        for folder in ("images", "raw_depth", "confidence"):
            (place / folder).mkdir(parents=True, exist_ok=True)
        center = np.asarray(shape["center"], dtype=np.float64)
        frames = []
        for index in range(views):
            angle = 2 * math.pi * index / views
            position = center + np.array([
                ORBIT_RADIUS * math.sin(angle), ORBIT_HEIGHT,
                ORBIT_RADIUS * math.cos(angle),
            ])
            pose = look_at(position, center)
            image, depth, confidence = render_view(shape, pose, intrinsics)
            write_png(place / "images" / f"{index:03d}.png", image)
            write_png(place / "raw_depth" / f"{index:03d}.png", depth)
            write_png(place / "confidence" / f"{index:03d}.png", confidence)
            frames.append({
                "file_path": f"images/{index:03d}.png",
                "raw_depth_file_path": f"raw_depth/{index:03d}.png",
                "confidence_file_path": f"confidence/{index:03d}.png",
                "transform_matrix": pose.tolist(),
                "tracking_state": "TRACKING",
            })
        transforms = {
            **intrinsics,
            "depth": {"depth_format": "png16_millimeter", "invalid_value": 0},
            "coordinate_convention": {
                "world_up": [0.0, 1.0, 0.0],
                "camera_axes": DEFAULT_AXES, **DEFAULT_DEPTH,
            },
            "capture": {"device_model": "synthetic"},
            "frames": frames,
        }
        (place / "transforms.json").write_text(
            json.dumps(transforms, indent=2), encoding="utf-8"
        )
    result = {
        "captures": root / "captures", "clouds": root / "clouds",
        "shapes": shapes,
    }

    return result
