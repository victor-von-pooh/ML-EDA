import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.loaders import (
    FOLLOW_LINKS_HELP, OPTION_HINT, Loader, check_inside, option_flag,
    points_outside, register_loader
)

# 撮影の決まりのファイルの名前
TRANSFORMS = "transforms.json"

# 撮影を探すディレクトリの数の上限 (並べたディレクトリの直下だけ)
MAX_CAPTURES = 10_000

# 読めるかを確かめる撮影の数 (先頭から)
SNIFF_CAPTURES = 3

# 信頼できる深度とする信頼度の既定 (0 から 255)
MIN_CONFIDENCE = 128

# 信頼できる深度の画素が少ないとする割合
LOW_CONFIDENT = 0.3

# 周回の欠けとする方位の隙間 (度)
WIDE_GAP = 90.0

# 物の周りを回ったとみなす, 中心からカメラまでの水平の距離の下限
# (深度の中央値に対する割合. 深度が無ければメートル)
ORBIT_RATIO = 0.1
ORBIT_METERS = 0.05

# 撮影の列の並び
COLUMNS = (
    "path", "frames", "tracked", "width", "height", "fl_x", "device",
    "depth_source", "depth_valid", "depth_confident", "distance_min",
    "distance_median", "distance_max", "view_gap", "points",
    "missing_files", "error",
)


def find_captures(path: Path) -> list[Path]:
    """
    撮影のディレクトリ (transforms.json を持つ) を探す関数

    Parameters
    ----------
    path: Path
        撮影のディレクトリか, それを並べたディレクトリ

    Returns
    ----------
    captures: list[Path]
        撮影のディレクトリ (名前の順. 上限まで. 無ければ空)
    """
    # 自身が撮影ならそれだけ, 違えば直下のディレクトリ
    if not path.is_dir():
        captures: list[Path] = []
        return captures
    if (path / TRANSFORMS).is_file():
        captures = [path]
        return captures
    captures = sorted(
        child for child in path.iterdir()
        if child.is_dir() and (child / TRANSFORMS).is_file()
    )[:MAX_CAPTURES]

    return captures


def sniff_capture(path: Path, follow_links: Any = False) -> float:
    """
    多視点の撮影として読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所
    follow_links: Any = False
        読み込みの設定 follow_links (真なら外を指すリンクの先の
        transforms.json も読む. 画面からの文字も読む)

    Returns
    ----------
    score: float
        確からしさ (先頭の撮影のどれかが読めれば, 画像のディレクトリより
        高く. 無ければ 0)
    """
    # 先頭の数件のどれかが frames を持てば撮影
    score = 0.0
    base = None if option_flag(
        {"follow_links": follow_links}, "follow_links", False
    ) else path
    for capture in find_captures(path)[:SNIFF_CAPTURES]:
        try:
            transforms = read_transforms(capture, base)
        except (OSError, ValueError):
            continue
        if transforms["frames"]:
            score = 0.97
            return score

    return score


def read_transforms(
    capture: Path, base: Optional[Path] = None, hint: str = OPTION_HINT
) -> dict:
    """
    撮影の transforms.json を読み, 形を確かめる関数

    Parameters
    ----------
    capture: Path
        撮影のディレクトリ
    base: Optional[Path] = None
        読み込んだディレクトリ (リンクをたどった後にこの外を指す
        transforms.json は読まない. None ならリンクの先をたどる)
    hint: str = OPTION_HINT
        外を指す場合の, 読む案内

    Returns
    ----------
    transforms: dict
        transforms.json の中身

    Raises
    ----------
    ValueError
        JSON でない場合, frames の並びが無い場合と, リンクをたどった後に
        base の外を指す場合
    """
    # 外を指すリンクでないことを確かめてから読む. frames は辞書の並び
    check_inside(
        base, capture / TRANSFORMS, f"{capture.name}/{TRANSFORMS}", hint
    )
    with open(capture / TRANSFORMS, encoding="utf-8") as file:
        transforms = json.load(file)
    if not isinstance(transforms, dict) or not isinstance(
        transforms.get("frames"), list
    ):
        raise ValueError(f"{TRANSFORMS} に frames の並びがありません")
    if not all(isinstance(frame, dict) for frame in transforms["frames"]):
        raise ValueError(f"{TRANSFORMS} の frames の形が違います")

    return transforms


def inside(
    capture: Path, name: Any, base: Optional[Path] = None
) -> Optional[Path]:
    """
    transforms.json に書かれた場所を, 撮影のディレクトリの中に限って返す
    関数

    Parameters
    ----------
    capture: Path
        撮影のディレクトリ
    name: Any
        書かれた場所 (撮影のディレクトリからの相対)
    base: Optional[Path] = None
        読み込んだディレクトリ (リンクをたどった後にこの外を指すファイルは
        無いものとする. None ならリンクの先をたどる)

    Returns
    ----------
    path: Optional[Path]
        ファイルの場所 (文字でない, 撮影のディレクトリの外 (絶対パスや
        ..), リンクをたどった後に base の外, 無い場合は None)
    """
    # 書かれた場所の文字の上で中にあり, リンクをたどった後も読み込んだ
    # ディレクトリの中にあるファイルだけ
    path: Optional[Path] = None
    if not isinstance(name, str) or not name or os.path.isabs(name):
        return path
    normal = os.path.normpath(name)
    if normal == ".." or normal.startswith(".." + os.sep):
        return path
    try:
        found = (capture / normal).is_file() and not (
            base is not None and points_outside(base, capture / normal)
        )
        if found:
            path = (capture / normal).absolute()
    except (OSError, ValueError):
        path = None

    return path


def is_tracked(frame: dict) -> bool:
    """
    1 枚が追跡できていたかを返す関数

    Parameters
    ----------
    frame: dict
        transforms.json の frames の 1 つ

    Returns
    ----------
    tracked: bool
        tracking_state が TRACKING か, 書かれていないか
    """
    # 書かれていない枚は追跡できたとする
    tracked = frame.get("tracking_state", "TRACKING") == "TRACKING"

    return tracked


def pose_of(frame: dict) -> Optional[np.ndarray]:
    """
    1 枚のカメラの姿勢 (カメラから世界への 4 x 4) を返す関数

    Parameters
    ----------
    frame: dict
        transforms.json の frames の 1 つ

    Returns
    ----------
    pose: Optional[np.ndarray]
        4 x 4 の行列 (無いか形が違えば None)
    """
    # 4 x 4 の有限の数の並び
    try:
        pose: Optional[np.ndarray] = np.asarray(
            frame["transform_matrix"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError):
        pose = None
        return pose
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        pose = None

    return pose


def forward_sign(transforms: dict) -> float:
    """
    カメラの視線がカメラの Z 軸のどちら向きかを返す関数

    Parameters
    ----------
    transforms: dict
        transforms.json の中身

    Returns
    ----------
    sign: float
        -1 (OpenGL の -Z が前. 既定) か 1 (OpenCV の +Z が前)
    """
    # 座標の決まりの説明に OpenCV か +Z forward とあれば +Z
    axes = str(
        (transforms.get("coordinate_convention") or {}).get("camera_axes", "")
    ).replace(" ", "").lower()
    sign = 1.0 if "opencv" in axes or "+zforward" in axes else -1.0

    return sign


def up_of(
    transforms: dict, poses: list[np.ndarray], sign: float
) -> np.ndarray:
    """
    世界の上の向きを返す関数

    Parameters
    ----------
    transforms: dict
        transforms.json の中身
    poses: list[np.ndarray]
        カメラから世界への 4 x 4 の姿勢
    sign: float
        視線がカメラの Z 軸のどちら向きか (OpenCV はカメラの上が -Y)

    Returns
    ----------
    up: np.ndarray
        長さ 1 の向き (world_up が無いか形が違えば, 各カメラの右の軸に
        最も垂直な向き. 符号はカメラの上の軸の平均に合わせる)
    """
    # 書かれた world_up. 無ければ, 傾けずに撮ればカメラの右の軸は水平
    # なので, 右の軸を並べた行列の最小の特異ベクトル (見下ろして撮っても
    # 傾かない)
    written = (transforms.get("coordinate_convention") or {}).get("world_up")
    try:
        up = np.asarray(written, dtype=np.float64)
        if up.shape != (3,) or not np.isfinite(up).all():
            raise ValueError("world_up の形が違います")
    except (TypeError, ValueError):
        rights = np.array([pose[:3, 0] for pose in poses])
        up = np.linalg.svd(rights)[2][-1]
        if float(up @ np.mean([pose[:3, 1] for pose in poses], axis=0)) * (
            -sign
        ) < 0:
            up = -up
    norm = float(np.linalg.norm(up))
    up = up / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])

    return up


def view_center(poses: list[np.ndarray]) -> Optional[np.ndarray]:
    """
    各枚の視線 (の直線) に最も近い点 (撮った物の中心) を求める関数

    視線の直線への距離は向きの符号に依らないので, 前か後ろかは使わない
    (前にあるかは orbit_of で確かめる)

    Parameters
    ----------
    poses: list[np.ndarray]
        カメラから世界への 4 x 4 の姿勢

    Returns
    ----------
    center: Optional[np.ndarray]
        世界の座標 (視線がほぼ平行で決まらなければ None)
    """
    # 視線への距離の 2 乗の和が最小の点 (最小二乗)
    system = np.zeros((3, 3))
    value = np.zeros(3)
    for pose in poses:
        axis = pose[:3, 2] / max(float(np.linalg.norm(pose[:3, 2])), 1e-12)
        projector = np.eye(3) - np.outer(axis, axis)
        system += projector
        value += projector @ pose[:3, 3]
    center: Optional[np.ndarray] = None
    if np.linalg.cond(system) < 1e6:
        center = np.linalg.solve(system, value)

    return center


def orbit_of(
    poses: list[np.ndarray], up: np.ndarray, sign: float,
    distance: Optional[float]
) -> Optional[dict]:
    """
    物の中心と, 上から見た (上の向きに垂直な面に写した) カメラの位置を
    求める関数

    Parameters
    ----------
    poses: list[np.ndarray]
        カメラから世界への 4 x 4 の姿勢
    up: np.ndarray
        世界の上の向き
    sign: float
        視線がカメラの Z 軸のどちら向きか
    distance: Optional[float]
        撮影距離 (深度の中央値. 無ければ None)

    Returns
    ----------
    orbit: Optional[dict]
        center (世界の座標), axes (面の 2 つの向き), flat (中心からの面の
        上の位置 (枚数, 2). メートル). 物の周りを回った撮影でない (中心が
        決まらない, 中心が半分以上のカメラの後ろ, カメラが中心からほぼ
        動いていない) 場合は None
    """
    # 中心が決まり, 半分以上のカメラの前にあること
    orbit: Optional[dict] = None
    center = view_center(poses)
    if center is None:
        return orbit
    ahead = [
        float((center - pose[:3, 3]) @ (sign * pose[:3, 2])) > 0
        for pose in poses
    ]
    if sum(ahead) * 2 < len(poses):
        return orbit

    # 上の向きに垂直な面の 2 つの向きと, 写した位置
    base = np.array([1.0, 0.0, 0.0])
    if abs(float(base @ up)) > 0.9:
        base = np.array([0.0, 0.0, 1.0])
    first = base - (base @ up) * up
    first = first / np.linalg.norm(first)
    second = np.cross(up, first)
    offsets = np.array([pose[:3, 3] for pose in poses]) - center
    flat = np.stack([offsets @ first, offsets @ second], axis=1)

    # 中心から十分に離れていること
    reach = float(np.median(np.linalg.norm(flat, axis=1)))
    limit = ORBIT_RATIO * distance if distance else ORBIT_METERS
    if reach < limit:
        return orbit
    orbit = {"center": center, "axes": (first, second), "flat": flat}

    return orbit


def widest_gap(flat: np.ndarray) -> tuple[float, float]:
    """
    上から見たカメラの方位の, 最も大きい隙間とその始まりを求める関数

    Parameters
    ----------
    flat: np.ndarray
        中心からの面の上の位置 (枚数, 2)

    Returns
    ----------
    gap: tuple[float, float]
        隙間 (度) と, 隙間の始まりの方位 (度. -180 から 180)
    """
    # 方位の隣どうしの差の最大 (一周を含む)
    angles = np.sort(np.degrees(np.arctan2(flat[:, 1], flat[:, 0])))
    steps = np.diff(np.concatenate([angles, [angles[0] + 360.0]]))
    widest = int(np.argmax(steps))
    gap = (float(steps[widest]), float(angles[widest]))

    return gap


def view_gap_of(
    poses: list[np.ndarray], up: np.ndarray, sign: float,
    distance: Optional[float]
) -> float:
    """
    中心から見たカメラの方位の, 最も大きい隙間 (度) を求める関数

    Parameters
    ----------
    poses: list[np.ndarray]
        追跡できた枚のカメラから世界への 4 x 4 の姿勢
    up: np.ndarray
        世界の上の向き
    sign: float
        視線がカメラの Z 軸のどちら向きか
    distance: Optional[float]
        撮影距離 (深度の中央値. 無ければ None)

    Returns
    ----------
    gap: float
        最も大きい隙間 (0 から 360). 物の周りを回った撮影でない場合は NaN
    """
    # 回った撮影の, 上から見た位置の方位
    orbit = orbit_of(poses, up, sign, distance)
    gap = math.nan if orbit is None else widest_gap(orbit["flat"])[0]

    return gap


def capture_layout(item: dict, distance: Optional[float] = None) -> dict:
    """
    1 回の撮影のカメラを, 物の中心から上から見た配置にする関数 (画面で
    周回の欠けを描くため)

    Parameters
    ----------
    item: dict
        load_item で読んだ撮影の中身
    distance: Optional[float] = None
        撮影距離 (表の distance_median. 回った撮影かの判定に使う)

    Returns
    ----------
    layout: dict
        orbit (回った撮影か), cameras (各枚の frame (番号), x, y (中心
        からの面の上の位置. メートル. 回った撮影でないか, 姿勢の無い枚は
        None), tracked), gap (最も大きい隙間 (度) か None), gap_from
        (隙間の始まりの方位 (度) か None). 方位は上から見て反時計回り
        (x 軸から y 軸へ. y は上向きの座標で描く), 隙間は gap_from から
        同じ向きに gap 度
    """
    # 追跡できた枚で決めた中心と面と, 全ての枚の面の上の位置
    convention = {"coordinate_convention": item.get("convention") or {}}
    sign = forward_sign(convention)
    tracked = [
        frame["pose"] for frame in item["frames"]
        if frame.get("pose") is not None and frame.get("tracking") == (
            "TRACKING"
        )
    ]
    orbit = None
    if len(tracked) >= 2:
        orbit = orbit_of(
            tracked, up_of(convention, tracked, sign), sign, distance
        )
    cameras = []
    for index, frame in enumerate(item["frames"]):
        place = None
        if orbit is not None and frame.get("pose") is not None:
            offset = frame["pose"][:3, 3] - orbit["center"]
            place = [float(offset @ axis) for axis in orbit["axes"]]
        cameras.append({
            "frame": index,
            "x": place[0] if place else None,
            "y": place[1] if place else None,
            "tracked": frame.get("tracking") == "TRACKING",
        })
    gap = widest_gap(orbit["flat"]) if orbit is not None else None
    layout = {
        "orbit": orbit is not None, "cameras": cameras,
        "gap": gap[0] if gap else None, "gap_from": gap[1] if gap else None,
    }

    return layout


def depth_scale(depth: dict) -> float:
    """
    深度の画像の値をメートルにする倍率を返す関数

    Parameters
    ----------
    depth: dict
        transforms.json の depth

    Returns
    ----------
    scale: float
        millimeter なら 0.001, centimeter なら 0.01, meter なら 1, 他は
        0.001
    """
    # 大文字と小文字は区別しない. 分からなければミリ
    form = str(depth.get("depth_format", "")).lower()
    scale = 0.001
    if "centi" in form:
        scale = 0.01
    elif "meter" in form and "milli" not in form:
        scale = 1.0

    return scale


def depth_stats(
    capture: Path, frames: list[dict], depth: dict, min_confidence: int,
    base: Optional[Path] = None
) -> dict:
    """
    各枚の深度と信頼度の画像から, 有効率と撮影距離を求める関数

    Parameters
    ----------
    capture: Path
        撮影のディレクトリ
    frames: list[dict]
        追跡できた枚の transforms.json の frames
    depth: dict
        transforms.json の depth (形式と欠けの値)
    min_confidence: int
        信頼できる深度とする信頼度 (0 から 255)
    base: Optional[Path] = None
        読み込んだディレクトリ (None ならリンクの先をたどる)

    Returns
    ----------
    stats: dict
        depth_source (raw, smoothed, mixed), depth_valid,
        depth_confident, distance_min, distance_median, distance_max
        (深度の画像が無ければ空)
    """
    # 生の深度 (信頼度と組) を優先. 穴を埋めた深度には欠けがほぼ無い
    from PIL import Image

    scale = depth_scale(depth)
    invalid = depth.get("invalid_value")
    invalid = invalid if isinstance(invalid, (int, float)) else 0
    valid_rates, confident_rates, medians, sources = [], [], [], set()
    for frame in frames:
        path = inside(capture, frame.get("raw_depth_file_path"), base)
        source = "raw"
        if path is None:
            path = inside(capture, frame.get("depth_file_path"), base)
            source = "smoothed"
        if path is None:
            continue
        sources.add(source)
        with Image.open(path) as image:
            values = np.asarray(image, dtype=np.float64)
        valid = values != invalid
        valid_rates.append(float(valid.mean()))
        if valid.any():
            medians.append(float(np.median(values[valid])) * scale)

        # 信頼度 (形が合わなければその枚は数えない)
        path = inside(capture, frame.get("confidence_file_path"), base)
        if path is None:
            continue
        with Image.open(path) as image:
            confidence = np.asarray(image, dtype=np.float64)
        if confidence.shape == valid.shape:
            confident_rates.append(
                float(((confidence >= min_confidence) & valid).mean())
            )
    stats: dict = {}
    if sources:
        stats["depth_source"] = sources.pop() if len(sources) == 1 else (
            "mixed"
        )
    if valid_rates:
        stats["depth_valid"] = float(np.mean(valid_rates))
    if confident_rates:
        stats["depth_confident"] = float(np.mean(confident_rates))
    if medians:
        stats.update(
            distance_min=float(np.min(medians)),
            distance_median=float(np.median(medians)),
            distance_max=float(np.max(medians)),
        )

    return stats


def ply_points(path: Path) -> Optional[int]:
    """
    PLY のヘッダーから点の数を読む関数

    Parameters
    ----------
    path: Path
        PLY のファイル

    Returns
    ----------
    count: Optional[int]
        点の数 (読めなければ None)
    """
    # ヘッダーの element vertex の行だけ (1 行は長さを限って読む)
    count: Optional[int] = None
    try:
        with open(path, "rb") as file:
            for _ in range(200):
                line = file.readline(1024).decode("ascii", "ignore").strip()
                if line.startswith("element vertex"):
                    count = int(line.split()[-1])
                if line == "end_header" or not line:
                    break
    except (OSError, ValueError):
        count = None

    return count


def describe_capture(
    capture: Path, min_confidence: int, with_depth: bool,
    base: Optional[Path] = None
) -> dict:
    """
    1 回の撮影を 1 行の値にまとめる関数

    Parameters
    ----------
    capture: Path
        撮影のディレクトリ
    min_confidence: int
        信頼できる深度とする信頼度
    with_depth: bool
        深度の画像を読むか (Pillow が無ければ読まない)
    base: Optional[Path] = None
        読み込んだディレクトリ (None ならリンクの先をたどる)

    Returns
    ----------
    row: dict
        COLUMNS の値 (path は呼び出し側が入れる). 深度, 距離, 方位は
        追跡できた枚だけで求める
    """
    # 枚数, 追跡, 画像の大きさ, 内部パラメータ, 端末, 無い画像
    transforms = read_transforms(capture, base)
    frames = transforms["frames"]
    tracked = [frame for frame in frames if is_tracked(frame)]
    row: dict = {
        "frames": len(frames), "tracked": len(tracked),
        "width": transforms.get("w"), "height": transforms.get("h"),
        "fl_x": transforms.get("fl_x"),
        "device": (transforms.get("capture") or {}).get("device_model"),
        "missing_files": sum(
            inside(capture, frame.get("file_path"), base) is None
            for frame in frames
        ),
    }

    # 深度と距離
    if with_depth:
        row.update(depth_stats(
            capture, tracked, transforms.get("depth") or {}, min_confidence,
            base
        ))

    # 物の中心から見た方位の隙間 (姿勢が 2 枚以上ある場合)
    poses = [pose for pose in map(pose_of, tracked) if pose is not None]
    if len(poses) >= 2:
        sign = forward_sign(transforms)
        row["view_gap"] = view_gap_of(
            poses, up_of(transforms, poses, sign), sign,
            row.get("distance_median")
        )

    # 点群
    cloud = inside(
        capture, (transforms.get("point_cloud") or {}).get("file_path"),
        base
    )
    if cloud is not None:
        row["points"] = ply_points(cloud)

    return row


def capture_item(
    root: Path, base: Optional[Path], frame: pd.DataFrame, index: int
) -> dict:
    """
    1 回の撮影の中身 (各枚の場所と姿勢) を読む関数

    Parameters
    ----------
    root: Path
        読み込んだ場所 (撮影の path はここからの相対)
    base: Optional[Path]
        読み込んだディレクトリ (None ならリンクの先をたどる)
    frame: pd.DataFrame
        撮影の表
    index: int
        行の位置

    Returns
    ----------
    item: dict
        path (撮影のディレクトリ), intrinsics (fl_x, fl_y, cx, cy, w, h),
        frames (image, raw_depth, depth, confidence の場所 (撮影の
        ディレクトリの中で, ある場合だけ. 無ければ None), pose (4 x 4 か
        None), tracking), point_cloud (場所か None), convention (座標の
        決まり)
    """
    # 場所は撮影のディレクトリの中に限った絶対パス
    name = str(frame["path"].iloc[index])
    capture = check_inside(base, root / name, name).resolve()
    transforms = read_transforms(capture, base)
    frames = []
    for entry in transforms["frames"]:
        places = {
            key: inside(capture, entry.get(field), base)
            for key, field in (
                ("image", "file_path"), ("raw_depth", "raw_depth_file_path"),
                ("depth", "depth_file_path"),
                ("confidence", "confidence_file_path"),
            )
        }
        frames.append({
            **{key: str(value) if value else None
               for key, value in places.items()},
            "pose": pose_of(entry),
            "tracking": entry.get("tracking_state", "TRACKING"),
        })
    cloud = inside(
        capture, (transforms.get("point_cloud") or {}).get("file_path"),
        base
    )
    item = {
        "path": str(capture),
        "intrinsics": {key: transforms.get(key)
                       for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")},
        "frames": frames,
        "point_cloud": str(cloud) if cloud else None,
        "convention": transforms.get("coordinate_convention") or {},
    }

    return item


def load_captures(path: Path, **options: Any) -> Data:
    """
    多視点の撮影を読み, 1 行が 1 回の撮影の表にする関数

    Parameters
    ----------
    path: Path
        撮影のディレクトリか, それを並べたディレクトリ
    **options: Any
        min_confidence (信頼できる深度とする信頼度. 既定 128),
        follow_links (既定 False. True なら外を指すリンクの先も読む)

    Returns
    ----------
    data: Data
        1 行が 1 回の撮影のデータ. 各枚は load_item で読む

    Raises
    ----------
    ValueError
        撮影が無い場合と, min_confidence が 0 から 255 の整数でない場合
    """
    # 撮影の一覧と, 深度を読めるか (Pillow). 根は絶対パス
    path = path.absolute()
    captures = find_captures(path)
    if not captures:
        raise ValueError(f"{path.name} に {TRANSFORMS} を持つ撮影がありません")
    min_confidence = options.get("min_confidence", MIN_CONFIDENCE)
    if isinstance(min_confidence, bool) or not isinstance(
        min_confidence, int
    ) or not 0 <= min_confidence <= 255:
        raise ValueError("min_confidence は 0 から 255 の整数にしてください")
    try:
        import PIL  # noqa: F401

        with_depth = True
    except ImportError:
        with_depth = False

    # 撮影ごとの 1 行 (読めない撮影と外を指すリンクは理由を残す). 外かは
    # 渡したディレクトリ (1 回の撮影ならその撮影) と比べる
    root = path.parent if (path / TRANSFORMS).is_file() else path
    base = None if option_flag(options, "follow_links", False) else path
    rows = []
    for capture in captures:
        row: dict = {"path": capture.relative_to(root).as_posix()}
        try:
            check_inside(base, capture, row["path"])
            row.update(describe_capture(
                capture, min_confidence, with_depth, base
            ))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["error"] = str(error) or type(error).__name__
        rows.append(row)
    frame = pd.DataFrame(rows).reindex(columns=list(COLUMNS))
    data = Data(
        kind="capture", frame=frame, roles={"path": "path"},
        item_loader=partial(capture_item, root, base, frame)
    )
    data.source = {
        "root": str(root), "min_confidence": min_confidence,
        "depth_read": with_depth, "capped": len(captures) >= MAX_CAPTURES,
    }

    return data


def describe_captures(data: Data) -> dict:
    """
    多視点の撮影の集計と気付きを求める関数

    Parameters
    ----------
    data: Data
        多視点の撮影のデータ

    Returns
    ----------
    result: dict
        kind, count, unreadable, frames (枚数の最少, 中央値, 最多),
        depth (深度の有効率と信頼できる割合の中央値), distance (撮影距離の
        範囲), findings
    """
    # 枚数, 深度, 距離の集計
    frame = data.frame
    readable = frame[frame["error"].isna()]
    result: dict = {
        "kind": data.kind, "count": int(len(frame)),
        "unreadable": int(frame["error"].notna().sum()),
        "frames": None if readable.empty else {
            "min": int(readable["frames"].min()),
            "median": float(readable["frames"].median()),
            "max": int(readable["frames"].max()),
        },
        "depth": {
            "valid": none_or(readable["depth_valid"].median()),
            "confident": none_or(readable["depth_confident"].median()),
        },
        "distance": {
            "min": none_or(readable["distance_min"].min()),
            "max": none_or(readable["distance_max"].max()),
        },
    }
    findings = []
    if data.source.get("capped"):
        findings.append({
            "severity": "warning", "code": "too_many_files",
            "message": (
                f"撮影が多いので, 名前の順で先頭の {MAX_CAPTURES} 件だけを"
                "読みました"
            ),
            "columns": [],
        })
    if data.source.get("depth_read") is False:
        findings.append({
            "severity": "info", "code": "missing_reader",
            "message": (
                "Pillow が無いので深度の画像を読んでいません (深度の列は"
                "欠損)"
            ),
            "columns": [],
        })
    if result["unreadable"]:
        findings.append({
            "severity": "warning", "code": "unreadable_items",
            "message": (
                f"読めない撮影が {result['unreadable']} 件あります (error の"
                "列に理由)"
            ),
            "columns": ["error"],
        })

    # 追跡できなかった枚, 無い画像
    lost = readable[readable["tracked"] < readable["frames"]]
    if not lost.empty:
        findings.append({
            "severity": "warning", "code": "tracking_lost",
            "message": (
                f"{len(lost)} 件の撮影に, 追跡できなかった枚があります (計 "
                f"{int((lost['frames'] - lost['tracked']).sum())} 枚). 姿勢が"
                "正しくないので, その枚は深度と方位の集計から除きました"
            ),
            "columns": ["tracked"],
        })
    missing = readable[readable["missing_files"].fillna(0) > 0]
    if not missing.empty:
        findings.append({
            "severity": "warning", "code": "missing_files",
            "message": (
                f"{len(missing)} 件の撮影で, transforms.json にある画像が"
                "無い (か撮影のディレクトリの外を指す) 枚があります (計 "
                f"{int(missing['missing_files'].sum())} 枚)"
            ),
            "columns": ["missing_files"],
        })

    # 深度の信頼度, 周回の欠け, 画像の大きさ
    low = readable[readable["depth_confident"] < LOW_CONFIDENT]
    if not low.empty:
        findings.append({
            "severity": "warning", "code": "low_depth_confidence",
            "message": (
                f"{len(low)} 件の撮影で, 信頼できる深度の画素が 3 割未満です "
                f"(最も低い {float(low['depth_confident'].min()):.1%}). 深度の"
                "画像の 1 画素より細い物や, 遠い物は信頼度が落ちます. 深度"
                "から形を作るなら, 近づいて撮るか, 使える撮影の下限を決めて"
                "ください"
            ),
            "columns": ["depth_confident"],
        })
    wide = readable[readable["view_gap"] >= WIDE_GAP]
    if not wide.empty:
        findings.append({
            "severity": "info", "code": "view_gap",
            "message": (
                f"{len(wide)} 件の撮影で, 物の周りを回った方位に "
                f"{WIDE_GAP:.0f} 度以上の隙間があります (最大 "
                f"{float(wide['view_gap'].max()):.0f} 度). その向きから"
                "見た形は, 撮った画像からは分かりません"
            ),
            "columns": ["view_gap"],
        })
    still = readable[readable["view_gap"].isna() & (readable["tracked"] >= 2)]
    if not still.empty:
        findings.append({
            "severity": "info", "code": "no_orbit",
            "message": (
                f"{len(still)} 件の撮影は, 物の周りを回った撮影と見なせない"
                "ので, 方位の隙間を求めていません (視線がほぼ平行, 物の中心"
                "が多くのカメラの後ろ, その場で向きだけを変えた, のどれか)"
            ),
            "columns": ["view_gap"],
        })
    if readable[["width", "height"]].dropna().drop_duplicates().shape[0] > 1:
        findings.append({
            "severity": "info", "code": "mixed_sizes",
            "message": "撮影ごとに画像の大きさが違います",
            "columns": ["width", "height"],
        })
    result["findings"] = findings

    return result


def none_or(value: Any) -> Optional[float]:
    """
    欠損なら None, それ以外は float にする関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    number: Optional[float]
        値
    """
    # NaN と None は None
    number = None if value is None or (
        isinstance(value, float) and math.isnan(value)
    ) else float(value)

    return number


def register_capture() -> None:
    """
    多視点の撮影の読み込みを登録簿に加える関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ディレクトリだけを読むので拡張子は無し
    register_loader(Loader(
        "capture", "capture", (), sniff_capture, load_captures,
        {"min_confidence": "信頼できる深度とする信頼度 (0 から 255. 既定 128)",
         "follow_links": FOLLOW_LINKS_HELP},
        "多視点の撮影 (transforms.json を持つディレクトリか, それを並べた"
        "ディレクトリ. 深度を読むには Pillow が要る)",
        sniff_options=("follow_links",)
    ))


register_capture()
