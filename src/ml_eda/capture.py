import json
import math
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.loaders import Loader, register_loader

# 撮影の決まりのファイルの名前
TRANSFORMS = "transforms.json"

# 撮影を探すディレクトリの数の上限 (並べたディレクトリの直下だけ)
MAX_CAPTURES = 10_000

# 信頼できる深度とする信頼度の既定 (0 から 255)
MIN_CONFIDENCE = 128

# 信頼できる深度の画素が少ないとする割合
LOW_CONFIDENT = 0.3

# 周回の欠けとする方位の隙間 (度)
WIDE_GAP = 90.0

# 撮影の列の並び
COLUMNS = (
    "path", "frames", "tracked", "width", "height", "fl_x", "device",
    "depth_valid", "depth_confident", "distance_min", "distance_median",
    "distance_max", "view_gap", "points", "missing_files", "error",
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
        撮影のディレクトリ (名前の順. 無ければ空)
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


def sniff_capture(path: Path) -> float:
    """
    多視点の撮影として読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ (撮影のディレクトリか, それを並べたディレクトリなら
        高く, 違えば 0)
    """
    # 画像のディレクトリとしても読めるので, それより高く
    captures = find_captures(path)
    if not captures:
        score = 0.0
        return score
    try:
        transforms = read_transforms(captures[0])
    except (OSError, ValueError):
        score = 0.0
        return score
    score = 0.97 if transforms.get("frames") else 0.0

    return score


def read_transforms(capture: Path) -> dict:
    """
    撮影の transforms.json を読み, 形を確かめる関数

    Parameters
    ----------
    capture: Path
        撮影のディレクトリ

    Returns
    ----------
    transforms: dict
        transforms.json の中身

    Raises
    ----------
    ValueError
        JSON でない場合と, frames の並びが無い場合
    """
    # frames は辞書の並び
    with open(capture / TRANSFORMS, encoding="utf-8") as file:
        transforms = json.load(file)
    if not isinstance(transforms, dict) or not isinstance(
        transforms.get("frames"), list
    ):
        raise ValueError(f"{TRANSFORMS} に frames の並びがありません")
    if not all(isinstance(frame, dict) for frame in transforms["frames"]):
        raise ValueError(f"{TRANSFORMS} の frames の形が違います")

    return transforms


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
    # 4 x 4 の数の並び
    try:
        pose: Optional[np.ndarray] = np.asarray(
            frame["transform_matrix"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError):
        pose = None
        return pose
    if pose is None or pose.shape != (4, 4) or not np.isfinite(pose).all():
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
    # 座標の決まりの説明に +Z forward とあれば OpenCV
    axes = str(
        (transforms.get("coordinate_convention") or {}).get("camera_axes", "")
    ).replace(" ", "").lower()
    sign = 1.0 if "+zforward" in axes else -1.0

    return sign


def view_center(poses: list[np.ndarray], sign: float) -> np.ndarray:
    """
    各枚の視線に最も近い点 (撮った物の中心) を求める関数

    Parameters
    ----------
    poses: list[np.ndarray]
        カメラから世界への 4 x 4 の姿勢
    sign: float
        視線がカメラの Z 軸のどちら向きか

    Returns
    ----------
    center: np.ndarray
        世界の座標 (視線がほぼ平行で決まらなければ, カメラの位置の平均)
    """
    # 視線への距離の 2 乗の和が最小の点 (最小二乗)
    positions = np.array([pose[:3, 3] for pose in poses])
    system = np.zeros((3, 3))
    value = np.zeros(3)
    for pose in poses:
        direction = sign * pose[:3, 2]
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        projector = np.eye(3) - np.outer(direction, direction)
        system += projector
        value += projector @ pose[:3, 3]
    if np.linalg.cond(system) > 1e6:
        center = positions.mean(axis=0)
        return center
    center = np.linalg.solve(system, value)

    return center


def view_gap(
    positions: np.ndarray, center: np.ndarray, up: np.ndarray
) -> float:
    """
    中心から見たカメラの方位の, 最も大きい隙間 (度) を求める関数

    Parameters
    ----------
    positions: np.ndarray
        カメラの位置 (枚数, 3)
    center: np.ndarray
        撮った物の中心
    up: np.ndarray
        世界の上の向き

    Returns
    ----------
    gap: float
        最も大きい隙間 (0 から 360. 1 枚なら 360)
    """
    # 上の向きに垂直な面に写した方位の, 隣どうしの差の最大 (一周を含む)
    up = up / max(np.linalg.norm(up), 1e-12)
    offsets = positions - center
    flat = offsets - np.outer(offsets @ up, up)
    base = np.array([1.0, 0.0, 0.0])
    if abs(base @ up) > 0.9:
        base = np.array([0.0, 0.0, 1.0])
    first = base - (base @ up) * up
    first = first / np.linalg.norm(first)
    second = np.cross(up, first)
    angles = np.sort(np.degrees(np.arctan2(flat @ second, flat @ first)))
    if len(angles) < 2:
        gap = 360.0
        return gap
    steps = np.diff(np.concatenate([angles, [angles[0] + 360.0]]))
    gap = float(steps.max())

    return gap


def depth_stats(
    capture: Path, frames: list[dict], depth: dict, min_confidence: int
) -> dict:
    """
    各枚の深度と信頼度の画像から, 有効率と撮影距離を求める関数

    Parameters
    ----------
    capture: Path
        撮影のディレクトリ
    frames: list[dict]
        transforms.json の frames
    depth: dict
        transforms.json の depth (形式と欠けの値)
    min_confidence: int
        信頼できる深度とする信頼度 (0 から 255)

    Returns
    ----------
    stats: dict
        depth_valid, depth_confident, distance_min, distance_median,
        distance_max (深度の画像が無ければ空)
    """
    # 16 bit の深度 (ミリかメートル) と 8 bit の信頼度. 欠けは invalid_value.
    # 生の深度があればそれを使う (信頼度は生の深度と組. 穴を埋めた深度は
    # 欠けが無い)
    from PIL import Image

    scale = 1.0 if "meter" in str(depth.get("depth_format", "")) and (
        "milli" not in str(depth.get("depth_format", ""))
    ) else 0.001
    invalid = depth.get("invalid_value", 0)
    valid_rates, confident_rates, medians = [], [], []
    for frame in frames:
        name = frame.get("raw_depth_file_path")
        if not isinstance(name, str) or not (capture / name).is_file():
            name = frame.get("depth_file_path")
        if not isinstance(name, str) or not (capture / name).is_file():
            continue
        with Image.open(capture / name) as image:
            values = np.asarray(image, dtype=np.float64)
        valid = values != invalid
        valid_rates.append(float(valid.mean()))
        if valid.any():
            medians.append(float(np.median(values[valid])) * scale)
        name = frame.get("confidence_file_path")
        if isinstance(name, str) and (capture / name).is_file():
            with Image.open(capture / name) as image:
                confidence = np.asarray(image, dtype=np.float64)
            confident_rates.append(
                float(((confidence >= min_confidence) & valid).mean())
            )
    stats: dict = {}
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
    # ヘッダーの element vertex の行だけ
    count: Optional[int] = None
    try:
        with open(path, "rb") as file:
            for _ in range(200):
                line = file.readline().decode("ascii", "ignore").strip()
                if line.startswith("element vertex"):
                    count = int(line.split()[-1])
                if line == "end_header" or not line:
                    break
    except (OSError, ValueError):
        count = None

    return count


def describe_capture(
    capture: Path, min_confidence: int, with_depth: bool
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

    Returns
    ----------
    row: dict
        COLUMNS の値 (path は呼び出し側が入れる)
    """
    # 枚数, 追跡, 画像の大きさ, 内部パラメータ, 端末
    transforms = read_transforms(capture)
    frames = transforms["frames"]
    row: dict = {
        "frames": len(frames),
        "tracked": sum(
            frame.get("tracking_state", "TRACKING") == "TRACKING"
            for frame in frames
        ),
        "width": transforms.get("w"), "height": transforms.get("h"),
        "fl_x": transforms.get("fl_x"),
        "device": (transforms.get("capture") or {}).get("device_model"),
        "missing_files": sum(
            not (capture / str(frame.get("file_path", ""))).is_file()
            for frame in frames
        ),
    }

    # 物の中心から見た方位の隙間 (姿勢が 2 枚以上ある場合)
    poses = [pose for pose in map(pose_of, frames) if pose is not None]
    if len(poses) >= 2:
        convention = transforms.get("coordinate_convention") or {}
        up = np.asarray(convention.get("world_up") or [0.0, 1.0, 0.0],
                        dtype=np.float64)
        center = view_center(poses, forward_sign(transforms))
        row["view_gap"] = view_gap(
            np.array([pose[:3, 3] for pose in poses]), center, up
        )

    # 深度と点群
    if with_depth:
        row.update(depth_stats(
            capture, frames, transforms.get("depth") or {}, min_confidence
        ))
    cloud = (transforms.get("point_cloud") or {}).get("file_path")
    if isinstance(cloud, str) and (capture / cloud).is_file():
        row["points"] = ply_points(capture / cloud)

    return row


def capture_item(root: Path, frame: pd.DataFrame, index: int) -> dict:
    """
    1 回の撮影の中身 (各枚の場所と姿勢) を読む関数

    Parameters
    ----------
    root: Path
        読み込んだ場所 (撮影の path はここからの相対)
    frame: pd.DataFrame
        撮影の表
    index: int
        行の位置

    Returns
    ----------
    item: dict
        intrinsics (fl_x, fl_y, cx, cy, w, h), frames (image, depth,
        confidence の場所, pose (4 x 4), tracking), point_cloud (場所か
        None), convention (座標の決まり), path (撮影のディレクトリ)
    """
    # 場所は絶対パスに
    capture = (root / str(frame["path"].iloc[index])).resolve()
    transforms = read_transforms(capture)
    frames = []
    for entry in transforms["frames"]:
        frames.append({
            "image": str(capture / str(entry.get("file_path", ""))),
            "depth": str(capture / entry["depth_file_path"])
            if isinstance(entry.get("depth_file_path"), str) else None,
            "confidence": str(capture / entry["confidence_file_path"])
            if isinstance(entry.get("confidence_file_path"), str) else None,
            "pose": pose_of(entry),
            "tracking": entry.get("tracking_state"),
        })
    cloud = (transforms.get("point_cloud") or {}).get("file_path")
    item = {
        "path": str(capture),
        "intrinsics": {key: transforms.get(key)
                       for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")},
        "frames": frames,
        "point_cloud": str(capture / cloud)
        if isinstance(cloud, str) else None,
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
        min_confidence (信頼できる深度とする信頼度. 既定 128)

    Returns
    ----------
    data: Data
        1 行が 1 回の撮影のデータ. 各枚は load_item で読む

    Raises
    ----------
    ValueError
        撮影が無い場合と, min_confidence が 0 から 255 の整数でない場合
    """
    # 撮影の一覧と, 深度を読めるか (Pillow)
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

    # 撮影ごとの 1 行 (読めない撮影は理由を残す)
    root = path.parent if (path / TRANSFORMS).is_file() else path
    rows = []
    for capture in captures:
        row: dict = {"path": capture.relative_to(root).as_posix()}
        try:
            row.update(describe_capture(capture, min_confidence, with_depth))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["error"] = str(error) or type(error).__name__
        rows.append(row)
    frame = pd.DataFrame(rows).reindex(columns=list(COLUMNS))
    data = Data(
        kind="capture", frame=frame, roles={"path": "path"},
        item_loader=partial(capture_item, root, frame)
    )
    data.source = {"root": str(root), "min_confidence": min_confidence}
    if not with_depth:
        data.source["notes"] = [
            "Pillow が無いので深度の画像を読んでいません (深度の列は欠損)"
        ]

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
        frames (枚数の最少, 中央値, 最多), depth (深度の有効率と信頼で
        きる割合の中央値), distance (撮影距離の範囲), unreadable,
        findings
    """
    # 枚数, 深度, 距離の集計
    frame = data.frame
    readable = frame[frame["error"].isna()]
    result: dict = {
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
    if result["unreadable"]:
        findings.append({
            "severity": "warning", "code": "unreadable_items",
            "message": (
                f"読めない撮影が {result['unreadable']} 件あります (error の"
                "列に理由)"
            ),
            "columns": ["error"],
        })

    # 追跡できなかった枚, 欠けたファイル
    lost = readable[readable["tracked"] < readable["frames"]]
    if not lost.empty:
        findings.append({
            "severity": "warning", "code": "tracking_lost",
            "message": (
                f"{len(lost)} 件の撮影に, 追跡できなかった枚があります (計 "
                f"{int((lost['frames'] - lost['tracked']).sum())} 枚). 姿勢が"
                "正しくないので, その枚は使わないでください"
            ),
            "columns": ["tracked"],
        })
    missing = readable[readable["missing_files"].fillna(0) > 0]
    if not missing.empty:
        findings.append({
            "severity": "warning", "code": "unreadable_items",
            "message": (
                f"{len(missing)} 件の撮影で, transforms.json にある画像の"
                f"ファイルが無い枚があります (計 "
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
        {"min_confidence": "信頼できる深度とする信頼度 (0 から 255. 既定 128)"},
        "多視点の撮影 (transforms.json を持つディレクトリか, それを並べた"
        "ディレクトリ. 深度を読むには Pillow が要る)"
    ))


register_capture()
