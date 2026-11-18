import json
import math
from pathlib import Path

import numpy as np
import pytest

from ml_eda import detect, load_data, prepare, profile

Image = pytest.importorskip("PIL.Image")


def look_at(
    position: np.ndarray, center: np.ndarray, up: np.ndarray
) -> list:
    """
    位置から中心を向くカメラの姿勢 (OpenGL の -Z が前, カメラから世界) を
    作る関数

    Parameters
    ----------
    position: np.ndarray
        カメラの位置
    center: np.ndarray
        見る点
    up: np.ndarray
        世界の上の向き

    Returns
    ----------
    pose: list
        4 x 4 の行列の並び
    """
    # 前 (-Z) が中心を向き, カメラの上 (+Y) が世界の上に近い
    forward = center - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = right, up, -forward
    pose[:3, 3] = position

    return pose.tolist()


def write_capture(
    place: Path, angles: list, confidence: int, lost: int = 0,
    z_up: bool = False, radius: float = 1.5
) -> Path:
    """
    中心の周りの方位から撮った撮影 (深度 1.5 m, 信頼度は一定) を書く関数

    Parameters
    ----------
    place: Path
        撮影のディレクトリ
    angles: list
        カメラの方位 (度)
    confidence: int
        信頼度の画像の値
    lost: int = 0
        追跡できなかった枚の数 (先頭から)
    z_up: bool = False
        世界の上を Z にし, 座標の決まり (world_up) を書かない
    radius: float = 1.5
        中心からカメラまでの水平の距離 (0 ならその場で向きだけ変える)

    Returns
    ----------
    place: Path
        撮影のディレクトリ
    """
    # 画像 (中身は使わない), 生の深度 (左の列だけ欠け), 信頼度
    for name in ("images", "raw_depth", "confidence"):
        (place / name).mkdir(parents=True)
    center = np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, 0.0, 1.0] if z_up else [0.0, 1.0, 0.0])
    frames = []
    for index, angle in enumerate(angles):
        rad = math.radians(angle)
        ring = [math.sin(rad), math.cos(rad), 0.3] if z_up else [
            math.sin(rad), 0.3, math.cos(rad)
        ]
        position = center + radius * np.array(ring)
        target = center if radius else position + np.array(
            [math.sin(rad), 0.0, math.cos(rad)]
        )
        Image.new("RGB", (8, 6)).save(place / "images" / f"{index}.jpg")
        depth = np.full((9, 16), 1500, dtype=np.uint16)
        depth[:, 0] = 0
        Image.fromarray(depth).save(place / "raw_depth" / f"{index}.png")
        conf = np.full((9, 16), confidence, dtype=np.uint8)
        conf[:, 0] = 0
        Image.fromarray(conf).save(place / "confidence" / f"{index}.png")
        frames.append({
            "file_path": f"images/{index}.jpg",
            "transform_matrix": look_at(position, target, up),
            "tracking_state": "PAUSED" if index < lost else "TRACKING",
            "raw_depth_file_path": f"raw_depth/{index}.png",
            "confidence_file_path": f"confidence/{index}.png",
        })
    transforms = {
        "fl_x": 10.0, "fl_y": 10.0, "cx": 4.0, "cy": 3.0, "w": 8, "h": 6,
        "depth": {"depth_format": "png16_millimeter", "invalid_value": 0},
        "capture": {"device_model": "試しの端末"},
        "frames": frames,
    }
    if not z_up:
        transforms["coordinate_convention"] = {"world_up": [0.0, 1.0, 0.0]}
    (place / "transforms.json").write_text(
        json.dumps(transforms), encoding="utf-8"
    )

    return place


def test_captures(tmp_path: Path) -> None:
    """
    多視点の撮影を 1 件 1 行で読み, 深度, 距離, 方位の隙間と気付きを
    出すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 一周 (30 度おき) と, 半周だけで信頼度の低い撮影, 追跡の切れた撮影
    root = tmp_path / "captures"
    write_capture(root / "a_round", list(range(0, 360, 30)), 200)
    write_capture(root / "b_half", [0, 45, 90, 135, 180], 50)
    write_capture(root / "c_lost", list(range(0, 360, 45)), 200, lost=2)
    (root / "d_broken").mkdir()
    (root / "d_broken" / "transforms.json").write_text("{}", "utf-8")
    assert detect(root)[0]["loader"] == "capture"
    data = load_data(root)
    assert data.kind == "capture"
    frame = data.frame.set_index("path")
    assert frame.loc["a_round", "frames"] == 12
    assert frame.loc["a_round", "view_gap"] == pytest.approx(30, abs=0.5)
    assert frame.loc["b_half", "view_gap"] == pytest.approx(180, abs=0.5)
    assert frame.loc["a_round", "depth_valid"] == pytest.approx(15 / 16)
    assert frame.loc["b_half", "depth_confident"] == 0
    assert frame.loc["a_round", "distance_median"] == pytest.approx(1.5)
    assert frame.loc["c_lost", "tracked"] == 6
    assert frame.loc["a_round", "device"] == "試しの端末"
    assert isinstance(frame.loc["d_broken", "error"], str)

    # 気付き: 読めない, 追跡, 信頼度, 周回の欠け
    codes = {item["code"] for item in profile(data)["findings"]}
    assert {"unreadable_items", "tracking_lost", "low_depth_confidence",
            "view_gap"} <= codes

    # 1 件の中身と, 撮影 1 つの場所
    item = data.load_item(0)
    assert len(item["frames"]) == 12
    assert item["frames"][0]["pose"].shape == (4, 4)
    assert Path(item["frames"][0]["image"]).is_file()
    one = load_data(root / "a_round")
    assert list(one.frame["path"]) == ["a_round"]
    assert one.load_item(0)["intrinsics"]["w"] == 8


def test_capture_limits(tmp_path: Path) -> None:
    """
    信頼度の下限の誤りと, 前処理が今は対象外であることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 信頼度の下限は 0 から 255 の整数. 下げると信頼できる画素が増える
    place = write_capture(tmp_path / "c", [0, 90, 180, 270], 50)
    with pytest.raises(ValueError, match="min_confidence"):
        load_data(place, min_confidence=300)
    low = load_data(place, min_confidence=10).frame
    assert low.loc[0, "depth_confident"] == pytest.approx(15 / 16)

    # 前処理と書き出しは対象外
    recipe = {"format_version": 1, "target": None, "task": "none",
              "steps": [], "split": {"method": "random", "valid": 0.5}}
    with pytest.raises(ValueError, match="読み込みと分析だけ"):
        prepare(load_data(place), recipe)


def test_capture_review_cases(tmp_path: Path) -> None:
    """
    撮影のディレクトリの外の場所, 単位の書き方, 形の違う信頼度, 上が Z の
    世界, その場で向きだけ変えた撮影, 追跡できなかった枚の扱いを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 外を指す場所 (絶対パスと ..) は無いものとし, 中身にも出さない
    place = write_capture(tmp_path / "out", list(range(0, 360, 60)), 200)
    (tmp_path / "secret.png").write_bytes(b"x")
    transforms = json.loads((place / "transforms.json").read_text("utf-8"))
    transforms["frames"][0]["file_path"] = str(tmp_path / "secret.png")
    transforms["frames"][1]["file_path"] = "../secret.png"
    transforms["frames"][2]["raw_depth_file_path"] = "../secret.png"
    transforms["point_cloud"] = {"file_path": "../secret.png"}
    transforms["depth"]["depth_format"] = "PNG16_Millimeter"
    (place / "transforms.json").write_text(json.dumps(transforms), "utf-8")
    data = load_data(place)
    row = data.frame.iloc[0]
    assert row["missing_files"] == 2 and np.isnan(row["points"])
    assert row["distance_median"] == pytest.approx(1.5)
    item = data.load_item(0)
    assert item["frames"][0]["image"] is None
    assert item["frames"][2]["raw_depth"] is None
    assert item["point_cloud"] is None and item["frames"][3]["raw_depth"]

    # 信頼度の画像の形が違う枚は数えない (撮影は読める)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
        place / "confidence" / "3.png"
    )
    assert load_data(place).frame["error"].isna().all()

    # 上が Z で world_up の無い一周 (60 度おき) は, カメラの上の軸から
    up_z = write_capture(tmp_path / "z", list(range(0, 360, 60)), 200,
                         z_up=True)
    assert load_data(up_z).frame.loc[0, "view_gap"] == pytest.approx(
        60, abs=0.5
    )

    # 見下ろして撮った半周でも, 上の向きは傾かない
    half = write_capture(tmp_path / "half", [0, 45, 90, 135, 180], 200,
                         z_up=True)
    assert load_data(half).frame.loc[0, "view_gap"] == pytest.approx(
        180, abs=0.5
    )

    # その場で向きだけ変えた撮影は, 方位の隙間を出さず, 気付きを出す
    still = write_capture(tmp_path / "still", [0, 30, 60, 90], 200,
                          radius=0.0)
    assert np.isnan(load_data(still).frame.loc[0, "view_gap"])
    codes = {item["code"] for item in profile(load_data(still))["findings"]}
    assert "no_orbit" in codes

    # 撮影の中の画像のディレクトリが別の場所へのシンボリックリンクでも読む
    linked = write_capture(tmp_path / "linked", [0, 120, 240], 200)
    (linked / "images").rename(tmp_path / "shared_images")
    (linked / "images").symlink_to(tmp_path / "shared_images")
    assert load_data(linked).frame.loc[0, "missing_files"] == 0

    # OpenCV とだけ書いた座標の決まり (+Z が前) でも一周を測る
    flipped = write_capture(tmp_path / "cv", list(range(0, 360, 60)), 200)
    transforms = json.loads((flipped / "transforms.json").read_text("utf-8"))
    for frame in transforms["frames"]:
        pose = np.array(frame["transform_matrix"])
        pose[:3, 1:3] *= -1
        frame["transform_matrix"] = pose.tolist()
    transforms["coordinate_convention"]["camera_axes"] = "OpenCV"
    (flipped / "transforms.json").write_text(json.dumps(transforms), "utf-8")
    assert load_data(flipped).frame.loc[0, "view_gap"] == pytest.approx(
        60, abs=0.5
    )

    # 追跡できなかった枚は方位から除く (一周の半分が追跡できない)
    lost = write_capture(tmp_path / "lost", list(range(0, 360, 60)), 200,
                         lost=3)
    assert load_data(lost).frame.loc[0, "view_gap"] == pytest.approx(
        240, abs=0.5
    )
