import json
import wave
from pathlib import Path

import numpy as np
import pytest

from ml_eda import (
    describe_loaders, export_image_masks, load_data, load_items, profile,
    read_capture
)
from ml_eda.synthetic import write_image_masks, write_png


def write_tone(path: Path) -> None:
    """
    短い 16 bit の WAVE を書く関数

    Parameters
    ----------
    path: Path
        書くファイル

    Returns
    ----------
    None
    """
    # 0.1 秒の正弦波
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.sin(np.linspace(0, 20 * np.pi, 1600)) * 0.5
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes((samples * 32767).astype("<i2").tobytes())


def write_points(path: Path) -> None:
    """
    3 点の .xyz を書く関数

    Parameters
    ----------
    path: Path
        書くファイル

    Returns
    ----------
    None
    """
    # 1 行 1 点
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0 0\n1 0 0\n0 1 0\n", encoding="utf-8")


def write_small_capture(place: Path) -> Path:
    """
    2 枚の画像を持つ小さな撮影を書く関数

    Parameters
    ----------
    place: Path
        撮影のディレクトリ

    Returns
    ----------
    place: Path
        撮影のディレクトリ
    """
    # 画像と transforms.json (姿勢は単位行列)
    (place / "images").mkdir(parents=True)
    frames = []
    for index in range(2):
        name = f"images/{index}.png"
        write_png(place / name, np.zeros((4, 4, 3), dtype=np.uint8))
        frames.append({
            "file_path": name, "transform_matrix": np.eye(4).tolist()
        })
    transforms = {"w": 4, "h": 4, "fl_x": 2.0, "frames": frames}
    (place / "transforms.json").write_text(
        json.dumps(transforms), encoding="utf-8"
    )

    return place


def test_audio_links(tmp_path: Path) -> None:
    """
    音声のディレクトリの, 外と中を指すリンクの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 実の件, 外を指すリンク, 中を指すリンク
    folder = tmp_path / "sounds"
    write_tone(folder / "a.wav")
    write_tone(tmp_path / "secret" / "x.wav")
    (folder / "out.wav").symlink_to(tmp_path / "secret" / "x.wav")
    (folder / "same.wav").symlink_to(folder / "a.wav")

    # 既定は外を読まず, 理由を残す. 中のリンクは読む
    data = load_data(folder)
    frame = data.frame.set_index("path")
    assert "follow_links" in frame.loc["out.wav", "error"]
    assert np.isnan(frame.loc["out.wav", "sample_rate"])
    assert frame.loc["same.wav", "sample_rate"] == 16000
    position = list(data.frame["path"]).index("out.wav")
    with pytest.raises(ValueError, match="外を指すリンク"):
        data.load_item(position)
    with pytest.raises(ValueError, match="外を指すリンク"):
        data.load_item(position, 0.05)
    codes = {item["code"] for item in profile(data)["findings"]}
    assert "unreadable_items" in codes

    # たどる設定 (画面からの文字も) では外も読む
    for flag in (True, "true"):
        data = load_data(folder, follow_links=flag)
        assert data.frame["error"].isna().all()
        assert data.load_item(position)["sample_rate"] == 16000
    assert load_data(folder, follow_links="false").frame[
        "error"
    ].notna().sum() == 1

    # 読み込んだディレクトリ自体のリンクは信頼し, たどった後の中で比べる
    (tmp_path / "alias").symlink_to(folder)
    data = load_data(tmp_path / "alias")
    frame = data.frame.set_index("path")
    assert frame.loc["same.wav", "sample_rate"] == 16000
    assert isinstance(frame.loc["out.wav", "error"], str)

    # ファイル 1 つのリンクは渡したものとして読む
    data = load_data(folder / "out.wav")
    assert data.frame["error"].isna().all()
    assert data.load_item(0)["sample_rate"] == 16000

    # 設定は読み込みの種類の説明に出る
    options = {item["name"]: item["options"] for item in describe_loaders()}
    for name in ("image_folder", "pointcloud", "capture", "audio_folder"):
        assert "follow_links" in options[name]


def test_point_links(tmp_path: Path) -> None:
    """
    点群のディレクトリの, 外を指すリンクの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # クラスのフォルダ自体が外へのリンク
    folder = tmp_path / "clouds"
    write_points(folder / "a" / "p.xyz")
    write_points(tmp_path / "shared" / "q.xyz")
    (folder / "b").symlink_to(tmp_path / "shared")
    data = load_data(folder, loader="pointcloud")
    frame = data.frame.set_index("path")
    assert "外を指すリンク" in frame.loc["b/q.xyz", "error"]
    assert frame.loc["a/p.xyz", "points"] == 3
    assert profile(data)["items"]["unreadable"] == 1
    with pytest.raises(ValueError, match="外を指すリンク"):
        data.load_item(list(data.frame["path"]).index("b/q.xyz"))

    # たどる設定
    data = load_data(folder, loader="pointcloud", follow_links=True)
    assert data.frame["points"].tolist() == [3, 3]


def test_image_links(tmp_path: Path) -> None:
    """
    画像のディレクトリの, 外と中を指すリンクの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 実の画像, 外を指すリンク, 中を指すリンク
    pytest.importorskip("PIL.Image")
    folder = tmp_path / "images"
    folder.mkdir()
    array = np.zeros((4, 6, 3), dtype=np.uint8)
    write_png(folder / "a.png", array)
    write_png(tmp_path / "secret.png", array)
    (folder / "out.png").symlink_to(tmp_path / "secret.png")
    (folder / "same.png").symlink_to(folder / "a.png")

    # 既定は外を読まない (大きさ, hash も求めない)
    data = load_data(folder)
    frame = data.frame.set_index("path")
    assert "follow_links" in frame.loc["out.png", "error"]
    assert np.isnan(frame.loc["out.png", "width"])
    assert frame.loc["same.png", "width"] == 6
    result = profile(data)["items"]
    assert result["unreadable"] == 1
    position = list(data.frame["path"]).index("out.png")
    with pytest.raises(ValueError, match="外を指すリンク"):
        data.load_item(position)

    # たどる設定では読み, 誤りの列は無い
    data = load_data(folder, follow_links=True)
    assert "error" not in data.frame.columns
    assert data.load_item(position).shape == (4, 6, 3)


def test_capture_links(tmp_path: Path) -> None:
    """
    撮影を並べたディレクトリの, 外を指すリンクの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 撮影自体が外へのリンクと, 撮影の中の画像が外へのリンク
    folder = tmp_path / "captures"
    write_small_capture(folder / "inner")
    write_small_capture(tmp_path / "elsewhere")
    (folder / "linked").symlink_to(tmp_path / "elsewhere")
    mixed = write_small_capture(folder / "mixed")
    (mixed / "images" / "1.png").unlink()
    (mixed / "images" / "1.png").symlink_to(
        tmp_path / "elsewhere" / "images" / "1.png"
    )

    # 既定: 外の撮影は理由付き, 外の画像は無いものとする
    data = load_data(folder, loader="capture")
    frame = data.frame.set_index("path")
    assert "外を指すリンク" in frame.loc["linked", "error"]
    assert frame.loc["inner", "missing_files"] == 0
    assert frame.loc["mixed", "missing_files"] == 1
    paths = list(data.frame["path"])
    with pytest.raises(ValueError, match="外を指すリンク"):
        data.load_item(paths.index("linked"))
    item = data.load_item(paths.index("mixed"))
    assert item["frames"][1]["image"] is None

    # たどる設定
    data = load_data(folder, loader="capture", follow_links=True)
    frame = data.frame.set_index("path")
    assert frame["error"].isna().all()
    assert frame.loc["mixed", "missing_files"] == 0
    assert data.load_item(paths.index("mixed"))["frames"][1]["image"]

    # 1 回の撮影を渡した場合は, その撮影と比べる
    data = load_data(mixed, loader="capture")
    assert data.frame.loc[0, "missing_files"] == 1


def test_item_links(tmp_path: Path) -> None:
    """
    版 2 のデータセットの表の場所が外を指すリンクの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出した 1 件の画像を, 外の同じ画像へのリンクに置き換える
    pytest.importorskip("PIL.Image")
    made = write_image_masks(tmp_path / "raw", count=4)
    export_image_masks(
        made["images"], made["masks"], tmp_path / "ds",
        split={"valid": 0.5}
    )
    first = load_items(tmp_path / "ds").split("train")[0]
    place = Path(load_items(tmp_path / "ds").split("train", read=False)[0][
        "image"
    ])
    copy = tmp_path / "outside.png"
    copy.write_bytes(place.read_bytes())
    place.unlink()
    place.symlink_to(copy)

    # 既定は誤り, たどる設定では同じ中身
    with pytest.raises(ValueError, match="外を指すリンク"):
        load_items(tmp_path / "ds").split("train")[0]
    again = load_items(tmp_path / "ds", follow_links=True).split("train")[0]
    assert np.array_equal(again["image"], first["image"])

    # データセットのディレクトリ自体のリンクは信頼する
    (tmp_path / "alias").symlink_to(tmp_path / "ds")
    assert load_items(tmp_path / "alias").split("valid")[0]["image"].ndim == 3


def test_read_capture_links(tmp_path: Path) -> None:
    """
    read_capture が撮影の外を指すリンクを無いものとすることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 画像のディレクトリが外へのリンク
    pytest.importorskip("PIL.Image")
    capture = write_small_capture(tmp_path / "new")
    (capture / "images").rename(tmp_path / "shared")
    (capture / "images").symlink_to(tmp_path / "shared")
    with pytest.raises(ValueError, match="画像の無い枚"):
        read_capture(capture)
    assert read_capture(capture, follow_links=True)["images"].shape == (
        2, 4, 4, 3
    )
