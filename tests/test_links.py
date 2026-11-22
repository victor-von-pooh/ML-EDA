import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    capture_shift, describe_loaders, detect, export_capture_clouds,
    export_image_masks, load_data, load_items, normalize_capture, profile,
    read_capture
)
from ml_eda.loaders import option_flag
from ml_eda.synthetic import (
    write_image_masks, write_png, write_shape_captures
)


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

    # 既定は誤り (引数の案内付き), たどる設定では同じ中身
    with pytest.raises(ValueError, match="follow_links=True"):
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


def test_transforms_link(tmp_path: Path) -> None:
    """
    撮影の transforms.json 自体が外を指すリンクの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 撮影の中の transforms.json だけを外の同じ中身へのリンクにする
    folder = tmp_path / "captures"
    capture = write_small_capture(folder / "one")
    write_small_capture(folder / "two")
    outside = tmp_path / "outside.json"
    outside.write_bytes((capture / "transforms.json").read_bytes())
    (capture / "transforms.json").unlink()
    (capture / "transforms.json").symlink_to(outside)

    # 撮影を並べたディレクトリ: 既定は理由付き, たどる設定では読む
    data = load_data(folder, loader="capture")
    frame = data.frame.set_index("path")
    assert "外を指すリンク" in frame.loc["one", "error"]
    assert frame.loc["two", "frames"] == 2
    with pytest.raises(ValueError, match="外を指すリンク"):
        data.load_item(list(data.frame["path"]).index("one"))
    data = load_data(folder, loader="capture", follow_links=True)
    assert data.frame["error"].isna().all()
    assert data.frame.set_index("path").loc["one", "frames"] == 2

    # 1 回の撮影を渡した場合も, その撮影と比べる
    data = load_data(capture, loader="capture")
    assert "外を指すリンク" in data.frame.loc[0, "error"]

    # read_capture は引数の案内付きの誤り
    pytest.importorskip("PIL.Image")
    with pytest.raises(ValueError, match="follow_links=True"):
        read_capture(capture)
    assert read_capture(capture, follow_links=True)["images"].shape[0] == 2


def test_dataset_yaml_link(tmp_path: Path) -> None:
    """
    版 2 のデータセットの dataset.yaml が外を指すリンクの扱いを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # dataset.yaml を外の同じ中身へのリンクにする
    pytest.importorskip("PIL.Image")
    made = write_image_masks(tmp_path / "raw", count=4)
    export_image_masks(made["images"], made["masks"], tmp_path / "ds")
    outside = tmp_path / "outside.yaml"
    outside.write_bytes((tmp_path / "ds" / "dataset.yaml").read_bytes())
    (tmp_path / "ds" / "dataset.yaml").unlink()
    (tmp_path / "ds" / "dataset.yaml").symlink_to(outside)

    # 既定は誤り, たどる設定では読む
    with pytest.raises(ValueError, match="follow_links=True"):
        load_items(tmp_path / "ds")
    dataset = load_items(tmp_path / "ds", follow_links=True)
    assert dataset.info["format_version"] == 2


def test_option_flag(tmp_path: Path) -> None:
    """
    真偽の設定の文字の読み方を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 受け付ける文字 (大文字と前後の空白は区別しない) と既定
    for word in ("true", "1", "yes", "on", " TRUE ", "On"):
        assert option_flag({"follow_links": word}, "follow_links", False)
    for word in ("false", "0", "no", "off", "", " Off "):
        assert not option_flag({"follow_links": word}, "follow_links", True)
    assert option_flag({}, "digest", True)
    assert not option_flag({"digest": False}, "digest", True)

    # 知らない文字は設定の名前と受け付ける値を挙げて誤り
    with pytest.raises(ValueError, match="follow_links.*true, 1, yes, on"):
        option_flag({"follow_links": "n"}, "follow_links", False)
    folder = tmp_path / "sounds"
    write_tone(folder / "a.wav")
    with pytest.raises(ValueError, match="follow_links"):
        load_data(folder, follow_links="maybe")

    # 画像の digest も同じ読み方 (off で hash を求めない)
    pytest.importorskip("PIL.Image")
    images = tmp_path / "images"
    images.mkdir()
    write_png(images / "a.png", np.zeros((2, 2, 3), dtype=np.uint8))
    assert "digest" not in load_data(images, digest="off").frame.columns
    assert "digest" in load_data(images).frame.columns
    with pytest.raises(ValueError, match="digest"):
        load_data(images, digest="n")


def link_out(place: Path, outside: Path) -> None:
    """
    ファイルかディレクトリを外へ移し, 元の場所にそこへのリンクを置く関数

    Parameters
    ----------
    place: Path
        元の場所
    outside: Path
        移す先 (読み込むディレクトリの外)

    Returns
    ----------
    None
    """
    # 中身は同じまま, 場所だけを外にする
    outside.parent.mkdir(parents=True, exist_ok=True)
    place.rename(outside)
    place.symlink_to(outside)


def test_text_links(tmp_path: Path) -> None:
    """
    文のディレクトリの, 外と中を指すリンクの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 実の文, 外を指すリンク, 中を指すリンク
    folder = tmp_path / "texts"
    folder.mkdir()
    (folder / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    (folder / "out.txt").symlink_to(tmp_path / "secret.txt")
    (folder / "same.txt").symlink_to(folder / "a.txt")

    # 既定は外を読まず, 理由を残す (空の文には数えない)
    data = load_data(folder, loader="text_lines")
    frame = data.frame.set_index("file")
    assert "follow_links を true にしてください" in frame.loc[
        "out.txt", "error"
    ]
    assert pd.isna(frame.loc["out.txt", "text"])
    assert frame.loc["same.txt", "text"] == "hello world"
    assert "secret" not in data.frame["text"].fillna("").tolist()
    result = profile(data)["items"]
    assert result["unreadable"] == 1
    assert result["empty"] == 0
    position = list(data.frame["file"]).index("out.txt")
    with pytest.raises(ValueError, match="外を指すリンク"):
        data.load_item(position)

    # たどる設定では読み, 誤りの列は無い
    data = load_data(folder, loader="text_lines", follow_links="true")
    assert "error" not in data.frame.columns
    assert data.load_item(position) == "secret"


def test_export_image_masks_links(tmp_path: Path) -> None:
    """
    画像と領域の書き出しが外を指すリンクを写さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 枚の画像と, 別の 1 枚の領域を外へのリンクにする
    pytest.importorskip("PIL.Image")
    made = write_image_masks(tmp_path / "raw", count=4)
    images = sorted(Path(made["images"]).iterdir())
    masks = sorted(Path(made["masks"]).iterdir())
    link_out(images[0], tmp_path / "outside" / images[0].name)
    link_out(masks[1], tmp_path / "outside" / ("mask_" + masks[1].name))

    # 既定: 外のファイルは除き, notes に案内付きで書く
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds",
        split={"method": "none"}
    )
    notes = " ".join(info["notes"])
    assert "画像のディレクトリの外を指すリンクが 1 件" in notes
    assert "領域のディレクトリの外を指すリンクが 1 件" in notes
    assert "follow_links=True にしてください" in notes
    assert len(load_items(tmp_path / "ds").items) == 2

    # 文字の "false" と "0" は読まない側
    for flag in ("false", "0"):
        info = export_image_masks(
            made["images"], made["masks"], tmp_path / f"text_{flag}",
            split={"method": "none"}, follow_links=flag
        )
        assert len(load_items(tmp_path / f"text_{flag}").items) == 2

    # たどる設定では全てを写す
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "all",
        split={"method": "none"}, follow_links=True
    )
    assert len(load_items(tmp_path / "all").items) == 4
    assert not any("外を指すリンク" in note for note in info["notes"])


def test_export_capture_links(tmp_path: Path) -> None:
    """
    撮影と点群の書き出しが外を指すリンクを写さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 撮影の 1 枚, 撮影自体, 点群を外へのリンクにする
    pytest.importorskip("PIL.Image")
    made = write_shape_captures(tmp_path / "raw", views=4, points=100)
    captures, clouds = Path(made["captures"]), Path(made["clouds"])
    outside = tmp_path / "outside"
    link_out(
        captures / "box_a" / "images" / "001.png", outside / "001.png"
    )
    link_out(captures / "box_b", outside / "box_b")
    link_out(clouds / "sphere_a.ply", outside / "sphere_a.ply")

    # 既定: 外の撮影と点群は組にせず, 外の枚は除いて notes に書く
    info = export_capture_clouds(
        captures, clouds, tmp_path / "ds", split={"method": "none"}
    )
    notes = " ".join(info["notes"])
    assert "読めない撮影が 1 件" in notes
    assert "点群のディレクトリの外を指すリンクが 1 件" in notes
    assert "外を指すリンクの) 1 枚を除きました" in notes
    dataset = load_items(tmp_path / "ds")
    assert sorted(dataset.items["id"]) == ["box_a", "sphere_b"]
    export_capture_clouds(
        captures, clouds, tmp_path / "text", split={"method": "none"},
        follow_links="false"
    )
    assert sorted(load_items(tmp_path / "text").items["id"]) == [
        "box_a", "sphere_b"
    ]
    frames = {
        item["id"]: len(item["capture"]["poses"])
        for item in dataset.split("train")
    }
    assert frames == {"box_a": 3, "sphere_b": 4}

    # たどる設定では全てを写す
    export_capture_clouds(
        captures, clouds, tmp_path / "all", split={"method": "none"},
        follow_links=True
    )
    dataset = load_items(tmp_path / "all")
    assert len(dataset.items) == 4
    assert all(
        len(item["capture"]["poses"]) == 4 for item in dataset.split("train")
    )


def test_normalize_capture_links(tmp_path: Path) -> None:
    """
    normalize_capture と capture_shift が外を指すリンクを読まないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 枚の画像を外へのリンクにする
    pytest.importorskip("PIL.Image")
    made = write_shape_captures(tmp_path / "raw", views=4, points=100)
    capture = Path(made["captures"]) / "box_a"
    link_out(capture / "images" / "002.png", tmp_path / "out" / "002.png")

    # 既定は除き, 平行移動も同じ枚から求める
    summary = normalize_capture(capture, tmp_path / "new")
    assert summary["frames"] == 3
    assert np.allclose(summary["shift"], capture_shift(capture))
    assert not (tmp_path / "new" / "images" / "002.png").exists()
    summary = normalize_capture(
        capture, tmp_path / "text", follow_links="false"
    )
    assert summary["frames"] == 3
    with pytest.raises(ValueError, match="follow_links"):
        capture_shift(capture, follow_links="maybe")

    # たどる設定では全ての枚
    summary = normalize_capture(capture, tmp_path / "all", follow_links=True)
    assert summary["frames"] == 4
    assert np.allclose(
        summary["shift"], capture_shift(capture, follow_links=True)
    )

    # transforms.json が外なら, 引数の案内付きの誤り
    link_out(capture / "transforms.json", tmp_path / "out" / "t.json")
    with pytest.raises(ValueError, match="follow_links=True にしてください"):
        normalize_capture(capture, tmp_path / "again")
    with pytest.raises(ValueError, match="follow_links=True"):
        capture_shift(capture)


def test_sniff_capture_links(tmp_path: Path) -> None:
    """
    推定が設定 follow_links を撮影の確かめに使うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 撮影の transforms.json が全て外へのリンク
    folder = tmp_path / "captures"
    capture = write_small_capture(folder / "one")
    link_out(capture / "transforms.json", tmp_path / "out.json")

    # 既定では撮影と見なさず, たどる設定 (文字も) では撮影
    assert "capture" not in [item["loader"] for item in detect(folder)]
    for flag in (True, "true"):
        assert detect(folder, follow_links=flag)[0]["loader"] == "capture"
    data = load_data(folder, follow_links=True)
    assert data.kind == "capture"
    assert data.frame.loc[0, "frames"] == 2
