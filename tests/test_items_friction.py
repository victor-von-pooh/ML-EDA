import json
from pathlib import Path

import numpy as np
import pytest

import ml_eda
from ml_eda import (
    export_capture_clouds, export_image_masks, load_items,
    normalize_capture, read_capture, read_cloud
)
from ml_eda.items import read_mask, source_name, write_ply
from ml_eda.media import read_image
from ml_eda.synthetic import (
    write_image_masks, write_png, write_shape_captures
)

Image = pytest.importorskip("PIL.Image")


def all_arrays(value: object) -> list[np.ndarray]:
    """
    辞書と並びの中の配列を全て集める関数

    Parameters
    ----------
    value: object
        1 件の中身 (配列, 辞書, 並び)

    Returns
    ----------
    arrays: list[np.ndarray]
        見つけた配列
    """
    # 辞書と並びは中をたどる
    arrays: list[np.ndarray] = []
    if isinstance(value, np.ndarray):
        arrays.append(value)
    elif isinstance(value, dict):
        for inner in value.values():
            arrays += all_arrays(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            arrays += all_arrays(inner)

    return arrays


def test_arrays_writable(tmp_path: Path) -> None:
    """
    load_items と読む関数の返す配列が, 種類によらず書き込めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 画像と領域
    made = write_image_masks(tmp_path / "raw", count=4)
    export_image_masks(
        made["images"], made["masks"], tmp_path / "masks",
        split={"method": "none"}
    )
    item = load_items(tmp_path / "masks").split("train")[0]
    for key in ("image", "mask"):
        assert item[key].flags.writeable, key
    image = read_image(made["images"] / "sample_000.png")
    assert image.flags.writeable

    # 灰色, 2 値, パレットの画像と領域
    write_png(tmp_path / "gray.png", np.zeros((4, 5), dtype=np.uint8))
    assert read_image(tmp_path / "gray.png").flags.writeable
    Image.new("1", (5, 4)).save(tmp_path / "binary.png")
    assert read_image(tmp_path / "binary.png").flags.writeable
    assert read_mask(tmp_path / "binary.png").flags.writeable
    Image.new("P", (5, 4)).save(tmp_path / "palette.png")
    assert read_mask(tmp_path / "palette.png").flags.writeable

    # 2 進の PLY の点, 色, 法線
    write_ply(tmp_path / "cloud.ply", {
        "points": np.zeros((3, 3)), "colors": np.zeros((3, 3)),
        "normals": np.ones((3, 3)),
    })
    cloud = read_cloud(tmp_path / "cloud.ply")
    assert all(cloud[key].flags.writeable
               for key in ("points", "colors", "normals"))

    # 撮影と点群 (撮影の中の配列も全て)
    shapes = write_shape_captures(tmp_path / "shapes", views=4, points=200)
    export_capture_clouds(
        shapes["captures"], shapes["clouds"], tmp_path / "clouds",
        split={"method": "none"}
    )
    item = load_items(tmp_path / "clouds").split("train")[0]
    arrays = all_arrays(item)
    assert len(arrays) >= 5
    assert all(array.flags.writeable for array in arrays)


def test_depth_convention(tmp_path: Path) -> None:
    """
    撮影の座標の決まりに深度の測り方と画素の中心が書かれ, その決まりで
    深度から戻した点が形の面に乗ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 合成の見本は transforms.json に書くので注意は出ない
    made = write_shape_captures(
        tmp_path / "raw", views=4, points=200,
        shapes=[{"name": "ball", "type": "sphere", "center": [0.7, 0.1, 0.2],
                 "radius": 0.15}]
    )
    info = export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "ds",
        split={"method": "none"}
    )
    convention = info["entries"]["capture"]["convention"]
    assert convention["depth"] == "z"
    assert convention["pixel_center"] == 0.5
    assert not any("pixel_center" in note for note in info["notes"])

    # z の深度と +0.5 の画素の中心で戻すと, 球の面 (正規化で中心が原点)
    # の上. 光線に沿った距離と見なすと面から外れる
    capture = load_items(tmp_path / "ds").split("train")[0]["capture"]
    assert capture["convention"]["depth"] == "z"
    intrinsics = capture["intrinsics"]
    pose, depth = capture["poses"][0], capture["depth"][0]
    rows, columns = np.nonzero(depth > 0)
    z = depth[rows, columns].astype(np.float64)
    direction = np.stack([
        (columns + 0.5 - intrinsics["cx"]) / intrinsics["fl_x"],
        -(rows + 0.5 - intrinsics["cy"]) / intrinsics["fl_y"],
        -np.ones(len(z)),
    ], axis=1)
    as_z = (direction * z[:, None]) @ pose[:3, :3].T + pose[:3, 3]
    gaps = np.abs(np.linalg.norm(as_z, axis=1) - 0.15)
    assert gaps.max() < 0.002
    unit = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    as_ray = (unit * z[:, None]) @ pose[:3, :3].T + pose[:3, 3]
    assert np.abs(np.linalg.norm(as_ray, axis=1) - 0.15).max() > 0.005


def test_depth_convention_missing(tmp_path: Path) -> None:
    """
    深度の決まりが無い撮影は既定 (z, 0.5) として注意に書き, 書かれた
    決まりはそのまま写すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 つ目は深度の決まりを消す
    made = write_shape_captures(tmp_path / "raw", views=4, points=200)
    file = made["captures"] / "sphere_a" / "transforms.json"
    transforms = json.loads(file.read_text(encoding="utf-8"))
    for key in ("depth", "pixel_center"):
        del transforms["coordinate_convention"][key]
    file.write_text(json.dumps(transforms), encoding="utf-8")
    assert read_capture(file.parent)["convention"]["depth"] == "z"
    info = export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "ds",
        split={"method": "none"}
    )
    convention = info["entries"]["capture"]["convention"]
    assert (convention["depth"], convention["pixel_center"]) == ("z", 0.5)
    assert any("pixel_center" in note for note in info["notes"])

    # 光線に沿った距離と書かれた撮影はそのまま
    for capture in made["captures"].iterdir():
        file = capture / "transforms.json"
        transforms = json.loads(file.read_text(encoding="utf-8"))
        transforms["coordinate_convention"].update(
            {"depth": "ray", "pixel_center": 0.0}
        )
        file.write_text(json.dumps(transforms), encoding="utf-8")
    info = export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "ray",
        split={"method": "none"}
    )
    convention = info["entries"]["capture"]["convention"]
    assert (convention["depth"], convention["pixel_center"]) == ("ray", 0.0)
    assert not any("pixel_center" in note for note in info["notes"])
    item = load_items(tmp_path / "ray").split("train")[0]
    assert item["capture"]["convention"]["depth"] == "ray"
    assert item["capture"]["convention"]["pixel_center"] == 0.0

    # null は書かれていないもの, 知らない測り方は誤り
    file = made["captures"] / "box_a" / "transforms.json"
    transforms = json.loads(file.read_text(encoding="utf-8"))
    transforms["coordinate_convention"]["depth"] = None
    file.write_text(json.dumps(transforms), encoding="utf-8")
    assert read_capture(file.parent)["convention"]["depth"] == "z"
    for key, value in (("depth", "range"), ("pixel_center", "0.5")):
        transforms["coordinate_convention"][key] = value
        file.write_text(json.dumps(transforms), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            read_capture(file.parent)
        with pytest.raises(ValueError, match=key):
            export_capture_clouds(
                made["captures"], made["clouds"], tmp_path / "bad",
                split={"method": "none"}
            )
        transforms["coordinate_convention"][key] = None


def test_public_readers(tmp_path: Path) -> None:
    """
    推論のときに使う読む関数が公開され, load_items と同じ形で読むことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 公開の名前
    for name in ("read_capture", "read_cloud", "normalize_capture",
                 "capture_shift"):
        assert name in ml_eda.__all__

    # 新しい撮影を正規化し, 文字列で渡した場所から読むと書き出したものと同じ
    made = write_shape_captures(tmp_path / "raw", views=4, points=200)
    export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "ds",
        split={"method": "none"}
    )
    dataset = load_items(tmp_path / "ds")
    item = next(
        entry for entry in dataset.split("train") if entry["id"] == "box_a"
    )
    normalize_capture(made["captures"] / "box_a", tmp_path / "new")
    capture = read_capture(str(tmp_path / "new"))
    assert sorted(capture) == sorted(item["capture"])
    for key in ("poses", "images", "depth", "confidence"):
        np.testing.assert_allclose(capture[key], item["capture"][key])
    assert capture["convention"] == item["capture"]["convention"]
    cloud = read_cloud(str(made["clouds"] / "box_a.ply"))
    assert cloud["points"].shape == (200, 3)


def test_source_names_only(tmp_path: Path) -> None:
    """
    版 2 の source に手元のパスが残らず, ディレクトリの名前だけになる
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 絶対パスで渡しても名前だけ
    made = write_image_masks(tmp_path / "raw", count=4)
    info = export_image_masks(
        made["images"].resolve(), made["masks"].resolve(),
        tmp_path / "masks", split={"method": "none"}
    )
    assert info["source"]["images"] == "images"
    assert info["source"]["masks"] == "masks"
    assert info["source"]["paths"] == "name_only"
    assert info["source"]["note"]
    shapes = write_shape_captures(tmp_path / "shapes", views=4, points=200)
    info = export_capture_clouds(
        shapes["captures"].resolve(), str(shapes["clouds"]) + "/",
        tmp_path / "clouds", split={"method": "none"}
    )
    assert info["source"]["captures"] == "captures"
    assert info["source"]["clouds"] == "clouds"
    assert info["source"]["paths"] == "name_only"

    # 1 つの撮影のディレクトリも, その名前
    info = export_capture_clouds(
        shapes["captures"] / "box_a", shapes["clouds"], tmp_path / "one",
        split={"method": "none"}
    )
    assert info["source"]["captures"] == "box_a"

    # 書いた説明のどこにも一時ディレクトリの場所は無い
    for name in ("masks", "clouds"):
        text = (tmp_path / name / "dataset.yaml").read_text(encoding="utf-8")
        assert str(tmp_path.resolve()) not in text
        assert str(tmp_path) not in text


def test_source_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    source に書く名前が, 渡した場所の最後の部分だけになることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        今のディレクトリの変更

    Returns
    ----------
    None
    """
    # 相対の場所, . と .., 終わりの区切り
    (tmp_path / "data" / "images").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "data")
    assert source_name("images") == "images"
    assert source_name("images/") == "images"
    assert source_name(Path("images") / ".." / "images") == "images"
    assert source_name(".") == "data"

    # シンボリックリンクはたどらず渡した名前. ~ だけと根は空
    (tmp_path / "data" / "link").symlink_to(tmp_path / "data" / "images")
    assert source_name("link") == "link"
    assert source_name("~") == ""
    assert source_name("~/") == ""
    assert source_name("/") == ""


def test_random_shapes(tmp_path: Path) -> None:
    """
    write_shape_captures の count と seed で, 乱数の形を決まった数だけ
    作れることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 数, 名前, 種類, 原点からの距離
    made = write_shape_captures(
        tmp_path / "a", count=6, seed=3, views=3, points=100
    )
    shapes = made["shapes"]
    assert [shape["name"] for shape in shapes] == [
        f"shape_{index:03d}" for index in range(6)
    ]
    assert {shape["type"] for shape in shapes} <= {"sphere", "box"}
    assert len(list(made["clouds"].glob("*.ply"))) == 6
    assert len(list(made["captures"].iterdir())) == 6
    for shape in shapes:
        flat = np.hypot(shape["center"][0], shape["center"][2])
        assert 0.5 <= flat <= 1.5
        assert ("radius" in shape) == (shape["type"] == "sphere")

    # 同じ種は同じ形, 違う種は違う形
    again = write_shape_captures(
        tmp_path / "b", count=6, seed=3, views=3, points=100
    )
    assert again["shapes"] == shapes
    other = write_shape_captures(
        tmp_path / "c", count=6, seed=4, views=3, points=100
    )
    assert other["shapes"] != shapes

    # 書き出せて, 各撮影に物が写る
    info = export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "ds",
        split={"valid": 0.5, "seed": 0}
    )
    assert sum(split["items"] for split in info["splits"].values()) == 6
    # 各撮影の全ての枚に物が写り, 縁にかからない
    dataset = load_items(tmp_path / "ds")
    for name in dataset.splits:
        for item in dataset.split(name):
            for depth in item["capture"]["depth"]:
                assert (depth > 0).any()
                edges = np.concatenate([
                    depth[0], depth[-1], depth[:, 0], depth[:, -1]
                ])
                assert (edges == 0).all()

    # 誤り: 形の並びと同時, 1 より小さい数
    with pytest.raises(ValueError, match="同時"):
        write_shape_captures(tmp_path / "d", shapes=shapes, count=2)
    with pytest.raises(ValueError, match="count"):
        write_shape_captures(tmp_path / "e", count=0)
    with pytest.raises(ValueError, match="整数"):
        write_shape_captures(tmp_path / "f", count=2.5)  # type: ignore
