from pathlib import Path

import numpy as np
import pytest
import yaml

from ml_eda import (
    ITEM_STEPS, describe_item_steps, export_capture_clouds,
    export_image_masks, load_items, normalize_capture, read_capture,
    transform_entry
)
from ml_eda.itemprep import check_entry_steps, nearest
from ml_eda.items import scale_intrinsics, scale_nearest
from ml_eda.media import read_image
from ml_eda.synthetic import (
    write_image_masks, write_png, write_shape_captures
)

Image = pytest.importorskip("PIL.Image")

# 大きさの混ざった見本の (高さ, 幅)
MIXED = [(24, 32), (40, 30), (17, 51)]

# 書き出しでそろえる大きさ
RESIZE = [{"op": "resize", "width": 32, "height": 32}]


def unproject(capture: dict) -> np.ndarray:
    """
    撮影の深度を, 内部パラメータと姿勢で世界の点に戻す関数

    Parameters
    ----------
    capture: dict
        read_capture の撮影 (深度の決まりは z, 画素の中心は 0.5)

    Returns
    ----------
    points: np.ndarray
        (点の数, 3) の世界の座標 (欠けの画素は除く)
    """
    # OpenGL の軸 (-Z が前) で, 画素の中心を通る光線
    intrinsics = capture["intrinsics"]
    found = []
    for pose, depth in zip(capture["poses"], capture["depth"]):
        rows, columns = np.nonzero(depth > 0)
        values = depth[rows, columns]
        camera = np.stack([
            (columns + 0.5 - intrinsics["cx"]) / intrinsics["fl_x"] * values,
            -(rows + 0.5 - intrinsics["cy"]) / intrinsics["fl_y"] * values,
            -values,
        ], axis=1)
        found.append(camera @ pose[:3, :3].T + pose[:3, 3])
    points = np.concatenate(found)

    return points


def mean_distance(points: np.ndarray, cloud: np.ndarray) -> float:
    """
    点から点群の最も近い点までの距離の平均を返す関数

    Parameters
    ----------
    points: np.ndarray
        (点の数, 3)
    cloud: np.ndarray
        (点の数, 3)

    Returns
    ----------
    distance: float
        距離の平均
    """
    # 全ての組の距離 (見本は小さい)
    gaps = np.linalg.norm(points[:, None, :] - cloud[None, :, :], axis=2)
    distance = float(gaps.min(axis=1).mean())

    return distance


def test_image_mask_steps(tmp_path: Path) -> None:
    """
    大きさの混ざった画像と領域を書き出しでそろえ, 手順を残すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 大きさの混ざった見本を 32 x 32 にそろえて書き出す (版 3)
    made = write_image_masks(tmp_path / "raw", count=6, sizes=MIXED)
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds",
        classes=made["classes"], item_steps={"image": RESIZE}
    )
    assert info["format_version"] == 3
    image, mask = info["entries"]["image"], info["entries"]["mask"]
    assert image["steps"] == RESIZE
    assert mask["steps"] == [
        {"op": "resize_mask", "width": 32, "height": 32}
    ]
    assert image["shape"] == [32, 32, 3] and mask["shape"] == [32, 32]
    assert image["formats"] == ["png"] and mask["formats"] == ["png"]
    assert mask["dtype"] == "uint8" and mask["values"] == [0, 1, 2]

    # 読み込みは手順をかけた後のファイル. 領域の値は混ざらず, 元の領域を
    # 最も近い画素で縮めたものと同じ
    dataset = load_items(tmp_path / "ds")
    seen = 0
    for name in dataset.splits:
        for item in dataset.split(name):
            assert item["image"].shape == (32, 32, 3)
            assert item["image"].dtype == np.uint8
            assert item["mask"].shape == (32, 32)
            source = np.asarray(
                Image.open(made["masks"] / f"{item['id']}.png")
            )
            assert np.array_equal(item["mask"], nearest(source, 32, 32))
            assert set(np.unique(item["mask"])) <= {0, 1, 2}
            seen += 1
    assert seen == 6

    # 推論: 新しい 1 件に同じ手順 (場所でも配列でも, 説明の辞書でも)
    first = dataset.split(dataset.splits[0])[0]
    place = made["images"] / f"{first['id']}.png"
    assert np.array_equal(
        transform_entry(dataset, "image", place), first["image"]
    )
    assert np.array_equal(
        transform_entry(dataset.info, "image", read_image(place)),
        first["image"]
    )
    mask_place = made["masks"] / f"{first['id']}.png"
    found = transform_entry(tmp_path / "ds", "mask", mask_place)
    assert np.array_equal(found, first["mask"])
    assert found.dtype == np.uint8


def test_steps_order_and_colors(tmp_path: Path) -> None:
    """
    色の手順は画像だけにかけ, 領域の手順は大きさだけになることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 灰色にしてから大きさをそろえる
    made = write_image_masks(tmp_path / "raw", count=4, sizes=MIXED)
    steps = [{"op": "grayscale"}, *RESIZE]
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "gray",
        item_steps={"image": steps}
    )
    assert info["entries"]["image"]["shape"] == [32, 32, 1]
    assert [s["op"] for s in info["entries"]["mask"]["steps"]] == [
        "resize_mask"
    ]
    item = load_items(tmp_path / "gray").split("train")[0]
    assert item["image"].shape == (32, 32, 1)

    # 色の手順だけなら領域は手順を持たない (大きさの同じ見本)
    same = write_image_masks(tmp_path / "same", count=4)
    info = export_image_masks(
        same["images"], same["masks"], tmp_path / "rgb",
        item_steps={"image": ["grayscale"]}
    )
    assert info["format_version"] == 3
    assert info["entries"]["image"]["steps"] == [{"op": "grayscale"}]
    assert "steps" not in info["entries"]["mask"]

    # 手順を渡さないか空なら, 今までと同じ版 2 (ファイルをそのまま写す)
    for given in (None, {}, {"image": []}):
        info = export_image_masks(
            same["images"], same["masks"], tmp_path / "v2",
            item_steps=given, overwrite=True
        )
        assert info["format_version"] == 2
        assert "steps" not in info["entries"]["image"]
    dataset = load_items(tmp_path / "v2")
    first = dataset.split("train")[0]
    raw = same["images"] / f"{first['id']}.png"
    assert np.array_equal(
        transform_entry(dataset, "image", raw), first["image"]
    )


def test_mask_values_kept(tmp_path: Path) -> None:
    """
    16 bit の領域の値が, 大きさを変えても混ざらずに残ることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 値 300 と 1000 の領域 (8 bit に収まらない)
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint16)
    mask[:, 10:20] = 300
    mask[:, 20:] = 1000
    for index in range(3):
        write_png(tmp_path / "images" / f"a{index}.png", image)
        Image.fromarray(mask).save(tmp_path / "masks" / f"a{index}.png")
    info = export_image_masks(
        tmp_path / "images", tmp_path / "masks", tmp_path / "ds",
        item_steps={"image": [{"op": "resize", "width": 15, "height": 7}]}
    )
    assert info["entries"]["mask"]["dtype"] == "uint16"
    assert info["entries"]["mask"]["values"] == [0, 300, 1000]
    item = load_items(tmp_path / "ds").split("train")[0]
    assert item["mask"].shape == (7, 15)
    assert set(np.unique(item["mask"])) == {0, 300, 1000}


def test_step_errors(tmp_path: Path) -> None:
    """
    ファイルに残せない手順, 知らない引数, 誤った値を書き出しの前に
    止めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出しの前に止まり, 何も書かない
    made = write_image_masks(tmp_path / "raw", count=4)
    cases = [
        ({"image": [{"op": "normalize"}]}, "使えない"),
        ({"image": [{"op": "sample_points", "points": 8}]}, "使えない"),
        ({"image": [{"op": "resize", "width": 8, "height": 8,
                     "method": "nearest"}]}, "知らない引数"),
        ({"image": [{"op": "resize", "width": 8}]}, "height"),
        ({"image": [{"op": "resize", "width": 0, "height": 8}]}, "整数"),
        ({"image": [{"op": "resize", "width": "8", "height": 8}]}, "整数"),
        ({"image": [{"op": "resize", "width": True, "height": 8}]}, "整数"),
        ({"mask": RESIZE}, "領域の手順"),
        ({"label": RESIZE}, "渡せません"),
        ({"image": {"op": "resize"}}, "リスト"),
        ([RESIZE], "辞書"),
    ]
    for given, message in cases:
        with pytest.raises(ValueError, match=message):
            export_image_masks(
                made["images"], made["masks"], tmp_path / "ds",
                item_steps=given
            )
        assert not (tmp_path / "ds").exists()

    # 撮影と点群の項目も, 使えない手順は止まる
    with pytest.raises(ValueError, match="使えない"):
        check_entry_steps([{"op": "normalize_cloud"}], "point_cloud")
    with pytest.raises(ValueError, match="使えない"):
        check_entry_steps([{"op": "normalize"}], "capture")
    with pytest.raises(ValueError, match="seed"):
        check_entry_steps(
            [{"op": "sample_points", "points": 8, "seed": -1}],
            "point_cloud"
        )

    # 領域の手順は登録簿にあり, 版 1 の種類には出ない
    assert ITEM_STEPS["resize_mask"].kind == "mask"
    assert {s["op"] for s in describe_item_steps()
            if s["kind"] == "image"} == {
        "resize", "grayscale", "rgb", "normalize"
    }


def test_transform_entry_stops(tmp_path: Path) -> None:
    """
    推論の手順のかけ方が, 知らない手順, 撮影, 違う版ではっきり止まる
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 新しい ML-EDA で足した手順 (読む側が知らない) は止まる
    made = write_image_masks(tmp_path / "raw", count=4)
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds",
        item_steps={"image": RESIZE}
    )
    newer = {**info, "entries": {
        **info["entries"],
        "image": {**info["entries"]["image"],
                  "steps": [{"op": "sharpen", "amount": 2}]},
    }}
    with pytest.raises(ValueError, match="sharpen"):
        transform_entry(newer, "image", np.zeros((4, 4, 3), np.uint8))

    # 無い項目, 撮影, 版 1 と知らない版
    with pytest.raises(ValueError, match="label"):
        transform_entry(info, "label", np.zeros((4, 4)))
    capture = {"format_version": 3, "entries": {
        "capture": {"kind": "capture", "placement": "directory"},
    }}
    with pytest.raises(ValueError, match="normalize_capture"):
        transform_entry(capture, "capture", tmp_path)
    for version in (1, 4):
        with pytest.raises(ValueError, match="手順をかけられません"):
            transform_entry(
                {**info, "format_version": version}, "image",
                np.zeros((4, 4, 3), np.uint8)
            )

    # 版 3 の説明はファイルにも書いてある
    written = yaml.safe_load(
        (tmp_path / "ds" / "dataset.yaml").read_text(encoding="utf-8")
    )
    assert written["format_version"] == 3
    assert written["entries"]["image"]["steps"] == RESIZE


def test_capture_steps(tmp_path: Path) -> None:
    """
    撮影の画像を縮めると, 深度と内部パラメータも合わせて直り, 深度から
    戻した点が正解の点群に重なることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 32 x 24 の撮影を 16 x 12 に縮め, 正解の点群を 256 点に
    made = write_shape_captures(
        tmp_path / "raw", views=4, size=(24, 32), points=500
    )
    steps = {
        "capture": [{"op": "resize", "width": 16, "height": 12}],
        "cloud": [{"op": "sample_points", "points": 256}],
    }
    info = export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "small",
        item_steps=steps, split={"method": "none"}
    )
    export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "full",
        split={"method": "none"}
    )
    assert info["format_version"] == 3
    assert info["entries"]["capture"]["steps"] == steps["capture"]
    assert info["entries"]["cloud"]["steps"] == steps["cloud"]
    assert info["entries"]["capture"]["shape"] == [4, 12, 16]
    assert info["entries"]["cloud"]["shape"] == [256, 3]

    # 画像, 深度, 信頼度は同じ大きさ. 内部パラメータは半分
    small = load_items(tmp_path / "small").split("train")
    full = load_items(tmp_path / "full").split("train")
    for reduced, original in zip(small, full):
        capture = reduced["capture"]
        assert capture["images"].shape == (4, 12, 16, 3)
        assert capture["depth"].shape == (4, 12, 16)
        assert capture["confidence"].shape == (4, 12, 16)
        for key in ("fl_x", "fl_y", "cx", "cy"):
            assert capture["intrinsics"][key] == pytest.approx(
                original["capture"]["intrinsics"][key] / 2
            )
        assert (capture["intrinsics"]["w"], capture["intrinsics"]["h"]) == (
            16, 12
        )
        assert reduced["cloud"].shape == (256, 3)

        # 深度は値を混ぜない (元の深度にある値だけ)
        assert set(np.unique(capture["depth"])) <= set(
            np.unique(original["capture"]["depth"])
        )

        # 深度から戻した点は, 縮めない場合と同じほど正解に近い
        cloud = original["cloud"]
        near = mean_distance(unproject(capture), cloud)
        base = mean_distance(unproject(original["capture"]), cloud)
        assert base < 0.02
        assert near < base * 2 + 0.005

    # 推論: 元の撮影に同じ正規化と手順をかけると, 書き出しと同じ
    first = small[0]
    source = made["captures"] / first["id"]
    summary = normalize_capture(
        source, tmp_path / "new", steps=info["entries"]["capture"]["steps"]
    )
    assert (summary["height"], summary["width"]) == (12, 16)
    again = read_capture(tmp_path / "new")
    assert np.array_equal(again["images"], first["capture"]["images"])
    assert np.array_equal(again["depth"], first["capture"]["depth"])
    assert np.allclose(again["poses"], first["capture"]["poses"])
    assert again["intrinsics"] == first["capture"]["intrinsics"]


def test_capture_gray_steps(tmp_path: Path) -> None:
    """
    撮影の色の手順だけでは大きさと内部パラメータが変わらないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 灰色にするだけ
    made = write_shape_captures(
        tmp_path / "raw", views=3, size=(12, 16), points=100
    )
    export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "gray",
        item_steps={"capture": ["grayscale"]}, split={"method": "none"}
    )
    capture = load_items(tmp_path / "gray").split("train")[0]["capture"]
    assert capture["images"].shape[1:] == (12, 16, 1)
    assert capture["depth"].shape[1:] == (12, 16)
    assert capture["intrinsics"]["w"] == 16


def test_scale_helpers(tmp_path: Path) -> None:
    """
    内部パラメータが画素の中心の決まりを保ち, 深度の大きさが画像との比を
    保つことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 画素の中心が 0 の決まり: 端からの座標で倍率をかける
    sizes = ((24, 32), (12, 16))
    holder = {"fl_x": 30.0, "fl_y": 28.0, "cx": 15.5, "cy": 11.5,
              "w": 32, "h": 24}
    scale_intrinsics(
        holder, {"coordinate_convention": {"pixel_center": 0}}, sizes
    )
    assert holder["cx"] == pytest.approx((15.5 + 0.5) / 2 - 0.5)
    assert holder["fl_y"] == pytest.approx(14.0)
    assert (holder["w"], holder["h"]) == (16, 12)

    # 画素の中心が 0.5 なら倍率をかけるだけ
    holder = {"cx": 16.0, "cy": 12.0}
    scale_intrinsics(holder, {}, sizes)
    assert (holder["cx"], holder["cy"]) == (8.0, 6.0)

    # 画像より小さい深度は, 同じ比で縮める (16 bit の値はそのまま)
    depth = (np.arange(48, dtype=np.uint16) * 1000).reshape(6, 8)
    Image.fromarray(depth).save(tmp_path / "d.png")
    scale_nearest(tmp_path / "d.png", sizes)
    with Image.open(tmp_path / "d.png") as opened:
        resized = np.array(opened)
    assert resized.shape == (3, 4)
    assert set(np.unique(resized)) <= set(np.unique(depth))
