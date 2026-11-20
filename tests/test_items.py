import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_eda import (
    capture_shift, export_capture_clouds, export_dataset,
    export_image_masks, load_data, load_dataset, load_items,
    normalize_capture, profile
)
from ml_eda.items import check_entry_names
from ml_eda.media import read_cloud
from ml_eda.synthetic import (
    write_image_masks, write_png, write_shape_captures
)

Image = pytest.importorskip("PIL.Image")

# 球と箱ごとのグループ
SHAPE_GROUPS = {
    "sphere_a": "sphere", "sphere_b": "sphere", "box_a": "box",
    "box_b": "box",
}


def rewrite_info(path: Path, **changes: object) -> None:
    """
    書き出した dataset.yaml の一部を書き換える関数

    Parameters
    ----------
    path: Path
        データセットのディレクトリ
    **changes: object
        書き換える項目

    Returns
    ----------
    None
    """
    # 読んで書き換えて書き戻し
    file = path / "dataset.yaml"
    info = yaml.safe_load(file.read_text(encoding="utf-8"))
    info.update(changes)
    file.write_text(yaml.safe_dump(info, allow_unicode=True), "utf-8")


def test_image_masks(tmp_path: Path) -> None:
    """
    画像と領域の対を版 2 で書き出し, load_items で同じ値が読めることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 見本と, 領域の無い画像 1 枚
    made = write_image_masks(tmp_path / "raw", count=8)
    shutil.copyfile(
        made["images"] / "sample_000.png", made["images"] / "lonely.png"
    )
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds",
        split={"valid": 0.25, "test": 0.25}, classes=made["classes"],
        origin={"type": "annotation", "name": "人の塗り分け"}
    )

    # 説明: 版 2, 受け持ちと正解の種類, task は無い
    assert info["format_version"] == 2
    assert info["role"] == "predict"
    assert info["target"]["kind"] == "mask"
    assert info["target"]["origin"]["type"] == "annotation"
    assert "task" not in info
    assert info["entries"]["image"]["shape"] == [24, 32, 3]
    assert info["entries"]["mask"]["values"] == [0, 1, 2]
    assert info["entries"]["mask"]["classes"][2]["name"] == "square"
    assert info["entries"]["image"]["pattern"] == (
        "items/{index:06d}/image{suffix}"
    )
    assert {k: v["items"] for k, v in info["splits"].items()} == {
        "train": 4, "valid": 2, "test": 2,
    }
    assert any("lonely" in note for note in info["notes"])
    written = yaml.safe_load(
        (tmp_path / "ds" / "dataset.yaml").read_text(encoding="utf-8")
    )
    assert written == info

    # 読み込み: 配列と場所. 領域は元の値のまま
    dataset = load_items(tmp_path / "ds")
    assert dataset.splits == ["train", "valid", "test"]
    seen = []
    for name in dataset.splits:
        for item in dataset.split(name):
            original = np.asarray(
                Image.open(made["masks"] / f"{item['id']}.png")
            )
            assert item["split"] == name
            assert item["image"].shape == (24, 32, 3)
            assert item["image"].dtype == np.uint8
            assert item["mask"].dtype == np.uint8
            np.testing.assert_array_equal(item["mask"], original)
            seen.append(item["id"])
    assert sorted(seen) == made["names"]
    paths = dataset.split("valid", read=False)
    assert len(paths) == 2
    assert Path(paths[-1]["image"]).is_file()
    with pytest.raises(IndexError):
        paths[2]
    with pytest.raises(ValueError, match="unlabeled"):
        dataset.split("unlabeled")


def test_version_stops(tmp_path: Path) -> None:
    """
    版 1 と版 2 の読み込みが, 違う版を見たらはっきり止まることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 2 を load_dataset で読むと load_items へ案内
    made = write_image_masks(tmp_path / "raw", count=4)
    export_image_masks(made["images"], made["masks"], tmp_path / "v2")
    with pytest.raises(ValueError, match="load_items"):
        load_dataset(tmp_path / "v2")

    # 版 1 を load_items で読むと load_dataset へ案内
    pd.DataFrame({"x": range(10), "y": [0, 1] * 5}).to_csv(
        tmp_path / "t.csv", index=False
    )
    export_dataset(
        load_data(tmp_path / "t.csv"),
        {"target": "y", "task": "classification", "steps": []},
        tmp_path / "v1"
    )
    with pytest.raises(ValueError, match="load_dataset"):
        load_items(tmp_path / "v1")

    # 知らない版と知らない置き方は読まない
    rewrite_info(tmp_path / "v2", format_version=3)
    with pytest.raises(ValueError, match="読めません"):
        load_items(tmp_path / "v2")
    with pytest.raises(ValueError, match="読めません"):
        load_dataset(tmp_path / "v2")
    rewrite_info(tmp_path / "v2", format_version=2, entries={
        "user": {"kind": "id", "placement": "lookup"},
    })
    with pytest.raises(ValueError, match="lookup"):
        load_items(tmp_path / "v2")

    # 1 件の辞書の決まった鍵と同じ名前の項目は読まない (書き出しも)
    rewrite_info(tmp_path / "v2", entries={
        "split": {"kind": "image", "placement": "file"},
    })
    with pytest.raises(ValueError, match="split"):
        load_items(tmp_path / "v2")
    with pytest.raises(ValueError, match="group"):
        check_entry_names({"image": {}, "group": {}})

    # 版 1 の書き出しは版 2 のデータセットを置き換えられる
    export_dataset(
        load_data(tmp_path / "t.csv"),
        {"target": "y", "task": "classification", "steps": []},
        tmp_path / "v2", overwrite=True
    )
    assert load_dataset(tmp_path / "v2").info["format_version"] == 1


def test_image_mask_errors(tmp_path: Path) -> None:
    """
    組の決まらないファイル, カラーの領域, 大きさの違い, 分割の誤り,
    外を指す場所を誤りにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ stem が 2 つ
    made = write_image_masks(tmp_path / "a", count=4)
    Image.open(made["images"] / "sample_000.png").save(
        made["images"] / "sample_000.jpg"
    )
    with pytest.raises(ValueError, match="2 つ"):
        export_image_masks(made["images"], made["masks"], tmp_path / "x")

    # カラーの領域と, 大きさの違う領域
    made = write_image_masks(tmp_path / "b", count=4)
    write_png(made["masks"] / "sample_001.png",
              np.zeros((24, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB"):
        export_image_masks(made["images"], made["masks"], tmp_path / "x")
    write_png(made["masks"] / "sample_001.png",
              np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="大きさ"):
        export_image_masks(made["images"], made["masks"], tmp_path / "x")
    assert not (tmp_path / "x").exists()
    assert not (tmp_path / ".x.writing").exists()

    # EXIF の向きの付いた画像 (読むと向きが変わり, 領域とずれる)
    exif = Image.Exif()
    exif[0x0112] = 3
    Image.open(made["images"] / "sample_001.png").convert("RGB").save(
        made["images"] / "sample_001.jpg", exif=exif
    )
    (made["images"] / "sample_001.png").unlink()
    write_png(made["masks"] / "sample_001.png",
              np.zeros((24, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="EXIF の向き \\(3\\)"):
        export_image_masks(made["images"], made["masks"], tmp_path / "x")

    # 分割の誤り, 正解の出どころの誤り
    made = write_image_masks(tmp_path / "c", count=4)
    for options, message in (
        ({"split": {"method": "stratified"}}, "stratified"),
        ({"split": {"method": "group"}}, "groups"),
        ({"groups": {"sample_000": "g"}}, "groups に無い"),
        ({"origin": {"type": "magic"}}, "magic"),
    ):
        with pytest.raises(ValueError, match=message):
            export_image_masks(
                made["images"], made["masks"], tmp_path / "x", **options
            )

    # データセットでないディレクトリは置き換えない
    (tmp_path / "plain").mkdir()
    with pytest.raises(FileExistsError):
        export_image_masks(
            made["images"], made["masks"], tmp_path / "plain",
            overwrite=True
        )

    # 表の場所が外を指せば読まない
    export_image_masks(made["images"], made["masks"], tmp_path / "ds")
    table = pd.read_parquet(tmp_path / "ds" / "items.parquet") if (
        tmp_path / "ds" / "items.parquet"
    ).is_file() else pd.read_csv(tmp_path / "ds" / "items.csv")
    table.loc[0, "image"] = "../c/images/sample_000.png"
    table.to_csv(tmp_path / "ds" / "items.csv", index=False)
    rewrite_info(tmp_path / "ds", items="items.csv")
    dataset = load_items(tmp_path / "ds")
    with pytest.raises(ValueError, match="外"):
        dataset.split(dataset.items.loc[0, "split"])[0]


def test_group_split(tmp_path: Path) -> None:
    """
    グループを渡すと同じグループが同じ分割にまとまり, 無作為の分割では
    またがりを注意にすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 4 つのグループ (2 件ずつ)
    made = write_image_masks(tmp_path / "raw", count=8)
    groups = {name: f"g{i // 2}" for i, name in enumerate(made["names"])}
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds", groups=groups,
        split={"valid": 0.25, "test": 0.25, "seed": 1}
    )
    assert info["split"]["method"] == "group"
    table = load_items(tmp_path / "ds").items
    assert table.groupby("group")["split"].nunique().max() == 1
    assert set(table["group"]) == {"g0", "g1", "g2", "g3"}

    # 無作為ではまたがりを注意
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds2", groups=groups,
        split={"method": "random", "valid": 0.5, "seed": 0}
    )
    assert any("グループ" in note for note in info["notes"])

    # グループが大きく検証用が 0 件になれば注意. none は割合を 0 に
    halves = {name: f"h{i // 4}" for i, name in enumerate(made["names"])}
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds3", groups=halves,
        split={"valid": 0.3}, classes={"0": "bg", "1": "circle"}
    )
    assert info["splits"] == {"train": {"items": 8}}
    assert any("0 件" in note for note in info["notes"])
    assert info["entries"]["mask"]["classes"][1] == {
        "value": 1, "name": "circle",
    }
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds4",
        split={"method": "none"}
    )
    assert (info["split"]["valid"], info["split"]["test"]) == (0.0, 0.0)
    assert not any("0 件" in note for note in info["notes"])


def test_capture_clouds(tmp_path: Path) -> None:
    """
    撮影と点群の対を版 2 で書き出し, 姿勢と点群に同じ平行移動がかかり,
    実寸が残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 球 2 つと箱 2 つ (点群の単位はミリで渡す)
    made = write_shape_captures(tmp_path / "raw", views=8, points=3000)
    for cloud in made["clouds"].glob("*.ply"):
        points = read_cloud(cloud)["points"]
        cloud.unlink()
        np.save(made["clouds"] / f"{cloud.stem}.npy", points * 1000.0)
    (made["clouds"] / "stray.npy").write_bytes(
        (made["clouds"] / "box_a.npy").read_bytes()
    )
    info = export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "ds",
        groups=SHAPE_GROUPS, split={"valid": 0.5},
        origin={"type": "process", "name": "外の復元", "settings": {"k": 1}},
        cloud_units="millimeter"
    )
    assert info["format_version"] == 2
    assert info["target"] == {
        "kind": "point_cloud", "entries": ["cloud"],
        "origin": {"type": "process", "name": "外の復元",
                   "settings": {"k": 1}},
    }
    assert "task" not in info
    assert info["entries"]["capture"]["shape"] == [8, 24, 32]
    assert info["entries"]["cloud"]["shape"] == [3000, 3]
    assert info["normalization"]["translation_only"] is True
    assert any("stray" in note for note in info["notes"])

    # 1 件ずつ: 元の座標 = 正規化した座標 - shift, 物の中心は原点の近く
    dataset = load_items(tmp_path / "ds")
    table = dataset.items
    assert table.groupby("group")["split"].nunique().max() == 1
    shapes = {shape["name"]: shape for shape in made["shapes"]}
    count = 0
    for name in dataset.splits:
        for item in dataset.split(name):
            count += 1
            shape = shapes[item["id"]]
            row = table[table["id"] == item["id"]].iloc[0]
            shift = row[["shift_x", "shift_y", "shift_z"]].to_numpy(float)
            np.testing.assert_allclose(
                -shift, shape["center"], atol=1e-6
            )
            cloud = item["cloud"]
            assert cloud.dtype == np.float32 and cloud.shape == (3000, 3)
            assert np.abs(cloud.mean(axis=0)).max() < 0.02
            check_depth_on_cloud(item["capture"], cloud)
    assert count == 4

    # 書き出した撮影は, 多視点の撮影の読み込みでそのまま読める
    first = dataset.split("train", read=False)[0]["capture"]
    data = load_data(first, loader="capture")
    assert data.frame["frames"].iloc[0] == 8
    assert data.frame["view_gap"].iloc[0] == pytest.approx(45.0)


def check_depth_on_cloud(capture: dict, cloud: np.ndarray) -> None:
    """
    深度から戻した点が, 正規化した点群の面の上にあることを確かめる関数

    深度はメートル, 姿勢は正規化した世界なので, 実寸と平行移動がそろって
    いれば点群の点の近くに戻る

    Parameters
    ----------
    capture: dict
        load_items で読んだ撮影
    cloud: np.ndarray
        (点の数, 3) の正規化した点群

    Returns
    ----------
    None
    """
    # 各枚の当たった画素をカメラの座標 (OpenGL. -Z が前) から世界へ
    intrinsics = capture["intrinsics"]
    for pose, depth in zip(capture["poses"], capture["depth"]):
        rows, columns = np.nonzero(depth > 0)
        z = depth[rows, columns].astype(np.float64)
        camera = np.stack([
            (columns + 0.5 - intrinsics["cx"]) / intrinsics["fl_x"] * z,
            -(rows + 0.5 - intrinsics["cy"]) / intrinsics["fl_y"] * z,
            -z,
        ], axis=1)
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        gaps = np.linalg.norm(
            world[:, None, :] - cloud[None, :, :], axis=2
        ).min(axis=1)
        assert len(world) > 0
        assert np.max(gaps) < 0.03


def test_capture_device_cloud(tmp_path: Path) -> None:
    """
    撮影が持つ点群にも同じ平行移動がかかり, 画像の無い枚と読めない撮影が
    除かれることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 つ目の撮影に端末の点群 (正解と同じ) と, 画像の無い枚を足す
    made = write_shape_captures(
        tmp_path / "raw", shapes=None, views=6, points=500
    )
    capture = made["captures"] / "sphere_a"
    shutil.copyfile(
        made["clouds"] / "sphere_a.ply", capture / "device.ply"
    )
    transforms = yaml.safe_load(
        (capture / "transforms.json").read_text(encoding="utf-8")
    )
    transforms["point_cloud"] = {"file_path": "device.ply"}
    far = np.array(transforms["frames"][0]["transform_matrix"])
    far[:3, 3] += [5.0, 0.0, 5.0]
    transforms["frames"].append({
        **transforms["frames"][0], "file_path": "images/none.png",
        "transform_matrix": far.tolist(),
    })
    (capture / "transforms.json").write_text(
        json.dumps(transforms), encoding="utf-8"
    )

    # 2 つ目の撮影の点群は無いファイルを指す
    other = made["captures"] / "sphere_b" / "transforms.json"
    described = json.loads(other.read_text(encoding="utf-8"))
    described["point_cloud"] = {"file_path": "../outside.ply"}
    other.write_text(json.dumps(described), encoding="utf-8")
    (made["captures"] / "broken").mkdir()
    (made["captures"] / "broken" / "transforms.json").write_text(
        "{", encoding="utf-8"
    )
    info = export_capture_clouds(
        made["captures"], made["clouds"], tmp_path / "ds", split={
            "method": "none",
        }
    )
    assert info["splits"] == {"train": {"items": 4}}
    assert info["target"]["origin"]["type"] == "unknown"
    assert any("broken" in note for note in info["notes"])
    assert any("1 枚" in note for note in info["notes"])
    assert any("outside.ply" in note for note in info["notes"])

    # 端末の点群と正解の点群は同じ座標になる
    dataset = load_items(tmp_path / "ds")
    item = next(
        entry for entry in dataset.split("train") if entry["id"] == "sphere_a"
    )
    copied = Path(
        dataset.split("train", read=False)[
            list(dataset.items["id"]).index("sphere_a")
        ]["capture"]
    )
    device = read_cloud(copied / "point_cloud.ply")["points"]
    np.testing.assert_allclose(device, item["cloud"], atol=1e-5)

    # 中心は画像のある枚だけから. 書き出した撮影から求め直すと 0,
    # 元の撮影から求めると表の shift (推論で同じ正規化をかけられる)
    row = dataset.items[dataset.items["id"] == "sphere_a"].iloc[0]
    shift = row[["shift_x", "shift_y", "shift_z"]].to_numpy(float)
    np.testing.assert_allclose(capture_shift(capture), shift, atol=1e-9)
    np.testing.assert_allclose(capture_shift(copied), 0.0, atol=1e-9)
    summary = normalize_capture(capture, tmp_path / "again")
    np.testing.assert_allclose(summary["shift"], shift, atol=1e-9)
    assert summary["frames"] == 6
    np.testing.assert_allclose(capture_shift(tmp_path / "again"), 0.0,
                               atol=1e-9)
    assert item["capture"]["poses"].shape == (6, 4, 4)


def test_shape_helpers() -> None:
    """
    説明の形とチャンネル数の補助が, 分からない値と色の形式を扱うことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 分からない次元 (w, h の無い撮影) とそろわない次元は None
    from ml_eda.items import common_shape, image_channels

    assert common_shape([(8, None, 32), (8, None, 32)]) == [8, None, 32]
    assert common_shape([(8, 24), (6, 24)]) == [None, 24]

    # read_image と同じチャンネル数
    assert [image_channels(mode) for mode in (
        "L", "I;16", "P", "RGB", "LA", "RGBA", "CMYK", None
    )] == [1, 1, 3, 3, 4, 4, 3, None]


def test_mask_suffix(tmp_path: Path) -> None:
    """
    領域のファイルの名前の終わり (_mask) を除いて組にすることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # a.png と a_mask.png の組
    made = write_image_masks(tmp_path / "raw", count=4)
    for mask in list(made["masks"].glob("*.png")):
        mask.rename(mask.with_name(f"{mask.stem}_mask.png"))
    with pytest.raises(ValueError, match="組がありません"):
        export_image_masks(made["images"], made["masks"], tmp_path / "x")
    info = export_image_masks(
        made["images"], made["masks"], tmp_path / "ds", mask_suffix="_mask",
        split={"method": "none"}
    )
    assert info["splits"] == {"train": {"items": 4}}
    assert sorted(load_items(tmp_path / "ds").items["id"]) == made["names"]


def test_profile_shape_captures(tmp_path: Path) -> None:
    """
    合成の撮影 (方位の隙間が浮動小数の誤差だけ違う) を分析できることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    made = write_shape_captures(tmp_path / "raw")
    result = profile(load_data(made["captures"], loader="capture"))
    histogram = result["columns"]["view_gap"]["histogram"]
    assert sum(histogram["counts"]) == 4
