import json
from pathlib import Path

import numpy as np
import pytest

from ml_eda import detect, export_dataset, load_data, load_dataset, profile

PIL = pytest.importorskip("PIL.Image")


def write_images(root: Path) -> None:
    """
    2 クラスの画像を, 大きさを混ぜて保存する関数

    Parameters
    ----------
    root: Path
        保存先のディレクトリ (<クラス>/<画像> の形)

    Returns
    ----------
    None
    """
    # 明るい画像 (cat) と暗い画像 (dog), 1 枚だけ大きさが違う
    rng = np.random.default_rng(0)
    for label, base in (("cat", 200), ("dog", 50)):
        (root / label).mkdir(parents=True)
        for index in range(6):
            size = (12, 10) if index == 0 and label == "cat" else (8, 8)
            pixels = np.clip(
                base + rng.normal(0, 10, (*size[::-1], 3)), 0, 255
            ).astype(np.uint8)
            PIL.fromarray(pixels).save(root / label / f"{index}.png")


def test_images(tmp_path: Path) -> None:
    """
    画像のディレクトリを読み, 分析し, 大きさをそろえて書き出せることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # クラスのフォルダからラベル付きで読む
    write_images(tmp_path / "pets")
    assert detect(tmp_path / "pets")[0]["loader"] == "image_folder"
    data = load_data(tmp_path / "pets")
    assert data.kind == "image" and len(data.frame) == 12
    assert set(data.frame["label"]) == {"cat", "dog"}
    assert data.load_item(0).shape == (10, 12, 3)

    # 大きさのばらつきの気付き
    result = profile(data, target="label")
    json.dumps(result)
    codes = {item["code"] for item in result["findings"]}
    assert "mixed_sizes" in codes and "id_like" not in codes
    assert result["items"]["sizes"][0] == {
        "width": 8, "height": 8, "count": 11
    }

    # 大きさをそろえず書き出すと理由付きの誤り
    recipe = {
        "target": "label", "task": "classification",
        "split": {"method": "stratified", "valid": 0.34},
    }
    with pytest.raises(ValueError, match="resize"):
        export_dataset(data, recipe, tmp_path / "bad")

    # そろえて正規化すると (件数, 3, 8, 8) の配列になる
    recipe["item_steps"] = [
        {"op": "resize", "width": 8, "height": 8}, {"op": "normalize"}
    ]
    info = export_dataset(data, recipe, tmp_path / "ds")
    assert info["splits"]["train"]["shape"] == [8, 3, 8, 8]
    dataset = load_dataset(tmp_path / "ds")
    x, y = dataset.arrays("valid")
    assert x.shape == (4, 3, 8, 8) and set(y) == {0, 1}
    assert dataset.transform_item(data.load_item(0)).shape == (3, 8, 8)


def test_text(tmp_path: Path) -> None:
    """
    1 行 1 文のファイルとクラスのフォルダの文を読み, 語彙で番号の並びに
    できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 行 1 文
    (tmp_path / "lines.txt").write_text(
        "the cat sat\nthe dog ran fast\n\nbirds fly\n", encoding="utf-8"
    )
    data = load_data(tmp_path / "lines.txt")
    assert data.kind == "text" and len(data.frame) == 3

    # クラスのフォルダ
    for label, words in (("pos", "good great"), ("neg", "bad awful")):
        (tmp_path / "reviews" / label).mkdir(parents=True)
        for index in range(5):
            (tmp_path / "reviews" / label / f"{index}.txt").write_text(
                f"this is {words} {index}", encoding="utf-8"
            )
    data = load_data(tmp_path / "reviews")
    assert data.kind == "text" and set(data.frame["label"]) == {"neg", "pos"}
    assert profile(data, target="label")["columns"]["text"]["type"] in (
        "text", "categorical", "id"
    )

    # 学習用で語彙を作り, 長さ 6 の番号の並びにする
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [{"op": "vocabulary", "length": 6}],
        "split": {"method": "stratified", "valid": 0.2},
    }
    export_dataset(data, recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    x, _ = dataset.arrays("train")
    assert x.shape == (8, 6) and x.dtype == np.int64
    assert dataset.transform_item("totally unknown words").tolist()[:3] == [
        1, 1, 1
    ]


def write_ply(path: Path, points: np.ndarray, binary: bool) -> None:
    """
    色付きの点群を PLY で保存する関数

    Parameters
    ----------
    path: Path
        保存先
    points: np.ndarray
        (N, 3) の座標
    binary: bool
        バイナリで書くか (False なら ASCII)

    Returns
    ----------
    None
    """
    # 座標と色の頂点
    form = "binary_little_endian" if binary else "ascii"
    header = (
        f"ply\nformat {form} 1.0\nelement vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    if binary:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        table = np.zeros(len(points), dtype=dtype)
        for k, axis in enumerate("xyz"):
            table[axis] = points[:, k]
        for k, name in enumerate(("red", "green", "blue")):
            table[name] = colors[:, k]
        path.write_bytes(header.encode() + table.tobytes())
    else:
        rows = "\n".join(
            " ".join(map(str, [*p, *c])) for p, c in zip(points, colors)
        )
        path.write_text(header + rows + "\n", encoding="ascii")


def test_pointclouds(tmp_path: Path) -> None:
    """
    PLY (ASCII とバイナリ), PCD, xyz, npy の点群を読み, 点の数をそろえて
    書き出せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 球 (sphere) と立方体 (cube) の点群をクラスのフォルダに
    rng = np.random.default_rng(0)
    for label in ("sphere", "cube"):
        (tmp_path / "shapes" / label).mkdir(parents=True)
        for index in range(4):
            count = 200 + 50 * index
            points = rng.normal(0, 1, (count, 3)) if label == "sphere" else (
                rng.uniform(-1, 1, (count, 3))
            )
            write_ply(
                tmp_path / "shapes" / label / f"{index}.ply",
                points.astype(np.float32), binary=index % 2 == 1
            )
    data = load_data(tmp_path / "shapes")
    assert data.kind == "pointcloud" and len(data.frame) == 8
    cloud = data.load_item(1)
    assert cloud["points"].shape == (250, 3) and cloud["colors"] is not None
    result = profile(data, target="label")
    codes = {item["code"] for item in result["findings"]}
    assert "mixed_points" in codes and "high_correlation" not in codes

    # 1 つのファイルの形式 (PCD, xyz, npy)
    (tmp_path / "a.pcd").write_text(
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        "WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n0 0 0\n1 0 0\n0 1 0\n",
        encoding="ascii"
    )
    (tmp_path / "a.xyz").write_text("0 0 0\n1 1 1\n2 2 2\n", encoding="ascii")
    np.save(tmp_path / "lidar_points.npy", rng.normal(0, 1, (100, 4)))
    for name in ("a.pcd", "a.xyz", "lidar_points.npy"):
        assert load_data(tmp_path / name).kind == "pointcloud", name

    # 点の数を 128 にそろえ, 大きさを正規化して書き出す
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [
            {"op": "sample_points", "points": 128},
            {"op": "normalize_cloud"},
        ],
        "split": {"method": "stratified", "valid": 0.25},
    }
    export_dataset(data, recipe, tmp_path / "ds")
    x, y = load_dataset(tmp_path / "ds").arrays("train")
    assert x.shape == (6, 128, 3) and len(y) == 6
    assert np.abs(np.linalg.norm(x, axis=2).max() - 1) < 1e-5


def test_review_fixes(tmp_path: Path) -> None:
    """
    16 ビットの画像, 無作為な当てはめの見本, 灰色だけの形, 灰色とカラーの
    混在, 隠しの名前のディレクトリ, 点群と文の読み方を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    from ml_eda.itemprep import FIT_SAMPLE, tokenize
    from ml_eda.media import parse_pcd, parse_xyz, read_image

    # 16 ビットの画像は値の範囲を保って 0 から 255 に縮める
    wide = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000)
    PIL.fromarray(wide).save(tmp_path / "w16.png")
    image = read_image(tmp_path / "w16.png")
    assert image.ndim == 2 and len(np.unique(image)) > 30

    # 隠しの名前のディレクトリでも読む. 明るさの違う 2 クラスを多く置き,
    # 正規化の平均は学習用の両方のクラスから求める
    root = tmp_path / ".pets"
    for label, value in (("a", 230), ("b", 20)):
        (root / label).mkdir(parents=True)
        for index in range(FIT_SAMPLE // 2 + 50):
            pixels = np.full((4, 4, 3), value, dtype=np.uint8)
            PIL.fromarray(pixels).save(root / label / f"{index}.png")
    data = load_data(root)
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [{"op": "normalize"}],
        "split": {"method": "stratified", "valid": 0.2},
    }
    from ml_eda import prepare

    mean = prepare(data, recipe).recipe["item_steps"][0]["state"]["mean"][0]
    assert 0.3 < mean < 0.7

    # 灰色だけでも (件数, 1, 高さ, 幅) で 0 から 1
    recipe["item_steps"] = [{"op": "grayscale"}]
    array = prepare(data, recipe).arrays["train"]
    assert array.shape[1:] == (1, 4, 4) and array.max() <= 1.0

    # 灰色とカラーの混在は rgb でそろう
    PIL.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
        root / "a" / "gray.png"
    )
    data = load_data(root)
    recipe["item_steps"] = [{"op": "normalize"}]
    with pytest.raises(ValueError):
        prepare(data, recipe)
    recipe["item_steps"] = [{"op": "rgb"}, {"op": "normalize"}]
    assert prepare(data, recipe).arrays["train"].shape[1] == 3

    # PCD の整数の rgb と, .pts の強度の後の色
    (tmp_path / "u.pcd").write_text(
        "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
        f"DATA ascii\n0 0 0 {255 << 16 | 128 << 8 | 7}\n",
        encoding="ascii"
    )
    assert parse_pcd(tmp_path / "u.pcd")["colors"].tolist() == [[255, 128, 7]]
    (tmp_path / "a.pts").write_text("1\n0 0 0 0.5 10 20 30\n", "ascii")
    assert parse_xyz(tmp_path / "a.pts")["colors"].tolist() == [[10, 20, 30]]

    # 大文字と記号付きの文字の語は切らない. 日本語は 1 文字ずつ
    assert tokenize("Hello café", {"lowercase": False}) == ["Hello", "café"]
    assert tokenize("猫が好き", {}) == ["猫", "が", "好", "き"]


def test_item_checks(tmp_path: Path) -> None:
    """
    中身を読まずに, 中身の手順の誤りと書き出しで失敗しそうなことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    from ml_eda import prepare

    # 大きさの混ざる画像と, 読めない 1 枚
    write_images(tmp_path / "pets")
    (tmp_path / "pets" / "cat" / "bad.png").write_bytes(b"not an image")
    data = load_data(tmp_path / "pets")
    recipe = {
        "target": "label", "task": "classification",
        "split": {"method": "stratified", "valid": 0.25},
    }

    # 誤った手順は中身を読まずに誤り
    for steps, message in (
        ("resize", "リスト"),
        ([{"op": "sample_points", "points": 8}], "使えない"),
        ([{"op": "resize", "width": 8}], "height"),
    ):
        with pytest.raises(ValueError, match=message):
            prepare(data, {**recipe, "item_steps": steps}, read_items=False)

    # 読めない件と大きさのばらつきの注意. 配列は作らない
    prepared = prepare(
        data, {**recipe, "item_steps": [{"op": "normalize"}]},
        read_items=False
    )
    assert prepared.arrays == {}
    assert len(prepared.notes) == 2
    assert "1 件" in prepared.notes[0] and "resize" in prepared.notes[1]

    # 読めない件を除き, 大きさをそろえれば注意は無い
    recipe["steps"] = [
        {"op": "filter_rows", "column": "width", "compare": "notna"}
    ]
    recipe["item_steps"] = [{"op": "resize", "width": 8, "height": 8}]
    assert prepare(data, recipe, read_items=False).notes == []

    # タスクを決めていなければ注意
    recipe["task"] = "none"
    assert "タスク" in prepare(data, recipe, read_items=False).notes[0]


def test_preview_items(tmp_path: Path) -> None:
    """
    中身の手順を少ない見本で当てはめて, 数件の前と後を返すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    from ml_eda import preview_items

    # 画像は前が元の配列, 後が (チャンネル, 高さ, 幅)
    write_images(tmp_path / "pets")
    data = load_data(tmp_path / "pets")
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [
            {"op": "resize", "width": 6, "height": 6}, {"op": "normalize"}
        ],
        "split": {"method": "stratified", "valid": 0.25},
    }
    result = preview_items(data, recipe, count=3, fit_limit=4)
    assert len(result["items"]) == 3 and result["fit_rows"] == 4
    item = result["items"][0]
    assert item["after"].shape == (3, 6, 6) and item["label"] in ("cat", "dog")
    assert "state" in result["steps"][1]

    # 表は中身を持たないので誤り
    import pandas as pd

    pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ValueError, match="中身"):
        preview_items(load_data(tmp_path / "t.csv"), {"steps": []})
