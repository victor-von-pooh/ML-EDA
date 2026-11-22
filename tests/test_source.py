import importlib.util
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import Data, export_dataset, load_data, load_dataset
from ml_eda.synthetic import write_shape_captures

# 種類ごとの読み込みの名前と拡張子
KINDS = {
    "image": ("image_folder", ".png"),
    "pointcloud": ("pointcloud", ".xyz"),
    "audio": ("audio_folder", ".wav"),
}


def write_item(path: Path, kind: str) -> None:
    """
    種類に合わせた小さな 1 件のファイルを書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    kind: str
        種類 (image, pointcloud, audio)

    Returns
    ----------
    None
    """
    # 4 x 4 の画像 (Pillow が要る), 3 点の点群, 100 標本の音声
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "image":
        image = pytest.importorskip("PIL.Image")
        image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(path)
    elif kind == "pointcloud":
        path.write_text("0 0 0\n1 1 1\n0 1 0\n", encoding="utf-8")
    else:
        numbers = (np.sin(np.arange(100)) * 1000).astype("<i2")
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(numbers.tobytes())


def write_folder(root: Path, kind: str) -> list[str]:
    """
    <クラス>/<ファイル> の形のディレクトリを書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ
    kind: str
        種類 (image, pointcloud, audio)

    Returns
    ----------
    places: list[str]
        root からの相対の場所 (名前の順)
    """
    # 2 クラスに 2 件ずつ
    suffix = KINDS[kind][1]
    places = [f"{label}/{index}{suffix}" for label in "ab" for index in (0, 1)]
    for place in places:
        write_item(root / place, kind)

    return places


@pytest.mark.parametrize("kind", list(KINDS))
def test_folder_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """
    ディレクトリを相対の場所で読んでも, root が絶対パスのそのディレクトリ
    になり, path がそこからの相対になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        今のディレクトリの変更
    kind: str
        種類

    Returns
    ----------
    None
    """
    # 相対の場所で読む
    places = write_folder(tmp_path / "data", kind)
    monkeypatch.chdir(tmp_path)
    data = load_data("data", loader=KINDS[kind][0])
    root = data.source["root"]
    assert isinstance(root, str) and Path(root).is_absolute()
    assert root == str(Path.cwd() / "data")
    assert sorted(data.frame["path"]) == places
    for place in data.frame["path"]:
        assert (Path(root) / place).is_file()

    # 今のディレクトリを変えても root / path で 1 件を読める
    monkeypatch.chdir(tmp_path / "data" / "a")
    assert data.load_item(0) is not None


@pytest.mark.parametrize("kind", list(KINDS))
def test_file_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """
    ファイル 1 つを読むと, root がそのファイルのあるディレクトリになり,
    path がファイルの名前になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        今のディレクトリの変更
    kind: str
        種類

    Returns
    ----------
    None
    """
    # 相対の場所でファイル 1 つを読む
    write_folder(tmp_path / "data", kind)
    monkeypatch.chdir(tmp_path)
    name = f"0{KINDS[kind][1]}"
    data = load_data(Path("data") / "a" / name, loader=KINDS[kind][0])
    assert data.source["root"] == str(Path.cwd() / "data" / "a")
    assert data.frame["path"].tolist() == [name]
    assert data.load_item(0) is not None


def test_capture_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    撮影を並べたディレクトリでは root がそのディレクトリ, 1 回の撮影では
    root がその親になり, path が撮影のディレクトリの名前になることを
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
    # 撮影を並べたディレクトリを相対の場所で読む (Pillow が要る)
    pytest.importorskip("PIL.Image")
    write_shape_captures(tmp_path / "set", views=3, points=100)
    monkeypatch.chdir(tmp_path / "set")
    names = sorted(
        child.name for child in Path("captures").iterdir() if child.is_dir()
    )
    data = load_data("captures", loader="capture")
    assert data.source["root"] == str(Path.cwd() / "captures")
    assert Path(data.source["root"]).is_absolute()
    assert sorted(data.frame["path"]) == names

    # 1 回の撮影を読むと, 根はその親で, 場所は撮影の名前
    single = load_data(Path("captures") / names[0], loader="capture")
    assert single.source["root"] == str(Path.cwd() / "captures")
    assert single.frame["path"].tolist() == [names[0]]

    # 今のディレクトリを変えても 1 件を読める
    monkeypatch.chdir(tmp_path)
    assert len(data.load_item(0)["frames"]) == 3


def test_root_keeps_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    root は渡した場所を絶対パスにしただけで, シンボリックリンクと .. を
    たどらないことを確かめる関数

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
    # リンクの名前と .. は root に残る. 読み込んだ場所の path はたどった後
    write_folder(tmp_path / "data", "image")
    (tmp_path / "link").symlink_to(tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    linked = load_data("link", loader="image_folder")
    assert linked.source["root"] == str(Path.cwd() / "link")
    assert linked.source["path"] == str((tmp_path / "data").resolve())
    climbed = load_data(Path("data") / ".." / "data", loader="image_folder")
    assert climbed.source["root"] == str(Path.cwd() / "data" / ".." / "data")


@pytest.mark.parametrize("place", ["table.csv", "sounds"])
def test_size_and_modified(tmp_path: Path, place: str) -> None:
    """
    source の size と modified が, 読んだ場所の大きさ (int) と更新時刻
    (float) になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    place: str
        読む場所 (表のファイルか, 音声のディレクトリ)

    Returns
    ----------
    None
    """
    # 表のファイルと, ディレクトリ (ディレクトリ自身の値)
    target = tmp_path / place
    if place == "table.csv":
        target.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    else:
        write_folder(target, "audio")
    data = load_data(target)
    stat = target.stat()
    assert type(data.source["size"]) is int
    assert type(data.source["modified"]) is float
    assert data.source["size"] == stat.st_size
    assert data.source["modified"] == stat.st_mtime
    assert data.source["loader"] in ("csv", "audio_folder")
    assert isinstance(data.source["options"], dict)


def test_export_names_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    版 1 の書き出しの source に手元の場所が残らず, 名前だけになることを
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
    # 表のファイル (絶対パス) と, 文と画像のディレクトリ (相対の場所)
    table = tmp_path / "shop.csv"
    rows = "\n".join(f"{i},{'ab'[i % 2]}" for i in range(20))
    table.write_text("x,label\n" + rows + "\n", encoding="utf-8")
    for label in "ab":
        (tmp_path / "reviews" / label).mkdir(parents=True)
        for index in range(4):
            (tmp_path / "reviews" / label / f"{index}.txt").write_text(
                f"{label} {index}", encoding="utf-8"
            )
    monkeypatch.chdir(tmp_path)
    split = {"method": "random", "valid": 0.25}
    settings = {
        "shop.csv": (load_data(table, sep=","), []),
        "reviews": (load_data("reviews"),
                    [{"op": "vocabulary", "length": 4}]),
    }
    if importlib.util.find_spec("PIL") is not None:
        write_folder(tmp_path / "pets", "image")
        settings["pets"] = (load_data(Path("pets") / ".." / "pets"),
                            [{"op": "resize", "width": 2, "height": 2}])
    for name, (data, item_steps) in settings.items():
        recipe = {"target": "label", "task": "classification",
                  "item_steps": item_steps, "split": split}
        info = export_dataset(data, recipe, tmp_path / "out" / name)
        source = info["source"]
        assert source["name"] == name
        assert "path" not in source and "root" not in source
        assert source["paths"] == "name_only" and source["note"]
        assert source["loader"] == data.source["loader"]
        assert source["options"] == data.source["options"]
        assert source["size"] == data.source["size"]

        # 書いた説明のどこにも一時ディレクトリの場所は無い
        text = (tmp_path / "out" / name / "dataset.yaml").read_text(
            encoding="utf-8"
        )
        assert str(tmp_path.resolve()) not in text
        assert str(tmp_path) not in text
        assert load_dataset(tmp_path / "out" / name).info["source"] == source
    assert settings["shop.csv"][0].source["options"] == {"sep": ","}

    # 場所の無い source (手元で作ったデータ) は写すだけ
    frame = pd.DataFrame({"x": range(20), "label": ["a", "b"] * 10})
    made = Data(kind="table", frame=frame, roles={"target": "label"})
    info = export_dataset(made, {"target": "label", "split": split},
                          tmp_path / "out" / "made")
    assert info["source"] == {}
