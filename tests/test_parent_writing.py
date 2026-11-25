import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    export_capture_clouds, export_dataset, export_image_masks, load_data,
    load_dataset, load_items, model_inputs
)
from ml_eda import items as items_module
from ml_eda import pinned, pipeline

# 書き出すデータセットの名前
NAME = "ds"

# 表のレシピ
RECIPE = {
    "target": "y", "task": "classification",
    "split": {"method": "random", "valid": 0.2, "seed": 0},
}


@pytest.fixture
def holders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    手元の一時の場所を, 確かめられるディレクトリに置く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        差し替え

    Returns
    ----------
    place: Path
        tempfile が使うディレクトリ
    """
    # tempfile の一時のディレクトリの差し替え
    place = tmp_path / "holders"
    place.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(place))

    return place


@pytest.fixture
def opened(tmp_path: Path) -> Iterator[tuple[Path, int]]:
    """
    プロジェクトの中の親 (project/a/b) を開く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Yields
    ----------
    places: tuple[Path, int]
        親の場所と, 開いた記述子
    """
    # 使う側と同じく, 開いた記述子
    place = tmp_path / "project" / "a" / "b"
    place.mkdir(parents=True)
    fd = os.open(place, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield place, fd
    finally:
        os.close(fd)


def small_table(tmp_path: Path) -> Any:
    """
    小さな表を読み込む関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    data: Any
        読み込んだデータ
    """
    # 20 行の 2 値の分類
    frame = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
    frame.to_csv(tmp_path / "table.csv", index=False)
    data = load_data(tmp_path / "table.csv")

    return data


def swap_while(
    monkeypatch: pytest.MonkeyPatch, module: Any, tmp_path: Path
) -> Path:
    """
    書いている最中に途中の場所 (project/a) をリンクに替える関数

    module の write_frame を, 呼ばれたときに project/a を project/moved に
    移し, 外 (outside) を指すリンクを置いてから書くものに差し替える

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        差し替え
    module: Any
        write_frame を使うモジュール
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    outside: Path
        リンクの先 (中に b を持つ外のディレクトリ)
    """
    # 外の同じ名前の途中 (リンクをたどると outside/b に書く)
    outside = tmp_path / "outside"
    (outside / "b").mkdir(parents=True)
    original: Callable[..., str] = module.write_frame

    def swapped(frame: pd.DataFrame, path: Path) -> str:
        # 最初に呼ばれたときだけの差し替え
        middle = tmp_path / "project" / "a"
        if not middle.is_symlink():
            middle.rename(tmp_path / "project" / "moved")
            middle.symlink_to(outside, target_is_directory=True)
        name = original(frame, path)

        return name

    monkeypatch.setattr(module, "write_frame", swapped)

    return outside


def test_parent_writes_dataset(
    tmp_path: Path, holders: Path, opened: tuple[Path, int]
) -> None:
    """
    開いた親を渡した版 1 の書き出しが, 親の中に書き, 置き換えも同じく
    できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # 書いたものの読み込みと, 手元の一時の場所と親の中の一時の名前の無さ
    place, fd = opened
    data = small_table(tmp_path)
    info = export_dataset(data, RECIPE, place / NAME, parent=fd)
    assert info["name"] == NAME
    assert len(load_dataset(place / NAME).frames["train"]) > 0
    assert list(holders.iterdir()) == []
    assert sorted(os.listdir(place)) == [NAME]

    # 指定の無い置き換えの誤りと, 指定した置き換え
    with pytest.raises(FileExistsError, match="既にあります"):
        export_dataset(data, RECIPE, place / NAME, parent=fd)
    (place / NAME / "old.txt").write_text("old", encoding="utf-8")
    export_dataset(data, RECIPE, place / NAME, overwrite=True, parent=fd)
    assert not (place / NAME / "old.txt").exists()
    assert list(holders.iterdir()) == []


def test_parent_refuses_links(
    tmp_path: Path, holders: Path, opened: tuple[Path, int]
) -> None:
    """
    開いた親の中のリンクとデータセットでないものを置き換えず, 名前に
    できない場所を断ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # 外のデータセットを指すリンク (置き換えを指定しても置き換えない)
    place, fd = opened
    data = small_table(tmp_path)
    export_dataset(data, RECIPE, tmp_path / "elsewhere")
    (place / NAME).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(FileExistsError, match="データセットではない"):
        export_dataset(data, RECIPE, place / NAME, overwrite=True, parent=fd)
    assert (tmp_path / "elsewhere" / "dataset.yaml").is_file()
    assert (place / NAME).is_symlink()

    # 説明の無いディレクトリ (これも置き換えない)
    (place / "plain").mkdir()
    with pytest.raises(FileExistsError, match="データセットではない"):
        export_dataset(
            data, RECIPE, place / "plain", overwrite=True, parent=fd
        )

    # 親の中の名前にできない場所
    with pytest.raises(ValueError, match="名前にできません"):
        export_dataset(data, RECIPE, place / "..", parent=fd)
    assert list(holders.iterdir()) == []


def test_parent_failure_cleans(
    tmp_path: Path, holders: Path, opened: tuple[Path, int]
) -> None:
    """
    開いた親に書く途中で失敗したら, 一時の場所を残さないことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # 無い目的の列での前処理の失敗
    place, fd = opened
    data = small_table(tmp_path)
    with pytest.raises(ValueError, match="目的の列"):
        export_dataset(
            data, {**RECIPE, "target": "missing"}, place / NAME, parent=fd
        )
    assert list(holders.iterdir()) == []
    assert os.listdir(place) == []


def test_parent_other_device(
    tmp_path: Path, holders: Path, opened: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    手元の一時の場所が別のファイルシステムでも, 親の中に写して入れ替える
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子
    monkeypatch: pytest.MonkeyPatch
        差し替え

    Returns
    ----------
    None
    """
    # 手元の一時の場所から親への移しだけの, 別のファイルシステムの誤り
    real = os.rename

    def cross(src: Any, dst: Any, **kwargs: Any) -> None:
        if kwargs.get("src_dir_fd") is None and kwargs.get("dst_dir_fd"):
            raise OSError(errno.EXDEV, "別のファイルシステム")
        real(src, dst, **kwargs)

    monkeypatch.setattr(os, "rename", cross)
    place, fd = opened
    export_dataset(small_table(tmp_path), RECIPE, place / NAME, parent=fd)
    assert len(load_dataset(place / NAME).frames["train"]) > 0
    assert list(holders.iterdir()) == []
    assert sorted(os.listdir(place)) == [NAME]


def test_parent_swap_dataset(
    tmp_path: Path, holders: Path, opened: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    版 1 の書き出しの最中に途中の場所をリンクに替えても, 外に書かない
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子
    monkeypatch: pytest.MonkeyPatch
        差し替え

    Returns
    ----------
    None
    """
    # 書いている最中の, project/a の外を指すリンクへの差し替え
    place, fd = opened
    data = small_table(tmp_path)
    outside = swap_while(monkeypatch, pipeline, tmp_path)
    export_dataset(data, RECIPE, place / NAME, parent=fd)

    # 外の空と, 開いた親 (移った先) への書き込み
    assert (tmp_path / "project" / "a").is_symlink()
    assert list((outside / "b").iterdir()) == []
    moved = tmp_path / "project" / "moved" / "b" / NAME
    assert len(load_dataset(moved).frames["train"]) > 0
    assert list(holders.iterdir()) == []


def test_parent_swap_pairs(
    tmp_path: Path, holders: Path, opened: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ファイルの組の書き出し (画像と領域, 撮影と点群) の最中に途中の場所を
    リンクに替えても, 外に書かないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子
    monkeypatch: pytest.MonkeyPatch
        差し替え

    Returns
    ----------
    None
    """
    # 見本 (画像の読み書きが要る)
    pytest.importorskip("PIL.Image")
    from ml_eda.synthetic import write_image_masks, write_shape_captures

    made = write_image_masks(tmp_path / "raw", count=6)
    shapes = write_shape_captures(tmp_path / "shapes")
    outside = swap_while(monkeypatch, items_module, tmp_path)
    place, fd = opened

    # 画像と領域の対: 外の空と, 開いた親 (移った先) への書き込み
    export_image_masks(
        made["images"], made["masks"], place / NAME, parent=fd
    )
    assert (tmp_path / "project" / "a").is_symlink()
    assert list((outside / "b").iterdir()) == []
    moved = tmp_path / "project" / "moved" / "b"
    assert len(load_items(moved / NAME).items) > 0

    # 撮影と点群の対も同じ (リンクのまま, 開いた親への書き込み)
    export_capture_clouds(
        shapes["captures"], shapes["clouds"], place / "clouds", parent=fd
    )
    assert list((outside / "b").iterdir()) == []
    assert len(load_items(moved / "clouds").items) > 0
    assert sorted(os.listdir(moved)) == ["clouds", NAME]
    assert list(holders.iterdir()) == []


def test_parent_keeps_outside(
    tmp_path: Path, holders: Path, opened: tuple[Path, int]
) -> None:
    """
    前のデータセットの中の外を指すリンクを, 置き換えでリンクだけ消し,
    外を消さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # 前のデータセットの中の, 外のファイルとディレクトリを指すリンク
    place, fd = opened
    data = small_table(tmp_path)
    export_dataset(data, RECIPE, place / NAME, parent=fd)
    outside = tmp_path / "outside"
    (outside / "inner").mkdir(parents=True)
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    (outside / "inner" / "keep.txt").write_text("keep", encoding="utf-8")
    (place / NAME / "file").symlink_to(outside / "keep.txt")
    (place / NAME / "dir").symlink_to(outside / "inner")

    # 置き換えの後も外はそのまま
    export_dataset(data, RECIPE, place / NAME, overwrite=True, parent=fd)
    assert not (place / NAME / "file").exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (outside / "inner" / "keep.txt").is_file()
    assert sorted(os.listdir(place)) == [NAME]


def test_parent_swap_window(
    tmp_path: Path, holders: Path, opened: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    確かめの後, 入れ替えの前にデータセットでないものに替えられても,
    それを消さずに残すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子
    monkeypatch: pytest.MonkeyPatch
        差し替え

    Returns
    ----------
    None
    """
    # 前のデータセット
    place, fd = opened
    data = small_table(tmp_path)
    export_dataset(data, RECIPE, place / NAME, parent=fd)

    # 退避の rename の直前の, 使う側のディレクトリへの差し替え
    real = os.rename

    def swapped(src: Any, dst: Any, **kwargs: Any) -> None:
        if src == NAME and dst == f".{NAME}.old":
            shutil.rmtree(place / NAME)
            (place / NAME).mkdir()
            (place / NAME / "mine.txt").write_text("mine", encoding="utf-8")
        real(src, dst, **kwargs)

    monkeypatch.setattr(os, "rename", swapped)
    with pytest.raises(FileExistsError, match="データセットではない"):
        export_dataset(data, RECIPE, place / NAME, overwrite=True, parent=fd)

    # 使う側のものは元の名前のまま残り, 一時の名前は残らない
    assert (place / NAME / "mine.txt").read_text(encoding="utf-8") == "mine"
    assert sorted(os.listdir(place)) == [NAME]
    assert list(holders.iterdir()) == []


def test_parent_leftovers(
    tmp_path: Path, holders: Path, opened: tuple[Path, int]
) -> None:
    """
    親の中に残った一時の名前を, 書き出しの印があるときだけ消すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # 印の無い一時の名前 (消さずに断る)
    place, fd = opened
    data = small_table(tmp_path)
    leftover = place / f".{NAME}.writing"
    leftover.mkdir()
    (leftover / "mine.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError, match="残りではない"):
        export_dataset(data, RECIPE, place / NAME, parent=fd)
    assert (leftover / "mine.txt").is_file()
    assert not (place / NAME).exists()
    assert list(holders.iterdir()) == []

    # 印 (dataset.yaml) のある一時の名前 (消して書く)
    (leftover / "dataset.yaml").write_text("{}", encoding="utf-8")
    export_dataset(data, RECIPE, place / NAME, parent=fd)
    assert sorted(os.listdir(place)) == [NAME]
    assert len(load_dataset(place / NAME).frames["train"]) > 0


def test_parent_copy_failure(
    tmp_path: Path, holders: Path, opened: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    別のファイルシステムへの写しの途中で失敗したら, 親の中の一時の名前も
    手元の一時の場所も残さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    holders: Path
        tempfile が使うディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子
    monkeypatch: pytest.MonkeyPatch
        差し替え

    Returns
    ----------
    None
    """
    # 別のファイルシステムの誤りと, 2 つ目のファイルの写しの失敗
    real = os.rename
    copy = shutil.copyfileobj
    calls: list[Any] = []

    def cross(src: Any, dst: Any, **kwargs: Any) -> None:
        if kwargs.get("src_dir_fd") is None and kwargs.get("dst_dir_fd"):
            raise OSError(errno.EXDEV, "別のファイルシステム")
        real(src, dst, **kwargs)

    def broken(reader: Any, writer: Any) -> None:
        calls.append(reader)
        if len(calls) == 2:
            raise OSError(errno.ENOSPC, "空きが無い")
        copy(reader, writer)

    monkeypatch.setattr(os, "rename", cross)
    monkeypatch.setattr(pinned.shutil, "copyfileobj", broken)
    place, fd = opened
    with pytest.raises(OSError, match="空きが無い"):
        export_dataset(
            small_table(tmp_path), RECIPE, place / NAME, parent=fd
        )
    assert len(calls) == 2
    assert os.listdir(place) == []
    assert list(holders.iterdir()) == []


def test_holder_private(holders: Path) -> None:
    """
    手元の一時の場所が, 自分だけが読み書きできるディレクトリの下にある
    ことを確かめる関数

    Parameters
    ----------
    holders: Path
        tempfile が使うディレクトリ

    Returns
    ----------
    None
    """
    # 中身を書くディレクトリの親の権限 0700
    target = pinned.make_holder(NAME)
    try:
        assert target.name == NAME
        assert target.parent.parent == holders
        mode = stat.S_IMODE(target.parent.stat().st_mode)
        assert mode == 0o700
    finally:
        pinned.drop_holder(target)
    assert list(holders.iterdir()) == []


def text_dataset(tmp_path: Path, path: Path, fd: int) -> None:
    """
    中身の配列と語彙を持つ文のデータセットを, 開いた親に書き出す関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    path: Path
        書き出す場所
    fd: int
        開いた親の記述子

    Returns
    ----------
    None
    """
    # 2 つのラベルの短い文と, 語の番号の並びにするレシピ
    reviews = tmp_path / "reviews"
    if not reviews.exists():
        for label, words in (("pos", "good great"), ("neg", "bad awful")):
            (reviews / label).mkdir(parents=True)
            for index in range(5):
                (reviews / label / f"{index}.txt").write_text(
                    f"this is {words} {index}", encoding="utf-8"
                )
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [{"op": "vocabulary", "length": 6}],
        "split": {"method": "stratified", "valid": 0.2},
    }
    export_dataset(load_data(reviews), recipe, path, parent=fd)


def test_parent_model_inputs(
    tmp_path: Path, opened: tuple[Path, int]
) -> None:
    """
    開いた親を渡した model_inputs が, パスで読んだ結果と同じで, 途中が
    替わっても替わった先を読まないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # 表と, 中身の配列と語彙を持つ文のデータセット
    place, fd = opened
    export_dataset(small_table(tmp_path), RECIPE, place / NAME, parent=fd)
    text_dataset(tmp_path, place / "text", fd)

    # パスで読んだものと同じ (語彙と配列の型も)
    for name in (NAME, "text"):
        assert model_inputs(place / name, parent=fd) == model_inputs(
            place / name
        )
    assert model_inputs(place / "text", parent=fd)["inputs"][0][
        "vocabulary"
    ] > 0

    # 途中を外を指すリンクに替えた後の, 開いた親の中の読み込み (外の
    # 偽物は読まない)
    outside = tmp_path / "outside"
    export_dataset(
        small_table(tmp_path), {**RECIPE, "task": "regression"},
        outside / "b" / NAME
    )
    (tmp_path / "project" / "a").rename(tmp_path / "project" / "moved")
    (tmp_path / "project" / "a").symlink_to(outside)
    assert model_inputs(place / NAME)["task"] == "regression"
    assert model_inputs(place / NAME, parent=fd)["task"] == (
        "classification"
    )


def test_parent_model_inputs_links(
    tmp_path: Path, opened: tuple[Path, int]
) -> None:
    """
    開いた親を渡した model_inputs が, 中のリンクと外を指す場所と,
    follow_links との組を誤りにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # データセットそのものがリンク
    place, fd = opened
    export_dataset(small_table(tmp_path), RECIPE, tmp_path / "elsewhere")
    (place / NAME).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(
        ValueError, match=f"データセットのディレクトリ {NAME} がシンボリック"
    ):
        model_inputs(place / NAME, parent=fd)

    # 説明がリンク
    (place / "inner").mkdir()
    (place / "inner" / "dataset.yaml").symlink_to(
        tmp_path / "elsewhere" / "dataset.yaml"
    )
    with pytest.raises(
        ValueError, match="ファイル dataset.yaml がシンボリックリンク"
    ):
        model_inputs(place / "inner", parent=fd)

    # 無い説明と, follow_links との組
    (place / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="dataset.yaml"):
        model_inputs(place / "empty", parent=fd)
    with pytest.raises(ValueError, match="follow_links"):
        model_inputs(place / "empty", follow_links=True, parent=fd)
    with pytest.raises(ValueError, match="データセットの外"):
        pinned.open_under(fd, "empty", "../x")


def test_parent_model_inputs_inner_links(
    tmp_path: Path, opened: tuple[Path, int]
) -> None:
    """
    開いた親を渡した model_inputs が, 中身の配列, レシピ, 途中の
    ディレクトリのリンクを, どれかを添えた誤りにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # 中身の配列のリンク (中身は同じ写し)
    place, fd = opened
    text_dataset(tmp_path, place / "text", fd)
    root = place / "text"
    shutil.copyfile(root / "train_items.npy", tmp_path / "copy.npy")
    (root / "train_items.npy").unlink()
    (root / "train_items.npy").symlink_to(tmp_path / "copy.npy")
    with pytest.raises(
        ValueError, match="ファイル train_items.npy がシンボリックリンク"
    ):
        model_inputs(root, parent=fd)

    # レシピのリンク
    (root / "train_items.npy").unlink()
    shutil.copyfile(tmp_path / "copy.npy", root / "train_items.npy")
    shutil.copyfile(root / "recipe.yaml", tmp_path / "recipe.yaml")
    (root / "recipe.yaml").unlink()
    (root / "recipe.yaml").symlink_to(tmp_path / "recipe.yaml")
    with pytest.raises(
        ValueError, match="ファイル recipe.yaml がシンボリックリンク"
    ):
        model_inputs(root, parent=fd)

    # 途中のディレクトリのリンクと, ディレクトリでないもの
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.npy").write_bytes(b"")
    (root / "sub").symlink_to(tmp_path / "sub", target_is_directory=True)
    with pytest.raises(
        ValueError, match="途中のディレクトリ sub がシンボリックリンク"
    ):
        pinned.open_under(fd, "text", "sub/x.npy")
    with pytest.raises(ValueError, match="dataset.yaml を開けない"):
        pinned.open_under(fd, "text", "dataset.yaml/x")


def test_parent_model_inputs_headers(
    tmp_path: Path, opened: tuple[Path, int]
) -> None:
    """
    開いた親を渡した model_inputs が, 版 2.0 の配列の頭を読み, 壊れた頭,
    物体型の配列, FIFO の説明を待たずに誤りにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    opened: tuple[Path, int]
        親の場所と, 開いた記述子

    Returns
    ----------
    None
    """
    # 版 2.0 の頭で書き直した配列 (パスで読んだものと同じ)
    place, fd = opened
    text_dataset(tmp_path, place / "text", fd)
    items = place / "text" / "train_items.npy"
    array = np.load(items)
    with open(items, "wb") as file:
        np.lib.format.write_array(file, array, version=(2, 0))
    with open(items, "rb") as file:
        assert np.lib.format.read_magic(file) == (2, 0)
    assert model_inputs(place / "text", parent=fd) == model_inputs(
        place / "text"
    )

    # 物体型の配列 (パスで読む場合も誤り)
    np.save(items, np.array([1, "a"], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="物体"):
        model_inputs(place / "text", parent=fd)
    with pytest.raises(ValueError):
        model_inputs(place / "text")

    # 壊れた頭
    items.write_bytes(b"not an array")
    with pytest.raises(ValueError):
        model_inputs(place / "text", parent=fd)

    # FIFO の説明 (待たずに誤り)
    (place / "fifo").mkdir()
    os.mkfifo(place / "fifo" / "dataset.yaml")
    with pytest.raises(ValueError, match="普通のファイルではありません"):
        model_inputs(place / "fifo", parent=fd)
