import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml_eda import (
    export_capture_clouds, export_dataset, export_image_masks, load_data,
    load_dataset, load_items
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
