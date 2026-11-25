import errno
import os
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
from ml_eda import pipeline

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
    # tempfile の一時のディレクトリを差し替え
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
    # 使う側と同じく, 開いて渡す
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
    # 外には同じ名前の途中を用意する (リンクをたどると outside/b に書く)
    outside = tmp_path / "outside"
    (outside / "b").mkdir(parents=True)
    original: Callable[..., str] = module.write_frame

    def swapped(frame: pd.DataFrame, path: Path) -> str:
        # 最初に呼ばれたときだけ替える
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
    # 書いて読め, 手元の一時の場所と親の中の一時ディレクトリは残らない
    place, fd = opened
    data = small_table(tmp_path)
    info = export_dataset(data, RECIPE, place / NAME, parent=fd)
    assert info["name"] == NAME
    assert len(load_dataset(place / NAME).frames["train"]) > 0
    assert list(holders.iterdir()) == []
    assert sorted(os.listdir(place)) == [NAME]

    # 置き換えは指定が要り, 指定すれば置き換える
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
    # 外のデータセットを指すリンクは, 置き換えを指定しても置き換えない
    place, fd = opened
    data = small_table(tmp_path)
    export_dataset(data, RECIPE, tmp_path / "elsewhere")
    (place / NAME).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(FileExistsError, match="データセットではない"):
        export_dataset(data, RECIPE, place / NAME, overwrite=True, parent=fd)
    assert (tmp_path / "elsewhere" / "dataset.yaml").is_file()
    assert (place / NAME).is_symlink()

    # 説明の無いディレクトリも置き換えない
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
    # 無い目的の列で前処理が失敗する
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
    # 手元の一時の場所から親への移しだけを, 別のファイルシステムの誤りに
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
    # 書いている最中に project/a が外を指すリンクに替わる
    place, fd = opened
    data = small_table(tmp_path)
    outside = swap_while(monkeypatch, pipeline, tmp_path)
    export_dataset(data, RECIPE, place / NAME, parent=fd)

    # 外には何も書かず, 開いた親 (移った先) に書く
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

    # 画像と領域の対: 外には何も書かず, 開いた親 (移った先) に書く
    export_image_masks(
        made["images"], made["masks"], place / NAME, parent=fd
    )
    assert (tmp_path / "project" / "a").is_symlink()
    assert list((outside / "b").iterdir()) == []
    moved = tmp_path / "project" / "moved" / "b"
    assert len(load_items(moved / NAME).items) > 0

    # 撮影と点群の対も同じ (リンクのまま, 開いた親に書く)
    export_capture_clouds(
        shapes["captures"], shapes["clouds"], place / "clouds", parent=fd
    )
    assert list((outside / "b").iterdir()) == []
    assert len(load_items(moved / "clouds").items) > 0
    assert sorted(os.listdir(moved)) == ["clouds", NAME]
    assert list(holders.iterdir()) == []
