import errno
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Optional

# 開いた親の下のディレクトリを開くときの印 (リンクはたどらない)
DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)

# 開いた親の下に写すファイルの印 (リンクはたどらず, 既にあれば誤り)
FILE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
)

# 開いた親に書くときの, 中身を書く手元の一時の場所の名前の頭
HOLDER_PREFIX = "ml_eda-writing-"


def entry_name(final: Path) -> str:
    """
    開いた親の中に書く名前を返す関数

    Parameters
    ----------
    final: Path
        書き出す場所 (最後の部分だけを使う)

    Returns
    ----------
    name: str
        親の中の名前

    Raises
    ----------
    ValueError
        名前が空か, . か .. の場合
    """
    # 親の中の 1 つの名前 (途中はたどらない)
    name = final.name
    if name in ("", ".", ".."):
        raise ValueError(
            f"{final} の最後の部分は, 開いた親の中の名前にできません"
        )

    return name


def entry_state(parent: int, name: str) -> Optional[str]:
    """
    開いた親の中の名前の様子を, リンクをたどらずに返す関数

    Parameters
    ----------
    parent: int
        親のディレクトリの記述子
    name: str
        親の中の名前

    Returns
    ----------
    state: Optional[str]
        無ければ None, dataset.yaml を普通のファイルで持つディレクトリなら
        dataset, 他 (リンクも) は other
    """
    # リンクは中を見ずに other
    try:
        mode = os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode
    except FileNotFoundError:
        state = None
        return state
    if not stat.S_ISDIR(mode):
        state = "other"
        return state

    # 中の説明も, リンクをたどらずに確かめる
    fd = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)
    try:
        mode = os.stat(
            "dataset.yaml", dir_fd=fd, follow_symlinks=False
        ).st_mode
        state = "dataset" if stat.S_ISREG(mode) else "other"
    except FileNotFoundError:
        state = "other"
    finally:
        os.close(fd)

    return state


def remove_entry(parent: int, name: str) -> None:
    """
    開いた親の中の名前を, リンクをたどらずに消す関数

    ディレクトリは中から消す. リンクはリンクそのものだけを消す

    Parameters
    ----------
    parent: int
        親のディレクトリの記述子
    name: str
        親の中の名前 (無ければ何もしない)

    Returns
    ----------
    None
    """
    # ディレクトリでないもの (リンクも) はそのものを消す
    try:
        mode = os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(mode):
        os.unlink(name, dir_fd=parent)
        return

    # 中を開いて 1 つずつ消してから, ディレクトリを消す
    fd = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)
    try:
        for child in os.listdir(fd):
            remove_entry(fd, child)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent)


def copy_entry(source: Path, parent: int, name: str) -> None:
    """
    手元のディレクトリを, 開いた親の中に新しく写す関数

    写す先はリンクをたどらずに作って開く (既にあれば誤り)

    Parameters
    ----------
    source: Path
        写すディレクトリ
    parent: int
        写す先の親のディレクトリの記述子
    name: str
        写す先の親の中の名前

    Returns
    ----------
    None

    Raises
    ----------
    OSError
        写す先が既にある場合と, 普通のファイルでもディレクトリでもない
        ものがある場合
    """
    # 写す先のディレクトリを作って開く
    os.mkdir(name, dir_fd=parent)
    fd = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)

    # 中のディレクトリは奥へ, ファイルは開いた先に新しく書く
    try:
        with os.scandir(source) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    copy_entry(Path(entry.path), fd, entry.name)
                elif entry.is_file(follow_symlinks=False):
                    with open(entry.path, "rb") as reader:
                        out = os.open(
                            entry.name, FILE_FLAGS, 0o666, dir_fd=fd
                        )
                        with os.fdopen(out, "wb") as writer:
                            shutil.copyfileobj(reader, writer)
                else:
                    raise OSError(
                        f"{entry.path} は普通のファイルでもディレクトリでも"
                        "ないので写しません"
                    )
    finally:
        os.close(fd)


def make_holder(name: str) -> Path:
    """
    開いた親に書く中身を置く, 手元の一時の場所を作る関数

    自分だけが読み書きできる一時のディレクトリの下に, 名前のディレクトリ
    を作る (中のディレクトリは普通の権限で作る)

    Parameters
    ----------
    name: str
        書き出す場所の名前

    Returns
    ----------
    target: Path
        中身を書くディレクトリ
    """
    # tempfile の一時のディレクトリ (途中の場所を含まない)
    holder = Path(tempfile.mkdtemp(prefix=HOLDER_PREFIX))
    target = holder / name
    try:
        target.mkdir()
    except BaseException:
        shutil.rmtree(holder, ignore_errors=True)
        raise

    return target


def drop_holder(target: Path) -> None:
    """
    手元の一時の場所を, 中身ごと消す関数

    Parameters
    ----------
    target: Path
        make_holder が返したディレクトリ

    Returns
    ----------
    None
    """
    # 自分だけが読み書きできる一時のディレクトリごと
    shutil.rmtree(target.parent, ignore_errors=True)


def move_into(target: Path, parent: int, name: str, final: Path) -> None:
    """
    手元に書いた中身を, 開いた親の中の名前に入れ替える関数

    親の中の一時ディレクトリ (.<名前>.writing) に移し (別のファイル
    システムならリンクをたどらずに写す), 前のデータセットを消して入れ
    替える. どれも親に対する相対の操作で, リンクをたどらない

    Parameters
    ----------
    target: Path
        make_holder が返し, 中身を書き終えたディレクトリ
    parent: int
        書き出す場所の親のディレクトリの記述子
    name: str
        親の中の名前
    final: Path
        書き出す場所 (理由の文に使う)

    Returns
    ----------
    None

    Raises
    ----------
    FileExistsError
        書き出す場所がデータセットではない (リンクも) 場合
    """
    # 前の失敗で残った一時ディレクトリは消してから移す
    writing = f".{name}.writing"
    try:
        remove_entry(parent, writing)
        try:
            os.rename(target, writing, dst_dir_fd=parent)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            copy_entry(target, parent, writing)

        # 置き換えられるのは前のデータセットだけ
        state = entry_state(parent, name)
        if state == "other":
            raise FileExistsError(
                f"{final} はデータセットではないので置き換えません"
            )
        if state == "dataset":
            remove_entry(parent, name)
        os.rename(writing, name, src_dir_fd=parent, dst_dir_fd=parent)
    except BaseException:
        try:
            remove_entry(parent, writing)
        except OSError:
            pass
        raise
    finally:
        drop_holder(target)
