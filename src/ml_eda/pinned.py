import errno
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Optional

# リンクをたどらない印
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

# 開いた親の下のディレクトリを開くときの印 (リンクはたどらない)
DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | NOFOLLOW

# 開いた親の下のファイルを読むときの印 (リンクはたどらず, FIFO などを
# 待たない)
FILE_READ_FLAGS = os.O_RDONLY | NOFOLLOW | getattr(os, "O_NONBLOCK", 0)

# 開いた親の下に写すファイルの印 (リンクはたどらず, 既にあれば誤り)
FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW

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
    # 親の中の 1 つの名前 (途中は使わない)
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

    # リンクをたどらない, 中の説明の確かめ
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
    # ディレクトリでないもの (リンクも) はそのものの消去
    try:
        mode = os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(mode):
        os.unlink(name, dir_fd=parent)
        return

    # 中を開いて 1 つずつ消した後の, ディレクトリの消去
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
    # 写す先のディレクトリの作成と記述子
    os.mkdir(name, dir_fd=parent)
    fd = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)

    # 中のディレクトリは奥へ, ファイルは開いた先への新しい書き込み
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
    # 自分だけが読み書きできる一時のディレクトリごとの消去
    shutil.rmtree(target.parent, ignore_errors=True)


def clear_leftover(parent: int, leftover: str, final: Path) -> None:
    """
    前の失敗で親の中に残った一時の名前を, 書き出しの印があるときだけ消す
    関数

    印は dataset.yaml (この書き出しは説明を書いてから親の中に移すので,
    残ったものは説明を持つ). 印の無いものは使う側のものかもしれないので
    消さずに断る

    Parameters
    ----------
    parent: int
        親のディレクトリの記述子
    leftover: str
        親の中の一時の名前 (.<名前>.writing か .<名前>.old)
    final: Path
        書き出す場所 (理由の文に使う)

    Returns
    ----------
    None

    Raises
    ----------
    FileExistsError
        一時の名前に印の無いもの (リンクも) がある場合
    """
    # 印のあるディレクトリだけの消去
    state = entry_state(parent, leftover)
    if state == "other":
        raise FileExistsError(
            f"{final} の隣の {leftover} はこの書き出しの残りではないので"
            "消しません. 確かめてからもう一度書き出してください"
        )
    if state == "dataset":
        remove_entry(parent, leftover)


def replace_entry(
    parent: int, writing: str, name: str, final: Path
) -> None:
    """
    親の中の一時ディレクトリを, 前のデータセットと入れ替える関数

    前のものは先に退避の名前 (.<名前>.old) に移し, 移した後のものを
    確かめてから消す (確かめと消去の間に替えられても, データセットで
    ないものを消さないため). データセットでなければ元の名前に戻す

    Parameters
    ----------
    parent: int
        親のディレクトリの記述子
    writing: str
        親の中の一時ディレクトリの名前
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
        前のものがデータセットではない (リンクも) 場合
    """
    # 前のものの退避 (無ければ退避しない)
    aside = f".{name}.old"
    clear_leftover(parent, aside, final)
    try:
        os.rename(name, aside, src_dir_fd=parent, dst_dir_fd=parent)
        moved = True
    except FileNotFoundError:
        moved = False

    # 退避したものの確かめ (データセットでなければ元の名前に戻して断る)
    if moved and entry_state(parent, aside) != "dataset":
        if entry_state(parent, name) is None:
            os.rename(aside, name, src_dir_fd=parent, dst_dir_fd=parent)
        raise FileExistsError(
            f"{final} はデータセットではないので置き換えません"
        )

    # 新しいものを置いてから, 退避した前のデータセットの消去
    os.rename(writing, name, src_dir_fd=parent, dst_dir_fd=parent)
    if moved:
        remove_entry(parent, aside)


def move_into(target: Path, parent: int, name: str, final: Path) -> None:
    """
    手元に書いた中身を, 開いた親の中の名前に入れ替える関数

    親の中の一時ディレクトリ (.<名前>.writing) に移し (別のファイル
    システムならリンクをたどらずに写す), 前のデータセットと入れ替える
    (replace_entry). どれも親に対する相対の操作で, リンクをたどらない

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
        書き出す場所がデータセットではない (リンクも) 場合と, 親の中の
        一時の名前に印の無いものがある場合
    """
    # 前の失敗の残り (印のあるものだけ) を消してからの移し
    writing = f".{name}.writing"
    placed = False
    try:
        clear_leftover(parent, writing, final)
        placed = True
        try:
            os.rename(target, writing, dst_dir_fd=parent)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            copy_entry(target, parent, writing)

        # 前のデータセットとの入れ替え
        replace_entry(parent, writing, name, final)
    except BaseException:
        # 自分で置いた一時ディレクトリだけの片付け
        if placed:
            try:
                remove_entry(parent, writing)
            except OSError:
                pass
        raise
    finally:
        drop_holder(target)


def open_under(parent: int, name: str, place: Any) -> BinaryIO:
    """
    開いた親の中のデータセットのファイルを, リンクをたどらずに開く関数

    データセットのディレクトリから一段ずつ O_NOFOLLOW で開き, 最後の
    ファイルも普通のファイルであることを開いた後に確かめる

    Parameters
    ----------
    parent: int
        データセットの親のディレクトリの記述子
    name: str
        親の中のデータセットの名前
    place: Any
        説明に書かれた場所 (データセットからの相対)

    Returns
    ----------
    file: BinaryIO
        読むために開いたファイル (閉じるのは呼ぶ側)

    Raises
    ----------
    ValueError
        場所が文字でないか外を指す (絶対パスや ..) 場合と, 途中か場所が
        リンクか普通のファイルでない場合
    FileNotFoundError
        ファイルが無い場合
    """
    # 書かれた場所の文字の上での確かめ (外と空は誤り)
    text = place if isinstance(place, str) else ""
    normal = os.path.normpath(text) if text else ""
    parts = [part for part in normal.split(os.sep) if part not in ("", ".")]
    if not parts or os.path.isabs(text) or ".." in parts:
        raise ValueError(f"データセットの外か空の場所です: {place}")

    # データセットから一段ずつの, リンクをたどらない開き
    fds: list[int] = []
    try:
        names = [name, *parts]
        for index, part in enumerate(names):
            last = index == len(parts)
            flags = FILE_READ_FLAGS if last else DIRECTORY_FLAGS
            where = fds[-1] if fds else parent
            try:
                fds.append(os.open(part, flags, dir_fd=where))
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"データセットに {text} がありません"
                ) from error
            except OSError as error:
                raise ValueError(
                    refused_reason(where, part, names[:index], last)
                ) from error
        if not stat.S_ISREG(os.fstat(fds[-1]).st_mode):
            raise ValueError(
                f"データセットの {text} は普通のファイルではありません"
            )
        file = os.fdopen(fds[-1], "rb")
        fds.pop()
    finally:
        for fd in fds:
            os.close(fd)

    return file


def refused_reason(
    where: int, part: str, before: list[str], last: bool
) -> str:
    """
    開いた親の中で開けなかった場所の理由の文を返す関数

    Parameters
    ----------
    where: int
        開けなかった場所の親のディレクトリの記述子
    part: str
        開けなかった場所の名前
    before: list[str]
        その場所より前の名前 (先頭はデータセットの名前. データセット自体
        なら空)
    last: bool
        読むファイルそのものか (False ならディレクトリのはずの場所)

    Returns
    ----------
    reason: str
        何がリンクか, ディレクトリでないかの文
    """
    # 場所の呼び方 (データセット, 途中のディレクトリ, 読むファイル)
    if not before and not last:
        what = f"データセットのディレクトリ {part}"
    elif last:
        what = f"データセットのファイル {'/'.join([*before[1:], part])}"
    else:
        what = (
            f"データセットの途中のディレクトリ "
            f"{'/'.join([*before[1:], part])}"
        )

    # リンクか, それ以外 (ディレクトリでないものなど) かの見分け
    try:
        mode = os.stat(part, dir_fd=where, follow_symlinks=False).st_mode
        linked = stat.S_ISLNK(mode)
    except OSError:
        linked = False
    reason = (
        f"{what} がシンボリックリンクなので読みません" if linked
        else f"{what} を開けないので読みません (ディレクトリでないもの"
        "などです)"
    )

    return reason
