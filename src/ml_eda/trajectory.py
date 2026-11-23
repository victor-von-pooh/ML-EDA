import csv
import io
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.profile import to_json

# 読める軌跡の説明の版
TRAJECTORY_FORMAT_VERSION = 1

# 読む大きさの上限 (表と説明)
MAX_TABLE_BYTES = 64 << 20
MAX_INFO_BYTES = 1 << 20

# 表のファイルの拡張子と形式
TABLE_FORMATS = {".csv": "csv", ".parquet": "parquet"}

# 観測と行動の列の名前の頭
OBSERVATION_PREFIX = "obs"
ACTION_PREFIX = "action"

# 名前で型の決まった列
KNOWN_TYPES = {
    "episode": "integer", "step": "integer", "reward": "number",
    "terminated": "boolean", "truncated": "boolean", "seed": "integer",
    "guard": "string", "guard_reasons": "string", "action_prob": "number",
}

# 行動を選んだ確率の列
PROB_COLUMN = "action_prob"

# 観測と行動の他に要る列
REQUIRED_COLUMNS = ("episode", "step", "reward", "terminated", "truncated")

# 列の型
COLUMN_TYPES = ("integer", "number", "boolean", "string")

# 守りで止めた判断 (行動と報酬が空で, 環境を進めていない行)
STOPPED_RESULTS = ("stop", "handover")

# 守りが働いた理由のつなぎ
REASON_SEPARATOR = "; "

# 行動の値ごとの数を返す, 異なる値の数の上限
MAX_ACTION_VALUES = 50

# Parquet の行の数の上限 (64 MiB の CSV の最も短い行と同じ程度)
MAX_PARQUET_ROWS = 4_000_000

# 要約に並べるエピソードの数の上限
MAX_EPISODE_ROWS = 1000

# 気付きに書く値の文字の長さの上限
MAX_NOTE_VALUE = 80

# int64 の範囲
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

# 文字の値の書き方
INTEGER_TEXT = re.compile(r"[+-]?\d+")
BOOLEAN_TEXT = {"true": True, "false": False}


@dataclass
class Trajectory:
    # 軌跡の名前 (表のファイルの名前から拡張子を除いたもの)
    name: str
    # 表の形式 (csv か parquet)
    format: str
    # 1 歩が 1 行の表
    frame: pd.DataFrame
    # 列ごとの型 (integer, number, boolean, string)
    types: dict
    # 観測の列 (要素の順)
    observation_columns: list
    # 行動の列 (要素の順)
    action_columns: list
    # 守りの結果の列 (guard, guard_reasons) があるか
    guarded: bool
    # 説明 (<名前>.json) の中身. 無いか読まなかった場合は None
    info: Optional[dict] = None
    # 読むときに気付いたこと (日本語の文)
    notes: list = field(default_factory=list)
    # 行動を選んだ確率の列 (action_prob) があるか
    with_action_prob: bool = False


def read_regular(path: Path, limit: int, what: str) -> bytes:
    """
    リンクでない普通のファイルを, 大きさの上限まで読む関数

    Parameters
    ----------
    path: Path
        ファイル
    limit: int
        読む大きさの上限 (バイト)
    what: str
        誤りに書くファイルの呼び名

    Returns
    ----------
    raw: bytes
        ファイルの中身

    Raises
    ----------
    FileNotFoundError
        無い場合
    ValueError
        リンク, 普通でないファイル, 上限を超える場合
    """
    # リンクは辿らず, FIFO で待たないように開く
    if path.is_symlink():
        raise ValueError(f"{what}がシンボリックリンクなので読みません: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_NONBLOCK", 0
    ) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise FileNotFoundError(f"{what}がありません: {path}") from None
    except OSError as error:
        raise ValueError(f"{what}を開けません: {path} ({error})") from error

    # 開いたものが普通のファイルで, 上限に収まる場合だけ
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode):
        os.close(descriptor)
        raise ValueError(f"{what}が普通のファイルではありません: {path}")
    if status.st_size > limit:
        os.close(descriptor)
        raise ValueError(
            f"{what}が {limit >> 20} MiB を超えるので読みません: {path}"
        )
    with os.fdopen(descriptor, "rb") as file:
        raw = file.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(
            f"{what}が {limit >> 20} MiB を超えるので読みません: {path}"
        )

    return raw


def parse_info(raw: bytes, path: Path) -> dict:
    """
    説明の JSON を読み, 版を確かめる関数

    Parameters
    ----------
    raw: bytes
        説明のファイルの中身
    path: Path
        説明のファイル (誤りに書く)

    Returns
    ----------
    info: dict
        説明の中身

    Raises
    ----------
    ValueError
        JSON の辞書でないか, 版が 1 以上の整数でない場合
    """
    # JSON の辞書で, 版が 1 以上の整数
    try:
        info = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"説明を JSON として読めません: {path}") from error
    if not isinstance(info, dict):
        raise ValueError(f"説明が JSON の辞書ではありません: {path}")
    version = info.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or (
        version < 1
    ):
        raise ValueError(
            f"説明の format_version が 1 以上の整数ではありません: {path}"
        )

    return info


def table_of_info(info: dict, path: Path) -> Path:
    """
    説明の file から, 同じディレクトリの表のファイルを返す関数

    Parameters
    ----------
    info: dict
        説明の中身
    path: Path
        説明のファイル

    Returns
    ----------
    table: Path
        表のファイル

    Raises
    ----------
    ValueError
        file が同じディレクトリの csv か parquet の名前でない場合
    """
    # 区切りと .. を含まない, 知っている拡張子の名前だけ
    name = info.get("file")
    plain = isinstance(name, str) and name not in ("", ".", "..") and (
        "/" not in name and "\\" not in name
    )
    if not plain or Path(name).suffix.lower() not in TABLE_FORMATS:
        raise ValueError(
            "説明の file が同じディレクトリの csv か parquet の名前ではあり"
            f"ません: {name!r}"
        )
    table = path.parent / name

    return table


def optional_info(path: Path, notes: list) -> Optional[dict]:
    """
    表の隣の説明 (<名前>.json) を, あれば読む関数

    Parameters
    ----------
    path: Path
        説明のファイル
    notes: list
        読まなかった理由を足す並び

    Returns
    ----------
    info: Optional[dict]
        説明の中身. 無いか読めない場合は None (説明の無い表として読む)
    """
    # 無ければ利用者の表. 読めなければ理由を残して説明の無い表として
    if not path.exists() and not path.is_symlink():
        return None
    try:
        info = parse_info(read_regular(path, MAX_INFO_BYTES, "説明"), path)
    except ValueError as error:
        notes.append(f"説明を使わず, 列の名前と値から読みます: {error}")
        return None

    return info


def text_values(
    text: pd.Series, kind: Optional[str], name: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    CSV の 1 列の文字を, 型の値と欠けの印にする関数

    Parameters
    ----------
    text: pd.Series
        列の文字
    kind: Optional[str]
        列の型 (None なら値から決める)
    name: str
        列の名前 (誤りに書く)

    Returns
    ----------
    values: np.ndarray
        値 (欠けの場所の値は使わない)
    missing: np.ndarray
        欠け (空の値) の印
    kind: str
        列の型

    Raises
    ----------
    ValueError
        値が型の書き方でない場合
    """
    # 空だけが欠け. nan, inf は数の書き方
    text = text.fillna("").astype(str)
    missing = (text == "").to_numpy()
    filled = text[~missing]
    lower = filled.str.strip().str.lower()
    if kind is None:
        kind = guess_type(filled, lower)

    # 型ごとに読み, 読めない値は列の名前と値を挙げて誤り
    if kind == "boolean":
        bad = ~lower.isin(list(BOOLEAN_TEXT))
        found = lower.map(BOOLEAN_TEXT)
    elif kind == "integer":
        bad = ~filled.str.strip().str.fullmatch(INTEGER_TEXT.pattern)
        found = filled.where(~bad, "0").map(int)
    elif kind == "number":
        found = pd.to_numeric(filled, errors="coerce")
        bad = found.isna() & ~lower.str.lstrip("+-").eq("nan")
    else:
        bad = pd.Series(False, index=filled.index)
        found = filled
    if bool(bad.any()):
        value = filled[bad].iloc[0]
        raise ValueError(f"列 {name} の値 {value!r} は {kind} ではありません")
    values = np.empty(len(text), dtype=object)
    values[~missing] = found.to_numpy()

    return values, missing, kind


def guess_type(filled: pd.Series, lower: pd.Series) -> str:
    """
    説明の無い列の型を, 値から決める関数

    Parameters
    ----------
    filled: pd.Series
        空でない値の文字
    lower: pd.Series
        前後の空白を除いて小文字にした値

    Returns
    ----------
    kind: str
        真偽値, 整数, 数, 文字の順に, 全ての値が読める型
    """
    # 値が無い列は数 (観測, 行動, 報酬の空の列と同じ)
    if filled.empty:
        return "number"
    if bool(lower.isin(list(BOOLEAN_TEXT)).all()):
        return "boolean"
    if bool(filled.str.strip().str.fullmatch(INTEGER_TEXT.pattern).all()):
        return "integer"
    numbers = pd.to_numeric(filled, errors="coerce")
    nan_text = lower.str.lstrip("+-").eq("nan")
    kind = "number" if bool((numbers.notna() | nan_text).all()) else "string"

    return kind


def arrow_values(
    column: Any, kind: Optional[str], name: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Parquet の 1 列を, 型の値と欠けの印にする関数

    Parameters
    ----------
    column: Any
        pyarrow の列
    kind: Optional[str]
        列の型 (None なら Parquet の型から決める)
    name: str
        列の名前 (誤りに書く)

    Returns
    ----------
    values: np.ndarray
        値 (欠けの場所の値は使わない)
    missing: np.ndarray
        欠け (null) の印. nan は欠けにしない
    kind: str
        列の型

    Raises
    ----------
    ValueError
        Parquet の型が列の型と合わない場合
    """
    import pyarrow as pa

    # 文字の列は CSV と同じ読み方
    array = column.combine_chunks()
    arrow_type = array.type
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(
        arrow_type
    ):
        text = pd.Series(array.to_pylist(), dtype=object)
        values, missing, kind = text_values(text, kind, name)
        return values, missing, kind

    # 全て null の列は, 欠けだけの列として決まった型で
    missing = array.is_null().to_numpy(zero_copy_only=False)
    if pa.types.is_null(arrow_type):
        values = np.full(len(array), None, dtype=object)
        return values, missing, kind or "number"

    # 型の合う列だけ. 整数は数としても読み, 値が全て整数の浮動小数は
    # 整数としても読む (pandas が欠けのある整数を浮動小数で書くため)
    found = (
        "boolean" if pa.types.is_boolean(arrow_type)
        else "integer" if pa.types.is_integer(arrow_type)
        else "number" if pa.types.is_floating(arrow_type) else "string"
    )
    kind = found if kind is None else kind
    allowed = {
        "boolean": ("boolean",), "integer": ("integer",),
        "number": ("integer", "number"),
        "string": ("boolean", "integer", "number", "string"),
    }[kind]
    values = np.array(array.to_pylist(), dtype=object)
    present = values[~missing]
    if kind == "integer" and found == "number" and all(
        math.isfinite(value) and float(value).is_integer()
        for value in present
    ):
        found = "integer"
        values[~missing] = [int(value) for value in present]
    if found not in allowed:
        raise ValueError(
            f"列 {name} の Parquet の型 {arrow_type} は {kind} ではありません"
        )
    if kind == "string":
        values = np.array(
            [None if value is None else str(value) for value in values],
            dtype=object
        )

    return values, missing, kind


def typed_series(
    values: np.ndarray, missing: np.ndarray, kind: str, name: str
) -> pd.Series:
    """
    値と欠けの印から, 型の列を作る関数

    Parameters
    ----------
    values: np.ndarray
        値
    missing: np.ndarray
        欠けの印
    kind: str
        列の型
    name: str
        列の名前 (誤りに書く)

    Returns
    ----------
    series: pd.Series
        欠けの無い列は int64, float64, bool, 文字の object. 欠けのある列は
        Int64, Float64 (nan は欠けと別に残す), boolean, None を持つ object

    Raises
    ----------
    ValueError
        整数が int64 に入らない場合
    """
    # 欠けの無い列は numpy の型
    if kind == "string":
        data = values.copy()
        data[missing] = None
        return pd.Series(data, dtype=object)
    dtype = {"integer": "int64", "number": "float64", "boolean": "bool"}[kind]
    filled = values.copy()
    filled[missing] = 0
    if kind == "integer":
        large = [
            value for value in filled
            if not INT64_MIN <= int(value) <= INT64_MAX
        ]
        if large:
            raise ValueError(
                f"列 {name} の値 {large[0]} は int64 に入りません"
            )
    data = filled.astype(dtype)
    if not missing.any():
        return pd.Series(data)

    # 欠けのある列は pandas の欠けを持てる型
    array = {
        "integer": pd.arrays.IntegerArray,
        "number": pd.arrays.FloatingArray,
        "boolean": pd.arrays.BooleanArray,
    }[kind](data, missing.astype(bool))
    series = pd.Series(array)

    return series


def element_columns(columns: list, prefix: str, notes: list) -> list:
    """
    観測か行動の列を, 名前で見分けて要素の順に並べる関数

    Parameters
    ----------
    columns: list
        表の列
    prefix: str
        列の名前の頭 (obs か action)
    notes: list
        番号の飛びを足す並び

    Returns
    ----------
    elements: list
        prefix か, prefix_<番号> の列を番号の順に並べたもの

    Raises
    ----------
    ValueError
        列が無いか, prefix と prefix_<番号> が並ぶ場合
    """
    # 1 つの値の列か, 番号の列
    pattern = re.compile(re.escape(prefix) + r"_(0|[1-9]\d*)")
    numbered = sorted(
        (int(match.group(1)), column) for column in columns
        if (match := pattern.fullmatch(column))
    )
    single = prefix in columns
    if single and numbered:
        raise ValueError(
            f"列 {prefix} と {prefix}_<番号> が同じ表に並んでいます"
        )
    if not single and not numbered:
        raise ValueError(
            f"{prefix} か {prefix}_<番号> の列がありません (軌跡の表の列)"
        )
    if single:
        return [prefix]
    numbers = [number for number, _ in numbered]
    if numbers != list(range(len(numbers))):
        notes.append(f"{prefix} の列の番号が 0 から続いていません: {numbers}")
    elements = [column for _, column in numbered]

    return elements


def read_columns(
    raw: bytes, table_format: str, declared: dict, notes: list
) -> tuple[pd.DataFrame, dict]:
    """
    表を読み, 列ごとに型の値にする関数

    Parameters
    ----------
    raw: bytes
        表のファイルの中身
    table_format: str
        csv か parquet
    declared: dict
        説明の列ごとの型 (説明が無ければ空)
    notes: list
        説明の型で読めなかった列を足す並び

    Returns
    ----------
    frame: pd.DataFrame
        型の値の表
    types: dict
        列ごとの型

    Raises
    ----------
    ValueError
        表として読めないか, 値が型に合わない場合
    """
    # 形式ごとに, 列の名前と中身
    if table_format == "csv":
        columns, text = csv_table(raw)
    else:
        arrow = parquet_table(raw)
        columns = [str(name) for name in arrow.column_names]
    if len(set(columns)) != len(columns):
        raise ValueError(f"表の列の名前が重なっています: {columns}")

    # 説明の型, 名前で決まった型, 値から決めた型の順. 説明の型で読め
    # なければ, 表を使う決まりに合わせて決め直す
    data = {}
    types = {}
    for name in columns:
        column = text[name] if table_format == "csv" else arrow.column(name)
        read = text_values if table_format == "csv" else arrow_values
        fallback = KNOWN_TYPES.get(name)
        kind = declared.get(name, fallback)
        try:
            values, missing, kind = read(column, kind, name)
            series = typed_series(values, missing, kind, name)
        except ValueError as error:
            if name not in declared or declared[name] == fallback:
                raise
            notes.append(
                f"説明の型 {declared[name]} で列 {name} を読めないので, "
                f"型を決め直します: {error}"
            )
            values, missing, kind = read(column, fallback, name)
            series = typed_series(values, missing, kind, name)

        # 守りの理由の空は, 働かなかったことなので空の文字
        if name == "guard_reasons" and kind == "string":
            series = series.fillna("")
        data[name] = series
        types[name] = kind
    frame = pd.DataFrame(data, columns=columns)

    return frame, types


def csv_table(raw: bytes) -> tuple[list, pd.DataFrame]:
    """
    CSV の表を, 見出しと文字の値の表にする関数

    Parameters
    ----------
    raw: bytes
        表のファイルの中身

    Returns
    ----------
    columns: list
        列の名前 (見出しの順)
    text: pd.DataFrame
        文字の値の表 (空の値は空の文字)

    Raises
    ----------
    ValueError
        UTF-8 でないか, 見出しが無いか, CSV として読めない場合
    """
    # 見出しは csv で読む (pandas は重なる名前に番号を付けるため)
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("表が UTF-8 ではありません") from error
    rows = csv.reader(io.StringIO(decoded))
    header = next(rows, None)
    if not header:
        raise ValueError("表に見出しの行がありません")
    columns = list(header)

    # 列の数が見出しと違う行は誤り (pandas は黙って切ることがあるため)
    for row in rows:
        if row and len(row) != len(columns):
            raise ValueError(
                f"表の {rows.line_num} 行目の値の数 ({len(row)}) が列の数 "
                f"({len(columns)}) と違います"
            )

    # 値は全て文字のまま読み, 型は列ごとに決める
    try:
        text = pd.read_csv(
            io.StringIO(decoded), dtype=str, keep_default_na=False,
            na_filter=False, header=None, skiprows=1, index_col=False,
            names=[str(index) for index in range(len(columns))]
        )
    except (pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"表を CSV として読めません ({error})") from error
    text.columns = columns

    return columns, text


def parquet_table(raw: bytes) -> Any:
    """
    Parquet の表を pyarrow の表にする関数

    Parameters
    ----------
    raw: bytes
        表のファイルの中身

    Returns
    ----------
    table: Any
        pyarrow の表

    Raises
    ----------
    ImportError
        pyarrow が無い場合
    ValueError
        Parquet として読めない場合
    """
    # nan と欠けを分けるため, pandas を通さず pyarrow で読む
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "parquet を読むには pyarrow が要ります (pip install pyarrow)"
        ) from error
    try:
        rows = pq.ParquetFile(pa.BufferReader(raw)).metadata.num_rows
        if rows > MAX_PARQUET_ROWS:
            raise ValueError(
                f"Parquet の表の行の数 ({rows}) が上限 "
                f"({MAX_PARQUET_ROWS}) を超えるので読みません"
            )
        table = pq.read_table(pa.BufferReader(raw))
    except (pa.ArrowException, OSError) as error:
        raise ValueError(f"表を Parquet として読めません ({error})") from error

    return table


def declared_types(info: Optional[dict], notes: list) -> dict:
    """
    説明の列ごとの型のうち, 知っている型だけを返す関数

    Parameters
    ----------
    info: Optional[dict]
        説明の中身
    notes: list
        使わなかった型を足す並び

    Returns
    ----------
    declared: dict
        列の名前から型
    """
    # 知らない型の列は値から決める
    types = (info or {}).get("types")
    if not isinstance(types, dict):
        return {}
    declared = {
        name: kind for name, kind in types.items() if kind in COLUMN_TYPES
    }
    unknown = sorted(str(name) for name in types if name not in declared)
    if unknown:
        notes.append(f"説明の型が分からない列は値から決めます: {unknown}")

    return declared


def check_columns(
    frame: pd.DataFrame, types: dict, observation: list, action: list
) -> None:
    """
    要る列があり, 型と欠けが決まりに合うかを確かめる関数

    Parameters
    ----------
    frame: pd.DataFrame
        型の値の表
    types: dict
        列ごとの型
    observation: list
        観測の列
    action: list
        行動の列

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        要る列が無いか, 型か欠けが決まりに合わない場合
    """
    # 要る列
    lacking = [name for name in REQUIRED_COLUMNS if name not in frame]
    if lacking:
        raise ValueError(f"軌跡の表に要る列がありません: {lacking}")

    # 番号と終わりの印は欠けの無い整数と真偽値, 観測, 行動, 報酬, 確率は数
    expected = {
        "episode": ("integer",), "step": ("integer",),
        "terminated": ("boolean",), "truncated": ("boolean",),
        "reward": ("integer", "number"),
    }
    expected.update(
        {name: ("integer", "number") for name in observation + action}
    )
    if PROB_COLUMN in frame:
        expected[PROB_COLUMN] = ("integer", "number")
    for name, kinds in expected.items():
        if types[name] not in kinds:
            raise ValueError(
                f"列 {name} の型は {' か '.join(kinds)} にしてください: "
                f"{types[name]}"
            )
    for name in ("episode", "step", "terminated", "truncated"):
        if bool(frame[name].isna().any()):
            raise ValueError(f"列 {name} に空の値があります")


def order_rows(frame: pd.DataFrame, notes: list) -> pd.DataFrame:
    """
    行をエピソード, 歩の順に並べる関数

    Parameters
    ----------
    frame: pd.DataFrame
        型の値の表
    notes: list
        並べ直したことと, 重なる歩を足す並び

    Returns
    ----------
    ordered: pd.DataFrame
        エピソード, 歩の順の表 (番号は 0 から振り直す)
    """
    # 決まりの順でなければ, 同じ歩の行の順を保って並べ直す
    keys = frame[["episode", "step"]].to_numpy(dtype=np.int64)
    ordered = frame
    if len(keys) > 1:
        later = (keys[1:, 0] > keys[:-1, 0]) | (
            (keys[1:, 0] == keys[:-1, 0]) & (keys[1:, 1] >= keys[:-1, 1])
        )
        if not bool(later.all()):
            notes.append("行がエピソード, 歩の順でないので並べ直しました")
            ordered = frame.sort_values(
                ["episode", "step"], kind="stable"
            ).reset_index(drop=True)
    if bool(ordered.duplicated(["episode", "step"]).any()):
        notes.append("同じエピソードの同じ歩の行が重なっています")

    return ordered


def compare_info(
    info: dict, frame: pd.DataFrame, observation: list, action: list,
    guarded: bool, with_action_prob: bool, notes: list
) -> None:
    """
    説明と表の食い違いを気付きにする関数

    Parameters
    ----------
    info: dict
        説明の中身
    frame: pd.DataFrame
        型の値の表
    observation: list
        名前で見分けた観測の列
    action: list
        名前で見分けた行動の列
    guarded: bool
        表に守りの結果の列があるか
    with_action_prob: bool
        表に行動を選んだ確率の列があるか
    notes: list
        食い違いを足す並び

    Returns
    ----------
    None
    """
    # 新しい版は読める項目だけを使う
    version = info["format_version"]
    if version > TRAJECTORY_FORMAT_VERSION:
        notes.append(
            f"説明の版 {version} はこの ML-EDA の知る版 "
            f"{TRAJECTORY_FORMAT_VERSION} より新しい ML-Found の出力です. "
            "読める項目だけを使います"
        )

    # 列と数は表の方を使い, 違えば知らせる
    episodes = int(frame["episode"].nunique())
    checks = (
        ("columns", list(frame.columns)), ("observation_columns", observation),
        ("action_columns", action), ("guarded", guarded),
        ("with_action_prob", with_action_prob), ("rows", len(frame)),
        ("episodes", episodes),
    )
    for key, actual in checks:
        if key in info and info[key] != actual:
            notes.append(
                f"説明の {key} ({short_text(info[key])}) が表 "
                f"({short_text(actual)}) と違います. 表の方を使います"
            )


def short_text(value: Any) -> str:
    """
    気付きに書く値を, 長さの上限までの文字にする関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    text: str
        repr の文字. 長ければ切って ... を付ける
    """
    # 上限を超える分を切る
    text = repr(value)
    if len(text) > MAX_NOTE_VALUE:
        text = text[:MAX_NOTE_VALUE - 3] + "..."

    return text


def load_trajectory(path: Union[str, Path]) -> Trajectory:
    """
    1 歩が 1 行の軌跡の表 (CSV か Parquet) を読む関数

    Parameters
    ----------
    path: Union[str, Path]
        表のファイル (<名前>.csv か <名前>.parquet) か, 説明 (<名前>.json.
        その file の表を読む)

    Returns
    ----------
    trajectory: Trajectory
        型の値の表, 観測と行動の列, 守りと確率の列があるか, 説明, 気付き

    Raises
    ----------
    FileNotFoundError
        表 (か渡した説明) が無い場合
    ImportError
        Parquet の表で pyarrow が無い場合
    ValueError
        リンク, 普通でないファイル, 上限を超えるファイル, 表として読め
        ないか, 列の決まりに合わない場合
    """
    # 説明を渡した場合は, その file の表. 表の場合は, 隣の説明があれば
    path = Path(path)
    notes = []
    if path.suffix.lower() == ".json":
        info = parse_info(read_regular(path, MAX_INFO_BYTES, "説明"), path)
        path = table_of_info(info, path)
    elif path.suffix.lower() in TABLE_FORMATS:
        info = optional_info(path.with_suffix(".json"), notes)
        if info is not None and info.get("file", path.name) != path.name:
            notes.append(
                f"隣の説明の file ({short_text(info.get('file'))}) が表の"
                f"名前 ({path.name}) と違うので, 説明を使いません"
            )
            info = None
    else:
        raise ValueError(
            f"軌跡の表は .csv か .parquet のファイルにしてください: {path}"
        )
    table_format = TABLE_FORMATS[path.suffix.lower()]

    # 表を型の値で読み, 観測と行動の列を名前で見分ける
    raw = read_regular(path, MAX_TABLE_BYTES, "表")
    frame, types = read_columns(
        raw, table_format, declared_types(info, notes), notes
    )
    observation = element_columns(
        list(frame.columns), OBSERVATION_PREFIX, notes
    )
    action = element_columns(list(frame.columns), ACTION_PREFIX, notes)
    check_columns(frame, types, observation, action)
    frame = order_rows(frame, notes)

    # 守りと確率の列は表にあるかで決め, 説明と違えば知らせる
    guarded = "guard" in frame.columns
    with_action_prob = PROB_COLUMN in frame.columns
    if info is not None:
        compare_info(
            info, frame, observation, action, guarded, with_action_prob,
            notes
        )
    trajectory = Trajectory(
        name=path.stem, format=table_format, frame=frame, types=types,
        observation_columns=observation, action_columns=action,
        guarded=guarded, info=info, notes=notes,
        with_action_prob=with_action_prob
    )

    return trajectory


def float_values(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    数の列を, 欠けを除いた float の値と欠けの印にする関数

    Parameters
    ----------
    series: pd.Series
        数の列

    Returns
    ----------
    values: np.ndarray
        float64 の値 (欠けの場所は nan)
    missing: np.ndarray
        欠けの印 (nan は欠けにしない)
    """
    # 欠けは欠けを持てる型の印だけ (numpy の float64 の nan は値)
    if isinstance(series.dtype, np.dtype):
        missing = np.zeros(len(series), dtype=bool)
    else:
        missing = series.isna().to_numpy()
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)

    return values, missing


def number_summary(values: np.ndarray) -> dict:
    """
    数の並びの最小, 最大, 平均, 標準偏差, 中央値を求める関数

    Parameters
    ----------
    values: np.ndarray
        有限の数

    Returns
    ----------
    summary: dict
        min, max, mean, std, median (値が無ければ全て None)
    """
    # 値の無い場合は None
    if values.size == 0:
        return dict.fromkeys(("min", "max", "mean", "std", "median"))
    summary = {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
    }

    return summary


def describe_element(series: pd.Series, name: str, kind: str) -> dict:
    """
    観測か行動の 1 列の値をまとめる関数

    Parameters
    ----------
    series: pd.Series
        列
    name: str
        列の名前
    kind: str
        列の型 (integer か number)

    Returns
    ----------
    element: dict
        name, type, missing (空の数), nan, infinite, min, max, mean,
        std, median (有限の値で), counts (整数の列で, 異なる値が上限以下
        なら値ごとの数の並び. 他は None)
    """
    # 欠け, nan, 無限大を分けて数え, 有限の値で要約
    values, missing = float_values(series)
    present = values[~missing]
    finite = present[np.isfinite(present)]
    element = {
        "name": name, "type": kind, "missing": int(missing.sum()),
        "nan": int(np.isnan(present).sum()),
        "infinite": int(np.isinf(present).sum()),
        **number_summary(finite), "counts": None,
    }

    # 整数の列 (番号の空間) は値ごとの数
    if kind == "integer":
        found, counts = np.unique(present.astype(np.int64), return_counts=True)
        if len(found) <= MAX_ACTION_VALUES:
            element["counts"] = [
                {"value": int(value), "count": int(count)}
                for value, count in zip(found, counts)
            ]

    return element


def episode_rows(trajectory: Trajectory) -> list[dict]:
    """
    エピソードごとの歩数, 報酬の和, 終わり方を求める関数

    Parameters
    ----------
    trajectory: Trajectory
        load_trajectory で読んだ軌跡

    Returns
    ----------
    episodes: list[dict]
        episode, rows (行の数), length (環境を進めた歩数. 報酬のある行),
        return (報酬の和. 空を除き, nan か無限大を含めば None), end
        (terminated, truncated, stop, handover か, 終わりの印の無い open),
        seed (最初の行の種か None) の並び
    """
    # 報酬の和と歩数は報酬のある行で
    frame = trajectory.frame
    rewards, missing = float_values(frame["reward"])
    guard = frame["guard"] if trajectory.guarded else None
    seeds = frame["seed"] if "seed" in frame else None
    episodes = []
    for episode, index in frame.groupby("episode", sort=False).indices.items():
        present = rewards[index][~missing[index]]
        total = float(present.sum())
        last = index[-1]

        # 最後の行で終わり方 (守りで止めた判断, 終了, 打ち切りの順)
        result = guard.iloc[last] if guard is not None else None
        if result in STOPPED_RESULTS:
            end = result
        elif bool(frame["terminated"].iloc[last]):
            end = "terminated"
        elif bool(frame["truncated"].iloc[last]):
            end = "truncated"
        else:
            end = "open"
        seed = seeds.iloc[index[0]] if seeds is not None else None
        episodes.append({
            "episode": int(episode), "rows": int(len(index)),
            "length": int(present.size),
            "return": total if math.isfinite(total) else None,
            "end": end,
            "seed": None if seed is None or pd.isna(seed) else seed,
        })

    return episodes


def describe_guard(frame: pd.DataFrame) -> dict:
    """
    守りの結果と理由の数をまとめる関数

    Parameters
    ----------
    frame: pd.DataFrame
        守りの列のある表

    Returns
    ----------
    guard: dict
        results (結果ごとの行の数. 知らない結果もそのまま数え, 空は
        missing), reasons (理由ごとの行の数の並び. 多い順)
    """
    # 結果ごとの行の数
    results = {}
    for value in frame["guard"]:
        key = "missing" if value is None or pd.isna(value) else str(value)
        results[key] = results.get(key, 0) + 1

    # 理由は ; でつないだものを分けて数える
    reasons = {}
    if "guard_reasons" in frame:
        for text in frame["guard_reasons"].dropna():
            for reason in str(text).split(REASON_SEPARATOR):
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
    guard = {
        "results": results,
        "reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                reasons.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ],
    }

    return guard


def describe_action_prob(series: pd.Series) -> dict:
    """
    行動を選んだ確率の列の欠けと範囲をまとめる関数

    Parameters
    ----------
    series: pd.Series
        確率の列

    Returns
    ----------
    prob: dict
        missing (空の数), out_of_range (0 より大きく 1 以下でない値の数.
        nan と無限大も含む), 範囲の中の値の min, max, mean, std, median
    """
    # 空を除いた値を, 範囲の中と外に分ける
    values, missing = float_values(series)
    present = values[~missing]
    inside = (present > 0.0) & (present <= 1.0)
    prob = {
        "missing": int(missing.sum()),
        "out_of_range": int((~inside).sum()),
        **number_summary(present[inside]),
    }

    return prob


def describe_trajectory(trajectory: Trajectory) -> dict:
    """
    軌跡のエピソード, 長さ, 報酬, 観測と行動, 守りの結果, 行動を選んだ
    確率をまとめる関数

    Parameters
    ----------
    trajectory: Trajectory
        load_trajectory で読んだ軌跡

    Returns
    ----------
    summary: dict
        JSON にできる辞書. 鍵は docs/DESIGN.md の「記録した軌跡」
    """
    # エピソードごとの値
    frame = trajectory.frame
    episodes = episode_rows(trajectory)
    lengths = np.array([row["length"] for row in episodes], dtype=float)
    returns = np.array(
        [row["return"] for row in episodes if row["return"] is not None],
        dtype=float
    )
    ends = dict.fromkeys(
        ("terminated", "truncated", "stop", "handover", "open"), 0
    )
    for row in episodes:
        ends[row["end"]] += 1

    # 説明の項目 (無ければ None)
    info = trajectory.info or {}
    known = set(KNOWN_TYPES) | set(trajectory.observation_columns) | set(
        trajectory.action_columns
    )
    summary = {
        "name": trajectory.name,
        "format": trajectory.format,
        "described": trajectory.info is not None,
        **{
            key: info.get(key) for key in (
                "format_version", "source", "env", "epoch", "description",
                "complete", "dropped_episodes", "observation_space",
                "action_space"
            )
        },
        "rows": int(len(frame)),
        "episodes": len(episodes),
        "steps": int(lengths.sum()),
        "columns": [
            {"name": name, "type": trajectory.types[name]}
            for name in frame.columns
        ],
        "observation_columns": list(trajectory.observation_columns),
        "action_columns": list(trajectory.action_columns),
        "extra_columns": [name for name in frame.columns if name not in known],
        "guarded": trajectory.guarded,
        "with_action_prob": trajectory.with_action_prob,
        "length": number_summary(lengths),
        "return": number_summary(returns),
        "unknown_returns": int(len(episodes) - returns.size),
        "ends": ends,
        "observations": [
            describe_element(frame[name], name, trajectory.types[name])
            for name in trajectory.observation_columns
        ],
        "actions": [
            describe_element(frame[name], name, trajectory.types[name])
            for name in trajectory.action_columns
        ],
        "guard": describe_guard(frame) if trajectory.guarded else None,
        "action_prob": describe_action_prob(
            frame[PROB_COLUMN]
        ) if trajectory.with_action_prob else None,
        "per_episode": episodes[:MAX_EPISODE_ROWS],
        "per_episode_truncated": len(episodes) > MAX_EPISODE_ROWS,
        "notes": list(trajectory.notes),
    }

    # 範囲の外の確率 (方策の見積もりに使えない値)
    prob = summary["action_prob"]
    if prob is not None and prob["out_of_range"]:
        summary["notes"].append(
            f"{PROB_COLUMN} に 0 より大きく 1 以下でない値が "
            f"{prob['out_of_range']} 件あります"
        )

    # エピソードごとの並びは先頭の上限まで (要約の値は全てのエピソードで)
    if summary["per_episode_truncated"]:
        summary["notes"].append(
            f"エピソードが {len(episodes)} 件あるので, per_episode には先頭"
            f"の {MAX_EPISODE_ROWS} 件だけを並べます"
        )
    summary = to_json(summary)

    return summary
