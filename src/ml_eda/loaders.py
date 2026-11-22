import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.data import Data

# 中身の推定に読む先頭のバイト数
SNIFF_BYTES = 64 * 1024

# 数の前に付けてよい通貨の記号と, 後に付けてよい単位 (これ以外の文字が
# 付いた値は, 型番や ID かもしれないので数にしない)
CURRENCIES = "¥$€£"
UNITS = (
    "万円", "千円", "円", "ドル", "%", "個", "件", "人", "本", "回", "点", "歳",
    "冊", "台", "枚", "杯", "時間", "分", "秒", "日", "kg", "mg", "g", "km",
    "cm", "mm", "m", "mL", "ml", "L"
)

# 設定 follow_links の説明 (画像, 点群, 撮影, 音声の読み込み)
FOLLOW_LINKS_HELP = (
    "読み込んだディレクトリの外を指すシンボリックリンクの先を読むか (既定 "
    "false. false では読めない件として理由を残す. 読み込んだディレクトリ"
    "自体がリンクの場合は, その先の中を読む)"
)

# 試す文字コード (日本語の csv によくあるものを含む)
ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "euc-jp", "latin-1")


@dataclass
class Loader:
    # 読み込みの名前
    name: str
    # 主に返す種類
    kind: str
    # 扱う拡張子 (小文字, 点付き)
    extensions: tuple
    # 読めるかの確からしさ (0 から 1) を返す関数
    sniff: Callable[[Path], float]
    # Data を返す関数
    load: Callable[..., Data]
    # 読み込みの設定の説明 (設定の名前から説明)
    options: dict = field(default_factory=dict)
    # 説明
    description: str = ""


# 読み込みの登録簿
LOADERS: dict[str, Loader] = {}


def register_loader(loader: Loader) -> None:
    """
    読み込みを登録簿に加える関数

    Parameters
    ----------
    loader: Loader
        加える読み込み (同じ名前があれば置き換え)

    Returns
    ----------
    None
    """
    # 名前で登録
    LOADERS[loader.name] = loader


def describe_loaders() -> list[dict]:
    """
    登録した読み込みの一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    loaders: list[dict]
        名前, 種類, 拡張子, 設定, 説明の辞書の並び
    """
    # JSON にできる形に変換
    loaders = [
        {
            "name": loader.name, "kind": loader.kind,
            "extensions": list(loader.extensions),
            "options": dict(loader.options),
            "description": loader.description,
        }
        for loader in LOADERS.values()
    ]

    return loaders


def option_flag(options: dict, name: str, default: bool) -> bool:
    """
    読み込みの真偽の設定を読む関数 (画面からの文字も読む)

    Parameters
    ----------
    options: dict
        読み込みの設定
    name: str
        設定の名前
    default: bool
        無い場合の値

    Returns
    ----------
    flag: bool
        設定の値 (文字の false, 0, no と空は False)
    """
    # 画面からの設定は文字で来る
    flag = options.get(name, default)
    if isinstance(flag, str):
        flag = flag.strip().lower() not in ("false", "0", "no", "")
    flag = bool(flag)

    return flag


def points_outside(base: Path, path: Path) -> bool:
    """
    リンクをたどった後の場所が, 読み込んだディレクトリの外かを返す関数

    Parameters
    ----------
    base: Path
        読み込んだディレクトリ (それ自体のリンクもたどって比べる)
    path: Path
        確かめる場所

    Returns
    ----------
    outside: bool
        たどった後の path が, たどった後の base の下に無いか (リンクが
        輪になってたどれない場合も True)
    """
    # 両方のリンクをたどってから比べる
    try:
        outside = not path.resolve().is_relative_to(base.resolve())
    except (OSError, RuntimeError):
        outside = True

    return outside


def outside_message(name: str) -> str:
    """
    外を指すリンクを読まない理由の文を返す関数

    Parameters
    ----------
    name: str
        件の場所 (読み込んだディレクトリからの相対)

    Returns
    ----------
    message: str
        理由と, 読む場合の設定の案内
    """
    # 設定 follow_links で読めることを添える
    message = (
        f"{name} は読み込んだディレクトリの外を指すリンクなので読みません "
        "(読むには読み込みの設定 follow_links を true にしてください)"
    )

    return message


def check_inside(base: Optional[Path], path: Path, name: str) -> Path:
    """
    読み込んだディレクトリの外を指すリンクを誤りにする関数

    Parameters
    ----------
    base: Optional[Path]
        読み込んだディレクトリ (None ならリンクの先をたどって読む)
    path: Path
        読む場所
    name: str
        誤りに書く件の場所

    Returns
    ----------
    path: Path
        中にあれば渡した場所

    Raises
    ----------
    ValueError
        リンクをたどった後に外を指す場合
    """
    # たどる設定でなければ, たどった後の場所で比べる
    if base is not None and points_outside(base, path):
        raise ValueError(outside_message(name))

    return path


def detect(path: Union[str, Path]) -> list[dict]:
    """
    場所を読めそうな読み込みを, 確からしい順に返す関数

    Parameters
    ----------
    path: Union[str, Path]
        ファイルかディレクトリ

    Returns
    ----------
    candidates: list[dict]
        読み込みの名前, 種類, 確からしさの辞書の並び (0 のものは除く)

    Raises
    ----------
    FileNotFoundError
        場所が無い場合
    """
    # 全ての読み込みに確からしさを聞く
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"{target} がありません")
    candidates = []
    for loader in LOADERS.values():
        try:
            score = float(loader.sniff(target))
        except Exception:
            score = 0.0
        if score > 0:
            candidates.append(
                {"loader": loader.name, "kind": loader.kind, "score": score}
            )
    candidates.sort(key=lambda item: -item["score"])

    return candidates


def load_data(
    path: Union[str, Path], loader: Optional[str] = None, **options: Any
) -> Data:
    """
    ファイルかディレクトリを読み込む関数

    Parameters
    ----------
    path: Union[str, Path]
        ファイルかディレクトリ
    loader: Optional[str] = None
        読み込みの名前, None の場合は最も確からしいものを選ぶ
    **options: Any
        読み込みの設定 (区切り文字, 日時の列など)

    Returns
    ----------
    data: Data
        読み込んだデータ

    Raises
    ----------
    ValueError
        読める読み込みが無い場合と, 知らない読み込みの名前の場合
    """
    # 読み込みの選択
    target = Path(path).expanduser()
    if loader is None:
        candidates = detect(target)
        if not candidates:
            raise ValueError(
                f"{target.name} を読める読み込みがありません (対応する形式: "
                f"{', '.join(sorted(LOADERS))})"
            )
        loader = candidates[0]["loader"]
    if loader not in LOADERS:
        raise ValueError(f"知らない読み込みです: {loader}")

    # 読み込みと, 場所と設定の記録
    data = LOADERS[loader].load(target, **options)
    stat = target.stat()
    data.source = {
        "path": str(target.resolve()),
        "loader": loader,
        "options": {key: value for key, value in options.items()},
        "size": stat.st_size,
        "modified": stat.st_mtime,
        **data.source,
    }

    return data


def read_head(path: Path, size: int = SNIFF_BYTES) -> bytes:
    """
    ファイルの先頭を読む関数

    Parameters
    ----------
    path: Path
        ファイル
    size: int = SNIFF_BYTES
        読むバイト数

    Returns
    ----------
    head: bytes
        先頭のバイト列
    """
    # 先頭だけを読み込み
    with open(path, "rb") as file:
        head = file.read(size)

    return head


def decodes(head: bytes, encoding: str) -> Optional[str]:
    """
    先頭のバイト列を文字コードで読めるか試す関数

    Parameters
    ----------
    head: bytes
        ファイルの先頭
    encoding: str
        文字コード

    Returns
    ----------
    text: Optional[str]
        読めた文字 (末尾で文字が途中で切れていても読む), 読めなければ None
    """
    # 末尾の最大 3 バイトを除いて試す
    for cut in range(0, 4):
        try:
            return (head[:len(head) - cut] if cut else head).decode(encoding)
        except UnicodeDecodeError:
            continue

    return None


def guess_encoding(head: bytes) -> str:
    """
    先頭のバイト列から文字コードを推定する関数

    Parameters
    ----------
    head: bytes
        ファイルの先頭

    Returns
    ----------
    encoding: str
        推定した文字コード
    """
    # UTF-8 で読めればそれ. 日本語の 2 つ (cp932 と euc-jp) は両方で
    # 読める場合があるので, 半角カナ (文字化けの印) の少ない方を選ぶ
    for encoding in ("utf-8-sig", "utf-8"):
        if decodes(head, encoding) is not None:
            return encoding
    readable = {
        encoding: text for encoding in ("cp932", "euc-jp")
        if (text := decodes(head, encoding)) is not None
    }
    if readable:
        return min(
            readable,
            key=lambda e: sum("\uff61" <= c <= "\uff9f" for c in readable[e])
        )
    encoding = "latin-1"

    return encoding


def guess_delimiter(text: str) -> str:
    """
    表の文字から区切り文字を推定する関数

    Parameters
    ----------
    text: str
        ファイルの先頭の文字

    Returns
    ----------
    delimiter: str
        区切り文字 (カンマ, タブ, セミコロン, 縦棒, 空白のどれか)
    """
    # csv.Sniffer で推定し, 失敗した場合は行ごとの数が揃う区切りを選ぶ
    lines = [line for line in text.splitlines()[:50] if line.strip()]
    sample = "\n".join(lines)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        return delimiter
    except csv.Error:
        pass
    best, best_score = ",", -1.0
    for candidate in (",", "\t", ";", "|", " "):
        counts = [line.count(candidate) for line in lines]
        if not counts or max(counts) == 0:
            continue
        # 行ごとの数が同じほど, 多いほど良い区切り
        score = max(counts) * (counts.count(counts[0]) / len(counts))
        if score > best_score:
            best, best_score = candidate, score
    delimiter = best

    return delimiter


# 系列を分ける列の値の種類の上限 (店や商品, センサーごとなど)
MAX_SERIES = 10000


def panel_of(frame: pd.DataFrame, time: str) -> Optional[str]:
    """
    日時の列と組にすると行が重ならない, 系列を分ける列を探す関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    time: str
        日時の列

    Returns
    ----------
    name: Optional[str]
        文字の列で, 値が 2 から 10000 種類, 1 つの値あたり平均 10 行以上,
        日時の列との組の重なりが 5 % 未満, 1 割以上の時刻に 2 つ以上の値が
        並ぶ最初の列. 無ければ None
    """
    # 系列ごとに十分な長さがあり, 同じ系列の中で時刻が重ならず, 時刻と系列
    # の組の半分以上が埋まっている列 (注文の表の顧客は, ほとんどの日に注文
    # が無いので系列にしない). 整数の店の番号なども候補
    times = frame[time].nunique()
    if times < 10:
        return None
    for name in frame.columns:
        column = frame[name]
        if name == time or pd.api.types.is_float_dtype(column) or (
            pd.api.types.is_datetime64_any_dtype(column)
            or pd.api.types.is_bool_dtype(column)
        ):
            continue
        count = column.nunique()
        if not (2 <= count <= MAX_SERIES and len(frame) / count >= 10):
            continue
        pairs = frame[[time, name]].drop_duplicates().shape[0]
        if frame.duplicated([time, name]).mean() >= 0.05 or (
            pairs < 0.5 * times * count
        ):
            continue
        # 系列は同じ時刻に並ぶもの (時刻が行ごとに違うレビューのラベルは,
        # 2 つの値で埋まり方が 5 割になっても系列にしない). 開始日のずれや
        # 週末だけの営業も系列にするよう, 並ぶ時刻は 1 割以上
        shared = (frame.groupby(time)[name].nunique() >= 2).mean()
        if shared >= 0.1:
            return str(name)

    return None


def table_kind(frame: pd.DataFrame) -> tuple[str, dict]:
    """
    表が時系列かを推定し, 日時の列を時刻の順に並べる関数

    Parameters
    ----------
    frame: pd.DataFrame
        読み込んだ表 (日時の列は変換を試みて書き換える)

    Returns
    ----------
    kind: str
        table か timeseries
    roles: dict
        推定した役割 (時系列なら time. 複数の系列なら系列を分ける列 series)
    """
    # 日時に変換できる列を探す (数値の列は除く)
    roles: dict = {}
    for name in frame.columns:
        column = frame[name]
        if pd.api.types.is_datetime64_any_dtype(column):
            converted = column
        elif pd.api.types.is_object_dtype(column) or (
            pd.api.types.is_string_dtype(column)
        ):
            converted = to_datetime(column)
            if converted is None:
                continue
            frame[name] = converted
        else:
            continue
        # 異なる時刻が多い最初の列を日時の列とする (他の日時の列も変換).
        # 店ごとの売上のように, 系列を分ける列と組にすると時刻が重ならない
        # 場合も日時の列とする
        valid = converted.dropna()
        if "time" not in roles and len(valid) >= 3:
            # 時刻が重なる場合は, まず系列を分ける列を探す
            series = (
                panel_of(frame, str(name))
                if valid.nunique() < len(valid) else None
            )
            if series is not None:
                roles["time"], roles["series"] = str(name), series
            elif valid.nunique() >= 0.5 * len(valid):
                roles["time"] = str(name)
    kind = "timeseries" if "time" in roles else "table"

    return kind, roles


def japanese_date(value: str) -> str:
    """
    日本語の日時の書き方 (2024年1月3日12時30分15秒, 2024年1月) を,
    pandas で読める書き方 (2024-1-3 12:30:15) に直す関数

    Parameters
    ----------
    value: str
        値

    Returns
    ----------
    text: str
        直した値 (日本語の日時でなければそのまま)
    """
    # 文字でない値 (欠損) はそのまま
    if not isinstance(value, str):
        return value

    # 年月日, 年月, 時分秒 (欠けた分と秒は 0)
    text = re.sub(r"(\d{4})年(\d{1,2})月(\d{1,2})日?\s*", r"\1-\2-\3 ", value)
    text = re.sub(r"(\d{4})年(\d{1,2})月(?!\d)", r"\1-\2", text)
    text = re.sub(
        r"(\d{1,2})時(?:(\d{1,2})分)?(?:(\d{1,2})秒)?",
        lambda m: f"{m[1]}:{m[2] or 0}:{m[3] or 0}", text
    )
    text = text.strip()

    return text


def to_datetime(column: pd.Series) -> Optional[pd.Series]:
    """
    文字の列を日時に変換できれば変換する関数

    Parameters
    ----------
    column: pd.Series
        文字の列

    Returns
    ----------
    converted: Optional[pd.Series]
        8 割以上が日時として読めた場合は変換した列, そうでなければ None
    """
    # 日付の区切り (- / : 年) を含まない値 (数やカンマ付きの数) は日時に
    # しない. 時差の違う値が混ざる場合は UTC にそろえる
    values = column.dropna().astype(str)
    if values.empty:
        return None
    dated = values.str.contains(r"\d[-/:年]\d|\d年|\d{4}-\d", regex=True)
    if dated.mean() < 0.8:
        return None
    # 2024年1月3日 (と 12時30分) の書き方は, - と : の書き方に直してから読む
    column = column.where(column.isna(), column.astype(str).map(japanese_date))
    try:
        converted = pd.to_datetime(column, errors="coerce", format="mixed")
        if not pd.api.types.is_datetime64_any_dtype(converted):
            converted = pd.to_datetime(
                column, errors="coerce", format="mixed", utc=True
            )
    except (ValueError, TypeError):
        try:
            converted = pd.to_datetime(
                column, errors="coerce", format="mixed", utc=True
            )
        except (ValueError, TypeError):
            return None
    years = converted.dropna().dt.year
    plausible = years.between(1800, 2200).mean() if len(years) else 0
    if converted.notna().sum() < 0.8 * column.notna().sum() or (
        plausible < 0.8
    ):
        return None

    return converted


def to_numbers(column: pd.Series) -> Optional[pd.Series]:
    """
    桁区切りのカンマ付きの数の列を数に変換する関数

    Parameters
    ----------
    column: pd.Series
        文字の列

    Returns
    ----------
    converted: Optional[pd.Series]
        8 割以上が 1,234 の形の数なら変換した列, そうでなければ None
    """
    # 1,234 や -12,345.6 の形と, 前後に同じ単位の付いた形 (¥1,200, 1,200円,
    # 12%, 3.5kg) を数にする. 全角の数字は半角に
    values = column.dropna().astype(str).map(
        lambda v: unicodedata.normalize("NFKC", v).strip()
    )
    if values.empty:
        return None
    number = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?"
    units = "|".join(re.escape(unit) for unit in UNITS)
    parts = values.str.extract(
        rf"^(?P<sign>[+-]?)\s*(?P<pre>[{re.escape(CURRENCIES)}]?)\s*"
        rf"(?P<num>{number})\s*(?P<suf>(?:{units})?)$"
    )
    matched = parts["num"].notna()
    affix = (parts["pre"].fillna("") + "|" + parts["suf"].fillna(""))[matched]
    if matched.mean() < 0.8 or affix.empty:
        return None
    common = affix.value_counts()
    unit = common.index[0]
    if common.iloc[0] < 0.8 * len(values):
        return None
    if unit == "|" and not values.str.contains(",").any():
        return None
    # 単位の付いた形は, 読めた値が全て同じ単位の場合だけ (1kg と 500g の
    # ように単位の違う値を欠損にしない). 0 で始まる値 (001件) は ID とみて
    # 数にしない
    if unit != "|" and (
        len(common) > 1
        or parts["num"][matched].str.match(r"[+-]?0\d").any()
    ):
        return None
    keep = matched & (
        parts["pre"].fillna("") + "|" + parts["suf"].fillna("") == unit
    )
    numbers = pd.to_numeric(
        parts["num"].where(keep).str.replace(",", "", regex=False),
        errors="coerce"
    )
    numbers = numbers.where(parts["sign"] != "-", -numbers)
    converted = pd.Series(np.nan, index=column.index, dtype=float)
    converted[numbers.index] = numbers

    return converted


def unique_names(names: list) -> list[str]:
    """
    列の名前を文字にし, 重なる名前に番号を付ける関数

    Parameters
    ----------
    names: list
        列の名前

    Returns
    ----------
    unique: list[str]
        重ならない文字の名前 (2 つ目からは name.1 のように付ける)
    """
    # 出てきた回数を数えて番号を付ける
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        text = str(name)
        count = seen.get(text, 0)
        seen[text] = count + 1
        unique.append(text if count == 0 else f"{text}.{count}")

    return unique


def finish_table(frame: pd.DataFrame, options: dict) -> Data:
    """
    読み込んだ表から, 種類と役割を決めて Data にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        読み込んだ表
    options: dict
        読み込みの設定 (time_column で日時の列, text_column で文の列,
        kind: table で表として読むことを指定できる)

    Returns
    ----------
    data: Data
        表, 時系列, 文のデータ
    """
    # 列の名前を重ならない文字にし, カンマ付きの数を数にする
    frame.columns = unique_names(list(frame.columns))
    for name in frame.columns:
        column = frame[name]
        if pd.api.types.is_object_dtype(column) or (
            pd.api.types.is_string_dtype(column)
        ):
            numbers = to_numbers(column)
            if numbers is not None:
                frame[name] = numbers

    # 文の列の指定があれば文のデータ (1 行が 1 つの文. 他の列も残す)
    text_column = options.get("text_column")
    if text_column:
        return text_table(frame, text_column)

    # 日時の列の指定があればそれを, 無ければ推定
    time_column = options.get("time_column")
    if time_column:
        if time_column not in frame.columns:
            raise ValueError(f"日時の列 {time_column} がありません")
        frame[time_column] = pd.to_datetime(
            frame[time_column], errors="coerce", format="mixed", utc=True
        ) if to_datetime(frame[time_column]) is None else to_datetime(
            frame[time_column]
        )
        kind, roles = "timeseries", {"time": time_column}
    elif options.get("kind") == "table":
        kind, roles = "table", {}
    else:
        kind, roles = table_kind(frame)

    # 時系列は時刻の順に並べる
    if kind == "timeseries":
        frame = frame.sort_values(roles["time"], kind="stable")
        frame = frame.reset_index(drop=True)
    data = Data(kind=kind, frame=frame, roles=roles)

    return data


def text_table(frame: pd.DataFrame, text_column: str) -> Data:
    """
    表の 1 つの列を文にした, 文のデータを作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        読み込んだ表
    text_column: str
        文の列

    Returns
    ----------
    data: Data
        文のデータ (元の列と, 文の長さ length と語の数 words を持つ.
        欠損の文は空の文)

    Raises
    ----------
    ValueError
        文の列が無い場合
    """
    # 欠損は空の文にし, 分析と手順の数の決め方に使う長さと語の数を足す
    if text_column not in frame.columns:
        raise ValueError(f"文の列 {text_column} がありません")
    frame[text_column] = frame[text_column].fillna("").astype(str)
    names = unique_names([*frame.columns, "length", "words"])
    length, words = names[-2], names[-1]
    frame[length] = frame[text_column].str.len()
    frame[words] = frame[text_column].str.split().map(len)
    roles = {"text": text_column, "length": length, "words": words}
    column = frame[text_column]
    data = Data(
        kind="text", frame=frame, roles=roles,
        item_loader=lambda i: column.iloc[i]
    )

    return data


def sniff_csv(path: Path) -> float:
    """
    区切り文字の表として読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # 拡張子と, 先頭の行の区切りの揃い方で判定
    if not path.is_file():
        return 0.0
    suffix = path.suffix.lower()
    head = read_head(path)
    if b"\x00" in head[:4096]:
        return 0.0
    text = head.decode(guess_encoding(head), errors="replace")
    lines = [line for line in text.splitlines()[:20] if line.strip()]
    if len(lines) < 2:
        return 0.2 if suffix in (".csv", ".tsv") else 0.0
    delimiter = guess_delimiter(text)
    counts = {line.count(delimiter) for line in lines[:-1]}
    regular = len(counts) == 1 and counts != {0}
    if suffix in (".csv", ".tsv"):
        score = 0.9 if regular else 0.6
    elif suffix in (".txt", ".dat", ".data", ""):
        score = 0.6 if regular and delimiter != " " else 0.0
    else:
        score = 0.0

    return score


def load_csv(path: Path, **options: Any) -> Data:
    """
    区切り文字の表を読み込む関数

    Parameters
    ----------
    path: Path
        ファイル
    **options: Any
        sep (区切り文字), encoding (文字コード), header (ヘッダーの行,
        無い場合は None), time_column (日時の列), kind (table で時系列の
        推定を止める)

    Returns
    ----------
    data: Data
        表か時系列のデータ
    """
    # 文字コードと区切り文字の推定
    head = read_head(path)
    encoding = options.get("encoding") or guess_encoding(head)
    text = head.decode(encoding, errors="replace")
    sep = options.get("sep") or guess_delimiter(text)
    header = options.get("header", "infer")

    # 読み込み (空白区切りは連続した空白を 1 つとして扱う). 先頭より後で
    # 読めない文字があれば, 次の文字コードで読み直す
    order = [encoding] + [e for e in ENCODINGS if e != encoding]
    if options.get("encoding"):
        order = [encoding]
    for candidate in order:
        try:
            frame = pd.read_csv(
                path, sep=r"\s+" if sep == " " else sep, encoding=candidate,
                header=header, engine="python" if sep == " " else "c",
                **({} if sep == " " else {"low_memory": False})
            )
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"{path.name} は空です") from error
    data = finish_table(frame, options)
    data.source = {"encoding": encoding, "sep": sep}

    return data


def sniff_json(path: Path) -> float:
    """
    JSON の表として読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # 拡張子と先頭の文字で判定
    if not path.is_file():
        return 0.0
    suffix = path.suffix.lower()
    head = read_head(path, 4096).lstrip()
    if suffix in (".jsonl", ".ndjson"):
        return 0.9 if head.startswith(b"{") else 0.3
    if suffix == ".json":
        return 0.9 if head[:1] in (b"[", b"{") else 0.2
    if head[:1] == b"{" and b"\n{" in head:
        return 0.5

    return 0.0


def load_json(path: Path, **options: Any) -> Data:
    """
    JSON (レコードの配列, 列の辞書, 1 行 1 レコード) を表として読む関数

    Parameters
    ----------
    path: Path
        ファイル
    **options: Any
        records (レコードの配列がある鍵の名前, 入れ子の場合),
        time_column, kind

    Returns
    ----------
    data: Data
        表か時系列のデータ
    """
    # 1 行 1 レコードの形
    text = path.read_text(encoding=options.get("encoding", "utf-8-sig"))
    stripped = text.lstrip()
    if path.suffix.lower() in (".jsonl", ".ndjson") or (
        stripped.startswith("{") and "\n{" in stripped
        and not looks_like_single_object(stripped)
    ):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        frame = pd.json_normalize(rows)
        return finish_table(frame, options)

    # レコードの配列か, 配列を持つ辞書か, 列の辞書の形
    value = json.loads(text)
    records = options.get("records")
    if records:
        value = value[records]
    elif isinstance(value, dict):
        lists = [key for key, item in value.items() if isinstance(item, list)]
        if len(lists) == 1 and all(
            isinstance(row, dict) for row in value[lists[0]][:10]
        ):
            value = value[lists[0]]
    if isinstance(value, list):
        frame = pd.json_normalize(value)
    elif isinstance(value, dict):
        frame = pd.DataFrame(value)
    else:
        raise ValueError("JSON の中身が表の形 (配列か辞書) ではありません")
    data = finish_table(frame, options)

    return data


def looks_like_single_object(text: str) -> bool:
    """
    文字が 1 つの JSON の値 (整形された辞書など) かを判定する関数

    Parameters
    ----------
    text: str
        ファイルの中身

    Returns
    ----------
    single: bool
        全体が 1 つの JSON として読めるか
    """
    # 全体を JSON として読めるかで判定
    try:
        json.loads(text)
        single = True
    except json.JSONDecodeError:
        single = False

    return single


def sniff_parquet(path: Path) -> float:
    """
    parquet として読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # 拡張子と先頭の印で判定
    if not path.is_file():
        return 0.0
    score = 0.95 if read_head(path, 4) == b"PAR1" else 0.0

    return score


def load_parquet(path: Path, **options: Any) -> Data:
    """
    parquet を読み込む関数

    Parameters
    ----------
    path: Path
        ファイル
    **options: Any
        time_column, kind

    Returns
    ----------
    data: Data
        表か時系列のデータ
    """
    # pyarrow で読み込み
    try:
        frame = pd.read_parquet(path)
    except ImportError as error:
        raise ImportError(
            "parquet を読むには pyarrow が要ります (pip install pyarrow)"
        ) from error
    data = finish_table(frame, options)

    return data


def sniff_excel(path: Path) -> float:
    """
    Excel として読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # 拡張子と, xlsx の中身の zip の印で判定
    if not path.is_file() or path.suffix.lower() not in (".xlsx", ".xlsm"):
        return 0.0
    score = 0.9 if read_head(path, 2) == b"PK" else 0.0

    return score


def load_excel(path: Path, **options: Any) -> Data:
    """
    Excel のシートを読み込む関数

    Parameters
    ----------
    path: Path
        ファイル
    **options: Any
        sheet (シートの名前か番号, 既定は最初のシート), time_column, kind

    Returns
    ----------
    data: Data
        表か時系列のデータ
    """
    # openpyxl で 1 つのシートを読み込み, 全てのシートの名前も記録 (他の
    # シートがあることを気付きにするため)
    sheet = options.get("sheet", 0)
    try:
        with pd.ExcelFile(path) as book:
            names = [str(name) for name in book.sheet_names]
            # 番号は範囲の中, 名前は有るものだけ (無いものは日本語の誤り)
            if isinstance(sheet, (int, np.integer)):
                if not -len(names) <= int(sheet) < len(names):
                    raise ValueError(
                        f"シートの番号 {sheet} がありません (0 から "
                        f"{len(names) - 1}. シート: {', '.join(names)})"
                    )
                current = names[int(sheet)]
                sheet = int(sheet)
            elif str(sheet) in names:
                current = sheet = str(sheet)
            else:
                raise ValueError(
                    f"シート {sheet} がありません (シート: {', '.join(names)})"
                )
            frame = pd.read_excel(book, sheet_name=sheet)
    except ImportError as error:
        raise ImportError(
            "Excel を読むには openpyxl が要ります (pip install openpyxl)"
        ) from error
    data = finish_table(frame, options)
    data.source = {**data.source, "sheets": names, "sheet": current}

    return data


def sniff_npy(path: Path) -> float:
    """
    numpy の配列のファイルとして読めるかの確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # 拡張子と先頭の印で判定
    if not path.is_file():
        return 0.0
    head = read_head(path, 6)
    if path.suffix.lower() == ".npy" and head == b"\x93NUMPY":
        return 0.8
    if path.suffix.lower() == ".npz" and head[:2] == b"PK":
        return 0.8

    return 0.0


def load_npy(path: Path, **options: Any) -> Data:
    """
    numpy の 1 次元か 2 次元の配列を表として読む関数

    Parameters
    ----------
    path: Path
        ファイル (.npy か .npz. pickle は使わない)
    **options: Any
        key (.npz の中の配列の名前, 既定は最初の配列), columns (列の名前)

    Returns
    ----------
    data: Data
        表のデータ

    Raises
    ----------
    ValueError
        3 次元以上の配列の場合
    """
    # pickle を許さずに読み込み
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        key = options.get("key") or loaded.files[0]
        array = loaded[key]
    else:
        array = loaded
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(
            f"表として読めるのは 1 次元か 2 次元の配列です (形 {array.shape})"
        )

    # 列の名前を付けて表にする
    columns = unique_names(options.get("columns") or [
        f"x{index}" for index in range(array.shape[1])
    ])
    frame = pd.DataFrame(array, columns=columns)
    data = Data(kind="table", frame=frame)

    return data


def register_builtin() -> None:
    """
    組み込みの読み込みを登録する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 表と時系列の読み込み
    table_options = {
        "time_column": "日時の列 (時系列として読む. 省略すると推定)",
        "kind": "table と書くと時系列の推定をしない",
    }
    register_loader(Loader(
        "csv", "table", (".csv", ".tsv", ".txt", ".dat", ".data"),
        sniff_csv, load_csv,
        {"sep": "区切り文字 (省略すると推定)",
         "encoding": "文字コード (省略すると推定)",
         "header": "ヘッダーの行 (無い場合は null)", **table_options},
        "区切り文字の表 (csv, tsv など)"
    ))
    register_loader(Loader(
        "json", "table", (".json", ".jsonl", ".ndjson"), sniff_json,
        load_json,
        {"records": "レコードの配列がある鍵 (入れ子の場合)", **table_options},
        "JSON (レコードの配列, 列の辞書, 1 行 1 レコード)"
    ))
    register_loader(Loader(
        "parquet", "table", (".parquet",), sniff_parquet, load_parquet,
        dict(table_options), "parquet (pyarrow が要る)"
    ))
    register_loader(Loader(
        "excel", "table", (".xlsx", ".xlsm"), sniff_excel, load_excel,
        {"sheet": "シートの名前か番号 (既定は最初)", **table_options},
        "Excel のシート (openpyxl が要る)"
    ))
    register_loader(Loader(
        "npy", "table", (".npy", ".npz"), sniff_npy, load_npy,
        {"key": ".npz の中の配列の名前", "columns": "列の名前の並び"},
        "numpy の 1 次元か 2 次元の配列"
    ))


register_builtin()
