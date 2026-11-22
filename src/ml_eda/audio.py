import struct
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.loaders import Loader, register_loader
from ml_eda.media import MAX_FILES, labels_of, list_files

# 標準ライブラリと numpy で読む拡張子 (WAVE)
WAVE_EXTENSIONS = (".wav", ".wave")

# soundfile (libsndfile) で読む拡張子 (FLAC, MP3, Ogg, AIFF)
LIBRARY_EXTENSIONS = (".flac", ".mp3", ".ogg", ".opus", ".aif", ".aiff")

# 音声の拡張子
AUDIO_EXTENSIONS = WAVE_EXTENSIONS + LIBRARY_EXTENSIONS

# libsndfile の中身の形式 (subtype) ごとの形式と bit 数 (他は compressed)
SUBTYPES = {
    "PCM_S8": ("pcm", 8), "PCM_U8": ("pcm", 8), "PCM_16": ("pcm", 16),
    "PCM_24": ("pcm", 24), "PCM_32": ("pcm", 32), "FLOAT": ("float", 32),
    "DOUBLE": ("float", 64),
}

# MP3 を読める libsndfile の版
MP3_LIBSNDFILE = (1, 1)

# 形式の番号 (WAVE の fmt の欄)
PCM = 1
IEEE_FLOAT = 3
EXTENSIBLE = 0xFFFE

# 無音とする RMS (満点を 1 とした値. 約 -60 dBFS)
SILENT_RMS = 1e-3

# 音の割れとする標本の割合 (振幅の端は clip_level で形式ごとに決める)
CLIPPED_SHARE = 0.01

# 長さのばらつきを気付きにする, 最も長いものと最も短いものの比
LENGTH_RATIO = 2.0

# 分析で 1 つの音声から読む長さ (秒. 先頭から. 長い録音で記憶を使い
# すぎないように)
ANALYSIS_SECONDS = 30.0


def wave_chunks(path: Path) -> dict:
    """
    WAVE のファイルの fmt と data の欄の位置と中身を読む関数

    Parameters
    ----------
    path: Path
        WAVE のファイル

    Returns
    ----------
    chunks: dict
        format (pcm か float), sample_rate, channels, bits, frames,
        offset (data の始まり), size (data のバイト数)

    Raises
    ----------
    ValueError
        WAVE でない場合, fmt か data の欄が無い場合, 読めない形式の場合,
        1 つの標本の組のバイト数 (block align) が足りない場合
    """
    # RIFF の頭と, 欄を順にたどる (data の中身は読まない)
    with open(path, "rb") as stream:
        head = stream.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{path.name} は WAVE (RIFF) ではありません")
        fmt = None
        data = None
        while data is None:
            header = stream.read(8)
            if len(header) < 8:
                break
            name, size = header[:4], struct.unpack("<I", header[4:])[0]
            if name == b"fmt ":
                fmt = stream.read(size)
                stream.seek(size % 2, 1)
            elif name == b"data":
                data = (stream.tell(), size)
            else:
                stream.seek(size + size % 2, 1)
        file_size = stream.seek(0, 2)
    if fmt is None or len(fmt) < 16 or data is None:
        raise ValueError(f"{path.name} に fmt か data の欄がありません")

    # 形式 (拡張の形式は中の形式の番号)
    tag, channels, rate, _, align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == EXTENSIBLE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]
    kinds = {PCM: "pcm", IEEE_FLOAT: "float"}
    if tag not in kinds:
        raise ValueError(
            f"{path.name} の形式 ({tag}) は読めません (PCM と浮動小数だけ)"
        )
    if channels < 1 or rate < 1 or align < 1 or bits not in (8, 16, 24, 32):
        raise ValueError(f"{path.name} の fmt の値が正しくありません")
    if kinds[tag] == "float" and bits != 32:
        raise ValueError(f"{path.name} の浮動小数は 32 bit だけ読めます")
    if align < channels * (bits // 8):
        raise ValueError(
            f"{path.name} の block align ({align}) が, チャンネル × バイト数 "
            f"({channels * (bits // 8)}) より小さいです"
        )

    # 書かれた大きさがファイルより長い場合 (書きかけ) は, ある分だけ
    offset, size = data
    size = min(size, max(file_size - offset, 0))
    chunks = {
        "format": kinds[tag], "sample_rate": int(rate),
        "channels": int(channels), "bits": int(bits),
        "frames": int(size // align), "offset": int(offset),
        "size": int(size // align * align), "align": int(align),
    }

    return chunks


def soundfile_module() -> Any:
    """
    soundfile を読み込む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: Any
        soundfile のモジュール

    Raises
    ----------
    ImportError
        soundfile が無い場合
    """
    # 任意の依存 (WAVE の他の形式だけに要る)
    try:
        import soundfile as module
    except ImportError as error:
        raise ImportError(
            "WAVE の他の音声 (FLAC, MP3, Ogg, AIFF) を読むには soundfile が"
            "要ります (pip install soundfile か ml-eda[audio])"
        ) from error

    return module


def library_chunks(path: Path) -> dict:
    """
    soundfile で読む音声のヘッダー (形式, 周波数, チャンネル, 長さ) を
    読む関数

    Parameters
    ----------
    path: Path
        音声のファイル (FLAC, MP3, Ogg, AIFF)

    Returns
    ----------
    chunks: dict
        format (pcm, float, compressed), sample_rate, channels, bits
        (compressed は None), frames

    Raises
    ----------
    ImportError
        soundfile が無い場合
    ValueError
        読めないファイルの場合, MP3 を読めない古い libsndfile の場合
    """
    # libsndfile の版 (MP3 は 1.1 から)
    module = soundfile_module()
    if path.suffix.lower() == ".mp3":
        version = str(module.__libsndfile_version__)
        numbers = tuple(
            int(part) for part in version.split(".")[:2] if part.isdigit()
        )
        if numbers < MP3_LIBSNDFILE or "MP3" not in (
            module.available_formats()
        ):
            raise ValueError(
                f"{path.name} の MP3 は, この libsndfile ({version}) では"
                "読めません (1.1 以上が要ります)"
            )

    # ヘッダーと中身の形式
    try:
        info = module.info(str(path))
    except RuntimeError as error:
        raise ValueError(f"{path.name} を読めません ({error})") from error
    if info.samplerate < 1 or info.channels < 1:
        raise ValueError(f"{path.name} の周波数かチャンネルが正しくありません")
    form, bits = SUBTYPES.get(str(info.subtype), ("compressed", None))
    chunks = {
        "format": form, "sample_rate": int(info.samplerate),
        "channels": int(info.channels), "bits": bits,
        "frames": max(int(info.frames), 0),
    }

    return chunks


def audio_chunks(path: Path) -> dict:
    """
    音声のファイルのヘッダーを, 拡張子に合う読み方で読む関数

    Parameters
    ----------
    path: Path
        音声のファイル

    Returns
    ----------
    chunks: dict
        format, sample_rate, channels, bits, frames (と WAVE の欄の位置)

    Raises
    ----------
    ImportError
        soundfile で読む形式で, soundfile が無い場合
    ValueError
        読めないファイルの場合
    """
    # WAVE は標準ライブラリと numpy, 他は soundfile
    if path.suffix.lower() in LIBRARY_EXTENSIONS:
        chunks = library_chunks(path)
        return chunks
    chunks = wave_chunks(path)

    return chunks


def read_library(path: Path, seconds: Optional[float] = None) -> dict:
    """
    soundfile で読む音声 (FLAC, MP3, Ogg, AIFF) の音を読む関数

    Parameters
    ----------
    path: Path
        音声のファイル
    seconds: Optional[float] = None
        先頭から読む長さ (秒. None は全て)

    Returns
    ----------
    audio: dict
        samples ((標本の数, チャンネル) の float32) と sample_rate

    Raises
    ----------
    ImportError
        soundfile が無い場合
    ValueError
        読めないファイルの場合
    """
    # 形式の確かめ (MP3 の版も) と, 先頭からの標本の組
    chunks = library_chunks(path)
    module = soundfile_module()
    try:
        with module.SoundFile(str(path)) as stream:
            count = -1
            if seconds is not None:
                count = max(int(seconds * chunks["sample_rate"]), 0)
            samples = stream.read(count, dtype="float32", always_2d=True)
    except RuntimeError as error:
        raise ValueError(f"{path.name} を読めません ({error})") from error
    audio = {
        "samples": np.ascontiguousarray(samples, dtype=np.float32),
        "sample_rate": chunks["sample_rate"],
    }

    return audio


def read_audio(
    path: Union[str, Path], seconds: Optional[float] = None
) -> dict:
    """
    音声のファイルの音を読む関数

    Parameters
    ----------
    path: Union[str, Path]
        音声のファイル (WAVE の PCM の 8, 16, 24, 32 bit と, 32 bit の
        浮動小数. FLAC, MP3, Ogg, AIFF は soundfile で)
    seconds: Optional[float] = None
        先頭から読む長さ (秒. None は全て)

    Returns
    ----------
    audio: dict
        samples ((標本の数, チャンネル) の float32. 整数の形式は満点を 1
        にそろえた値, 浮動小数はそのままの値) と sample_rate (1 秒の標本の
        数)

    Raises
    ----------
    ImportError
        soundfile で読む形式で, soundfile が無い場合
    ValueError
        読めない形式の場合
    """
    # WAVE の他の形式は soundfile
    place = Path(path)
    if place.suffix.lower() in LIBRARY_EXTENSIONS:
        audio = read_library(place, seconds)
        return audio

    # 読む標本の組の数だけのバイト
    chunks = wave_chunks(place)
    count = chunks["frames"]
    if seconds is not None:
        count = min(count, max(int(seconds * chunks["sample_rate"]), 0))
    align = chunks["align"]
    with open(place, "rb") as stream:
        stream.seek(chunks["offset"])
        raw = stream.read(count * align)
    count = len(raw) // align
    width = chunks["bits"] // 8
    channels = chunks["channels"]

    # 組の後ろの埋め草を除く (多くのファイルは埋め草が無いので写さない)
    packed = width * channels
    if align == packed:
        data = memoryview(raw)[:count * align]
    else:
        data = np.frombuffer(raw, dtype=np.uint8)[:count * align].reshape(
            count, align
        )[:, :packed].tobytes()

    # 型ごとに float32 の数にする (float64 を挟まない)
    if chunks["format"] == "float":
        values = np.frombuffer(data, dtype="<f4").astype(np.float32)
    elif width == 1:
        # 8 bit は符号なし (128 が 0)
        values = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        values -= np.float32(128.0)
        values /= np.float32(128.0)
    elif width == 3:
        # 24 bit は 3 バイトの符号付きを 4 バイトに広げる
        triple = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        number = triple[:, 0].astype(np.int32)
        number |= triple[:, 1].astype(np.int32) << 8
        number |= triple[:, 2].astype(np.int32) << 16
        number[number >= 1 << 23] -= 1 << 24
        values = number.astype(np.float32)
        values /= np.float32(1 << 23)
    else:
        kind = {2: "<i2", 4: "<i4"}[width]
        values = np.frombuffer(data, dtype=kind).astype(np.float32)
        values /= np.float32(1 << (8 * width - 1))
    audio = {
        "samples": values.reshape(count, channels),
        "sample_rate": chunks["sample_rate"],
    }

    return audio


def sniff_audio(path: Path) -> float:
    """
    音声 (1 つのファイルか, 音声を集めたディレクトリ) として読めるかの
    確からしさを返す関数

    Parameters
    ----------
    path: Path
        場所

    Returns
    ----------
    score: float
        確からしさ
    """
    # 1 つの音声か, 音声が多いディレクトリ (soundfile が無くても, 拡張子
    # が合えば読み込みで理由を返す)
    if path.is_file():
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return 0.0
        try:
            audio_chunks(path)
        except ImportError:
            return 0.6
        except (ValueError, OSError, struct.error):
            return 0.2
        return 0.8
    files = list_files(path, AUDIO_EXTENSIONS)
    every = [p for _, p in zip(range(2000), path.rglob("*")) if p.is_file()]
    if not files:
        return 0.0
    score = 0.9 if len(files) >= 0.5 * max(len(every), 1) else 0.4

    return score


def load_audio(path: Path, **options: Any) -> Data:
    """
    音声 (1 つのファイルか, 音声のファイルを集めたディレクトリ) を読む
    関数

    Parameters
    ----------
    path: Path
        ファイルかディレクトリ
    **options: Any
        使わない

    Returns
    ----------
    data: Data
        1 行が 1 つの音声のデータ (path, label, 標本化の周波数, チャンネル,
        長さなど. ヘッダーだけを読む). 音は load_item で読む

    Raises
    ----------
    ValueError
        音声のファイルが無い場合
    ImportError
        全てのファイルが soundfile で読む形式で, soundfile が無い場合
    """
    # ファイルの一覧とラベル
    root = path if path.is_dir() else path.parent
    files = list_files(path, AUDIO_EXTENSIONS) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"{path.name} に音声のファイルがありません")
    labels = labels_of(root, files) if path.is_dir() else None

    # ファイルごとのヘッダー (soundfile が無くて読めない件は数える)
    rows = []
    missing = 0
    for index, file in enumerate(files):
        row: dict = {"path": file.relative_to(root).as_posix()}
        if labels:
            row["label"] = labels[index]
        try:
            chunks = audio_chunks(file)
            row.update({
                "sample_rate": chunks["sample_rate"],
                "channels": chunks["channels"], "bits": chunks["bits"],
                "format": chunks["format"], "frames": chunks["frames"],
                "duration": chunks["frames"] / chunks["sample_rate"],
                "bytes": file.stat().st_size,
            })
        except ImportError as error:
            row["error"] = str(error)
            missing += 1
        except (ValueError, OSError, struct.error) as error:
            row["error"] = str(error) or type(error).__name__
        rows.append(row)
    if missing == len(files):
        soundfile_module()
    frame = pd.DataFrame(rows)
    for name in ("sample_rate", "duration", "error"):
        if name not in frame.columns:
            frame[name] = None
    roles = {"path": "path"}
    if labels:
        roles["label"] = "label"
    data = Data(
        kind="audio", frame=frame, roles=roles,
        item_loader=lambda i: read_audio(root / frame["path"].iloc[i]),
        head_loader=lambda i, seconds: read_audio(
            root / frame["path"].iloc[i], seconds
        )
    )
    data.source = {
        "root": str(root), "capped": len(files) >= MAX_FILES,
        "missing_reader": missing,
    }

    return data


def clip_level(bits: Optional[int], form: str) -> np.float32:
    """
    音の割れとする振幅 (満点を 1 とした端の値) を, 形式と bit 数から
    求める関数

    Parameters
    ----------
    bits: Optional[int]
        量子化の bit 数 (compressed は None)
    form: str
        pcm, float, compressed

    Returns
    ----------
    level: np.float32
        整数の形式は正の最大の値 ((2^(bits-1) - 1) / 2^(bits-1). 8 bit は
        127 / 128), 浮動小数と compressed は 1 (1 を超える値も割れと
        数える)
    """
    # 読んだ振幅と同じ float32 で比べる (32 bit の PCM の端は 1 に丸まる).
    # 圧縮した形式は, 割れた所の復号した波が 1 を超える
    if form != "pcm" or bits is None or pd.isna(bits):
        level = np.float32(1.0)
        return level
    full = float(1 << (int(bits) - 1))
    level = np.float32((full - 1.0) / full)

    return level


def loudness(samples: np.ndarray, limit: np.float32) -> dict:
    """
    1 つの音の大きさ (RMS, 最大の振幅, 割れた標本の割合) を求める関数

    Parameters
    ----------
    samples: np.ndarray
        (標本の数, チャンネル) の振幅
    limit: np.float32
        音の割れとする振幅 (clip_level)

    Returns
    ----------
    level: dict
        rms, peak, clipped (振幅が limit 以上の標本の割合)
    """
    # 空の音は全て 0. float32 のまま求め, 和だけを float64 で足す
    if samples.size == 0:
        level = {"rms": 0.0, "peak": 0.0, "clipped": 0.0}
        return level
    values = np.abs(samples.astype(np.float32, copy=False))
    level = {
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "peak": float(values.max()),
        "clipped": float(np.mean(values >= limit)),
    }

    return level


def describe_audio(data: Data, seed: int = 0, sample: int = 200) -> dict:
    """
    音声の種類ごとの分析と気付きを求める関数

    Parameters
    ----------
    data: Data
        音声のデータ
    seed: int = 0
        見本を選ぶ乱数のシード
    sample: int = 200
        中身を読む見本の件数 (音の大きさ. 1 件は先頭の ANALYSIS_SECONDS 秒)

    Returns
    ----------
    result: dict
        count, unreadable, sample_rates, channels, duration (秒の最小,
        中央値, 最大, 合計), level (見本の RMS と最大の振幅の中央値),
        silent, clipped (見本の数), missing_reader (soundfile が無くて
        読めない数) と気付き (findings)
    """
    # 読めた音声の長さ, 標本化の周波数, チャンネル
    frame = data.frame
    valid = frame[frame["error"].isna()].dropna(subset=["sample_rate"])
    unreadable = int(len(frame) - len(valid))
    durations = valid["duration"].astype(float)
    result: dict = {
        "kind": "audio", "count": int(len(frame)), "unreadable": unreadable,
        "sample_rates": {
            str(int(k)): int(v)
            for k, v in valid["sample_rate"].value_counts().items()
        },
        "channels": {
            str(int(k)): int(v)
            for k, v in valid["channels"].value_counts().items()
        } if "channels" in valid.columns else {},
        "duration": {
            "min": float(durations.min()), "median": float(durations.median()),
            "max": float(durations.max()), "total": float(durations.sum()),
        } if len(durations) else None,
    }

    # 見本の音の大きさ (無音と割れ. 1 つの音声の先頭の決まった長さだけ)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        valid.index.to_numpy(), min(sample, len(valid)), replace=False
    ) if len(valid) else []
    levels = []
    for index in sorted(int(k) for k in chosen):
        row = frame.iloc[int(frame.index.get_loc(index))]
        place = row["path"]
        try:
            audio = read_audio(
                Path(str(data.source.get("root", "."))) / str(place),
                ANALYSIS_SECONDS
            )
        except (ValueError, ImportError, OSError, struct.error):
            continue
        bits = None if pd.isna(row["bits"]) else int(row["bits"])
        limit = clip_level(bits, str(row["format"]))
        levels.append(loudness(audio["samples"], limit))
    result["sampled"] = len(levels)
    result["level"] = {
        "rms": float(np.median([item["rms"] for item in levels])),
        "peak": float(np.median([item["peak"] for item in levels])),
    } if levels else None
    result["silent"] = sum(item["rms"] < SILENT_RMS for item in levels)
    result["clipped"] = sum(
        item["clipped"] >= CLIPPED_SHARE for item in levels
    )
    result["missing_reader"] = int(data.source.get("missing_reader", 0))
    result["findings"] = audio_findings(data, result)

    return result


def audio_findings(data: Data, result: dict) -> list[dict]:
    """
    音声の集計から気付きを作る関数

    Parameters
    ----------
    data: Data
        音声のデータ
    result: dict
        describe_audio の集計

    Returns
    ----------
    findings: list[dict]
        too_many_files, unreadable_items, missing_reader, mixed_sizes,
        mixed_lengths, silent_items, clipped_items の気付き
    """
    # ファイルの上限と読めないファイル (soundfile が無くて読めない件は別)
    findings = []
    missing = result.get("missing_reader", 0)
    unreadable = result["unreadable"] - missing
    if data.source.get("capped"):
        findings.append({
            "severity": "warning", "code": "too_many_files",
            "message": (
                f"ファイルが多いので, 名前の順で先頭の {MAX_FILES} 件だけを"
                "読みました (後ろのクラスが欠けている場合があります)"
            ),
            "columns": [],
        })
    if unreadable:
        findings.append({
            "severity": "warning", "code": "unreadable_items",
            "message": (
                f"読めないファイルが {unreadable} 件あります (理由は error "
                "の列. WAVE は PCM と 32 bit の浮動小数だけを読みます)"
            ),
            "columns": ["error"],
        })
    if missing:
        findings.append({
            "severity": "info", "code": "missing_reader",
            "message": (
                f"soundfile が無いので, {missing} 件の音声 (FLAC, MP3, Ogg, "
                "AIFF) を読んでいません (pip install soundfile か "
                "ml-eda[audio])"
            ),
            "columns": ["error"],
        })

    # 標本化の周波数とチャンネルのばらつき
    if len(result["sample_rates"]) > 1 or len(result["channels"]) > 1:
        findings.append({
            "severity": "warning", "code": "mixed_sizes",
            "message": (
                "標本化の周波数かチャンネルの数がそろいません (周波数 "
                f"{', '.join(result['sample_rates'])} Hz, チャンネル "
                f"{', '.join(result['channels'])}). 同じ長さの音でも配列の"
                "長さが変わるので, 学習の前にそろえてください"
            ),
            "columns": ["sample_rate", "channels"],
        })

    # 長さのばらつき, 無音と割れ (見本で)
    duration = result["duration"]
    if duration and duration["max"] > 0 and (
        duration["min"] == 0
        or duration["max"] / duration["min"] >= LENGTH_RATIO
    ):
        findings.append({
            "severity": "info", "code": "mixed_lengths",
            "message": (
                f"音の長さが {duration['min']:.2f} 秒から "
                f"{duration['max']:.2f} 秒までばらつきます. 学習では決まった"
                "長さに切るか埋めることになります"
            ),
            "columns": ["duration"],
        })
    if result["silent"]:
        findings.append({
            "severity": "warning", "code": "silent_items",
            "message": (
                f"見本の {result['sampled']} 件のうち {result['silent']} 件が"
                "ほぼ無音です (RMS が約 -60 dBFS 未満). 録音の失敗でないかを"
                "確かめてください"
            ),
            "columns": [],
        })
    if result["clipped"]:
        findings.append({
            "severity": "info", "code": "clipped_items",
            "message": (
                f"見本の {result['sampled']} 件のうち {result['clipped']} 件で"
                f", 標本の {CLIPPED_SHARE:.0%} 以上が音の大きさの上限 (形式の満点) に"
                "張り付いています (音が割れている). 録音の大きさを確かめて"
                "ください"
            ),
            "columns": [],
        })

    return findings


def register_audio() -> None:
    """
    音声の読み込みを登録する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # WAVE は標準ライブラリと numpy, 他の形式は soundfile で読む
    register_loader(Loader(
        "audio_folder", "audio", AUDIO_EXTENSIONS, sniff_audio, load_audio,
        {},
        "音声 (.wav の PCM と 32 bit の浮動小数. .flac, .mp3, .ogg, .opus, "
        ".aif, .aiff は soundfile が要る. 1 つか, <クラス>/<音声> の"
        "ディレクトリ)"
    ))


register_audio()
