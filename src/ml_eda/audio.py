import struct
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.loaders import Loader, register_loader
from ml_eda.media import MAX_FILES, labels_of, list_files

# 音声の拡張子
AUDIO_EXTENSIONS = (".wav", ".wave")

# 形式の番号 (WAVE の fmt の欄)
PCM = 1
IEEE_FLOAT = 3
EXTENSIBLE = 0xFFFE

# 無音とする RMS (満点を 1 とした値. 約 -60 dBFS)
SILENT_RMS = 1e-3

# 音の割れとする振幅 (満点を 1 とした値. 8 bit の正の端 127 / 128 も
# 割れと数える) と, 割れた標本の割合
CLIPPED_LEVEL = 127 / 128
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


def read_audio(
    path: Union[str, Path], seconds: Optional[float] = None
) -> dict:
    """
    WAVE のファイルの音を読む関数

    Parameters
    ----------
    path: Union[str, Path]
        WAVE のファイル (PCM の 8, 16, 24, 32 bit と, 32 bit の浮動小数)
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
    ValueError
        読めない形式の場合
    """
    # 読む標本の組の数だけのバイト
    place = Path(path)
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
    # 1 つの WAVE か, WAVE が多いディレクトリ
    if path.is_file():
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return 0.0
        try:
            wave_chunks(path)
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
    """
    # ファイルの一覧とラベル
    root = path if path.is_dir() else path.parent
    files = list_files(path, AUDIO_EXTENSIONS) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"{path.name} に音声 (.wav) のファイルがありません")
    labels = labels_of(root, files) if path.is_dir() else None

    # ファイルごとのヘッダー
    rows = []
    for index, file in enumerate(files):
        row: dict = {"path": file.relative_to(root).as_posix()}
        if labels:
            row["label"] = labels[index]
        try:
            chunks = wave_chunks(file)
            row.update({
                "sample_rate": chunks["sample_rate"],
                "channels": chunks["channels"], "bits": chunks["bits"],
                "format": chunks["format"], "frames": chunks["frames"],
                "duration": chunks["frames"] / chunks["sample_rate"],
                "bytes": file.stat().st_size,
            })
        except (ValueError, OSError, struct.error) as error:
            row["error"] = str(error) or type(error).__name__
        rows.append(row)
    frame = pd.DataFrame(rows)
    for name in ("sample_rate", "duration", "error"):
        if name not in frame.columns:
            frame[name] = None
    roles = {"path": "path"}
    if labels:
        roles["label"] = "label"
    data = Data(
        kind="audio", frame=frame, roles=roles,
        item_loader=lambda i: read_audio(root / frame["path"].iloc[i])
    )
    data.source = {"root": str(root), "capped": len(files) >= MAX_FILES}

    return data


def loudness(samples: np.ndarray) -> dict:
    """
    1 つの音の大きさ (RMS, 最大の振幅, 割れた標本の割合) を求める関数

    Parameters
    ----------
    samples: np.ndarray
        (標本の数, チャンネル) の振幅

    Returns
    ----------
    level: dict
        rms, peak, clipped (振幅が CLIPPED_LEVEL 以上の標本の割合)
    """
    # 空の音は全て 0. float32 のまま求め, 和だけを float64 で足す
    if samples.size == 0:
        level = {"rms": 0.0, "peak": 0.0, "clipped": 0.0}
        return level
    values = np.abs(samples.astype(np.float32, copy=False))
    level = {
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "peak": float(values.max()),
        "clipped": float(np.mean(values >= CLIPPED_LEVEL)),
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
        silent, clipped (見本の数) と気付き (findings)
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
        place = frame["path"].iloc[int(frame.index.get_loc(index))]
        try:
            audio = read_audio(
                Path(str(data.source.get("root", "."))) / str(place),
                ANALYSIS_SECONDS
            )
        except (ValueError, OSError, struct.error):
            continue
        levels.append(loudness(audio["samples"]))
    result["sampled"] = len(levels)
    result["level"] = {
        "rms": float(np.median([item["rms"] for item in levels])),
        "peak": float(np.median([item["peak"] for item in levels])),
    } if levels else None
    result["silent"] = sum(item["rms"] < SILENT_RMS for item in levels)
    result["clipped"] = sum(
        item["clipped"] >= CLIPPED_SHARE for item in levels
    )
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
        too_many_files, unreadable_items, mixed_sizes, mixed_lengths,
        silent_items, clipped_items の気付き
    """
    # ファイルの上限と読めないファイル
    findings = []
    if data.source.get("capped"):
        findings.append({
            "severity": "warning", "code": "too_many_files",
            "message": (
                f"ファイルが多いので, 名前の順で先頭の {MAX_FILES} 件だけを"
                "読みました (後ろのクラスが欠けている場合があります)"
            ),
            "columns": [],
        })
    if result["unreadable"]:
        findings.append({
            "severity": "warning", "code": "unreadable_items",
            "message": (
                f"読めないファイルが {result['unreadable']} 件あります "
                "(PCM と 32 bit の浮動小数の WAVE だけを読みます)"
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
                f", 標本の {CLIPPED_SHARE:.0%} 以上が最大の振幅に張り付いて"
                "います (音が割れている). 録音の大きさを確かめてください"
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
    # WAVE だけ (標準ライブラリと numpy で読む)
    register_loader(Loader(
        "audio_folder", "audio", AUDIO_EXTENSIONS, sniff_audio, load_audio,
        {},
        "音声 (.wav. PCM と 32 bit の浮動小数. 1 つか, <クラス>/<音声> の"
        "ディレクトリ)"
    ))


register_audio()
