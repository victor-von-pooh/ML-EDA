import math
from typing import Any

import numpy as np

# 多相のフィルタの片側の零点の数 (下げる場合は比で広げる)
SINC_ZEROS = 16

# Kaiser の窓の beta
KAISER_BETA = 8.6

# 周波数のそろえ方
RESAMPLE_METHODS = ("sinc", "linear")

# 長さをそろえるときに残す位置
ALIGNS = ("start", "center")


def audio_parts(audio: Any, op: str) -> tuple[np.ndarray, int]:
    """
    音声の中身から, 標本と標本化の周波数を取り出す関数

    Parameters
    ----------
    audio: Any
        samples ((標本の数, チャンネル)) と sample_rate の辞書
    op: str
        手順の名前 (誤りの文に使う)

    Returns
    ----------
    parts: tuple[np.ndarray, int]
        (標本の数, チャンネル) の float32 と標本化の周波数

    Raises
    ----------
    ValueError
        辞書でない, 標本が 2 次元でない, 周波数が 1 以上でない場合
    """
    # read_audio の返す形だけを受け取る
    if not isinstance(audio, dict) or "samples" not in audio:
        raise ValueError(
            f"{op} には read_audio の形 (samples と sample_rate の辞書) を"
            "渡してください"
        )
    samples = np.asarray(audio["samples"], dtype=np.float32)
    if samples.ndim != 2:
        raise ValueError(
            f"{op} の samples は (標本の数, チャンネル) の 2 次元にして"
            f"ください (今は {samples.ndim} 次元)"
        )
    rate = audio.get("sample_rate")
    if isinstance(rate, bool) or not isinstance(
        rate, (int, np.integer)
    ) or rate < 1:
        raise ValueError(
            f"{op} の sample_rate は 1 以上の整数にしてください (今は {rate})"
        )
    parts = (samples, int(rate))

    return parts


def whole_number(op: str, name: str, value: Any) -> int:
    """
    手順の引数を 1 以上の整数として確かめる関数

    Parameters
    ----------
    op: str
        手順の名前
    name: str
        引数の名前
    value: Any
        引数の値

    Returns
    ----------
    number: int
        値

    Raises
    ----------
    ValueError
        1 以上の整数でない場合
    """
    # 真偽値と小数は受け取らない
    if isinstance(value, bool) or not isinstance(
        value, (int, np.integer)
    ) or value < 1:
        raise ValueError(
            f"{op} の {name} は 1 以上の整数にしてください (今は {value})"
        )
    number = int(value)

    return number


def sinc_bank(up: int, down: int) -> tuple[np.ndarray, int]:
    """
    多相のフィルタの位相ごとの重みを作る関数

    Parameters
    ----------
    up: int
        周波数の比の分子 (先の周波数 / 最大公約数)
    down: int
        周波数の比の分母 (元の周波数 / 最大公約数)

    Returns
    ----------
    bank: tuple[np.ndarray, int]
        (位相, 重み) の配列と, 片側の重みの数
    """
    # 下げる場合は先の周波数の半分で切る (零点の間を比で広げる)
    cutoff = min(1.0, up / down)
    half = int(math.ceil(SINC_ZEROS / cutoff))
    offsets = np.arange(-half + 1, half + 1)

    # 位相 p の出力の位置 (元の標本の単位) から見た, 各元の標本の隔たり
    distance = np.arange(up)[:, None] / up - offsets[None, :]
    window = np.i0(
        KAISER_BETA * np.sqrt(np.clip(1.0 - (distance / half) ** 2, 0, 1))
    ) / np.i0(KAISER_BETA)
    weights = cutoff * np.sinc(cutoff * distance) * window

    # 位相ごとに重みの和を 1 (直流をそのまま通す)
    weights /= weights.sum(axis=1, keepdims=True)
    bank = (weights, half)

    return bank


def resample_sinc(samples: np.ndarray, up: int, down: int) -> np.ndarray:
    """
    多相のフィルタで標本化の周波数を変える関数

    Parameters
    ----------
    samples: np.ndarray
        (標本の数, チャンネル) の float32
    up: int
        周波数の比の分子
    down: int
        周波数の比の分母

    Returns
    ----------
    result: np.ndarray
        (ceil(標本の数 × up / down), チャンネル) の float32
    """
    # 出力の各標本の, 元の標本の位置の整数部と位相
    count = len(samples)
    total = -(-count * up // down)
    weights, half = sinc_bank(up, down)
    numerator = np.arange(total, dtype=np.int64) * down
    base = numerator // up
    phase = numerator % up

    # 両側を 0 で埋め, 重みの列ごとに足し込む (記憶は出力の大きさだけ)
    padded = np.zeros((count + 2 * half, samples.shape[1]), dtype=np.float64)
    padded[half:half + count] = samples
    accumulated = np.zeros((total, samples.shape[1]), dtype=np.float64)
    for column, offset in enumerate(range(-half + 1, half + 1)):
        accumulated += (
            padded[base + offset + half] * weights[phase, column][:, None]
        )
    result = accumulated.astype(np.float32)

    return result


def resample_linear(
    samples: np.ndarray, up: int, down: int
) -> np.ndarray:
    """
    線形補間で標本化の周波数を変える関数

    Parameters
    ----------
    samples: np.ndarray
        (標本の数, チャンネル) の float32
    up: int
        周波数の比の分子
    down: int
        周波数の比の分母

    Returns
    ----------
    result: np.ndarray
        (ceil(標本の数 × up / down), チャンネル) の float32
    """
    # 出力の各標本の元の位置 (最後の標本より後は最後の値)
    count = len(samples)
    total = -(-count * up // down)
    places = np.arange(total, dtype=np.float64) * down / up
    known = np.arange(count, dtype=np.float64)
    result = np.empty((total, samples.shape[1]), dtype=np.float32)
    for channel in range(samples.shape[1]):
        result[:, channel] = np.interp(places, known, samples[:, channel])

    return result


def resample(audio: Any, params: dict, state: dict) -> dict:
    """
    音声の標本化の周波数をそろえる関数

    Parameters
    ----------
    audio: Any
        samples と sample_rate の辞書
    params: dict
        rate (そろえる周波数), method (sinc か linear. 既定 sinc)
    state: dict
        使わない

    Returns
    ----------
    resampled: dict
        rate の周波数にした samples と sample_rate

    Raises
    ----------
    ValueError
        rate が 1 以上の整数でない場合と, 知らない method の場合
    """
    # 引数の確認
    samples, source = audio_parts(audio, "resample")
    rate = whole_number("resample", "rate", params.get("rate"))
    method = params.get("method", "sinc")
    if method not in RESAMPLE_METHODS:
        raise ValueError(
            f"resample の method は {', '.join(RESAMPLE_METHODS)} のどれかに"
            f"してください: {method}"
        )
    if rate == source or len(samples) == 0:
        resampled = {**audio, "samples": samples, "sample_rate": rate}
        return resampled

    # 既約の比で変える
    divisor = math.gcd(rate, source)
    up, down = rate // divisor, source // divisor
    change = resample_sinc if method == "sinc" else resample_linear
    resampled = {
        **audio, "samples": change(samples, up, down), "sample_rate": rate
    }

    return resampled


def mix_channels(audio: Any, params: dict, state: dict) -> dict:
    """
    音声のチャンネルの数をそろえる関数

    Parameters
    ----------
    audio: Any
        samples と sample_rate の辞書
    params: dict
        channels (そろえるチャンネルの数)
    state: dict
        使わない

    Returns
    ----------
    mixed: dict
        channels のチャンネルにした samples と sample_rate

    Raises
    ----------
    ValueError
        channels が 1 以上の整数でない場合と, 混ぜ方の決まらない組の場合
    """
    # 1 は平均, 1 チャンネルからは同じ音を並べる
    samples, _ = audio_parts(audio, "mix_channels")
    channels = whole_number("mix_channels", "channels", params.get("channels"))
    have = samples.shape[1]
    if have == channels:
        mixed = {**audio, "samples": samples}
        return mixed
    if channels == 1:
        values = samples.mean(axis=1, keepdims=True, dtype=np.float32)
    elif have == 1:
        values = np.repeat(samples, channels, axis=1)
    else:
        raise ValueError(
            f"{have} チャンネルの音声を {channels} チャンネルにする混ぜ方は"
            "決まっていません (1 チャンネルにするか, 同じチャンネルの数の"
            "音声だけにしてください)"
        )
    mixed = {**audio, "samples": values}

    return mixed


def fix_length(audio: Any, params: dict, state: dict) -> dict:
    """
    音声を決まった長さに切るか, 0 で埋める関数

    Parameters
    ----------
    audio: Any
        samples と sample_rate の辞書
    params: dict
        seconds (そろえる長さの秒), align (start か center. 既定 start)
    state: dict
        使わない

    Returns
    ----------
    fixed: dict
        seconds × sample_rate (四捨五入) の標本の数にした samples と
        sample_rate

    Raises
    ----------
    ValueError
        seconds が 0 より大きい数でない場合と, 知らない align の場合
    """
    # 引数の確認と, そろえる標本の数
    samples, rate = audio_parts(audio, "fix_length")
    seconds = params.get("seconds")
    if isinstance(seconds, bool) or not isinstance(
        seconds, (int, float, np.integer, np.floating)
    ) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            f"fix_length の seconds は 0 より大きい数にしてください (今は "
            f"{seconds})"
        )
    align = params.get("align", "start")
    if align not in ALIGNS:
        raise ValueError(
            f"fix_length の align は {', '.join(ALIGNS)} のどれかにして"
            f"ください: {align}"
        )
    length = max(int(round(seconds * rate)), 1)

    # 長ければ切り, 短ければ 0 で埋める (center は両側, 余りは後ろ)
    count = len(samples)
    gap = abs(count - length)
    before = gap // 2 if align == "center" else 0
    if count >= length:
        values = samples[before:before + length]
    else:
        values = np.zeros((length, samples.shape[1]), dtype=np.float32)
        values[before:before + count] = samples
    fixed = {**audio, "samples": values}

    return fixed


def audio_array(audio: Any) -> np.ndarray:
    """
    音声の中身を, 学習に渡す (チャンネル, 標本の数) の配列にする関数

    Parameters
    ----------
    audio: Any
        samples と sample_rate の辞書

    Returns
    ----------
    array: np.ndarray
        (チャンネル, 標本の数) の float32
    """
    # PyTorch の音声の慣例の並び
    samples, _ = audio_parts(audio, "音声の配列")
    array = np.ascontiguousarray(samples.T)

    return array
