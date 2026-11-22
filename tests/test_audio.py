import json
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from ml_eda import detect, load_data, prepare, profile, read_audio


def write_wave(
    path: Path, samples: np.ndarray, rate: int = 16000, bits: int = 16
) -> None:
    """
    (標本の数, チャンネル) の振幅 (-1 から 1) を PCM の WAVE に書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    samples: np.ndarray
        振幅
    rate: int = 16000
        標本化の周波数
    bits: int = 16
        量子化の bit 数 (8, 16, 24, 32)

    Returns
    ----------
    None
    """
    # bit 数ごとの整数にして, 標準ライブラリの wave で書く
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(1 << (bits - 1))
    numbers = np.clip(np.round(samples * scale), -scale, scale - 1)
    numbers = numbers.astype(np.int64).reshape(-1)
    if bits == 8:
        raw = (numbers + 128).astype(np.uint8).tobytes()
    elif bits == 24:
        raw = b"".join(int(n).to_bytes(3, "little", signed=True)
                       for n in numbers)
    else:
        raw = numbers.astype(f"<i{bits // 8}").tobytes()
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(samples.shape[1])
        stream.setsampwidth(bits // 8)
        stream.setframerate(rate)
        stream.writeframes(raw)


def write_float_wave(path: Path, samples: np.ndarray, rate: int) -> None:
    """
    (標本の数, チャンネル) の振幅を 32 bit の浮動小数の WAVE に書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    samples: np.ndarray
        振幅
    rate: int
        標本化の周波数

    Returns
    ----------
    None
    """
    # 形式の番号 3 の fmt と data の欄
    path.parent.mkdir(parents=True, exist_ok=True)
    data = samples.astype("<f4").tobytes()
    channels = samples.shape[1]
    fmt = struct.pack(
        "<HHIIHH", 3, channels, rate, rate * 4 * channels, 4 * channels, 32
    )
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(data)) + data)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def tone(seconds: float, rate: int, channels: int = 1) -> np.ndarray:
    """
    440 Hz の正弦波を作る関数

    Parameters
    ----------
    seconds: float
        長さ (秒)
    rate: int
        標本化の周波数
    channels: int = 1
        チャンネルの数

    Returns
    ----------
    samples: np.ndarray
        (標本の数, チャンネル) の振幅 (最大 0.5)
    """
    # 全てのチャンネルに同じ波
    times = np.arange(int(seconds * rate)) / rate
    wave_ = 0.5 * np.sin(2 * np.pi * 440 * times)
    samples = np.repeat(wave_[:, None], channels, axis=1)

    return samples


def test_read_audio_formats(tmp_path: Path) -> None:
    """
    PCM の 8, 16, 24, 32 bit と浮動小数の WAVE を同じ振幅で読むことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 チャンネルの 0.1 秒
    samples = tone(0.1, 8000, channels=2)
    for bits in (8, 16, 24, 32):
        write_wave(tmp_path / f"b{bits}.wav", samples, 8000, bits)
        audio = read_audio(tmp_path / f"b{bits}.wav")
        assert audio["sample_rate"] == 8000
        assert audio["samples"].shape == (800, 2)
        assert audio["samples"].dtype == np.float32
        np.testing.assert_allclose(
            audio["samples"], samples, atol=2.0 / (1 << (bits - 1))
        )
    write_float_wave(tmp_path / "float.wav", samples, 8000)
    np.testing.assert_allclose(
        read_audio(tmp_path / "float.wav")["samples"], samples, atol=1e-6
    )

    # WAVE でないファイルと, 知らない形式は誤り
    (tmp_path / "bad.wav").write_bytes(b"not a wave")
    with pytest.raises(ValueError, match="WAVE"):
        read_audio(tmp_path / "bad.wav")
    data = (tmp_path / "float.wav").read_bytes()
    (tmp_path / "alaw.wav").write_bytes(data[:20] + b"\x06" + data[21:])
    with pytest.raises(ValueError, match="形式"):
        read_audio(tmp_path / "alaw.wav")


def test_audio_folder(tmp_path: Path) -> None:
    """
    <クラス>/<音声> のディレクトリを load_data で読み, 分析の結果と気付きが
    返ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 クラス. 周波数の違う音, 長い音, 無音, 読めないファイルを含む
    root = tmp_path / "sounds"
    for index in range(4):
        write_wave(root / "dog" / f"d{index}.wav", tone(0.5, 16000))
        write_wave(root / "cat" / f"c{index}.wav", tone(0.5, 16000))
    write_wave(root / "cat" / "long.wav", tone(2.0, 16000))
    write_wave(root / "cat" / "low.wav", tone(0.5, 8000), 8000)
    write_wave(root / "dog" / "quiet.wav", np.zeros((8000, 1)))
    (root / "dog" / "broken.wav").write_bytes(b"RIFF....WAVEjunk")

    # 読み込み: 選ばれる読み込み, 表の列, ラベル, 1 件の中身
    assert detect(root)[0]["loader"] == "audio_folder"
    data = load_data(root)
    assert data.kind == "audio" and data.roles["label"] == "label"
    assert len(data.frame) == 12
    row = data.frame.set_index("path").loc["cat/long.wav"]
    assert row["sample_rate"] == 16000 and row["duration"] == 2.0
    assert row["channels"] == 1 and row["bits"] == 16
    position = int(np.flatnonzero(data.frame["path"] == "cat/c0.wav")[0])
    assert data.load_item(position)["samples"].shape == (8000, 1)

    # 分析の結果と気付き
    result = profile(data)
    json.dumps(result)
    items = result["items"]
    assert items["kind"] == "audio" and items["count"] == 12
    assert items["unreadable"] == 1
    assert items["sample_rates"] == {"16000": 10, "8000": 1}
    assert items["duration"]["max"] == 2.0 and items["silent"] == 1
    codes = {item["code"] for item in result["findings"]}
    assert {"unreadable_items", "mixed_sizes", "mixed_lengths",
            "silent_items"} <= codes

    # 1 つのファイルも読め, 前処理と書き出しはまだ誤り
    single = load_data(root / "cat" / "c0.wav")
    assert single.kind == "audio" and len(single.frame) == 1
    assert "label" not in single.frame.columns
    with pytest.raises(ValueError, match="音声は"):
        prepare(data, {"split": {"method": "none"}})
