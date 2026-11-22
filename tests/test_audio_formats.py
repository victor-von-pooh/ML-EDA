import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from ml_eda import (
    detect, export_dataset, load_data, load_dataset, profile, read_audio
)
from ml_eda.audio import clip_level


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
    times = np.arange(int(round(seconds * rate))) / rate
    wave_ = 0.5 * np.sin(2 * np.pi * 440 * times)
    samples = np.repeat(wave_[:, None], channels, axis=1)

    return samples


def write_wave(path: Path, samples: np.ndarray, rate: int) -> None:
    """
    (標本の数, チャンネル) の振幅を 16 bit の PCM の WAVE に書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    samples: np.ndarray
        振幅 (-1 から 1)
    rate: int
        標本化の周波数

    Returns
    ----------
    None
    """
    # 標準ライブラリの wave で書く (soundfile が無くても)
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers = np.clip(np.round(samples * 32768), -32768, 32767)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(samples.shape[1])
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(numbers.astype("<i2").tobytes())


def write_sound(
    path: Path, samples: np.ndarray, rate: int, subtype: str
) -> None:
    """
    soundfile で音声のファイル (拡張子の形式) を書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    samples: np.ndarray
        (標本の数, チャンネル) の振幅
    rate: int
        標本化の周波数
    subtype: str
        libsndfile の中身の形式 (PCM_16, FLOAT, MPEG_LAYER_III など)

    Returns
    ----------
    None
    """
    # 拡張子から形式を決めて書く
    soundfile = pytest.importorskip("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    form = {
        ".flac": "FLAC", ".mp3": "MP3", ".ogg": "OGG", ".opus": "OGG",
        ".aiff": "AIFF", ".aif": "AIFF",
    }[path.suffix]
    soundfile.write(str(path), samples, rate, format=form, subtype=subtype)


def test_library_formats(tmp_path: Path) -> None:
    """
    FLAC, MP3, Ogg, AIFF を load_data で読み, bit 数と形式, 中身, 分析の
    結果と割れの判定を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 形式ごとに 1 つ (2 クラス) と, 読めないファイル
    pytest.importorskip("soundfile")
    root = tmp_path / "sounds"
    samples = tone(0.5, 16000)
    files = {
        "dog/a16.flac": "PCM_16", "dog/a24.flac": "PCM_24",
        "dog/a8.flac": "PCM_S8", "cat/b.aiff": "FLOAT",
        "cat/c.mp3": "MPEG_LAYER_III", "cat/d.ogg": "VORBIS",
    }
    for name, subtype in files.items():
        write_sound(root / name, samples, 16000, subtype)
    (root / "dog" / "broken.flac").write_bytes(b"fLaCjunk")

    # 読み込み: 選ばれる読み込み, bit 数と形式 (圧縮は bit 数が欠損)
    assert detect(root)[0]["loader"] == "audio_folder"
    data = load_data(root)
    assert data.kind == "audio" and len(data.frame) == 7
    table = data.frame.set_index("path")
    expected = {
        "dog/a16.flac": ("pcm", 16), "dog/a24.flac": ("pcm", 24),
        "dog/a8.flac": ("pcm", 8), "cat/b.aiff": ("float", 32),
    }
    for name, (form, bits) in expected.items():
        assert table.loc[name, "format"] == form
        assert table.loc[name, "bits"] == bits
        assert table.loc[name, "sample_rate"] == 16000
    for name in ("cat/c.mp3", "cat/d.ogg"):
        assert table.loc[name, "format"] == "compressed"
        assert np.isnan(table.loc[name, "bits"])
        assert table.loc[name, "duration"] == pytest.approx(0.5, abs=0.05)
    assert "読めません" in table.loc["dog/broken.flac", "error"]

    # 中身: read_audio と同じ形 ((標本の数, チャンネル) の float32)
    for name in files:
        audio = read_audio(root / name)
        assert audio["samples"].dtype == np.float32
        assert audio["samples"].ndim == 2 and audio["samples"].shape[1] == 1
        assert audio["sample_rate"] == 16000
    np.testing.assert_allclose(
        read_audio(root / "dog/a24.flac")["samples"], samples, atol=1e-6
    )
    head = read_audio(root / "dog/a16.flac", seconds=0.1)
    assert head["samples"].shape == (1600, 1)
    with pytest.raises(ValueError, match="読めません"):
        read_audio(root / "dog" / "broken.flac")

    # 分析の結果と気付き (soundfile があるので missing_reader は無い)
    result = profile(data)
    json.dumps(result)
    items = result["items"]
    assert items["count"] == 7 and items["unreadable"] == 1
    assert items["missing_reader"] == 0 and items["sampled"] == 6
    assert items["clipped"] == 0 and items["silent"] == 0
    codes = {item["code"] for item in result["findings"]}
    assert "unreadable_items" in codes and "missing_reader" not in codes

    # 割れ: FLAC は bit 数の端, 圧縮した形式は 1 以上で数える
    harsh = tmp_path / "harsh"
    clipped = np.clip(tone(0.5, 16000) * 4, -1, 1)
    write_sound(harsh / "clip.flac", clipped, 16000, "PCM_16")
    write_sound(harsh / "clip.mp3", clipped, 16000, "MPEG_LAYER_III")
    write_sound(harsh / "clip.ogg", clipped, 16000, "VORBIS")
    write_sound(harsh / "loud.flac", tone(0.5, 16000) * 1.99, 16000,
                "PCM_24")
    result = profile(load_data(harsh))
    assert result["items"]["clipped"] == 3
    assert clip_level(None, "compressed") == np.float32(1.0)
    assert clip_level(float("nan"), "compressed") == np.float32(1.0)


def test_library_export(tmp_path: Path) -> None:
    """
    FLAC, MP3, Ogg の混ざった音声を中身の手順 (resample, mix_channels,
    fix_length) で書き出し, 読み直しと推論の 1 件が合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 クラス. 形式, 周波数, チャンネル, 長さが混ざる
    pytest.importorskip("soundfile")
    root = tmp_path / "sounds"
    for index in range(4):
        write_sound(root / "dog" / f"d{index}.flac", tone(0.5, 22050),
                    22050, "PCM_24")
        write_sound(root / "cat" / f"c{index}.ogg", tone(0.8, 44100, 2),
                    44100, "VORBIS")
    write_sound(root / "cat" / "m.mp3", tone(1.5, 16000, 2), 16000,
                "MPEG_LAYER_III")
    data = load_data(root)
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [
            {"op": "resample", "rate": 16000},
            {"op": "mix_channels", "channels": 1},
            {"op": "fix_length", "seconds": 1.0},
        ],
        "split": {"method": "stratified", "valid": 0.3, "seed": 0},
    }

    # 書き出しと読み直し
    info = export_dataset(data, recipe, tmp_path / "ds")
    assert info["kind"] == "audio" and info["classes"] == ["cat", "dog"]
    assert info["splits"]["train"]["shape"][1:] == [1, 16000]
    total = sum(split["rows"] for split in info["splits"].values())
    assert total == 9
    dataset = load_dataset(tmp_path / "ds")
    x, y = dataset.arrays("train")
    assert x.shape[1:] == (1, 16000) and x.dtype == np.float32
    assert len(x) == len(y)

    # 推論: ファイルの場所でも read_audio の辞書でも, 学習と同じ配列
    first = root / dataset.frames["train"]["path"].iloc[0]
    assert np.array_equal(dataset.transform_item(first), x[0])
    assert np.array_equal(
        dataset.transform_item(read_audio(first)), x[0]
    )


def test_old_libsndfile_mp3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    MP3 を読めない古い libsndfile では, MP3 の行に理由が入ることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        libsndfile の版を差し替える

    Returns
    ----------
    None
    """
    # 書いてから版を古くする
    soundfile = pytest.importorskip("soundfile")
    write_sound(tmp_path / "a.mp3", tone(0.5, 16000), 16000,
                "MPEG_LAYER_III")
    write_sound(tmp_path / "b.flac", tone(0.5, 16000), 16000, "PCM_16")
    monkeypatch.setattr(soundfile, "__libsndfile_version__", "1.0.31")
    frame = load_data(tmp_path).frame.set_index("path")
    assert "1.0.31" in frame.loc["a.mp3", "error"]
    assert frame["error"].isna()["b.flac"]


def test_missing_soundfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    soundfile が無い場合に, missing_reader の気付きか理由付きの誤りに
    なることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        soundfile を読み込めなくする

    Returns
    ----------
    None
    """
    # soundfile を読み込めなくする (中身は読まないので名前だけのファイル)
    monkeypatch.setitem(sys.modules, "soundfile", None)
    root = tmp_path / "mixed"
    write_wave(root / "dog" / "a.wav", tone(0.5, 16000), 16000)
    write_wave(root / "cat" / "b.wav", tone(0.5, 16000), 16000)
    (root / "cat" / "c.flac").parent.mkdir(parents=True, exist_ok=True)
    (root / "cat" / "c.flac").write_bytes(b"fLaC")

    # WAVE と混ざる場合: その行の誤りと missing_reader の気付き
    data = load_data(root)
    assert "soundfile" in data.frame.set_index("path").loc[
        "cat/c.flac", "error"
    ]
    result = profile(data)
    assert result["items"]["missing_reader"] == 1
    codes = {item["code"] for item in result["findings"]}
    assert "missing_reader" in codes and "unreadable_items" not in codes

    # 全てがその形式の場合と, 1 件の読み込みは理由付きの誤り
    with pytest.raises(ImportError, match="soundfile"):
        load_data(root / "cat" / "c.flac")
    only = tmp_path / "only"
    only.mkdir()
    (only / "a.mp3").write_bytes(b"ID3")
    (only / "b.ogg").write_bytes(b"OggS")
    with pytest.raises(ImportError, match="soundfile"):
        load_data(only)
    with pytest.raises(ImportError, match="soundfile"):
        read_audio(root / "cat" / "c.flac")
