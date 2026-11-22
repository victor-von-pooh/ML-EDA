import json
import struct
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    Data, detect, export_dataset, load_data, load_dataset, model_inputs,
    prepare, profile, read_audio
)
from ml_eda.audio import clip_level
from ml_eda.audioprep import fix_length, mix_channels, resample


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

    # 1 つのファイルも読める
    single = load_data(root / "cat" / "c0.wav")
    assert single.kind == "audio" and len(single.frame) == 1
    assert "label" not in single.frame.columns


def test_load_item_seconds(tmp_path: Path) -> None:
    """
    load_item の seconds で音声の先頭の秒だけを読み, 音声の他の種類と 0 以下
    の秒が誤りになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 チャンネルの 0.5 秒の音
    root = tmp_path / "sounds"
    write_wave(root / "a.wav", tone(0.5, 16000, 2))
    data = load_data(root)
    whole = read_audio(root / "a.wav")

    # None は全て, 秒は先頭だけ, ファイルより長い秒は全てで, 辞書は同じ形
    assert data.load_item(0)["samples"].shape == (8000, 2)
    head = data.load_item(0, seconds=0.1)
    assert set(head) == {"samples", "sample_rate"}
    assert head["sample_rate"] == 16000
    assert head["samples"].dtype == np.float32
    np.testing.assert_array_equal(head["samples"], whole["samples"][:1600])
    for seconds in (1, 10.0):
        longer = data.load_item(0, seconds=seconds)
        np.testing.assert_array_equal(longer["samples"], whole["samples"])

    # 0, 負, 無限, 数でない秒は誤り
    for seconds in (0, -1.0, float("inf"), float("nan"), "1", True):
        with pytest.raises(ValueError, match="0 より大きい数"):
            data.load_item(0, seconds=seconds)

    # 音声の他の種類は理由付きの誤り, 表は中身を持たない誤り
    image = Data(kind="image", frame=pd.DataFrame({"path": ["a.png"]}),
                 item_loader=lambda i: np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="seconds は音声"):
        image.load_item(0, seconds=1.0)
    table = Data(kind="table", frame=pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="中身を持ちません"):
        table.load_item(0, seconds=1.0)


def test_audio_edge_cases(tmp_path: Path) -> None:
    """
    拡張の形式, 書きかけ, 壊れた欄, 足りない block align, 先頭だけの読み
    込みと, 割れと長さ 0 の気付きを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 拡張の形式 (中の形式は PCM) の 16 bit
    samples = tone(0.1, 8000)
    write_wave(tmp_path / "plain.wav", samples, 8000)
    plain = (tmp_path / "plain.wav").read_bytes()
    data = plain[plain.index(b"data"):]
    fmt = struct.pack("<HHIIHHHHI", 0xFFFE, 1, 8000, 16000, 2, 16, 22, 16,
                      4) + struct.pack("<H", 1) + bytes(14)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + data
    (tmp_path / "ext.wav").write_bytes(
        b"RIFF" + struct.pack("<I", len(body)) + body
    )
    np.testing.assert_allclose(
        read_audio(tmp_path / "ext.wav")["samples"],
        read_audio(tmp_path / "plain.wav")["samples"]
    )

    # 書きかけ (data の大きさより短い) はある分だけ, 先頭だけも読める
    (tmp_path / "cut.wav").write_bytes(plain[:-401])
    assert read_audio(tmp_path / "cut.wav")["samples"].shape == (599, 1)
    head = read_audio(tmp_path / "plain.wav", seconds=0.05)
    assert head["samples"].shape == (400, 1)

    # data の欄の無いファイルと, 足りない block align は誤り
    (tmp_path / "nodata.wav").write_bytes(plain[:plain.index(b"data")])
    with pytest.raises(ValueError, match="data の欄"):
        read_audio(tmp_path / "nodata.wav")
    start = plain.index(b"fmt ") + 8
    short = plain[:start + 12] + struct.pack("<H", 1) + plain[start + 14:]
    (tmp_path / "align.wav").write_bytes(short)
    with pytest.raises(ValueError, match="block align"):
        read_audio(tmp_path / "align.wav")

    # 割れた音 (8 bit の正の端も) と長さ 0 の音の気付き
    root = tmp_path / "set"
    loud = np.clip(tone(0.5, 8000) * 4, -1, 1)
    write_wave(root / "loud8.wav", loud, 8000, bits=8)
    write_wave(root / "loud16.wav", loud, 8000)
    write_wave(root / "empty.wav", np.zeros((0, 1)), 8000)
    result = profile(load_data(root))
    assert result["items"]["clipped"] == 2
    codes = {item["code"] for item in result["findings"]}
    assert {"clipped_items", "mixed_lengths"} <= codes


def test_audio_clipping(tmp_path: Path) -> None:
    """
    音の割れの判定が, 形式と bit 数ごとの満点の端で数えることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest の一時ディレクトリ

    Returns
    ----------
    None
    """
    # 16 bit 以上と浮動小数の割れていない大きな音 (最大 0.995) は数えない
    loud = tone(0.5, 8000) * 1.99
    quiet = tmp_path / "quiet"
    for bits in (16, 24, 32):
        write_wave(quiet / f"loud{bits}.wav", loud, 8000, bits=bits)
    write_float_wave(quiet / "loudf.wav", loud, 8000)
    result = profile(load_data(quiet))
    assert result["items"]["sampled"] == 4
    assert result["items"]["clipped"] == 0
    codes = {item["code"] for item in result["findings"]}
    assert "clipped_items" not in codes

    # 端に張り付いた音は全ての bit 数で, 浮動小数は 1 を超える値も数える
    clipped = np.clip(tone(0.5, 8000) * 4, -1, 1)
    harsh = tmp_path / "harsh"
    for bits in (8, 16, 24, 32):
        write_wave(harsh / f"clip{bits}.wav", clipped, 8000, bits=bits)
    write_float_wave(harsh / "clipf.wav", clipped, 8000)
    write_float_wave(harsh / "overf.wav", tone(0.5, 8000) * 4, 8000)
    result = profile(load_data(harsh))
    assert result["items"]["clipped"] == 6
    codes = {item["code"] for item in result["findings"]}
    assert "clipped_items" in codes


def test_clip_level_bounds() -> None:
    """
    音の割れとする振幅の端の境界を, 形式と bit 数ごとに確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 整数の形式は正の最大の値から (1 つ下は数えない)
    for bits in (8, 16, 24):
        full = 1 << (bits - 1)
        level = clip_level(bits, "pcm")
        assert np.float32((full - 1) / full) >= level
        assert np.float32((full - 2) / full) < level

    # 32 bit の PCM の端は float32 で 1 に丸まり, 0.995 は数えない
    assert clip_level(32, "pcm") == np.float32(1.0)
    assert np.float32(0.995) < clip_level(32, "pcm")

    # 浮動小数は 1 から (1 を超える値も)
    level = clip_level(32, "float")
    assert np.float32(1.0) >= level
    assert np.float32(1.5) >= level
    assert np.float32(0.9999) < level


def sine(frequency: float, seconds: float, rate: int) -> np.ndarray:
    """
    決まった周波数の正弦波を (標本の数, 1) で作る関数

    Parameters
    ----------
    frequency: float
        周波数 (Hz)
    seconds: float
        長さ (秒)
    rate: int
        標本化の周波数

    Returns
    ----------
    samples: np.ndarray
        (標本の数, 1) の float32 (振幅 0.5)
    """
    # 時刻ごとの値
    times = np.arange(int(round(seconds * rate))) / rate
    samples = (0.5 * np.sin(2 * np.pi * frequency * times))[:, None]
    samples = samples.astype(np.float32)

    return samples


def test_resample() -> None:
    """
    多相のフィルタと線形補間で周波数を変え, 長さ, 値, 折り返しを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 48000 から 16000: 長さは ceil, 440 Hz の波は先の周波数の波と同じ
    # (端の外は 0 なので両端を除く)
    audio = {"samples": sine(440, 1.0, 48000), "sample_rate": 48000}
    for method in ("sinc", "linear"):
        done = resample(audio, {"rate": 16000, "method": method}, {})
        assert done["sample_rate"] == 16000
        assert done["samples"].shape == (16000, 1)
        assert done["samples"].dtype == np.float32
        expected = sine(440, 1.0, 16000)
        gap = np.abs(done["samples"] - expected)[200:-200].max()
        assert gap < 2e-3

    # 割り切れない比 (44100 から 16000) と, 上げる向き (8000 から 22050)
    odd = {"samples": sine(440, 0.5, 44100), "sample_rate": 44100}
    done = resample(odd, {"rate": 16000}, {})
    assert done["samples"].shape == (-(-22050 * 160 // 441), 1)
    gap = np.abs(done["samples"][:8000] - sine(440, 0.5, 16000))[200:-200]
    assert gap.max() < 2e-3
    low = {"samples": sine(440, 0.5, 8000), "sample_rate": 8000}
    done = resample(low, {"rate": 22050}, {})
    assert done["samples"].shape == (11025, 1)
    gap = np.abs(done["samples"] - sine(440, 0.5, 22050))[400:-400]
    assert gap.max() < 2e-3

    # 先の周波数の半分より高い音 (7 kHz を 8 kHz に) の落ちと, 線形補間の
    # 折り返し. 半分より低い音 (6 kHz を 16 kHz に) はそのまま
    high = {"samples": sine(7000, 0.5, 48000), "sample_rate": 48000}
    sinc = resample(high, {"rate": 8000}, {})["samples"][200:-200]
    linear = resample(
        high, {"rate": 8000, "method": "linear"}, {}
    )["samples"][200:-200]
    assert np.sqrt(np.mean(sinc ** 2)) < 0.01
    assert np.sqrt(np.mean(linear ** 2)) > 0.1
    near = {"samples": sine(6000, 0.5, 48000), "sample_rate": 48000}
    kept = resample(near, {"rate": 16000}, {})["samples"][200:-200]
    assert abs(np.sqrt(np.mean(kept ** 2)) - 0.5 / np.sqrt(2)) < 0.01

    # 互いに素な比 (44099 から 48000) と, 極端に下げる比 (出力の標本ごとの
    # 繰り返し), 線形補間で上げる向き
    prime = {"samples": sine(440, 0.2, 44099), "sample_rate": 44099}
    done = resample(prime, {"rate": 48000}, {})["samples"]
    assert done.shape == (-(-len(prime["samples"]) * 48000 // 44099), 1)
    gap = np.abs(done[:9600] - sine(440, 0.2, 48000))[300:-300]
    assert gap.max() < 2e-3
    flat_long = {"samples": np.full((48000, 1), 0.5, np.float32),
                 "sample_rate": 48000}
    tiny = resample(flat_long, {"rate": 100}, {})["samples"]
    assert tiny.shape == (100, 1)
    assert np.allclose(tiny[20:-20], 0.5, atol=1e-3)
    assert resample(flat_long, {"rate": 1}, {})["samples"].shape == (1, 1)
    done = resample(low, {"rate": 22050, "method": "linear"}, {})
    assert done["samples"].shape == (11025, 1)
    gap = np.abs(done["samples"] - sine(440, 0.5, 22050))[400:-400]
    assert gap.max() < 2e-2

    # 直流はそのまま, 同じ周波数と空の音声はそのまま, ステレオも
    flat = {"samples": np.full((3000, 2), 0.25, np.float32),
            "sample_rate": 44100}
    done = resample(flat, {"rate": 16000}, {})["samples"]
    assert done.shape[1] == 2
    assert np.allclose(done[100:-100], 0.25, atol=1e-5)
    same = resample(audio, {"rate": 48000}, {})
    assert np.array_equal(same["samples"], audio["samples"])
    empty = {"samples": np.zeros((0, 1), np.float32), "sample_rate": 8000}
    assert resample(empty, {"rate": 16000}, {})["samples"].shape == (0, 1)

    # 引数の誤り
    for params in ({"rate": 0}, {"rate": 1.5}, {"rate": True},
                   {"rate": 16000, "method": "cubic"}):
        with pytest.raises(ValueError, match="resample"):
            resample(audio, params, {})


def test_mix_and_fix_length() -> None:
    """
    チャンネルと長さをそろえる手順の値と誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # チャンネル: 平均, 並べる, 同じ数はそのまま, 決まらない組は誤り
    stereo = np.stack([np.ones(4), np.zeros(4)], axis=1).astype(np.float32)
    audio = {"samples": stereo, "sample_rate": 8000}
    mono = mix_channels(audio, {"channels": 1}, {})["samples"]
    assert mono.shape == (4, 1) and np.allclose(mono, 0.5)
    wide = mix_channels(
        {"samples": mono, "sample_rate": 8000}, {"channels": 2}, {}
    )["samples"]
    assert wide.shape == (4, 2) and np.allclose(wide, 0.5)
    assert np.array_equal(
        mix_channels(audio, {"channels": 2}, {})["samples"], stereo
    )
    with pytest.raises(ValueError, match="混ぜ方"):
        mix_channels(audio, {"channels": 3}, {})
    with pytest.raises(ValueError, match="channels"):
        mix_channels(audio, {"channels": 0}, {})

    # 長さ: 先頭を残した切り, 後ろの 0 の埋め, 真ん中を残した切りと埋め
    ramp = np.arange(10, dtype=np.float32)[:, None]
    audio = {"samples": ramp, "sample_rate": 10}
    cut = fix_length(audio, {"seconds": 0.4}, {})["samples"]
    assert cut[:, 0].tolist() == [0, 1, 2, 3]
    padded = fix_length(audio, {"seconds": 1.3}, {})["samples"]
    assert padded.shape == (13, 1) and padded[10:, 0].tolist() == [0] * 3
    middle = fix_length(audio, {"seconds": 0.5, "align": "center"}, {})
    assert middle["samples"][:, 0].tolist() == [2, 3, 4, 5, 6]
    around = fix_length(audio, {"seconds": 1.3, "align": "center"}, {})
    assert around["samples"][:, 0].tolist() == [0, *range(10), 0, 0]
    for params in ({"seconds": 0}, {"seconds": "1"},
                   {"seconds": 1.0, "align": "end"}):
        with pytest.raises(ValueError, match="fix_length"):
            fix_length(audio, params, {})


def test_audio_export(tmp_path: Path) -> None:
    """
    周波数, チャンネル, 長さの混ざった音声を, レシピで prepare と
    export_dataset に通し, 書き出したデータセットを読めることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 2 クラス. 周波数, チャンネル, 長さが混ざり, 読めないファイルも 1 つ
    root = tmp_path / "sounds"
    for index in range(5):
        write_wave(root / "dog" / f"d{index}.wav", tone(0.5, 16000))
        write_wave(root / "cat" / f"c{index}.wav", tone(0.8, 44100, 2),
                   44100)
    write_wave(root / "cat" / "low.wav", tone(1.5, 8000), 8000, bits=8)
    (root / "dog" / "broken.wav").write_bytes(b"RIFF....WAVEjunk")
    data = load_data(root)
    steps = [
        {"op": "resample", "rate": 16000},
        {"op": "mix_channels", "channels": 1},
        {"op": "fix_length", "seconds": 1.0},
    ]
    recipe = {
        "target": "label", "task": "classification", "item_steps": steps,
        "split": {"method": "stratified", "valid": 0.3, "seed": 0},
    }

    # 中身を読まない確かめ: 読めない件, そろえる手順の無いもの, 引数の誤り
    notes = prepare(
        data, {**recipe, "item_steps": []}, read_items=False
    ).notes
    assert any("読めない件が 1 件" in note for note in notes)
    for op in ("resample", "mix_channels", "fix_length"):
        assert any(op in note for note in notes)
    with pytest.raises(ValueError, match="method"):
        prepare(data, {**recipe, "item_steps": [
            {"op": "resample", "rate": 16000, "method": "cubic"}
        ]}, read_items=False)
    for spec in ({"op": "resample", "rate": 0},
                 {"op": "mix_channels", "channels": 1.5},
                 {"op": "fix_length", "seconds": -1}):
        with pytest.raises(ValueError, match=spec["op"]):
            prepare(data, {**recipe, "item_steps": [spec]},
                    read_items=False)
    with pytest.raises(ValueError, match="audio に使えない"):
        prepare(data, {**recipe, "item_steps": [{"op": "resize"}]},
                read_items=False)

    # 周波数をそろえる場合, 長さは秒で比べる (16 kHz の 1 秒と 8 kHz の
    # 2 秒は標本の数が同じでも, そろえた後は違う)
    pair = tmp_path / "pair"
    write_wave(pair / "a" / "one.wav", tone(1.0, 16000))
    write_wave(pair / "b" / "two.wav", tone(2.0, 8000), 8000)
    write_wave(pair / "b" / "same.wav", tone(1.0, 44100), 44100)
    notes = prepare(load_data(pair), {
        **recipe, "item_steps": [{"op": "resample", "rate": 16000}],
        "split": {"method": "none"},
    }, read_items=False).notes
    assert any("長さ" in note for note in notes)

    # 読めない件での, 除き方を添えた誤り
    with pytest.raises(ValueError, match="error がある"):
        export_dataset(data, recipe, tmp_path / "bad")

    # 読めない件を除いた書き出し
    recipe["steps"] = [
        {"op": "filter_rows", "column": "error", "compare": "isna"}
    ]
    info = export_dataset(data, recipe, tmp_path / "ds")
    assert info["format_version"] == 1 and info["kind"] == "audio"
    assert info["classes"] == ["cat", "dog"] and info["features"] == []
    assert info["splits"]["train"]["shape"][1:] == [1, 16000]
    total = sum(split["rows"] for split in info["splits"].values())
    assert total == 11

    # 読み込み: 配列とテンソルの形, 書き出した手順
    dataset = load_dataset(tmp_path / "ds")
    x, y = dataset.arrays("train")
    assert x.shape[1:] == (1, 16000) and x.dtype == np.float32
    assert len(x) == len(y) and set(y.tolist()) <= {0, 1}
    assert [s["op"] for s in dataset.recipe["item_steps"]] == [
        "resample", "mix_channels", "fix_length"
    ]
    shapes = model_inputs(tmp_path / "ds")
    assert shapes["inputs"][0]["shape"] == [None, 1, 16000]
    assert shapes["outputs"] == 2

    # 推論: ファイルの場所でも read_audio の辞書でも, 学習と同じ配列
    frame = dataset.frames["train"]
    first = root / frame["path"].iloc[0]
    from_path = dataset.transform_item(first)
    assert np.array_equal(from_path, x[0])
    assert np.array_equal(dataset.transform_item(read_audio(first)), x[0])
    torch = pytest.importorskip("torch")
    tensors = dataset.tensors("train")
    assert tensors[0].dtype == torch.float32
    assert tuple(tensors[0].shape[1:]) == (1, 16000)
