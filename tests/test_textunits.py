from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    compare_tokenizers, export_dataset, load_data, load_dataset, profile,
    textunits
)
from ml_eda.itemprep import check_item_steps, encode_tokens

# 部分語の試験は tokenizers が要る
needs_subword = pytest.mark.skipif(
    not textunits.has_subword(), reason="tokenizers がありません"
)


def japanese_reviews(count: int, seed: int = 0) -> list[str]:
    """
    句を組み合わせた日本語の文を作る関数

    Parameters
    ----------
    count: int
        文の数
    seed: int = 0
        乱数のシード

    Returns
    ----------
    texts: list[str]
        空白で区切らない文
    """
    # 同じ句が何度も出る (部分語で短くなる)
    rng = np.random.default_rng(seed)
    subjects = ["本体は", "配送は", "説明書は", "付属品は", "梱包は"]
    states = ["問題なかった", "期待通りではない", "しっかりしている",
              "悪くない", "壊れていた"]
    texts = [
        "、".join(
            str(rng.choice(subjects)) + str(rng.choice(states))
            for _ in range(int(rng.integers(2, 5)))
        ) + "。"
        for _ in range(count)
    ]

    return texts


def test_compare_char_for_few_kinds() -> None:
    """
    並びが短くならない文では文字を勧め, 測った値の形を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 文字がばらばらな短い文は, どの区切り方でも長さが変わらない
    rng = np.random.default_rng(0)
    letters = list("abcdefghij")
    texts = ["".join(rng.choice(letters, 6)) for _ in range(300)]
    result = compare_tokenizers(texts)
    units = [c["unit"] for c in result["candidates"]]
    assert units[:2] == ["char", "word"]
    char = result["candidates"][0]
    assert char["vocabulary"] == 12 and char["unknown_rate"] == 0.0
    assert char["tokens_p95"] == 6.0 and char["length"] == 8
    assert result["recommended"]["unit"] == "char"

    # 空の文だけなら勧めない
    empty = compare_tokenizers(["", "  "])
    assert empty["recommended"] is None and empty["candidates"] == []


def test_compare_word_for_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    同じ語が繰り返す英語では, 文字より短い語を勧める関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        部分語を外すため

    Returns
    ----------
    None
    """
    # 部分語が無い場合は, その旨を注意に書く
    monkeypatch.setattr(textunits, "has_subword", lambda: False)
    rng = np.random.default_rng(1)
    words = ["good", "movie", "bad", "story", "acting", "great", "boring"]
    texts = [" ".join(rng.choice(words, 8)) for _ in range(400)]
    result = compare_tokenizers(texts)
    assert [c["unit"] for c in result["candidates"]] == ["char", "word"]
    assert result["notes"] and "tokenizers" in result["notes"][0]
    assert result["recommended"]["unit"] == "word"
    assert result["recommended"]["length"] == 8

    # 部分語の手順は tokenizers が無いと誤り
    with pytest.raises(ValueError, match="tokenizers"):
        check_item_steps([{"op": "vocabulary", "unit": "subword"}], "text")
    with pytest.raises(ValueError, match="unit"):
        check_item_steps([{"op": "vocabulary", "unit": "byte"}], "text")


@needs_subword
def test_subword_export(tmp_path: Path) -> None:
    """
    日本語の文で部分語を勧め, 書き出しと新しい文の変換が同じ区切りに
    なることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 分析で部分語を勧める (文字より並びが短い)
    texts = japanese_reviews(600)
    frame = pd.DataFrame({
        "review": texts, "label": ["良い", "悪い"] * 300,
    })
    path = tmp_path / "reviews.csv"
    frame.to_csv(path, index=False)
    data = load_data(path, text_column="review")
    result = profile(data, "label")["items"]["tokenizers"]
    char, _, subword = result["candidates"]
    assert subword["unit"] == "subword"
    assert subword["tokens_p95"] < char["tokens_p95"] * 0.8
    assert subword["unknown_rate"] <= 0.01
    recommended = result["recommended"]
    assert recommended["unit"] == "subword"

    # 勧めた手順で書き出し, 新しい文にも同じ区切りをかける
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [recommended],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    export_dataset(data, recipe, tmp_path / "ds")
    dataset = load_dataset(tmp_path / "ds")
    arrays, _ = dataset.arrays("train")
    assert arrays.shape[1] == recommended["length"]
    state = dataset.recipe["item_steps"][0]["state"]
    assert state["vocabulary"][:2] == ["<pad>", "<unk>"]
    first = dataset.transform_item(texts[0])
    again = dataset.transform_item(texts[0])
    assert first.tolist() == again.tolist() and first[0] > 1
    # 文字より短く, 埋め草は 0. 知らない文字は 1
    assert int((first > 0).sum()) < len(texts[0])
    assert 1 in dataset.transform_item("☃☃☃").tolist()

    # 部分語の区切りが無い状態は誤り
    with pytest.raises(ValueError, match="区切り"):
        encode_tokens(texts[0], {"unit": "subword"}, {"vocabulary": []})
