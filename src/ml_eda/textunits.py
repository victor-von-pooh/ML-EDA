import math
from typing import Any

import numpy as np

# 比べる見本の件数の上限
COMPARE_SAMPLE = 5000

# 比べる部分語の語彙の大きさ
SUBWORD_SIZES = (500, 2000, 8000)

# 勧める長さの上限
MAX_LENGTH = 512

# 語彙に無い語の割合の許容
UNKNOWN_LIMIT = 0.01

# 大きい語彙に移るのに要る, 並びが短くなる割合
SHORTER = 0.2

# 当てはめた部分語の区切り (同じ語彙は 1 度だけ作る)
SUBWORD_CACHE: dict[str, Any] = {}


def has_subword() -> bool:
    """
    部分語の区切りに使うライブラリ (tokenizers) があるかを返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    available: bool
        tokenizers を読み込めるか
    """
    # 入っていない場合は部分語の候補の除外
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        return False

    return True


def fit_subword(texts: list[str], size: int) -> str:
    """
    文から BPE の部分語の区切りを作る関数

    Parameters
    ----------
    texts: list[str]
        学習用の文 (小文字にするなどは済ませたもの)
    size: int
        語彙の大きさの上限 (埋め草と語彙に無い語を含む)

    Returns
    ----------
    serialized: str
        区切りの JSON (レシピに保存する)
    """
    # 0 は埋め草, 1 は語彙に無い語. 空白と記号で区切ってから部分語へ
    # (1 つの部分語は 16 文字まで)
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()
    ])
    # 文字も語彙の上限に含める (多い文字から. 外れた文字は語彙に無い語)
    size = max(int(size), 3)
    trainer = trainers.BpeTrainer(
        vocab_size=size, special_tokens=["<pad>", "<unk>"],
        limit_alphabet=size - 2, max_token_length=16, show_progress=False
    )
    tokenizer.train_from_iterator(
        [text for text in texts if text.strip()], trainer
    )
    serialized = tokenizer.to_str()

    return serialized


def subword_tokenizer(serialized: str) -> Any:
    """
    保存した部分語の区切りを読み込む関数

    Parameters
    ----------
    serialized: str
        区切りの JSON

    Returns
    ----------
    tokenizer: Any
        tokenizers の Tokenizer
    """
    # 読み込みは重いので, 4 つまで保持 (古いものから破棄)
    tokenizer = SUBWORD_CACHE.get(serialized)
    if tokenizer is None:
        from tokenizers import Tokenizer

        if len(SUBWORD_CACHE) >= 4:
            SUBWORD_CACHE.pop(next(iter(SUBWORD_CACHE)))
        tokenizer = Tokenizer.from_str(serialized)
        SUBWORD_CACHE[serialized] = tokenizer

    return tokenizer


def subword_vocabulary(serialized: str) -> list[str]:
    """
    部分語の区切りの語彙を番号の順に返す関数

    Parameters
    ----------
    serialized: str
        区切りの JSON

    Returns
    ----------
    vocabulary: list[str]
        番号の順の部分語
    """
    # 番号の順の並び
    pieces = subword_tokenizer(serialized).get_vocab()
    vocabulary = [
        piece for piece, _ in sorted(pieces.items(), key=lambda p: p[1])
    ]

    return vocabulary


def round_length(tokens: float) -> int:
    """
    トークンの数から並びの長さを決める関数

    Parameters
    ----------
    tokens: float
        1 つの文のトークンの数 (95 % 点)

    Returns
    ----------
    length: int
        8 の倍数に切り上げた長さ (8 以上 512 以下)
    """
    # 8 の倍数は計算の効率のため
    length = int(min(MAX_LENGTH, max(8, math.ceil(tokens / 8) * 8)))

    return length


def measure(
    unit: str, fit_texts: list[str], test_texts: list[str],
    lowercase: bool, size: int
) -> dict:
    """
    1 つの区切り方で語彙を作り, 別の文で測る関数

    Parameters
    ----------
    unit: str
        char, word, subword
    fit_texts: list[str]
        語彙を作る文
    test_texts: list[str]
        測る文
    lowercase: bool
        小文字にするか
    size: int
        部分語の語彙の大きさ (char と word では使わない)

    Returns
    ----------
    result: dict
        unit, vocabulary (語彙の大きさ), tokens_mean, tokens_p95,
        unknown_rate, length (勧める長さ), truncated_rate, max_tokens
    """
    # 語彙に無いトークンは 1 番
    from ml_eda.itemprep import encode_tokens, fit_vocabulary

    params: dict = {"unit": unit, "lowercase": lowercase}
    if unit == "subword":
        params["max_tokens"] = size
    state = fit_vocabulary(fit_texts, params)
    encoded = [encode_tokens(text, params, state) for text in test_texts]
    counts = np.array([len(ids) for ids in encoded], dtype=np.float64)
    total = max(float(counts.sum()), 1.0)
    unknown = sum(int(np.sum(np.asarray(ids) == 1)) for ids in encoded)
    p95 = float(np.percentile(counts, 95)) if len(counts) else 0.0
    length = round_length(p95)
    result = {
        "unit": unit,
        "vocabulary": len(state["vocabulary"]),
        "tokens_mean": float(counts.mean()) if len(counts) else 0.0,
        "tokens_p95": p95,
        "unknown_rate": unknown / total,
        "length": length,
        "truncated_rate": (
            float(np.mean(counts > length)) if len(counts) else 0.0
        ),
        "max_tokens": size if unit == "subword" else 20000,
    }

    return result


def unit_label(candidate: dict) -> str:
    """
    区切り方の候補を画面に出す名前にする関数

    Parameters
    ----------
    candidate: dict
        measure の結果

    Returns
    ----------
    label: str
        文字, 語, 部分語 (部分語は語彙の大きさも)
    """
    # 部分語は語彙の大きさで区別
    names = {"char": "文字", "word": "語", "subword": "部分語"}
    label = names.get(candidate["unit"], candidate["unit"])
    if candidate["unit"] == "subword":
        label += f" (語彙 {candidate['vocabulary']})"

    return label


def recommend(candidates: list[dict]) -> tuple[dict, str]:
    """
    測った候補から勧める区切り方と理由を決める関数

    Parameters
    ----------
    candidates: list[dict]
        measure の結果 (1 つ以上)

    Returns
    ----------
    best: dict
        勧める候補
    reason: str
        勧めた理由 (画面に出す)
    """
    # 語彙に無い語が少ない候補を語彙の小さい順にたどり, 並びが 2 割以上
    # 短くなる場合だけ大きい語彙へ
    usable = [c for c in candidates if c["unknown_rate"] <= UNKNOWN_LIMIT]
    if not usable:
        best = min(candidates, key=lambda c: c["unknown_rate"])
        reason = (
            "どの区切り方でも語彙に無い語が 1 % を超えるので, 最も少ない"
            f"{unit_label(best)}にしました"
        )
    elif len(usable) == 1:
        best = usable[0]
        reason = (
            f"語彙に無い語が 1 % 以下なのは{unit_label(best)}だけなので"
            "選びました"
        )
    else:
        ordered = sorted(usable, key=lambda c: c["vocabulary"])
        best = ordered[0]
        previous = None
        for candidate in ordered[1:]:
            if candidate["tokens_p95"] <= best["tokens_p95"] * (1 - SHORTER):
                previous, best = best, candidate
        if previous is None:
            reason = (
                "語彙の大きい区切り方でも並びが 2 割以上短くならないので, "
                f"語彙の最も小さい{unit_label(best)}にしました"
            )
        else:
            reason = (
                f"{unit_label(best)}は, 語彙に無い語が 1 % 以下で, 語彙の"
                f"小さい{unit_label(previous)}より並びが 2 割以上短い "
                f"(95 % の文が {best['tokens_p95']:.0f} 個以下. "
                f"{unit_label(previous)}は {previous['tokens_p95']:.0f} 個)"
                " ので選びました"
            )

    return best, reason


def compare_tokenizers(
    texts: list[str], seed: int = 0, sample: int = COMPARE_SAMPLE
) -> dict:
    """
    文を区切る単位 (文字, 語, 部分語) を比べ, vocabulary の引数を勧める関数

    Parameters
    ----------
    texts: list[str]
        文 (空の文は除く)
    seed: int = 0
        見本を選び, 分ける乱数のシード
    sample: int = COMPARE_SAMPLE
        比べる見本の件数の上限

    Returns
    ----------
    result: dict
        candidates (区切り方ごとの measure の結果), recommended
        (vocabulary の手順), reason (勧めた理由), notes (注意)
    """
    # 見本を 8 対 2 に分け, 8 割で語彙の作成, 2 割で計測
    values = [str(text) for text in texts if str(text).strip()]
    if not values:
        return {
            "candidates": [], "recommended": None,
            "reason": "空でない文がありません", "notes": [],
        }
    rng = np.random.default_rng(seed)
    if len(values) > sample:
        chosen = rng.choice(len(values), sample, replace=False)
        values = [values[int(k)] for k in chosen]
    order = rng.permutation(len(values))
    cut = int(len(values) * 0.8)
    fit_texts = [values[int(k)] for k in order[:cut]]
    test_texts = [values[int(k)] for k in order[cut:]]
    notes = []
    if len(test_texts) < 10:
        # 少ない文は, 語彙を作った文での計測 (語彙に無い割合は甘くなる)
        fit_texts = test_texts = values
        notes.append("文が少ないので, 語彙を作った文で測りました")

    # 部分語は語彙の大きさを変えての比較 (文字の種類より十分に大きいもの)
    kinds = len({c for text in fit_texts for c in text.lower()})
    units = [("char", 0), ("word", 0)]
    if has_subword():
        units += [
            ("subword", size) for size in SUBWORD_SIZES if size > kinds + 10
        ]
    else:
        notes.append(
            "tokenizers が無いので, 部分語は比べていません "
            "(pip install tokenizers)"
        )
    candidates: list[dict] = []
    for unit, size in units:
        measured = measure(unit, fit_texts, test_texts, True, size)
        # 文が少なく語彙が上限まで育たない場合は, 同じ部分語を 1 つに
        grown = [c["vocabulary"] for c in candidates if c["unit"] == unit]
        if unit == "subword" and measured["vocabulary"] in grown:
            continue
        candidates.append(measured)

    best, reason = recommend(candidates)
    recommended = {
        "op": "vocabulary", "unit": best["unit"], "length": best["length"],
        "max_tokens": best["max_tokens"],
    }
    result = {
        "candidates": candidates, "recommended": recommended,
        "reason": reason, "notes": notes,
    }

    return result
