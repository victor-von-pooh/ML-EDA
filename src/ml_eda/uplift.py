import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.loaders import load_data
from ml_eda.profile import association, finding, infer_type, to_json
from ml_eda.recipe import OUTCOME_KINDS

# 介入の記録の表として読む種類
UPLIFT_KINDS = ("table", "timeseries")

# 少なすぎる群の行の数と, 処置の割合の下限 (上限は 1 から引いた値)
MIN_GROUP_ROWS = 30
MIN_TREATED_RATE = 0.05

# 釣り合いが大きく崩れたと見なす標準化した平均の差の絶対値
IMBALANCE = 0.25

# 処置と結果の後に決まる列と見なす関連の強さ
TREATMENT_LEAK = 0.9
OUTCOME_LEAK = 0.98

# 片方の群でだけ値が変わる列を確かめる群の行の数の下限
CONSTANT_GROUP_ROWS = 5

# 標準化した平均の差を求めるカテゴリの値の数の上限
BALANCE_VALUES = 20

# 釣り合いを求める分析の型
BALANCE_TYPES = ("numeric", "boolean", "categorical")

# 気付きの文に挙げる列の数
FINDING_EXAMPLES = 5


def zero_one(values: pd.Series) -> tuple[Optional[pd.Series], list[str]]:
    """
    欠けの無い値を, 0 と 1 の整数にする関数

    Parameters
    ----------
    values: pd.Series
        欠けの無い値

    Returns
    ----------
    labels: Optional[pd.Series]
        int64 の 0 と 1 (真偽値でも 0 と 1 だけの数でもなければ None)
    others: list[str]
        0 と 1 に読めない値の例 (最大 3 つ)
    """
    # 真偽値と, 0 と 1 だけの数
    if pd.api.types.is_bool_dtype(values):
        labels = values.astype("int64")
        return labels, []
    numeric = (
        pd.api.types.is_numeric_dtype(values)
        or values.map(lambda value: isinstance(value, bool)).all()
    )
    if not numeric:
        others = values.astype(str).unique().tolist()[:3]
        return None, others
    others = [
        str(value) for value in values.unique().tolist()
        if value not in (0, 1)
    ][:3]
    if others:
        return None, others
    labels = values.astype("int64")

    return labels, []


def treatment_values(values: pd.Series, name: str) -> pd.Series:
    """
    処置の列の値を確かめ, 処置の群は 1, 対照の群は 0 の整数にする関数

    Parameters
    ----------
    values: pd.Series
        処置の列の値
    name: str
        処置の列の名前 (誤りの文に使う)

    Returns
    ----------
    labels: pd.Series
        int64 の 0 と 1

    Raises
    ----------
    ValueError
        欠けがある場合と, 0 と 1 (か真偽値) でない値がある場合
    """
    # 欠けと, 0 と 1 に読めない値
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(
            f"処置の列 {name} に欠けがあります ({missing} 行). 前もって"
            "落としてください (filter_rows の compare: isna など)"
        )
    labels, others = zero_one(values)
    if labels is None:
        raise ValueError(
            f"処置の列 {name} は, 処置の群を 1 か真, 対照の群を 0 か偽に"
            f"してください (他の値: {', '.join(others)})"
        )

    return labels


def choose_outcome_kind(
    values: pd.Series, name: str, kind: Optional[str] = None
) -> str:
    """
    結果の列の種類 (binary か amount) を確かめるか, 値から決める関数

    Parameters
    ----------
    values: pd.Series
        欠けの無い結果の列の値
    name: str
        結果の列の名前 (誤りの文に使う)
    kind: Optional[str] = None
        書いた結果の種類, None の場合は値から決める

    Returns
    ----------
    chosen: str
        binary か amount

    Raises
    ----------
    ValueError
        知らない種類の場合, binary で 0 と 1 (か真偽値) でない値がある
        場合, 数でない場合
    """
    # 書いた種類の確かめ
    if kind is not None and kind not in OUTCOME_KINDS:
        raise ValueError(
            f"結果の種類は binary か amount にしてください: {kind}"
        )
    labels, others = zero_one(values)
    if kind == "binary" and labels is None:
        raise ValueError(
            f"結果の列 {name} は, binary では 1 か真, 0 か偽にしてください "
            f"(他の値: {', '.join(others)})"
        )
    numeric = labels is not None or (
        pd.api.types.is_numeric_dtype(values)
        and not pd.api.types.is_bool_dtype(values)
    )
    if not numeric:
        raise ValueError(
            f"結果の列 {name} が数ではありません (例: {', '.join(others)}). "
            "買ったかは 0 と 1, 買った額は数にしてください"
        )

    # 書かなければ, 0 と 1 だけなら binary
    chosen = kind or ("binary" if labels is not None else "amount")

    return chosen


def outcome_values(values: pd.Series, kind: str) -> pd.Series:
    """
    欠けの無い結果の列の値を, 書き出す型にする関数

    Parameters
    ----------
    values: pd.Series
        欠けの無い結果の列の値 (choose_outcome_kind で確かめたもの)
    kind: str
        binary か amount

    Returns
    ----------
    converted: pd.Series
        binary は int64 の 0 と 1, amount は数のまま
    """
    # binary は 0 と 1 の整数, amount はそのまま
    if kind == "binary":
        converted = values.astype("int64")
        return converted
    converted = values

    return converted


def uplift_table(
    source: Union[str, Path, Data], loader: Optional[str], options: dict
) -> Data:
    """
    介入の記録の表を load_data で読むか, 読んだデータを使う関数

    Parameters
    ----------
    source: Union[str, Path, Data]
        表のファイルか, load_data で読んだデータ
    loader: Optional[str]
        読み込みの名前 (None は推定)
    options: dict
        読み込みの設定

    Returns
    ----------
    data: Data
        読んだ表のデータ

    Raises
    ----------
    ValueError
        読んだデータに読み込みの名前か設定を渡した場合と, 表でない場合
    """
    # 読んだデータはそのまま使う (読み方は読んだときのもの)
    if isinstance(source, Data):
        if loader is not None or options:
            raise ValueError(
                "読んだデータ (Data) を渡す場合は, 読み込みの名前と設定を"
                "渡さないでください"
            )
        data = source
    else:
        data = load_data(source, loader=loader, **options)
    if data.kind not in UPLIFT_KINDS:
        raise ValueError(
            f"介入の記録は表にしてください ({data.kind} のデータです)"
        )

    return data


def column_of(frame: pd.DataFrame, name: Any, label: str) -> pd.Series:
    """
    名前の列が 1 つだけあることを確かめて返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        列の名前を文字にそろえた表
    name: Any
        列の名前
    label: str
        列の呼び名 (誤りの文に使う)

    Returns
    ----------
    column: pd.Series
        その列

    Raises
    ----------
    ValueError
        列が無いか 2 つ以上ある場合
    """
    # 同じ名前の列は取り違えるので誤り
    count = list(frame.columns).count(str(name))
    if count != 1:
        raise ValueError(f"{label} {name} がありません (か 2 つ以上)")
    column = frame[str(name)]

    return column


def balance_columns(
    frame: pd.DataFrame, columns: Any, reserved: list[str], roles: dict
) -> list[tuple[str, str]]:
    """
    釣り合いを求める列と, その分析の型を決める関数

    Parameters
    ----------
    frame: pd.DataFrame
        列の名前を文字にそろえた表
    columns: Any
        列の名前の並び, None の場合は数, 真偽, カテゴリの型の全ての列
    reserved: list[str]
        除く列 (処置, 結果, 行の id)
    roles: dict
        列の役割 (型の推定に使う)

    Returns
    ----------
    chosen: list[tuple[str, str]]
        (列の名前, 分析の型) の並び

    Raises
    ----------
    ValueError
        渡した列が無いか, 除く列を渡した場合
    """
    # 省くと, 除く列の他の数, 真偽, カテゴリの列
    if columns is None:
        chosen = []
        for name in frame.columns:
            kind = infer_type(frame[name], roles)
            if name not in reserved and kind in BALANCE_TYPES:
                chosen.append((name, kind))
        return chosen

    # 渡した列の確かめ
    names = [columns] if isinstance(columns, str) else list(columns)
    chosen = []
    for name in names:
        column = column_of(frame, name, "釣り合いの列")
        if str(name) in reserved:
            raise ValueError(
                f"釣り合いの列に処置, 結果, 行の id の列は渡せません: {name}"
            )
        chosen.append((str(name), infer_type(column, roles)))

    return chosen


def smd(treated: np.ndarray, control: np.ndarray) -> Optional[float]:
    """
    2 つの群の標準化した平均の差を求める関数

    Parameters
    ----------
    treated: np.ndarray
        処置の群の欠けの無い値
    control: np.ndarray
        対照の群の欠けの無い値

    Returns
    ----------
    value: Optional[float]
        (処置の群の平均 − 対照の群の平均) / √((分散 + 分散) / 2). 群に値が
        2 つ未満か, 分母が 0 で平均が違う場合は None
    """
    # 不偏の分散で, 分母が 0 なら平均が同じかで決める
    if len(treated) < 2 or len(control) < 2:
        return None
    difference = float(treated.mean() - control.mean())
    scale = math.sqrt(
        (float(treated.var(ddof=1)) + float(control.var(ddof=1))) / 2
    )
    if scale == 0:
        value = 0.0 if difference == 0 else None
        return value
    value = difference / scale

    return value


def column_balance(
    column: pd.Series, kind: str, treated: np.ndarray
) -> dict:
    """
    1 つの列の処置の群と対照の群の釣り合いを求める関数

    Parameters
    ----------
    column: pd.Series
        列 (処置と結果の両方が分かる行)
    kind: str
        分析の型
    treated: np.ndarray
        行ごとの処置の群か (真偽)

    Returns
    ----------
    result: dict
        column, type, smd, value (カテゴリの列で SMD の最も大きい値)
    """
    # 数と, 数か真偽値の 2 値の列は値のまま
    result = {"column": str(column.name), "type": kind, "smd": None,
              "value": None}
    known = column.notna().to_numpy()
    numeric = pd.api.types.is_numeric_dtype(column) or (
        pd.api.types.is_bool_dtype(column)
    )
    if kind in ("numeric", "boolean") and numeric:
        values = column.to_numpy(dtype=float, na_value=np.nan)
        result["smd"] = smd(
            values[known & treated], values[known & ~treated]
        )
        return result

    # カテゴリの列は値ごとの 0 と 1 の割合 (値の多い列は求めない)
    text = column.astype(str).to_numpy()
    values = pd.unique(text[known])
    if len(values) > BALANCE_VALUES:
        return result
    best: Optional[float] = None
    for value in values:
        flags = (text == value).astype(float)
        found = smd(flags[known & treated], flags[known & ~treated])
        if found is not None and (best is None or abs(found) > abs(best)):
            best = found
            result["value"] = str(value)
    result["smd"] = best

    return result


def splits_constant(column: pd.Series, treated: np.ndarray) -> bool:
    """
    片方の群でだけ値が変わる (もう片方は値が 1 つか全て欠け) かを返す関数

    Parameters
    ----------
    column: pd.Series
        列 (処置と結果の両方が分かる行)
    treated: np.ndarray
        行ごとの処置の群か (真偽)

    Returns
    ----------
    found: bool
        両方の群が 5 行以上あり, 片方の値が 1 つ以下でもう片方が 2 つ以上
    """
    # 群ごとの異なる値の数
    if treated.sum() < CONSTANT_GROUP_ROWS:
        return False
    if (~treated).sum() < CONSTANT_GROUP_ROWS:
        return False
    text = column.astype(str).where(column.notna(), None)
    in_treated = text[treated].nunique()
    in_control = text[~treated].nunique()
    found = min(in_treated, in_control) <= 1 < max(in_treated, in_control)

    return found


def post_treatment_findings(
    frame: pd.DataFrame, chosen: list[tuple[str, str]],
    treatment: pd.Series, outcome: pd.Series, kind: str
) -> list[dict]:
    """
    処置の後に決まる列 (処置か結果とほぼ同じ情報, 片方の群でだけ値が
    変わる) を気付きにする関数

    Parameters
    ----------
    frame: pd.DataFrame
        処置と結果の両方が分かる行の表
    chosen: list[tuple[str, str]]
        (列の名前, 分析の型) の並び
    treatment: pd.Series
        処置の 0 と 1
    outcome: pd.Series
        結果の値
    kind: str
        結果の種類 (binary か amount)

    Returns
    ----------
    findings: list[dict]
        列ごとの post_treatment の気付き
    """
    # 列ごとに, 最初に当たった理由を 1 つ
    findings = []
    treated = (treatment == 1).to_numpy()
    task = "classification" if kind == "binary" else "regression"
    for name, column_type in chosen:
        column = frame[name]
        reason = None
        to_treatment = association(
            column, column_type, treatment, "classification"
        )
        to_outcome = association(column, column_type, outcome, task)
        if to_treatment is not None and to_treatment >= TREATMENT_LEAK:
            reason = (
                f"処置とほぼ同じ情報を持っています (関連の強さ "
                f"{to_treatment:.2f})"
            )
        elif to_outcome is not None and to_outcome >= OUTCOME_LEAK:
            reason = (
                f"結果とほぼ同じ情報を持っています (関連の強さ "
                f"{to_outcome:.2f})"
            )
        elif splits_constant(column, treated):
            reason = "片方の群でだけ値が変わります"
        if reason is None:
            continue
        findings.append(finding(
            "warning", "post_treatment",
            f"{name} は{reason}. 処置の後に決まる列 (クーポンを使ったか, "
            "結果から作った列など) は効果を取り違えるので, 落としてください",
            [name]
        ))

    return findings


def group_findings(
    treated_rows: int, control_rows: int, treatment: str
) -> list[dict]:
    """
    処置の群か対照の群が少なすぎることを気付きにする関数

    Parameters
    ----------
    treated_rows: int
        処置の群の行の数
    control_rows: int
        対照の群の行の数
    treatment: str
        処置の列の名前

    Returns
    ----------
    findings: list[dict]
        small_group の気付き (無ければ空)
    """
    # 行の数と処置の割合
    total = treated_rows + control_rows
    rate = treated_rows / total if total else 0.0
    small = min(treated_rows, control_rows) < MIN_GROUP_ROWS
    skewed = not MIN_TREATED_RATE <= rate <= 1 - MIN_TREATED_RATE
    if not small and not skewed:
        return []
    findings = [finding(
        "warning", "small_group",
        f"処置の群が {treated_rows} 行, 対照の群が {control_rows} 行です "
        f"(処置の割合 {rate:.1%}). 群の行が少ないと効果の差が揺れ, 分割に"
        "両方の群を置けないことがあります. 記録を足すか, 配った割合を"
        "確かめてください", [treatment]
    )]

    return findings


def imbalance_findings(
    balance: list[dict], leaked: set[str], treatment: str
) -> list[dict]:
    """
    群どうしの釣り合いが大きく崩れた列を気付きにする関数

    Parameters
    ----------
    balance: list[dict]
        列ごとの釣り合い
    leaked: set[str]
        処置の後に決まる列 (除く)
    treatment: str
        処置の列の名前

    Returns
    ----------
    findings: list[dict]
        treatment_imbalance の気付き (無ければ空)
    """
    # 標準化した平均の差の絶対値が大きい列
    names = [
        item["column"] for item in balance
        if item["column"] not in leaked and item["smd"] is not None
        and abs(item["smd"]) >= IMBALANCE
    ]
    if not names:
        return []
    shown = ", ".join(names[:FINDING_EXAMPLES])
    more = f" ほか {len(names) - FINDING_EXAMPLES} 列" if (
        len(names) > FINDING_EXAMPLES
    ) else ""
    findings = [finding(
        "warning", "treatment_imbalance",
        f"処置の群と対照の群で {shown}{more} の釣り合いが大きく崩れています "
        f"(標準化した平均の差の絶対値が {IMBALANCE} 以上). 無作為に配った"
        "記録に見えないので, 配り方を確かめてください (偏りのある記録の"
        "傾向の見積もりは今の約束の外です)", names + [treatment]
    )]

    return findings


def missing_findings(
    treatment_missing: int, outcome_missing: int, treatment: str,
    outcome: str
) -> list[dict]:
    """
    処置か結果の欠けた行を気付きにする関数

    Parameters
    ----------
    treatment_missing: int
        処置の欠けた行の数
    outcome_missing: int
        処置が分かり, 結果の欠けた行の数
    treatment: str
        処置の列の名前
    outcome: str
        結果の列の名前

    Returns
    ----------
    findings: list[dict]
        treatment_missing と outcome_missing の気付き
    """
    # 処置の欠けは書き出しで誤り, 結果の欠けは落とすか残す
    findings = []
    if treatment_missing:
        findings.append(finding(
            "warning", "treatment_missing",
            f"処置の列 {treatment} の欠けた行が {treatment_missing} 行あり"
            "ます. 書き出しでは誤りになるので, 前もって落としてください "
            "(filter_rows の compare: isna)", [treatment]
        ))
    if outcome_missing:
        findings.append(finding(
            "info", "outcome_missing",
            f"結果の列 {outcome} の欠けた行が {outcome_missing} 行あります. "
            "書き出しでは split.unlabeled で落とすか unlabeled の分割に"
            "残します", [outcome]
        ))

    return findings


def describe_uplift(
    source: Union[str, Path, Data], treatment: str, outcome: str,
    outcome_kind: Optional[str] = None, columns: Any = None,
    id: Optional[str] = None, loader: Optional[str] = None, **options: Any
) -> dict:
    """
    介入の記録の表の, 処置の群と対照の群の行の数, 結果の平均と差, 特徴の
    釣り合いと気付きを返す関数

    Parameters
    ----------
    source: Union[str, Path, Data]
        表のファイルか, load_data で読んだデータ
    treatment: str
        処置の列 (処置の群は 1 か真, 対照の群は 0 か偽)
    outcome: str
        結果の列
    outcome_kind: Optional[str] = None
        結果の種類 (binary か amount), None の場合は値から決める
    columns: Any = None
        釣り合いを求める列, None の場合は処置, 結果, 行の id の列を除く
        数, 真偽, カテゴリの列
    id: Optional[str] = None
        行の id の列 (釣り合いの列から除く)
    loader: Optional[str] = None
        読み込みの名前, None の場合は推定 (load_data と同じ)
    **options: Any
        読み込みの設定 (load_data と同じ)

    Returns
    ----------
    summary: dict
        JSON にできる介入の記録の要約 (groups, effect, balance, findings
        など)

    Raises
    ----------
    ValueError
        表でない場合, 列が無い場合, 処置と結果が同じ列の場合, 処置か結果の
        値が読み方に合わない場合
    """
    # 表と, 処置と結果の列
    data = uplift_table(source, loader, options)
    frame = data.frame.reset_index(drop=True).copy(deep=False)
    frame.columns = [str(name) for name in frame.columns]
    treat = column_of(frame, treatment, "処置の列")
    result = column_of(frame, outcome, "結果の列")
    if str(treatment) == str(outcome):
        raise ValueError("処置の列と結果の列は違う列にしてください")
    if id is not None:
        column_of(frame, id, "行の id の列")
    reserved = [str(treatment), str(outcome)] + (
        [str(id)] if id is not None else []
    )
    chosen = balance_columns(frame, columns, reserved, data.roles)

    # 欠けた行を除いた, 処置の 0 と 1 と結果の値
    treatment_missing = int(treat.isna().sum())
    outcome_missing = int((treat.notna() & result.isna()).sum())
    known = (treat.notna() & result.notna()).to_numpy()
    kind = choose_outcome_kind(result[known], str(outcome), outcome_kind)
    labels = treatment_values(treat[known], str(treatment))
    values = outcome_values(result[known], kind).astype(float)
    rows = frame[known].reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    values = values.reset_index(drop=True)
    treated = (labels == 1).to_numpy()

    # 群ごとの行の数, 結果の平均, 差
    groups = {}
    for name, flags in (("treated", treated), ("control", ~treated)):
        count = int(flags.sum())
        mean = float(values[flags].mean()) if count else None
        groups[name] = {"rows": count, "mean": mean}
    effect = (
        groups["treated"]["mean"] - groups["control"]["mean"]
        if groups["treated"]["rows"] and groups["control"]["rows"] else None
    )
    total = int(len(rows))

    # 釣り合い (SMD の絶対値の大きい順. None は最後)
    balance = [
        column_balance(rows[name], column_type, treated)
        for name, column_type in chosen
    ]
    balance.sort(key=lambda item: (
        item["smd"] is None, -abs(item["smd"] or 0.0)
    ))

    # 気付き (warning を先に)
    findings = group_findings(
        groups["treated"]["rows"], groups["control"]["rows"], str(treatment)
    )
    leaks = post_treatment_findings(rows, chosen, labels, values, kind)
    leaked = {item["columns"][0] for item in leaks}
    findings += leaks
    findings += imbalance_findings(balance, leaked, str(treatment))
    findings += missing_findings(
        treatment_missing, outcome_missing, str(treatment), str(outcome)
    )
    findings.sort(key=lambda item: item["severity"] != "warning")
    summary = to_json({
        "treatment": str(treatment),
        "outcome": str(outcome),
        "outcome_kind": kind,
        "rows": total,
        "missing": {
            "treatment": treatment_missing, "outcome": outcome_missing
        },
        "groups": groups,
        "treated_rate": groups["treated"]["rows"] / total if total else None,
        "effect": effect,
        "balance": balance,
        "findings": findings,
        "source": dict(data.source),
    })

    return summary
