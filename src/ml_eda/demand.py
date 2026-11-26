from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ml_eda.data import Data
from ml_eda.price import listed, numeric_values, product_values
from ml_eda.profile import finding, to_json
from ml_eda.uplift import column_of, uplift_table

# 需要の記録の表として読む種類
DEMAND_KINDS = ("table", "timeseries")

# 書き出しの分割に置く, 先の期間の売れた数の合計の列 (目的の値)
DEMAND = "demand"

# 書き出しの分割に置く, 前の期間の売れた数の合計の列 (比べる相手)
PREVIOUS = "demand_previous"

# 先の期間の日数の既定
HORIZON = 7

# 記録の短い商品と見なす日数 (先の期間の日数の倍)
SHORT_FACTOR = 4

# 要約に並べる商品の数の上限
PRODUCT_ROWS = 50

# 曜日の数 (0 が月曜)
WEEKDAYS = 7


def date_values(values: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """
    日付の列を日ごとの日時に読み, 読めない行と時刻のある行を見分ける関数

    Parameters
    ----------
    values: pd.Series
        日付の列の値

    Returns
    ----------
    days: pd.Series
        日ごとの日時 (読めない行と時刻のある行は NaT)
    timed: np.ndarray
        時刻 (0 時の他) のある行の印

    Raises
    ----------
    ValueError
        数か真偽値の列の場合 (日付として読まない)
    """
    # 数は日付として読まない (整数を日時に読むと 1970 年からの数になる)
    if pd.api.types.is_bool_dtype(values) or (
        pd.api.types.is_numeric_dtype(values)
    ):
        raise ValueError(
            f"日付の列 {values.name} が日付ではありません ({values.dtype})"
        )
    parsed = values
    if not pd.api.types.is_datetime64_any_dtype(values):
        # 書き方の混ざる値も 1 つずつ読む (時差の違う値は UTC にそろえる)
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        if not pd.api.types.is_datetime64_any_dtype(parsed):
            parsed = pd.to_datetime(
                values, errors="coerce", format="mixed", utc=True
            )
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    days = parsed.dt.normalize()
    timed = (parsed.notna() & (parsed != days)).to_numpy()
    days = days.where(~timed)

    return days, timed


def demand_dates(values: pd.Series, name: str) -> pd.Series:
    """
    日付の列の全ての行が日ごとの日付であることを確かめる関数

    Parameters
    ----------
    values: pd.Series
        日付の列の値 (全ての行)
    name: str
        日付の列の名前

    Returns
    ----------
    days: pd.Series
        日ごとの日時

    Raises
    ----------
    ValueError
        日付でない列, 欠けたか読めない行, 時刻のある行がある場合
    """
    # 欠けと読めない値, 時刻のある値
    days, timed = date_values(values.rename(name))
    unread = int((days.isna().to_numpy() & ~timed).sum())
    if unread:
        raise ValueError(
            f"日付の列 {name} に欠けたか読めない行があります ({unread} 行). "
            "読み込む前の表で直してください"
        )
    if timed.any():
        examples = values[timed].astype(str).unique().tolist()[:3]
        raise ValueError(
            f"日付の列 {name} に時刻のある行があります ({int(timed.sum())} "
            f"行. 例: {', '.join(examples)}). 日ごとの日付にしてください"
        )

    return days


def demand_units(values: pd.Series, name: str) -> pd.Series:
    """
    日ごとの売れた数の列を確かめ, float64 にする関数 (欠けは抜けの日)

    Parameters
    ----------
    values: pd.Series
        売れた数の列の値 (全ての行)
    name: str
        売れた数の列の名前

    Returns
    ----------
    units: pd.Series
        float64 の売れた数 (欠けは nan)

    Raises
    ----------
    ValueError
        数でない場合と, 負か無限大の値がある場合
    """
    # 負か無限大の値 (欠けは記録の無い日と同じに扱う)
    units = numeric_values(values, name, "売れた数の列")
    array = units.to_numpy()
    wrong = ~np.isnan(array) & (~np.isfinite(array) | (array < 0))
    if wrong.any():
        examples = values[wrong].astype(str).unique().tolist()[:3]
        raise ValueError(
            f"売れた数の列 {name} に負か無限大の値があります "
            f"({int(wrong.sum())} 行. 例: {', '.join(examples)}). 読み込む"
            "前の表で直してください"
        )

    return units


def stock_values(values: pd.Series, name: str) -> pd.Series:
    """
    在庫の数の列を確かめ, float64 にする関数

    Parameters
    ----------
    values: pd.Series
        在庫の数の列の値 (全ての行)
    name: str
        在庫の数の列の名前

    Returns
    ----------
    stock: pd.Series
        float64 の在庫の数

    Raises
    ----------
    ValueError
        数でない場合, 欠けがある場合, 負か無限大の値がある場合
    """
    # 欠けと, 負か無限大の値
    stock = numeric_values(values, name, "在庫の数の列")
    missing = int(stock.isna().sum())
    if missing:
        raise ValueError(
            f"在庫の数の列 {name} に欠けがあります ({missing} 行). 読み込む"
            "前の表で埋めるか落としてください"
        )
    array = stock.to_numpy()
    wrong = ~np.isfinite(array) | (array < 0)
    if wrong.any():
        examples = stock[wrong].astype(str).unique().tolist()[:3]
        raise ValueError(
            f"在庫の数の列 {name} に負か無限大の値があります "
            f"({int(wrong.sum())} 行. 例: {', '.join(examples)}). 読み込む"
            "前の表で直してください"
        )

    return stock


def demand_periods(recipe: dict) -> tuple[int, int]:
    """
    レシピの先の期間の日数と, 発注を決める日の間隔を返す関数

    Parameters
    ----------
    recipe: dict
        読み込んだレシピ (horizon, interval)

    Returns
    ----------
    horizon: int
        先の期間の日数 (既定 7)
    interval: int
        発注を決める日の間隔の日数 (既定は先の期間の日数)
    """
    # 書かなければ既定
    horizon = int(recipe.get("horizon", HORIZON))
    interval = int(recipe.get("interval", horizon))

    return horizon, interval


def window_sums(
    values: np.ndarray, start: np.ndarray, length: int
) -> np.ndarray:
    """
    日の並びの, 位置 start から length 日の合計を求める関数 (1 日でも
    値が無ければ nan)

    Parameters
    ----------
    values: np.ndarray
        1 つの商品の, 最初の日から毎日の値 (記録の無い日は nan)
    start: np.ndarray
        合計を始める位置 (範囲の外は nan)
    length: int
        日数

    Returns
    ----------
    sums: np.ndarray
        位置ごとの合計
    """
    # 累積の和と, 値のある日の累積の数
    known = ~np.isnan(values)
    total = np.concatenate([[0.0], np.cumsum(np.where(known, values, 0.0))])
    count = np.concatenate([[0], np.cumsum(known)])
    inside = (start >= 0) & (start + length <= len(values))
    begin = np.where(inside, start, 0)
    end = begin + length
    full = inside & (count[np.minimum(end, len(values))] - count[begin] == (
        length
    ))
    sums = np.where(
        full, total[np.minimum(end, len(values))] - total[begin], np.nan
    )

    return sums


def demand_windows(
    products: pd.Series, days: pd.Series, units: pd.Series,
    stock: Optional[pd.Series], horizon: int, interval: int
) -> pd.DataFrame:
    """
    行ごとの先の期間と前の期間の売れた数の合計, 発注を決める日の印,
    先の期間の在庫が 0 の日の数を求める関数

    Parameters
    ----------
    products: pd.Series
        行ごとの商品の id (欠けの無いもの)
    days: pd.Series
        行ごとの日ごとの日時 (欠けの無いもの)
    units: pd.Series
        行ごとの売れた数 (欠けは nan)
    stock: Optional[pd.Series]
        行ごとの在庫の数 (無ければ None)
    horizon: int
        先の期間の日数
    interval: int
        発注を決める日の間隔の日数

    Returns
    ----------
    windows: pd.DataFrame
        行の位置の順の demand (その日からの先の期間の合計), previous (その
        日の前の期間の合計), scheduled (発注を決める日), decision (発注を
        決める日で, 前の期間が揃う行), stockout (先の期間の記録にある在庫が
        0 の日の数. 在庫の数が無ければ nan), day (日)
    """
    # 全ての商品の最初の日から interval 日ごとが発注を決める日
    count = len(days)
    day_values = days.to_numpy(dtype="datetime64[D]")
    offsets = (day_values - day_values.min()).astype(np.int64) if count else (
        np.zeros(0, dtype=np.int64)
    )
    demand = np.full(count, np.nan)
    previous = np.full(count, np.nan)
    stockout = np.full(count, np.nan)
    keys = products.astype(str).to_numpy()
    unit_values = units.to_numpy(dtype=float)
    stocks = None if stock is None else stock.to_numpy(dtype=float)

    # 商品ごとに, 最初の日から毎日の並びにした売れた数と在庫が 0 の日
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        first = offsets[rows].min()
        position = offsets[rows] - first
        length = int(position.max()) + 1
        daily = np.full(length, np.nan)
        daily[position] = unit_values[rows]
        demand[rows] = window_sums(daily, position, horizon)
        previous[rows] = window_sums(daily, position - horizon, horizon)
        if stocks is not None:
            empty = np.zeros(length)
            empty[position] = stocks[rows] == 0
            empty = np.concatenate([empty, np.zeros(horizon)])
            total = np.concatenate([[0.0], np.cumsum(empty)])
            stockout[rows] = total[position + horizon] - total[position]
    scheduled = offsets % interval == 0
    windows = pd.DataFrame({
        "demand": demand, "previous": previous,
        "scheduled": scheduled, "decision": scheduled & ~np.isnan(previous),
        "stockout": stockout, "day": days.to_numpy(),
    })

    return windows


def duplicate_days(products: pd.Series, days: pd.Series) -> np.ndarray:
    """
    同じ商品の同じ日の, 2 つ目より後の行の印を返す関数

    Parameters
    ----------
    products: pd.Series
        行ごとの商品の id
    days: pd.Series
        行ごとの日

    Returns
    ----------
    twins: np.ndarray
        2 つ目より後の行の印
    """
    # 商品の id の文字と日の組の重なり
    pairs = pd.DataFrame({
        "product": products.astype(str).to_numpy(),
        "day": days.to_numpy(),
    })
    twins = pairs.duplicated(keep="first").to_numpy()

    return twins


def product_rows(
    products: np.ndarray, days: np.ndarray, units: np.ndarray,
    empty: Optional[np.ndarray], horizon: int
) -> dict:
    """
    商品ごとの記録の日数, 最初と最後の日, 抜けの日, 在庫が 0 の日と
    売れた数の平均を求める関数

    Parameters
    ----------
    products: np.ndarray
        使う行の商品の id
    days: np.ndarray
        使う行の日 (datetime64)
    units: np.ndarray
        使う行の売れた数
    empty: Optional[np.ndarray]
        使う行の在庫が 0 の印 (在庫の数が無ければ None)
    horizon: int
        先の期間の日数 (記録の短い商品の目安)

    Returns
    ----------
    summary: dict
        count, short, gaps (抜けの日のある商品の数), rows (記録の日数の
        少ない順で最大 50 件), shorts と gapped (商品の文字の並び),
        missing_days (抜けの日の数の合計)
    """
    # 商品ごとの集計 (商品の id は文字. 同じ日の行は 1 日に数える)
    frame = pd.DataFrame({
        "product": pd.Series(products).astype(str),
        "day": pd.Series(days, dtype="datetime64[ns]"),
        "units": units,
        "empty": empty if empty is not None else np.zeros(len(units)),
    })
    grouped = frame.groupby("product", sort=True)
    first = grouped["day"].min()
    last = grouped["day"].max()
    recorded = grouped["day"].nunique()
    span = (last - first).dt.days + 1
    table = pd.DataFrame({
        "days": recorded,
        "first": first.dt.strftime("%Y-%m-%d"),
        "last": last.dt.strftime("%Y-%m-%d"),
        "gaps": span - recorded,
        "stockout": grouped["empty"].sum().astype(int)
        if empty is not None else None,
        "units_mean": grouped["units"].mean(),
    }).reset_index()
    table = table.sort_values(["days", "product"], kind="stable")

    # 記録の短い商品と抜けの日のある商品
    shorts = table.loc[
        table["days"] < SHORT_FACTOR * horizon, "product"
    ].tolist()
    gapped = table.loc[table["gaps"] > 0, "product"].tolist()
    summary = {
        "count": int(len(table)),
        "short": int(len(shorts)),
        "gaps": int(len(gapped)),
        "rows": table.head(PRODUCT_ROWS).to_dict(orient="records"),
        "shorts": shorts,
        "gapped": gapped,
        "missing_days": int(table["gaps"].sum()),
    }

    return summary


def weekday_rows(days: np.ndarray, units: np.ndarray) -> list[dict]:
    """
    曜日ごとの行の数と売れた数の平均を求める関数

    Parameters
    ----------
    days: np.ndarray
        使う行の日 (datetime64)
    units: np.ndarray
        使う行の売れた数

    Returns
    ----------
    rows: list[dict]
        曜日 (0 が月曜) ごとの weekday, rows, units_mean (行の無い曜日は
        units_mean が None)
    """
    # 曜日ごとの集計
    weekdays = pd.Series(
        days, dtype="datetime64[ns]"
    ).dt.day_of_week.to_numpy()
    rows = []
    for weekday in range(WEEKDAYS):
        flags = weekdays == weekday
        rows.append({
            "weekday": weekday,
            "rows": int(flags.sum()),
            "units_mean": float(units[flags].mean()) if flags.any() else None,
        })

    return rows


def value_findings(
    missing: dict, invalid: dict, duplicates: int, names: dict
) -> list[dict]:
    """
    商品の id, 日付, 売れた数, 在庫の数の欠けと値の誤り, 同じ日の行を
    気付きにする関数

    Parameters
    ----------
    missing: dict
        欠けた行の数 (product, date, units, stock)
    invalid: dict
        値の誤りの行の数 (date, units, stock)
    duplicates: int
        同じ商品の同じ日の 2 つ目より後の行の数
    names: dict
        列の名前 (product, date, units, stock)

    Returns
    ----------
    findings: list[dict]
        product_missing, date_missing, duplicate_days, units_invalid,
        stock_invalid, units_missing の気付き
    """
    # 書き出しでは誤りになるもの
    fix = "書き出しでは誤りになるので, 読み込む前の表で直してください"
    findings = []
    checks = [
        ("product_missing", missing["product"], "product", "商品の id の列",
         "欠けた行"),
        ("date_missing", missing["date"] + invalid["date"], "date",
         "日付の列", "欠けたか, 読めないか, 時刻のある行"),
        ("duplicate_days", duplicates, "date", "日付の列",
         "同じ商品の同じ日の行 (2 つ目より後)"),
        ("units_invalid", invalid["units"], "units", "売れた数の列",
         "負か無限大の行"),
        ("stock_invalid", (missing["stock"] or 0) + (invalid["stock"] or 0),
         "stock", "在庫の数の列", "欠けたか, 負か無限大の行"),
    ]
    for code, count, key, label, kind in checks:
        if count:
            findings.append(finding(
                "warning", code,
                f"{label} {names[key]} の{kind}が {count} 行あります. {fix}",
                [names[key]]
            ))

    # 売れた数の欠け (記録の無い日と同じ扱い)
    if missing["units"]:
        findings.append(finding(
            "info", "units_missing",
            f"売れた数の列 {names['units']} の欠けた行が {missing['units']} "
            "行あります. 書き出しでは記録の無い日 (抜けの日) と同じく, その"
            "日を含む期間の合計を作りません", [names["units"]]
        ))

    return findings


def history_findings(
    products: dict, stockout: Optional[dict], horizon: int, names: dict
) -> list[dict]:
    """
    日の抜けのある商品, 記録の短い商品, 在庫が 0 の日を気付きにする関数

    Parameters
    ----------
    products: dict
        商品ごとの要約 (product_rows)
    stockout: Optional[dict]
        在庫が 0 の日の要約 (在庫の数の列が無ければ None)
    horizon: int
        先の期間の日数
    names: dict
        列の名前 (product, date, units, stock)

    Returns
    ----------
    findings: list[dict]
        date_gaps, short_history, stockout_days の気付き
    """
    # 日の抜け
    findings = []
    if products["gapped"]:
        findings.append(finding(
            "warning", "date_gaps",
            f"記録の無い日 (抜けの日) のある商品が {products['gaps']} 件"
            f"あります ({listed(products['gapped'])}. 抜けの日は計 "
            f"{products['missing_days']} 日). 抜けの日を含む先の期間と前の"
            "期間は合計を作れず, 書き出しの行が減ります. 売れなかった日なら"
            ", 読み込む前の表に売れた数 0 の行を足してください",
            [names["product"], names["date"]]
        ))

    # 記録の短い商品
    if products["shorts"]:
        findings.append(finding(
            "warning", "short_history",
            f"記録の日数が {SHORT_FACTOR * horizon} 日 (先の期間の "
            f"{SHORT_FACTOR} 倍) 未満の商品が {products['short']} 件あります"
            f" ({listed(products['shorts'])}). 前の期間と先の期間で "
            f"{2 * horizon} 日を使うので, 発注を決める日の行がわずかか, "
            "作れません", [names["product"]]
        ))

    # 在庫が 0 の日 (売れた数が需要より少なく記録される)
    if stockout is not None and stockout["rows"]:
        findings.append(finding(
            "warning", "stockout_days",
            f"在庫が 0 の日が {stockout['rows']} 行 ({stockout['products']} "
            "件の商品) あります. その日の売れた数は需要より少なく記録され"
            "るので, 学んだ需要が少なめになります (書き出しでは, 先の期間に"
            "在庫が 0 の日を含む行の数を説明の demand に残します)",
            [names["stock"]]
        ))

    return findings


def describe_demand(
    source: Union[str, Path, Data], product: str, date: str, units: str,
    stock: Optional[str] = None, horizon: int = HORIZON,
    loader: Optional[str] = None, **options: Any
) -> dict:
    """
    需要の記録の表の日付の範囲, 売れた数, 曜日ごとの売れた数, 商品ごとの
    記録の日数と抜けの日, 在庫が 0 の日と気付きを返す関数

    Parameters
    ----------
    source: Union[str, Path, Data]
        表のファイルか, load_data で読んだデータ
    product: str
        商品の id の列
    date: str
        日付の列
    units: str
        日ごとの売れた数の列
    stock: Optional[str] = None
        在庫の数の列
    horizon: int = 7
        先の期間の日数 (記録の短い商品の目安)
    loader: Optional[str] = None
        読み込みの名前, None の場合は推定 (load_data と同じ)
    **options: Any
        読み込みの設定 (load_data と同じ)

    Returns
    ----------
    summary: dict
        JSON にできる需要の記録の要約 (dates, unit_stats, weekdays,
        products, stockout, findings など)

    Raises
    ----------
    ValueError
        表でない場合, 列が無い場合, 役割の列が重なる場合, 売れた数と在庫の
        数の列が数でない場合, 日付の列が数の場合, horizon が 1 以上の整数
        でない場合
    """
    # 先の期間の日数
    if isinstance(horizon, bool) or not isinstance(horizon, int) or (
        horizon < 1
    ):
        raise ValueError(f"horizon は 1 以上の整数にしてください: {horizon}")

    # 表と役割の列
    data = uplift_table(source, loader, options, "需要の記録")
    frame = data.frame.reset_index(drop=True).copy(deep=False)
    frame.columns = [str(name) for name in frame.columns]
    names = {"product": product, "date": date, "units": units, "stock": stock}
    given = {key: str(name) for key, name in names.items() if name is not None}
    if len(set(given.values())) != len(given):
        raise ValueError(
            "商品の id, 日付, 売れた数, 在庫の数の列は違う列にしてください"
        )
    labels = {"product": "商品の id の列", "date": "日付の列",
              "units": "売れた数の列", "stock": "在庫の数の列"}
    picked = {
        key: column_of(frame, name, labels[key])
        for key, name in given.items()
    }
    days, timed = date_values(picked["date"])
    units_all = numeric_values(picked["units"], given["units"], "売れた数の列")
    stock_all = numeric_values(
        picked["stock"], given["stock"], "在庫の数の列"
    ) if stock is not None else None

    # 欠けと値の誤りの行の数と除外 (在庫の数は数えるだけ)
    product_known = picked["product"].notna().to_numpy()
    date_known = picked["date"].notna().to_numpy()
    date_ok = days.notna().to_numpy()
    units_array = units_all.to_numpy()
    units_known = ~np.isnan(units_array)
    units_ok = units_known & np.isfinite(units_array) & (units_array >= 0)
    stock_array = stock_all.to_numpy() if stock_all is not None else None
    stock_ok = (
        ~np.isnan(stock_array) & np.isfinite(stock_array) & (stock_array >= 0)
    ) if stock_array is not None else None
    missing = {
        "product": int((~product_known).sum()),
        "date": int((~date_known).sum()),
        "units": int((product_known & date_ok & ~units_known).sum()),
        "stock": int(np.isnan(stock_array).sum())
        if stock_array is not None else None,
    }
    invalid = {
        "date": int((date_known & ~date_ok).sum()),
        "units": int((units_known & ~units_ok).sum()),
        "stock": int((~np.isnan(stock_array) & ~stock_ok).sum())
        if stock_array is not None else None,
    }
    placed = product_known & date_ok
    twins = np.zeros(len(frame), dtype=bool)
    twins[placed] = duplicate_days(
        picked["product"][placed], days[placed]
    )
    used = placed & units_ok
    keys = picked["product"].to_numpy()[used]
    used_days = days.to_numpy()[used]
    sold = units_array[used]
    empty = (
        stock_ok[used] & (stock_array[used] == 0)
        if stock_array is not None else None
    )

    # 日付の範囲, 売れた数, 曜日ごとと商品ごとの要約
    rows = int(used.sum())
    dates = {
        "min": pd.Timestamp(used_days.min()).strftime("%Y-%m-%d")
        if rows else None,
        "max": pd.Timestamp(used_days.max()).strftime("%Y-%m-%d")
        if rows else None,
        "days": int(
            (used_days.max() - used_days.min()) // np.timedelta64(1, "D")
        ) + 1 if rows else 0,
    }
    unit_stats = {
        "min": float(sold.min()) if rows else None,
        "max": float(sold.max()) if rows else None,
        "mean": float(sold.mean()) if rows else None,
        "zero_rate": float((sold == 0).mean()) if rows else None,
    }
    products = product_rows(keys, used_days, sold, empty, horizon)
    stockout = {
        "rows": int(empty.sum()),
        "products": int(len(set(pd.Series(keys[empty]).astype(str)))),
        "rate": float(empty.mean()) if rows else None,
    } if empty is not None else None

    # 気付き (warning を先に)
    findings = value_findings(
        missing, invalid, int(twins.sum()),
        given | {"stock": given.get("stock")}
    )
    findings += history_findings(
        products, stockout, horizon, given | {"stock": given.get("stock")}
    )
    findings.sort(key=lambda item: item["severity"] != "warning")
    product_view = {
        key: value for key, value in products.items()
        if key not in ("shorts", "gapped")
    }
    summary = to_json({
        "product": given["product"],
        "date": given["date"],
        "units": given["units"],
        "stock": given.get("stock"),
        "horizon": horizon,
        "rows": rows,
        "missing": missing,
        "invalid": invalid,
        "duplicates": int(twins.sum()),
        "dates": dates,
        "unit_stats": unit_stats,
        "weekdays": weekday_rows(used_days, sold),
        "products": product_view,
        "stockout": stockout,
        "findings": findings,
        "source": dict(data.source),
    })

    return summary


def check_demand_values(
    frame: pd.DataFrame, names: dict, horizon: int, interval: int
) -> pd.DataFrame:
    """
    需要の記録の表の役割の列を全ての行で確かめ, 行ごとの期間の合計を
    求める関数

    Parameters
    ----------
    frame: pd.DataFrame
        読み込んだ表 (列の名前は文字)
    names: dict
        列の名前 (target, product, date と, 書いた場合の stock)
    horizon: int
        先の期間の日数
    interval: int
        発注を決める日の間隔の日数

    Returns
    ----------
    windows: pd.DataFrame
        行の位置の順の demand, previous, scheduled, decision, stockout,
        day (demand_windows)

    Raises
    ----------
    ValueError
        値が決まりに合わない場合と, 同じ商品の同じ日の行がある場合
    """
    # 商品の id, 日付, 売れた数, 在庫の数
    products = product_values(
        column_of(frame, names["product"], "商品の id の列"),
        names["product"]
    )
    days = demand_dates(
        column_of(frame, names["date"], "日付の列"), names["date"]
    )
    units = demand_units(
        column_of(frame, names["target"], "売れた数の列"), names["target"]
    )
    stock = stock_values(
        column_of(frame, names["stock"], "在庫の数の列"), names["stock"]
    ) if names.get("stock") else None

    # 同じ商品の同じ日は, 合計に入れる行を決められない
    twins = duplicate_days(products, days)
    if twins.any():
        examples = [
            f"{key} {day:%Y-%m-%d}" for key, day in zip(
                products[twins].astype(str).tolist()[:3],
                days[twins].tolist()[:3]
            )
        ]
        raise ValueError(
            f"同じ商品の同じ日の行があります ({int(twins.sum())} 行. 例: "
            f"{', '.join(examples)}). 読み込む前の表で 1 日 1 行に"
            "まとめてください"
        )
    windows = demand_windows(
        products.reset_index(drop=True), days.reset_index(drop=True),
        units.reset_index(drop=True),
        None if stock is None else stock.reset_index(drop=True),
        horizon, interval
    )

    return windows
