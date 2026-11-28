import json
from pathlib import Path

import pandas as pd
import pytest

import ml_eda
from ml_eda import Reactions, describe_reactions, load_data, load_reactions


def write_csv(path: Path, lines: list[str]) -> Path:
    """
    反応の表の CSV を書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    lines: list[str]
        見出しと行

    Returns
    ----------
    path: Path
        書いたファイル
    """
    # 改行でつないで書く
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return path


def test_public_names() -> None:
    """
    反応の表の関数が公開の名前にあるかを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ml_eda から読める
    for name in ("Reactions", "load_reactions", "describe_reactions"):
        assert name in ml_eda.__all__


def test_drops_like_the_skeleton(tmp_path: Path) -> None:
    """
    欠けた行と, 強さが 0 以下か数でない行を除いて数えるかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 欠け 2 行 (品物と時刻), 強さが 0, 負, 数でない行が 1 つずつ
    path = write_csv(tmp_path / "reactions.csv", [
        "member,item_id,ts,qty,note",
        "a,x,3,1,p",
        "a,y,1,2,q",
        "b,x,2,1,r",
        "b,,4,1,s",
        "c,z,,1,t",
        "c,y,5,0,u",
        "c,x,6,-1,v",
        "c,z,7,many,w",
        "a,x,8,1.5,x",
    ])
    reactions = load_reactions(
        path, user="member", item="item_id", time="ts", reaction="qty"
    )

    # 除いた数, 時刻の順, 残す列
    assert isinstance(reactions, Reactions)
    assert reactions.rows == 9
    assert reactions.dropped_missing == 2
    assert reactions.dropped_not_positive == 3
    assert reactions.time_kind == "number"
    assert list(reactions.frame.columns) == ["user", "item", "time",
                                             "reaction"]
    assert reactions.frame["time"].tolist() == [1, 2, 3, 8]
    assert reactions.frame["reaction"].tolist() == [2.0, 1.0, 1.0, 1.5]
    assert reactions.columns == {
        "user": "member", "item": "item_id", "time": "ts",
        "reaction": "qty", "shown": None, "presentation": None
    }
    assert reactions.shown is None
    assert reactions.source["loader"] == "csv"
    assert len(reactions.notes) == 2


def test_without_strength_keeps_all_rows(tmp_path: Path) -> None:
    """
    反応の強さを渡さなければ全ての行を反応とみなすかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 強さの列は読まない
    path = write_csv(tmp_path / "r.csv", [
        "u,i,t,qty", "a,x,1,0", "b,y,2,-3",
    ])
    reactions = load_reactions(path, user="u", item="i", time="t")
    summary = describe_reactions(reactions)

    # 2 行とも残り, 強さの要約は無い
    assert len(reactions.frame) == 2
    assert list(reactions.frame.columns) == ["user", "item", "time"]
    assert summary["columns"]["reaction"] is None
    assert summary["reaction"] is None
    assert summary["dropped"]["total"] == 0


def test_datetime_time(tmp_path: Path) -> None:
    """
    日時の時刻を読み, 読めない値を欠けとして除くかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 読めない日時が 1 つ
    path = write_csv(tmp_path / "r.csv", [
        "u,i,t",
        "a,x,2026-01-03",
        "a,y,2026-01-01",
        "b,x,someday",
    ])
    reactions = load_reactions(path, user="u", item="i", time="t")
    summary = describe_reactions(reactions)

    # 日時の範囲と, 読めない値は欠けとして数える
    assert reactions.time_kind == "datetime"
    assert reactions.dropped_missing == 1
    assert reactions.unreadable_time == 1
    assert summary["time"] == {
        "kind": "datetime", "min": "2026-01-01T00:00:00",
        "max": "2026-01-03T00:00:00", "distinct": 2
    }
    assert summary["dropped"]["unreadable_time"] == 1


def test_errors(tmp_path: Path) -> None:
    """
    列の名前の誤りと, 読めない時刻で止まるかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 無い列, 同じ列を 2 つの役割, 数でも日時でもない時刻
    path = write_csv(tmp_path / "r.csv", ["u,i,t", "a,x,soon", "b,y,later"])
    with pytest.raises(ValueError, match="qty"):
        load_reactions(path, user="u", item="i", time="t", reaction="qty")
    with pytest.raises(ValueError, match="別々の列"):
        load_reactions(path, user="u", item="u", time="t")
    with pytest.raises(ValueError, match="time"):
        load_reactions(path, user="u", item="i", time=None)
    with pytest.raises(ValueError, match="数か日時"):
        load_reactions(path, user="u", item="i", time="t")


def test_reads_loaded_data(tmp_path: Path) -> None:
    """
    load_data で読んだデータを渡せるかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 読んだデータの表と読み方を使う
    path = write_csv(tmp_path / "r.csv", ["u,i,t", "a,x,1", "b,y,2"])
    data = load_data(path)
    reactions = load_reactions(data, user="u", item="i", time="t")
    assert len(reactions.frame) == 2
    assert reactions.source == data.source

    # 読んだデータに読み方は渡さない
    with pytest.raises(ValueError, match="読み込み"):
        load_reactions(data, user="u", item="i", time="t", loader="csv")


def test_summary(tmp_path: Path) -> None:
    """
    利用者, 品物, 反応の数, 分布, 人気の偏りの要約を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 利用者 a は 4 回 (x を 2 回), b は 2 回, c は 1 回
    path = write_csv(tmp_path / "r.csv", [
        "u,i,t,w",
        "a,x,1,1", "a,x,2,1", "a,y,3,2", "a,z,4,1",
        "b,x,5,1", "b,y,6,3",
        "c,x,7,1",
        "c,y,8,0",
    ])
    reactions = load_reactions(path, user="u", item="i", time="t",
                               reaction="w")
    summary = describe_reactions(reactions)

    # JSON にできる
    json.dumps(summary)

    # 数
    assert summary["rows"] == 8
    assert summary["reactions"] == 7
    assert summary["users"] == 3
    assert summary["items"] == 3
    assert summary["pairs"] == 6
    assert summary["repeated"] == 1
    assert summary["density"] == pytest.approx(6 / 9)
    assert summary["dropped"] == {
        "missing": 0, "unreadable_time": 0, "not_positive": 1,
        "unreadable_shown": 0, "conflict": 0, "duplicate_shown": 0,
        "total": 1
    }

    # 利用者ごとの数の分布 (4, 2, 1)
    per_user = summary["per_user"]
    assert per_user["min"] == 1 and per_user["max"] == 4
    assert per_user["median"] == 2
    assert per_user["once"] == 1
    bins = {(row["from"], row["to"]): row["count"] for row in per_user["bins"]}
    assert bins[(1, 1)] == 1 and bins[(2, 2)] == 1 and bins[(3, 5)] == 1
    assert bins[(101, None)] == 0

    # 品物の人気 (x 4, y 2, z 1)
    popularity = summary["popularity"]
    assert popularity["top_items"] == 1
    assert popularity["top_share"] == pytest.approx(4 / 7)
    assert popularity["popular"][0] == {"item": "x", "reactions": 4}
    assert 0 < popularity["gini"] < 1

    # 時刻と強さ
    assert summary["time"]["kind"] == "number"
    assert summary["time"]["min"] == 1 and summary["time"]["max"] == 7
    assert summary["reaction"]["max"] == 3
    assert summary["source"]["loader"] == "csv"


def test_empty_and_single(tmp_path: Path) -> None:
    """
    反応が残らない表と, 品物が 1 つの表の要約を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 全ての行が 0 の強さ
    path = write_csv(tmp_path / "r.csv", ["u,i,t,w", "a,x,1,0", "b,x,2,0"])
    summary = describe_reactions(
        load_reactions(path, user="u", item="i", time="t", reaction="w")
    )
    json.dumps(summary)
    assert summary["reactions"] == 0
    assert summary["density"] is None
    assert summary["popularity"]["top_share"] is None
    assert summary["time"]["min"] is None
    assert any("残る行" in note for note in summary["notes"])

    # 品物が 1 つ
    path = write_csv(tmp_path / "s.csv", ["u,i,t", "a,x,1", "b,x,2"])
    summary = describe_reactions(
        load_reactions(path, user="u", item="i", time="t")
    )
    assert any("並べて確かめられません" in note for note in summary["notes"])


def test_number_ids_stay_numbers(tmp_path: Path) -> None:
    """
    数の利用者と品物の値が, 人気の品物の並びで数のままかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # Parquet でなく表から直に (読んだデータの形)
    frame = pd.DataFrame({"u": [1, 2, 2], "i": [10, 10, 11],
                          "t": [1.0, 2.0, 3.0]})
    data = ml_eda.Data(kind="table", frame=frame)
    summary = describe_reactions(
        load_reactions(data, user="u", item="i", time="t")
    )
    json.dumps(summary)
    assert summary["popularity"]["popular"][0] == {"item": 10, "reactions": 2}



def test_shown_rows(tmp_path: Path) -> None:
    """
    見せただけの印の列がある表で, 反応と見せただけの記録に分けるかを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 行 = 見せた品物 1 件. 欠けた印は偽, 真の行は強さを読まない
    path = write_csv(tmp_path / "r.csv", [
        "u,i,t,qty,seen",
        "a,x,2,1,",
        "a,y,2,,1",
        "a,z,2,,TRUE",
        "b,x,1,2,false",
        "b,z,1,0,0",
        "c,w,3,,1",
    ])
    reactions = load_reactions(
        path, "u", "i", "t", "qty", shown="seen"
    )
    assert reactions.columns["shown"] == "seen"
    assert reactions.frame["item"].tolist() == ["x", "x"]
    assert reactions.frame["user"].tolist() == ["b", "a"]
    assert list(reactions.shown.columns) == ["user", "item", "time"]
    assert reactions.shown["item"].tolist() == ["y", "z", "w"]
    assert reactions.dropped_not_positive == 1
    assert reactions.dropped_shown == 0

    # 提示は利用者と時刻の組. 反応の表に無い品物は数える
    summary = describe_reactions(reactions)
    json.dumps(summary)
    assert summary["columns"]["shown"] == "seen"
    assert summary["shown"] == {
        "negatives": 3, "users": 2, "items": 3, "by": "time",
        "presentations": 3, "chosen_presentations": 2, "unseen_items": 3,
    }
    assert (summary["rows"] == summary["reactions"]
            + summary["shown"]["negatives"] + summary["dropped"]["total"])
    assert any("反応の表に無い品物" in note for note in summary["notes"])


def test_without_shown_column(tmp_path: Path) -> None:
    """
    印を渡さない読み方が, 選んだ記録だけの今の読み方と同じかを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 印の列の無い表と, 印の列があっても渡さない場合
    plain = write_csv(tmp_path / "plain.csv", ["u,i,t", "a,x,1", "b,y,2"])
    marked = write_csv(
        tmp_path / "marked.csv", ["u,i,t,seen", "a,x,1,0", "b,y,2,1"]
    )
    for path in (plain, marked):
        reactions = load_reactions(path, "u", "i", "t")
        assert reactions.shown is None
        assert reactions.frame["item"].tolist() == ["x", "y"]
        summary = describe_reactions(reactions)
        assert summary["shown"] is None
        assert summary["columns"]["shown"] is None
        assert summary["columns"]["presentation"] is None
        assert summary["dropped"]["total"] == 0

    # 5 つ目の位置の引数は今までどおり読み込みの名前
    reactions = load_reactions(plain, "u", "i", "t", None, "csv")
    assert reactions.source["loader"] == "csv"


def test_shown_broken_values(tmp_path: Path) -> None:
    """
    見せただけの印の壊れた値を除くか, 誤りで止まるかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 読めない印の行は除き, 欠けた印は偽
    path = write_csv(tmp_path / "r.csv", [
        "u,i,t,seen", "a,x,1,0", "a,y,1,yes", "a,z,1,", "a,w,1,2",
        "b,y,2, 1 ",
    ])
    reactions = load_reactions(path, "u", "i", "t", shown="seen")
    assert reactions.dropped_shown == 2
    assert reactions.frame["item"].tolist() == ["x", "z"]
    assert reactions.shown["item"].tolist() == ["y"]
    summary = describe_reactions(reactions)
    assert summary["dropped"]["unreadable_shown"] == 2
    assert summary["dropped"]["total"] == 2
    assert any("印 (seen) を読めない 2 行" in note
               for note in summary["notes"])

    # どの値も読めない印, 全て欠けた印, 選んだ反応が残らない表は誤り
    cases = {
        "unreadable.csv": (["a,x,1,yes", "b,y,2,no"], "読めません"),
        "absent.csv": (["a,x,1,", "b,y,2,"], "全て欠けて"),
        "only_shown.csv": (["a,x,1,1", "b,y,2,true"], "選んだ反応"),
    }
    for name, (lines, message) in cases.items():
        broken = write_csv(tmp_path / name, ["u,i,t,seen", *lines])
        with pytest.raises(ValueError, match=message):
            load_reactions(broken, "u", "i", "t", shown="seen")

    # 欠けた行を除いた後で判定 (印の残る行は欠けた利用者の行だけ)
    late = write_csv(
        tmp_path / "late.csv", ["u,i,t,seen", ",x,1,1", "a,y,2,"]
    )
    with pytest.raises(ValueError, match="全て欠けて"):
        load_reactions(late, "u", "i", "t", shown="seen")

    # 無い列, 他の役割と同じ列, 印無しの提示は誤り
    with pytest.raises(ValueError, match="flag"):
        load_reactions(path, "u", "i", "t", shown="flag")
    with pytest.raises(ValueError, match="別々の列"):
        load_reactions(path, "u", "i", "t", shown="t")
    with pytest.raises(ValueError, match="一緒に"):
        load_reactions(path, "u", "i", "t", presentation="seen")


def test_shown_conflict_and_duplicate() -> None:
    """
    選んだ行と重なる見せただけの行と, 同じ見せただけの行を除くかを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 強さで外れた選んだ行 (11) と重なる行も負の例にしない
    frame = pd.DataFrame({
        "u": [1, 1, 1, 1, 1, 1], "i": [10, 11, 10, 11, 12, 12],
        "t": [1.0] * 6, "qty": [1, 0, None, None, None, None],
        "seen": [False, False, True, True, True, True],
    })
    data = ml_eda.Data(kind="table", frame=frame)
    reactions = load_reactions(data, "u", "i", "t", "qty", shown="seen")
    assert reactions.dropped_not_positive == 1
    assert reactions.dropped_conflict == 2
    assert reactions.dropped_duplicate == 1
    assert reactions.shown["item"].tolist() == [12]
    summary = describe_reactions(reactions)
    assert summary["dropped"]["conflict"] == 2
    assert summary["dropped"]["duplicate_shown"] == 1
    assert summary["rows"] == 1 + 1 + summary["dropped"]["total"]


def test_shown_datetime_conflict(tmp_path: Path) -> None:
    """
    日時の時刻で, 同じ提示の重なりを見分けるかを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ日時の重なりは除き, 1 秒違う行は別の提示
    path = write_csv(tmp_path / "r.csv", [
        "u,i,t,seen",
        "a,x,2024-01-01 10:00:00,0",
        "a,x,2024-01-01 10:00:00,1",
        "a,x,2024-01-01 10:00:01,1",
        "a,y,2024-01-01 10:00:00,1",
    ])
    reactions = load_reactions(path, "u", "i", "t", shown="seen")
    assert reactions.time_kind == "datetime"
    assert reactions.dropped_conflict == 1
    assert reactions.shown["item"].tolist() == ["y", "x"]
    summary = describe_reactions(reactions)
    assert summary["shown"]["presentations"] == 2
    assert summary["shown"]["chosen_presentations"] == 1


def test_shown_presentation(tmp_path: Path) -> None:
    """
    提示の列で, 時刻の違う選んだ行と見せただけの行を結び付けるかを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 選んだ行の時刻 (クリック) は提示の時刻より後
    path = write_csv(tmp_path / "r.csv", [
        "u,i,t,seen,shot",
        "a,x,5,0,p1",
        "a,x,1,1,p1",
        "a,y,1,1,p1",
        "a,x,7,1,p2",
        "b,y,3,1,",
    ])
    reactions = load_reactions(
        path, "u", "i", "t", shown="seen", presentation="shot"
    )
    assert reactions.dropped_missing == 1
    assert reactions.dropped_conflict == 1
    assert list(reactions.shown.columns) == [
        "user", "item", "time", "presentation"
    ]
    assert reactions.shown["presentation"].tolist() == ["p1", "p2"]
    assert reactions.frame["presentation"].tolist() == ["p1"]
    summary = describe_reactions(reactions)
    assert summary["columns"]["presentation"] == "shot"
    assert summary["shown"]["by"] == "presentation"
    assert summary["shown"]["presentations"] == 2
    assert summary["shown"]["chosen_presentations"] == 1

    # 提示の列が無ければ, 時刻の違う行は別の提示で, 重ならない
    reactions = load_reactions(path, "u", "i", "t", shown="seen")
    assert reactions.dropped_conflict == 0
    assert len(reactions.shown) == 4


def test_shown_nullable_types() -> None:
    """
    nullable の boolean と Int64 の印を読めるかを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 欠けた印 (pd.NA) は偽, 読めない数 (2) は除く
    frame = pd.DataFrame({
        "u": [1, 1, 2, 2], "i": [10, 11, 10, 11],
        "t": [1.0, 1.0, 2.0, 2.0],
    })
    marks = {
        "boolean": pd.array([None, True, False, True], dtype="boolean"),
        "Int64": pd.array([pd.NA, 1, 0, 2], dtype="Int64"),
    }
    for kind, values in marks.items():
        data = ml_eda.Data(kind="table", frame=frame.assign(seen=values))
        reactions = load_reactions(data, "u", "i", "t", shown="seen")
        assert reactions.frame["item"].tolist() == [10, 10], kind
        expected = [11, 11] if kind == "boolean" else [11]
        assert reactions.shown["item"].tolist() == expected, kind
        assert reactions.dropped_shown == (0 if kind == "boolean" else 1)
