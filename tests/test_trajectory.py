import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_eda import (
    TRAJECTORY_FORMAT_VERSION, describe_trajectory, load_trajectory
)
from ml_eda import trajectory as trajectory_module

# ML-Found の save_trajectory が書いた軌跡 (説明付き)
SAMPLES = Path(__file__).parent / "data" / "trajectories"

# 利用者の表の見出し (説明無し)
HEADER = "episode,step,obs,action,reward,terminated,truncated"


def write_table(path: Path, lines: list[str]) -> Path:
    """
    軌跡の CSV を書く関数

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


def copy_sample(tmp_path: Path, name: str, info: bool = True) -> Path:
    """
    ML-Found の軌跡を一時ディレクトリに写す関数

    Parameters
    ----------
    tmp_path: Path
        写す先
    name: str
        軌跡の名前
    info: bool = True
        説明も写すか

    Returns
    ----------
    path: Path
        写した表
    """
    # 表と, 必要なら説明
    path = tmp_path / f"{name}.csv"
    shutil.copy(SAMPLES / f"{name}.csv", path)
    if info:
        shutil.copy(SAMPLES / f"{name}.json", tmp_path / f"{name}.json")

    return path


def test_reads_found_trajectory() -> None:
    """
    ML-Found の評価の軌跡を説明付きで読み, 要約できることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 5 歩で終わる 2 エピソード. 行動はいつも 1 で報酬 1
    trajectory = load_trajectory(SAMPLES / "valid.csv")
    assert trajectory.format == "csv"
    assert trajectory.info["format_version"] == TRAJECTORY_FORMAT_VERSION
    assert trajectory.observation_columns == ["obs_0"]
    assert trajectory.action_columns == ["action"]
    assert not trajectory.guarded
    assert trajectory.notes == []
    frame = trajectory.frame
    assert frame["episode"].dtype == np.int64
    assert frame["terminated"].dtype == bool
    assert frame["action"].dtype == np.int64

    # 要約は JSON にでき, 説明の項目を写す
    summary = describe_trajectory(trajectory)
    json.dumps(summary, allow_nan=False)
    assert summary["described"]
    assert summary["epoch"] == 4
    assert summary["source"] == "evaluation_loader"
    assert summary["action_space"] == {"type": "discrete", "n": 2, "start": 0}
    assert summary["episodes"] == 2
    assert summary["rows"] == summary["steps"] == 10
    assert summary["length"]["mean"] == 5.0
    assert summary["return"]["max"] == 5.0
    assert summary["ends"]["terminated"] == 2
    assert summary["actions"][0]["counts"] == [{"value": 1, "count": 10}]
    assert summary["observations"][0]["max"] == 4.0
    assert summary["per_episode"][1] == {
        "episode": 1, "rows": 5, "length": 5, "return": 5.0,
        "end": "terminated", "seed": 4,
    }
    assert summary["guard"] is None


def test_reads_guarded_trajectory() -> None:
    """
    守りの列のある軌跡で, 止めた判断の行と守りの結果を数えることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 3 歩と止めた判断の 1 行のエピソードが 2 つ
    trajectory = load_trajectory(SAMPLES / "guarded.csv")
    assert trajectory.guarded
    frame = trajectory.frame
    assert frame["reward"].isna().sum() == 2
    assert frame["guard_reasons"].iloc[0] == ""

    # 歩数と報酬の和は報酬のある行で, 終わり方は止めた判断
    summary = describe_trajectory(trajectory)
    assert summary["steps"] == 6
    assert [row["length"] for row in summary["per_episode"]] == [3, 3]
    assert [row["rows"] for row in summary["per_episode"]] == [4, 4]
    assert summary["return"]["mean"] == 3.0
    assert summary["ends"]["stop"] == 2
    assert summary["actions"][0]["missing"] == 2
    assert summary["guard"]["results"] == {"ok": 6, "stop": 2}
    assert summary["guard"]["reasons"] == [
        {"reason": "観測の要素 0 が範囲の外 (3)", "count": 2}
    ]


def test_same_without_info(tmp_path: Path) -> None:
    """
    説明の無い表を, 列の名前と値から同じ型で読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 説明付きと同じ型, 列, 要約 (説明の項目は除く)
    for name in ("valid", "guarded", "box"):
        bare = load_trajectory(copy_sample(tmp_path, name, info=False))
        full = load_trajectory(SAMPLES / f"{name}.csv")
        assert bare.info is None
        assert bare.types == full.types
        assert bare.observation_columns == full.observation_columns
        assert bare.action_columns == full.action_columns
        assert bare.guarded == full.guarded
        pd.testing.assert_frame_equal(bare.frame, full.frame)
        summary = describe_trajectory(bare)
        assert not summary["described"]
        assert summary["epoch"] is None
        assert summary["per_episode"] == (
            describe_trajectory(full)["per_episode"]
        )


def test_reads_from_info(tmp_path: Path) -> None:
    """
    説明を渡すと, その file の表を読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 説明から表
    trajectory = load_trajectory(SAMPLES / "box.json")
    assert trajectory.name == "box"
    assert trajectory.action_columns == ["action_0"]

    # 別のディレクトリを指す file は誤り
    info = json.loads((SAMPLES / "box.json").read_text(encoding="utf-8"))
    info["file"] = "../box.csv"
    (tmp_path / "bad.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="file"):
        load_trajectory(tmp_path / "bad.json")


def test_nan_is_not_missing(tmp_path: Path) -> None:
    """
    nan と無限大を値として残し, 空だけを欠けにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 観測の nan と inf, 報酬の nan, 形の合わない観測の空
    path = write_table(tmp_path / "t.csv", [
        HEADER,
        "0,0,nan,0,1.0,false,false",
        "0,1,inf,1,1.0,false,false",
        "0,2,,1,1.0,true,false",
        "1,0,0.5,0,nan,true,false",
    ])
    trajectory = load_trajectory(path)
    frame = trajectory.frame
    assert frame["obs"].isna().tolist() == [False, False, True, False]
    assert math.isnan(frame["obs"].iloc[0])
    assert frame["reward"].dtype == np.float64

    # 要約は nan, 無限大, 欠けを分けて数え, nan を含む和は None
    summary = describe_trajectory(trajectory)
    observation = summary["observations"][0]
    assert (observation["missing"], observation["nan"]) == (1, 1)
    assert observation["infinite"] == 1
    assert observation["max"] == 0.5
    assert summary["per_episode"][1]["return"] is None
    assert summary["unknown_returns"] == 1
    assert summary["return"]["mean"] == 3.0


def test_element_columns_by_number(tmp_path: Path) -> None:
    """
    観測と行動の列を名前で見分け, 番号の順に並べ, 知らない列を残す
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 列の並びは番号の順でなく, 後で足される確率の列もある
    path = write_table(tmp_path / "t.csv", [
        "episode,step,obs_10,obs_2,obs_0,action_1,action_0,reward,"
        "terminated,truncated,action_prob",
        "0,0,1,2,3,0.5,0.25,1,false,true,0.9",
    ])
    trajectory = load_trajectory(path)
    assert trajectory.observation_columns == ["obs_0", "obs_2", "obs_10"]
    assert trajectory.action_columns == ["action_0", "action_1"]
    assert any("番号" in note for note in trajectory.notes)
    summary = describe_trajectory(trajectory)
    assert summary["extra_columns"] == ["action_prob"]
    assert trajectory.types["action_prob"] == "number"
    assert summary["ends"]["truncated"] == 1
    assert summary["observations"][0]["counts"] == [{"value": 3, "count": 1}]


@pytest.mark.parametrize(("header", "message"), [
    ("episode,step,obs,obs_0,action,reward,terminated,truncated", "並んで"),
    ("episode,step,action,reward,terminated,truncated", "obs"),
    ("episode,step,obs,reward,terminated,truncated", "action"),
    ("episode,step,obs,action,terminated,truncated", "reward"),
    ("episode,step,obs,action,reward,terminated,terminated", "重なって"),
])
def test_rejects_bad_columns(
    tmp_path: Path, header: str, message: str
) -> None:
    """
    列の決まりに合わない表を誤りにすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    header: str
        見出し
    message: str
        誤りに含む文字

    Returns
    ----------
    None
    """
    # 値は全て 0 (真偽値の列も読めるように false)
    values = ",".join(
        "false" if name in ("terminated", "truncated") else "0"
        for name in header.split(",")
    )
    path = write_table(tmp_path / "t.csv", [header, values])
    with pytest.raises(ValueError, match=message):
        load_trajectory(path)


@pytest.mark.parametrize("line", [
    "x,0,0,0,1,false,false",
    "0,0,0,0,1,maybe,false",
    "0,0,0,0,1,,false",
    "0,0,a,0,1,false,false",
    "0,0,0,0,1,false,false,9",
])
def test_rejects_bad_values(tmp_path: Path, line: str) -> None:
    """
    型に合わない値と, 番号と終わりの印の空を誤りにすることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    line: str
        1 行の値

    Returns
    ----------
    None
    """
    # どれも列の名前か, 列の数との食い違いを挙げて誤り
    path = write_table(tmp_path / "t.csv", [HEADER, line])
    with pytest.raises(ValueError, match="列"):
        load_trajectory(path)


def test_reorders_rows(tmp_path: Path) -> None:
    """
    エピソード, 歩の順でない表を並べ直して知らせ, 終わりの印の無い
    エピソードを open にすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # エピソード 1 が先, エピソード 0 は終わりの印が無い
    path = write_table(tmp_path / "t.csv", [
        HEADER,
        "1,1,0,0,1,true,false",
        "0,0,0,0,1,false,false",
        "1,0,0,0,1,false,false",
    ])
    trajectory = load_trajectory(path)
    assert trajectory.frame["episode"].tolist() == [0, 1, 1]
    assert trajectory.frame["step"].tolist() == [0, 0, 1]
    assert any("並べ直し" in note for note in trajectory.notes)
    summary = describe_trajectory(trajectory)
    assert summary["ends"]["open"] == 1
    assert summary["ends"]["terminated"] == 1


def test_info_notes(tmp_path: Path) -> None:
    """
    説明の新しい版, 誤った版, 表との食い違いを知らせることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 新しい版は読める項目を使う. 行の数の食い違いも知らせる
    path = copy_sample(tmp_path, "valid")
    info_path = tmp_path / "valid.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update(format_version=TRAJECTORY_FORMAT_VERSION + 1, rows=3)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    trajectory = load_trajectory(path)
    assert trajectory.info["epoch"] == 4
    assert any("新しい" in note for note in trajectory.notes)
    assert any("rows" in note for note in trajectory.notes)

    # 版が整数でない説明は使わず, 説明の無い表として読む
    info["format_version"] = "1"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    trajectory = load_trajectory(path)
    assert trajectory.info is None
    assert any("format_version" in note for note in trajectory.notes)


def test_reads_regular_files_only(tmp_path: Path) -> None:
    """
    リンク, 普通でないファイル, 上限を超えるファイルを読まないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表のリンクとディレクトリは誤り
    path = copy_sample(tmp_path, "valid")
    link = tmp_path / "link.csv"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="リンク"):
        load_trajectory(link)
    (tmp_path / "folder.csv").mkdir()
    with pytest.raises(ValueError, match="普通"):
        load_trajectory(tmp_path / "folder.csv")
    with pytest.raises(FileNotFoundError):
        load_trajectory(tmp_path / "none.csv")
    with pytest.raises(ValueError, match="csv か .parquet"):
        load_trajectory(tmp_path / "valid.txt")

    # FIFO は待たずに誤り
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "fifo.csv")
        with pytest.raises(ValueError, match="普通"):
            load_trajectory(tmp_path / "fifo.csv")

    # 説明のリンクは使わず, 説明の無い表として読む
    other = tmp_path / "other"
    other.mkdir()
    table = copy_sample(other, "valid", info=False)
    (other / "valid.json").symlink_to(path.with_suffix(".json"))
    trajectory = load_trajectory(table)
    assert trajectory.info is None
    assert any("リンク" in note for note in trajectory.notes)


def test_size_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    表と説明の大きさの上限を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        上限を小さくする

    Returns
    ----------
    None
    """
    # 説明が上限を超えれば使わない
    path = copy_sample(tmp_path, "valid")
    monkeypatch.setattr(trajectory_module, "MAX_INFO_BYTES", 100)
    trajectory = load_trajectory(path)
    assert trajectory.info is None
    assert any("MiB" in note for note in trajectory.notes)

    # 表が上限を超えれば誤り
    monkeypatch.setattr(trajectory_module, "MAX_TABLE_BYTES", 100)
    with pytest.raises(ValueError, match="MiB"):
        load_trajectory(path)


def test_reads_parquet(tmp_path: Path) -> None:
    """
    Parquet の表を CSV と同じ口で読み, 欠けと nan を分けることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    # 守りで止めた行の行動と報酬は null, 報酬の nan は値
    table = pa.table({
        "episode": [0, 0, 1], "step": [0, 1, 0],
        "obs_0": [0.0, 1.0, 2.0], "obs_1": [1.0, 1.0, 1.0],
        "action": pa.array([1, None, 0]),
        "reward": pa.array([float("nan"), None, 2.0]),
        "terminated": [False, False, True], "truncated": [False] * 3,
        "guard": ["ok", "handover", "clipped"],
        "guard_reasons": ["", "人に渡す", ""],
    })
    path = tmp_path / "t.parquet"
    pq.write_table(table, path)
    trajectory = load_trajectory(path)
    assert trajectory.format == "parquet"
    assert trajectory.observation_columns == ["obs_0", "obs_1"]
    assert trajectory.types["action"] == "integer"
    reward = trajectory.frame["reward"]
    assert reward.isna().tolist() == [False, True, False]
    assert math.isnan(reward.iloc[0])
    assert trajectory.frame["terminated"].dtype == bool

    # 要約
    summary = describe_trajectory(trajectory)
    assert summary["format"] == "parquet"
    assert [row["end"] for row in summary["per_episode"]] == [
        "handover", "terminated"
    ]
    assert summary["per_episode"][0]["return"] is None
    assert summary["guard"]["results"] == {
        "ok": 1, "handover": 1, "clipped": 1
    }

    # 型の合わない列は誤り
    pq.write_table(table.set_column(
        4, "action", pa.array(["a", "b", "c"])
    ), path)
    with pytest.raises(ValueError, match="action"):
        load_trajectory(path)
