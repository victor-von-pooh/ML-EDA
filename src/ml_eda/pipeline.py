import datetime as dt
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ml_eda.data import Data
from ml_eda.profile import to_json
from ml_eda.recipe import STEPS, load_recipe, to_plain

# データセットの説明の形式の版
DATASET_FORMAT_VERSION = 1

# 分割を表す列 (処理の間だけ使う)
SPLIT_COLUMN = "__split__"

# 分割の名前
SPLITS = ("train", "valid", "test")


@dataclass
class Prepared:
    # 分割ごとの前処理した表 (目的の列を含む)
    frames: dict
    # 当てはめた値を含むレシピ
    recipe: dict
    # 学習に使う列 (目的の列を除く)
    features: list
    # 目的の列 (無ければ None)
    target: Optional[str]
    # 分類のクラス (番号の順)
    classes: list = field(default_factory=list)
    # 前処理の後の注意 (数値でない列が残っているなど)
    notes: list = field(default_factory=list)


def assign_splits(
    frame: pd.DataFrame, split: dict, roles: dict
) -> pd.Series:
    """
    各行を学習用, 検証用, テスト用に割り当てる関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    split: dict
        method (random, stratified, time, group, none), valid, test
        (割合), seed, column (group の列)
    roles: dict
        列の役割 (target と time)

    Returns
    ----------
    assigned: pd.Series
        行ごとの分割の名前

    Raises
    ----------
    ValueError
        分割に要る列が無い場合と, 割合の誤りの場合
    """
    # 割合の確認
    method = split.get("method", "random")
    valid = float(split.get("valid", 0.2))
    test = float(split.get("test", 0.0))
    if method == "none":
        return pd.Series("train", index=frame.index)
    if valid < 0 or test < 0 or valid + test >= 1:
        raise ValueError("valid と test の割合は 0 以上で, 合計を 1 未満に")
    rng = np.random.default_rng(int(split.get("seed", 0)))

    # 時刻の順は, 最後の部分を検証用とテスト用にする
    if method == "time":
        time = roles.get("time")
        if not time:
            raise ValueError("時刻の順の分割には日時の列が要ります")
        order = np.argsort(pd.to_datetime(frame[time]).to_numpy(),
                           kind="stable")
        return label_by_position(frame.index, order, valid, test)

    # グループごとに同じ分割にする
    if method == "group":
        column = split.get("column")
        if column not in frame.columns:
            raise ValueError(f"グループの列 {column} がありません")
        groups = frame[column].astype(str).unique()
        rng.shuffle(groups)
        labels = label_by_position(
            pd.Index(groups), np.arange(len(groups)), valid, test
        )
        return frame[column].astype(str).map(labels)

    # 層化は目的の列の値ごとに, 無作為はまとめて割り当て
    if method == "stratified":
        target = roles.get("target")
        if not target:
            raise ValueError("層化の分割には目的の列が要ります")
        assigned = pd.Series("train", index=frame.index)
        for _, group in frame.groupby(frame[target].astype(str)):
            order = rng.permutation(len(group))
            assigned.loc[group.index] = label_by_position(
                group.index, order, valid, test
            )
        return assigned
    order = rng.permutation(len(frame))
    assigned = label_by_position(frame.index, order, valid, test)

    return assigned


def label_by_position(
    index: pd.Index, order: np.ndarray, valid: float, test: float
) -> pd.Series:
    """
    並べた順の最後の部分を検証用とテスト用にする関数

    Parameters
    ----------
    index: pd.Index
        行の名前
    order: np.ndarray
        行の位置の並べ方
    valid: float
        検証用の割合
    test: float
        テスト用の割合

    Returns
    ----------
    labels: pd.Series
        行ごとの分割の名前
    """
    # 件数を割合から求め, 並べた順の後ろから割り当てる
    count = len(order)
    n_test = int(round(count * test))
    n_valid = int(round(count * valid))
    if valid > 0 and n_valid == 0 and count - n_test >= 2:
        n_valid = 1
    names = np.array(["train"] * count, dtype=object)
    names[order[count - n_test - n_valid:count - n_test]] = "valid"
    if n_test:
        names[order[count - n_test:]] = "test"
    labels = pd.Series(names, index=index)

    return labels


def run_steps(
    frame: pd.DataFrame, recipe: dict, roles: dict,
    upto: Optional[int] = None
) -> pd.DataFrame:
    """
    レシピの手順を順にかける関数 (値を学ぶ手順は学習用の行で当てはめ)

    Parameters
    ----------
    frame: pd.DataFrame
        分割の列 (SPLIT_COLUMN) を持つ表
    recipe: dict
        レシピ (当てはめた値を各手順の state に書き込む)
    roles: dict
        列の役割
    upto: Optional[int] = None
        この数の手順までかける (None は全て)

    Returns
    ----------
    result: pd.DataFrame
        手順をかけた表
    """
    # 手順ごとに当てはめと適用
    result = frame
    steps = recipe["steps"][:upto] if upto is not None else recipe["steps"]
    for number, spec in enumerate(steps, 1):
        step = STEPS[spec["op"]]
        params = {k: v for k, v in spec.items() if k not in ("op", "state")}
        try:
            if step.fit is not None:
                train = result[result[SPLIT_COLUMN] == "train"]
                spec["state"] = to_plain(step.fit(train, params, roles))
            result = step.apply(result, params, spec.get("state", {}))
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(
                f"手順 {number} ({spec['op']}) で失敗しました: {error}"
            ) from error

    return result


def prepare(
    data: Data, recipe: Union[str, Path, dict]
) -> Prepared:
    """
    レシピで分割と前処理をして, 学習用の表にする関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: Union[str, Path, dict]
        レシピ

    Returns
    ----------
    prepared: Prepared
        分割ごとの表, 当てはめたレシピ, 学習に使う列, クラス
    """
    # 目的の列の欠損の行を落としてから分割
    recipe = load_recipe(recipe)
    target = recipe.get("target") or data.roles.get("target")
    roles = {**data.roles, "target": target} if target else dict(data.roles)
    frame = data.frame.copy()
    if target:
        if target not in frame.columns:
            raise ValueError(f"目的の列 {target} がありません")
        frame = frame[frame[target].notna()]
    frame[SPLIT_COLUMN] = assign_splits(frame, recipe["split"], roles)

    # 手順をかけ, 分割ごとに分ける
    result = run_steps(frame, recipe, roles)
    classes = []
    if target and recipe["task"] == "classification":
        classes = sorted(result[target].astype(str).unique().tolist())
        index = {value: number for number, value in enumerate(classes)}
        result[target] = result[target].astype(str).map(index).astype(int)
    frames = {
        name: result[result[SPLIT_COLUMN] == name].drop(
            columns=[SPLIT_COLUMN]
        ).reset_index(drop=True)
        for name in SPLITS
        if (result[SPLIT_COLUMN] == name).any()
    }

    # 学習に使う列と, 数値でない列の注意
    reserved = {target, roles.get("id"), SPLIT_COLUMN}
    features = [name for name in result.columns if name not in reserved]
    notes = []
    for name in features:
        if not pd.api.types.is_numeric_dtype(result[name]):
            notes.append(
                f"{name} は数値ではありません ({result[name].dtype}). "
                "学習の前に符号化 (one_hot, label_encode) か日時の分解を"
                "してください"
            )
        elif result[name].isna().any():
            notes.append(f"{name} に欠損が残っています (impute で埋める)")
    prepared = Prepared(
        frames=frames, recipe=recipe, features=features, target=target,
        classes=classes, notes=notes
    )

    return prepared


def preview(
    data: Data, recipe: Union[str, Path, dict], upto: int, rows: int = 20
) -> dict:
    """
    手順を 1 つずつかけた前と後の表の様子を返す関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: Union[str, Path, dict]
        レシピ
    upto: int
        この数の手順までかけた後を見る (前は upto - 1 まで)
    rows: int = 20
        返す行の数

    Returns
    ----------
    result: dict
        前と後の行数, 列, 先頭の行, 足した列と消えた列
    """
    # 分割してから, 手順を upto - 1 と upto までかける
    recipe = load_recipe(recipe)
    target = recipe.get("target") or data.roles.get("target")
    roles = {**data.roles, "target": target} if target else dict(data.roles)
    frame = data.frame.copy()
    if target and target in frame.columns:
        frame = frame[frame[target].notna()]
    frame[SPLIT_COLUMN] = assign_splits(frame, recipe["split"], roles)
    before = run_steps(frame, load_recipe(recipe), roles, max(upto - 1, 0))
    after = run_steps(frame, load_recipe(recipe), roles, upto)

    # 列の差と先頭の行
    before = before.drop(columns=[SPLIT_COLUMN])
    after = after.drop(columns=[SPLIT_COLUMN])
    result = {
        "before": frame_view(before, rows),
        "after": frame_view(after, rows),
        "added": [c for c in after.columns if c not in before.columns],
        "removed": [c for c in before.columns if c not in after.columns],
    }
    result = to_json(result)

    return result


def frame_view(frame: pd.DataFrame, rows: int) -> dict:
    """
    表の行数, 列と型, 先頭の行を JSON にできる形にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    rows: int
        返す行の数

    Returns
    ----------
    view: dict
        行数, 列の名前と型, 先頭の行 (値は文字か数)
    """
    # 先頭の行を JSON にできる値にする
    head = frame.head(rows)
    records = [
        [
            None if pd.isna(value) else (
                value if isinstance(value, (int, float, bool, np.number))
                else str(value)
            )
            for value in row
        ]
        for row in head.itertuples(index=False)
    ]
    view = {
        "rows": int(len(frame)),
        "columns": [
            {"name": str(name), "dtype": str(dtype)}
            for name, dtype in frame.dtypes.items()
        ],
        "head": records,
    }

    return view


def write_frame(frame: pd.DataFrame, path: Path) -> str:
    """
    表を parquet (無ければ csv) で保存する関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    path: Path
        拡張子を除いた保存先

    Returns
    ----------
    name: str
        保存したファイルの名前
    """
    # pyarrow があれば型を保てる parquet
    try:
        target = path.with_suffix(".parquet")
        frame.to_parquet(target, index=False)
    except ImportError:
        target = path.with_suffix(".csv")
        frame.to_csv(target, index=False)
    name = target.name

    return name


def export_dataset(
    data: Data, recipe: Union[str, Path, dict], path: Union[str, Path],
    name: Optional[str] = None, overwrite: bool = False
) -> dict:
    """
    前処理して分割したデータを, 学習用のディレクトリに書き出す関数

    Parameters
    ----------
    data: Data
        読み込んだデータ
    recipe: Union[str, Path, dict]
        レシピ
    path: Union[str, Path]
        書き出すディレクトリ
    name: Optional[str] = None
        データセットの名前 (省略するとディレクトリの名前)
    overwrite: bool = False
        既にあるディレクトリを置き換えるか

    Returns
    ----------
    info: dict
        書き出した dataset.yaml の中身

    Raises
    ----------
    FileExistsError
        ディレクトリが既にあり, overwrite が False の場合
    """
    # 書き出す先の用意
    target = Path(path).expanduser()
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} は既にあります")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    # 前処理と, 分割ごとの保存
    prepared = prepare(data, recipe)
    splits = {}
    for split, frame in prepared.frames.items():
        splits[split] = {
            "file": write_frame(frame, target / split),
            "rows": int(len(frame)),
        }
    (target / "recipe.yaml").write_text(
        yaml.safe_dump(to_plain(prepared.recipe), allow_unicode=True,
                       sort_keys=False),
        encoding="utf-8"
    )

    # 説明
    first = next(iter(prepared.frames.values()))
    reference = prepared.frames.get("train", first)
    info = {
        "format_version": DATASET_FORMAT_VERSION,
        "name": name or target.name,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "kind": data.kind,
        "task": prepared.recipe["task"],
        "target": prepared.target,
        "features": [
            {"name": str(c), "dtype": str(reference[c].dtype)}
            for c in prepared.features
        ],
        "classes": prepared.classes,
        "splits": splits,
        "source": to_plain(data.source),
        "recipe": "recipe.yaml",
        "notes": prepared.notes,
    }
    (target / "dataset.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    return info


@dataclass
class Dataset:
    # 書き出したディレクトリ
    path: Path
    # dataset.yaml の中身
    info: dict
    # 分割ごとの表
    frames: dict
    # 当てはめた値を含むレシピ
    recipe: dict

    @property
    def features(self) -> list[str]:
        """
        学習に使う列の名前の並びを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        names: list[str]
            列の名前
        """
        # 説明の features の名前
        names = [item["name"] for item in self.info["features"]]

        return names

    @property
    def target(self) -> Optional[str]:
        """
        目的の列の名前を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        target: Optional[str]
            目的の列 (無ければ None)
        """
        # 説明の target
        target = self.info.get("target")

        return target

    def arrays(self, split: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        分割の特徴量と目的の値を numpy の配列で返す関数

        Parameters
        ----------
        split: str
            train, valid, test のどれか

        Returns
        ----------
        x: np.ndarray
            (件数, 列の数) の float32 の配列
        y: Optional[np.ndarray]
            目的の値 (分類は int64 の番号, 回帰は float32), 無ければ None
        """
        # 特徴量と目的の値を取り出す
        frame = self.frames[split]
        x = frame[self.features].to_numpy(dtype=np.float32)
        y = None
        if self.target:
            dtype = (
                np.int64 if self.info["task"] == "classification"
                else np.float32
            )
            y = frame[self.target].to_numpy(dtype=dtype)

        return x, y

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        新しいデータ (推論のときなど) に, 当てはめた前処理をかける関数

        Parameters
        ----------
        frame: pd.DataFrame
            元のデータと同じ列を持つ表

        Returns
        ----------
        result: pd.DataFrame
            学習に使う列だけの表 (当てはめた値をそのまま使う)
        """
        # 当てはめ直さずに, 保存した値で手順を順にかける
        result = frame.copy()
        for spec in self.recipe["steps"]:
            params = {
                k: v for k, v in spec.items() if k not in ("op", "state")
            }
            result = STEPS[spec["op"]].apply(
                result, params, spec.get("state", {})
            )
        result = result[[c for c in self.features if c in result.columns]]

        return result

    def tensors(self, split: str) -> tuple:
        """
        分割の特徴量と目的の値を PyTorch のテンソルで返す関数

        Parameters
        ----------
        split: str
            train, valid, test のどれか

        Returns
        ----------
        tensors: tuple
            (特徴量, 目的の値) のテンソル (目的の列が無ければ特徴量だけ)
        """
        # PyTorch は使うときだけ読み込む
        import torch

        x, y = self.arrays(split)
        tensors = (torch.from_numpy(x),) if y is None else (
            torch.from_numpy(x), torch.from_numpy(y)
        )

        return tensors


def load_dataset(path: Union[str, Path]) -> Dataset:
    """
    書き出したデータセットを読み込む関数

    Parameters
    ----------
    path: Union[str, Path]
        export_dataset で書き出したディレクトリ

    Returns
    ----------
    dataset: Dataset
        説明, 分割ごとの表, 当てはめたレシピ

    Raises
    ----------
    ValueError
        新しい版の場合
    """
    # 説明と版の確認
    root = Path(path).expanduser()
    info = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    version = info.get("format_version")
    if not isinstance(version, int) or version > DATASET_FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} のデータセットは, このライブラリ (版 "
            f"{DATASET_FORMAT_VERSION} まで) では読めません"
        )

    # 分割ごとの表とレシピ
    frames = {}
    for split, item in info["splits"].items():
        file = root / item["file"]
        frames[split] = (
            pd.read_parquet(file) if file.suffix == ".parquet"
            else pd.read_csv(file)
        )
    recipe = yaml.safe_load(
        (root / info.get("recipe", "recipe.yaml")).read_text(encoding="utf-8")
    )
    dataset = Dataset(path=root, info=info, frames=frames, recipe=recipe)

    return dataset

