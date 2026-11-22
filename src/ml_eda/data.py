from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

# データの種類
KINDS = (
    "table", "timeseries", "image", "text", "pointcloud", "capture", "audio"
)

# 列の役割
ROLES = ("target", "time", "id", "text", "path", "label")


@dataclass
class Data:
    # 種類 (KINDS のどれか)
    kind: str
    # 1 行が 1 件 (表の 1 行, 画像の 1 枚など) の表
    frame: pd.DataFrame
    # 読み込んだ場所, 読み込みの名前, 読み込みの設定
    source: dict = field(default_factory=dict)
    # 列の役割 (役割の名前から列の名前)
    roles: dict = field(default_factory=dict)
    # 画像や点群の 1 件の中身を読む関数 (表には無い)
    item_loader: Optional[Callable[[int], Any]] = None

    def load_item(self, index: int) -> Any:
        """
        画像や点群など, 1 件の中身を読む関数

        Parameters
        ----------
        index: int
            frame の行の位置

        Returns
        ----------
        item: Any
            中身 (画像は numpy の配列, 点群は点の配列と属性の辞書など)

        Raises
        ----------
        ValueError
            中身を別に持たない種類 (表など) の場合
        """
        # 中身を別に持つ種類だけ読む
        if self.item_loader is None:
            raise ValueError(f"{self.kind} のデータは 1 件ずつの中身を持ちません")
        item = self.item_loader(index)

        return item

    def summary(self) -> dict:
        """
        種類, 件数, 列, 役割, 読み込んだ場所を辞書にする関数

        Parameters
        ----------
        None

        Returns
        ----------
        summary: dict
            JSON にできる概要
        """
        # 列の名前と pandas の型の並び
        columns = [
            {"name": str(name), "dtype": str(dtype)}
            for name, dtype in self.frame.dtypes.items()
        ]
        summary = {
            "kind": self.kind,
            "rows": int(len(self.frame)),
            "columns": columns,
            "roles": dict(self.roles),
            "source": dict(self.source),
            "has_items": self.item_loader is not None,
        }

        return summary
