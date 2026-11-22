from pathlib import Path

import pandas as pd
import pytest
import yaml

from ml_eda import export_dataset, load_data, load_dataset


def test_dataset_inside(tmp_path: Path) -> None:
    """
    版 1 の load_dataset がデータセットの中だけを読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 小さな表を書き出す
    frame = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
    frame.to_csv(tmp_path / "table.csv", index=False)
    recipe = {
        "target": "y", "task": "classification",
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    export_dataset(
        load_data(tmp_path / "table.csv"), recipe, tmp_path / "ds"
    )
    root = tmp_path / "ds"
    info = yaml.safe_load((root / "dataset.yaml").read_text("utf-8"))
    original = info["splits"]["train"]["file"]
    assert len(load_dataset(root).frames["train"]) > 0

    # 説明の場所が .. や絶対パスで外を指すと誤り
    secret = tmp_path / "secret.csv"
    secret.write_bytes((root / original).read_bytes())
    for name in ("../secret.csv", str(secret)):
        info["splits"]["train"]["file"] = name
        (root / "dataset.yaml").write_text(yaml.safe_dump(info), "utf-8")
        with pytest.raises(ValueError, match="データセットの外"):
            load_dataset(root)

    # 分割のデータが外を指すリンクなら, 引数の案内付きの誤り
    info["splits"]["train"]["file"] = original
    (root / "dataset.yaml").write_text(yaml.safe_dump(info), "utf-8")
    (root / original).unlink()
    (root / original).symlink_to(secret)
    with pytest.raises(ValueError, match="follow_links=True"):
        load_dataset(root)
    assert len(load_dataset(root, follow_links=True).frames["train"]) > 0

    # レシピが外を指すリンクでも誤り. 無いファイルは FileNotFoundError
    (root / original).unlink()
    (root / original).write_bytes(secret.read_bytes())
    recipe_file = root / info.get("recipe", "recipe.yaml")
    outside = tmp_path / "recipe.yaml"
    recipe_file.rename(outside)
    recipe_file.symlink_to(outside)
    with pytest.raises(ValueError, match="外を指すリンク"):
        load_dataset(root)
    recipe_file.unlink()
    with pytest.raises(FileNotFoundError):
        load_dataset(root)
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing")
