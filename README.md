# ML-EDA

機械学習をする前に, 入力となるデータを探索的に分析する (EDA) ライブラリ.

表, 時系列, 画像, 文, 点群のように形の違うデータを読み込み, 列の型, 分布, 欠損, 関係, 学習の前に確かめた方がよいこと (気付き) を調べ, 前処理のレシピで学習用のデータセットに書き出す. 書き出したデータセットは, [ML-Found](https://github.com/victor-von-pooh/ML-Found) の学習スクリプトがそのまま読み込める.

- **読み込み**: 拡張子と中身から形を判定する (csv, tsv, json, jsonl, parquet, xlsx, npy など. 文字コードと区切り文字も推定)
- **分析**: 列の型の推定, 統計とヒストグラム, 相関, 目的の列との関連の強さ, 時系列の間隔と周期, 気付き (欠損の多い列, ID らしい列, リークの疑い, クラスの偏りなど)
- **前処理**: 手順の並び (レシピ) を YAML で書く. 値を学ぶ手順 (標準化, 欠損の補完など) は分割した後の学習用のデータだけで当てはめ, 検証用とテスト用には同じ値を使う
- **書き出し**: 分割ごとのデータと, 当てはめた値を含むレシピと, 説明 (`dataset.yaml`) を 1 つのディレクトリに書く

GUI のアプリ ML-Fathom はこのライブラリを使う. 設計の方針は [docs/DESIGN.md](docs/DESIGN.md) を参照.

## 導入

```bash
git clone https://github.com/victor-von-pooh/ML-EDA.git
cd ML-EDA
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"     # parquet, Excel, 画像, 文の部分語も使う場合 (使うだけなら pip install -e . でもよい)
pip install -e ".[dev,all]" # テストも動かす場合
```

PyPI には公開していないので, 別の環境 (学習のプロジェクトなど) には `pip install "ml-eda[all] @ git+https://github.com/victor-von-pooh/ML-EDA.git"` で入れる. Python 3.10 以上.

## 使い方

### 読み込みと分析

```python
from ml_eda import detect, load_data, profile

detect("titanic.csv")                 # 読める読み込みの候補 (確からしい順)
data = load_data("titanic.csv")       # 形を判定して読み込む (Data)
data.kind                             # table, timeseries など
data.frame                            # pandas の DataFrame
result = profile(data, target="Survived")   # JSON にできる辞書

for item in result["findings"]:       # 気付き
    print(item["severity"], item["code"], item["message"])
```

`profile` の結果は `overview` (行数, 重複など), `columns` (列ごとの型と統計), `correlations` (数値の列の相関), `associations` (カテゴリの列の関連), `target` (目的の列の分布と, 各列との関連の強さの順位), `timeseries` (時系列の間隔, 抜け, 周期の候補), `findings` (気付き).

日時の列を持つ表は時系列として読み, 時刻の順に並べる (`time_column` で列を指定, `kind="table"` で時系列にしない). 読み込みの設定は `load_data(path, sep=";", encoding="cp932")` のように渡す.

### 前処理と書き出し

```yaml
# recipe.yaml
format_version: 1
target: Survived
task: classification          # classification, regression, forecasting, none
steps:
  - {op: drop_columns, columns: [PassengerId, Name, Ticket, Cabin]}
  - {op: impute, columns: [Age], strategy: median, indicator: true}
  - {op: impute, columns: [Embarked], strategy: most_frequent}
  - {op: one_hot, columns: auto}
  - {op: standardize, columns: [Age, Fare]}
split: {method: stratified, valid: 0.2, test: 0.1, seed: 0}
```

```python
from ml_eda import export_dataset, preview

preview(data, "recipe.yaml", 4)      # 4 つ目の手順の前と後
export_dataset(data, "recipe.yaml", "datasets/titanic")
```

使える手順は `describe_steps()` で一覧にできる (列を落とす, 行を絞る, 型を変える, 欠損の補完と行の削除, 重複の削除, 外れ値の切り詰め, 対数, 標準化, 最小と最大の正規化, ワンホット, 番号付け, 日時の分解, 時系列の過去の値と移動平均). `columns: auto` は目的の列などを除く, 手順に合う型 (数値かカテゴリ) の列.

分割は `random`, `stratified` (目的の列の割合を保つ), `time` (時刻の順. 最後の部分を検証用とテスト用に), `group` (`column` の値ごとに同じ分割), `none`.

### 画像, 文, 点群

画像のディレクトリ (`<クラス>/<画像>` ならラベル付き), 1 行 1 文の `.txt` か `<クラス>/<文>.txt`, 点群 (`.ply`, `.pcd`, `.xyz`, `.npy`. 1 つか `<クラス>/<点群>`) も `load_data` で読める. 表 (`data.frame`) は 1 行 1 件の様子 (画像なら幅, 高さ, ラベル) で, 中身は `data.load_item(i)` で読む. 分析には大きさのばらつきなどの気付きが加わる.

学習用にするには, レシピに中身の手順 (`item_steps`) を書く.

```yaml
target: label
task: classification
item_steps:
  - {op: resize, width: 64, height: 64}   # 画像. 文は {op: vocabulary, length: 128}
  - {op: normalize}                       # 点群は sample_points と normalize_cloud
split: {method: stratified, valid: 0.2}
```

文の区切り方 (文字 `char`, 語 `word`, 部分語 `subword`) は, 分析の結果の `result["items"]["tokenizers"]` が見本の文で比べて勧める (語彙の大きさ, 1 つの文のトークンの数, 語彙に無いトークンの割合). `recommended` をそのまま `item_steps` に入れられる. 部分語は `tokenizers` が要る (`[all]` か `[text]` で入る).

`load_dataset(...).tensors("train")` は (画像なら (件数, チャンネル, 高さ, 幅)) の配列とラベルを返す.

### 学習で使う

```python
from ml_eda import load_dataset

dataset = load_dataset("datasets/titanic")
x_train, y_train = dataset.tensors("train")     # PyTorch のテンソル (arrays で numpy)
x_valid, y_valid = dataset.tensors("valid")
dataset.info["classes"]                         # 分類のクラス (番号の順)
dataset.transform(new_frame)                    # 新しいデータに同じ前処理 (推論のとき)
```

ML-Found の学習スクリプトでは, `TensorDataset(*dataset.tensors("train"))` を `DataLoader` に渡す. 入力の次元は `len(dataset.features)`, 分類のクラスの数は `len(dataset.info["classes"])`.

## 開発

```bash
pytest
```

## ライセンス

MIT
