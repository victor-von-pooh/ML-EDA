# ML-EDA

「機械学習を全て GUI 上で行う」うちの「データの分析」の段階のライブラリ. 機械学習に使うデータを読み込み, 分析し, 前処理して学習用に書き出す. 画面のアプリは [ML-Fathom](https://github.com/victor-von-pooh/ML-Fathom).

## できること

表, 時系列, 画像, 文, 点群, 音声のように形の違うデータを読み込み, 探索的に分析し (EDA), 前処理のレシピで学習用のデータセットに書き出す. 書き出したデータセットは, [ML-Found](https://github.com/victor-von-pooh/ML-Found) の学習スクリプトがそのまま読み込める.

- **読み込み**: 拡張子と中身から形を判定する (csv, tsv, json, jsonl, parquet, xlsx, npy など. 文字コードと区切り文字も推定)
- **分析**: 列の型の推定, 統計とヒストグラム, 相関, 目的の列との関連の強さ, 時系列の間隔と周期, 気付き (欠損の多い列, ID らしい列, リークの疑い, クラスの偏りなど)
- **前処理**: 手順の並び (レシピ) を YAML で書く. 値を学ぶ手順 (標準化, 欠損の補完など) は分割した後の学習用のデータだけで当てはめ, 検証用とテスト用には同じ値を使う
- **書き出し**: 分割ごとのデータと, 当てはめた値を含むレシピと, 説明 (`dataset.yaml`) を 1 つのディレクトリに書く
- **ファイルの組**: 画像と領域, 多視点の撮影と点群のように正解がファイルのデータを, データセットの版 2 で書き出して読む. 項目ごとの中身の手順 (大きさをそろえるなど) をかけると版 3
- **モデルに要るもの**: 書き出したデータセットから, モデルの入力の形と出す値の数を, 分割のデータを読まずに返す (`model_inputs`)
- **推論のときの同じ前処理**: 新しいデータに, 学習のときと同じレシピと中身の手順をかける
- **記録した軌跡**: 強化学習の 1 歩が 1 行の軌跡 (ML-Found が評価で書くものと, 手元の CSV と Parquet) を読み, エピソードの長さ, 報酬の和, 観測と行動の値, 守りの結果, 行動を選んだ確率の列の有無と範囲を要約する
- **並べるの反応の表**: 推薦の学習に使う 1 行が 1 つの反応の表を, ML-Quest の並べるの骨組みと同じ決まりで読み, 利用者, 品物, 反応の数, 利用者ごとの数の分布, 品物の人気の偏り, 時刻の範囲, 除いた行の数を要約する
- **群分けの表**: 群分け (まとめる) の 1 行が 1 つの対象の表を, 分析の保存として版 2 のデータセットに書き出して読み (学習には表のファイルを渡す), 学習の前に列の特徴をそのまま k-means で分けた群 (か渡した群) の要約 (群ごとの行の数, 割合, 安定性, 列の平均と最も多い値) を, ML-Quest のまとめるの骨組みの群の要約と同じ形で返す
- **異常の検知の表**: 正常の行が多く, 異常の正解が無いか少しだけある表を分析し (異常の行の数と割合, 正解の列の誤りなどの気付き), 学習用には正常の行だけを置き, 異常の行を検証用とテスト用に配って書き出す (`task: anomaly`)
- **介入の記録の表**: クーポンを無作為に配った記録のような, 処置の列と結果の列を持つ表を分析し (処置の群と対照の群の行の数, 結果の差, 特徴の釣り合い, 処置の後に決まる列などの気付き), 全ての分割に両方の群を置いて書き出す (`task: uplift`)
- **値段を変えた記録の表**: 試しに値段を変えた売れ行きの記録 (商品の特徴, 値段, 売れた数, 原価) を分析し (値段の幅と数, 値段と売れた数の関わり, 商品ごとの試した値段, 値段と他の列の関わりなどの気付き), 値段を入力に入れて売れた数を予測する学習のために, 値段が分割ごとに偏らないように書き出す (`task: price`)
- **需要の記録の表**: 商品ごとの日ごとの売れた数の記録 (商品の id, 日付, 売れた数, 商品の特徴, 在庫の数) を分析し (日付の範囲, 曜日ごとの売れた数, 商品ごとの記録の日数と抜けの日, 在庫が 0 の日などの気付き), 先の期間の需要を分位点で予測する学習のために, (商品, 発注を決める日) の行を, 先の期間が分割の境目をまたがないように書き出す (`task: demand`)

## どのアプリから使われるか

| 相手 | 使うもの |
| --- | --- |
| ML-Fathom (データの分析) | 読み込み, 分析, 前処理のレシピ, 書き出し, 表の 1 ページ (`frame_view`), 書き出したデータセットの一覧と `model_inputs`, 群の要約 (`describe_groups`), 介入の記録の要約 (`describe_uplift`), 値段を変えた記録の要約 (`describe_price`), 需要の記録の要約 (`describe_demand`) |
| ML-Quest (問題の設定) | データの読み方 (`Data.source`), レシピの雛形と書き戻し, ファイルの組の書き出しの設定. 手元のデータの答えの候補 |
| ML-CreateArchitecture (モデルの作成) | `model_inputs` でモデルの定義の入力と出力をデータセットに合わせる |
| ML-Found | `task_from_dataset` が `dataset.yaml` を読む (`ml_eda` は使わない). 評価の軌跡 (`trajectories/`) は ML-EDA が読む |
| ML-Inference, ML-Provide | 書き出したデータセットの読み込み (`load_dataset`, `load_items`), 推論のときの同じ前処理, 書き出し (`export_dataset` など). ML-Inference は群の要約の形 (`describe_groups`) も使う |

## 導入

```bash
git clone https://github.com/victor-von-pooh/ML-EDA.git
cd ML-EDA
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"     # parquet, Excel, 画像, 文の部分語, WAVE の他の音声も使う場合 (使うだけなら pip install -e . でもよい)
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

`profile` の結果は `overview` (行数, 重複など), `columns` (列ごとの型と統計), `correlations` (数値の列の相関), `associations` (カテゴリの列の関連), `target` (目的の列の分布と, 各列との関連の強さの順位), `timeseries` (時系列の間隔, 抜け, 周期の候補), `anomaly` (`task="anomaly"` の場合の, 異常の正解の付いた行と異常の行の数と割合), `findings` (気付き).

日時の列を持つ表は時系列として読み, 時刻の順に並べる (`time_column` で列を指定, `kind="table"` で時系列にしない). 読み込みの設定は `load_data(path, sep=";", encoding="cp932")` のように渡す. 画像, 文, 点群, 撮影, 音声のディレクトリの中の, 読み込んだディレクトリの外を指すシンボリックリンクは既定で読まず, 理由を表の `error` に残す (共有の保存先へのリンクを読むには `load_data(path, follow_links=True)`. 版 2 の書き出しと `normalize_capture` も既定で外を写さず, 引数 `follow_links=True` でたどる).

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

分割は `random`, `stratified` (目的の列の割合を保つ), `time` (時刻の順. 最後の部分を検証用とテスト用に), `group` (`column` の値ごとに同じ分割), `group_time` (`column` の値ごとに同じ分割にし, 後から現れた値を検証用とテスト用に), `stratified_group` (`column` の値ごとに同じ分割にし, 目的の列の割合も近くする), `none`.

目的の列の欠けた行は既定 (`unlabeled: drop`) で落とす. 分割に `unlabeled: keep` を書くと, 落とさずに `unlabeled` の分割 (`unlabeled.parquet`) に残す (半教師あり学習や異常の検知のため). 前処理の値は学習用だけで当てはめ, 正解の無い行には推論のときと同じようにかける. `tensors("unlabeled")` は特徴量だけを返す.

異常の検知 (正常の行が多く, 異常の正解が無いか少しだけある表) は `task: anomaly` で書く. 正解の列 (`target`) は異常を 1 か真, 正常を 0 か偽にし, 正解の無い表は書かない (全ての行を正常と見なす). 分割は `random` だけで, 学習用には正常の行だけを置き, 異常の行は検証用とテスト用に配る. `unlabeled: normal` は正解の欠けた行を正常と見なす. レシピの `id` に行の id の列を書くと, 全ての分割に元の値のまま残る (書かなければ行の位置の `row_id`). `dataset.yaml` の `anomaly` に分割ごとの正常, 正常と見なした行, 異常の行の数が入る.

```yaml
task: anomaly
target: is_anomaly
id: reading_id
steps:
  - {op: standardize, columns: auto}
split: {method: random, valid: 0.2, test: 0.2, seed: 0, unlabeled: normal}
```

効果を見積もる (クーポンを配ったときに顧客ごとに買う確率や額がどれだけ上がるか) は `task: uplift` で書く. 処置は無作為に配った記録を前提にする. 結果の列 (`target`) と処置の列 (`treatment`. 配ったら 1 か真, 配らなければ 0 か偽) を必ず書き, 結果の種類 (`outcome`) は `binary` (買ったかの 0 と 1) か `amount` (買った額. 書かなければ値から決める). 分割は `stratified` だけで, 処置 (`binary` は処置と結果の組) で層を作り, 全ての分割に処置の群と対照の群を置く. 処置の列は学習に使う列から除き, 全ての分割に元の値の 0 と 1 で残る (`dataset.treatment(split)`). 行の id は異常の検知と同じ. `dataset.yaml` の `uplift` に分割ごとの群の行の数, 結果の平均と差が入る.

```yaml
task: uplift
target: bought
treatment: coupon
outcome: binary
id: customer_id
steps:
  - {op: standardize, columns: auto}
split: {method: stratified, valid: 0.2, test: 0.2, seed: 0}
```

```python
from ml_eda import describe_uplift

summary = describe_uplift("coupons.csv", "coupon", "bought", id="customer_id")
summary["groups"]      # treated と control の rows と mean (結果の平均)
summary["effect"]      # 結果の平均の差 (処置の群 - 対照の群)
summary["balance"]     # 列ごとの標準化した平均の差 (smd) の大きい順
summary["findings"]    # small_group, treatment_imbalance, post_treatment など
```

値段を変える判断 (値段を変えたときの売れた数を予測し, 候補の中から利益の大きい値段を選ぶ) は `task: price` で書く. 値段は試しに変えた記録 (無作為に近い) を前提にする. 売れた数の列 (`target`) と値段の列 (`price`) を必ず書き, 原価の列 (`cost`) と商品の id の列 (`product`) は任意. 値段の列は学習に使う列の最後に入り, 値段の列を変えられる手順は `clip`, `log`, `standardize`, `minmax` だけ. 全ての分割に手順の前の値段 (`price_raw`. `dataset.price(split)`), 原価 (`dataset.cost(split)`), 商品の id が元の値で残る. 分割は `stratified` だけで, 値段の順位で 5 つに分けた段で層を作る. 候補の値段での入力は `dataset.price_inputs(split, prices)` で得る (値段の列だけを, 学習用で当てはめた手順をかけた候補に置き換える). 行の id は異常の検知と同じ. `dataset.yaml` の `price` に分割ごとの値段の範囲, 売れた数と原価の平均が入る.

```yaml
task: price
target: units
price: price
cost: unit_cost
product: product_id
steps:
  - {op: log, columns: [price]}
  - {op: standardize, columns: auto}
split: {method: stratified, valid: 0.2, test: 0.2, seed: 0}
```

```python
from ml_eda import describe_price, load_dataset

summary = describe_price("sales.csv", "price", "units", cost="unit_cost",
                         product="product_id")
summary["price_stats"]  # 値段の範囲, 種類の数, 幅
summary["relation"]     # 順位の相関, 弾力性の目安, 値段の段ごとの売れた数
summary["products"]     # 商品ごとの試した値段の数と範囲
summary["findings"]     # single_price_products, narrow_price_range など

dataset = load_dataset("datasets/sales")
x = dataset.price_inputs("test", dataset.price("test") * 1.1)  # 1 割上げた値段
```

需要の予測 (先の期間の需要を分位点で予測し, 損の比から決まる分位点で発注の数を決める) は `task: demand` で書く. 日ごとの売れた数の列 (`target`), 商品の id の列 (`product`), 日付の列 (`date`) を必ず書き, 在庫の数の列 (`stock`) は任意. 書き出す 1 行は (商品, 発注を決める日) で, 発注を決める日は最も早い日から `interval` 日ごと (既定は `horizon`). 目的の値は発注を決める日から `horizon` 日 (既定 7) の売れた数の合計 `demand`, 比べる相手は前の `horizon` 日の合計 `demand_previous` (`dataset.previous(split)`). 抜けの日を含む期間は合計を作らない. 在庫の数は発注を決める日の値が残る (`dataset.stock(split)`). 過去の売れた数は値を学ぶ手順より前の `lag` で日ごとの行から作る. 分割は `time` だけで, 先の期間が分割の境目をまたぐ行は落とす. `dataset.yaml` の `demand` に分割ごとの日の範囲, 合計の平均, 先の期間に在庫が 0 の日を含む行の数が入る.

```yaml
task: demand
target: units
product: product_id
date: date
stock: stock
horizon: 7
steps:
  - {op: lag, columns: [units], lags: [1, 7], windows: [7, 28],
     by: [product_id], time: date, freq: 1D}
  - {op: datetime_parts, columns: [date], parts: [weekday, month]}
  - {op: standardize, columns: auto}
split: {method: time, valid: 0.2, test: 0.2}
```

```python
from ml_eda import describe_demand, load_dataset

summary = describe_demand("daily_sales.csv", "product_id", "date", "units",
                          stock="stock", horizon=7)
summary["weekdays"]  # 曜日ごとの売れた数の平均
summary["products"]  # 商品ごとの記録の日数, 抜けの日, 在庫が 0 の日
summary["findings"]  # date_gaps, short_history, stockout_days など

dataset = load_dataset("datasets/daily_sales")
x, y = dataset.arrays("test")         # y は先の期間の売れた数の合計
baseline = dataset.previous("test")   # 前の期間と同じ数を発注する決め方
stock = dataset.stock("test")         # 発注を決める日の在庫の数
```

### 画像, 文, 点群, 音声

画像のディレクトリ (`<クラス>/<画像>` ならラベル付き), 1 行 1 文の `.txt` か `<クラス>/<文>.txt`, 点群 (`.ply`, `.pcd`, `.xyz`, `.npy`. 1 つか `<クラス>/<点群>`) も `load_data` で読める. 表 (`data.frame`) は 1 行 1 件の様子 (画像なら幅, 高さ, ラベル) で, 中身は `data.load_item(i)` で読む. 分析には大きさのばらつきなどの気付きが加わる.

音声 (`.wav` の PCM と 32 bit の浮動小数. 1 つか `<クラス>/<音声>`) も `load_data` で読める (種類 `audio`). FLAC, MP3, Ogg (Vorbis, Opus), AIFF は soundfile が要る (`[all]` か `[audio]` で入る). 表は標本化の周波数, チャンネル数, 長さなどで, `data.load_item(i)` か `read_audio(ファイル)` が `samples` ((標本の数, チャンネル) の float32) と `sample_rate` を返す. 先頭の秒だけは `data.load_item(i, seconds)` で読める. 分析には周波数のばらつき, 長さ, 無音と音の割れの気付きが加わる.

学習用にするには, レシピに中身の手順 (`item_steps`) を書く.

```yaml
target: label
task: classification
item_steps:
  - {op: resize, width: 64, height: 64}   # 画像. 文は {op: vocabulary, length: 128}
  - {op: normalize}                       # 点群は sample_points と normalize_cloud
split: {method: stratified, valid: 0.2}
```

音声は, 標本化の周波数, チャンネル, 長さをそろえる.

```yaml
target: label
task: classification
item_steps:
  - {op: resample, rate: 16000}           # method: sinc (既定) か linear
  - {op: mix_channels, channels: 1}
  - {op: fix_length, seconds: 1.0}        # align: start (既定) か center
steps:
  - {op: filter_rows, column: error, compare: isna}   # 読めない件を除く
split: {method: stratified, valid: 0.2}
```

文の区切り方 (文字 `char`, 語 `word`, 部分語 `subword`) は, 分析の結果の `result["items"]["tokenizers"]` が見本の文で比べて勧める (語彙の大きさ, 1 つの文のトークンの数, 語彙に無いトークンの割合). `recommended` をそのまま `item_steps` に入れられる. 部分語は `tokenizers` が要る (`[all]` か `[text]` で入る).

`load_dataset(...).tensors("train")` は (画像なら (件数, チャンネル, 高さ, 幅), 音声なら (件数, チャンネル, 標本の数)) の配列とラベルを返す.

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

モデルに要るもの (入力の形と型, 出す値の数. 文なら語彙の大きさ `vocabulary` も) は, 分割のデータを読まずに `model_inputs` で分かる.

```python
from ml_eda import model_inputs

model_inputs("datasets/titanic")
# {"inputs": [{"name": "x", "shape": [None, 10], "dtype": "float32"}],
#  "task": "classification", "target": "Survived",
#  "classes": ["0", "1"], "target_kind": None, "outputs": 2}
```

`outputs` は分類ならクラスの数, 回帰と時系列の予測なら 1. ファイルの組 (版 2 と 3) では, 画像と領域は `[None, チャンネル, 高さ, 幅]` の入力と背景を含む領域のクラスの数, 撮影と点群は `[None, 入力の点の数, 3]` の入力と出す点の数を返す (docs/DESIGN.md の「モデルの入力」).

### 入力と正解がファイルの組 (版 2 と 3)

画像と領域, 多視点の撮影と点群のように, 正解が列でなくファイルのデータは `dataset.yaml` の版 2 で書き出し, `load_items` で読む (列の正解のデータは版 1). 詳しくは [docs/DESIGN.md](docs/DESIGN.md) の「書き出しの版 2」.

```python
from ml_eda import export_capture_clouds, export_image_masks, load_items

export_image_masks("raw/images", "raw/masks", "datasets/shapes",
                   classes={0: "background", 1: "object"},
                   origin={"type": "annotation", "name": "塗り分け"})
dataset = load_items("datasets/shapes")
item = dataset.split("train")[0]     # {"id", "image", "mask", ...} の配列
paths = dataset.split("train", read=False)[0]   # 配列の代わりに場所
```

正解のファイルの無い入力 (領域の無い画像など) は既定で除く. `split={"unlabeled": "keep"}` を渡すと `unlabeled` の分割に書き, `load_items` はその件の正解を None で返す.

撮影と点群の対 (`export_capture_clouds`) は, 姿勢と点群に同じ平行移動 (姿勢から求めた物の中心を原点に) をかけ, 実寸 (メートル) を残す. 深度の測り方 (`depth: z`) と画素の中心 (`pixel_center: 0.5`) は説明の `entries.capture.convention` に書く. 推論のときは, 新しい撮影に `normalize_capture` で同じ正規化をかけ, `read_capture` で `load_items` と同じ形に読む (点群は `read_cloud`). 確かめ用の見本は `ml_eda.synthetic` (`write_image_masks`, `write_shape_captures`. `count` と `seed` で形の数をばらつかせる) で作れる.

```python
from ml_eda import normalize_capture, read_capture

summary = normalize_capture("new/capture_001", "work/capture_001")
capture = read_capture("work/capture_001")   # intrinsics, poses, depth など
# 予測した点群 - summary["shift"] で元の座標に戻す
```

書き出しの `item_steps` に項目ごとの中身の手順を渡すと, 大きさをそろえるなどをかけたファイルを書き, かけた手順を `dataset.yaml` の `entries.<名前>.steps` に残す (版 3). 画像の `resize` には, 領域に同じ大きさの `resize_mask` (最も近い画素. 値を混ぜない) を書き出しがかける. 撮影の `resize` は深度と内部パラメータも合わせて直す. 推論のときは, 新しい 1 件に `transform_entry` (撮影は `normalize_capture` の `steps`) で同じ手順をかける.

```python
from ml_eda import transform_entry

export_image_masks("raw/images", "raw/masks", "datasets/sized",
                   item_steps={"image": [
                       {"op": "resize", "width": 128, "height": 128}]})
dataset = load_items("datasets/sized")
image = transform_entry(dataset, "image", "new/photo.png")   # (128, 128, 3)

export_capture_clouds("raw/captures", "raw/clouds", "datasets/scenes",
                      item_steps={
                          "capture": [{"op": "resize", "width": 256,
                                       "height": 192}],
                          "cloud": [{"op": "sample_points", "points": 4096}]})
steps = load_items("datasets/scenes").entries["capture"]["steps"]
normalize_capture("new/capture_001", "work/capture_001", steps=steps)
```

### 記録した軌跡

```python
from ml_eda import describe_trajectory, load_trajectory

# run_dir は ML-Found の実験のディレクトリ
trajectory = load_trajectory(run_dir / "trajectories" / "valid_best.csv")
trajectory.frame                      # 1 歩が 1 行の表 (pandas の DataFrame)
trajectory.observation_columns        # ["obs_0", "obs_1", ...] (要素の順)
summary = describe_trajectory(trajectory)   # JSON にできる辞書
summary["return"]                     # エピソードごとの報酬の和の min, max, mean など
summary["per_episode"]                # エピソードごとの長さ, 報酬の和, 終わり方
summary["guard"]                      # 守りの結果と理由の数 (守りの列のある表だけ)
summary["action_prob"]                # 行動を選んだ確率の欠けと範囲の外の数, 範囲の中の値の min, max, mean など (確率の列のある表だけ)
```

隣に説明 (`valid_best.json`) があれば型と環境の名前などを使い, 無ければ列の名前 (`obs`, `obs_<番号>`, `action`, `action_<番号>`) と値から読む. 手元の Parquet (`trajectory.parquet`) も同じ口で読む (pyarrow が要る). 列の決まりは [docs/DESIGN.md](docs/DESIGN.md) の「記録した軌跡」.

### 並べるの反応の表

```python
from ml_eda import describe_reactions, load_reactions

# 列の名前は問題の設定で答えたもの (反応の強さは任意)
reactions = load_reactions("interactions.csv", user="member",
                           item="item_id", time="ts", reaction="qty")
reactions.frame                       # user, item, time, reaction の列 (時刻の順)
summary = describe_reactions(reactions)     # JSON にできる辞書
summary["users"], summary["items"], summary["reactions"]
summary["dropped"]                    # 欠けた行と, 強さが 0 以下か数でない行の数
summary["per_user"]                   # 利用者ごとの反応の数の min, max, 区切りごとの数など
summary["popularity"]                 # 上位 10% の品物の反応の割合, ジニ係数, 人気の品物
summary["time"]                       # 時刻の型と範囲
```

ファイルは `load_data` と同じ読み方で読む (`loader` と読み込みの設定も渡せる. `load_data` で読んだ `Data` も渡せる). 時刻は数か日時. 利用者, 品物か時刻の欠けた行と, 反応の強さが 0 以下か数でない行は除く (ML-Quest の並べるの骨組みと同じ決まり). 反応の表はファイル 1 つのまま骨組みに渡し, 分割は骨組みが時刻で行う. 決まりは [docs/DESIGN.md](docs/DESIGN.md) の「並べるの反応の表」.

### 群分けの表

```python
from ml_eda import describe_groups, export_group_rows, load_items

# 群を作る列は問題の設定で答えたもの (行を見分ける列は任意)
columns = ["spend", "visits", "region"]
summary = describe_groups("customers.csv", columns, k=3)  # JSON にできる辞書
summary["groups"]                     # 群ごとの rows, share, stability と列の平均か最も多い値. 最後は「全体」
summary["stability"]                  # 数え方 (method) で作り直した群との一致 (ari, ari_min)
summary["silhouette"]                 # 群のまとまり
describe_groups("customers.csv", columns, k=3, stability="sample")  # 既定の seed は種を変える. sample は行の 8 割を引き直して作り直す数え方

# 分析の保存として版 2 のデータセットに書き出して読む (受け持ちは group, 正解は無い. 学習には使わない)
export_group_rows("customers.csv", "datasets/customers", columns, id="cid")
dataset = load_items("datasets/customers")
dataset.split("train")[0]             # {"id", "split", "group", "spend", "visits", "region"}
describe_groups(dataset, k=3)         # 書き出したデータセットからも同じ要約
```

群は ML-Quest のまとめるの骨組みと同じ符号化 (数の列は標準化, 他の列は値ごとの 0 と 1) をかけた列の特徴を, そのまま k-means で分けたもの (骨組みの比べる相手と同じ作り方). 安定性の数え方 `stability` は ML-Quest の問題の設定の答え `group.stability` と同じ `seed` (種を変える. 既定) か `sample` (行を引き直す. 引く割合は `fraction`) で, 骨組みが学習の後に数える数え方と同じ. `labels` に行ごとの群か群の列の名前を渡すと, その群を要約する. 群の要約の行は骨組みの `evaluations/group_summary.json` の行と同じ形. 版 2 の書き出しは分析の保存で, 群分けの学習 (ML-Quest のまとめるの骨組み) には元の表のファイルを渡す. 決まりは [docs/DESIGN.md](docs/DESIGN.md) の「群の要約」と「群分けの表」.

## 文書

- [docs/DESIGN.md](docs/DESIGN.md): 今の設計と決まり, 他のリポジトリとの約束 (レシピとデータセットの形と版, 読む軌跡と反応の表の形, 群の要約の形)
- [TODO.md](TODO.md): まだやっていないこと (優先度の順)
- [CODING_STYLE.md](CODING_STYLE.md): コードの書き方
- [CLAUDE.md](CLAUDE.md): このリポジトリで作業するときの方針

## ライセンス

MIT
