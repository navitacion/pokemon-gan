# Pokemon GAN

## 概要
GANを用いて新しいポケモンを生み出すWebアプリです。

DCGAN, SAGANを試してみました。

## 前提
poetryがインストールされていること

MacあるいはLinuxであること

## 使用データ
「Pokemon Images Dataset」（Kaggle Datasetより）

[Pokemon Images Dataset | Kaggle](https://www.kaggle.com/kvpratama/pokemon-images-dataset)

## 仕様
出力画像：RGB；128×128

モデル：DCGAN, SAGAN

WebアプリではEpoch数0 ~ 1,000回の結果を40ごとに表示できるようにしています。

## 利用方法

* 環境の構築

以下を実行し、必要なライブラリを読み込む

```
poetry install
```

* アプリの起動

以下を実行

```
poetry run streamlit run app.py
```

## 学習方法
[Pokemon Images Dataset | Kaggle](https://www.kaggle.com/kvpratama/pokemon-images-dataset)

上のURLからデータをダウンロードし、data/pokemonに格納します。

その後、下記を実行します。

```
poetry run python train.py
```

-gan_typeで学習するGANを指定できます。