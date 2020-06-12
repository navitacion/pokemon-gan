# Pokemon GAN

## 概要

ポケモンの画像を使ってDCGAN, SAGANを実行してみた

出力はグレースケール・64×64

それぞれの結果をstreamlitを用いてアプリ化した


## 前提

poetryがインストール済み


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


## 使用データ＆学習

「Pokemon Images Dataset」（Kaggle Datasetより）

[Pokemon Images Dataset | Kaggle](https://www.kaggle.com/kvpratama/pokemon-images-dataset)

学習する時は上のURLからデータをダウンロードし、

data/pokemonに格納したのち、下記を実行する

```
poetry run python train.py
```

-gan_typeで学習するGANを指定できる
