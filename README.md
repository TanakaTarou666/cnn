## はじめに
本プロジェクトでは、単純な構造の CNN を実装します。

## 環境構築
Anaconda を使用します。以下のコマンドで仮想環境を作成し、アクティブ化してください。
Pytorchがcondaで入れられないので、パッケージのインストールは全てpipでお願いします。(ライブラリの競合を避けるため)

```bash
conda create --name [仮想環境名] python=3.12.0
conda activate [仮想環境名]
pip install -r requirements.txt
```

PyTorch のインストールについては、[公式サイト](https://pytorch.org/) を参照にコマンドを入手してください。

![Test Image 1](img/pytorch_install.png)

## データのダウンロード
以下のリンクからデモデータをダウンロードし、`./data/small-places/` となるように配置してください。

- デモデータ: https://drive.google.com/drive/folders/1IheiqhXHuR5DgX7-DY6Usc6sItROpYbL?usp=drive_link
- ※元データ: https://www.kaggle.com/datasets/benjaminkz/places365

## 実行方法
1. `config.py` に分類したいクラスのディレクトリを記述します。
2. 以下のコマンドを実行してください。   
    ```bash
    cd src
    python main.py
    ```