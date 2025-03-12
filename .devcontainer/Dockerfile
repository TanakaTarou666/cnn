FROM ubuntu:24.04

# 必要なパッケージをインストール
RUN apt-get update \
    && apt-get upgrade -y apt-utils curl vim python3.12 python3-pip python3.12-venv git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonの仮想環境を作成
RUN python3 -m venv /opt/venv

# 仮想環境をデフォルトに設定
ENV PATH="/opt/venv/bin:$PATH"

# 必要なファイルをコンテナにコピー
COPY requirements.txt .

# Pythonパッケージをインストール
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm requirements.txt

# PyTorchをインストール
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspaces