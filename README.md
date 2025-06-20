# ADKではじめる！RAGエージェント構築チュートリアル

このドキュメントは、ADK (Agent Development Kit) を使用して、RAG (Retrieval Augmented Generation) の基本的な仕組みを学びながら、初めてのRAGエージェントを構築するためのチュートリアルです。

環境構築からエージェントの実行までをステップバイステップで説明します。

## 1. uv のインストール

プロジェクトのパッケージ管理には `uv` を使用します。

- **公式ドキュメント:** [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **GitHubリポジトリ:** [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

お使いのOSに合わせて、以下のコマンドを実行してください。

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. uv の初期化とライブラリの追加

プロジェクトディレクトリで以下のコマンドを実行し、`uv` を初期化して必要なライブラリをインストールします。

```bash
uv init
# ADK本体 (必須)
uv add google-adk

# 埋め込みモデル用ライブラリ (どちらか一方、または両方を選択)
# Google Gemini を使用する場合
uv add google-generativeai

# OpenAI API を使用する場合
uv add openai
```

## 3. APIキーの設定 (.envファイル)

APIキーを管理するために、プロジェクトのルートディレクトリに `.env` ファイルを作成します。
まず、`.env.example` ファイルをコピーして `.env` ファイルを作成してください。

作成した `.env` ファイルを編集します。

以下のサイトからAPIキーを取得してください。

- **Google AI Studio (必須):** [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- **OpenAI Platform (OpenAI API を使用する場合のみ):** [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

取得したAPIキーを `.env` ファイルに以下のように記述します。

```env
# .env ファイルの例

# Google API を利用するための設定
GOOGLE_GENAI_USE_VERTEXAI="False"
GOOGLE_API_KEY="ここにあなたのGoogle APIキーを入力してください"

# OpenAI API を利用する場合の設定 (任意)
# OPENAI_API_KEY="ここにあなたのOpenAI APIキーを入力してください"
```

## 4. プロジェクトのフォルダ構成 (例)

基本的なフォルダ構成は以下の通りです。

```text
.env
agent/
├── __init__.py
└── agent.py
README.md
```

## 5. エージェントの実行

環境構築が完了したら、以下のいずれかの方法でエージェントを実行できます。

### 1) 検索関数のテスト実行

`agent.py` に実装されている検索関数 (`retrieve` や `get_random_chunks`) の動作を個別に確認できます。

```bash
uv run --env-file .env python .\agent\agent.py
```

### 2) AIエージェントの起動 (Webインターフェース)

ADKのWebインターフェースを起動し、ブラウザ経由でAIエージェントと対話します。

```bash
uv run adk web
```

Webインターフェースが起動したら、表示されるURLにブラウザでアクセスしてください。
