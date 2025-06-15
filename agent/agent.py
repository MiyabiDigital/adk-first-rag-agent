# -----------------------------------------------------------------------------
# ライブラリのインポート
# -----------------------------------------------------------------------------
from google.adk.agents import Agent
from pathlib import Path
import os
import pickle
import random
from typing import List, Tuple, Optional

# -----------------------------------------------------------------------------
# 型定義と設定
# -----------------------------------------------------------------------------
# --- 型エイリアス ---
# 「Embedding」は、テキストの意味を表現する浮動小数点数のリスト（ベクトル）です。
Embedding = List[float]
# 「VectorDBEntry」は、元のテキストチャンクとそのベクトル表現のペアです。
VectorDBEntry = Tuple[str, Embedding]
# 「VectorDBType」は、VectorDBEntryのリストで、私たちのベクトルデータベース全体を表します。
VectorDBType = List[VectorDBEntry]

# --- LLMプロバイダー設定 ---
# Trueに設定するとGoogle Generative AIを、Falseに設定するとOpenAIを使用します。
USE_GOOGLE_GENAI = False

# --- グローバル変数 ---
# ベクトルデータベースを保存するファイル名。使用するAPIによって名前が変わります。
VECTOR_DB_FILENAME = f"vector_db_{'google' if USE_GOOGLE_GENAI else 'openai'}.pkl"
VECTOR_DB_FILEPATH = Path(__file__).resolve().parent / VECTOR_DB_FILENAME

# ベクトルデータベース本体。起動時にファイルから読み込むか、新規作成されます。
VECTOR_DB: VectorDBType = []

# APIクライアントの初期化
# USE_GOOGLE_GENAI の値に応じて、使用するAPIクライアントを準備します。
if USE_GOOGLE_GENAI:
    try:
        import google.generativeai as genai
        # 環境変数 `GOOGLE_API_KEY` を使って認証します。
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("環境変数 'GOOGLE_API_KEY' が設定されていません。")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    except ImportError:
        print("エラー: 'google.generativeai' パッケージが見つかりません。")
        print("インストールしてください: uv add google-generativeai")
        exit()
    except Exception as e:
        print(f"エラー: Google Generative AI の初期化に失敗しました: {e}")
        exit()
else:
    try:
        from openai import OpenAI
        # 環境変数 `OPENAI_API_KEY` を使って認証します。
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("環境変数 'OPENAI_API_KEY' が設定されていません。")
        openai_client = OpenAI()
    except ImportError:
        print("エラー: 'openai' パッケージが見つかりません。")
        print("インストールしてください: uv add openai")
        exit()
    except Exception as e:
        print(f"エラー: OpenAIクライアントの初期化に失敗しました: {e}")
        exit()

# -----------------------------------------------------------------------------
# ベクトルデータベースのヘルパー関数
# -----------------------------------------------------------------------------
def load_vector_db() -> Optional[VectorDBType]:
    """ベクトルデータベースをファイルから読み込みます。"""
    if VECTOR_DB_FILEPATH.exists():
        print(f"情報: 既存のベクトルDB '{VECTOR_DB_FILENAME}' を読み込みます...")
        try:
            with open(VECTOR_DB_FILEPATH, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"警告: DBファイル '{VECTOR_DB_FILENAME}' の読み込みに失敗しました。ファイルが破損している可能性があります。エラー: {e}")
            return None
    return None

def save_vector_db(db: VectorDBType) -> None:
    """ベクトルデータベースをファイルに保存します。"""
    print(f"情報: ベクトルDBを '{VECTOR_DB_FILENAME}' に保存しています...")
    with open(VECTOR_DB_FILEPATH, 'wb') as f:
        pickle.dump(db, f)

def get_embedding(text: str, is_query: bool = False) -> Embedding:
    """指定されたテキストのベクトル表現（Embedding）を計算して返します。"""
    try:
        if USE_GOOGLE_GENAI:
            # Googleのモデルは用途（検索用クエリか、DB格納用ドキュメントか）でタスクタイプを分ける
            task_type = "retrieval_query" if is_query else "retrieval_document"
            return genai.embed_content(model="models/text-embedding-004",
                                     content=text,
                                     task_type=task_type)["embedding"]
        else:
            # OpenAIのモデルはタスクタイプの区別は不要
            response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
            return response.data[0].embedding
    except Exception as e:
        print(f"エラー: ベクトル化（Embedding）の取得中にエラーが発生しました: {e}")
        print("APIキーが正しいか、ネットワーク接続を確認してください。")
        # エラーが発生した場合は、プログラムを終了するか、適切なデフォルト値を返す
        exit()

def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """2つのベクトル間のコサイン類似度を計算します。値は-1から1の範囲で、1に近いほど類似していることを意味します。"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    # ゼロ除算を避けるための微小値
    return dot_product / ((norm_a * norm_b) + 1e-9)

# -----------------------------------------------------------------------------
# データベースの構築処理
# -----------------------------------------------------------------------------
def build_database():
    """データソースからテキストを読み込み、ベクトルデータベースを構築します。"""
    global VECTOR_DB
    script_dir = Path(__file__).resolve().parent
    source_file_path = script_dir.parent / 'sample.txt'

    try:
        print(f"情報: データソース '{source_file_path.name}' を読み込んでいます...")
        with open(source_file_path, 'r', encoding='utf-8') as file:
            # ファイルを1行ずつ読み込み、各行をチャンクとする
            dataset = file.readlines()
            print(f"情報: {len(dataset)}個のチャンクを読み込みました。")
    except FileNotFoundError:
        print(f"エラー: データソースファイル '{source_file_path}' が見つかりません。プログラムを終了します。")
        exit()

    print("情報: 新規にベクトルDBを構築します。これには時間がかかる場合があります...")
    new_db = []
    for i, chunk_text in enumerate(dataset):
        chunk_text = chunk_text.strip()
        if not chunk_text: continue # 空行はスキップ
        
        embedding = get_embedding(chunk_text, is_query=False)
        new_db.append((chunk_text, embedding))
        print(f'  チャンク {i+1}/{len(dataset)} をベクトル化し、DBに追加しました。')
    
    VECTOR_DB = new_db
    save_vector_db(VECTOR_DB)
    print("情報: ベクトルDBの構築が完了しました。")

# --- メインの初期化処理 ---
VECTOR_DB = load_vector_db()
if VECTOR_DB is None:
    build_database()
else:
    print(f"情報: ベクトルDBを '{VECTOR_DB_FILENAME}' から正常にロードしました。エントリ数: {len(VECTOR_DB)}")

# -----------------------------------------------------------------------------
# ADKエージェントのツール定義
# -----------------------------------------------------------------------------
def retrieve(query: str, top_n: int = 3) -> List[Tuple[str, float]]:
    """
    質問（クエリ）を受け取り、データベース内で最も関連性の高い知識（チャンク）を
    類似度スコアと共に上位N件検索して返します。

    Args:
        query (str): 検索したい質問やキーワード。
        top_n (int): 取得する上位チャンクの数。デフォルトは3。

    Returns:
        (チャンク文字列, 類似度スコア) のタプルのリスト。
    """
    if not VECTOR_DB:
        return [("データベースが空です。", 0.0)]

    # 1. 質問文をベクトルに変換する
    query_embedding = get_embedding(query, is_query=True)

    # 2. DB内の全知識と類似度を計算する
    similarities = []
    for chunk, chunk_embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((chunk, similarity))

    # 3. 類似度が高い順に並べ替える
    similarities.sort(key=lambda item: item[1], reverse=True)

    # 4. 上位N件の結果を返す
    return similarities[:top_n]

def get_random_chunks(count: int = 3) -> List[str]:
    """
    データベースからランダムにいくつかの知識（チャンク）を返します。
    データベースにどのような情報が入っているか概要を掴むのに役立ちます。

    Args:
        count (int): 取得するランダムなチャンクの数。デフォルトは3。

    Returns:
        ランダムに選ばれたチャンク（文字列）のリスト。
    """
    if not VECTOR_DB:
        return ["データベースは現在空です。"]
    
    all_chunks = [entry[0] for entry in VECTOR_DB]
    
    if len(all_chunks) <= count:
        return all_chunks
        
    return random.sample(all_chunks, count)

# -----------------------------------------------------------------------------
# ADKエージェントの定義
# -----------------------------------------------------------------------------
root_agent = Agent(
    name="rag_agent",
    model="gemini-2.0-flash",
    description="文書データベースに関する質問に日本語で回答するエージェントです。",
    instruction=(
        "あなたは、与えられた文書データベースの内容に基づいて質問に回答するアシスタントです。"
        "回答は、必ず `retrieve` ツールを使って検索した情報を根拠に作成してください。"
        "特定のトピックについて質問された場合は、`retrieve` ツールを使用してください。"
        "データベースにどのような情報があるか、といった曖昧な質問の場合は `get_random_chunks` ツールを使用してください。"
    ),
    tools=[retrieve, get_random_chunks]
)

# -----------------------------------------------------------------------------
# 動作確認用のテストコード
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n----------------------------------------")
    print("--- retrieve 関数の直接実行テスト ---")
    print("----------------------------------------")
    
    test_query = "深層学習の成功事例を教えてください"
    print(f"\nクエリ: 「{test_query}」")
    
    retrieved_knowledge = retrieve(test_query, top_n=3)
    
    print('\n検索結果:')
    if retrieved_knowledge:
        for i, (chunk, similarity) in enumerate(retrieved_knowledge):
            print(f'  {i+1}. (類似度: {similarity:.3f}) {chunk.strip()}')
    else:
        print("関連する知識は見つかりませんでした。")

    print("\n----------------------------------------")
    print("--- get_random_chunks 関数の直接実行テスト ---")
    print("----------------------------------------")
    random_chunks = get_random_chunks(count=2)
    print('\nランダムに取得された知識:')
    if random_chunks:
        for i, chunk in enumerate(random_chunks):
            print(f'  {i+1}. {chunk.strip()}')
    else:
        print("データベースは空か、チャンクを取得できませんでした。")
    print("\n")
