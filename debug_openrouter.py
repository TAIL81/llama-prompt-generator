import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

import argparse

def main():
    # コマンドライン引数でモデルID指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                       default='microsoft/mai-ds-r1:free',
                       help='OpenRouter model ID (e.g. "microsoft/mai-ds-r1:free", "deepseek/deepseek-r1-0528:free")')
    args = parser.parse_args()

    # 環境変数読み込み
    env_path = "d:/Users/onisi/Documents/AI/llama-prompt-generator/src/.env" # 絶対パスを直接指定
    
    if not os.path.exists(env_path):
        print(f"❌ 致命的エラー: .envファイルが見つかりません: {env_path}")
        return # スクリプトを終了

    load_dotenv(dotenv_path=env_path)
    
    # API設定
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    print(f"🔍 環境変数チェック: OPENAI_API_KEY={'***' if not OPENAI_API_KEY else OPENAI_API_KEY[:3] + '...'}")
    print(f"📁 .envファイルパス: {env_path}") # パス表示を追加
    print(f"✅ .envファイルを発見: {env_path}")
    print(f"🛠️ 使用モデル: {args.model}")

    # 最小テストリクエスト
    try:
        print("⚡ OpenRouter APIにリクエスト送信中...")
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "こんにちは！日本語で簡単な挨拶を返してください。"}]
        )
        print("\n✅ APIレスポンス成功!")
        print(f"レスポンスタイプ: {type(response)}")
        print(f"選択肢数: {len(response.choices)}")
        if response.choices:
            msg = response.choices[0].message
            print(f"メッセージ構造: {type(msg)}")
            print(f"コンテンツ: '{msg.content}'")
    except Exception as e:
        print(f"\n❌ エラー発生: {str(e)}")
        if not OPENAI_API_KEY:
            print("⚠️ 原因: OPENAI_API_KEYが設定されていません")
        elif "401" in str(e):
            print("⚠️ 原因: 無効なAPIキー")
        elif "404" in str(e):
            print("⚠️ 原因: 不正なエンドポイントURL")

if __name__ == "__main__":
    main()
