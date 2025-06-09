import os
import json
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# 環境変数の読み込み
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

print("=== GROQ_API_KEY確認 ===")
api_key = os.getenv("GROQ_API_KEY")
print(f"APIキー: {api_key[:6]}...{api_key[-6:] if api_key else 'Not found'}")

client = Groq(api_key=api_key)

try:
    print("\n=== モデルリスト取得テスト ===")
    models = client.models.list()
    print(f"取得モデル数: {len(models.data)}")
    scout_exists = any(m.id == "meta-llama/llama-4-scout-17b-16e-instruct" for m in models.data)
    print(f"Scoutモデル存在: {'Yes' if scout_exists else 'No'}")
    
    print("\n=== 最小リクエストテスト ===")
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print(f"レスポンス: {completion.choices[0].message.content}")
    
    print("\n=== アプリケーション同等リクエストテスト ===")
    lang_example = json.dumps({"lang": "ch"})
    prompt = f"""
Please determine what language the document below is in? English (en) or Chinese (ch)?

<document>
{"This is a test document"}
</document>
    
Use JSON format with key `lang` when return result. Please only output the result in json format, and do the json format check and return, don't include other extra text! An example of output is as follows:
Output example: {lang_example}
""".strip()
    
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "{"}
    ]
    
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        max_tokens=1000,
        temperature=0.8,
    )
    print("レスポンス内容:")
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"\n!!! エラー発生: {str(e)}")
