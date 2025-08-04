import os
from typing import Any, Dict, Generator, List

import gradio as gr
from dotenv import load_dotenv

# OpenAI 互換クライアント（公式 openai パッケージの v1 スタイル）
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()  # .env から環境変数を読み込み

# 環境変数の取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openrouter/horizon-beta")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")  # Azure などで使用

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY が設定されていません。`.env` に設定するか環境変数をセットしてください。"
    )

# クライアントの初期化
# Azure 互換など base_url 指定が必要な場合に対応
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

SYSTEM_PROMPT = (
    "Act as a helpful assistant. Think step by step. Reply in fluent Japanese."
)


def _convert_history_to_messages(
    history: List[List[str]],
) -> List[ChatCompletionMessageParam]:
    """
    gr.ChatInterface の history を OpenAI API が要求する形式に変換する。
    """
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages


def chat_reply(
    message: str, history: List[List[str]]
) -> Generator[str | None, Any, Any]:
    """
    Gradio ChatInterface 用のコールバック。
    ユーザー入力と履歴を受け取り、OpenAI 互換 API に問い合わせ、ストリーミングで返す。
    """
    messages = _convert_history_to_messages(history)
    messages.append({"role": "user", "content": message})

    # Azure の場合は API バージョンが必要
    extra_params = {}
    if OPENAI_API_VERSION:
        extra_params["api_version"] = OPENAI_API_VERSION

    try:
        # ストリーミングレスポンス
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
            temperature=0.7,
            **extra_params,
        )
        # チャンクを逐次連結して送る
        full_text = []
        for chunk in stream:
            delta = chunk.choices[0].delta  # OpenAI 互換の delta
            if delta and getattr(delta, "content", None):
                token = delta.content
                full_text.append(token)
                yield token
    except AuthenticationError:
        yield "[認証エラー] API キーまたは認可情報を確認してください。"
    except RateLimitError:
        yield "[レート制限] 少し時間をおいて再試行してください。"
    except (APITimeoutError, APIConnectionError):
        yield "[接続エラー] ネットワークまたはエンドポイントを確認してください。"
    except BadRequestError as e:
        yield f"[リクエストエラー] {getattr(e, 'message', str(e))}"
    except APIError as e:
        yield f"[サーバーエラー] {getattr(e, 'message', str(e))}"
    except Exception as e:
        yield f"[不明なエラー] {str(e)}"


def create_chat_tab():
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            bot_message_generator = chat_reply(message, chat_history)
            chat_history.append([message, ""])
            for token in bot_message_generator:
                chat_history[-1][1] += token
                yield "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
