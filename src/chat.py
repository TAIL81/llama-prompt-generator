import os
from typing import Any, Generator, List

from dotenv import load_dotenv
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

class ChatService:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        self.model = os.getenv("OPENAI_MODEL", "openrouter/horizon-beta")
        self.api_version = os.getenv("OPENAI_API_VERSION")
        self.client = self._create_client()

    def _create_client(self) -> OpenAI | None:
        if not self.api_key:
            return None
        return OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def _convert_history_to_messages(
        self, history: List[List[str]]
    ) -> List[ChatCompletionMessageParam]:
        """
        gr.ChatInterface の history を OpenAI API が要求する形式に変換する。
        """
        messages: List[ChatCompletionMessageParam] = []
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    def chat_completion_stream(
        self,
        message: str,
        history: List[List[str]],
        system_prompt: str,
        temperature: float,
    ) -> Generator[str | None, Any, Any]:
        if not self.client:
            yield "[エラー] OpenAIクライアントが初期化されていません。APIキーを確認してください。"
            return

        # システムプロンプトを追加し、履歴を結合
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(self._convert_history_to_messages(history))
        messages.append({"role": "user", "content": message})

        # Azure の場合は API バージョンが必要
        extra_params = {}
        if self.api_version:
            extra_params["api_version"] = self.api_version

        try:
            # ストリーミングレスポンス
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=temperature,
                **extra_params,
            )
            # チャンクを逐次連結して送る
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    token = delta.content
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