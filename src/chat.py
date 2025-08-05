import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

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
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall

# ロガーの設定
# logging.basicConfig() は呼び出し元で設定することを想定し、ここでは削除。
# ライブラリとして利用される際に、利用側のロギング設定を上書きしないようにするため。
logger = logging.getLogger(__name__)


class ChatService:
    def _estimate_cost(self, usage: Dict[str, int]) -> float:
        """
        価格は 100万トークンあたりの USD を環境変数から受け取り、トークン数に比例計算。
        """
        prompt_rate = (
            self.prompt_price_per_million / 1_000_000.0
            if self.prompt_price_per_million > 0
            else 0.0
        )
        completion_rate = (
            self.completion_price_per_million / 1_000_000.0
            if self.completion_price_per_million > 0
            else 0.0
        )
        cost = (
            usage.get("prompt_tokens", 0) * prompt_rate
            + usage.get("completion_tokens", 0) * completion_rate
        )
        return float(cost)

    def _append_usage_log(
        self,
        success: bool,
        model: str,
        usage: Dict[str, int],
        cost_usd: float,
        system_prompt: str,
        history_count: int,
        error_message: Optional[str],
    ) -> None:
        """
        1 リクエストごとのメタ情報を JSONL で追記保存。
        """
        try:
            # ディレクトリ作成
            log_path = Path(self.usage_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "success": success,
                "model": model,
                "usage": usage,
                "cost_usd": round(cost_usd, 8),
                "system_prompt_sha256": hashlib.sha256(
                    (system_prompt or "").encode("utf-8")
                ).hexdigest(),
                "history_count": history_count,
                "error": error_message,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append usage log.")

    def __init__(
        self,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
        max_context_tokens: int = 8192,
    ):
        """
        ChatServiceを初期化します。

        引数:
        - max_retries: APIエラー時の最大再試行回数。
        - retry_backoff_base: 再試行時の指数バックオフの底。
        - max_context_tokens: モデルに送信する最大トークン数。
        """
        load_dotenv()

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        self.model = os.getenv("OPENAI_MODEL", "openrouter/horizon-beta")
        self.api_version = os.getenv("OPENAI_API_VERSION")

        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_backoff_base = float(
            os.getenv("RETRY_BACKOFF_BASE", retry_backoff_base)
        )
        self.max_context_tokens = int(
            os.getenv("MAX_CONTEXT_TOKENS", max_context_tokens)
        )

        self.client = self._create_client()
        # 概算トークン計測用の係数（文字数/トークン）
        self.chars_per_token = float(os.getenv("TOKEN_CHARS_PER_TOKEN", "4.0"))
        # コスト推定用（USD, 100万トークンあたり）
        self.prompt_price_per_million = float(
            os.getenv("PROMPT_PRICE_PER_MILLION", "0")
        )
        self.completion_price_per_million = float(
            os.getenv("COMPLETION_PRICE_PER_MILLION", "0")
        )
        # ログ出力先
        self.usage_log_path = os.getenv("USAGE_LOG_PATH", "logs/chat_usage.jsonl")

    def _create_client(self) -> Optional[OpenAI]:
        """
        OpenAI互換クライアントを作成して返します。
        APIキーが設定されていない場合はNoneを返します。
        """
        if not self.api_key:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.api_base)

    def _convert_history_to_messages(
        self, history: List[List[str]]
    ) -> List[ChatCompletionMessageParam]:
        """
        Gradioの履歴形式をOpenAIのメッセージ形式に変換します。
        """
        messages: List[ChatCompletionMessageParam] = []
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                # tool_callsを持つアシスタントメッセージも考慮する必要があるかもしれないが、
                # Gradioの基本形式では単純な文字列と仮定する。
                messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    def _estimate_tokens_from_text(self, text: str) -> int:
        """
        概算: 文字数 / chars_per_token でトークン数を見積もる。
        """
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def _num_tokens_from_messages(
        self, messages: List[ChatCompletionMessageParam]
    ) -> int:
        """
        OpenAI Cookbook のオーバーヘッド近似を維持しつつ、各文字列を概算でカウント。
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # メッセージごとのオーバーヘッド
            for key, value in message.items():
                if value and isinstance(value, str):
                    num_tokens += self._estimate_tokens_from_text(value)
                if key == "name":
                    num_tokens -= 1  # name がある場合の補正
        return num_tokens + 3  # リプライのプライミング

    def _trim_messages(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """
        メッセージ履歴がmax_context_tokensを超えないようにトリミングします。
        システムプロンプトと最新のユーザーメッセージは常に保持されます。
        履歴は古いものからペア（ユーザー/アシスタント）で削除されます。
        """
        # 先にトークン数を計算
        token_count = self._num_tokens_from_messages(messages)
        if token_count <= self.max_context_tokens:
            return messages

        # 各パーツを分離
        system_message = []
        if messages and messages[0]["role"] == "system":
            system_message = messages[:1]
            messages = messages[1:]

        # 最新のメッセージは常に保持
        latest_message = messages[-1:]
        history = messages[:-1]

        # 履歴を古いペアから削除していく
        while history:
            current_messages = system_message + history + latest_message
            token_count = self._num_tokens_from_messages(current_messages)
            if token_count <= self.max_context_tokens:
                break
            # 履歴の先頭から2件（ユーザーとアシスタントのペアを想定）を削除
            history = history[2:]

        final_messages = system_message + history + latest_message
        final_token_count = self._num_tokens_from_messages(final_messages)

        logger.info(
            f"Messages trimmed to {final_token_count} tokens to fit within the {self.max_context_tokens} limit."
        )
        return final_messages

    def chat_completion_stream(
        self,
        message: str,
        history: List[List[str]],
        system_prompt: str,
        temperature: float,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Generator[Dict[str, Any], Any, Dict[str, Any]]:
        """
        ストリーミングチャット補完リクエストを送信し、イベントをyieldします。

        - 指数バックオフ付きの自動再試行
        - トークン使用量の追跡とロギング
        - コンテキスト長の自動トリミング
        - ツール呼び出しのサポート

        Yields:
            Dict[str, Any]: イベントを表す辞書。以下のtypeを持つ。
                - 'content': テキストトークン (`{'type': 'content', 'value': str}`)
                - 'tool_calls': 完成したツール呼び出しのリスト (`{'type': 'tool_calls', 'value': List[Dict]}`)
                - 'error': エラーメッセージ (`{'type': 'error', 'value': str}`)

        Returns:
            Dict[str, Any]: トークン使用量とコストの統計情報。
        """
        if not self.client:
            yield {
                "type": "error",
                "value": "[エラー] OpenAIクライアントが初期化されていません。APIキーを確認してください。",
            }
            return {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "cost_usd": 0.0,
            }

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(self._convert_history_to_messages(history))
        messages.append({"role": "user", "content": message})

        messages = self._trim_messages(messages)
        prompt_tokens = self._num_tokens_from_messages(messages)

        extra_params: Dict[str, Any] = {}
        if self.api_version:
            extra_params["api_version"] = self.api_version
        if tools:
            extra_params["tools"] = tools
        if tool_choice:
            extra_params["tool_choice"] = tool_choice

        for attempt in range(self.max_retries):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    **extra_params,
                )

                full_response_content = ""
                # tool_callsをインデックスごとに格納する辞書
                collected_tool_calls: Dict[int, Dict[str, Any]] = {}

                for chunk in stream:
                    delta: ChoiceDelta = chunk.choices[0].delta
                    if delta.content:
                        full_response_content += delta.content
                        yield {"type": "content", "value": delta.content}

                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index
                            if idx not in collected_tool_calls:
                                # 新しいツール呼び出しの開始
                                collected_tool_calls[idx] = {
                                    "id": tc_chunk.id,
                                    "type": "function",
                                    "function": {
                                        "name": (
                                            tc_chunk.function.name
                                            if tc_chunk.function
                                            else ""
                                        ),
                                        "arguments": (
                                            tc_chunk.function.arguments
                                            if tc_chunk.function
                                            else ""
                                        ),
                                    },
                                }
                            else:
                                # 既存のツール呼び出しに引数を追記
                                if tc_chunk.function and tc_chunk.function.arguments:
                                    collected_tool_calls[idx]["function"][
                                        "arguments"
                                    ] += tc_chunk.function.arguments

                if collected_tool_calls:
                    yield {
                        "type": "tool_calls",
                        "value": list(collected_tool_calls.values()),
                    }

                completion_tokens = self._estimate_tokens_from_text(
                    full_response_content
                )
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                cost_usd = self._estimate_cost(usage)
                self._append_usage_log(
                    success=True,
                    model=self.model,
                    usage=usage,
                    cost_usd=cost_usd,
                    system_prompt=system_prompt,
                    history_count=len(history),
                    error_message=None,
                )
                logger.info(
                    f"Request successful. Token usage: {usage}, cost_usd: {cost_usd:.8f}"
                )
                return {"usage": usage, "cost_usd": cost_usd}

            except (AuthenticationError, BadRequestError) as e:
                error_msg = f"[{type(e).__name__}] {getattr(e, 'message', str(e))}"
                logger.error(f"Fatal API error: {error_msg}")
                yield {"type": "error", "value": error_msg}
                break  # Fatal errors should not be retried

            except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
                if attempt >= self.max_retries - 1:
                    error_msg = f"[{type(e).__name__}] {getattr(e, 'message', str(e))}"
                    logger.error(
                        f"API error after {self.max_retries} attempts: {error_msg}"
                    )
                    self._append_usage_log(
                        success=False,
                        model=self.model,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": 0,
                            "total_tokens": prompt_tokens,
                        },
                        cost_usd=0.0,
                        system_prompt=system_prompt,
                        history_count=len(history),
                        error_message=error_msg,
                    )
                    yield {"type": "error", "value": error_msg}
                    break

                wait_time = self.retry_backoff_base ** (attempt + 1)
                logger.warning(
                    f"API error: {type(e).__name__}. Retrying in {wait_time:.2f} seconds..."
                )
                time.sleep(wait_time)

            except Exception as e:
                logger.exception("An unexpected error occurred.")
                error_msg = f"[不明なエラー] {str(e)}"
                self._append_usage_log(
                    success=False,
                    model=self.model,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 0,
                        "total_tokens": prompt_tokens,
                    },
                    cost_usd=0.0,
                    system_prompt=system_prompt,
                    history_count=len(history),
                    error_message=error_msg,
                )
                yield {"type": "error", "value": error_msg}
                break

        # すべてのリトライが失敗した場合
        final_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }
        return {"usage": final_usage, "cost_usd": 0.0}
