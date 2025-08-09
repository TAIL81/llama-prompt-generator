import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

from dotenv import load_dotenv
from groq import APIError, APIStatusError, Groq, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)


class ChatService:
    # --- デフォルト設定値 ---
    DEFAULT_MODEL = "openai/gpt-oss-120b"
    DEFAULT_MAX_RETRIES = 3
    # リトライ時の指数バックオフのベース値 (SDKが処理)
    DEFAULT_RETRY_BACKOFF_BASE = 2.0
    DEFAULT_CHARS_PER_TOKEN = 4.0
    DEFAULT_PROMPT_PRICE = 0.0
    DEFAULT_COMPLETION_PRICE = 0.0
    DEFAULT_USAGE_LOG_PATH = "logs/chat_usage.jsonl"

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE,  # SDK側で利用
    ):
        # .envファイルから環境変数を読み込む
        load_dotenv()

        # Groq APIキーを環境変数から取得
        self.api_key = os.getenv("GROQ_API_KEY")
        # モデル名はGroqのデフォルトを設定（必要に応じて環境変数で上書き）
        self.model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)

        # 最大リトライ回数を環境変数またはデフォルト値から取得
        self.max_retries = self._get_env_var("MAX_RETRIES", max_retries, int)

        # 1トークンあたりの文字数を環境変数またはデフォルト値から取得
        self.chars_per_token = self._get_env_var(
            "TOKEN_CHARS_PER_TOKEN", self.DEFAULT_CHARS_PER_TOKEN, float
        )
        # プロンプトの100万トークンあたりの価格を環境変数またはデフォルト値から取得
        self.prompt_price_per_million = self._get_env_var(
            "PROMPT_PRICE_PER_MILLION", self.DEFAULT_PROMPT_PRICE, float
        )
        # 完了の100万トークンあたりの価格を環境変数またはデフォルト値から取得
        self.completion_price_per_million = self._get_env_var(
            "COMPLETION_PRICE_PER_MILLION", self.DEFAULT_COMPLETION_PRICE, float
        )
        self.usage_log_path = os.getenv("USAGE_LOG_PATH", self.DEFAULT_USAGE_LOG_PATH)

        # Groqクライアントを初期化
        self.client = self._init_client()

        # 最後の完全な応答テキストを保持する内部バッファ
        self._last_full_response_text: str = ""

    def _get_env_var(self, key: str, default: Any, cast_type: type) -> Any:
        """環境変数から値を取得し、指定された型にキャストするヘルパー関数。
        取得に失敗した場合はデフォルト値を返す。"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return cast_type(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid value for {key}: '{value}'. Using default: {default}"
            )
            return default

    def _init_client(self) -> Optional[Groq]:
        """Groqクライアントを初期化する。"""
        if not self.api_key:
            return None
        try:
            return Groq(api_key=self.api_key, max_retries=self.max_retries)
        except Exception:
            logger.exception("Failed to initialize Groq client.")
            return None

    # 使用量に基づいてコストを推定する
    def _estimate_cost(self, usage: Dict[str, int]) -> float:
        prompt_rate = self.prompt_price_per_million / 1_000_000.0
        completion_rate = self.completion_price_per_million / 1_000_000.0
        cost = (
            usage.get("prompt_tokens", 0) * prompt_rate
            + usage.get("completion_tokens", 0) * completion_rate
        )
        return float(cost)

    def _append_usage_log(self, **kwargs: Any) -> None:
        """使用ログをJSONLファイルに追記する。"""
        try:
            log_path = Path(self.usage_log_path)
            # ログファイルの親ディレクトリが存在しない場合は作成
            log_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {"timestamp": datetime.utcnow().isoformat() + "Z", **kwargs}
            # ログエントリをJSON形式でファイルに書き込む
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append usage log.")

    def _log_api_result(
        self,
        *,
        success: bool,
        model: str,
        usage: Dict[str, int],
        cost_usd: float,
        system_prompt: str,
        history_count: int,
        error_message: Optional[str] = None,
    ) -> None:
        """API呼び出しの結果をログに記録する。"""
        system_prompt_hash = hashlib.sha256(
            (system_prompt or "").encode("utf-8")
        ).hexdigest()
        self._append_usage_log(
            success=success,
            model=model,
            usage=usage,
            cost_usd=round(cost_usd, 8),
            system_prompt_sha256=system_prompt_hash,
            history_count=history_count,
            error=error_message,
        )

    # テキストからトークン数を推定する
    def _estimate_tokens_from_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def _num_tokens_from_messages(self, messages: List[Dict[str, Any]]) -> int:
        """メッセージリストからトークン数を推定する。"""
        # 既存の概算ロジックを維持（OpenAI推定式の近似）
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                if value and isinstance(value, str):
                    num_tokens += self._estimate_tokens_from_text(value)
                if key == "name":
                    num_tokens -= 1
        return num_tokens + 3

    def _process_stream_chunk(
        self,
        chunk: Any,
        collected_tool_calls: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """ストリームのチャンクを処理し、イベントを返す。"""
        if not chunk.choices:
            return None

        delta = chunk.choices[0].delta
        if delta is None:
            return None

        # コンテンツの値を抽出
        if content_val := delta.content:
            self._last_full_response_text += content_val
            return {"type": "content", "value": content_val}

        # ツール呼び出しを収集
        if tool_calls := delta.tool_calls:
            for tc in tool_calls:
                idx = tc.index
                if idx not in collected_tool_calls:
                    collected_tool_calls[idx] = {
                        "id": tc.id or "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.function:
                    if name := tc.function.name:
                        collected_tool_calls[idx]["function"]["name"] = name
                    if args := tc.function.arguments:
                        collected_tool_calls[idx]["function"]["arguments"] += args
        return None

    def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """チャット補完をストリーム形式で取得する。"""
        if not self.client:
            yield {
                "type": "error",
                "value": "[エラー] Groqクライアントが初期化されていません。GROQ_API_KEYを設定してください。",
            }
            return {
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost_usd": 0.0,
            }

        prompt_tokens = self._num_tokens_from_messages(messages)
        raw_system_prompt = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        system_prompt = (
            raw_system_prompt
            if isinstance(raw_system_prompt, str)
            else json.dumps(raw_system_prompt)
        )
        history_count = sum(1 for m in messages if m.get("role") == "user")

        self._last_full_response_text = ""
        collected_tool_calls: Dict[int, Dict[str, Any]] = {}

        # toolsが指定されていない場合、デフォルトのツールを設定
        if tools is None:
            tools = [
                {"type": "browser_search"},
                {"type": "code_interpreter"},
            ]
            if tool_choice is None:
                tool_choice = "auto"

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
            )

            for chunk in stream:
                event = self._process_stream_chunk(chunk, collected_tool_calls)
                if event:
                    yield event

            if collected_tool_calls:
                yield {"type": "tool_calls", "value": list(collected_tool_calls.values())}

            # ストリーム終了後にusage推定
            completion_tokens = self._estimate_tokens_from_text(
                self._last_full_response_text
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            cost_usd = self._estimate_cost(usage)

            self._log_api_result(
                success=True,
                model=self.model,
                usage=usage,
                cost_usd=cost_usd,
                system_prompt=system_prompt,
                history_count=history_count,
            )
            return {"usage": usage, "cost_usd": cost_usd}

        except (RateLimitError, APIConnectionError, APIStatusError, APIError) as e:
            error_message = f"[{type(e).__name__}] {str(e)}"
            logger.error(f"Groq API error: {error_message}")
            self._log_api_result(
                success=False,
                model=self.model,
                usage={"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens},
                cost_usd=0.0,
                system_prompt=system_prompt,
                history_count=history_count,
                error_message=error_message,
            )
            yield {"type": "error", "value": error_message}

        except Exception as e:
            error_message = f"[不明なエラー] {str(e)}"
            logger.exception("An unexpected error occurred during chat completion.")
            self._log_api_result(
                success=False,
                model=self.model,
                usage={"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens},
                cost_usd=0.0,
                system_prompt=system_prompt,
                history_count=history_count,
                error_message=error_message,
            )
            yield {"type": "error", "value": error_message}

        # エラー発生時の戻り値
        return {
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens},
            "cost_usd": 0.0,
        }
