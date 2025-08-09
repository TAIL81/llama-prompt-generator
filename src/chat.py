import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import openai
from dotenv import load_dotenv
from openai import NotGiven
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from typing_extensions import cast

logger = logging.getLogger(__name__)


class ChatService:
    # --- デフォルト設定値 ---
    DEFAULT_API_BASE = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "openai/gpt-oss-120b"
    DEFAULT_MAX_RETRIES = 3
    # リトライ時の指数バックオフのベース値
    DEFAULT_RETRY_BACKOFF_BASE = 2.0
    DEFAULT_CHARS_PER_TOKEN = 4.0
    DEFAULT_PROMPT_PRICE = 0.0
    DEFAULT_COMPLETION_PRICE = 0.0
    DEFAULT_USAGE_LOG_PATH = "logs/chat_usage.jsonl"

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE,
    ):
        # .envファイルから環境変数を読み込む
        load_dotenv()

        # Groq APIキーを環境変数から取得
        self.api_key = os.getenv("GROQ_API_KEY")
        # GroqはOpenAI互換エンドポイントだが、baseは環境変数で上書き可能
        self.api_base = os.getenv("GROQ_API_BASE", self.DEFAULT_API_BASE)
        # モデル名はGroqのデフォルトを設定（必要に応じて環境変数で上書き）
        self.model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        # 任意: バージョンをクエリで付ける
        self.api_version = os.getenv("GROQ_API_VERSION")

        # 最大リトライ回数を環境変数またはデフォルト値から取得
        self.max_retries = self._get_env_var("MAX_RETRIES", max_retries, int)
        # リトライ時のバックオフベース値を環境変数またはデフォルト値から取得
        self.retry_backoff_base = self._get_env_var(
            "RETRY_BACKOFF_BASE", retry_backoff_base, float
        )

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

        # OpenAI互換クライアント (Groq)
        self.client = self._create_client()

        # 最後の完全な応答テキストを保持する内部バッファ
        self._last_full_response_text: str = ""

    def _get_env_var(self, key: str, default: Any, cast_type: type) -> Any:
        """環境変数から値を取得し、指定された型にキャストするヘルパー関数。
        取得に失敗した場合はデフォルト値を返す。"""
        value = os.getenv(key, default)
        try:
            return cast_type(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid value for {key}: '{value}'. Using default: {default}"
            )
            return default

    def _create_client(self):
        """OpenAI互換クライアントを作成（GroqのChat Completions APIを使用）。"""
        if not self.api_key:
            return None
        # openai>=1.x 形式のクライアント
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base.rstrip("/"),
            )
            return client
        except Exception:
            logger.exception("Failed to create OpenAI-compatible client.")
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
        import hashlib

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

    def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        stream: bool = True,  # stream をデフォルトTrueに変更
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """チャット補完をストリーム形式または非ストリーム形式で取得する。"""
        if not self.client:
            yield {
                "type": "error",
                "value": "[エラー] Groqクライアントが初期化されていません。GROQ_API_KEYを設定してください。",
            }
            return {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "cost_usd": 0.0,
            }

        prompt_tokens = self._num_tokens_from_messages(messages)

        # system_promptはstr想定に正規化
        raw_system_prompt = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        if isinstance(raw_system_prompt, str):
            system_prompt = raw_system_prompt
        else:
            try:
                parts: List[str] = []
                for part in raw_system_prompt or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                system_prompt = " ".join(parts)
            except Exception:
                system_prompt = ""
        history_count = sum(1 for m in messages if m.get("role") == "user")

        # OpenAI SDK の型に合わせる
        sdk_messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionMessageParam, msg) for msg in messages
        ]
        sdk_tools: Union[List[ChatCompletionToolParam], NotGiven] = (
            [cast(ChatCompletionToolParam, tool) for tool in tools]
            if tools is not None
            else NotGiven()
        )
        sdk_tool_choice = NotGiven() if tool_choice is None else tool_choice

        for attempt in range(self.max_retries):
            try:
                if stream:
                    # ストリーム: OpenAIクライアントのイベントストリームを使用
                    full_text_parts: List[str] = []
                    tool_calls = []
                    stream_resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=sdk_messages,
                        temperature=temperature,
                        tools=sdk_tools,
                        tool_choice=sdk_tool_choice,  # type: ignore
                        stream=True,
                    )
                    for chunk in stream_resp:
                        if not isinstance(chunk, ChatCompletionChunk):
                            continue

                        # content
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            yield {"type": "content", "value": content}
                            full_text_parts.append(content)

                        # tool_calls
                        if chunk.choices and chunk.choices[0].delta.tool_calls:
                            for tc in chunk.choices[0].delta.tool_calls:
                                if tc.function:
                                    tool_calls.append(tc.function.dict())

                    if tool_calls:
                        yield {"type": "tool_calls", "value": tool_calls}

                    # usage (ストリームの最後で取得できる場合がある)
                    # Groq APIはストリーミングレスポンスにusageを含めない場合がある
                    completion_tokens = self._estimate_tokens_from_text(
                        "".join(full_text_parts)
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
                else:
                    # 非ストリーム: 単発応答
                    resp: ChatCompletion = self.client.chat.completions.create(
                        model=self.model,
                        messages=sdk_messages,
                        temperature=temperature,
                        tools=sdk_tools,
                        tool_choice=sdk_tool_choice,  # type: ignore
                        stream=False,
                    )
                    # テキスト
                    output_text = resp.choices[0].message.content or ""
                    if output_text:
                        yield {"type": "content", "value": output_text}

                    # ツール呼び出し
                    if resp.choices[0].message.tool_calls:
                        tool_calls = [
                            tc.function.dict()
                            for tc in resp.choices[0].message.tool_calls
                        ]
                        yield {"type": "tool_calls", "value": tool_calls}

                    # usage
                    usage = {}
                    if resp.usage:
                        usage = {
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                            "total_tokens": resp.usage.total_tokens,
                        }
                    else:
                        completion_tokens = self._estimate_tokens_from_text(output_text)
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

            except Exception as e:
                # OpenAIクライアント内で429/5xx等も例外になる想定。指数バックオフでリトライ。
                if attempt >= self.max_retries - 1:
                    error_msg = f"[OpenAIClientError] {str(e)}"
                    self._log_api_result(
                        success=False,
                        model=self.model,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": 0,
                            "total_tokens": prompt_tokens,
                        },
                        cost_usd=0.0,
                        system_prompt=system_prompt,
                        history_count=history_count,
                        error_message=error_msg,
                    )
                    yield {"type": "error", "value": error_msg}
                    break
                wait_time = self.retry_backoff_base ** (attempt + 1)
                logger.warning(f"Client error. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        # 最終的な使用量とコストを返す（エラーの場合）
        final_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }
        return {"usage": final_usage, "cost_usd": 0.0}
