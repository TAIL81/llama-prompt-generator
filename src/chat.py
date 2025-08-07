import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Union, cast

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
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
)

# ChatCompletionMessageToolCall, ChatCompletionMessageToolCallFunction は存在しない環境があるため削除
# TypedDict も未使用のため削除

logger = logging.getLogger(__name__)


class ChatService:
    # --- Default configuration values ---
    DEFAULT_API_BASE = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "openai/gpt-oss-120b"
    DEFAULT_MAX_RETRIES = 3
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
        load_dotenv()

        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", self.DEFAULT_API_BASE)
        self.model = os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self.api_version = os.getenv("OPENAI_API_VERSION")

        self.max_retries = self._get_env_var("MAX_RETRIES", max_retries, int)
        self.retry_backoff_base = self._get_env_var(
            "RETRY_BACKOFF_BASE", retry_backoff_base, float
        )

        self.client = self._create_client()
        self.chars_per_token = self._get_env_var(
            "TOKEN_CHARS_PER_TOKEN", self.DEFAULT_CHARS_PER_TOKEN, float
        )
        self.prompt_price_per_million = self._get_env_var(
            "PROMPT_PRICE_PER_MILLION", self.DEFAULT_PROMPT_PRICE, float
        )
        self.completion_price_per_million = self._get_env_var(
            "COMPLETION_PRICE_PER_MILLION", self.DEFAULT_COMPLETION_PRICE, float
        )
        self.usage_log_path = os.getenv("USAGE_LOG_PATH", self.DEFAULT_USAGE_LOG_PATH)

    def _get_env_var(self, key: str, default: Any, cast_type: type) -> Any:
        value = os.getenv(key, default)
        try:
            return cast_type(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid value for {key}: '{value}'. Using default: {default}"
            )
            return default

    def _create_client(self) -> Optional[OpenAI]:
        if not self.api_key:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.api_base)

    def _estimate_cost(self, usage: Dict[str, int]) -> float:
        prompt_rate = self.prompt_price_per_million / 1_000_000.0
        completion_rate = self.completion_price_per_million / 1_000_000.0
        cost = (
            usage.get("prompt_tokens", 0) * prompt_rate
            + usage.get("completion_tokens", 0) * completion_rate
        )
        return float(cost)

    def _append_usage_log(self, **kwargs: Any) -> None:
        try:
            log_path = Path(self.usage_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {"timestamp": datetime.utcnow().isoformat() + "Z", **kwargs}
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

    def _estimate_tokens_from_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def _num_tokens_from_messages(
        self, messages: List[ChatCompletionMessageParam]
    ) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                if value and isinstance(value, str):
                    num_tokens += self._estimate_tokens_from_text(value)
                if key == "name":
                    num_tokens -= 1
        return num_tokens + 3

    def _prepare_api_params(
        self,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if self.api_version:
            params["api_version"] = self.api_version

        params["reasoning_effort"] = "low"
        params["max_completion_tokens"] = 32766

        if tools is None:
            params["tools"] = [
                {"type": "browser_search"},
                {"type": "code_interpreter"},
            ]
            params["tool_choice"] = "auto"
        else:
            params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
        return params

    def _process_stream(
        self, stream: Iterable[ChatCompletionChunk]
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        full_response_content = ""
        collected_tool_calls: Dict[int, Dict[str, Any]] = {}

        for chunk in stream:
            # OpenAI SDK の型は Union/Optional を含むため、適切にガードしつつ処理する
            chunk = cast(ChatCompletionChunk, chunk)
            delta: ChoiceDelta = chunk.choices[0].delta  # type: ignore[assignment]

            # content は Optional[str]。存在し、かつ str の場合だけ扱う（TypedDict バリアント警告を回避）
            content_val = getattr(delta, "content", None)
            if isinstance(content_val, str):
                full_response_content += content_val
                yield {"type": "content", "value": content_val}

            # tool_calls は Optional[List[ChoiceDeltaToolCall]]。存在する場合のみ処理
            tool_calls: Optional[List[ChoiceDeltaToolCall]] = getattr(delta, "tool_calls", None)  # type: ignore[assignment]
            if tool_calls:
                for tc_chunk in tool_calls:
                    idx = getattr(tc_chunk, "index", 0)
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": getattr(tc_chunk, "id", ""),
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "",
                            },
                        }
                    # function は ChoiceDeltaToolCallFunction | None
                    fn: Optional[ChoiceDeltaToolCallFunction] = getattr(tc_chunk, "function", None)  # type: ignore[assignment]
                    if fn:
                        name_val = getattr(fn, "name", None)
                        if isinstance(name_val, str):
                            collected_tool_calls[idx]["function"]["name"] = name_val
                        args_val = getattr(fn, "arguments", None)
                        if isinstance(args_val, str):
                            collected_tool_calls[idx]["function"][
                                "arguments"
                            ] += args_val

        if collected_tool_calls:
            yield {"type": "tool_calls", "value": list(collected_tool_calls.values())}

        return {"full_response_content": full_response_content}

    def chat_completion_stream(
        self,
        messages: List[ChatCompletionMessageParam],
        temperature: float,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
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

        prompt_tokens = self._num_tokens_from_messages(messages)
        # system_prompt は str 前提でログに渡すため、str へ正規化
        raw_system_prompt = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        if isinstance(raw_system_prompt, str):
            system_prompt = raw_system_prompt
        elif raw_system_prompt is None:
            system_prompt = ""
        else:
            # Iterable[ChatCompletionContentPartParam] の可能性を想定し、text を抽出
            parts: List[str] = []
            try:
                for part in cast(
                    Iterable[ChatCompletionContentPartParam], raw_system_prompt
                ):
                    if isinstance(part, dict):
                        # dict 形の場合は text を優先（Pylance 型明示）
                        part_dict = cast(Dict[str, Any], part)
                        txt = part_dict.get("text")
                        if isinstance(txt, str):
                            parts.append(txt)
                    else:
                        # 型クラスの場合: type 属性で分岐
                        ptype = getattr(part, "type", None)
                        if ptype == "text":
                            txt2 = getattr(part, "text", None)
                            if isinstance(txt2, str):
                                parts.append(txt2)
                system_prompt = " ".join(parts)
            except Exception:
                system_prompt = ""
        history_count = sum(1 for m in messages if m.get("role") == "user")

        api_params = self._prepare_api_params(tools, tool_choice)

        for attempt in range(self.max_retries):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    **api_params,
                )

                stream_processor = self._process_stream(stream)
                final_stream_result = {}
                for event in stream_processor:
                    yield event
                    if event.get("type") == "content":
                        # This is a bit of a hack to get the final result from the generator
                        # A better approach might be to have the generator return a final value
                        pass

                # The generator _process_stream should return the final result
                # but since it's a generator, we can't get the return value directly.
                # Let's assume the last yielded value is the final result.
                # A better implementation would use a different pattern.
                # For now, we will re-implement the logic to get the full response.

                # Re-creating stream to get full response, this is inefficient.
                stream_for_full_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    **api_params,
                )
                # content 文字列だけを安全に連結（TypedDict バリアントに依存しない）
                parts: List[str] = []
                for ch in stream_for_full_response:
                    # choices は Optional の可能性があるためガード
                    choices = getattr(ch, "choices", None)
                    if not choices:
                        continue
                    first = choices[0]
                    # delta は Pydantic BaseModel/TypedDict なので文字列化で厳密型チェックを回避しつつテキストのみ抽出
                    delta_any: Any = getattr(first, "delta", None)  # type: ignore[assignment]
                    if not delta_any:
                        continue
                    # BaseModel なら model_dump で辞書化、そうでなければ __dict__ / asdict 相当を試す
                    to_dict = getattr(delta_any, "model_dump", None)
                    delta_dict_any: Any
                    if callable(to_dict):
                        delta_dict_any = to_dict()
                    elif hasattr(delta_any, "__dict__"):
                        delta_dict_any = delta_any.__dict__
                    else:
                        try:
                            delta_dict_any = dict(delta_any)
                        except Exception:
                            delta_dict_any = {}
                    # Pylance 対策: 明示的に Dict[str, Any] へキャストしつつ、dict でなければスキップ
                    if not isinstance(delta_dict_any, dict):
                        continue
                    delta_dict = cast(Dict[str, Any], delta_dict_any)
                    content_val = delta_dict.get("content")
                    if isinstance(content_val, str):
                        parts.append(content_val)
                    elif isinstance(content_val, list):
                        # まれに content が parts の配列になる SDK 実装に対応
                        for p in content_val:
                            if isinstance(p, str):
                                parts.append(p)
                            elif isinstance(p, dict):
                                # dict のときのみ .get を使用（Pylance への型明示）
                                p_dict = cast(Dict[str, Any], p)
                                txt = p_dict.get("text")
                                if isinstance(txt, str):
                                    parts.append(txt)
                            elif hasattr(p, "text"):
                                # dict 以外では .get は使用しない。text 属性のみ参照する
                                txt_attr = getattr(p, "text", None)
                                if isinstance(txt_attr, str):
                                    parts.append(txt_attr)
                            # それ以外の型は無視
                        # 上記以外の要素は無視
                full_response_content = "".join(parts)

                completion_tokens = self._estimate_tokens_from_text(
                    full_response_content
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
                logger.info(
                    f"Request successful. Token usage: {usage}, cost_usd: {cost_usd:.8f}"
                )
                return {"usage": usage, "cost_usd": cost_usd}

            except (AuthenticationError, BadRequestError) as e:
                error_msg = f"[{type(e).__name__}] {getattr(e, 'message', str(e))}"
                logger.error(f"Fatal API error: {error_msg}")
                yield {"type": "error", "value": error_msg}
                break

            except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
                if attempt >= self.max_retries - 1:
                    error_msg = f"[{type(e).__name__}] {getattr(e, 'message', str(e))}"
                    logger.error(
                        f"API error after {self.max_retries} attempts: {error_msg}"
                    )
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
                logger.warning(
                    f"API error: {type(e).__name__}. Retrying in {wait_time:.2f} seconds..."
                )
                time.sleep(wait_time)

            except Exception as e:
                logger.exception("An unexpected error occurred.")
                error_msg = f"[不明なエラー] {str(e)}"
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

        final_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }
        return {"usage": final_usage, "cost_usd": 0.0}
