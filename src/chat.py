import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import requests
from dotenv import load_dotenv

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
        # GroqはOpenAI互換エンドポイントだが、baseは環境変数で上書き可能
        self.api_base = os.getenv("GROQ_API_BASE", self.DEFAULT_API_BASE)
        # モデル名はGroqのデフォルトを設定（必要に応じて環境変数で上書き）
        self.model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        # 任意: バージョンをクエリで付ける
        self.api_version = os.getenv("GROQ_API_VERSION")

        self.max_retries = self._get_env_var("MAX_RETRIES", max_retries, int)
        self.retry_backoff_base = self._get_env_var(
            "RETRY_BACKOFF_BASE", retry_backoff_base, float
        )

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

        # HTTP session
        self.session = self._create_session()

        # 内部バッファ
        self._last_full_response_text: str = ""

    def _get_env_var(self, key: str, default: Any, cast_type: type) -> Any:
        value = os.getenv(key, default)
        try:
            return cast_type(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid value for {key}: '{value}'. Using default: {default}"
            )
            return default

    def _create_session(self) -> Optional[requests.Session]:
        if not self.api_key:
            return None
        s = requests.Session()
        s.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        return s

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

    def _estimate_tokens_from_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def _num_tokens_from_messages(self, messages: List[Dict[str, Any]]) -> int:
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

    def _prepare_api_payload(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: float,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        # Groq拡張: reasoning_effort, max_completion_tokens
        payload["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low")
        payload["max_completion_tokens"] = int(
            os.getenv("GROQ_MAX_COMPLETION_TOKENS", "32766")
        )

        # tools と tool_choice はOpenAI互換形式でGroqも対応
        if tools is None:
            payload["tools"] = [
                {"type": "browser_search"},
                {"type": "code_interpreter"},
            ]
            payload["tool_choice"] = "auto"
        else:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        return payload

    def _parse_sse_stream(
        self, resp: requests.Response
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Groq Chat CompletionsのSSEを逐次パースして、以下のイベントをyield:
        - {"type": "content", "value": str}
        - {"type": "tool_calls", "value": List[...]}  // function呼び出しが完了したタイミングでまとめて
        - {"type": "error", "value": str}            // 受信中にJSONデコード等で問題があれば
        """
        collected_tool_calls: Dict[int, Dict[str, Any]] = {}
        full_response_parts: List[str] = []

        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                data = raw[len("data: ") :]
            elif raw.startswith("data:"):
                data = raw[len("data:") :].lstrip()
            else:
                continue

            if data == "[DONE]":
                if collected_tool_calls:
                    yield {
                        "type": "tool_calls",
                        "value": list(collected_tool_calls.values()),
                    }
                break

            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                yield {"type": "error", "value": f"JSON decode error: {data[:200]}..."}
                continue

            try:
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                content_val = delta.get("content")
                if isinstance(content_val, str) and content_val:
                    full_response_parts.append(content_val)
                    yield {"type": "content", "value": content_val}

                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        fn = tc.get("function") or {}
                        name_val = fn.get("name")
                        if isinstance(name_val, str):
                            collected_tool_calls[idx]["function"]["name"] = name_val
                        args_val = fn.get("arguments")
                        if isinstance(args_val, str):
                            collected_tool_calls[idx]["function"][
                                "arguments"
                            ] += args_val
            except Exception as e:
                yield {"type": "error", "value": f"SSE parse error: {str(e)}"}
                continue

        self._last_full_response_text = "".join(full_response_parts)

    def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        if not self.session:
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
            # 配列やその他は単純化してテキスト連結（SDKに依存せず安全に）
            try:
                parts: List[str] = []
                for part in raw_system_prompt or []:
                    if isinstance(part, dict):
                        txt = part.get("text")
                        if isinstance(txt, str):
                            parts.append(txt)
                system_prompt = " ".join(parts)
            except Exception:
                system_prompt = ""
        history_count = sum(1 for m in messages if m.get("role") == "user")

        payload = self._prepare_api_payload(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )

        params = {}
        if self.api_version:
            params["api_version"] = self.api_version

        url = f"{self.api_base.rstrip('/')}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    url,
                    params=params,
                    data=json.dumps(payload),
                    stream=True,
                    timeout=600,
                )

                # エラーステータスは本文を読みつつ適切に処理
                if resp.status_code >= 400:
                    try:
                        err_json = resp.json()
                        message = err_json.get("error", {}).get(
                            "message", json.dumps(err_json, ensure_ascii=False)
                        )
                    except Exception:
                        message = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    yield {"type": "error", "value": message}
                    # 4xxは致命的、429のみリトライ対象にしても良いがここではstatusで判別
                    if resp.status_code in (429, 408, 409, 425, 500, 502, 503, 504):
                        # リトライ可能とみなす
                        raise RuntimeError(message)
                    else:
                        # 非リトライ
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
                            error_message=message,
                        )
                        return {
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": 0,
                                "total_tokens": prompt_tokens,
                            },
                            "cost_usd": 0.0,
                        }

                # ストリームを逐次処理
                for event in self._parse_sse_stream(resp):
                    yield event

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

            except (requests.Timeout, requests.ConnectionError) as e:
                # 最終試行で失敗したらログして終了
                if attempt >= self.max_retries - 1:
                    msg = f"[{type(e).__name__}] {str(e)}"
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
                        error_message=msg,
                    )
                    yield {"type": "error", "value": msg}
                    break
                wait_time = self.retry_backoff_base ** (attempt + 1)
                logger.warning(
                    f"Network error: {type(e).__name__}. Retrying in {wait_time:.2f} seconds..."
                )
                time.sleep(wait_time)
            except RuntimeError as e:
                # 上でHTTPエラーをRuntimeErrorへラップしている場合のリトライ
                if attempt >= self.max_retries - 1:
                    msg = f"[HTTPError] {str(e)}"
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
                        error_message=msg,
                    )
                    yield {"type": "error", "value": msg}
                    break
                wait_time = self.retry_backoff_base ** (attempt + 1)
                logger.warning(f"HTTP error. Retrying in {wait_time:.2f} seconds...")
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
