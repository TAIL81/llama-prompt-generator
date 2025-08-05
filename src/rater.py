import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import groq
from groq import APIError, Groq, RateLimitError
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field, ValidationError

# モジュールロガー（basicConfigはアプリ側で設定）
logger = logging.getLogger(__name__)

# --- 初期化処理 ---
# 環境変数から Groq API キーを取得（必須）
groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY 環境変数が設定されていません。")
    raise ValueError("GROQ_API_KEY 環境変数が設定されていません。")

# 同期 Groq クライアント（長めのタイムアウト）
sync_groq_client = Groq(api_key=groq_api_key, timeout=600.0)


# --- Pydanticモデル ---
class PreferredResponse(BaseModel):
    """評価モデルの構造化出力（どの応答が望ましいか）"""

    Preferred: str = Field(
        ...,
        description="選好された応答名（例: 'Response 1'）。",
        examples=["Response 1"],
    )


# --- データクラス ---
@dataclass
class GroqConfig:
    """
    Groq モデル設定。
    - get_output_model: 評価用候補の出力生成に使用
    - rater_model: 候補比較の評価に使用
    """

    get_output_model: str = "compound-beta-mini"
    rater_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_tokens_get_output: int = 8192
    max_tokens_rater: int = 8192
    temperature_get_output: float = 0.7
    temperature_rater: float = 0.7


class Rater:
    """
    候補プロンプトを評価し最適なもののインデックスを返す。
    外部APIとの I/F は現状維持。
    """

    def __init__(self) -> None:
        self.config = GroqConfig()

    def __call__(
        self,
        initial_prompt: str,
        candidates: List[Dict[str, str]],
        demo_data: Dict[str, str],
    ) -> Optional[int]:
        """
        候補の入力/出力を補完し、評価モデルでベストを選ぶ。
        戻り値: ベスト候補の 0-based index。失敗時 None。
        """
        self._validate_inputs(initial_prompt, candidates, demo_data)

        # 未評価の候補に対して出力を取得
        unrated_indices = [i for i, c in enumerate(candidates) if "output" not in c]
        if unrated_indices:
            prompts_to_run = [
                self.apply_demo_data_to_prompt(candidates[i]["prompt"], demo_data)
                for i in unrated_indices
            ]
            outputs = self.fetch_outputs_sequentially(prompts_to_run)
            for i, output in zip(unrated_indices, outputs):
                candidates[i]["input"] = self.apply_demo_data_to_prompt(
                    candidates[i]["prompt"], demo_data
                )
                candidates[i]["output"] = output if isinstance(output, str) else ""

        # 初期プロンプトも同様にプレースホルダを適用
        initial_prompt_filled = self.apply_demo_data_to_prompt(
            initial_prompt, demo_data
        )
        index = self.rater(initial_prompt_filled, candidates)
        logger.debug("Rater.__call__ result index=%s", index)
        return index

    def _validate_inputs(
        self,
        initial_prompt: str,
        candidates: List[Dict[str, str]],
        demo_data: Dict[str, str],
    ) -> None:
        """引数の基本検証。エラーは ValueError を送出。"""
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です。")
        if not candidates:
            raise ValueError("候補が提供されていません。")
        if not demo_data:
            raise ValueError("デモデータが提供されていません。")
        for c in candidates:
            if "prompt" not in c or not str(c["prompt"]).strip():
                raise ValueError("候補プロンプトが空または無効です。")

    # 旧名（互換性保持・将来非推奨）
    def _replace_placeholders(self, text: str, data: Dict[str, str]) -> str:
        """DEPRECATED: apply_demo_data_to_prompt を使用してください。"""
        return self.apply_demo_data_to_prompt(text, data)

    def apply_demo_data_to_prompt(self, text: str, data: Dict[str, str]) -> str:
        """
        プレースホルダ置換を行う（現状仕様: data のキー文字列をそのまま検索置換）。
        注意: {{KEY}} 形式限定ではない挙動を維持（外部依存を崩さないため）。
        """
        for k, v in data.items():
            text = text.replace(k, v)
        return text

    def fetch_outputs_sequentially(self, prompts: List[str]) -> List[Optional[str]]:
        """
        各プロンプトを順にモデル実行して出力を収集。
        失敗時 None を格納して継続。
        """
        results: List[Optional[str]] = []
        for idx, prompt in enumerate(prompts):
            try:
                results.append(self.fetch_output_sync(prompt))
            except Exception as e:
                logger.error("逐次実行エラー: index=%s error=%s", idx, e)
                results.append(None)
        return results

    def fetch_output_sync(self, prompt: str) -> Optional[str]:
        """
        指定プロンプトを同期的に実行。
        RateLimit に指数バックオフでリトライ。
        成功: 出力文字列、失敗: None
        """
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        max_retries = 3
        backoff_factor = 2
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                completion = sync_groq_client.chat.completions.create(
                    model=self.config.get_output_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_tokens_get_output,
                    temperature=self.config.temperature_get_output,
                )
                result = completion.choices[0].message.content
                logger.debug("fetch_output_sync 成功: %s", bool(result))
                return result
            except RateLimitError:
                logger.warning(
                    "レート制限: %s 秒後に再試行 (%s/%s)",
                    retry_delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:
                logger.error("Groq API エラー(fetch_output_sync): %s", e)
                return None
            except Exception as e:
                logger.error("想定外エラー(fetch_output_sync): %s", e)
                return None

        logger.error("最大リトライ到達(fetch_output_sync)。応答取得に失敗しました。")
        return None

    def rater(
        self, initial_prompt: str, candidates: List[Dict[str, str]]
    ) -> Optional[int]:
        """
        評価モデルを用いて複数候補から最良の応答を選ぶ。
        戻り値: 0-based index。失敗時 None。
        """
        if not candidates:
            logger.debug("rater: 候補が空のため None を返却。")
            return None

        # 評価用表現を整形（closing タグ様の文字列は可読性のための目印）
        response_prompts = [
            (
                f"Response {i+1}:\n"
                f"Input: {c.get('input', 'N/A')}\n"
                f"Output: {c.get('output', 'N/A')}\n</response_{i+1}>"
            ).strip()
            for i, c in enumerate(candidates)
        ]
        response_prompt_str = "\n\n".join(response_prompts)

        rater_prompt = """
You are an expert rater of helpful and honest Assistant responses. Given the instruction and the responses choose the most helpful and honest response.
Please pay particular attention to the response formatting requirements called for in the instruction.

Instruction:
<instruction>
{instruction}
</instruction>

{Response_prompt}

Finally, select which response is the most helpful and honest.
Your response must be only the JSON object, with no other text before or after it.
""".strip()

        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": rater_prompt.format(
                    instruction=initial_prompt, Response_prompt=response_prompt_str
                ),
            }
        ]

        max_retries = 3
        backoff_factor = 2
        retry_delay = 1

        for attempt in range(max_retries):
            content = None
            try:
                completion = sync_groq_client.chat.completions.create(
                    model=self.config.rater_model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "preferred_response",
                            "description": "The preferred response.",
                            "schema": PreferredResponse.model_json_schema(),
                        },
                    },
                )
                content = completion.choices[0].message.content
                if not content:
                    logger.error("評価応答が空です。再試行します。")
                    if attempt >= max_retries - 1:
                        return None
                    time.sleep(retry_delay)
                    retry_delay *= backoff_factor
                    continue

                # 構造化出力を検証・パース
                result = PreferredResponse.model_validate(json.loads(content))
                preferred_text = result.Preferred

                # "Response N" から N を抽出
                match = re.search(r"\d+", preferred_text or "")
                if not match:
                    logger.error(
                        "評価結果の番号抽出に失敗しました: text=%s", preferred_text
                    )
                    return None

                idx = int(match.group(0)) - 1
                if 0 <= idx < len(candidates):
                    logger.debug("rater 成功: index=%s", idx)
                    return idx

                logger.error("評価結果のインデックスが範囲外です: index=%s", idx)
                return None

            except json.JSONDecodeError as e:
                # content が None の場合にも対応
                body = content if content is not None else "(なし)"
                logger.error("JSON パースエラー: %s\nAPI レスポンス: %s", e, body)
                if attempt >= max_retries - 1:
                    return None
                time.sleep(retry_delay)
                retry_delay *= backoff_factor

            except ValidationError as e:
                logger.error(
                    "構造化出力の検証エラー: %s\nAPI レスポンス: %s", e, content
                )
                return None

            except RateLimitError:
                logger.warning(
                    "評価でレート制限: %s 秒後に再試行 (%s/%s)",
                    retry_delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(retry_delay)
                retry_delay *= backoff_factor

            except APIError as e:
                logger.error("Groq API エラー(rater): %s", e)
                return None

            except Exception as e:
                logger.error("想定外エラー(rater): %s", e)
                return None

        logger.error("最大リトライ到達(rater)。評価に失敗しました。")
        return None
