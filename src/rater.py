import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import groq
from groq import APIError, AsyncGroq, Groq, RateLimitError
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# --- 初期化処理 ---
groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY") # Groq APIキーを環境変数から取得
if not groq_api_key:
    logging.error("GROQ_API_KEY環境変数が設定されていません。")
    raise ValueError("GROQ_API_KEY環境変数が設定されていません。")

async_groq_client = AsyncGroq(api_key=groq_api_key, timeout=600.0) # 非同期Groqクライアントの初期化
sync_groq_client = Groq(api_key=groq_api_key, timeout=600.0) # 同期Groqクライアントの初期化

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- データクラス ---
@dataclass
class GroqConfig:
    get_output_model: str = "meta-llama/llama-guard-4-12b"
    rater_model: str = "llama-3.3-70b-versatile"
    max_tokens_get_output: int = 1024
    max_tokens_rater: int = 8192
    temperature_get_output: float = 0.0
    temperature_rater: float = 0.0


# --- メインクラス ---
class Rater:
    def __init__(self) -> None:
        self.config = GroqConfig()

    async def __call__(
        self,
        initial_prompt: str,
        candidates: List[Dict[str, str]],
        demo_data: Dict[str, str],
    ) -> Optional[int]:
        """複数のプロンプト候補を評価し、最も良いものを選択します。"""
        self._validate_inputs(initial_prompt, candidates, demo_data)

        unrated_indices = [i for i, c in enumerate(candidates) if "output" not in c]
        if unrated_indices:
            unrated_prompts = [
                self._replace_placeholders(candidates[i]["prompt"], demo_data)
                for i in unrated_indices
            ]
            outputs = await self._get_outputs_parallel(unrated_prompts)
            for i, output in zip(unrated_indices, outputs):
                candidates[i]["input"] = self._replace_placeholders(
                    candidates[i]["prompt"], demo_data
                )
                candidates[i]["output"] = output if isinstance(output, str) else ""

        initial_prompt_filled = self._replace_placeholders(initial_prompt, demo_data)
        rate = self.rater(initial_prompt_filled, candidates)
        logging.info(f"Rater.__call__ return: {rate}")
        return rate

    def _validate_inputs(
        self,
        initial_prompt: str,
        candidates: List[Dict[str, str]],
        demo_data: Dict[str, str],
    ) -> None:
        """入力パラメータの検証を行います。"""
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です")
        if not candidates:
            raise ValueError("候補が提供されていません")
        if not demo_data:
            raise ValueError("デモデータが提供されていません")
        for c in candidates:
            if "prompt" not in c or not c["prompt"].strip():
                raise ValueError("候補プロンプトが空または無効です")

    def _replace_placeholders(self, text: str, data: Dict[str, str]) -> str:
        """テキスト内のプレースホルダをデモデータで置換します。"""
        for k, v in data.items():
            text = text.replace(k, v)
        return text

    async def _get_outputs_parallel(self, prompts: List[str]) -> List[Optional[str]]:
        """複数のプロンプトに対して非同期でGroqモデルを実行し、出力を取得します。"""
        tasks = [self._get_output_async(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [res if isinstance(res, str) else None for res in results]

    async def _get_output_async(self, prompt: str) -> Optional[str]:
        """指定されたプロンプトでGroqモデルを非同期で実行し、出力を取得します。"""
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        max_retries = 3
        backoff_factor = 2
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                completion = await async_groq_client.chat.completions.create(
                    model=self.config.get_output_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_tokens_get_output,
                    temperature=self.config.temperature_get_output,
                )
                result = completion.choices[0].message.content
                logging.info(f"Rater._get_output_async successful, result: {result}")
                return result
            except RateLimitError:
                logging.warning(
                    f"Rate limit exceeded. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:
                logging.error(f"Rater._get_output_async - Groq APIError: {e}")
                return None
            except Exception as e:
                logging.error(f"Rater._get_output_async - Unexpected error: {e}")
                return None
        logging.error(
            "Max retries reached for _get_output_async. Failed to get a response from Groq API."
        )
        return None

    def rater(
        self, initial_prompt: str, candidates: List[Dict[str, str]]
    ) -> Optional[int]:
        """Groqモデルを使用して、複数の候補応答の中から最も良いものを評価させます。"""
        if not candidates:
            logging.debug("Rater.rater - No candidates provided. Returning None.")
            return None

        rater_example = json.dumps({"Preferred": "Response 1"})
        response_prompts = [
            f"Response {i+1}:\nInput: {c.get('input', 'N/A')}\nOutput: {c.get('output', 'N/A')}\n</response_{i+1}>".strip()
            for i, c in enumerate(candidates)
        ]
        response_prompt_str = "\n\n".join(response_prompts)

        rater_prompt = """
You        You are an expert rater of helpful and honest Assistant responses. Given the instruction and the two responses choose the most helpful and honest response.
        Please pay particular attention to the response formatting requirements called for in the instruction.

        Instruction:
        <instruction>
        {instruction}
        </instruction>

        {Response_prompt}

        Finally, select which response is the most helpful and honest.

        Use JSON format with key `Preferred` when returning results. The value should be a string like "Response 1". Please only output the result in json format, and do the json format check and return, don't include other extra text!
        An example of output is as follows:
        Output example: {rater_example}
        """.strip()

        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": rater_prompt.format(
                    instruction=initial_prompt,
                    Response_prompt=response_prompt_str,
                    rater_example=rater_example,
                ),
            }
        ]

        max_retries = 3
        backoff_factor = 2
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                completion = sync_groq_client.chat.completions.create(
                    model=self.config.rater_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_tokens_rater,
                    temperature=self.config.temperature_rater,
                )
                content = completion.choices[0].message.content
                if not content:
                    raise json.JSONDecodeError("No content from LLM", "", 0)

                result_json = json.loads(content)
                preferred_text = result_json.get("Preferred")
                if not preferred_text:
                    raise ValueError("LLM response JSON is missing 'Preferred' key.")

                # 正規表現で応答から数値を抽出
                match = re.search(r"\d+", preferred_text)
                if match:
                    # 1-based indexを0-basedに変換
                    final_result = int(match.group(0)) - 1
                    if 0 <= final_result < len(candidates):
                        logging.info(f"Rater.rater successful, result: {final_result}")
                        return final_result

                raise ValueError(
                    f"Could not parse a valid index from LLM response: '{preferred_text}'"
                )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logging.error(f"Rater.rater - Error processing LLM response: {e}")
                if attempt >= max_retries - 1:
                    return None  # 最終試行でも失敗したらNoneを返す
            except RateLimitError:
                logging.warning(
                    f"Rate limit exceeded. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}"
                )
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:
                logging.error(f"Rater.rater - Groq APIError: {e}")
                return None
            except Exception as e:
                logging.error(f"Rater.rater - Unexpected error: {e}")
                return None

        logging.error(
            "Max retries reached for rater. Failed to get a response from Groq API."
        )
        return None
