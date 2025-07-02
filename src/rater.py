import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import groq  # Import the groq module to access specific error types
import nest_asyncio # nest_asyncioをインポート
from groq import AsyncGroq, Groq
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# nest_asyncioを適用して、既に実行中のイベントループ内で新しいイベントループをネストできるようにします
nest_asyncio.apply()

groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logging.error("GROQ_API_KEY環境変数が設定されていません。")
    raise ValueError("GROQ_API_KEY環境変数が設定されていません。")

# 非同期Groqクライアントを初期化
async_groq_client = AsyncGroq(api_key=groq_api_key, timeout=600.0)
# 同期Groqクライアントを初期化
sync_groq_client = Groq(api_key=groq_api_key, timeout=600.0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class GroqConfig:
    get_output_model: str = "meta-llama/llama-guard-4-12b"
    rater_model: str = "llama-3.3-70b-versatile"
    max_tokens_get_output: int = 1024
    max_tokens_rater: int = 8192
    temperature_get_output: float = 0.0
    temperature_rater: float = 0.0


# プロンプト候補を評価するクラス
class Rater:
    def __init__(self) -> None:
        self.config = GroqConfig()

    def __call__(
        self, initial_prompt: str, candidates: List[Dict[str, str]], demo_data: Dict[str, str]
    ) -> Optional[int]:
        """
        複数のプロンプト候補を評価し、最も良いものを選択します。

        Args:
            initial_prompt (str): 元の指示プロンプト。
            candidates (List[Dict[str, str]]): 評価対象のプロンプト候補のリスト。
                                     各要素は {"prompt": "候補プロンプト"} の形式。
            demo_data (Dict[str, str]): デモデータ（キーと値のペア）。プロンプト内のプレースホルダを置換するために使用。

        Returns:
            Optional[int]: 最も評価の高かった候補のインデックス。エラー時はNone。
        """
        self._validate_inputs(initial_prompt, candidates, demo_data)

        # 既に評価済みの候補を特定
        unrated_candidates_indices = [i for i, candidate in enumerate(candidates) if "output" not in candidate]

        # 未評価の候補に対して非同期で出力を取得
        if unrated_candidates_indices:
            unrated_prompts = [
                self._replace_placeholders(candidates[i]["prompt"], demo_data) for i in unrated_candidates_indices  # 未評価のプロンプトをデモデータで置換
            ]

            # nest_asyncioが適用されているため、asyncio.run()を安全に呼び出せます
            outputs = asyncio.run(self._get_outputs_parallel(unrated_prompts))

            for i, output in zip(unrated_candidates_indices, outputs):
                candidates[i]["input"] = self._replace_placeholders(candidates[i]["prompt"], demo_data)
                if isinstance(output, str):
                    candidates[i]["output"] = output  # 候補の入力と出力を格納
                else:
                    candidates[i]["output"] = ""  # or handle error appropriately

        # 元の指示プロンプトもデモデータで置換
        initial_prompt_filled = self._replace_placeholders(initial_prompt, demo_data)

        # 評価を実行
        rate = self.rater(initial_prompt_filled, candidates)
        logging.info(f"Rater.__call__ return: {rate}")
        return rate

    def _validate_inputs(
        self, initial_prompt: str, candidates: List[Dict[str, str]], demo_data: Dict[str, str]
    ) -> None:
        """
        入力パラメータの検証を行います。
        """
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です")
        if not candidates:
            raise ValueError("候補が提供されていません")
        if not demo_data:
            raise ValueError("デモデータが提供されていません")
        for candidate in candidates:
            if "prompt" not in candidate or not candidate["prompt"].strip():
                raise ValueError("候補プロンプトが空または無効です")

    def _replace_placeholders(self, text: str, data: Dict[str, str]) -> str:
        """
        テキスト内のプレースホルダをデモデータで置換します。
        """
        for k, v in data.items():  # デモデータのキーと値のペアでプレースホルダを置換
            text = text.replace(k, v)
        return text

    async def _get_outputs_parallel(self, prompts: List[str]) -> List[Optional[str]]:
        """
        複数のプロンプトに対して非同期でGroqモデルを実行し、出力を取得します。
        """
        tasks = [self._get_output_async(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [res if isinstance(res, str) else None for res in results]

    async def _get_output_async(self, prompt: str) -> Optional[str]:
        """指定されたプロンプトでGroqモデルを非同期で実行し、出力を取得します。"""
        messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
        try:
            completion = await async_groq_client.chat.completions.create(  # Groq APIを呼び出し
                model=self.config.get_output_model,
                messages=messages,
                max_completion_tokens=self.config.max_tokens_get_output,
                temperature=self.config.temperature_get_output,
            )
            result = completion.choices[0].message.content
            if result is None:
                return None
            logging.info(f"Rater._get_output_async successful, result: {result}")
            return result
        except groq.InternalServerError as e:
            error_message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logging.error(f"Rater._get_output_async - Groq InternalServerError: {error_message} (Details: {e})")
            return None
        except groq.APIError as e:
            error_message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logging.error(f"Rater._get_output_async - Groq APIError: {error_message} (Details: {e})")
            return None
        except Exception as e:
            logging.error(f"Rater._get_output_async - Unexpected error: {e}")
            return None

    def rater(self, initial_prompt: str, candidates: List[Dict[str, str]]) -> Optional[int]:
        """
        Groqモデルを使用して、複数の候補応答の中から最も良いものを評価させます。

        Args:
            initial_prompt (str): 元の指示。
            candidates (List[Dict[str, str]]): 評価対象の候補。各要素は {"input": "入力プロンプト", "output": "モデル出力"} を含む。

        Returns:
            Optional[int]: 最も良いと評価された候補のインデックス。エラー時はNone。
        """
        if not candidates:
            logging.debug("Rater.rater - No candidates provided for LLM rating. Returning None.")
            return None

        rater_example: str = json.dumps({"Preferred": "Response 1"})
        Response_prompt: List[str] = []
        for candidate_idx, candidate in enumerate(candidates):
            # 各候補の情報を整形して評価用プロンプトに含めます
            Response_template: str = (
                f"""
                Response {candidate_idx+1}:
                Input: {candidate.get('input', 'N/A')}
                Output: {candidate.get('output', 'N/A')}
                </response_{candidate_idx+1}>
                """.strip()
            )
            Response_prompt.append(Response_template)
        Response_prompt_str: str = "\n\n".join(Response_prompt)

        # 評価のための指示プロンプトテンプレート
        rater_prompt: str = (
            """
            You are an expert rater of helpful and honest Assistant responses. Given the instruction and the two responses choose the most helpful and honest response.
            Please pay particular attention to the response formatting requirements called for in the instruction.

            Instruction:
            <instruction>
            {instruction}
            </instruction>

            {Response_prompt}

            Finally, select which response is the most helpful and honest.

            Use JSON format with key `Preferred` when returning results. Please only output the result in json format, and do the json format check and return, don't include other extra text! An example of output is as follows:
            Output example: {rater_example}
            """.strip()
        )
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": rater_prompt.format(
                    instruction=initial_prompt,
                    Response_prompt=Response_prompt_str,
                    rater_example=rater_example,
                ),
            },
        ]
        try:
            # Groq APIを呼び出して評価を実行します
            completion = sync_groq_client.chat.completions.create(
                model=self.config.rater_model,
                messages=messages,
                max_completion_tokens=self.config.max_tokens_rater,
                temperature=self.config.temperature_rater,
            )
            content = completion.choices[0].message.content
            if content is None:
                raise json.JSONDecodeError("No content from LLM", "", 0)
            result_json: Dict[str, str] = json.loads(content)
            final_result: Optional[int] = None
            for idx in range(len(candidates)):
                if str(idx + 1) in result_json["Preferred"]:
                    final_result = idx
                    break

            if final_result is None:
                logging.warning(
                    "Rater.rater - LLM did not return a clear preferred choice. Falling back to random choice."
                )
                final_result = random.randint(0, len(candidates) - 1)

            logging.info(f"Rater.rater successful, result: {final_result}")
            return final_result

        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Rater.rater - Error parsing LLM response or key error: {e}")
            logging.warning("Rater.rater - Falling back to random choice due to parsing error.")
            return random.randint(0, len(candidates) - 1) if candidates else None
        except Exception as e:
            logging.error(f"Rater.rater - Unexpected error during LLM rating: {e}")
            logging.warning("Rater.rater - Falling back to random choice due to unexpected error.")
            return random.randint(0, len(candidates) - 1) if candidates else None
