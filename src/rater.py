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

# --- 初期化処理 ---
# Groq APIキーを環境変数から取得します。
# 環境変数に設定されていない場合はエラーを発生させます。
groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logging.error("GROQ_API_KEY環境変数が設定されていません。")
    raise ValueError("GROQ_API_KEY環境変数が設定されていません。")

# 同期Groqクライアントの初期化。
sync_groq_client = Groq(api_key=groq_api_key, timeout=600.0)

# ロギングの基本設定。
# INFOレベル以上のメッセージをコンソールに出力します。
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Pydanticモデル ---
class PreferredResponse(BaseModel):
    """プロンプト評価APIのレスポンスを定義するPydanticモデル"""

    Preferred: str = Field(
        ...,
        description="The name of the preferred response (e.g., 'Response 1').",
        examples=["Response 1"],
    )


# --- データクラス ---
@dataclass
class GroqConfig:
    """
    Groqモデルの設定を保持するデータクラス。
    モデル名、最大トークン数、温度などのパラメータを定義します。
    """

    # 出力生成に使用するモデル名
    get_output_model: str = "compound-beta-mini"
    # 評価（レーティング）に使用するモデル名
    rater_model: str = "llama-3.3-70b-versatile"
    # 出力生成時の最大トークン数
    max_tokens_get_output: int = 8192
    # 評価時の最大トークン数
    max_tokens_rater: int = 8192
    # 出力生成時の温度（ランダム性）
    temperature_get_output: float = 0.7
    # 評価時の温度（ランダム性）
    temperature_rater: float = 0.7


# --- メインクラス ---
class Rater:
    """
    Groqモデルを使用してプロンプト候補を評価し、最適なものを選択するクラス。
    """

    def __init__(self) -> None:
        """
        Raterクラスのコンストラクタ。
        GroqConfigのインスタンスを初期化します。
        """
        self.config = GroqConfig()

    def __call__(
        self,
        initial_prompt: str,
        candidates: List[Dict[str, str]],
        demo_data: Dict[str, str],
    ) -> Optional[int]:
        """
        複数のプロンプト候補を評価し、最も良いものを選択します。
        このメソッドは同期で実行されます。

        Args:
            initial_prompt (str): 評価の基準となる初期プロンプト。
            candidates (List[Dict[str, str]]): 評価対象となるプロンプト候補のリスト。
                                                各辞書は'prompt'キーを含む必要があります。
            demo_data (Dict[str, str]): プロンプト内のプレースホルダを置換するためのデモデータ。

        Returns:
            Optional[int]: 最も良いと評価された候補のインデックス（0-based）。
                           評価に失敗した場合はNoneを返します。
        """
        # 入力パラメータの検証
        self._validate_inputs(initial_prompt, candidates, demo_data)

        # まだ出力が生成されていない候補のインデックスを特定
        unrated_indices = [i for i, c in enumerate(candidates) if "output" not in c]
        if unrated_indices:
            # 未評価のプロンプトに対してプレースホルダを置換
            unrated_prompts = [
                self._replace_placeholders(candidates[i]["prompt"], demo_data)
                for i in unrated_indices
            ]
            # 逐次的にGroqモデルから出力を取得
            outputs = self._get_outputs_sequentially(unrated_prompts)
            # 取得した出力を対応する候補に格納
            for i, output in zip(unrated_indices, outputs):
                candidates[i]["input"] = self._replace_placeholders(
                    candidates[i]["prompt"], demo_data
                )
                candidates[i]["output"] = output if isinstance(output, str) else ""

        # 初期プロンプトのプレースホルダを置換
        initial_prompt_filled = self._replace_placeholders(initial_prompt, demo_data)
        # 評価モデルを使用して最適な候補を決定
        rate = self.rater(initial_prompt_filled, candidates)
        logging.debug(f"Rater.__call__ return: {rate}")
        return rate

    def _validate_inputs(
        self,
        initial_prompt: str,
        candidates: List[Dict[str, str]],
        demo_data: Dict[str, str],
    ) -> None:
        """
        入力パラメータの検証を行います。
        空のプロンプト、候補リスト、またはデモデータが渡された場合にValueErrorを発生させます。

        Args:
            initial_prompt (str): 検証する初期プロンプト。
            candidates (List[Dict[str, str]]): 検証するプロンプト候補のリスト。
            demo_data (Dict[str, str]): 検証するデモデータ。

        Raises:
            ValueError: いずれかの入力が空または無効な場合。
        """
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
        """
        テキスト内のプレースホルダ（例: `{{VARIABLE_NAME}}`）をデモデータで置換します。

        Args:
            text (str): プレースホルダを含む元のテキスト。
            data (Dict[str, str]): プレースホルダと置換する値の辞書。

        Returns:
            str: プレースホルダが置換されたテキスト。
        """
        for k, v in data.items():
            text = text.replace(k, v)
        return text

    def _get_outputs_sequentially(self, prompts: List[str]) -> List[Optional[str]]:
        """
        複数のプロンプトに対してGroqモデルを逐次実行し、出力を取得します。

        Args:
            prompts (List[str]): 出力を取得するプロンプトのリスト。

        Returns:
            List[Optional[str]]: 各プロンプトに対応するモデル出力のリスト。
                                 エラーが発生した場合はNoneが含まれます。
        """
        results = []
        for prompt in prompts:
            try:
                results.append(self._get_output_sync(prompt))
            except Exception as e:
                logging.error(f"Error in sequential Groq call: {e}")
                results.append(None)
        return results

    def _get_output_sync(self, prompt: str) -> Optional[str]:
        """
        指定されたプロンプトでGroqモデルを同期で実行し、出力を取得します。
        レートリミットエラーに対してリトライメカニズムを実装しています。

        Args:
            prompt (str): 実行するプロンプト。

        Returns:
            Optional[str]: モデルの出力文字列。エラーが発生した場合はNone。
        """
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        max_retries = 3  # 最大リトライ回数
        backoff_factor = 2  # リトライ遅延のバックオフ係数
        retry_delay = 1  # 初期リトライ遅延（秒）

        # 指定された回数だけリトライを試みるループ
        for attempt in range(max_retries):
            try:
                # Groq APIを同期で呼び出し
                completion = sync_groq_client.chat.completions.create(
                    model=self.config.get_output_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_tokens_get_output,
                    temperature=self.config.temperature_get_output,
                )
                result = completion.choices[0].message.content
                logging.debug(f"Rater._get_output_sync successful, result: {result}")
                return result
            except RateLimitError:
                # レートリミットエラーの場合、警告をログに出力し、指定された時間待機してからリトライ
                logging.warning(
                    f"Rate limit exceeded. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}"
                )
                time.sleep(retry_delay)
                retry_delay *= backoff_factor  # 遅延時間を指数関数的に増加
            except APIError as e:
                # Groq APIからのエラーの場合、エラーをログに出力し、Noneを返す
                logging.error(f"Rater._get_output_sync - Groq APIError: {e}")
                return None
            except Exception as e:
                # その他の予期せぬエラーの場合、エラーをログに出力し、Noneを返す
                logging.error(f"Rater._get_output_sync - Unexpected error: {e}")
                return None
        # 最大リトライ回数に達しても成功しなかった場合
        logging.error(
            "Max retries reached for _get_output_sync. Failed to get a response from Groq API."
        )
        return None

    def rater(
        self, initial_prompt: str, candidates: List[Dict[str, str]]
    ) -> Optional[int]:
        """
        Groqモデルを使用して、複数の候補応答の中から最も良いものを評価させます。
        GroqのStructured Outputs機能を使用して、信頼性の高いJSON出力を保証します。

        Args:
            initial_prompt (str): 評価の基準となる初期プロンプト。
            candidates (List[Dict[str, str]]): 評価対象となるプロンプト候補のリスト。
                                                各辞書は'input'と'output'キーを含む必要があります。

        Returns:
            Optional[int]: 最も良いと評価された候補のインデックス（0-based）。
                           評価に失敗した場合はNoneを返します。
        """
        if not candidates:
            logging.debug("Rater.rater - No candidates provided. Returning None.")
            return None

        # 各候補プロンプトの入力と出力を整形して、評価プロンプトに含める文字列を作成
        response_prompts = [
            f"Response {i+1}:\nInput: {c.get('input', 'N/A')}\nOutput: {c.get('output', 'N/A')}\n</response_{i+1}>".strip()
            for i, c in enumerate(candidates)
        ]
        response_prompt_str = "\n\n".join(response_prompts)

        # 評価モデルに与えるプロンプトテンプレート
        rater_prompt = """
        You are an expert rater of helpful and honest Assistant responses. Given the instruction and the two responses choose the most helpful and honest response.
        Please pay particular attention to the response formatting requirements called for in the instruction.

        Instruction:
        <instruction>
        {instruction}
        </instruction>

        {Response_prompt}

        Finally, select which response is the most helpful and honest.
        Your response must be only the JSON object, with no other text before or after it.
        """.strip()

        # 評価モデルへのメッセージリストを構築
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": rater_prompt.format(
                    instruction=initial_prompt,
                    Response_prompt=response_prompt_str,
                ),
            }
        ]

        max_retries = 3
        backoff_factor = 2
        retry_delay = 1

        for attempt in range(max_retries):
            content = None  # 変数を事前に初期化
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
                    logging.error("No content from LLM")
                    continue  # リトライ

                # Pydanticモデルでレスポンスを検証・パース
                result = PreferredResponse.model_validate(json.loads(content))
                preferred_text = result.Preferred

                # "Response N" から "N" を抽出
                match = re.search(r"\d+", preferred_text)
                if match:
                    final_result = int(match.group(0)) - 1
                    if 0 <= final_result < len(candidates):
                        logging.debug(f"Rater.rater successful, result: {final_result}")
                        return final_result

                logging.error(
                    f"Could not parse a valid index from LLM response: '{preferred_text}'"
                )
                # 不正なレスポンスでも、スキーマには合致しているためリトライしない
                return None

            except json.JSONDecodeError as e:
                # contentがNoneの場合のハンドリングを追加
                error_msg = f"JSONパースエラー: {e}"
                if content is not None:
                    error_msg += f"\nAPIレスポンス: {content}"
                else:
                    error_msg += "\nAPIレスポンス: なし"
                logging.error(error_msg)
                # JSONモードでも失敗する場合があるのでリトライ
                if attempt >= max_retries - 1:
                    return None
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except ValidationError as e:
                logging.error(f"Pydantic検証エラー: {e}\nAPIレスポンス: {content}")
                # スキーマ違反はリトライしても治らない可能性が高い
                return None
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
