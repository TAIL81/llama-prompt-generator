import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam

# .env 読み込み（アプリ側と二重設定しない前提で軽量に実行）
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# モジュールロガー（basicConfigはアプリ側で設定）
logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """Groq API の設定"""

    metaprompt_model: str = "moonshotai/kimi-k2-instruct"
    max_tokens: int = 16384
    temperature: float = 0.3


# 現在のスクリプトディレクトリとファイルパス
current_script_path = os.path.dirname(os.path.abspath(__file__))
metaprompt_txt_path = os.path.join(current_script_path, "metaprompt.txt")


@lru_cache(maxsize=1)
def load_metaprompt_content(path: str) -> str:
    """metaprompt.txt を読み込みキャッシュする。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class MetaPrompt:
    """
    メタプロンプト生成器:
    - タスク/変数からメタプロンプトを構築
    - Groq の JSON モード応答からテンプレートと変数一覧を抽出
    """

    def __init__(self) -> None:
        self.metaprompt: str = load_metaprompt_content(metaprompt_txt_path)

        groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY 環境変数が設定されていません。")
            raise ValueError("GROQ_API_KEY 環境変数が設定されていません。")

        self.groq_client = Groq(api_key=groq_api_key)
        self.config = GroqConfig()

    def __call__(self, task: str, variables: str) -> Tuple[str, str]:
        """
        タスクと変数（改行区切り）からプロンプトテンプレートと変数名一覧を抽出。
        戻り値: (prompt_template, "変数\n改行区切り")
        """
        self._validate_inputs(task, variables)

        parsed_variables: List[str] = [
            v.strip() for v in variables.split("\n") if v.strip()
        ]
        variable_string: str = "\n".join(
            f"{{{{{v.upper()}}}}}" for v in parsed_variables
        )

        prompt: str = self.metaprompt.replace("{{TASK}}", task).replace(
            "{{VARIABLES}}", variable_string
        )
        prompt += '\n\nPlease provide the rewritten prompt in a JSON object with two keys: "prompt_template" and "variables". The "variables" key should contain a list of all variables found in the "prompt_template".'
        prompt += "\nPlease use Japanese for rewriting."

        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]

        logger.debug("MetaPrompt リクエスト: %s", messages)
        logger.info("Groq API 呼び出し: model=%s", self.config.metaprompt_model)

        raw_response: str = ""
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.config.metaprompt_model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
            )
            raw_response = completion.choices[0].message.content or ""
            logger.info("Groq API 応答を受信")
            logger.debug("API 応答(raw): %s", raw_response)

            data = json.loads(raw_response)
            prompt_template: str = data.get("prompt_template", "") or ""
            extracted_variables: List[str] = data.get("variables", []) or []

            return prompt_template.strip(), "\n".join(extracted_variables)

        except json.JSONDecodeError as e:
            logger.error("JSON デコード失敗: %s", e)
            logger.error("Raw 応答: %s", raw_response)
            return "Error: Failed to parse response.", ""
        except Exception as e:
            logger.error("想定外エラー: %s", e)
            return f"Error: {e}", ""

    # 入力検証
    def _validate_inputs(self, task: str, variables: str) -> None:
        """task/variables の簡易検証。"""
        if not task.strip():
            raise ValueError("タスクが空です。")
        if not variables.strip():
            logger.warning("変数が空ですが、処理を続行します。")
