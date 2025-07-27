import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam

# 環境変数を .env ファイルから読み込みます
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class GroqConfig:
    """Groq APIの設定を保持するデータクラス。"""

    metaprompt_model: str = "compound-beta-kimi"
    max_tokens: int = 8192
    temperature: float = 0.6


# 現在のスクリプトが配置されているディレクトリを取得します
current_script_path = os.path.dirname(os.path.abspath(__file__))

# metaprompt.txt へのフルパスを構築します
metaprompt_txt_path = os.path.join(current_script_path, "metaprompt.txt")


@lru_cache(maxsize=1)
def load_metaprompt_content(path: str) -> str:
    """
    metaprompt.txt ファイルを読み込み、キャッシュします。
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# メタプロンプトを生成し、関連情報を抽出するクラス
class MetaPrompt:
    def __init__(self) -> None:
        self.metaprompt: str = load_metaprompt_content(metaprompt_txt_path)
        """
        MetaPromptクラスの初期化。

        - metaprompt.txtからメタプロンプトの内容を読み込む。
        - 環境変数からGROQ_API_KEYを取得し、Groqクライアントを初期化する。
        - GroqConfigからAPI設定を読み込む。
        """
        groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.error("GROQ_API_KEY環境変数が設定されていません。")
            raise ValueError("GROQ_API_KEY環境変数が設定されていません。")
        self.groq_client = Groq(api_key=groq_api_key)
        self.config = GroqConfig()

    def __call__(self, task: str, variables: str) -> Tuple[str, str]:
        """
        タスクと変数に基づいてメタプロンプトを生成し、プロンプトテンプレートと変数を抽出します。

        Args:
            task (str): ユーザーが定義したタスク。
            variables (str): 改行区切りの変数文字列。

        Returns:
            Tuple[str, str]: (抽出されたプロンプトテンプレート, 改行区切りの変数文字列)
        """
        self._validate_inputs(task, variables)

        # 入力された改行区切りの変数文字列をリストに変換
        parsed_variables: List[str] = [
            v.strip() for v in variables.split("\n") if v.strip()
        ]

        # メタプロンプトで使用する変数文字列を生成
        variable_string: str = "\n".join(
            [f"{{{{{v.upper()}}}}}" for v in parsed_variables]
        )

        # 基本的なメタプロンプトを読み込み、タスクと変数を挿入
        prompt: str = self.metaprompt.replace("{{TASK}}", task)
        prompt = prompt.replace("{{VARIABLES}}", variable_string)

        # JSONモードで応答を要求する指示を追加
        prompt += '\n\nPlease provide the rewritten prompt in a JSON object with two keys: "prompt_template" and "variables". The "variables" key should contain a list of all variables found in the "prompt_template".'
        prompt += "\nPlease use Japanese for rewriting."

        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt},
        ]

        logging.info(
            f"MetaPrompt Request JSON: {json.dumps(messages, ensure_ascii=False, indent=2)}"
        )
        logging.info(f"Calling Groq API with model: {self.config.metaprompt_model}")

        message: str = ""  # この行をtryブロックの外に移動

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.config.metaprompt_model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
            )
            message = completion.choices[0].message.content or ""

            logging.info("Received response from Groq API")
            logging.info(f"API Response: {message}")

            # API応答（JSON文字列）をパース
            response_data = json.loads(message)
            extracted_prompt_template: str = response_data.get("prompt_template", "")
            extracted_variables: List[str] = response_data.get("variables", [])

            return extracted_prompt_template.strip(), "\n".join(extracted_variables)

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response: {e}")
            logging.error(f"Raw response: {message}")
            return "Error: Failed to parse response.", ""
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return f"Error: {e}", ""

    # --- 入力検証 ---
    def _validate_inputs(self, task: str, variables: str) -> None:
        """
        入力パラメータの検証を行います。
        """
        if not task.strip():
            raise ValueError("タスクが空です")
        if not variables.strip():
            # 変数が空の場合でもエラーとせず、警告をログに出力して処理を続行
            logging.warning("変数が空ですが、処理を続行します。")
