import logging
import json
import os
import re
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class GroqConfig:
    """Groq APIの設定を保持するデータクラス。"""

    metaprompt_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_tokens: int = 8192
    temperature: float = 0.0


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
        parsed_variables: List[str] = [v for v in variables.split("\n") if v]

        # メタプロンプトで使用する変数文字列を生成
        variable_string: str = ""
        for variable in parsed_variables:
            variable_string += "\n{{" + variable.upper() + "}}"

        # 基本的なメタプロンプトを読み込み、タスクで置き換える
        prompt: str = self.metaprompt.replace("{{TASK}}", task)
        # Prompt に日本語での書き換え指示を追加
        prompt += "\nPlease use Japanese for rewriting. The xml tag name is still in English."

        # API に送信するメッセージを準備。system ロールは使用しない
        assistant_partial: str = "<Inputs>"
        if variable_string:
            assistant_partial += variable_string + "\n</Inputs>\n<Instructions Structure>"

        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt},  # ユーザーの質問（メタプロンプト）
            {"role": "assistant", "content": assistant_partial},  # アシスタントの途中応答（変数リストと指示）
        ]

        # Groq API を呼び出してプロンプトを生成
        logging.info(f"MetaPrompt Request JSON: {json.dumps(messages, ensure_ascii=False, indent=2)}")

        # Groq API を呼び出して応答を生成
        # ロギング: API 呼び出しの詳細をログに記録
        logging.info(f"Calling Groq API with model: {self.config.metaprompt_model}")
        logging.debug(f"API Request: {messages}")

        # API 呼び出しと応答処理
        completion = self.groq_client.chat.completions.create(
            model=self.config.metaprompt_model,
            messages=messages, 
            max_completion_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        message: str = completion.choices[0].message.content or ""

        # ロギング: API 応答をログに記録
        logging.info("Received response from Groq API")
        logging.info(f"API Response: {message}")
        logging.debug(message)

        # API応答からプロンプトと変数を抽出
        extracted_prompt_template: str = self.extract_prompt(message)
        extracted_variables: Set[str] = self.extract_variables(message)

        # 抽出されたプロンプトと変数を返す
        return extracted_prompt_template.strip(), "\n".join(extracted_variables)

    # --- 入力検証 ---
    def _validate_inputs(self, task: str, variables: str) -> None:
        """
        入力パラメータの検証を行います。
        """
        if not task.strip():
            raise ValueError("タスクが空です")
        if not variables.strip():
            # 変数が空の場合でもエラーとしない（空リストとして扱う）
            raise ValueError("変数が空です")

    def extract_between_tags(self, tag: str, string: str, strip: bool = False) -> List[str]:
        """
        文字列内から指定されたタグに囲まれた部分を抽出します。

        Args:
            tag (str): 抽出対象のタグ名。
            string (str): 検索対象の文字列。
            strip (bool, optional): 抽出された文字列の前後の空白を削除するかどうか。デフォルトは False。

        Returns:
            List[str]: 抽出された文字列のリスト。
        """
        ext_list: List[str] = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
        if strip:
            ext_list = [e.strip() for e in ext_list]
        return ext_list

    def remove_empty_tags(self, text: str) -> str:
        """
        空のXMLタグ（例: <Tag></Tag>）をテキストから削除します。
        """
        return re.sub(r"\n<(\w+)>\s*</\1>\n", "", text, flags=re.DOTALL)

    def extract_prompt(self, metaprompt_response: str) -> str:
        """
        メタプロンプトの応答から主要な指示部分を抽出します。

        Args:
            metaprompt_response (str): モデルからのメタプロンプト応答。

        Returns:
            str: 抽出されたプロンプト指示。
        """
        # <Instructions> タグ内のコンテンツを取得し、前後の空白を削除します。
        instructions_list: List[str] = self.extract_between_tags("Instructions", metaprompt_response, strip=True)

        if instructions_list and instructions_list[0]:
            # 抽出されたコンテンツが空でなければ、それを返します。
            return instructions_list[0].strip()

        # タグが見つからないか、内容が空の場合の処理。
        logging.warning("API応答から<Instructions>タグが見つからないか、内容が空です。")
        return ""

    def extract_variables(self, prompt_content: str) -> Set[str]:
        """
        プロンプト文字列から {{変数名}} 形式の変数を抽出します。

        Args:
            prompt_content (str): 変数を抽出する対象のプロンプト文字列。

        Returns:
            Set[str]: 抽出された変数名のセット（重複なし）。
        """
        # {{変数名}}, {$変数名}, {変数名} (変数名の大文字・小文字を区別しない) にマッチする
        # 変数名のみを抽出し、{{...}} などの記号は除く
        # パターンの順序は重要で、{変数名} が {{変数名}} の一部にマッチしないようにする
        patterns: List[str] = [
            r"\{\{([^{}]+?)\}\}",  # {{変数名}}
            # r"\{\$([^{}]+?)\}",  # {$変数名}
            # r"\{([^{}]+?)\}",  # {変数名}
        ]
        variables: Set[str] = set()
        temp_string: str = prompt_content
        for pattern in patterns:
            matches: List[str] = re.findall(pattern, temp_string)
            for m_content in matches:
                variables.add(m_content)
            # Remove matched parts to avoid re-matching by simpler patterns
            temp_string = re.sub(pattern, "", temp_string)
        return variables
