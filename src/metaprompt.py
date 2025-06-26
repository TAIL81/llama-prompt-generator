import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from groq import Groq

# 環境変数を .env ファイルから読み込みます
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class GroqConfig:
    metaprompt_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_tokens: int = 8192
    temperature: float = 0.1


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

        parsed_variables: List[str] = [v for v in variables.split("\n") if v]
        # 変数リストをメタプロンプト用の文字列形式に変換します
        variable_string: str = ""
        for variable in parsed_variables:
            variable_string += "\n{{" + variable.upper() + "}}"

        prompt: str = self.metaprompt.replace("{{TASK}}", task)
        prompt += "Please use Japanese for rewriting. The xml tag name is still in English."  # 指示文を追加

        assistant_partial: str = "<Inputs>"
        if variable_string:
            assistant_partial += variable_string + "\n</Inputs>\n<Instructions Structure>"

        messages: List[Dict[str, str]] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_partial},
        ]

        # Groq APIを使用してメタプロンプトから指示を生成します
        completion = self.groq_client.chat.completions.create(
            model=self.config.metaprompt_model,
            messages=messages,
            max_completion_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        message: str = completion.choices[0].message.content

        # デバッグ用の整形出力関数
        def pretty_print(message_content: str) -> None:
            """
            この関数は、与えられたメッセージを整形し、コンソールに出力します。
            長い行を100文字で折り返し、段落ごとに改行を挿入して可読性を高めます。
            """
            print(
                "\n\n".join(
                    "\n".join(line.strip() for line in re.findall(r".{1,100}(?:\s+|$)", paragraph.strip("\n")))
                    for paragraph in re.split(r"\n\n+", message_content)
                )
            )

        # pretty_print関数を呼び出して整形されたメッセージをコンソールに出力します
        pretty_print(message)

        # 生成されたメッセージからプロンプトテンプレートと変数を抽出します
        extracted_prompt_template: str = self.extract_prompt(message)
        extracted_variables: Set[str] = self.extract_variables(message)

        return extracted_prompt_template.strip(), "\n".join(extracted_variables)

    def _validate_inputs(self, task: str, variables: str) -> None:
        """
        入力パラメータの検証を行います。
        """
        if not task.strip():
            raise ValueError("タスクが空です")
        if not variables.strip():
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
            r"\{\$([^{}]+?)\}",  # {$変数名}
            r"\{([^{}]+?)\}",  # {変数名}
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
