import json
import os
import re

from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
# 環境変数を .env ファイルから読み込みます
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# メタプロンプトを生成し、関連情報を抽出するクラス
class MetaPrompt:
    def __init__(self):
        # 現在のスクリプトが配置されているディレクトリを取得します
        current_script_path = os.path.dirname(os.path.abspath(__file__))

        # metaprompt.txt へのフルパスを構築します
        prompt_guide_path = os.path.join(current_script_path, "metaprompt.txt")

        # metaprompt.txt を読み込みます
        with open(prompt_guide_path, "r", encoding="utf-8") as f:
            self.metaprompt = f.read()

        # Groq APIキーを取得し、クライアントを初期化します
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_api_key)

    def __call__(self, task, variables):
        """
        タスクと変数に基づいてメタプロンプトを生成し、プロンプトテンプレートと変数を抽出します。

        Args:
            task (str): ユーザーが定義したタスク。
            variables (str): 改行区切りの変数文字列。

        Returns:
            tuple: (抽出されたプロンプトテンプレート, 改行区切りの変数文字列)
        """
        variables = variables.split("\n")
        variables = [variable for variable in variables if len(variable)]
        # 変数リストをメタプロンプト用の文字列形式に変換します
        variable_string = ""
        for variable in variables:
            variable_string += "\n{$" + variable.upper() + "}"
        prompt = self.metaprompt.replace("{{TASK}}", task)
        prompt += "Please use Japanese for rewriting. The xml tag name is still in English." # 指示文を追加
        assistant_partial = "<Inputs>"
        if variable_string:
            assistant_partial += (
                variable_string + "\n</Inputs>\n<Instructions Structure>"
            )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_partial},
        ]
        # Groq APIを使用してメタプロンプトから指示を生成します
        completion = self.groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.0,
        )
        message = completion.choices[0].message.content

        # デバッグ用の整形出力関数
        def pretty_print(message):
            print(
                "\n\n".join(
                    "\n".join(
                        line.strip()
                        for line in re.findall(
                            r".{1,100}(?:\s+|$)", paragraph.strip("\n")
                        )
                    )
                    for paragraph in re.split(r"\n\n+", message)
                )
            )

        # pretty_print関数を呼び出して整形されたメッセージをコンソールに出力します
        # pretty_print(message)

        # 生成されたメッセージからプロンプトテンプレートと変数を抽出します
        extracted_prompt_template = self.extract_prompt(message)
        variables = self.extract_variables(message)

        return extracted_prompt_template.strip(), "\n".join(variables)

    def extract_between_tags(
        # 指定されたXMLタグ間のコンテンツを抽出するヘルパー関数
        self, tag: str, string: str, strip: bool = False
    ) -> list[str]:
        """
        文字列内から指定されたタグに囲まれた部分を抽出します。

        Args:
            tag (str): 抽出対象のタグ名。
            string (str): 検索対象の文字列。
            strip (bool, optional): 抽出された文字列の前後の空白を削除するかどうか。デフォルトは False。

        Returns:
            list[str]: 抽出された文字列のリスト。
        """
        ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
        if strip:
            ext_list = [e.strip() for e in ext_list]
        return ext_list

    def remove_empty_tags(self, text):
        return re.sub(r"\n<(\w+)>\s*</\1>\n", "", text, flags=re.DOTALL)
        # 空のXMLタグ（例: <Tag></Tag>）をテキストから削除します。

    def extract_prompt(self, metaprompt_response):
        """
        メタプロンプトの応答から主要な指示部分を抽出します。

        Args:
            metaprompt_response (str): モデルからのメタプロンプト応答。

        Returns:
            str: 抽出されたプロンプト指示。
        """
        # <Instructions> タグ内のコンテンツを取得し、前後の空白を削除します。
        instructions_list = self.extract_between_tags("Instructions", metaprompt_response, strip=True)

        if instructions_list and instructions_list[0]:
            # 抽出されたコンテンツが空でなければ、それを返します。
            return instructions_list[0].strip()

        # タグが見つからないか、内容が空の場合の処理。
        # 空文字列を返すか、あるいは必要に応じてprint文で警告を出します。
        print("警告: API応答から<Instructions>タグが見つからないか、内容が空です。")
        return ""

    def extract_variables(self, prompt):
        """
        プロンプト文字列から {{変数名}} 形式の変数を抽出します。

        Args:
            prompt (str): 変数を抽出する対象のプロンプト文字列。

        Returns:
            set: 抽出された変数名のセット（重複なし）。
        """
        # Matches {{VAR}}, {$VAR}, and {VAR} (case-insensitive for VAR content)
        # Extracts only the content VAR.
        # Order of patterns matters to avoid {{VAR}} being matched by {VAR} partially.
        patterns = [
            r"\{\{([^{}]+?)\}\}",  # {{VAR}}
            r"\{\$([^{}]+?)\}",   # {$VAR} (extracts VAR from {$VAR})
            r"\{([^{}]+?)\}",    # {VAR}
        ]
        variables = set()
        temp_string = prompt
        for pattern in patterns:
            matches = re.findall(pattern, temp_string)
            for m_content in matches:
                variables.add(m_content)
            # Remove matched parts to avoid re-matching by simpler patterns
            temp_string = re.sub(pattern, "", temp_string)
        return variables

# テスト用のコード (現在はコメントアウト)
# test = MetaPrompt()
# TASK = "Draft an email responding to a customer complaint" # Replace with your task!
# VARIABLES = ["CUSTOMER_COMPLAINT", "COMPANY_NAME"]
# test(TASK, VARIABLES)
