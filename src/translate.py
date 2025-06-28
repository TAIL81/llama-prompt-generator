import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

# 環境変数を .env ファイルから読み込みます
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class GroqConfig:
    rewrite_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    detect_lang_model: str = "llama-3.1-8b-instant"
    judge_model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 8192
    temperature_rewrite: float = 0.2
    temperature_detect_lang: float = 0.0
    temperature_judge: float = 0.0


# 現在のスクリプトが配置されているディレクトリを取得します
current_script_path = os.path.dirname(os.path.abspath(__file__))

# PromptGuide.md へのフルパスを構築します
prompt_guide_path = os.path.join(current_script_path, "PromptGuide.md")


@lru_cache(maxsize=1)
def load_prompt_guide(path: str) -> str:
    """
    PromptGuide.md ファイルを読み込み、キャッシュします。
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


PromptGuide = load_prompt_guide(prompt_guide_path)


# プロンプトガイドに基づいてプロンプトを書き換えるクラス
class GuideBased:
    def __init__(self) -> None:
        groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.error("GROQ_API_KEY環境変数が設定されていません。")
            raise ValueError("GROQ_API_KEY環境変数が設定されていません。")
        self.groq_client = Groq(api_key=groq_api_key)
        self.config = GroqConfig()

    def __call__(self, initial_prompt: str) -> str:
        """
        初期プロンプトをプロンプトガイドに基づいて書き換えます。
        言語を検出し、適切な言語で書き換えを行うよう指示します。

        Args:
            initial_prompt (str): 書き換え対象の初期プロンプト。

        Returns:
            str: 書き換えられたプロンプト。""を返却することはほぼないと思われる
        """
        self._validate_initial_prompt(initial_prompt)
        lang: str = self.detect_lang(initial_prompt)
        # 検出された言語に応じて、プロンプトの言語指示を設定します
        if lang == "ja":
            lang_prompt: str = "Please use Japanese for rewriting. The xml tag name is still in English."
        elif lang == "ch":
            lang_prompt = "Please use Chinese for rewriting. The xml tag name is still in English."
        elif lang == "en":
            lang_prompt = "Please use English for rewriting."
        else:
            lang_prompt = "Please use same language as the initial instruction for rewriting. The xml tag name is still in English."  # 初期指示と同じ言語を使用

        # プロンプト書き換えのための指示テンプレート
        prompt: str = (
            """
            You are a instruction engineer. Your task is to rewrite the initial instruction in <initial_instruction></initial_instruction> xml tag based on the suggestions in the instruction guide in <instruction_guide></instruction_guide> xml tag.
            This instruction is then sent to Llama to get the expected output.

            <instruction_guide>
            {guide}
            </instruction_guide>

            You are a instruction engineer. Your task is to rewrite the initial instruction in <initial_instruction></initial_instruction> xml tag based on the suggestions in the instruction guide in <instruction_guide></instruction_guide> xml tag.
            This instruction is then sent to claude to get the expected output.

            Here are some important rules for rewrite:
            1. Something like `{{variable}}` is customizable text that will be replaced when sent to Llama. It needs to be retained in the rewrite.
            2. {lang_prompt}
            3. Only output the rewrite instruction return them in <rerwited></rerwited>XML tags
            4. If examples are already included in the initial prompt, do not remove the examples after the rewrite.

            You are a instruction engineer. Your task is to rewrite the initial instruction in <initial_instruction></initial_instruction> xml tag based on the suggestions in the instruction guide in <instruction_guide></instruction_guide> xml tag.
            This instruction is then sent to claude to get the expected output.

            Example:
            <initial_instruction>
            You are a research assistant. You will answer the following question based on the document in triple quotes, if the question cannot be answered please output "Cannot answer the question from the document"
            ```
            {{full_text}}
            ```
            You will also need to find the original quote from the document that is most relevant to answering the question. If there is no relevant citation, output "No relevant quotes".
            Your output should start by listing all the quotes, putting one quote per line and starting with a numerical index. Then answer the question by adding the index of the quote where it is needed.

            The question is:
            {{question}}
            </initial_instruction>

            <rerwited>
            You are an expert research assistant. Here is a document you will answer questions about:
            <doc>
            {{full_text}}
            </doc>

            First, find the quotes from the document that are most relevant to answering the question, and then print them in numbered order. Quotes should be relatively short.

            If there are no relevant quotes, write "No relevant quotes" instead.

            Then, answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer. Don't say "According to Quote [1]" when answering. Instead make references to quotes relevant to each section of the answer solely by adding their bracketed numbers at the end of relevant sentences.

            Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly.
            <example>
            Quotes:
            [1] "Company X reported revenue of $12 million in 2021."
            [2] "Almost 90% of revenue came from widget sales, with gadget sales making up the remaining 10%."

            Answer:
            Company X earned $12 million. [1] Almost 90% of it was from widget sales. [2]
            </example>

            If the question cannot be answered by the document, say "Cannot answer the question from the document".

            <question>
            {{question}}
            </question>
            </rerwited>

            <initial_instruction>
            {initial}
            </initial_instruction>
            """.strip()  # プロンプトテンプレートの終わり
        )

        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": prompt.format(guide=PromptGuide, initial=initial_prompt, lang_prompt=lang_prompt),
            },
        ]  # LLMへのメッセージ
        # Groq APIを使用してプロンプトの書き換えをリクエストします
        completion = self.groq_client.chat.completions.create(
            model=self.config.rewrite_model,
            messages=messages,  # メッセージ
            max_completion_tokens=self.config.max_tokens,  # 最大トークン数
            temperature=self.config.temperature_rewrite,  # 温度
        )
        result: str = completion.choices[0].message.content  # LLMの応答
        # LLMからの応答をデバッグ出力
        logging.debug(f"__call__ LLM response: \n{result}\n")
        # 結果から不要なXMLタグを除去します
        if result.startswith("<instruction>"):  # 開始タグを除去
            result = result[13:]
        if result.endswith("</instruction>"):  # 終了タグを除去
            result = result[:-14]
        result = result.strip()
        return result

    def _validate_initial_prompt(self, initial_prompt: str) -> None:
        """
        initial_promptの入力検証を行います。
        """
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です")

    def detect_lang(self, initial_prompt: str) -> str:
        """
        与えられたプロンプトの言語を検出します (英語、中国語または日本語)。

        Args:
            initial_prompt (str): 言語を検出する対象のプロンプト。

        Returns:
            str: 検出された言語コード ("en", "ch" または "ja")。エラーの場合は空文字列。
        """
        lang_example: str = json.dumps({"lang": "ja"})
        prompt: str = (
            """
            Please determine what language the document below is in? English (en), Chinese (ch) or Japanese (ja)?

            <document>
            {document}
            </document>

            Use JSON format with key `lang` when return result. Please only output the result in json format, and do the json format check and return, don't include other extra text! An example of output is as follows:
            Output example: {lang_example}
            """.strip()  # プロンプトテンプレートの終わり
        )
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": prompt.format(document=initial_prompt, lang_example=lang_example),
            },
        ]
        # Groq APIを使用して言語検出をリクエストします
        completion = self.groq_client.chat.completions.create(
            model=self.config.detect_lang_model,
            messages=messages,
            max_completion_tokens=self.config.max_tokens,
            temperature=self.config.temperature_detect_lang,
        )  # API呼び出し
        # LLMからの応答をデバッグ出力
        logging.debug(f"detect_lang LLM response: {completion.choices[0].message.content}")
        try:
            # 結果のJSONをパースして言語コードを取得します
            lang: str = json.loads(completion.choices[0].message.content)["lang"]  # 言語コードを抽出
        except Exception as e:
            # エラーが発生した場合は空文字列を返します
            logging.error(f"Error detecting language: {e}")
            lang = ""  # エラー時は空文字列
        return lang

    def judge(self, candidates: List[str]) -> Optional[int]:
        """ # ドキュメント文字列
        複数のプロンプト候補を評価し、最も良いものを選択します。

        Args:
            candidates (List[str]): 評価対象のプロンプト候補のリスト。

        Returns:
            Optional[int]: 最も良いと判断された候補のインデックス。エラーの場合はNone。
        """
        Instruction_prompts: List[str] = []
        for idx, candidate in enumerate(candidates):  # 各候補を処理
            Instruction_prompts.append(f"Instruction {idx+1}:\n<instruction>\n{candidate}\n</instruction>")
        example: str = json.dumps({"Preferred": "Instruction 1"})
        # プロンプト評価のための指示テンプレート
        prompt: str = (
            """
            You are an instruction engineer. Your task is to evaluate which of the three instructions given below is better based on the guide in the <guide> xml tag.

            Instruction guide:
            <guide>
            {guide}
            </guide>

            {Instruction_prompts}

            Use JSON format when returning results. Please only output the result in json format, and do the json format check and return, don't include other extra text! An example of output is as follows:
            {example}
            """.strip()  # プロンプトテンプレートの終わり
        )
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": prompt.format(
                    guide=PromptGuide,
                    Instruction_prompts="\n\n".join(Instruction_prompts),
                    example=example,
                ),
            },
        ]
        # Groq APIを使用してプロンプト候補の評価をリクエストします
        completion = self.groq_client.chat.completions.create(
            model=self.config.judge_model,
            messages=messages,
            max_completion_tokens=self.config.max_tokens,
            temperature=self.config.temperature_judge,
        )  # API呼び出し
        # LLMからの応答をデバッグ出力
        logging.debug(f"judge LLM response (raw): {completion.choices[0].message.content}")
        final_result: Optional[int] = None
        try:
            result: Dict[str, str] = json.loads(completion.choices[0].message.content)  # JSONをパース
            # 結果のJSONから優先される指示の番号を抽出し、インデックスに変換します
            for idx in range(len(candidates)):  # candidatesの長さに合わせてループ
                if str(idx + 1) in result["Preferred"]:  # 優先される候補を特定
                    final_result = idx
                    break
        except (json.JSONDecodeError, KeyError) as e:
            # JSONパースエラーなどが発生した場合はNoneのまま
            logging.error(f"Error parsing judge LLM response: {e}")
            pass  # 何もしない
        except Exception as e:
            logging.error(f"Unexpected error in judge method: {e}")
            pass  # 何もしない
        return final_result
