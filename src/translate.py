import json
import logging
import os
import re  # 正規表現用モジュールを追加
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# 環境変数を .env ファイルから読み込みます
env_path = Path(__file__).parent.parent / ".env"  # .envファイルのパスを構築
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class GroqConfig:
    rewrite_model: str = (
        "meta-llama/llama-4-scout-17b-16e-instruct"  # プロンプト書き換えに使用するモデル
    )
    detect_lang_model: str = (
        "meta-llama/llama-4-scout-17b-16e-instruct"  # 言語検出に使用するモデル
    )
    judge_model: str = (
        "meta-llama/llama-4-scout-17b-16e-instruct"  # 評価に使用するモデル
    )
    max_tokens: int = 8192  # モデルからの応答の最大トークン数
    temperature_rewrite: float = 0.6
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
    with open(path, "r", encoding="utf-8") as f:  # ファイルを読み込みモードで開く
        return f.read()


PromptGuide = load_prompt_guide(prompt_guide_path)  # プロンプトガイドの内容を読み込む


# プロンプトガイドに基づいてプロンプトを書き換えるクラス
class GuideBased:
    def __init__(self) -> None:
        groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logging.error("GROQ_API_KEY環境変数が設定されていません。")
            raise ValueError("GROQ_API_KEY環境変数が設定されていません。")
        self.groq_client = Groq(api_key=groq_api_key)
        self.config = GroqConfig()  # GroqConfigのインスタンスを作成

    def __call__(self, initial_prompt: str) -> str:
        """
        初期プロンプトをプロンプトガイドに基づいて書き換えます。
        言語を検出し、適切な言語で書き換えを行うよう指示します。

        Args:
            initial_prompt (str): 書き換え対象の初期プロンプト。

        Returns:
            str: 書き換えられたプロンプト。""を返却することはほぼないと思われる
        """
        self._validate_initial_prompt(initial_prompt)  # 初期プロンプトの検証
        lang: str = self.detect_lang(initial_prompt)
        lang_prompt: str = ""  # Initialize with empty string
        # 検出された言語に応じて、プロンプトの言語指示を設定します
        if lang == "ja":
            lang_prompt = "Please use Japanese for rewriting. The xml tag name is still in English."
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

        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": prompt.format(
                    guide=PromptGuide, initial=initial_prompt, lang_prompt=lang_prompt
                ),
            },
        ]  # LLMへのメッセージ
        max_retries = 3
        backoff_factor = 2
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                # Groq APIを使用してプロンプトの書き換えをリクエストします
                completion = self.groq_client.chat.completions.create(
                    model=self.config.rewrite_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_tokens,  # 最大トークン数
                    temperature=self.config.temperature_rewrite,  # 温度
                )
                result: str = (  # LLMの応答を取得
                    completion.choices[0].message.content or ""
                )  # LLMの応答 (handle None)
                # LLMからの応答をログに出力
                logging.info(f"__call__ LLM response: \n{result}\n")
                # 結果から不要なXMLタグを除去します
                if result.startswith("<instruction>"):  # 開始タグを除去
                    result = result[13:]
                if result.endswith("</instruction>"):  # 終了タグを除去
                    result = result[:-14]
                result = result.strip()
                return result
            except RateLimitError as e:  # レート制限エラーが発生した場合
                logging.warning(
                    f"Rate limit exceeded in __call__. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}"
                )
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:  # Groq APIエラーが発生した場合
                logging.error(f"GuideBased.__call__ - Groq APIError: {e}")
                return ""
            except Exception as e:  # その他の予期せぬエラーが発生した場合
                logging.error(f"GuideBased.__call__ - Unexpected error: {e}")
                return ""
        logging.error(  # 最大リトライ回数に達した場合
            "Max retries reached for __call__. Failed to get a response from Groq API."
        )
        return ""

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
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": prompt.format(
                    document=initial_prompt, lang_example=lang_example
                ),
            },
        ]
        max_retries = 3
        backoff_factor = 2
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                # Groq APIを使用して言語検出をリクエストします
                completion = self.groq_client.chat.completions.create(
                    model=self.config.detect_lang_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_tokens,
                    temperature=self.config.temperature_detect_lang,
                )  # API呼び出し
                # LLMからの応答をログに出力
                content = completion.choices[0].message.content
                logging.info(f"detect_lang LLM response: {content}")
                if not content:  # 応答内容が空の場合
                    logging.error("No content in response for language detection")
                    return ""
                # 結果のJSONをパースして言語コードを取得します
                lang = json.loads(content)["lang"]  # 言語コードを抽出
                return lang
            except (json.JSONDecodeError, KeyError) as e:
                # エラーが発生した場合は空文字列を返します
                logging.error(f"言語検出のためのLLM応答のパースエラー: {e}")
                return ""
            except RateLimitError as e:  # レート制限エラーが発生した場合
                logging.warning(
                    f"Rate limit exceeded in detect_lang. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}"
                )
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:  # Groq APIエラーが発生した場合
                logging.error(f"GuideBased.detect_lang - Groq APIError: {e}")
                return ""
            except Exception as e:  # その他の予期せぬエラーが発生した場合
                logging.error(f"GuideBased.detect_lang - Unexpected error: {e}")
                return ""
        logging.error(  # 最大リトライ回数に達した場合
            "Max retries reached for detect_lang. Failed to get a response from Groq API."
        )
        return ""

    def _clean_json_response(self, content: str) -> str:
        """
        JSONレスポンスからコードブロックマーカーや不要な文字を除去

        Args:
            content (str): 生のLLMレスポンス

        Returns:
            str: クリーニングされたJSON文字列
        """
        # ```json ... ``` のパターンを除去
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        # 一般的な ``` ... ``` ブロックを除去
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        # その他の不正文字をフィルタリング
        return content.replace("```", "").strip()

    def judge(self, candidates: List[str]) -> Optional[int]:
        """
        複数のプロンプト候補を評価し、最も良いものを選択します。
        レスポンスフォーマットの変動に対応するため、前処理とフォールバック処理を追加。

        Args:
            candidates (List[str]): 評価対象のプロンプト候補のリスト。

        Returns:
            Optional[int]: 最も良いと判断された候補のインデックス。エラーの場合はNone。
        """
        Instruction_prompts: List[str] = []
        for idx, candidate in enumerate(candidates):
            Instruction_prompts.append(
                f"Instruction {idx+1}:\n<instruction>\n{candidate}\n</instruction>"
            )
        example: str = json.dumps({"Preferred": "Instruction 1"})
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
            """.strip()
        )
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": prompt.format(
                    guide=PromptGuide,
                    Instruction_prompts="\n\n".join(Instruction_prompts),
                    example=example,
                ),
            },
        ]
        max_retries = 3
        backoff_factor = 2
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.config.judge_model,
                    messages=messages,
                    max_completion_tokens=self.config.max_tokens,
                    temperature=self.config.temperature_judge,
                )
                content = completion.choices[0].message.content
                logging.info(f"judge LLM response (raw): {content}")

                if not content:
                    logging.error("No content in response for judge")
                    return None

                # JSONレスポンスのクリーニング
                cleaned_content = self._clean_json_response(content)
                logging.info(f"Cleaned JSON content: {cleaned_content}")

                try:
                    # クリーニング後のコンテンツをJSONパース
                    result: Dict[str, str] = json.loads(cleaned_content)
                except json.JSONDecodeError as e:
                    # JSONパース失敗時のフォールバック処理
                    logging.error(
                        f"JSONパースエラー: {e}\nコンテンツ: {cleaned_content}"
                    )

                    # 正規表現でJSONオブジェクトを抽出
                    if match := re.search(
                        r'\{"Preferred":\s*"Instruction\s*\d+"\}', cleaned_content
                    ):
                        result = json.loads(match.group(0))
                        logging.info(f"Regex fallback successful: {result}")
                    else:
                        logging.error("JSONオブジェクトを抽出できませんでした")
                        return None

                # 結果の抽出
                final_result = None
                for idx in range(len(candidates)):
                    if str(idx + 1) in result["Preferred"]:
                        final_result = idx
                        break

                return final_result

            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"LLM応答のパースエラー: {e}")
                return None
            except RateLimitError as e:
                logging.warning(
                    f"Rate limit exceeded in judge. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}"
                )
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:
                logging.error(f"GuideBased.judge - Groq APIError: {e}")
                return None
            except Exception as e:
                logging.error(f"Unexpected error in judge method: {e}")
                return None

        logging.error(
            "Max retries reached for judge. Failed to get a response from Groq API."
        )
        return None
