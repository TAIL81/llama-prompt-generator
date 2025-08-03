import json
import logging
import os
import re  # 正規表現用モジュールを追加
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field, ValidationError

# 環境変数を .env ファイルから読み込みます
env_path = Path(__file__).parent.parent / ".env"  # .envファイルのパスを構築
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Language(BaseModel):
    """言語検出APIのレスポンスを定義するPydanticモデル"""

    lang: str = Field(
        ...,
        description="Detected language code (e.g., 'en', 'ja', 'ch').",
        examples=["ja"],
    )


class PreferredInstruction(BaseModel):
    """プロンプト評価APIのレスポンスを定義するPydanticモデル"""

    preferred: str = Field(
        ...,
        description="The name of the preferred instruction (e.g., 'Instruction 1').",
        examples=["Instruction 1"],
    )


@dataclass
class GroqConfig:
    rewrite_model: str = (
        "meta-llama/llama-4-scout-17b-16e-instruct"  # プロンプト書き換えに使用するモデル
    )
    detect_lang_model: str = "llama-3.3-70b-versatile"  # 言語検出に使用するモデル
    judge_model: str = "llama-3.3-70b-versatile"  # 評価に使用するモデル
    max_tokens: int = 8192  # モデルからの応答の最大トークン数
    reasoning_effort: Literal["none", "default"] = (
        "none"  # "default" for thinking, "none" for non-thinking
    )
    temperature_rewrite: float = 0.7
    top_p_rewrite: float = 1.0
    temperature_detect_lang: float = 0.0
    temperature_judge: float = 0.0
    translate_model: str = "llama-3.1-8b-instant"  # 翻訳に使用するモデル


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
                # Groq APIパラメータ（ライブラリの互換性確保）
                api_params = {
                    "model": self.config.rewrite_model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature_rewrite,
                    "top_p": self.config.top_p_rewrite,
                }
                # reasoning_effortは新しいパラメータなので条件付きで追加
                if hasattr(self.groq_client.chat.completions, "reasoning_effort"):
                    api_params["reasoning_effort"] = self.config.reasoning_effort

                completion = self.groq_client.chat.completions.create(**api_params)
                result: str = (  # LLMの応答を取得
                    completion.choices[0].message.content or ""
                )  # LLMの応答 (handle None)
                # LLMからの応答をログに出力
                logging.debug(f"__call__ LLM response: \n{result}\n")
                # 結果から不要なXMLタグを除去します
                if result.startswith("<instruction>"):  # 開始タグを除去
                    result = result[13:]
                if result.endswith("</instruction>"):  # 終了タグを除去
                    result = result[:-14]
                result = result.strip()
                return result
            except RateLimitError as e:  # レート制限エラーが発生した場合
                logging.warning(
                    f"Rate limit exceeded in __call__. Attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay} seconds."
                )
                if e.response:
                    logging.info(f"Response headers: {e.response.headers}")
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:  # Groq APIエラーが発生した場合
                logging.error(f"GuideBased.__call__ - Groq APIError: {e}")
                raise  # エラーを再発生させて呼び出し元に通知
            except Exception as e:  # その他の予期せぬエラーが発生した場合
                logging.error(f"GuideBased.__call__ - Unexpected error: {e}")
                raise  # エラーを再発生させて呼び出し元に通知
        logging.error(  # 最大リトライ回数に達した場合
            "Max retries reached for __call__. Failed to get a response from Groq API."
        )
        raise Exception("Failed to rewrite prompt after multiple retries.")

    def _validate_initial_prompt(self, initial_prompt: str) -> None:
        """
        initial_promptの入力検証を行います。
        """
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です")

    def detect_lang(self, initial_prompt: str) -> str:
        """
        与えられたプロンプトの言語を検出します (英語、中国語または日本語)。
        GroqのJSON Object Modeを使用して、信頼性の高いJSON出力を保証します。

        Args:
            initial_prompt (str): 言語を検出する対象のプロンプト。

        Returns:
            str: 検出された言語コード ("en", "ch" または "ja")。エラーの場合は空文字列を返す。
        """
        prompt = f"""
        Please determine what language the document below is in? English (en), Chinese (ch) or Japanese (ja)?
        Your response must be only the JSON object, with no other text before or after it.
        The JSON object must have a key named "lang" with the detected language code as its value (e.g., {{"lang": "ja"}}).

        <document>
        {initial_prompt}
        </document>
        """
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "You are an AI assistant that detects the language of a given text and responds in a structured JSON format as requested by the user.",
            },
            {"role": "user", "content": prompt},
        ]
        max_retries = 3
        backoff_factor = 2
        retry_delay = 1
        for attempt in range(max_retries):
            content = None  # 変数を事前に初期化
            try:
                # Groq APIを使用して言語検出をリクエストします
                completion = self.groq_client.chat.completions.create(
                    model=self.config.detect_lang_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content
                logging.debug(f"detect_lang LLM response: {content}")
                if not content:
                    logging.error("No content in response for language detection")
                    return ""

                # Pydanticモデルでレスポンスを検証・パース
                detected_language = Language.model_validate(json.loads(content))
                return detected_language.lang

            except json.JSONDecodeError as e:
                # contentがNoneの場合のハンドリングを追加
                error_msg = f"JSONパースエラー: {e}"
                if content is not None:
                    error_msg += f"\nAPIレスポンス: {content}"
                else:
                    error_msg += "\nAPIレスポンス: なし"
                logging.error(error_msg)
                return ""
            except ValidationError as e:
                # contentがNoneの場合のハンドリングを追加
                error_msg = f"Pydantic検証エラー: {e}"
                if content is not None:
                    error_msg += f"\nAPIレスポンス: {content}"
                else:
                    error_msg += "\nAPIレスポンス: なし"
                logging.error(error_msg)
                return ""
            except RateLimitError as e:  # レート制限エラーが発生した場合
                logging.warning(
                    f"Rate limit exceeded in detect_lang. Attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay} seconds."
                )
                if e.response:
                    logging.info(f"Response headers: {e.response.headers}")
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:  # Groq APIエラーが発生した場合
                logging.error(f"GuideBased.detect_lang - Groq APIError: {e}")
                return ""
            except Exception as e:  # その他の予期せぬエラーが発生した場合
                logging.error(f"GuideBased.detect_lang - Unexpected error: {e}")
                return ""
        logging.error(  # 最大リトライ次に達した場合
            "Max retries reached for detect_lang. Failed to get a response from Groq API."
        )
        return ""

    def judge(self, candidates: List[str]) -> Optional[int]:
        """
        複数のプロンプト候補を評価し、最も良いものを選択します。
        GroqのJSON Object Modeを使用して、信頼性の高いJSON出力を保証します。

        Args:
            candidates (List[str]): 評価対象のプロンプト候補のリスト。

        Returns:
            Optional[int]: 最も良いと判断された候補のインデックス。エラーの場合はNoneを返す。
        """
        Instruction_prompts: List[str] = []
        for idx, candidate in enumerate(candidates):
            Instruction_prompts.append(
                f"Instruction {idx+1}:\n<instruction>\n{candidate}\n</instruction>"
            )

        prompt: str = (
            """
            You are an instruction engineer. Your task is to evaluate which of the three instructions given below is better based on the guide in the <guide> xml tag.

            Instruction guide:
            <guide>
            {guide}
            </guide>

            {Instruction_prompts}

            Your response must be only the JSON object, with no other text before or after it.
            The JSON object must have a key named "preferred" with the name of the preferred instruction as its value (e.g., {{"preferred": "Instruction 1"}}).
            """.strip()
        )
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": prompt.format(
                    guide=PromptGuide,
                    Instruction_prompts="\n\n".join(Instruction_prompts),
                ),
            },
        ]
        max_retries = 3
        backoff_factor = 2
        retry_delay = 1

        for attempt in range(max_retries):
            content = None  # 変数を事前に初期化
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.config.judge_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content
                logging.debug(f"judge LLM response (raw): {content}")

                if not content:
                    logging.error("No content in response for judge")
                    return None

                # Pydanticモデルでレスポンスを検証・パース
                result = PreferredInstruction.model_validate(json.loads(content))

                # 結果の抽出
                final_result = None
                for idx in range(len(candidates)):
                    if str(idx + 1) in result.preferred:
                        final_result = idx
                        break

                return final_result

            except json.JSONDecodeError as e:
                # contentがNoneの場合のハンドリングを追加
                error_msg = f"JSONパースエラー: {e}"
                if content is not None:
                    error_msg += f"\nAPIレスポンス: {content}"
                else:
                    error_msg += "\nAPIレスポンス: なし"
                logging.error(error_msg)
                return None
            except ValidationError as e:
                # contentがNoneの場合のハンドリングを追加
                error_msg = f"Pydantic検証エラー: {e}"
                if content is not None:
                    error_msg += f"\nAPIレスポンス: {content}"
                else:
                    error_msg += "\nAPIレスポンス: なし"
                logging.error(error_msg)
                return None
            except RateLimitError as e:
                logging.warning(
                    f"Rate limit exceeded in judge. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}"
                )
                if e.response:
                    logging.info(f"Response headers: {e.response.headers}")
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
            except APIError as e:
                logging.error(f"GuideBased.judge - Groq APIError: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error in judge method: {e}")
                raise

        logging.error(
            "Max retries reached for judge. Failed to get a response from Groq API."
        )
        raise Exception("Failed to judge prompt after multiple retries.")

    def translate(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """
        テキストを指定された言語に翻訳します。
        レート制限エラーが発生した場合は、指数バックオフでリトライします。
        """
        max_retries = 5
        base_delay = 1  # seconds

        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": f"You are a translator. Translate the following text from {source_lang} to {target_lang}.",
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        for attempt in range(max_retries):
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=messages,
                    model=self.config.translate_model,
                )
                return chat_completion.choices[0].message.content or ""

            except RateLimitError as e:
                logging.warning(
                    f"Rate limit exceeded during translation. Attempt {attempt + 1} of {max_retries}."
                )
                # Groq APIのレート制限ヘッダーを確認 (存在する場合)
                if e.response and e.response.headers:
                    logging.info(f"Response headers: {e.response.headers}")

                # 指数バックオフ
                delay = base_delay * (2**attempt)
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

            except Exception as e:
                logging.error(f"An unexpected error occurred during translation: {e}")
                # その他のエラーはリトライせずに終了
                raise

        raise Exception("Failed to translate after multiple retries.")
