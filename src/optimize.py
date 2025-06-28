import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import gradio as gr  # Gradioをインポート
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 環境変数の読み込み
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 定数の定義
TEMPERATURE: float = 0.2
MAX_TOKENS = 8192

# デフォルトのシステムプロンプトのテンプレートを定義します
DEFAULT_SYSTEM_TEMPLATE = "You are a helpful and knowledgeable assistant who is able to provide detailed and accurate information on a wide range of topics. You are also able to provide clear and concise answers to questions and are always willing to go the extra mile to help others."

# 応答評価用のプロンプトテンプレート
evaluate_response_prompt_template = """
You are an expert in linguistics and able to observe subtle differences in content between two paragraphs. Your task is to analyze responses from OpenAI and Llama and provide detailed feedback.
{language_instruction_for_evaluation}

Here are the OpenAI response:
<response>
{_OpenAI}
</response>

Here are the Llama response:
<response>
{_Groq}
</response>

Please follow these steps:
1. Carefully analyze both responses in terms of content accuracy, logical organization, and expression style.
2. Summarize the differences between the Llama response and the OpenAI response.
3. Provide recommendations on how the Llama response could be refactored to better align with the OpenAI response.
4. Encapsulate your analysis, including the differences, within <auto_feedback></auto_feedback> tags using bullet points.
5. Encapsulate recommendations, within <recommendation></recommendation> tags using bullet points.
""".strip()

# 改訂プロンプト生成用のプロンプトテンプレート
generate_revised_prompt_template = """
You are an expert in prompt engineering for both OpenAI and Llama model and able to follow the human feedback to adjust the prompt to attain the optimal effect, you will be given the original Llama prompt, responses from OpenAI, responses from Llama and human feedback to revise the Llama prompt.
{language_instruction_for_revision}

Here are the original Llama prompt:
<prompt>
{_prompt}
</prompt>

Here are the OpenAI response:
<response>
{_OpenAI}
</response>

Here are the Llama response:
<response>
{_Groq}
</response>

Here are the human feedback:
<evaluation_summary>
{_feedback}
</evaluation_summary>

Please analyze whether Llama's response strictly aligns with OpenAI's response based on the human feedback. Then, consider how the original Llama prompt can be improved accordingly. Your revised prompt should only involve slight adjustments and must not drastically change the original prompt. Use the human feedback to guide your revision.

Finally, provide the revised prompt within the following XML tags:

<revised_prompt>
[Your revised prompt]
</revised_prompt>
""".strip()

# 評価モデルIDの定数定義
EVAL_MODEL_ID: str = "llama-3.3-70b-versatile"

# APIキーとベースURLを (.env ファイルからロードされた) 環境変数から取得します
openai_api_key = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
groq_api_key = os.getenv("GROQ_API_KEY")


# プロンプトの最適化と評価を行うクラス
class Alignment:
    def __init__(self, lang_store: Optional[Dict[str, Any]] = None, language: str = "ja") -> None:
        self.lang_store: Optional[Dict[str, Any]] = lang_store
        self.language = language

        try:
            self.OpenAI_client = OpenAI(
                base_url=OPENAI_BASE_URL,
                api_key=openai_api_key,
            )
        except Exception as e:
            logging.error(f"OpenAI client initialization failed: {e}")
            self.OpenAI_client = None

        try:
            self.groq_client = Groq(api_key=groq_api_key)
        except Exception as e:
            logging.error(f"Groq client initialization failed: {e}")
            self.groq_client = None

        if self.language == "ja":
            language_instructions = {
                "generation": "応答は日本語で生成してください。",
                "evaluation": "フィードバックと推奨事項は日本語で生成してください。",
                "revision": "改訂されたプロンプトは日本語で生成してください。XMLタグ名は英語のままにしてください。",
            }
        else:
            language_instructions = {}

        # 英語の指示をデフォルトとして設定
        self.generation_language_instruction = language_instructions.get(
            "generation", "Please generate the response in English."
        )
        system_prompt_suffix = (
            f"\n{self.generation_language_instruction}" if hasattr(self, "generation_language_instruction") else ""
        )
        self.groq_system_content = DEFAULT_SYSTEM_TEMPLATE + system_prompt_suffix
        self.OpenAI_system_content = DEFAULT_SYSTEM_TEMPLATE + system_prompt_suffix

        self.evaluate_response_prompt = evaluate_response_prompt_template
        self.generate_revised_prompt_template = generate_revised_prompt_template

        if hasattr(self, "evaluation_language_instruction"):
            self.evaluate_response_prompt = self.evaluate_response_prompt.format(
                language_instruction_for_evaluation=self.evaluation_language_instruction
            )
        if hasattr(self, "revision_language_instruction"):
            self.generate_revised_prompt_template = self.generate_revised_prompt_template.format(
                language_instruction_for_revision=self.revision_language_instruction
            )
        self.evaluation_language_instruction = language_instructions.get(
            "evaluation", "Please generate the feedback and recommendations in English."
        )
        self.revision_language_instruction = language_instructions.get(
            "revision", "Please generate the revised prompt in English. XML tag names should remain in English."
        )

    def _validate_response(self, response: Any) -> bool:
        """APIレスポンスの構造を検証するヘルパーメソッド"""
        return (
            hasattr(response, "choices")
            and response.choices
            and len(response.choices) > 0
            and hasattr(response.choices[0], "message")
            and hasattr(response.choices[0].message, "content")
        )

    def _generate_response(
        self, client: Union[OpenAI, Groq], model_id: str, system_content: str, user_prompt: str
    ) -> str:
        """APIクライアントを使用して応答を生成する共通メソッド"""
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_content}, {"role": "user", "content": user_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )  # Groq API を呼び出し、応答を生成
            if not self._validate_response(completion):
                logging.error(f"Invalid response structure from Groq API: {completion}")
                return "Error: Invalid response structure from Groq API"

            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API Error: {e}")
            return f"Groq API Error: {str(e)}"

    def generate_groq_response(self, prompt: str, model_id: str) -> str:
        """Groq APIを使用して応答を生成"""
        if not self.groq_client:
            return "GroqError: API client not initialized. Check GROQ_API_KEY."
        return self._generate_response(self.groq_client, model_id, self.groq_system_content, prompt)

    def generate_OpenAI_response(self, prompt: str, model_id: str) -> str:
        """OpenAI APIを使用して応答を生成"""
        if not self.OpenAI_client:
            return "OpenAIError: API client not initialized. Check OPENAI_API_KEY."
        return self._generate_response(self.OpenAI_client, model_id, self.OpenAI_system_content, prompt)

    # ストリーミング出力メソッド（未実装のためコメントアウト）
    # def stream_groq_response(self, prompt, model_id, output_component):
    #     if not self.groq_client:
    #         logging.error("GroqError: API client not initialized. Check GROQ_API_KEY.")
    #         output_component.update("GroqError: API client not initialized. Check GROQ_API_KEY.")
    #         return
    #     try:
    #         stream = self.groq_client.chat.completions.create(
    #             model=model_id,
    #             messages=[
    #                 {"role": "system", "content": self.groq_system_content},
    #                 {"role": "user", "content": prompt},
    #             ],
    #             stream=True,
    #             temperature=TEMPERATURE,
    #             max_tokens=MAX_TOKENS,
    #         )
    #         for chunk in stream:
    #             if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
    #                 output_
    def stream_OpenAI_response(self, prompt, model_id, output_component):
        if not self.OpenAI_client:
            logging.error("OpenAIError: API client not initialized. Check OPENAI_API_KEY.")
            output_component.update("OpenAIError: API client not initialized. Check OPENAI_API_KEY.")
            return
        try:
            stream = self.OpenAI_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": self.OpenAI_system_content},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=TEMPERATURE,
                max_tokens=8192,
            )
            for chunk in stream:  # ストリームからのチャンクを処理
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                    output_component.update(chunk.choices[0].delta.content, append=True)
        except Exception as e:
            logging.error(f"OpenAI API Error during streaming: {e}")
            output_component.update(f"OpenAI API Error during streaming: {str(e)}")

    def invoke_prompt(
        self,
        original_prompt_replace,
        revised_prompt_replace,
        original_prompt,
        revised_prompt,
        OpenAI_model_id,
        groq_model_id,
    ) -> Generator[Tuple[Any, Any], None, None]:
        """
        元のプロンプトと改訂されたプロンプトをそれぞれOpenAIとGroqで実行し、結果を返します。ジェネレータを使用して、途中結果をyieldする。

        Args:
            original_prompt_replace (str): 変数が置換された元のプロンプト。
            revised_prompt_replace (str): 変数が置換された改訂プロンプト。
            original_prompt (str): 元のプロンプト（置換前）。
            revised_prompt (str): 改訂プロンプト（置換前）。
            OpenAI_model_id (str): OpenAIで使用するモデルID。
            groq_model_id (str): Groqで使用するモデルID。

        Returns:
            tuple: (OpenAIからの応答, Groqからの応答)
        """
        # 置換後のプロンプトが空の場合、置換前のプロンプトを使用します
        if len(original_prompt_replace) == 0:
            original_prompt_replace = original_prompt  # 元のプロンプトを使用
        if len(revised_prompt_replace) == 0:
            revised_prompt_replace = revised_prompt

        original_prompt_output = ""
        evaluation_prompt_output = ""

        processing_message = "Processing..."

        # 元のプロンプトはOpenAIで実行
        if self.OpenAI_client is None:
            error_msg = "OpenAIError: API client not initialized. Check OPENAI_API_KEY and OPENAI_BASE_URL."
            logging.error(error_msg)
            yield gr.update(value=error_msg), gr.update(value=error_msg)
            return
        # OpenAI API を呼び出し、応答を生成
        original_prompt_output = self.generate_OpenAI_response(original_prompt_replace, OpenAI_model_id)
        yield gr.update(value=original_prompt_output), gr.update(value=processing_message)

        if isinstance(original_prompt_output, str) and (
            original_prompt_output.startswith("OpenAIError:") or original_prompt_output.startswith("Error:")
        ):
            yield gr.update(value=original_prompt_output), gr.update(value=original_prompt_output)
            return

        # 評価用プロンプトはGroqで実行
        if self.groq_client is None:
            error_msg = "GroqError: API client not initialized. Check GROQ_API_KEY."
            logging.error(error_msg)
            yield gr.update(value=error_msg), gr.update(value=error_msg)
            return
        # Groq API を呼び出し、応答を生成
        evaluation_prompt_output = self.generate_groq_response(revised_prompt_replace, groq_model_id)
        yield gr.update(value=original_prompt_output), gr.update(value=evaluation_prompt_output)

    def evaluate_response(self, openai_output: str, groq_output: str) -> str:
        """
        固定モデルを使用して OpenAI と Groq の応答を比較評価し、フィードバックと推奨事項を生成します。

        OpenAIとGroqの応答を比較評価し、フィードバックと推奨事項を生成します。

        Args:
            openai_output (str): OpenAI (OpenAI) からの応答。
            groq_output (str): Groqからの応答。

        Returns:
            str: 自動フィードバックと推奨事項を含む文字列。
        """
        if not self.groq_client:
            logging.error("GroqError: API client for evaluation not initialized. Check GROQ_API_KEY.")
            return "GroqError: API client for evaluation not initialized. Check GROQ_API_KEY."

        current_eval_instruction = self.evaluation_language_instruction  # 現在の評価指示
        formatted_evaluate_prompt = evaluate_response_prompt_template.format(
            _OpenAI=openai_output, _Groq=groq_output, language_instruction_for_evaluation=current_eval_instruction
        )
        # generate_groq_responseからの戻り値をチェック
        groq_result = self.generate_groq_response(formatted_evaluate_prompt, EVAL_MODEL_ID)
        if isinstance(groq_result, str) and groq_result.startswith("Groq API Error:"):
            logging.error(f"Evaluation Error: {groq_result}")
            return f"Evaluation Error: {groq_result}"  # エラーメッセージを返す

        # 生成された結果からフィードバックと推奨事項を抽出します
        pattern = r"<auto_feedback>(.*?)</auto_feedback>"
        feedback_match = re.findall(pattern, groq_result, re.DOTALL)
        feedback = feedback_match[0] if feedback_match else "Feedback not found."

        pattern = r"<recommendation>(.*?)</recommendation>"
        recommendation_match = re.findall(pattern, groq_result, re.DOTALL)  # 推奨事項を抽出
        recommendation = recommendation_match[0] if recommendation_match else "Recommendation not found."

        return feedback + f"\n<recommendation>{recommendation}</recommendation>"

    def insert_kv(self, user_prompt, kv_string):
        """
        ユーザープロンプト内のプレースホルダをキーバリュー文字列に基づいて置換します。

        Args:
            user_prompt (str): 置換対象のプロンプト。
            kv_string (str): "key1:value1;key2:value2" 形式のキーバリュー文字列。

        Returns:
            str: 置換後のプロンプト。
        """
        kv_pairs = kv_string.split(";")
        for pair in kv_pairs:
            if ":" in pair:
                key, value = pair.split(":", 1)
                # プロンプト内の {key} 形式のプレースホルダを value で置換します
                user_prompt = user_prompt.replace(f"{{{key}}}", value)
        return user_prompt

    def generate_revised_prompt(self, feedback, prompt, openai_response, groq_response) -> str:
        """
        固定モデルを使用して、フィードバック、元のプロンプト、および両モデルの応答に基づいて、改訂されたプロンプトを生成します。


        フィードバック、元のプロンプト、および両モデルの応答に基づいて、改訂されたプロンプトを生成します。

        Args:
            feedback (str): プロンプト改善のためのフィードバック。
            groq_response (str): Llama (Groq) からの応答。

        Returns:
            str: 改訂されたプロンプト。
        """
        pattern = r"<recommendation>(.*?)</recommendation>"
        matches = re.findall(pattern, feedback, re.DOTALL)
        if len(matches):
            feedback = matches[0]

        current_revision_instruction = self.revision_language_instruction  # 現在の修正指示
        formatted_revised_prompt_content = generate_revised_prompt_template.format(
            _feedback=feedback,
            _prompt=prompt,
            _OpenAI=openai_response,
            _Groq=groq_response,
            language_instruction_for_revision=current_revision_instruction,
        )
        if not self.groq_client:
            logging.error("GroqError: API client for prompt revision not initialized. Check GROQ_API_KEY.")
            return "GroqError: API client for prompt revision not initialized. Check GROQ_API_KEY."
        # generate_groq_responseからの戻り値をチェック
        groq_result = self.generate_groq_response(formatted_revised_prompt_content, EVAL_MODEL_ID)
        if isinstance(groq_result, str) and groq_result.startswith("Groq API Error:"):  # Groq API エラーをチェック
            logging.error(f"Prompt Revision Error: {groq_result}")
            return f"Prompt Revision Error: {groq_result}"  # エラーメッセージを返す

        # 生成された結果から改訂プロンプトを抽出します
        pattern = r"<revised_prompt>(.*?)</revised_prompt>"
        matches = re.findall(pattern, groq_result, re.DOTALL)
        matches = matches[0] if matches else "Revised prompt not found."
        return matches.strip()
