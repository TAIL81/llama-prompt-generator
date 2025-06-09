import json
import os
import re

from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# 環境変数を .env ファイルから読み込みます
# プロジェクトルートの .env ファイルから環境変数を読み込みます
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# デフォルトのシステムプロンプトを定義します
default_system = "You are a helpful and knowledgeable assistant who is able to provide detailed and accurate information on a wide range of topics. You are also able to provide clear and concise answers to questions and are always willing to go the extra mile to help others."
Groq_default_system = default_system
OpenRouter_default_system = default_system

# 応答評価用のプロンプトテンプレート
evaluate_response_prompt_template = """
You are an expert in linguistics and able to observe subtle differences in content between two paragraphs. Your task is to analyze responses from OpenAI and Llama and provide detailed feedback.

Here are the OpenAI response: 
<response>
{_OpenAI}
</response>

Here are the Llama response:
<response>
{_Bedrock}
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
{_Bedrock}
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

# APIキーとベースURLを (.env ファイルからロードされた) 環境変数から取得します
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
groq_api_key = os.getenv("GROQ_API_KEY")

# プロンプトの最適化と評価を行うクラス
class Alignment:
    def __init__(self):
        try:
            self.openrouter_client = OpenAI(
                base_url=openai_base_url,
                api_key=openai_api_key,
            )
            # # 追加：実際のbase_urlをコンソールに出力
            # if self.openrouter_client:
            #     print(f"🔍 [DEBUG] Initialized OpenRouter client with base_url: {self.openrouter_client.base_url}")
        except Exception as e:
            # APIキーがない場合、クライアントはNoneになります
            print(f"OpenRouter client initialization failed: {e}") # エラーログを追加
            self.openrouter_client = None
        try:
            self.groq_client = Groq(api_key=groq_api_key)
        except Exception as e:
            # APIキーがない場合、クライアントはNoneになります
            print(f"Groq client initialization failed: {e}") # エラーログを追加
            self.groq_client = None

    def generate_groq_response(self, prompt, model_id):
        """
        Groq APIを使用して応答を生成します。

        Args:
            prompt (str): ユーザープロンプト。
            model_id (str): 使用するGroqモデルのID。

        Returns:
            str: Groqモデルからの応答、またはエラーメッセージ。
        """
        if not self.groq_client:
            return "GroqError: API client not initialized. Check GROQ_API_KEY."
        try:
            completion = self.groq_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": Groq_default_system},
                    {"role": "user", "content": prompt},
                ],
            )
            # レスポンス構造のバリデーションを追加
            if hasattr(completion, 'choices') and completion.choices and hasattr(completion.choices[0], 'message'):
                 msg = completion.choices[0].message
                 return msg.content if hasattr(msg, 'content') else "Error: Content missing in Groq response"
            else:
                return "Error: Invalid response structure from Groq API"
        except Exception as e:
            return f"Groq API Error: {str(e)}"

    def generate_openrouter_response(self, prompt, model_id):
        """
        OpenRouter API (OpenAI互換) を使用して応答を生成します。

        Args:
            prompt (str): ユーザープロンプト。
            model_id (str): 使用するOpenRouterモデルのID。

        Returns:
            str: OpenRouterモデルからの応答、またはエラーメッセージ。
        """
        if not self.openrouter_client:
            return "OpenRouterError: API client not initialized. Check OPENAI_API_KEY."
        try:
            # print(f"🔍 [DEBUG] APIリクエスト送信: model={model_id}, prompt_length={len(prompt)}")
            completion = self.openrouter_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": OpenRouter_default_system},
                    {"role": "user", "content": prompt},
                ],
            )

            # APIからのレスポンスが文字列型の場合、エラーとして処理します
            if isinstance(completion, str):
                # print(f"🔍 [DEBUG] OpenRouter API returned a string: {completion}")
                return f"OpenRouter API Error: Received unexpected string response: {completion}"

            # レスポンス構造デバッグ
            # print(f"🔍 [DEBUG] レスポンスタイプ: {type(completion)}")
            # print(f"🔍 [DEBUG] レスポンス属性: {dir(completion)}")

            if hasattr(completion, 'choices'):
                # print(f"🔍 [DEBUG] choices数: {len(completion.choices)}")
                if completion.choices and len(completion.choices) > 0: # choicesが空でないことを確認
                    first_choice = completion.choices[0]
                    # print(f"🔍 [DEBUG] 最初のchoiceタイプ: {type(first_choice)}")
                    # print(f"🔍 [DEBUG] 最初のchoice属性: {dir(first_choice)}")
                    if hasattr(first_choice, 'message'):
                        msg = first_choice.message
                        # print(f"🔍 [DEBUG] messageタイプ: {type(msg)}")
                        # print(f"🔍 [DEBUG] message属性: {dir(msg)}")
                        if hasattr(msg, 'content'):
                            return msg.content
            # 上記のいずれの条件にも一致しない場合、エラーメッセージを返します
            error_message = "Error: Invalid or empty response structure from OpenRouter API."
            # print(f"🔍 [DEBUG] {error_message} Response: {completion}") # 詳細なレスポンス内容をログに出力
            return error_message
        except Exception as e:
            return f"OpenRouter API Error: {str(e)}"

    def stream_groq_response(self, prompt, model_id, output_component):
        # TODO: Gradioの出力コンポーネントへのストリーミング出力を実装する
        if not self.groq_client:
            output_component.update("GroqError: API client not initialized. Check GROQ_API_KEY.")
            return
        try:
            stream = self.groq_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": Groq_default_system},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    output_component.update(chunk.choices[0].delta.content, append=True)
        except Exception as e:
            output_component.update(f"Groq API Error during streaming: {str(e)}")


    def stream_openrouter_response(self, prompt, model_id, output_component):
        # TODO: Gradioの出力コンポーネントへのストリーミング出力を実装する
        if not self.openrouter_client:
            output_component.update("OpenRouterError: API client not initialized. Check OPENAI_API_KEY.")
            return
        try:
            stream = self.openrouter_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": OpenRouter_default_system},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    output_component.update(chunk.choices[0].delta.content, append=True)
        except Exception as e:
             output_component.update(f"OpenRouter API Error during streaming: {str(e)}")


    def invoke_prompt(
        self,
        original_prompt_replace,
        revised_prompt_replace,
        original_prompt,
        revised_prompt,
        openrouter_model_id,
        groq_model_id,
    ):
        """
        元のプロンプトと改訂されたプロンプトをそれぞれOpenRouterとGroqで実行し、結果を返します。

        Args:
            original_prompt_replace (str): 変数が置換された元のプロンプト。
            revised_prompt_replace (str): 変数が置換された改訂プロンプト。
            original_prompt (str): 元のプロンプト（置換前）。
            revised_prompt (str): 改訂プロンプト（置換前）。
            openrouter_model_id (str): OpenRouterで使用するモデルID。
            groq_model_id (str): Groqで使用するモデルID。

        Returns:
            tuple: (OpenRouterからの応答, Groqからの応答)
        """
        # 置換後のプロンプトが空の場合、置換前のプロンプトを使用します
        if len(original_prompt_replace) == 0:
            original_prompt_replace = original_prompt
        if len(revised_prompt_replace) == 0:
            revised_prompt_replace = revised_prompt

        # APIクライアントが初期化されていない場合の具体的なエラーメッセージ
        if self.openrouter_client is None:
            openai_result = "OpenRouterError: API client not initialized. Check OPENAI_API_KEY and OPENAI_BASE_URL."
        else:
            # generate_openrouter_responseからの戻り値をチェック
            openai_result = self.generate_openrouter_response(
                original_prompt_replace, openrouter_model_id
            )
            # generate_openrouter_responseがエラー文字列を返す場合があるため、ここで処理
            if isinstance(openai_result, str) and openai_result.startswith("OpenRouter API Error:"):
                pass # エラーメッセージをそのまま使用

        if self.groq_client is None:
            groq_result = "GroqError: API client not initialized. Check GROQ_API_KEY."
        else:
            # generate_groq_responseからの戻り値をチェック
            groq_result = self.generate_groq_response(
                revised_prompt_replace, groq_model_id
            )
            # generate_groq_responseがエラー文字列を返す場合があるため、ここで処理
            if isinstance(groq_result, str) and groq_result.startswith("Groq API Error:"):
                pass # エラーメッセージをそのまま使用

        return openai_result, groq_result

    def evaluate_response(self, openai_output, groq_output, eval_model_id):
        """
        OpenAIとGroqの応答を比較評価し、フィードバックと推奨事項を生成します。

        Args:
            openai_output (str): OpenAI (OpenRouter) からの応答。
            groq_output (str): Groqからの応答。
            eval_model_id (str): 評価に使用するGroqモデルのID。

        Returns:
            str: 自動フィードバックと推奨事項を含む文字列。
        """
        if not self.groq_client:
            return "GroqError: API client for evaluation not initialized. Check GROQ_API_KEY."
        revised_prompt = evaluate_response_prompt_template.format(
            _OpenAI=openai_output, _Bedrock=groq_output
        )
        # generate_groq_responseからの戻り値をチェック
        groq_result = self.generate_groq_response(revised_prompt, eval_model_id)
        if isinstance(groq_result, str) and groq_result.startswith("Groq API Error:"):
             return f"Evaluation Error: {groq_result}" # エラーメッセージを返す

        # 生成された結果からフィードバックと推奨事項を抽出します
        pattern = r"<auto_feedback>(.*?)</auto_feedback>"
        feedback_match = re.findall(pattern, groq_result, re.DOTALL)
        feedback = feedback_match[0] if feedback_match else "Feedback not found."

        pattern = r"<recommendation>(.*?)</recommendation>"
        recommendation_match = re.findall(pattern, groq_result, re.DOTALL)
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
        # Split the key-value string by ';' to get individual pairs
        kv_pairs = kv_string.split(";")
        for pair in kv_pairs:
            if ":" in pair:
                key, value = pair.split(":", 1)  # Only split on the first ':'
                # プロンプト内の {key} 形式のプレースホルダを value で置換します
                user_prompt = user_prompt.replace(f"{{{key}}}", value)
        return user_prompt

    def generate_revised_prompt(
        self, feedback, prompt, openai_response, groq_response, eval_model_id
    ):
        """
        フィードバック、元のプロンプト、および両モデルの応答に基づいて、改訂されたプロンプトを生成します。

        Args:
            feedback (str): プロンプト改善のためのフィードバック。
            prompt (str): 元のLlama (Groq) プロンプト。
            openai_response (str): OpenAI (OpenRouter) からの応答。
            groq_response (str): Llama (Groq) からの応答。
            eval_model_id (str): プロンプト改訂に使用するGroqモデルのID。

        Returns:
            str: 改訂されたプロンプト。
        """
        pattern = r"<recommendation>(.*?)</recommendation>"
        matches = re.findall(pattern, feedback, re.DOTALL)
        if len(matches):
            feedback = matches[0]
        revised_prompt = generate_revised_prompt_template.format(
            _feedback=feedback,
            _prompt=prompt,
            _OpenAI=openai_response,
            _Bedrock=groq_response,
        )
        if not self.groq_client:
            return "GroqError: API client for prompt revision not initialized. Check GROQ_API_KEY."
        # generate_groq_responseからの戻り値をチェック
        groq_result = self.generate_groq_response(revised_prompt, eval_model_id)
        if isinstance(groq_result, str) and groq_result.startswith("Groq API Error:"):
             return f"Prompt Revision Error: {groq_result}" # エラーメッセージを返す

        # 生成された結果から改訂プロンプトを抽出します
        pattern = r"<revised_prompt>(.*?)</revised_prompt>"
        matches = re.findall(pattern, groq_result, re.DOTALL)
        matches = matches[0] if matches else "Revised prompt not found."
        return matches.strip()
