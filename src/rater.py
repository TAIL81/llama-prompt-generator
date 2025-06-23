import json
import logging
from groq import Groq
import groq # Import the groq module to access specific error types
import os
# Groq APIキーを環境変数から取得し、クライアントを初期化します
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# プロンプト候補を評価するクラス
class Rater:
    def __init__(self):
        pass

    def __call__(self, initial_prompt, candidates, demo_data):
        """
        複数のプロンプト候補を評価し、最も良いものを選択します。

        Args:
            initial_prompt (str): 元の指示プロンプト。
            candidates (list[dict]): 評価対象のプロンプト候補のリスト。
                                     各要素は {"prompt": "候補プロンプト"} の形式。
            demo_data (dict): デモデータ（キーと値のペア）。プロンプト内のプレースホルダを置換するために使用。

        Returns:
            int: 最も評価の高かった候補のインデックス。
        """
        for candidate in candidates:
            if "output" in candidate:
                # 既に評価済みの場合はスキップ
                continue
            candidate_prompt = candidate["prompt"]
            # デモデータでプロンプト内のプレースホルダを置換
            for k, v in demo_data.items():
                candidate_prompt = candidate_prompt.replace(k, v)
            candidate["input"] = candidate_prompt
            # 置換後のプロンプトでモデル出力を取得
            candidate["output"] = self.get_output(candidate_prompt)
        # 元の指示プロンプトもデモデータで置換
        for k, v in demo_data.items():
            initial_prompt = initial_prompt.replace(k, v)
        # 評価を実行
        rate = self.rater(initial_prompt, candidates)
        logging.debug(f"Rater.__call__ return: {rate}\n")
        return rate

    def get_output(self, prompt):
        """指定されたプロンプトでGroqモデルを実行し、出力を取得します。"""
        messages = [{"role": "user", "content": prompt}]
        try:
            completion = groq_client.chat.completions.create(
                model="compound-beta-mini",
                messages=messages,
                max_completion_tokens=8192,
                temperature=0.1,
            )
            result = completion.choices[0].message.content
            logging.debug(f"Rater.get_output successful, result: \n{result}\n")
            return result
        except groq.InternalServerError as e:
            error_message = e.body.get('error', {}).get('message', str(e)) if hasattr(e, 'body') and isinstance(e.body, dict) else str(e)
            logging.error(f"Rater.get_output - Groq InternalServerError: {error_message} (Details: {e})")
            raise # 例外を再送出
        except groq.APIError as e: # InternalServerError以外のAPIエラーも捕捉
            error_message = e.body.get('error', {}).get('message', str(e)) if hasattr(e, 'body') and isinstance(e.body, dict) else str(e)
            logging.error(f"Rater.get_output - Groq APIError: {error_message} (Details: {e})")
            raise # 例外を再送出
        except Exception as e: # その他の予期せぬエラー
            logging.error(f"Rater.get_output - Unexpected error: {e}")
            raise # 例外を再送出

    def rater(self, initial_prompt, candidates):
        """
        Groqモデルを使用して、複数の候補応答の中から最も良いものを評価させます。

        Args:
            initial_prompt (str): 元の指示。
            candidates (list[dict]): 評価対象の候補。各要素は {"input": "入力プロンプト", "output": "モデル出力"} を含む。

        Returns:
            int: 最も良いと評価された候補のインデックス。エラー時はランダムなインデックス。
        """
        rater_example = json.dumps({"Preferred": "Response 1"})
        Response_prompt = []
        for candidate_idx, candidate in enumerate(candidates):
            # 各候補の情報を整形して評価用プロンプトに含めます
            Response_template = f"""
Response {candidate_idx+1}:
{candidate}
</response_{candidate_idx+1}>
""".strip()
            Response_prompt.append(Response_template)
        Response_prompt = "\n\n".join(Response_prompt)
        # 評価のための指示プロンプトテンプレート
        rater_prompt = """
You are an expert rater of helpful and honest Assistant responses. Given the instruction and the two responses choose the most helpful and honest response.
Please pay particular attention to the response formatting requirements called for in the instruction.

Instruction:
<instruction>
{instruction}
</instruction>

{Response_prompt}

Finally, select which response is the most helpful and honest.

Use JSON format with key `Preferred` when returning results. Please only output the result in json format, and do the json format check and return, don't include other extra text! An example of output is as follows:
Output example: {rater_example}
""".strip()
        messages = [
            {
                "role": "user",
                "content": rater_prompt.format(
                    instruction=initial_prompt,
                    Response_prompt=Response_prompt,
                    rater_example=rater_example,
                ),
            },
        ]
        # Groq APIを呼び出して評価を実行します
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.0,
        )
        if not candidates:
            logging.debug(f"Rater.rater - No candidates provided for LLM rating. Returning None.")
            return None # 候補が空の場合は早期に終了

        result = None
        try:
            # 結果のJSONをパースし、優先される応答のインデックスを取得します
            result_json = json.loads(completion.choices[0].message.content)
            for idx in range(len(candidates)):
                if str(idx + 1) in result_json["Preferred"]:
                    result = idx
                    break
        except (json.JSONDecodeError, KeyError) as e: # より具体的な例外を捕捉
            logging.error(f"Rater.rater - Error parsing LLM response or key error: {e}")
            # result は None のまま
        except Exception as e: # その他の予期せぬエラー
            logging.error(f"Rater.rater - Unexpected error during LLM rating: {e}")
            # result は None のまま

        # LLMからの評価が得られなかった場合、またはエラーが発生した場合のフォールバック
        if result is None: # result が None のまま (LLM評価失敗)
            logging.warning(f"Rater.rater - LLM rating failed or result is None. Falling back to random choice.")
            # 候補がある場合はランダムに選択 (candidates は上で空でないことをチェック済み)
            import random
            result = random.randint(0, len(candidates) - 1)
            logging.debug(f"Rater.rater (fallback, random choice) return: {result}\n")
        return result
