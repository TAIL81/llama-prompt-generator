import json
from groq import Groq
import groq # Import the groq module to access specific error types
import os
# Groq APIキーを環境変数から取得し、クライアントを初期化します
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

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
        print(f"DEBUG: Rater.__call__ return: {rate}\n")
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
            print(f"DEBUG: Rater.get_output successful, result: \n{result}\n")
            return result
        except groq.InternalServerError as e:
            error_message = e.body.get('error', {}).get('message', str(e)) if hasattr(e, 'body') and isinstance(e.body, dict) else str(e)
            print(f"ERROR: Rater.get_output - Groq InternalServerError: {error_message} (Details: {e})")
            # 呼び出し元がエラーを処理できるように、エラー情報を含む文字列を返すか、例外を再送出します。
            # ここではエラーメッセージを返します。
            return f"Groq API Internal Server Error: {error_message}"
        except groq.APIError as e: # InternalServerError以外のAPIエラーも捕捉
            error_message = e.body.get('error', {}).get('message', str(e)) if hasattr(e, 'body') and isinstance(e.body, dict) else str(e)
            print(f"ERROR: Rater.get_output - Groq APIError: {error_message} (Details: {e})")
            return f"Groq API Error: {error_message}"
        except Exception as e: # その他の予期せぬエラー
            print(f"ERROR: Rater.get_output - Unexpected error: {e}")
            return f"Unexpected error during Groq API call: {str(e)}"

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
            # {"role": "assistant", "content": "{"},
        ]
        # Groq APIを呼び出して評価を実行します
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.0,
        )
        result = None
        try:
            # 結果のJSONをパースし、優先される応答のインデックスを取得します
            # candidates が空の場合、LLMに問い合わせる意味がないかもしれないが、現状のフローを維持
            if candidates: # 候補がある場合のみLLMに評価を依頼
                result_json = json.loads(completion.choices[0].message.content)
                for idx in range(len(candidates)):
                    if str(idx + 1) in result_json["Preferred"]:
                        result = idx
                        break
            else: # 候補がない場合は評価スキップ
                print(f"DEBUG: Rater.rater - No candidates provided for LLM rating.")
                result = None
        except Exception as e: # より具体的な例外 (json.JSONDecodeError, KeyErrorなど) を捕捉する方が望ましい
            print(f"DEBUG: Rater.rater - Error parsing LLM response or key error: {e}")
            # result は None のまま

        # LLMからの評価が得られなかった場合、またはエラーが発生した場合のフォールバック
        if result is None: # result が None のまま (LLM評価失敗または候補なし)
            print(f"DEBUG: Rater.rater - LLM rating failed or result is None. Falling back.")
            if not candidates:
                # 候補が全くない場合は、有効なインデックスは返せない
                print(f"DEBUG: Rater.rater - No candidates to choose from in fallback. Returning None.")
                return None # 呼び出し元で None を処理する必要がある
            else:
                # 候補がある場合はランダムに選択
                import random
                result = random.randint(0, len(candidates) - 1) # candidates は空でないことが保証されている
                print(f"DEBUG: Rater.rater (fallback, random choice) return: {result}\n")
        return result
