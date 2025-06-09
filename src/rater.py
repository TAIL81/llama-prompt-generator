import json
from groq import Groq
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
        return rate

    def get_output(self, prompt):
        """指定されたプロンプトでGroqモデルを実行し、出力を取得します。"""
        messages = [{"role": "user", "content": prompt}]
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.8,
        )
        result = completion.choices[0].message.content
        return result

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
            {"role": "assistant", "content": "{"},
        ]
        # Groq APIを呼び出して評価を実行します
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.8,
        )
        result = None
        try:
            # 結果のJSONをパースし、優先される応答のインデックスを取得します
            result_json = json.loads(completion.choices[0].message.content)
            for idx in range(len(candidates)):
                if str(idx + 1) in result_json["Preferred"]:
                    result = idx
                    break
        except:
            # JSONパースエラーなどが発生した場合は、ランダムなインデックスを返します
            # TODO: エラーハンドリングをより詳細に行うべき
            import random
            result = random.randint(0, len(candidates) - 1)
        if result is None:
            import random
            result = random.randint(0, len(candidates) - 1)
        return result
