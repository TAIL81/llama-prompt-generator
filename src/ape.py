import json
import os

from groq import Groq
from dotenv import load_dotenv
# 環境変数を読み込みます
load_dotenv()

# 現在のスクリプトが配置されているディレクトリを取得します
current_script_path = os.path.dirname(os.path.abspath(__file__))

# ファイルへのフルパスを構築します
prompt_guide_path = os.path.join(current_script_path, "PromptGuide.md")

# フルパスを使用してファイルを開きます
with open(prompt_guide_path, "r", encoding="utf-8") as f:
    PromptGuide = f.read()

# Groq APIキーを取得し、クライアントを初期化します
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

from rater import Rater

# APE (Automatic Prompt Engineering) を行うクラス
class APE:
    def __init__(self):
        # プロンプト評価用のRaterクラスを初期化します
        self.rater = Rater()

    def __call__(self, initial_prompt, epoch, demo_data):
        """
        APE処理を実行します。

        Args:
            initial_prompt (str): 初期プロンプト。
            epoch (int): 最適化の繰り返し回数。
            demo_data (dict): デモデータ（キーと値のペア）。

        Returns:
            dict: 最も評価の高かったプロンプト候補。
        """
        candidates = []
        for _ in range(2):
            candidates.append(self.rewrite(initial_prompt)) # ここでは candidates_raw ではなく直接 candidates に追加
        # candidates_raw = candidates.copy() # この行は不要になるか、上のループ結果を raw とするか検討
        customizable_variable_list = list(demo_data.keys())
        filtered_candidates = [ # 変数名を filtered_candidates に変更
            {"prompt": candidate}
            # カスタマイズ可能な変数がすべて含まれている候補のみをフィルタリングします
            for candidate in candidates
            if all(
                [
                    customizable_variable in candidate
                    for customizable_variable in customizable_variable_list
                ]
            )
        ]
        if not filtered_candidates:
            print("Warning: No candidates left after filtering for customizable variables. Returning initial prompt.")
            return {"prompt": initial_prompt, "error": "No valid candidates after filtering."}

        # 候補プロンプトを評価し、最良のものを選択します
        best_candidate_idx = self.rater(initial_prompt, filtered_candidates, demo_data)

        if best_candidate_idx is None:
            print("Error: Rater did not return a valid candidate index. Using the first available filtered candidate.")
            # filtered_candidates は上で空でないことをチェック済みなので、少なくとも1要素はあるはず
            best_candidate_obj = filtered_candidates[0]
        else:
            best_candidate_obj = filtered_candidates[best_candidate_idx]

        for i in range(epoch): # epoch の回数だけループ
            # 最良の候補を基にさらに候補を生成します
            more_candidate_prompt = self.generate_more(
                initial_prompt, best_candidate_obj["prompt"] # オブジェクトのプロンプトを使用
            )
            # 新しい候補と現在の最良候補でリストを作成
            current_rating_candidates = [best_candidate_obj, {"prompt": more_candidate_prompt}]

            # 再度評価し、最良のものを選択します
            rated_idx_loop = self.rater(initial_prompt, current_rating_candidates, demo_data)

            if rated_idx_loop is None:
                print(f"Warning: Rater failed in epoch {i+1}. Keeping previous best candidate.")
                # 評価に失敗した場合は、現在の best_candidate_obj を維持
            else:
                best_candidate_obj = current_rating_candidates[rated_idx_loop]

        print(f"DEBUG: APE.__call__ return: {best_candidate_obj}")
        return best_candidate_obj

    def rewrite(self, initial_prompt):
        """
        初期プロンプトをInstruction guideに基づいて書き換えます。

        Args:
            initial_prompt (str): 書き換え対象の初期プロンプト。

        Returns:
            str: 書き換えられたプロンプト。
        """
        prompt = """
You are a instruction engineer. Your task is to rewrite the initial instruction in <instruction> xml tag based on the suggestions in the instruction guide in <guide> xml tag.

Instruction guide:
<guide>
{guide}
</guide>

You are a instruction engineer. Your task is to rewrite the initial instruction in <instruction> xml tag based on the suggestions in the instruction guide in <guide> xml tag.
which is included using double pointed brackets is customizable text that will be replaced at runtime. This needs to be kept as is.
Please same language as the initial instruction for rewriting.

<instruction>
{initial}
</instruction>


Please only output the rewrite result.
""".strip()
        messages = [
            {
                "role": "user",
                "content": prompt.format(guide=PromptGuide, initial=initial_prompt),
            }
        ]
        # Groq APIを使用してプロンプトの書き換えをリクエストします
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.0,
        )
        result = completion.choices[0].message.content
        # 結果から不要なXMLタグを除去します
        if result.startswith("<instruction>"):
            result = result[13:]
        if result.endswith("</instruction>"):
            result = result[:-14]
        result = result.strip()
        print(f"DEBUG: APE.rewrite return: {result}")
        return result

    def generate_more(self, initial_prompt, example):
        """
        初期プロンプトと既存の良い例を基に、さらにプロンプト候補を生成します。

        Args:
            initial_prompt (str): 初期プロンプト。
            example (str): 参考となる既存のプロンプト例。

        Returns:
            str: 新たに生成されたプロンプト。
        """
        prompt = """
You are a instruction engineer. Your task is to rewrite the initial instruction in <instruction> xml tag based on the suggestions in the instruction guide in <guide> xml tag.

Instruction guide:
<guide>
{guide}
</guide>

You are a instruction engineer. Your task is to rewrite the initial instruction in <instruction> xml tag based on the suggestions in the instruction guide in <guide> xml tag.
which is included using double pointed brackets is customizable text that will be replaced at runtime. This needs to be kept as is.
Please same language as the initial instruction for rewriting.

<instruction>
{initial}
</instruction>

<example>
{demo}
</example>

Please only output the rewrite result.
""".strip()
        messages = [
            {
                "role": "user",
                "content": prompt.format(
                    guide=PromptGuide, initial=initial_prompt, demo=example
                ),
            }
        ]
        # Groq APIを使用して追加のプロンプト候補を生成します
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.0,
        )
        result = completion.choices[0].message.content
        # 結果から不要なXMLタグを除去します
        if result.startswith("<instruction>"):
            result = result[13:]
        if result.endswith("</instruction>"):
            result = result[:-14]
        result = result.strip()
        print(f"DEBUG: APE.generate_more return: {result}")
        return result
