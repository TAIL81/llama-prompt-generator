import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import groq
from dotenv import load_dotenv
from groq import Groq

# 環境変数を読み込みます
load_dotenv()

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class GroqConfig:
    rewrite_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    rating_model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 8192
    temperature: float = 0.1


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

# Groq APIキーを取得し、クライアントを初期化します
groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logging.error("GROQ_API_KEY環境変数が設定されていません。")
    raise ValueError("GROQ_API_KEY環境変数が設定されていません。")
groq_client = Groq(api_key=groq_api_key)

from rater import Rater


# APE (Automatic Prompt Engineering) を行うクラス
class APE:
    def __init__(self) -> None:
        # プロンプト評価用のRaterクラスを初期化します
        self.rater = Rater()
        self.config = GroqConfig()  # 設定を初期化

    def __call__(self, initial_prompt: str, epoch: int, demo_data: Dict[str, str]) -> Dict[str, Union[str, None]]:
        """
        APE処理を実行します。

        Args:
            initial_prompt (str): 初期プロンプト。
            epoch (int): 最適化の繰り返し回数。
            demo_data (Dict[str, str]): デモデータ（キーと値のペア）。

        Returns:
            Dict[str, Union[str, None]]: 最も評価の高かったプロンプト候補。
        """
        self._validate_inputs(initial_prompt, demo_data)

        candidates: List[str] = []
        for _ in range(2):
            rewritten_prompt: Optional[str] = self.rewrite(initial_prompt)
            if rewritten_prompt:  # rewriteが成功した場合のみ追加
                candidates.append(rewritten_prompt)

        if not candidates:  # 2回のrewriteが両方失敗した場合
            logging.error("Initial prompt rewriting failed for all attempts. Returning initial prompt.")
            return {"prompt": initial_prompt, "error": "Initial prompt rewriting failed."}

        customizable_variable_list: List[str] = list(demo_data.keys())
        filtered_candidates: List[Dict[str, str]] = [
            {"prompt": candidate}
            # カスタマイズ可能な変数がすべて含まれている候補のみをフィルタリングします
            for candidate in candidates
            if all([customizable_variable in candidate for customizable_variable in customizable_variable_list])
        ]
        if not filtered_candidates:
            logging.warning("No candidates left after filtering for customizable variables. Returning initial prompt.")
            return {"prompt": initial_prompt, "error": "No valid candidates after filtering."}

        # 候補プロンプトを評価し、最良のものを選択します
        best_candidate_idx: Optional[int] = self.rater(initial_prompt, filtered_candidates, demo_data)

        if best_candidate_idx is None:
            logging.error("Rater did not return a valid candidate index. Using the first available filtered candidate.")
            # filtered_candidates は上で空でないことをチェック済みなので、少なくとも1要素はあるはず
            best_candidate_obj: Dict[str, str] = filtered_candidates[0]
        else:
            best_candidate_obj = filtered_candidates[best_candidate_idx]

        for i in range(epoch):  # epoch の回数だけループ
            # 最良の候補を基にさらに候補を生成します
            more_candidate_prompt: Optional[str] = self.generate_more(
                initial_prompt, best_candidate_obj["prompt"]  # オブジェクトのプロンプトを使用
            )
            if more_candidate_prompt:  # generate_moreが成功した場合
                # 新しい候補と現在の最良候補でリストを作成
                current_rating_candidates: List[Dict[str, str]] = [
                    best_candidate_obj,
                    {"prompt": more_candidate_prompt},
                ]

                # 再度評価し、最良のものを選択します
                rated_idx_loop: Optional[int] = self.rater(initial_prompt, current_rating_candidates, demo_data)

                if rated_idx_loop is None:
                    logging.warning(f"Rater failed in epoch {i+1}. Keeping previous best candidate.")
                    # 評価に失敗した場合は、現在の best_candidate_obj を維持
                else:
                    best_candidate_obj = current_rating_candidates[rated_idx_loop]
            else:
                logging.warning(f"generate_more failed in epoch {i+1}. Keeping previous best candidate.")
                # generate_more に失敗した場合も、現在の best_candidate_obj を維持

        logging.debug("APE.__call__ return:")
        for key, value in best_candidate_obj.items():
            logging.debug(f"  {key}:")
            # 値が文字列の場合、改行を維持してインデント付きで表示
            if isinstance(value, str):
                for line in value.splitlines():
                    logging.debug(f"    {line}")
            else:
                # 文字列以外の場合は、そのままインデント付きで表示
                logging.debug(f"    {value}")
        return best_candidate_obj

    def _validate_inputs(self, initial_prompt: str, demo_data: Dict[str, str]) -> None:
        """
        入力パラメータの検証を行います。
        """
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です")
        if not demo_data:
            raise ValueError("デモデータが提供されていません")

    def rewrite(self, initial_prompt: str) -> Optional[str]:
        """
        初期プロンプトをInstruction guideに基づいて書き換えます。

        Args:
            initial_prompt (str): 書き換え対象の初期プロンプト。

        Returns:
            Optional[str]: 書き換えられたプロンプト。エラーの場合はNone。
        """
        prompt: str = (
            """
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
        )
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": prompt.format(guide=PromptGuide, initial=initial_prompt),
            }
        ]
        try:
            # Groq APIを使用してプロンプトの書き換えをリクエストします
            completion = groq_client.chat.completions.create(
                model=self.config.rewrite_model,
                messages=messages,
                max_completion_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            result: str = completion.choices[0].message.content
            # 結果から不要なXMLタグを除去します
            if result.startswith("<instruction>"):
                result = result[13:]
            if result.endswith("</instruction>"):
                result = result[:-14]
            result = result.strip()
            logging.debug(f"APE.rewrite successful, result: \n{result}\n")
            return result
        except groq.InternalServerError as e:
            error_message: str = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logging.error(f"APE.rewrite - Groq InternalServerError: {error_message} (Details: {e})")
            return None
        except groq.APIError as e:
            error_message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logging.error(f"APE.rewrite - Groq APIError: {error_message} (Details: {e})")
            return None
        except Exception as e:
            logging.error(f"APE.rewrite - Unexpected error: {e}")
            return None

    def generate_more(self, initial_prompt: str, example: str) -> Optional[str]:
        """
        初期プロンプトと既存の良い例を基に、さらにプロンプト候補を生成します。

        Args:
            initial_prompt (str): 初期プロンプト。
            example (str): 参考となる既存のプロンプト例。

        Returns:
            Optional[str]: 新たに生成されたプロンプト。エラーの場合はNone。
        """
        prompt: str = (
            """
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
        )
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": prompt.format(guide=PromptGuide, initial=initial_prompt, demo=example),
            }
        ]
        try:
            # Groq APIを使用して追加のプロンプト候補を生成します
            completion = groq_client.chat.completions.create(
                model=self.config.rewrite_model,
                messages=messages,
                max_completion_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            result: str = completion.choices[0].message.content
            # 結果から不要なXMLタグを除去します
            if result.startswith("<instruction>"):
                result = result[13:]
            if result.endswith("</instruction>"):
                result = result[:-14]
            result = result.strip()
            logging.debug(f"APE.generate_more successful, result: \n{result}\n")
            return result
        except groq.InternalServerError as e:
            error_message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logging.error(f"APE.generate_more - Groq InternalServerError: {error_message} (Details: {e})")
            return None
        except groq.APIError as e:
            error_message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logging.error(f"APE.generate_more - Groq APIError: {error_message} (Details: {e})")
            return None
        except Exception as e:
            logging.error(f"APE.generate_more - Unexpected error: {e}")
            return None
