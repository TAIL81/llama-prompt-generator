import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Union

import groq
from dotenv import load_dotenv  # 環境変数をロードするためのライブラリ
from groq import Groq  # Groq APIクライアント
from groq.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)  # チャット補完メッセージの型ヒント
from groq.types.chat.completion_create_params import ResponseFormat

from src.rater import Rater  # Raterクラスを絶対インポートに変更

# 環境変数を読み込みます
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 定数定義 ---
# プロンプトテンプレートを共通化
BASE_PROMPT_TEMPLATE = """
You are an instruction engineer. Your task is to rewrite the initial instruction in the <instruction> XML tag based on the suggestions in the instruction guide in the <guide> XML tag.

Instruction guide:
<guide>
{guide}
</guide>

The rewritten instruction must be a JSON object with two keys: "prompt_text" and "variables".
- "prompt_text": The rewritten instruction text.
- "variables": A list of strings, where each string is a customizable variable found in the "prompt_text". Each variable in this list must be in the double curly brace format (e.g., `{{variable_name}}`).

Customizable variables are enclosed in double curly braces (e.g., {{{{variable_name}}}}). You must preserve these variables exactly as they appear in the initial instruction.

Please use the same language as the initial instruction for rewriting.

<instruction>
{initial}
</instruction>
""".strip()


EXAMPLE_TEMPLATE = """
<example>
{demo}
</example>
""".strip()


# --- データクラス ---
@dataclass
class GroqConfig:
    rewrite_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_tokens: int = 8192  # 生成される応答の最大トークン数
    temperature: float = 0.7  # 応答の多様性を制御する温度パラメータ
    response_format: Optional[ResponseFormat] = field(
        default_factory=lambda: {"type": "json_object"}
    )


# --- 初期化処理 ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
prompt_guide_path = os.path.join(current_script_path, "PromptGuide.md")


@lru_cache(maxsize=1)
def load_prompt_guide(path: str) -> str:
    """PromptGuide.md ファイルを読み込み、キャッシュします。"""  # PromptGuide.mdファイルを読み込む関数
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


PromptGuide = load_prompt_guide(prompt_guide_path)  # プロンプトガイドの内容をロード

groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
if not groq_api_key:  # Groq APIキーが設定されているか確認
    logging.error("GROQ_API_KEY環境変数が設定されていません。")
    raise ValueError("GROQ_API_KEY環境変数が設定されていません。")
groq_client = Groq(api_key=groq_api_key)


# --- メインクラス ---
class APE:
    def __init__(self) -> None:
        self.rater = Rater()
        self.config = GroqConfig()

    def _call_groq_api(
        self,
        messages: List[ChatCompletionMessageParam],
        method_name: str,
    ) -> Optional[str]:
        """Groq APIを呼び出し、エラーハンドリングを共通化します。"""
        try:
            completion = groq_client.chat.completions.create(
                model=self.config.rewrite_model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format=self.config.response_format,
            )
            result = completion.choices[0].message.content or ""
            logging.debug(f"APE.{method_name} successful, result: {result}")
            return result
        except groq.InternalServerError as e:
            error_message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logging.error(
                f"APE.{method_name} - Groq InternalServerError: {error_message} (Details: {e})"
            )
            return None
        except groq.APIError as e:
            error_message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logging.error(
                f"APE.{method_name} - Groq APIError: {error_message} (Details: {e})"
            )
            return None
        except Exception as e:
            logging.error(f"APE.{method_name} - Unexpected error: {e}")
            return None

    def rewrite(self, initial_prompt: str) -> Optional[str]:
        """初期プロンプトをInstruction guideに基づいて書き換えます。"""
        prompt = BASE_PROMPT_TEMPLATE.format(guide=PromptGuide, initial=initial_prompt)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        return self._call_groq_api(messages, "rewrite")

    def generate_more(self, initial_prompt: str, example: str) -> Optional[str]:
        """初期プロンプトと既存の良い例を基に、さらにプロンプト候補を生成します。"""
        prompt_with_example = f"{BASE_PROMPT_TEMPLATE}\n\n{EXAMPLE_TEMPLATE}".format(
            guide=PromptGuide, initial=initial_prompt, demo=example
        )
        final_prompt = f"{prompt_with_example}\n\nPlease only output the rewrite result in JSON format."
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": final_prompt}
        ]
        return self._call_groq_api(messages, "generate_more")

    def _validate_inputs(self, initial_prompt: str, demo_data: Dict[str, str]) -> None:
        """入力パラメータの検証を行います。"""
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です")
        if not demo_data:
            logging.warning(
                "デモデータが提供されていません。APEの実行に影響する可能性があります。"
            )

    def _log_final_candidate(self, candidate: Dict[str, str]) -> None:
        """最終的な候補をデバッグログに出力します。"""
        logging.debug("APE.__call__ return:")
        for key, value in candidate.items():
            logging.debug(f"  {key}:")
            if isinstance(value, str):
                for line in value.splitlines():
                    logging.debug(f"    {line}")
            else:
                logging.debug(f"    {value}")

    def _generate_initial_candidates(
        self, initial_prompt: str, num_candidates: int = 2
    ) -> List[Dict[str, Any]]:
        """初期プロンプトから複数の候補を生成します。"""
        raw_candidates = []
        for _ in range(num_candidates):
            response_str = self.rewrite(initial_prompt)
            if not response_str:
                continue
            try:
                rewritten_data = json.loads(response_str)
                if "prompt_text" in rewritten_data and "variables" in rewritten_data:
                    raw_candidates.append(rewritten_data)
                else:
                    logging.warning(f"Invalid JSON format from rewrite: {response_str}")
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON from rewrite: {response_str}")
        return raw_candidates

    def _filter_candidates(
        self, candidates: List[Dict[str, Any]], demo_data: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """カスタマイズ可能な変数に基づいて候補をフィルタリングします。"""
        customizable_variable_set = set(demo_data.keys())
        filtered = [
            c
            for c in candidates
            if set(c.get("variables", [])) == customizable_variable_set
        ]

        if not filtered:
            logging.warning("No candidates left after filtering for customizable variables.")
            candidates_log_str = "\n".join(
                f"--- Candidate {i+1} ---\n{json.dumps(c, indent=2)}"
                for i, c in enumerate(candidates)
            )
            logging.warning(
                f"The following candidates were filtered out:\n{candidates_log_str}"
            )
        return filtered

    def _rate_and_select_best_candidate(
        self,
        initial_prompt: str,
        candidates: List[Dict[str, Any]],
        demo_data: Dict[str, str],
    ) -> Dict[str, Any]:
        """候補を評価し、最良のものを選択します。"""
        logging.info("Rating initial candidates generated by APE.rewrite...")
        rater_candidates = [{"prompt": c["prompt_text"]} for c in candidates]
        best_candidate_idx = self.rater(initial_prompt, rater_candidates, demo_data)
        logging.info(f"Initial rating completed. Best candidate index: {best_candidate_idx}")

        if best_candidate_idx is not None and 0 <= best_candidate_idx < len(candidates):
            return candidates[best_candidate_idx]
        
        logging.warning("Rater returned invalid index or failed, using first candidate as fallback.")
        return candidates[0]

    def _iteratively_improve(
        self,
        initial_prompt: str,
        best_candidate_obj: Dict[str, Any],
        demo_data: Dict[str, str],
        epoch: int,
    ) -> Dict[str, Any]:
        """最良の候補を反復的に改善します。"""
        customizable_variable_set = set(demo_data.keys())
        for i in range(epoch):
            more_candidate_response = self.generate_more(
                initial_prompt, best_candidate_obj["prompt_text"]
            )
            if not more_candidate_response:
                logging.warning(f"generate_more failed in epoch {i+1}. Keeping previous best candidate.")
                continue

            try:
                more_candidate_data = json.loads(more_candidate_response)
                if not ("prompt_text" in more_candidate_data and "variables" in more_candidate_data):
                    logging.warning(f"Invalid JSON format from generate_more: {more_candidate_response}")
                    continue
                
                if set(more_candidate_data.get("variables", [])) != customizable_variable_set:
                    logging.warning(f"generate_more produced a prompt with incorrect variables: {more_candidate_response}")
                    continue

                # Rate new candidate against the current best
                current_rating_candidates_obj = [best_candidate_obj, more_candidate_data]
                rater_loop_candidates = [{"prompt": c["prompt_text"]} for c in current_rating_candidates_obj]

                logging.info(f"Rating candidates in epoch {i+1}: [current_best vs new_generated]")
                rated_idx_loop = self.rater(initial_prompt, rater_loop_candidates, demo_data)
                logging.info(f"Epoch {i+1} rating completed. Winning index: {rated_idx_loop}")

                if rated_idx_loop is not None:
                    best_candidate_obj = current_rating_candidates_obj[rated_idx_loop]
                else:
                    logging.warning(f"Rater failed in epoch {i+1}. Keeping previous best candidate.")

            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON from generate_more: {more_candidate_response}")
        
        return best_candidate_obj

    def __call__(
        self, initial_prompt: str, epoch: int, demo_data: Dict[str, str]
    ) -> Mapping[str, Union[str, None]]:
        """APE処理を実行します。"""
        self._validate_inputs(initial_prompt, demo_data)

        # 1. 初期候補生成
        raw_candidates = self._generate_initial_candidates(initial_prompt)
        if not raw_candidates:
            logging.error("Initial prompt rewriting failed for all attempts.")
            return {
                "prompt": initial_prompt,
                "error": "Initial prompt rewriting failed.",
            }

        # 2. 候補フィルタリング
        filtered_candidates = self._filter_candidates(raw_candidates, demo_data)
        if not filtered_candidates:
            return {
                "prompt": initial_prompt,
                "error": "No valid candidates after filtering. The rewritten prompts might be missing required variables.",
            }

        # 3. 初期評価と最良候補の選択
        best_candidate_obj = self._rate_and_select_best_candidate(
            initial_prompt, filtered_candidates, demo_data
        )

        # 4. 反復的な改善
        best_candidate_obj = self._iteratively_improve(
            initial_prompt, best_candidate_obj, demo_data, epoch
        )

        final_prompt = {"prompt": best_candidate_obj["prompt_text"]}
        self._log_final_candidate(final_prompt)
        return final_prompt
