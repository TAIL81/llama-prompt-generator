import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import groq
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from groq.types.chat.completion_create_params import ResponseFormat

from src.rater import Rater

# .env 読み込み（アプリ側と二重設定しない前提で軽量に実行）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# モジュールロガー（basicConfigはアプリ側で設定）
logger = logging.getLogger(__name__)

# --- 定数定義 ---
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
    """Groq API 呼び出し設定"""

    rewrite_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_tokens: int = 8192
    temperature: float = 0.7
    response_format: Optional[ResponseFormat] = field(
        default_factory=lambda: {"type": "json_object"}
    )


# --- 初期化処理 ---
current_script_dir = Path(__file__).resolve().parent
prompt_guide_path = current_script_dir / "PromptGuide.md"


@lru_cache(maxsize=1)
def load_prompt_guide(path: Path) -> str:
    """PromptGuide.md を読み込み、キャッシュする。"""
    return path.read_text(encoding="utf-8")


PromptGuide = load_prompt_guide(prompt_guide_path)

groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY 環境変数が設定されていません。")
    raise ValueError("GROQ_API_KEY 環境変数が設定されていません。")
groq_client = Groq(api_key=groq_api_key)


class APE:
    """
    APE: Automatic Prompt Engineering
    - 初期プロンプトをガイドに基づき書き換え、候補を生成/評価/反復して最良を返す。
    """

    def __init__(self) -> None:
        self.rater = Rater()
        self.config = GroqConfig()

    # 旧I/F維持のための内部共通呼び出し
    def _call_groq_api(
        self,
        messages: List[ChatCompletionMessageParam],
        method_name: str,
    ) -> Optional[str]:
        """
        Groq API を呼び出す共通関数。
        成功: content(str)、失敗: None
        """
        try:
            completion = groq_client.chat.completions.create(
                model=self.config.rewrite_model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format=self.config.response_format,
            )
            content = completion.choices[0].message.content or ""
            logger.debug("APE.%s 成功: len=%s", method_name, len(content))
            return content
        except groq.InternalServerError as e:
            # e.body の安全な取り扱い
            message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logger.error("APE.%s - Groq InternalServerError: %s", method_name, message)
            return None
        except groq.APIError as e:
            message = (
                e.body.get("error", {}).get("message", str(e))
                if hasattr(e, "body") and isinstance(e.body, dict)
                else str(e)
            )
            logger.error("APE.%s - Groq APIError: %s", method_name, message)
            return None
        except Exception as e:
            logger.error("APE.%s - 想定外エラー: %s", method_name, e)
            return None

    def rewrite(self, initial_prompt: str) -> Optional[str]:
        """初期プロンプトをガイドに基づき JSON 形式へ書き換える。"""
        prompt_text = BASE_PROMPT_TEMPLATE.format(
            guide=PromptGuide, initial=initial_prompt
        )
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt_text}
        ]
        return self._call_groq_api(messages, "rewrite")

    def generate_more(self, initial_prompt: str, example: str) -> Optional[str]:
        """
        初期プロンプト + 良い例(example) を元に、追加候補を 1 つ生成。
        JSON 形式のみの出力を指示。
        """
        base_with_example = f"{BASE_PROMPT_TEMPLATE}\n\n{EXAMPLE_TEMPLATE}".format(
            guide=PromptGuide, initial=initial_prompt, demo=example
        )
        final_prompt = f"{base_with_example}\n\nPlease only output the rewrite result in JSON format."
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": final_prompt}
        ]
        return self._call_groq_api(messages, "generate_more")

    def _validate_inputs(self, initial_prompt: str, demo_data: Dict[str, str]) -> None:
        """入力を検証。デモデータ未提供は warning に留める。"""
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です。")
        if not demo_data:
            logger.warning(
                "デモデータが提供されていません。APE の評価精度に影響する可能性があります。"
            )

    def _log_final_candidate(self, candidate: Dict[str, str]) -> None:
        """最終候補をデバッグログへ整形出力。"""
        logger.debug("APE.__call__ return(最終候補):")
        for key, value in candidate.items():
            logger.debug("  %s:", key)
            if isinstance(value, str):
                for line in value.splitlines():
                    logger.debug("    %s", line)
            else:
                logger.debug("    %s", value)

    def _create_initial_candidates(
        self, initial_prompt: str, num_candidates: int = 2
    ) -> List[Dict[str, Any]]:
        """
        初期プロンプトから複数候補を生成して JSON をパース。
        不正フォーマットは除外。
        """
        candidates: List[Dict[str, Any]] = []
        for _ in range(num_candidates):
            response_str = self.rewrite(initial_prompt)
            if not response_str:
                continue
            try:
                data = json.loads(response_str)
                if "prompt_text" in data and "variables" in data:
                    candidates.append(data)
                else:
                    logger.warning("rewrite の JSON 形式が不正: %s", response_str)
            except json.JSONDecodeError:
                logger.error("rewrite の JSON デコードに失敗: %s", response_str)
        return candidates

    def _filter_candidates_by_variables(
        self, candidates: List[Dict[str, Any]], demo_data: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """デモデータのキー集合と variables が一致する候補のみを残す。"""
        expected_vars = set(demo_data.keys())
        filtered = [
            c for c in candidates if set(c.get("variables", [])) == expected_vars
        ]

        if not filtered:
            logger.warning("変数一致フィルタ後に候補が残りませんでした。")
            dump = "\n".join(
                f"--- Candidate {i+1} ---\n{json.dumps(c, indent=2, ensure_ascii=False)}"
                for i, c in enumerate(candidates)
            )
            logger.warning("除外された候補一覧:\n%s", dump)
        return filtered

    def _rate_and_select_best(
        self,
        initial_prompt: str,
        candidates: List[Dict[str, Any]],
        demo_data: Dict[str, str],
    ) -> Dict[str, Any]:
        """Rater で評価しベスト候補を選ぶ。失敗時は先頭をフォールバック。"""
        logger.info("APE.rewrite の初期候補を評価中...")
        rater_inputs = [{"prompt": c["prompt_text"]} for c in candidates]
        best_idx = self.rater(initial_prompt, rater_inputs, demo_data)
        logger.info("初期評価完了。ベスト候補 index=%s", best_idx)

        if best_idx is not None and 0 <= best_idx < len(candidates):
            return candidates[best_idx]

        logger.warning("Rater 失敗/不正 index のため先頭候補を採用。")
        return candidates[0]

    def _iterate_improvement(
        self,
        initial_prompt: str,
        best: Dict[str, Any],
        demo_data: Dict[str, str],
        epoch: int,
    ) -> Dict[str, Any]:
        """
        ベスト候補に対し epoch 回の改善ループを行い、勝者を都度更新。
        variables は demo_data.keys() と一致するもののみ採用。
        """
        expected_vars = set(demo_data.keys())
        current_best = best

        for i in range(epoch):
            more_json = self.generate_more(initial_prompt, current_best["prompt_text"])
            if not more_json:
                logger.warning("generate_more 失敗: epoch=%s。現状維持。", i + 1)
                continue

            try:
                candidate = json.loads(more_json)
                if not ("prompt_text" in candidate and "variables" in candidate):
                    logger.warning("generate_more の JSON 形式が不正: %s", more_json)
                    continue

                if set(candidate.get("variables", [])) != expected_vars:
                    logger.warning("generate_more の variables が不一致: %s", more_json)
                    continue

                pair = [current_best, candidate]
                rater_pair = [{"prompt": c["prompt_text"]} for c in pair]

                logger.info("epoch=%s の評価を実行 (current_best vs new)", i + 1)
                winner_idx = self.rater(initial_prompt, rater_pair, demo_data)
                logger.info("epoch=%s 勝者 index=%s", i + 1, winner_idx)

                if winner_idx is not None:
                    current_best = pair[winner_idx]
                else:
                    logger.warning("Rater 失敗: epoch=%s。現状維持。", i + 1)

            except json.JSONDecodeError:
                logger.error("generate_more の JSON デコード失敗: %s", more_json)

        return current_best

    def __call__(
        self, initial_prompt: str, epoch: int, demo_data: Dict[str, str]
    ) -> Mapping[str, Union[str, None]]:
        """
        APE のメイン処理:
        1) 候補生成 → 2) 変数一致フィルタ → 3) 初期評価 → 4) 反復改善 → 5) 最終出力
        """
        self._validate_inputs(initial_prompt, demo_data)

        # 1. 初期候補生成
        candidates = self._create_initial_candidates(initial_prompt)
        if not candidates:
            logger.error("初期書き換えがすべて失敗しました。")
            return {
                "prompt": initial_prompt,
                "error": "Initial prompt rewriting failed.",
            }

        # 2. 変数一致でフィルタ
        filtered = self._filter_candidates_by_variables(candidates, demo_data)
        if not filtered:
            return {
                "prompt": initial_prompt,
                "error": "No valid candidates after filtering. The rewritten prompts might be missing required variables.",
            }

        # 3. 初期評価
        best = self._rate_and_select_best(initial_prompt, filtered, demo_data)

        # 4. 改善ループ
        best = self._iterate_improvement(initial_prompt, best, demo_data, epoch)

        # 5. 最終結果
        final_prompt = {"prompt": best["prompt_text"]}
        self._log_final_candidate(final_prompt)
        return final_prompt
