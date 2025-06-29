import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Union

import groq
from dotenv import load_dotenv
from groq import Groq

from rater import Rater

# 環境変数を読み込みます
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass  # データクラス。API設定を保持
class GroqConfig:
    rewrite_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_tokens: int = 8192
    temperature: float = 0.2


# 現在のスクリプトが配置されているディレクトリを取得します
current_script_path = os.path.dirname(os.path.abspath(__file__))

# PromptGuide.md へのフルパスを構築します
prompt_guide_path = os.path.join(current_script_path, "PromptGuide.md")


@lru_cache(maxsize=1)
def load_prompt_guide(path: str) -> str:  # プロンプトガイドを読み込み、キャッシュ
    """
    PromptGuide.md ファイルを読み込み、キャッシュします。
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


PromptGuide = load_prompt_guide(prompt_guide_path)  # プロンプトガイドをロード

# Groq APIキーを取得し、クライアントを初期化します
groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logging.error("GROQ_API_KEY環境変数が設定されていません。")
    raise ValueError("GROQ_API_KEY環境変数が設定されていません。")
groq_client = Groq(api_key=groq_api_key)


# APE (Automatic Prompt Engineering) を行うクラス
class APE:
    def __init__(self) -> None:  # 初期化
        # プロンプト評価用のRaterクラスを初期化します
        self.rater = Rater()
        self.config = GroqConfig()  # 設定を初期化

    def __call__(self, initial_prompt: str, epoch: int, demo_data: Dict[str, str]) -> Mapping[str, Union[str, None]]:
        """
        APE処理を実行します。

        Args:
            initial_prompt (str): 初期プロンプト。
            epoch (int): 最適化の繰り返し回数。
            demo_data (Dict[str, str]): デモデータ（キーと値のペア）。

        Returns:
            Dict[str, Union[str, None]]: 最も評価の高かったプロンプト候補。
        """
        self._validate_inputs(initial_prompt, demo_data)  # 入力を検証

        candidates: List[str] = []
        for _ in range(2):  # 2回プロンプトを書き換え
            rewritten_prompt: Optional[str] = self.rewrite(initial_prompt)  # プロンプトを書き換え
            if rewritten_prompt:  # 書き換えに成功した場合
                candidates.append(rewritten_prompt)  # 候補リストに追加

        if not candidates:  # 書き換え候補が1つも生成されなかった場合
            logging.error("Initial prompt rewriting failed for all attempts. Returning initial prompt.")
            return {"prompt": initial_prompt, "error": "Initial prompt rewriting failed."}

        # 候補をDict[str, str]の形式に変換
        candidate_dicts: List[Dict[str, str]] = [{"prompt": c} for c in candidates]

        # デモデータからカスタマイズ可能な変数のリストを取得し、フィルタリング
        customizable_variable_list: List[str] = [f"{{{k}}}" for k in demo_data.keys()]
        filtered_candidates: List[Dict[str, str]] = [
            c for c in candidate_dicts if all(var in c["prompt"] for var in customizable_variable_list)
        ]
        # フィルタリングの結果、候補が残らなかった場合
        if not filtered_candidates:
            logging.warning("No candidates left after filtering for customizable variables. Returning initial prompt.")
            # 候補リストを整形して、1つのログエントリで見やすく出力します
            candidates_log_str = "\n".join(f"--- Candidate {i+1} ---\n{c}" for i, c in enumerate(candidates))
            logging.warning(f"The following candidates were filtered out:\n{candidates_log_str}")
            return {
                "prompt": initial_prompt,
                "error": "No valid candidates after filtering. The rewritten prompts might be missing some required variables.",
            }

        # 候補プロンプトを評価し、最良のものを選択します
        best_candidate_idx: Optional[int] = self.rater(initial_prompt, filtered_candidates, demo_data)

        if best_candidate_idx is None and filtered_candidates:
            logging.error("Rater did not return a valid candidate index. Using the first available filtered candidate.")
            # filtered_candidates は上で空でないことをチェック済みなので、少なくとも1要素はあるはず
            best_candidate_obj: Dict[str, str] = filtered_candidates[0]
        elif best_candidate_idx is not None:
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
            else: # type: ignore
                logging.warning(f"generate_more failed in epoch {i+1}. Keeping previous best candidate.")
                # generate_more に失敗した場合も、現在の best_candidate_obj を維持

        logging.debug("APE.__call__ return:")  # デバッグログ出力
        for key, value in best_candidate_obj.items():  # 最良の候補の情報をログ出力
            logging.debug(f"  {key}:")  # キーを出力
            if isinstance(value, str):  # 値が文字列の場合
                for line in value.splitlines():  # 改行ごとに
                    logging.debug(f"    {line}")  # インデントして出力
            else:  # 文字列以外の場合
                logging.debug(f"    {value}")  # そのままインデントして出力
        return best_candidate_obj  # type: ignore
 
    def _validate_inputs(self, initial_prompt: str, demo_data: Dict[str, str]) -> None:
        """
        入力パラメータの検証を行います。
        """
        if not initial_prompt.strip():
            raise ValueError("初期プロンプトが空です")
        # demo_dataが空の場合でもエラーをスローしないように変更
        # ただし、空の場合は警告をログに出力する
        if not demo_data:
            logging.warning("デモデータが提供されていません。APEの実行に影響する可能性があります。")

    def rewrite(self, initial_prompt: str) -> Optional[str]:
        """初期プロンプトをInstruction guideに基づいて書き換えます。

        Args:
            initial_prompt (str): 書き換え対象の初期プロンプト。

        Returns:
            Optional[str]: 書き換えられたプロンプト。エラーの場合はNone。
        """  # ドキュメント文字列
        # プロンプトのテンプレートを定義
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
        # ユーザーに送信するメッセージを作成
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": prompt.format(guide=PromptGuide, initial=initial_prompt),  # プロンプトをフォーマット
            }
        ]
        try:
            # Groq APIを使用してプロンプトの書き換えをリクエストします
            completion = groq_client.chat.completions.create(  # Groq APIを呼び出し
                model=self.config.rewrite_model, # type: ignore
                messages=messages, # type: ignore # メッセージを渡す
                max_completion_tokens=self.config.max_tokens,  # 最大トークン数を設定
                temperature=self.config.temperature,  # 温度パラメータを設定
            )
            result: str = completion.choices[0].message.content or ""  # APIの応答からコンテンツを取得
            # 結果から不要なXMLタグを除去します
            if result.startswith("<instruction>"):  # <instruction>タグで始まる場合
                result = result[13:]  # タグを取り除く
            if result.endswith("</instruction>"):  # </instruction>タグで終わる場合
                result = result[:-14]  # タグを取り除く
            result = result.strip()  # 前後の空白を削除
            logging.debug(f"APE.rewrite successful, result: \n{result}\n")  # デバッグログを出力
            return result  # 書き換えられたプロンプトを返す
        # APIエラーをハンドル
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
                model=self.config.rewrite_model,  # モデルを指定
                messages=messages,  # type: ignore
                max_completion_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            result: str = completion.choices[0].message.content or ""
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
