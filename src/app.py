import ast
import json
import logging
import operator
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union, cast

import gradio as gr
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from ape import APE
from application.soe_prompt import SOEPrompt
from calibration import CalibrationPrompt
from metaprompt import MetaPrompt
from optimize import Alignment
from translate import GuideBased


# 設定管理クラスの導入
class AppConfig(BaseSettings):
    """アプリケーション設定を管理するクラス。

    設定は環境変数と翻訳ファイルから読み込まれます。
    """

    LANGUAGE: str = Field(default="ja", env="LANGUAGE")
    TRANSLATIONS_PATH: str = Field(
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "translations.json"
        )
    )
    lang_store: Dict[str, Any] = {}

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.lang_store = self._load_translations()

    def _load_translations(self) -> Dict[str, Any]:
        """翻訳ファイルを読み込みます。"""
        try:
            with open(self.TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"翻訳ファイルが見つかりません: {self.TRANSLATIONS_PATH}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"翻訳ファイルの形式が不正です: {self.TRANSLATIONS_PATH}. エラー: {e}")
            return {}
        except Exception as e:
            logging.error(
                f"翻訳ファイルの読み込み中に予期せぬエラーが発生しました: {self.TRANSLATIONS_PATH}. エラー: {e}"
            )
            return {}

    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        env_file_encoding = "utf-8"


# AppConfig のインスタンスを作成
config = AppConfig()
language = config.language
lang_store = config.lang_store


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


setup_logging()


# コンポーネントの遅延初期化を管理するクラス
class ComponentManager:
    """
    アプリケーションコンポーネントの初期化を遅延させるクラス。

    必要な時にコンポーネントを初期化し、リソースを効率的に利用します。
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._components: Dict[str, Any] = {
            "ape": None,
            "rewrite": None,
            "alignment": None,
            "metaprompt": None,
            "soeprompt": None,
            "calibration": None,
        }

    def __getattr__(self, name: str) -> Any:
        """存在しない属性へのアクセスを処理し、コンポーネントを初期化して返します。"""
        if name in self._components:
            if self._components[name] is None:
                self._initialize_component(name)
            return self._components[name]
        raise AttributeError(f"Component {name} not found")

    def _initialize_component(self, name: str) -> None:  # コンポーネントの初期化を行う
        logging.info(f"Initializing component: {name}")
        if name == "ape":
            self._components[name] = APE()  # APEコンポーネントの初期化
        elif name == "rewrite":
            self._components[name] = GuideBased()  # GuideBasedコンポーネントの初期化
        elif name == "alignment":
            self._components[name] = Alignment(
                lang_store=self._config.lang_store, language=self._config.language
            )
        elif name == "metaprompt":  # メタプロンプトコンポーネントの初期化
            self._components[name] = MetaPrompt()  # MetaPrompt のインスタンス化
        elif name == "soeprompt":
            self._components[name] = SOEPrompt()
        elif name == "calibration":
            self._components[name] = CalibrationPrompt()
        else:
            raise ValueError(f"Unknown component: {name}")


# コンポーネントマネージャーのインスタンスを作成
component_manager = ComponentManager(config)

# 各コンポーネントへのアクセスは component_manager.ape のように変更
# 例: ape = component_manager.ape
# ただし、既存のコードの変更を最小限にするため、ここでは直接変数に割り当てます
ape = component_manager.ape
rewrite = component_manager.rewrite
alignment = component_manager.alignment
metaprompt = component_manager.metaprompt
soeprompt = component_manager.soeprompt
calibration = component_manager.calibration

# --- クリアボタン用の関数群 ---


def clear_metaprompt_tab() -> Tuple[str, str, str, str, str, str]:
    """メタプロンプトタブの入出力をクリアします。"""
    return "", "", "", "", "", ""


def clear_translation_tab() -> Tuple[str, gr.Textbox, gr.Textbox, gr.Textbox]:
    """プロンプト翻訳タブの入出力をクリアします。"""
    # 最初のテキストボックスのみ表示し、他は非表示にリセット
    return (
        "",
        gr.Textbox(value="", visible=True),
        gr.Textbox(value="", visible=False),
        gr.Textbox(value="", visible=False),
    )


def clear_evaluation_tab() -> Tuple[str, str, str, str, str, str, str, str, str, str]:
    """プロンプト評価タブの入出力をクリアします。"""
    return "", "", "", "", "", "", "", "", "", ""


def clear_soe_tab() -> Tuple[str, str, str, str, None, str]:
    """SOE最適化タブの入出力をクリアします。"""
    return "", "", "", "", None, ""


def clear_calibration_tab() -> Tuple[str, str, str, None, int, str]:
    """プロンプトキャリブレーションタブの入出力をクリアします。"""
    default_code = "def postprocess(llm_output):\n    return llm_output"
    # task, original_prompt, postprocess_code, dataset_file, steps_num, revised_prompt
    return "", "", default_code, None, 1, ""


# generate_prompt関数の分割と型ヒントの追加
def create_single_textbox(value: str) -> List[gr.Textbox]:
    return [
        cast(
            gr.Textbox,
            gr.Textbox(
                label=lang_store[language]["Prompt Template Generated"],
                value=value,
                lines=3,
                show_copy_button=True,
                interactive=False,
            ),
        )  # 閉じ括弧を追加
    ] + [
        gr.Textbox(visible=False)
    ] * 2  # テキストボックスのリストを返す


def create_multiple_textboxes(
    candidates: List[str], judge_result: int
) -> List[gr.Textbox]:
    textboxes = []
    for i in range(3):
        is_best = "Y" if judge_result == i else "N"
        textboxes.append(
            cast(
                gr.Textbox,
                gr.Textbox(
                    label=f"{lang_store[language]['Prompt Template Generated']} #{i+1} {is_best}",
                    value=candidates[i],
                    lines=3,
                    show_copy_button=True,
                    visible=True,
                    interactive=False,
                ),
            )  # 閉じ括弧を追加
        )
    return textboxes


def generate_single_prompt(original_prompt: str) -> List[gr.Textbox]:
    """一回生成モードでプロンプトを生成し、結果をテキストボックスに表示します。"""
    """一回生成モード"""
    result = rewrite(original_prompt)
    return create_single_textbox(result)


def generate_multiple_prompts_async(original_prompt: str) -> List[str]:
    """複数回生成モードでプロンプトを非同期に生成します。"""
    """複数回生成モード (非同期実行用)"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(rewrite, original_prompt) for _ in range(3)]
        candidates = [future.result() for future in futures]
    return candidates


# プロンプト生成のメイン関数
def generate_prompt(original_prompt: str, level: str) -> List[gr.Textbox]:
    """プロンプト生成のメイン関数。
    最適化レベルに応じて、一回生成または複数回生成を行います。

    Args:
        original_prompt (str): 元のプロンプト。
        level (str): 最適化レベル ("One-time Generation" または "Multiple-time Generation")。

    Returns:
        list: Gradioテキストボックスコンポーネントのリスト。
    """
    if level == "One-time Generation":  # 一回生成モードの場合
        return generate_single_prompt(original_prompt)
    elif level == "Multiple-time Generation":
        candidates = generate_multiple_prompts_async(original_prompt)
        judge_result = rewrite.judge(candidates)
        return create_multiple_textboxes(candidates, judge_result)
    return []  # デフォルトの戻り値


from safe_executor import SafeCodeExecutor

# SafeCodeExecutor のインスタンスを作成
safe_code_executor = SafeCodeExecutor()


def ape_prompt(original_prompt: str, user_data: str) -> List[gr.Textbox]:
    """APE (Automatic Prompt Engineering) を使用してプロンプトを生成します。

    Args:
        original_prompt (str): 元のプロンプト。
        user_data (str): JSON形式のユーザーデータ。
    Returns:
        List[gr.Textbox]: Gradioテキストボックスコンポーネントのリスト。生成されたプロンプトを含む。
    """
    logging.debug("ape_prompt function was called!")
    logging.debug(f"original_prompt = {original_prompt}")
    logging.debug(f"user_data = {user_data}")

    try:
        parsed_user_data = json.loads(user_data)
    except json.JSONDecodeError as e:
        logging.error(
            f"Error decoding user_data in ape_prompt: {e}. User data was: '{user_data}'"
        )
        # エラーが発生した場合、空の辞書を返すか、適切なエラー処理を行う
        # ここでは、エラーメッセージを表示するために空のテキストボックスを返す
        return [
            gr.Textbox(
                label="Error",  # この行に "label=" が追加されました
                value=f"Error processing user data: {e}. Please ensure it's valid JSON.",
                lines=3,
                show_copy_button=True,
                interactive=False,
            )
        ] + [gr.Textbox(visible=False)] * 2

    result = ape(original_prompt, 1, parsed_user_data)
    return [
        gr.Textbox(
            label="Prompt Generated",
            value=result["prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
    ] + [
        gr.Textbox(visible=False)
    ] * 2  # 他のタブとの互換性のための非表示テキストボックス


# メタプロンプト出力に対して APE を実行する関数
async def run_ape_on_metaprompt_output(
    metaprompt_template: str, metaprompt_variables_str: str
) -> Tuple[str, str]:
    """
    メタプロンプトで生成されたテンプレートと変数文字列を元に APE を実行します。

    Args:
        metaprompt_template (str): メタプロンプトによって生成されたプロンプトテンプレート。
        metaprompt_variables_str (str): 改行区切りの変数名文字列 (例: "VAR1\nVAR2")。
                                         metaprompt.py の修正により、変数名の中身のみが含まれます。

    Returns:
        tuple: (APEによって生成された新しいプロンプトテンプレート, 元の変数文字列)
    """
    variable_names = [var for var in metaprompt_variables_str.split("\n") if var]
    demo_data = {}
    if not variable_names:
        demo_data = {"{{DUMMY_VARIABLE}}": "dummy_value"}
        logging.warning(
            "メタプロンプトの変数が空のため、APE用にダミーデータを生成しました。"
        )
    else:
        for var_name in variable_names:
            placeholder_key_for_ape = f"{{{{{var_name}}}}}"
            demo_data[placeholder_key_for_ape] = f"dummy_{var_name.lower()}"

    try:
        result_dict = await ape(
            initial_prompt=metaprompt_template, epoch=1, demo_data=demo_data
        )
        if result_dict and isinstance(result_dict.get("prompt"), str):
            prompt = result_dict["prompt"]
            return prompt, metaprompt_variables_str
        else:
            error_info = (
                result_dict.get("error", "APE returned invalid data")
                if result_dict
                else "APE returned None"
            )
            logging.error(f"APE実行エラー: {error_info}")
            return f"エラー: {error_info}", metaprompt_variables_str
    except ValueError as e:
        error_message = str(e)
        logging.error(f"APE実行エラー: {error_message}")
        return f"エラー: {error_message}", metaprompt_variables_str


# メタプロンプト生成ボタンクリック時のラッパー関数
def metaprompt_wrapper(task: str, variables_str: str) -> Tuple[str, str, str, str]:
    """
    metapromptを実行し、APE結果表示用のテキストボックスをクリアするための値を返すラッパー。
    """
    try:
        prompt, new_vars = metaprompt(task, variables_str)
        # メタプロンプトの出力と、APE側をクリアするための空文字列を返す
        # Gradioの出力コンポーネント数に合わせて4つの値を返す
        return prompt, new_vars, "", ""
    except ValueError as e:
        error_message = str(e)
        # エラーメッセージをユーザーに表示し、他のフィールドをクリア
        return f"エラー: {error_message}", "", "", ""


# Gradioインターフェースを定義
with gr.Blocks(
    title=config.lang_store[config.language]["Automatic Prompt Engineering"],
    theme="soft",
) as demo:
    clear_button_label = config.lang_store[config.language].get("Clear", "Clear")
    gr.Markdown(
        f"# {config.lang_store[config.language]['Automatic Prompt Engineering']}"
    )

    # 「メタプロンプト」タブ
    with gr.Tab(config.lang_store[config.language]["Meta Prompt"]):
        original_task = gr.Textbox(
            label=config.lang_store[config.language]["Task"],
            lines=3,
            info=config.lang_store[config.language]["Please input your task"],
            placeholder=config.lang_store[config.language][
                "Draft an email responding to a customer complaint"
            ],
        )
        variables = gr.Textbox(
            label=config.lang_store[config.language]["Variables"],
            info=config.lang_store[config.language][
                "Please input your variables, one variable per line"
            ],
            lines=5,
            placeholder=config.lang_store[config.language][
                "CUSTOMER_COMPLAINT\nCOMPANY_NAME"
            ],
        )
        with gr.Column(scale=2):
            with gr.Row():
                metaprompt_button = gr.Button(
                    config.lang_store[config.language]["Generate Prompt"], scale=1
                )
                ape_on_metaprompt_button = gr.Button(
                    config.lang_store[config.language]["APE on MetaPrompt Output"],
                    scale=1,
                )
                clear_button_meta = gr.Button(clear_button_label, scale=1)

        with gr.Row():
            with gr.Column():
                prompt_result_meta = gr.Textbox(
                    label=config.lang_store[config.language][
                        "MetaPrompt Output: Prompt Template"
                    ],
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                )
                variables_result_meta = gr.Textbox(
                    label=config.lang_store[config.language][
                        "MetaPrompt Output: Variables"
                    ],
                    lines=5,
                    show_copy_button=True,
                    interactive=False,
                )
            with gr.Column():
                prompt_result_ape = gr.Textbox(
                    label=config.lang_store[config.language][
                        "APE Output: Prompt Template"
                    ],
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                    scale=1,
                )
                variables_result_ape = gr.Textbox(
                    label=config.lang_store[config.language]["APE Output: Variables"],
                    lines=5,
                    show_copy_button=True,
                    interactive=False,
                    scale=1,
                )

        metaprompt_button.click(
            metaprompt_wrapper,
            inputs=[original_task, variables],
            outputs=[
                prompt_result_meta,
                variables_result_meta,
                prompt_result_ape,
                variables_result_ape,
            ],
        )
        clear_button_meta.click(
            clear_metaprompt_tab,
            inputs=[],
            outputs=[
                original_task,
                variables,
                prompt_result_meta,
                variables_result_meta,
                prompt_result_ape,
                variables_result_ape,
            ],
        )

        ape_on_metaprompt_button.click(
            run_ape_on_metaprompt_output,
            inputs=[prompt_result_meta, variables_result_meta],
            outputs=[prompt_result_ape, variables_result_ape],
        )
    # 「プロンプト翻訳」タブ (実際にはプロンプトの書き換え・改善機能)
    with gr.Tab(config.lang_store[config.language]["Prompt Translation"]):
        original_prompt = gr.Textbox(
            label=config.lang_store[config.language][
                "Please input your original prompt"
            ],
            lines=3,
            placeholder=config.lang_store[config.language][
                'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
            ],
        )
        gr.Markdown(r"Use {\{xxx\}} to express custom variable, e.g. {\{document\}}")
        with gr.Row():
            with gr.Column(scale=2):
                level = gr.Radio(  # 最適化レベルを選択するラジオボタン
                    ["One-time Generation", "Multiple-time Generation"],
                    label=config.lang_store[config.language]["Optimize Level"],
                    value="One-time Generation",
                )
                with gr.Row():
                    b1 = gr.Button(
                        config.lang_store[config.language]["Generate Prompt"], scale=4
                    )
                    clear_button_translate = gr.Button(clear_button_label, scale=1)
                textboxes = []
                for i in range(3):
                    t = gr.Textbox(
                        label=config.lang_store[config.language][
                            "Prompt Template Generated"
                        ],
                        elem_id="textbox_id",
                        lines=3,
                        show_copy_button=True,
                        interactive=False,
                        visible=False if i > 0 else True,
                    )
                    textboxes.append(t)
                b1.click(
                    generate_prompt, inputs=[original_prompt, level], outputs=textboxes
                )
                clear_button_translate.click(
                    clear_translation_tab,
                    inputs=[],
                    outputs=[original_prompt] + textboxes,
                )

    # プロンプト評価タブの定義
    with gr.Tab(config.lang_store[config.language]["Prompt Evaluation"]):
        # プロンプト入力と変数置換のセクション
        with gr.Row():
            # 元のプロンプト入力欄（OpenAIモデル用）
            user_prompt_original = gr.Textbox(
                label=config.lang_store[config.language][
                    "Please input your original prompt"
                ],
                lines=3,
            )
            # 評価対象プロンプト入力欄
            user_prompt_eval = gr.Textbox(
                label=config.lang_store[config.language][
                    "Please input the prompt need to be evaluate"
                ],
                lines=3,
            )

            # 評価対象プロンプトの入力と変数置換のセクション
        with gr.Row():
            # 変数置換入力欄（元のプロンプト用）
            # 形式: "key1:value1;key2:value2"
            # user_prompt_original 内のプレースホルダ（例: {key1}）を置換するために使用
            kv_input_original = gr.Textbox(
                label=config.lang_store[config.language][
                    "[Optional]Input the template variable need to be replaced"
                ],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            # 変数置換後の元のプロンプト表示欄 (インタラクティブではない)
            user_prompt_original_replaced = gr.Textbox(
                label=config.lang_store[config.language]["Replace Result"],
                lines=3,
                interactive=False,
            )
            # 変数置換入力欄（評価対象のプロンプト用）
            # 形式: "key1:value1;key2:value2"
            # user_prompt_eval 内のプレースホルダ（例: {key1}）を置換するために使用
            kv_input_eval = gr.Textbox(
                label=config.lang_store[config.language][
                    "[Optional]Input the template variable need to be replaced"
                ],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            # 変数置換後の評価対象プロンプト表示欄 (インタラクティブではない)
            user_prompt_eval_replaced = gr.Textbox(
                label=config.lang_store[config.language]["Replace Result"],
                lines=3,
                interactive=False,
            )

        # 変数置換ボタンの定義
        with gr.Row():
            # 元のプロンプトの変数置換ボタン
            insert_button_original = gr.Button(
                config.lang_store[config.language][
                    "Replace Variables in Original Prompt"
                ]
            )
            insert_button_original.click(
                alignment.insert_kv,
                inputs=[user_prompt_original, kv_input_original],
                outputs=user_prompt_original_replaced,
            )

            # 改訂されたプロンプトの変数置換ボタン
            insert_button_revise = gr.Button(
                config.lang_store[config.language][
                    "Replace Variables in Revised Prompt"
                ]
            )
            insert_button_revise.click(
                alignment.insert_kv,
                inputs=[user_prompt_eval, kv_input_eval],
                outputs=user_prompt_eval_replaced,
            )

        # モデル選択と実行ボタンの定義
        with gr.Row():
            # OpenAIモデル選択ドロップダウン
            OpenAI_model_dropdown = gr.Dropdown(
                label=config.lang_store[config.language].get(
                    "Choose OpenAI Model", "Choose OpenAI Model"
                ),
                choices=[
                    "deepseek/deepseek-chat-v3-0324:free",
                    "deepseek/deepseek-r1-0528:free",
                ],
                value="deepseek/deepseek-chat-v3-0324:free",
            )
            # Groqモデル選択ドロップダウン
            groq_model_dropdown = gr.Dropdown(
                label=config.lang_store[config.language].get(
                    "Choose Groq Model", "Choose Groq Model"
                ),
                choices=[
                    "compound-beta-mini",
                    "compound-beta",
                ],
                value="compound-beta-mini",
            )

        with gr.Row():
            # プロンプト実行ボタン
            invoke_button = gr.Button(
                config.lang_store[config.language]["Execute prompt"], scale=4
            )
            clear_button_eval = gr.Button(clear_button_label, scale=1)

        # モデル実行結果表示エリア
        with gr.Row():
            # OpenAIモデルの出力表示テキストボックス
            OpenAI_output = gr.Textbox(
                label=config.lang_store[config.language][
                    "Original Prompt Output (OpenAI)"
                ],
                lines=3,
                interactive=True,
                show_copy_button=True,
            )
            # Groqモデルの出力表示テキストボックス
            groq_output = gr.Textbox(
                label=config.lang_store[config.language][
                    "Evaluation Prompt Output (Groq)"
                ],
                lines=3,
                interactive=False,
                show_copy_button=True,
            )
            # プロンプト実行イベント
            invoke_button.click(
                alignment.invoke_prompt,
                inputs=[
                    user_prompt_original_replaced,
                    user_prompt_eval_replaced,
                    user_prompt_original,
                    user_prompt_eval,
                    OpenAI_model_dropdown,
                    groq_model_dropdown,
                ],
                outputs=[OpenAI_output, groq_output],
            )

        # フィードバックと評価、改善プロンプト生成エリアのUI定義
        with gr.Row():
            # フィードバック入力欄（手動入力または自動生成）
            feedback_input = gr.Textbox(
                label=config.lang_store[config.language]["Evaluate the Prompt Effect"],
                placeholder=config.lang_store[config.language][
                    "Input your feedback manually or by model"
                ],
                lines=3,
                show_copy_button=True,
            )
            # 改訂されたプロンプトの出力表示テキストボックス
            revised_prompt_output = gr.Textbox(
                label=config.lang_store[config.language]["Revised Prompt"],
                lines=3,
                interactive=False,
                show_copy_button=True,
            )

            clear_button_eval.click(
                clear_evaluation_tab,
                inputs=[],
                outputs=[
                    user_prompt_original,
                    user_prompt_eval,
                    kv_input_original,
                    kv_input_eval,
                    user_prompt_original_replaced,
                    user_prompt_eval_replaced,
                    OpenAI_output,
                    groq_output,
                    feedback_input,
                    revised_prompt_output,
                ],
            )

        with gr.Row():
            evaluate_button = gr.Button(
                config.lang_store[config.language]["Auto-evaluate the Prompt Effect"]
            )
            evaluate_button.click(
                alignment.evaluate_response,
                inputs=[OpenAI_output, groq_output],
                outputs=[feedback_input],
            )
            # プロンプト改善ボタン（フィードバックに基づいてプロンプトを改訂）
            revise_button = gr.Button(
                config.lang_store[config.language]["Iterate the Prompt"]
            )
            revise_button.click(
                alignment.generate_revised_prompt,  # eval_model_id は削除
                inputs=[feedback_input, user_prompt_eval, OpenAI_output, groq_output],
                outputs=revised_prompt_output,
            )

    # 「SOE最適化商品説明」タブのUI定義
    with gr.Tab(
        config.lang_store[config.language]["SOE-Optimized Product Description"]
    ):
        with gr.Row():
            with gr.Column():
                product_category = gr.Textbox(
                    label=config.lang_store[config.language]["Product Category"],
                    placeholder=config.lang_store[config.language][
                        "Enter the product category"
                    ],
                )
                brand_name = gr.Textbox(
                    label=config.lang_store[config.language]["Brand Name"],
                    placeholder=config.lang_store[config.language][
                        "Enter the brand name"
                    ],
                )
                usage_description = gr.Textbox(
                    label=config.lang_store[config.language]["Usage Description"],
                    placeholder=config.lang_store[config.language][
                        "Enter the usage description"
                    ],
                )
                target_customer = gr.Textbox(
                    label=config.lang_store[config.language]["Target Customer"],
                    placeholder=config.lang_store[config.language][
                        "Enter the target customer"
                    ],
                )
            with gr.Column():
                # 画像アップロードとプレビュー
                image_preview = gr.Gallery(
                    label=config.lang_store[config.language]["Uploaded Images"],
                    show_label=False,
                    elem_id="image_preview",
                )
                image_upload = gr.UploadButton(
                    config.lang_store[config.language][
                        "Upload Product Image (Optional)"
                    ],
                    file_types=["image", "video"],
                    file_count="multiple",
                )
                generate_button = gr.Button(
                    config.lang_store[config.language]["Generate Product Description"]
                )

        with gr.Row():
            product_description = gr.Textbox(
                label=config.lang_store[config.language][
                    "Generated Product Description"
                ],
                lines=10,
                interactive=False,
            )
        # 商品説明生成イベントの定義
        generate_button.click(
            soeprompt.generate_description,
            inputs=[
                product_category,
                brand_name,
                usage_description,
                target_customer,
                image_upload,
            ],
            outputs=product_description,
        )
        image_upload.upload(
            lambda images: images, inputs=image_upload, outputs=image_preview
        )

    # 「プロンプトキャリブレーション」タブのUI定義
    with gr.Tab(
        config.lang_store[config.language]["Prompt Calibration"]
    ):  # 例: タブ名の変更（必要に応じて）
        default_code = """
def postprocess(llm_output):
    return llm_output
""".strip()
        with gr.Row():
            with gr.Column(scale=2):  # カラムの定義
                calibration_task = gr.Textbox(
                    label=config.lang_store[config.language]["Please input your task"],
                    lines=3,
                )
                calibration_prompt_original = gr.Textbox(
                    label=config.lang_store[config.language][
                        "Please input your original prompt"
                    ],
                    lines=5,
                    placeholder=config.lang_store[config.language][
                        'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
                    ],
                )
            with gr.Column(scale=2):  # カラムの定義
                postprocess_code = gr.Textbox(  # ポストプロセスコード入力欄
                    label=config.lang_store[config.language][
                        "Please input your postprocess code"
                    ],
                    lines=3,
                    value=default_code,
                )  # 例: ラベルの変更（より明確に）
                dataset_file = gr.File(
                    file_types=["csv"], type="binary"
                )  # データセットファイルアップロード
        with gr.Row():  # 行の定義
            calibration_task_type = gr.Radio(
                ["classification"],
                value="classification",
                label=config.lang_store[config.language]["Task type"],
            )  # タスクタイプ選択
            steps_num = gr.Slider(
                1, 5, value=1, step=1, label=config.lang_store[config.language]["Epoch"]
            )  # 最適化ステップ数
        # 例: 不要なラベルの削除（またはより具体的なラベルに変更）
        #  calibration_prompt = gr.Textbox(label=config.lang_store[config.language]["Revised Prompt"],
        #                                  lines=3, show_copy_button=True, interactive=False)
        calibration_optimization = gr.Button(
            config.lang_store[config.language]["Optimization based on prediction"]
        )
        calibration_prompt = gr.Textbox(
            label=config.lang_store[config.language]["Revised Prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
        # プロンプトキャリブレーション実行イベント
        calibration_optimization.click(
            calibration.optimize,
            inputs=[
                calibration_task,
                calibration_prompt_original,
                dataset_file,
                postprocess_code,
                steps_num,
            ],
            outputs=calibration_prompt,
        )


# シグナルハンドラ
def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    time.sleep(1)  # タスク完了のための猶予時間（必要に応じて調整）
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    demo.launch()
