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
from ui.metaprompt_tab import create_metaprompt_tab
from ui.translation_tab import create_translation_tab
from translate import GuideBased


# 設定管理クラスの導入
class AppConfig(BaseSettings):
    """アプリケーション設定を管理するクラス。

    設定は環境変数と翻訳ファイルから読み込まれます。
    """

    language: str = Field(default="ja", validation_alias="LANGUAGE")
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
            logging.error(
                f"翻訳ファイルの形式が不正です: {self.TRANSLATIONS_PATH}. エラー: {e}"
            )
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





# Gradioインターフェースを定義
with gr.Blocks(
    title=config.lang_store[config.language]["Automatic Prompt Engineering"],
    theme="soft",
) as demo:
    clear_button_label = config.lang_store[config.language].get("Clear", "Clear")
    gr.Markdown(
        f"# {config.lang_store[config.language]['Automatic Prompt Engineering']}"
    )

    
    create_metaprompt_tab(component_manager, config)
    create_translation_tab(component_manager, config)
    

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
