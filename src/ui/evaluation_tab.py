from typing import Any, Dict, Tuple

import gradio as gr

from src.optimize import Alignment
from src.ui.utils import notify_error

# モデルの選択肢を定数として定義
OPENAI_MODEL_CHOICES = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1-0528:free",
]
GROQ_MODEL_CHOICES = [
    "compound-beta-mini",
    "compound-beta-kimi",
    "compound-beta",
]


def clear_evaluation_tab() -> Tuple[str, str, str, str, str, str, str, str, str, str]:
    """
    プロンプト評価タブのすべての入力フィールドと出力フィールドをクリアします。

    Returns:
        Tuple[str, ...]: クリアされたフィールドの空文字列タプル。
    """
    return "", "", "", "", "", "", "", "", "", ""


def create_evaluation_tab(component_manager: Any, config: Any):
    """
    プロンプト評価タブのUIを作成し、イベントハンドラを登録します。

    Args:
        component_manager: アプリケーションのコンポーネントを管理するオブジェクト。
        config: アプリケーションの設定オブジェクト。
    """
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
            kv_input_eval = gr.Textbox(
                label=config.lang_store[config.language][
                    "[Optional]Input the template variable need to be replaced"
                ],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            # 変数置換後の評価対象プロンプ���表示欄 (インタラクティブではない)
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
            # 元のプロンプトの変数置換ボタンクリックイベント
            insert_button_original.click(
                component_manager.get(Alignment).insert_kv,
                inputs=[user_prompt_original, kv_input_original],
                outputs=user_prompt_original_replaced,
            )

            # 改訂されたプロンプトの変数置換ボタン
            insert_button_revise = gr.Button(
                config.lang_store[config.language][
                    "Replace Variables in Revised Prompt"
                ]
            )
            # ��訂されたプロンプトの変数置換ボタンクリックイベント
            insert_button_revise.click(
                component_manager.get(Alignment).insert_kv,
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
                choices=OPENAI_MODEL_CHOICES,
                value=OPENAI_MODEL_CHOICES[0],
            )
            # Groqモデル選択ドロップダウン
            groq_model_dropdown = gr.Dropdown(
                label=config.lang_store[config.language].get(
                    "Choose Groq Model", "Choose Groq Model"
                ),
                choices=GROQ_MODEL_CHOICES,
                value=GROQ_MODEL_CHOICES[0],
            )

        with gr.Row():
            # プロンプト実行ボタン
            invoke_button = gr.Button(
                config.lang_store[config.language]["Execute prompt"], scale=4
            )
            # クリアボタン
            clear_button_eval = gr.Button(
                config.lang_store[config.language].get("Clear", "Clear"), scale=1
            )

        # モデル実行結果表示エリア
        with gr.Row():
            # OpenAIモデルの出力表示テキストボックス
            OpenAI_output = gr.Textbox(
                label=config.lang_store[config.language][
                    "Original Prompt Output (OpenAI)"
                ],
                lines=16,  # 標準化
                interactive=False,  # 標準化
                show_copy_button=True,
            )
            # Groqモデルの出力表示テキストボックス
            groq_output = gr.Textbox(
                label=config.lang_store[config.language][
                    "Evaluation Prompt Output (Groq)"
                ],
                lines=16,  # 標準化
                interactive=False,  # 標準化
                show_copy_button=True,
            )

            # 入力検証付きのラッパー
            def _invoke_wrapper(
                uo_r: str,
                ue_r: str,
                uo: str,
                ue: str,
                openai_model: str,
                groq_model: str,
            ):
                if not (uo_r and uo_r.strip()):
                    notify_error("元のプロンプトの置換結果が空です。変数置換を実行してください。")
                    return "", ""
                if not (ue_r and ue_r.strip()):
                    notify_error("評価対象プロンプトの置換結果が空です。変数置換を実行してください。")
                    return "", ""
                try:
                    return component_manager.get(Alignment).invoke_prompt(
                        uo_r, ue_r, uo, ue, openai_model, groq_model
                    )
                except Exception as e:
                    notify_error(f"実行中にエラーが発生しました: {e}")
                    return "", ""

            # プロンプト実行イベント
            invoke_button.click(
                _invoke_wrapper,
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
                lines=16,  # 標準化
                show_copy_button=True,
            )
            # 改訂されたプロンプトの出力表示テキストボックス
            revised_prompt_output = gr.Textbox(
                label=config.lang_store[config.language]["Revised Prompt"],
                lines=16,  # 標準化
                interactive=False,
                show_copy_button=True,
            )

            # クリアボタンクリックイベント
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
            # 自動評価ボタン
            evaluate_button = gr.Button(
                config.lang_store[config.language]["Auto-evaluate the Prompt Effect"]
            )
            # 自動評価ボタンクリックイベント
            evaluate_button.click(
                component_manager.get(Alignment).evaluate_response,
                inputs=[OpenAI_output, groq_output],
                outputs=[feedback_input],
            )
            # プロンプト改善ボタン（フィードバックに基づいてプロンプトを改訂）
            revise_button = gr.Button(
                config.lang_store[config.language]["Iterate the Prompt"]
            )
            # プロンプト改善ボタンクリックイベント
            revise_button.click(
                component_manager.get(
                    Alignment
                ).generate_revised_prompt,  # eval_model_id は削除
                inputs=[feedback_input, user_prompt_eval, OpenAI_output, groq_output],
                outputs=revised_prompt_output,
            )
