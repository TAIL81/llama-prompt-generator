from typing import Tuple

import gradio as gr

from src.calibration import CalibrationPrompt

# モジュールレベルでデフォルトコードを定義し、重複を排除
DEFAULT_POSTPROCESS_CODE = """
def postprocess(llm_output):
    return llm_output
""".strip()


def create_calibration_tab(component_manager, config):
    """
    プロンプトキャリブレーションタブのUIを作成し、イベントハンドラを登録します。

    Args:
        component_manager: アプリケーションのコンポーネントを管理するオブジェクト。
        config: アプリケーションの設定オブジェクト。
    """
    with gr.Tab(config.lang_store[config.language]["Prompt Calibration"]):
        # 入力セクション
        with gr.Row():
            with gr.Column(scale=2):
                # キャリブレーションタスク入力用のテキストボックス
                calibration_task = gr.Textbox(
                    label=config.lang_store[config.language]["Please input your task"],
                    lines=3,
                )
                # 元のプロンプト入力用のテキストボックス
                calibration_prompt_original = gr.Textbox(
                    label=config.lang_store[config.language][
                        "Please input your original prompt"
                    ],
                    lines=5,
                    placeholder=config.lang_store[config.language][
                        'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
                    ],
                )
            with gr.Column(scale=2):
                # ポストプロセスコード入力用のテキストボックス
                postprocess_code = gr.Textbox(
                    label=config.lang_store[config.language][
                        "Please input your postprocess code"
                    ],
                    lines=3,
                    value=DEFAULT_POSTPROCESS_CODE,  # デフォルト値を設定
                )
                # データセットファイルアップロード用のコンポーネント
                dataset_file = gr.File(file_types=["csv"], type="binary")

        # タスクタイプとエポック数選択セクション
        with gr.Row():
            # タスクタイプ選択用のラジオボタン
            calibration_task_type = gr.Radio(
                ["classification"],
                value="classification",
                label=config.lang_store[config.language]["Task type"],
            )
            # エポック数選択用のスライダー
            steps_num = gr.Slider(
                1, 5, value=1, step=1, label=config.lang_store[config.language]["Epoch"]
            )

        # ボタンセクション
        with gr.Row():
            # 最適化実行ボタン
            calibration_optimization = gr.Button(
                config.lang_store[config.language]["Optimization based on prediction"],
                scale=4,
            )
            # クリアボタン
            clear_button_calibration = gr.Button(
                config.lang_store[config.language].get("Clear", "Clear"), scale=1
            )

        # 改訂されたプロンプト表示用のテキストボックス
        calibration_prompt = gr.Textbox(
            label=config.lang_store[config.language]["Revised Prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )

        # イベントハンドラの登録
        # 最適化ボタンがクリックされたときの処理
        calibration_optimization.click(
            component_manager.get(CalibrationPrompt).optimize,
            inputs=[
                calibration_task,
                calibration_prompt_original,
                dataset_file,
                postprocess_code,
                steps_num,
            ],
            outputs=calibration_prompt,
        )

        # クリアボタンがクリックされたときの処理
        clear_button_calibration.click(
            clear_calibration_tab,
            inputs=[],
            outputs=[
                calibration_task,
                calibration_prompt_original,
                postprocess_code,
                dataset_file,
                steps_num,
                calibration_prompt,
            ],
        )


def clear_calibration_tab() -> Tuple[str, str, str, None, int, str]:
    """
    プロンプトキャリブレーションタブのすべての入力フィールドと出力フィールドをクリアします。

    Returns:
        Tuple[str, str, str, None, int, str]: クリアされたフィールドの空文字列タプル、
                                             デフォルトのポストプロセスコード、None（ファイル用）、
                                             デフォルトのエポック数、空文字列（プロンプト用）。
    """
    return "", "", DEFAULT_POSTPROCESS_CODE, None, 1, ""
