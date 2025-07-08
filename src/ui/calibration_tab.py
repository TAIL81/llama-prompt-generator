import gradio as gr
from typing import Tuple

# モジュールレベルでデフォルトコードを定義し、重複を排除
DEFAULT_POSTPROCESS_CODE = """
def postprocess(llm_output):
    return llm_output
""".strip()

def create_calibration_tab(component_manager, config):
    """プロンプトキャリブレーションタブのUIを作成し、イベントハンドラを登録します。"""
    with gr.Tab(
        config.lang_store[config.language]["Prompt Calibration"]
    ):
        with gr.Row():
            with gr.Column(scale=2):
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
            with gr.Column(scale=2):
                postprocess_code = gr.Textbox(
                    label=config.lang_store[config.language][
                        "Please input your postprocess code"
                    ],
                    lines=3,
                    value=DEFAULT_POSTPROCESS_CODE,
                )
                dataset_file = gr.File(
                    file_types=["csv"], type="binary"
                )
        with gr.Row():
            calibration_task_type = gr.Radio(
                ["classification"],
                value="classification",
                label=config.lang_store[config.language]["Task type"],
            )
            steps_num = gr.Slider(
                1, 5, value=1, step=1, label=config.lang_store[config.language]["Epoch"]
            )
        
        with gr.Row():
            calibration_optimization = gr.Button(
                config.lang_store[config.language]["Optimization based on prediction"],
                scale=4
            )
            clear_button_calibration = gr.Button(
                config.lang_store[config.language].get("Clear", "Clear"),
                scale=1
            )

        calibration_prompt = gr.Textbox(
            label=config.lang_store[config.language]["Revised Prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
        
        calibration_optimization.click(
            component_manager.calibration.optimize,
            inputs=[
                calibration_task,
                calibration_prompt_original,
                dataset_file,
                postprocess_code,
                steps_num,
            ],
            outputs=calibration_prompt,
        )
        
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
    """プロンプトキャリブレーションタブの入出力をクリアします。"""
    return "", "", DEFAULT_POSTPROCESS_CODE, None, 1, ""
