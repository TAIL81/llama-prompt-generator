
import gradio as gr
import logging
from typing import Tuple

def create_metaprompt_tab(component_manager, config):
    lang_store = config.lang_store
    language = config.language
    metaprompt = component_manager.metaprompt
    ape = component_manager.ape

    def clear_metaprompt_tab() -> Tuple[str, str, str, str, str, str]:
        """メタプロンプトタブの入出力をクリアします。"""
        return "", "", "", "", "", ""

    async def run_ape_on_metaprompt_output(
        metaprompt_template: str, metaprompt_variables_str: str
    ) -> Tuple[str, str]:
        """
        メタプロンプトで生成されたテンプレートと変数文字列を元に APE を実行します。
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
                # APEは `{{variable}}` 形式のプレースホルダを期待するため、
                # メタプロンプトの変数名をAPEが認識できる形式に変換します。
                # f-string の仕様上、`{` は `{{`、`}` は `}}` とエスケープする必要があるため、
                # `{{{{var_name}}}}` のように複雑な記述になっています。
                # 例: `CUSTOMER_NAME` -> `{{CUSTOMER_NAME}}`
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

    def metaprompt_wrapper(task: str, variables_str: str) -> Tuple[str, str, str, str]:
        """
        metapromptを実行し、APE結果表示用のテキストボックスをクリアするための値を返すラッパー。
        """
        try:
            prompt, new_vars = metaprompt(task, variables_str)
            return prompt, new_vars, "", ""
        except ValueError as e:
            error_message = str(e)
            return f"エラー: {error_message}", "", "", ""

    with gr.Tab(lang_store[language]["Meta Prompt"]):
        clear_button_label = lang_store[language].get("Clear", "Clear")
        original_task = gr.Textbox(
            label=lang_store[language]["Task"],
            lines=3,
            info=lang_store[language]["Please input your task"],
            placeholder=lang_store[language][
                "Draft an email responding to a customer complaint"
            ],
        )
        variables = gr.Textbox(
            label=lang_store[language]["Variables"],
            info=lang_store[language][
                "Please input your variables, one variable per line"
            ],
            lines=5,
            placeholder=lang_store[language][
                "CUSTOMER_COMPLAINT\nCOMPANY_NAME"
            ],
        )
        with gr.Column(scale=2):
            with gr.Row():
                metaprompt_button = gr.Button(
                    lang_store[language]["Generate Prompt"], scale=1
                )
                ape_on_metaprompt_button = gr.Button(
                    lang_store[language]["APE on MetaPrompt Output"],
                    scale=1,
                )
                clear_button_meta = gr.Button(clear_button_label, scale=1)

        with gr.Row():
            with gr.Column():
                prompt_result_meta = gr.Textbox(
                    label=lang_store[language][
                        "MetaPrompt Output: Prompt Template"
                    ],
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                )
                variables_result_meta = gr.Textbox(
                    label=lang_store[language][
                        "MetaPrompt Output: Variables"
                    ],
                    lines=5,
                    show_copy_button=True,
                    interactive=False,
                )
            with gr.Column():
                prompt_result_ape = gr.Textbox(
                    label=lang_store[language][
                        "APE Output: Prompt Template"
                    ],
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                    scale=1,
                )
                variables_result_ape = gr.Textbox(
                    label=lang_store[language]["APE Output: Variables"],
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
