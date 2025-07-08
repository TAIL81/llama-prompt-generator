
import gradio as gr
from typing import List, Tuple, cast
from concurrent.futures import ThreadPoolExecutor

# This function will be called from app.py, so it needs access to the component_manager and config
def create_translation_tab(component_manager, config):
    lang_store = config.lang_store
    language = config.language
    rewrite = component_manager.rewrite

    def clear_translation_tab() -> Tuple[str, gr.Textbox, gr.Textbox, gr.Textbox]:
        """プロンプト翻訳タブの入出力をクリアします。"""
        # 最初のテキストボックスのみ表示し、他は非表示にリセット
        return (
            "",
            gr.Textbox(value="", visible=True),
            gr.Textbox(value="", visible=False),
            gr.Textbox(value="", visible=False),
        )

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
            )
        ] + [
            gr.Textbox(visible=False)
        ] * 2

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
                        label=f'{lang_store[language]["Prompt Template Generated"]} #{i+1} {is_best}',
                        value=candidates[i],
                        lines=3,
                        show_copy_button=True,
                        visible=True,
                        interactive=False,
                    ),
                )
            )
        return textboxes

    def generate_single_prompt(original_prompt: str) -> List[gr.Textbox]:
        """一回生成モードでプロンプトを生成し、結果をテキストボックスに表示します。"""
        result = rewrite(original_prompt)
        return create_single_textbox(result)

    def generate_multiple_prompts_async(original_prompt: str) -> List[str]:
        """複数回生成モードでプロンプトを非同期に生成します。"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(rewrite, original_prompt) for _ in range(3)]
            candidates = [future.result() for future in futures]
        return candidates

    def generate_prompt(original_prompt: str, level: str) -> List[gr.Textbox]:
        """プロンプト生成のメイン関数。"""
        if level == "One-time Generation":
            return generate_single_prompt(original_prompt)
        elif level == "Multiple-time Generation":
            candidates = generate_multiple_prompts_async(original_prompt)
            judge_result = rewrite.judge(candidates)
            return create_multiple_textboxes(candidates, judge_result)
        return []

    with gr.Tab(lang_store[language]["Prompt Translation"]):
        original_prompt = gr.Textbox(
            label=lang_store[language]["Please input your original prompt"],
            lines=3,
            placeholder=lang_store[language][
                'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
            ],
        )
        gr.Markdown(r"Use {\{xxx\}} to express custom variable, e.g. {\{document\}}")
        with gr.Row():
            with gr.Column(scale=2):
                level = gr.Radio(
                    ["One-time Generation", "Multiple-time Generation"],
                    label=lang_store[language]["Optimize Level"],
                    value="One-time Generation",
                )
                with gr.Row():
                    b1 = gr.Button(
                        lang_store[language]["Generate Prompt"], scale=4
                    )
                    clear_button_translate = gr.Button(config.lang_store[config.language].get("Clear", "Clear"), scale=1)
                textboxes = []
                for i in range(3):
                    t = gr.Textbox(
                        label=lang_store[language]["Prompt Template Generated"],
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
