
import gradio as gr
from typing import List, Tuple, cast
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# 最適化レベルをEnumで定義し、マジックストリングを排除
class OptimizeLevel(Enum):
    """
    プロンプト生成の最適化レベルを定義するEnum。
    - SINGLE: 一回生成モード
    - MULTIPLE: 複数回生成モード
    """
    SINGLE = "One-time Generation"
    MULTIPLE = "Multiple-time Generation"

def create_translation_tab(component_manager, config):
    """
    プロンプト翻訳タブのUIを作成し、イベントハンドラを登録します。

    Args:
        component_manager: アプリケーションのコンポーネントを管理するオブジェクト。
        config: アプリケーションの設定オブジェクト。
    """
    # 設定とコンポーネントを初期化
    lang_store = config.lang_store
    language = config.language
    rewrite = component_manager.rewrite

    def clear_translation_tab() -> Tuple[str, gr.Textbox, gr.Textbox, gr.Textbox]:
        """
        プロンプト翻訳タブのすべての入力フィールドと出力フィールドをクリアします。

        Returns:
            Tuple[str, gr.Textbox, gr.Textbox, gr.Textbox]: クリアされたフィールドの空文字列タプルと、
                                                           表示状態をリセットしたGradio Textboxコンポーネント。
        """
        # 最初のテキストボックスのみ表示し、他は非表示にリセット
        return (
            "",
            gr.Textbox(value="", visible=True),
            gr.Textbox(value="", visible=False),
            gr.Textbox(value="", visible=False),
        )

    def create_single_textbox(value: str) -> List[gr.Textbox]:
        """
        一回生成モード用の単一のテキストボックスコンポーネントを作成します。

        Args:
            value (str): テキストボックスに表示する値。

        Returns:
            List[gr.components.Textbox]: 単一の表示テキストボックスと、残りの非表示テキストボックスのリスト。
        """
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
        ] * 2 # 他のタブの出力コンポーネント数と合わせるため、3つの出力のうち2つは非表示にする

    def create_multiple_textboxes(
        candidates: List[str], judge_result: int
    ) -> List[gr.Textbox]:
        """
        複数回生成モード用の複数のテキストボックスコンポーネントを作成します。

        Args:
            candidates (List[str]): 生成されたプロンプト候補のリスト。
            judge_result (int): 最も良いと判断されたプロンプトのインデックス。

        Returns:
            List[gr.components.Textbox]: 複数の表示テキストボックスのリスト。
        """
        textboxes = []
        for i in range(3):
            # 最も良いプロンプトには 'Y'、それ以外には 'N' を表示
            is_best = "Y" if judge_result == i else "N"
            textboxes.append(
                cast(
                    gr.Textbox,
                    gr.Textbox(
                        label=f'{lang_store[language]["Prompt Template Generated"]} #{i+1} {is_best}',
                        value=candidates[i],
                        lines=3,
                        show_copy_button=True,
                        visible=True, # 複数回生成モードではすべて表示
                        interactive=False,
                    ),
                )
            )
        return textboxes

    def generate_single_prompt(original_prompt: str) -> List[gr.Textbox]:
        """
        一回生成モードでプロンプトを生成し、結果をテキストボックスに表示します。

        Args:
            original_prompt (str): 元のプロンプト。

        Returns:
            List[gr.components.Textbox]: 生成されたプロンプトを含むテキストボックスのリスト。
        """
        result = rewrite(original_prompt) # rewriteコンポーネントを使用してプロンプトを生成
        return create_single_textbox(result)

    def generate_multiple_prompts_async(original_prompt: str) -> List[str]:
        """
        複数回生成モードでプロンプトを非同期に生成します。

        Args:
            original_prompt (str): 元のプロンプト。

        Returns:
            List[str]: 生成されたプロンプト候補のリスト。
        """
        # ThreadPoolExecutor を使用して3つのプロンプトを並行して生成
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(rewrite, original_prompt) for _ in range(3)]
            candidates = [future.result() for future in futures]
        return candidates

    def generate_prompt(original_prompt: str, level: str) -> List[gr.Textbox]:
        """
        プロンプト生成のメイン関数。選択された最適化レベルに基づいてプロンプトを生成します。

        Args:
            original_prompt (str): 元のプロンプト。
            level (str): 最適化レベル（"One-time Generation" または "Multiple-time Generation"）。

        Returns:
            List[gr.components.Textbox]: 生成されたプロンプトを含むテキストボックスのリスト。
        """
        if level == OptimizeLevel.SINGLE.value:
            # 一回生成モードの場合
            return generate_single_prompt(original_prompt)
        elif level == OptimizeLevel.MULTIPLE.value:
            # 複数回生成モードの場合
            candidates = generate_multiple_prompts_async(original_prompt) # 複数の候補を非同期で生成
            judge_result = rewrite.judge(candidates) # 最も良い候補を判断
            return create_multiple_textboxes(candidates, judge_result)
        return [] # どのレベルにも一致しない場合は空リストを返す

    # Gradio UIの定義
    with gr.Tab(lang_store[language]["Prompt Translation"]):
        # 元のプロンプト入力用のテキストボックス
        original_prompt = gr.Textbox(
            label=lang_store[language]["Please input your original prompt"],
            lines=3,
            placeholder=lang_store[language][
                'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
            ],
        )
        # カスタム変数の使用方法を説明するMarkdown
        gr.Markdown(r"Use {\{xxx\}} to express custom variable, e.g. {\{document\}}")
        with gr.Row():
            with gr.Column(scale=2):
                # 最適化レベル選択用のラジオボタン
                level = gr.Radio(
                    choices=[level.value for level in OptimizeLevel],
                    label=lang_store[language]["Optimize Level"],
                    value=OptimizeLevel.SINGLE.value, # デフォルトは一回生成
                )
                with gr.Row():
                    # プロンプト生成ボタン
                    b1 = gr.Button(
                        lang_store[language]["Generate Prompt"], scale=4
                    )
                    # クリアボタン
                    clear_button_translate = gr.Button(config.lang_store[config.language].get("Clear", "Clear"), scale=1)
                
                # 生成されたプロンプトを表示するためのテキストボックスのリスト
                textboxes = []
                for i in range(3):
                    t = gr.Textbox(
                        label=lang_store[language]["Prompt Template Generated"],
                        elem_id=f"textbox_id_{i}", # ユニークなIDを付与
                        lines=3,
                        show_copy_button=True,
                        interactive=False,
                        visible=False if i > 0 else True, # 最初のテキストボックスのみデフォルトで表示
                    )
                    textboxes.append(t)
                
                # イベントハンドラの登録
                # プロンプト生成ボタンがクリックされたときの処理
                b1.click(
                    generate_prompt, inputs=[original_prompt, level], outputs=textboxes
                )
                # クリアボタンがクリックされたときの処理
                clear_button_translate.click(
                    clear_translation_tab,
                    inputs=[],
                    outputs=[original_prompt] + textboxes,
                )
