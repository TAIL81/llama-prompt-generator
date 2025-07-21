import logging
from typing import Tuple

import gradio as gr

from src.ape import APE
from src.metaprompt import MetaPrompt


def create_metaprompt_tab(component_manager, config):
    """
    メタプロンプトタブのUIを作成し、イベントハンドラを登録します。

    Args:
        component_manager: アプリケーションのコンポーネントを管理するオブジェクト。
        config: アプリケーションの設定オブジェクト。
    """
    # 設定とコンポーネントを初期化
    lang_store = config.lang_store
    language = config.language
    metaprompt = component_manager.get(MetaPrompt)
    ape = component_manager.get(APE)

    def clear_metaprompt_tab() -> Tuple[str, str, str, str, str, str]:
        """
        メタプロンプトタブのすべての入力フィールドと出力フィールドをクリアします。

        Returns:
            Tuple[str, str, str, str, str, str]: クリアされたフィールドの空文字列タプル。
        """
        return "", "", "", "", "", ""

    def run_ape_on_metaprompt_output(
        metaprompt_template: str, metaprompt_variables_str: str
    ) -> Tuple[str, str]:
        """
        メタプロンプトで生成されたテンプレートと変数文字列を元に APE (Automatic Prompt Engineering) を実行します。

        Args:
            metaprompt_template (str): メタプロンプトによって生成されたプロンプトテンプレート。
            metaprompt_variables_str (str): メタプロンプトによって生成された変数名の文字列（改行区切り）。

        Returns:
            Tuple[str, str]: APEによって生成されたプロンプトと、元の変数文字列。
                             エラーが発生した場合はエラーメッセージを返します。
        """
        # 変数文字列を解析し、変数名のリストを作成
        variable_names = [var for var in metaprompt_variables_str.split("\n") if var]
        demo_data = {}

        # 変数名が空の場合、APE用にダミーデータを生成
        if not variable_names:
            demo_data = {"{{DUMMY_VARIABLE}}": "dummy_value"}
            logging.warning(
                "メタプロンプトの変数が空のため、APE用にダミーデータを生成しました。"
            )
        else:
            # 各変数名に対してAPEが期待する形式のプレースホルダとダミー値を生成
            for var_name in variable_names:
                # APEは `{{variable}}` 形式のプレースホルダを期待するため、
                # メタプロンプトの変数名をAPEが認識できる形式に変換します。
                # f-string の仕様上、`{` は `{{`、`}` は `}}` とエスケープする必要があるため、
                # `{{{{var_name}}}}` のように複雑な記述になっています。
                # 例: `CUSTOMER_NAME` -> `{{CUSTOMER_NAME}}`
                placeholder_key_for_ape = f"{{{{{var_name}}}}}"
                demo_data[placeholder_key_for_ape] = f"dummy_{var_name.lower()}"

        try:
            # APEを実行
            result_dict = ape(
                initial_prompt=metaprompt_template, epoch=1, demo_data=demo_data
            )
            # APEの結果が有効なプロンプトを含んでいるか確認
            if result_dict and isinstance(result_dict.get("prompt"), str):
                prompt = result_dict["prompt"]
                return prompt, metaprompt_variables_str
            else:
                # APEが有効なデータを返さなかった場合のエラー処理
                error_info = (
                    result_dict.get("error", "APE returned invalid data")
                    if result_dict
                    else "APE returned None"
                )
                logging.error(f"APE実行エラー: {error_info}")
                return f"エラー: {error_info}", metaprompt_variables_str
        except ValueError as e:
            # ValueErrorが発生した場合のエラー処理
            error_message = str(e)
            logging.error(f"APE実行エラー: {error_message}")
            return f"エラー: {error_message}", metaprompt_variables_str

    def metaprompt_wrapper(task: str, variables_str: str) -> Tuple[str, str, str, str]:
        """
        metapromptを実行し、APE結果表示用のテキストボックスをクリアするための値を返すラッパー関数。

        Args:
            task (str): ユーザーが入力した元のタスク。
            variables_str (str): ユーザーが入力した変数名の文字列（改行区切り）。

        Returns:
            Tuple[str, str, str, str]: メタプロンプトによって生成されたプロンプトと変数、
                                       およびAPE結果表示用のクリア値。
                                       エラーが発生した場合はエラーメッセージを返します。
        """
        try:
            # メタプロンプトを実行
            prompt, new_vars = metaprompt(task, variables_str)
            # メタプロンプトの結果と、APE結果表示用のテキストボックスをクリアするための空文字列を返す
            return prompt, new_vars, "", ""
        except ValueError as e:
            # ValueErrorが発生した場合のエラー処理
            error_message = str(e)
            return f"エラー: {error_message}", "", "", ""

    # Gradio UIの定義
    with gr.Tab(lang_store[language]["Meta Prompt"]):
        # クリアボタンのラベルを取得
        clear_button_label = lang_store[language].get("Clear", "Clear")

        # 元のタスク入力用のテキストボックス
        original_task = gr.Textbox(
            label=lang_store[language]["Task"],
            lines=3,
            info=lang_store[language]["Please input your task"],
            placeholder=lang_store[language][
                "Draft an email responding to a customer complaint"
            ],
        )
        # 変数入力用のテキストボックス
        variables = gr.Textbox(
            label=lang_store[language]["Variables"],
            info=lang_store[language][
                "Please input your variables, one variable per line"
            ],
            lines=5,
            placeholder=lang_store[language]["CUSTOMER_COMPLAINT\nCOMPANY_NAME"],
        )
        # ボタンを配置するためのカラムと行
        with gr.Column(scale=2):
            with gr.Row():
                # プロンプト生成ボタン
                metaprompt_button = gr.Button(
                    lang_store[language]["Generate Prompt"], scale=1
                )
                # メタプロンプト出力に対してAPEを実行するボタン
                ape_on_metaprompt_button = gr.Button(
                    lang_store[language]["APE on MetaPrompt Output"],
                    scale=1,
                )
                # クリアボタン
                clear_button_meta = gr.Button(clear_button_label, scale=1)

        # 結果表示用のテキストボックスを配置するための行
        with gr.Row():
            with gr.Column():
                # メタプロンプト出力：プロンプトテンプレート
                prompt_result_meta = gr.Textbox(
                    label=lang_store[language]["MetaPrompt Output: Prompt Template"],
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                )
                # メタプロンプト出力：変数
                variables_result_meta = gr.Textbox(
                    label=lang_store[language]["MetaPrompt Output: Variables"],
                    lines=5,
                    show_copy_button=True,
                    interactive=False,
                )
            with gr.Column():
                # APE出力：プロンプトテンプレート
                prompt_result_ape = gr.Textbox(
                    label=lang_store[language]["APE Output: Prompt Template"],
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                    scale=1,
                )
                # APE出力：変数
                variables_result_ape = gr.Textbox(
                    label=lang_store[language]["APE Output: Variables"],
                    lines=5,
                    show_copy_button=True,
                    interactive=False,
                    scale=1,
                )

        # イベントハンドラの登録
        # メタプロンプト生成ボタンがクリックされたときの処理
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
        # クリアボタンがクリックされたときの処理
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

        # メタプロンプト出力に対してAPEを実行するボタンがクリックされたときの処理
        ape_on_metaprompt_button.click(
            run_ape_on_metaprompt_output,
            inputs=[prompt_result_meta, variables_result_meta],
            outputs=[prompt_result_ape, variables_result_ape],
        )
