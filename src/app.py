import json
import os
import threading

import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
from ape import APE
from calibration import CalibrationPrompt
from metaprompt import MetaPrompt
from optimize import Alignment
from translate import GuideBased
from application.soe_prompt import SOEPrompt

# 各コンポーネントを初期化します
# 環境変数を読み込みます
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
language = os.getenv("LANGUAGE", "ja")

# JSONファイルから翻訳を読み込みます
translations_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'translations.json')
with open(translations_path, 'r', encoding='utf-8') as f:
    lang_store = json.load(f)

ape = APE()
rewrite = GuideBased()
alignment = Alignment(lang_store=lang_store, language=language) # lang_storeとlanguageを渡す
metaprompt = MetaPrompt()
soeprompt = SOEPrompt()
calibration = CalibrationPrompt()

# JSONファイルから翻訳を読み込みます
translations_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'translations.json')
with open(translations_path, 'r', encoding='utf-8') as f:
    lang_store = json.load(f)

def generate_prompt(original_prompt, level):
    """
    元のプロンプトと最適化レベルに基づいてプロンプトを生成します。

    Args:
        original_prompt (str): 元のプロンプト。
        level (str): 最適化レベル ("One-time Generation" または "Multiple-time Generation")。

    Returns:
        list: Gradioテキストボックスコンポーネントのリスト。
    """
    if level == "One-time Generation":
        result = rewrite(original_prompt)
        return [
            gr.Textbox(
                label=lang_store[language]["Prompt Template Generated"],
                value=result,
                lines=3,
                show_copy_button=True,
                interactive=False,
            )
        ] + [gr.Textbox(visible=False)] * 2 # 複数回生成用の非表示テキストボックス
    elif level == "Multiple-time Generation":
        candidates = []
        for i in range(3):
            result = rewrite(original_prompt)
            candidates.append(result)
        judge_result = rewrite.judge(candidates)
        textboxes = []
        for i in range(3):
            is_best = "Y" if judge_result == i else "N"
            textboxes.append( # 生成された各プロンプト候補を表示するテキストボックス
                gr.Textbox(
                    label=f"{lang_store[language]['Prompt Template Generated']} #{i+1} {is_best}",
                    value=candidates[i],
                    lines=3,
                    show_copy_button=True,
                    visible=True,
                    interactive=False,
                )
            )
        return textboxes

def ape_prompt(original_prompt, user_data):
    print("DEBUG: ape_prompt function was called!") # デバッグメッセージを追加
    print(f"DEBUG: original_prompt = {original_prompt}")
    print(f"DEBUG: user_data = {user_data}")
    """
    APE (Automatic Prompt Engineering) を使用してプロンプトを生成します。

    Args:
        original_prompt (str): 元のプロンプト。
        user_data (str): JSON形式のユーザーデータ。

    Returns:
        list: Gradioテキストボックスコンポーネントのリスト。
    """
    result = ape(original_prompt, 1, json.loads(user_data))
    return [
        gr.Textbox(
            label="Prompt Generated",
            value=result["prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
    ] + [gr.Textbox(visible=False)] * 2 # 他のタブとの互換性のための非表示テキストボックス

# Gradioインターフェースを定義します
with gr.Blocks(title=lang_store[language]["Automatic Prompt Engineering"], theme="soft") as demo:
    gr.Markdown(f"# {lang_store[language]['Automatic Prompt Engineering']}")

    # 「メタプロンプト」タブ
    with gr.Tab(lang_store[language]["Meta Prompt"]):
        original_task = gr.Textbox(
            # タスク入力用のテキストボックス            
            label=lang_store[language]["Task"],
            lines=3,
            info=lang_store[language]["Please input your task"],
            placeholder=lang_store[language]["Draft an email responding to a customer complaint"],
        )
        variables = gr.Textbox(
            label=lang_store[language]["Variables"],
            # 変数入力用のテキストボックス（1行に1変数）
            info=lang_store[language]["Please input your variables, one variable per line"],
            lines=5,
            placeholder=lang_store[language]["CUSTOMER_COMPLAINT\nCOMPANY_NAME"],
        )
        metaprompt_button = gr.Button(lang_store[language]["Generate Prompt"])
        prompt_result = gr.Textbox(
            label=lang_store[language]["Prompt Template Generated"],
            # 生成されたプロンプトテンプレート表示用
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
        variables_result = gr.Textbox(
            label=lang_store[language]["Variables Generated"],
            lines=3,
            # 生成された変数表示用
            show_copy_button=True,
            interactive=False,
        )
        metaprompt_button.click(
            metaprompt,
            inputs=[original_task, variables],
            outputs=[prompt_result, variables_result],
        )
    # 「プロンプト翻訳」タブ (実際にはプロンプトの書き換え・改善機能)
    with gr.Tab(lang_store[language]["Prompt Translation"]):
        original_prompt = gr.Textbox(
            label=lang_store[language]["Please input your original prompt"],
            # 元のプロンプト入力用
            lines=3,
            placeholder=lang_store[language]["Summarize the text delimited by triple quotes.\n\n\"\"\"{{insert text here}}\"\"\""],
        )
        gr.Markdown("Use {\{xxx\}} to express custom variable, e.g. {\{document\}}")
        with gr.Row():
            with gr.Column(scale=2):
                level = gr.Radio(
                    ["One-time Generation", "Multiple-time Generation"],
                    # 最適化レベル選択用ラジオボタン
                    label=lang_store[language]["Optimize Level"],
                    value="One-time Generation",
                )
                b1 = gr.Button(lang_store[language]["Generate Prompt"])
                textboxes = []
                for i in range(3):
                    t = gr.Textbox(
                        # 生成されたプロンプト表示用 (最大3つ)
                        label=lang_store[language]["Prompt Template Generated"],
                        elem_id="textbox_id",
                        lines=3,
                        show_copy_button=True,
                        interactive=False,
                        visible=False if i > 0 else True,
                    )
                    textboxes.append(t)
                b1.click(generate_prompt, inputs=[original_prompt, level], outputs=textboxes)

    with gr.Tab(lang_store[language]["Prompt Evaluation"]):
        with gr.Row():
            user_prompt_original = gr.Textbox(
                label=lang_store[language]["Please input your original prompt"], lines=3
            )
            kv_input_original = gr.Textbox(
                label=lang_store[language]["[Optional]Input the template variable need to be replaced"],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            user_prompt_original_replaced = gr.Textbox(
                label=lang_store[language]["Replace Result"], lines=3, interactive=False
            )
            user_prompt_eval = gr.Textbox(
                label=lang_store[language]["Please input the prompt need to be evaluate"], lines=3
            )
            kv_input_eval = gr.Textbox(
                label=lang_store[language]["[Optional]Input the template variable need to be replaced"],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            user_prompt_eval_replaced = gr.Textbox(
                label=lang_store[language]["Replace Result"], lines=3, interactive=False
            )

        # 変数置換ボタンの定義
        with gr.Row():
            insert_button_original = gr.Button(lang_store[language]["Replace Variables in Original Prompt"])
            insert_button_original.click(
                alignment.insert_kv,
                inputs=[user_prompt_original, kv_input_original],
                outputs=user_prompt_original_replaced,
            )

            insert_button_revise = gr.Button(lang_store[language]["Replace Variables in Revised Prompt"])
            insert_button_revise.click(
                alignment.insert_kv,
                inputs=[user_prompt_eval, kv_input_eval],
                outputs=user_prompt_eval_replaced,
            )

        # モデル選択と実行ボタンの定義
        with gr.Row():
            model_provider_radio = gr.Radio(
                choices=["OpenRouter", "Groq"],
                label=lang_store[language].get("Choose Model Provider for Comparison", "Choose Model Provider for Comparison"),
                value="OpenRouter", # デフォルト値
                interactive=True
            )
            openrouter_model_dropdown = gr.Dropdown(
                label=lang_store[language].get("Choose OpenRouter Model", "Choose OpenRouter Model"),
                choices=[
                    "deepseek/deepseek-chat-v3-0324:free",
                    "deepseek/deepseek-r1-0528:free",
                ],
                value="deepseek/deepseek-chat-v3-0324:free",
            )
            groq_model_dropdown = gr.Dropdown(
                label=lang_store[language].get("Choose Groq Model", "Choose Groq Model"),
                choices=[
                    "compound-beta-mini",
                    "compound-beta",
                ],
                value="compound-beta-mini",
            )

            invoke_button = gr.Button(lang_store[language]["Execute prompt"])

        # モデル実行結果表示エリア
        with gr.Row():
            # ラジオボタンのデフォルト値に基づいて初期ラベルを設定
            # model_provider_radio はこの時点で value を持っています。
            default_provider = model_provider_radio.value
            initial_label_original_key = "Original Prompt Output by {provider}"
            initial_label_eval_key = "Evaluation Prompt Output by {provider}"

            openrouter_output = gr.Textbox(
                label=lang_store[language].get(initial_label_original_key, "Output for Original Prompt ({provider})").format(provider=default_provider),
                lines=3,
                interactive=False,
                show_copy_button=True
            )
            groq_output = gr.Textbox(
                label=lang_store[language].get(initial_label_eval_key, "Output for Evaluation Prompt ({provider})").format(provider=default_provider),
                lines=3,
                interactive=False,
                show_copy_button=True,
            )

        # ラジオボタン変更時に出力ラベルを更新する関数
        def update_output_labels(provider_choice):
            # language と lang_store はこの関数のスコープからアクセス可能です
            label_original_key = "Original Prompt Output by {provider}"
            label_eval_key = "Evaluation Prompt Output by {provider}"

            new_label_original = lang_store[language].get(label_original_key, "Output for Original Prompt ({provider})").format(provider=provider_choice)
            new_label_eval = lang_store[language].get(label_eval_key, "Output for Evaluation Prompt ({provider})").format(provider=provider_choice)

            return gr.update(label=new_label_original), gr.update(label=new_label_eval)

        model_provider_radio.change(
            update_output_labels,
            inputs=[model_provider_radio],
            outputs=[openrouter_output, groq_output]
        )

        # プロンプト実行イベント
        invoke_button.click(
            alignment.invoke_prompt,
            inputs=[
                user_prompt_original_replaced,
                user_prompt_eval_replaced,
                user_prompt_original,
                user_prompt_eval,
                model_provider_radio,
                openrouter_model_dropdown,
                groq_model_dropdown,
            ],
            outputs=[openrouter_output, groq_output],
        )

        # フィードバックと評価、改善プロンプト生成エリア
        with gr.Row():
            feedback_input = gr.Textbox(
                label=lang_store[language]["Evaluate the Prompt Effect"],
                placeholder=lang_store[language]["Input your feedback manually or by model"],
                lines=3,
                show_copy_button=True,
            )
            eval_model_dropdown = gr.Dropdown(
                label=lang_store[language]["Choose the Evaluation Model"],
                choices=[
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "meta-llama/llama-4-maverick-17b-128e-instruct",
                ],
                value="meta-llama/llama-4-scout-17b-16e-instruct",
            )
            # 自動評価ボタン            
            evaluate_button = gr.Button(lang_store[language]["Auto-evaluate the Prompt Effect"])
            evaluate_button.click(
                alignment.evaluate_response,
                inputs=[openrouter_output, groq_output, eval_model_dropdown],
                outputs=[feedback_input],
            )

            # プロンプト改善ボタン
            revise_button = gr.Button(lang_store[language]["Iterate the Prompt"])
            revised_prompt_output = gr.Textbox(
                label=lang_store[language]["Revised Prompt"], lines=3, interactive=False, show_copy_button=True
            )
            revise_button.click(
                alignment.generate_revised_prompt,
                inputs=[
                    feedback_input,
                    user_prompt_eval,
                    openrouter_output,
                    groq_output,
                    eval_model_dropdown,
                ],
                outputs=revised_prompt_output,
            )

    # 「SOE最適化商品説明」タブ
    with gr.Tab(lang_store[language]["SOE-Optimized Product Description"]):
        with gr.Row():
            with gr.Column():
                product_category = gr.Textbox(label=lang_store[language]["Product Category"], placeholder=lang_store[language]["Enter the product category"])
                brand_name = gr.Textbox(label=lang_store[language]["Brand Name"], placeholder=lang_store[language]["Enter the brand name"])
                usage_description = gr.Textbox(label=lang_store[language]["Usage Description"], placeholder=lang_store[language]["Enter the usage description"])
                target_customer = gr.Textbox(label=lang_store[language]["Target Customer"], placeholder=lang_store[language]["Enter the target customer"])
            with gr.Column():
                # 画像アップロードとプレビュー                
                image_preview = gr.Gallery(label=lang_store[language]["Uploaded Images"], show_label=False, elem_id="image_preview")
                image_upload = gr.UploadButton(lang_store[language]["Upload Product Image (Optional)"], file_types=["image", "video"], file_count="multiple")
                generate_button = gr.Button(lang_store[language]["Generate Product Description"])
        
        with gr.Row():
            product_description = gr.Textbox(label=lang_store[language]["Generated Product Description"], lines=10, interactive=False)
            # 商品説明生成イベント
            generate_button.click(
                soeprompt.generate_description,
                inputs=[product_category, brand_name, usage_description, target_customer, image_upload],
                outputs=product_description,
            )
            image_upload.upload(lambda images: images, inputs=image_upload, outputs=image_preview)

    # 「プロンプトキャリブレーション」タブ
    with gr.Tab(lang_store[language]["Prompt Calibration"]):
# デフォルトの後処理コード
        default_code = '''
def postprocess(llm_output):
    return llm_output
'''.strip()
        with gr.Row():
            with gr.Column(scale=2):
                calibration_task = gr.Textbox(label=lang_store[language]["Please input your task"], lines=3)
                calibration_prompt_original = gr.Textbox(label=lang_store[language]["Please input your original prompt"], lines=5, placeholder=lang_store[language]["Summarize the text delimited by triple quotes.\n\n\"\"\"{{insert text here}}\"\"\""])
                # 元のプロンプト入力
            with gr.Column(scale=2):
                postprocess_code = gr.Textbox(label=lang_store[language]["Please input your postprocess code"], lines=3, value=default_code)
                # 後処理コード入力
                dataset_file = gr.File(file_types=['csv'], type='binary')
                # データセットファイルアップロード
        with gr.Row():
            with gr.Column(scale=2):
                calibration_task = gr.Radio(["classification"], value="classification", label=lang_store[language]["Task type"])
            with gr.Column(scale=2):
                steps_num = gr.Slider(1, 5, value=1, step=1, label=lang_store[language]["Epoch"])
                # タスクタイプ選択
            calibration_optimization = gr.Button(lang_store[language]["Optimization based on prediction"])
            # 最適化実行ボタン
            calibration_prompt = gr.Textbox(label=lang_store[language]["Revised Prompt"], lines=3, show_copy_button=True, interactive=False)
            # 改善されたプロンプト表示
            calibration_optimization.click(
                calibration.optimize, inputs=[calibration_task, calibration_prompt_original, dataset_file, postprocess_code, steps_num],
                outputs=calibration_prompt
            )
# Gradioアプリを起動します
demo.launch()
