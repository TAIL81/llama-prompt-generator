import gradio as gr
from typing import Tuple

def create_soe_tab(component_manager, config):
    """
    SOE最適化商品説明タブのUIを作成し、イベントハンドラを登録します。

    Args:
        component_manager: アプリケーションのコンポーネントを管理するオブジェクト。
        config: アプリケーションの設定オブジェクト。
    """
    with gr.Tab(
        config.lang_store[config.language]["SOE-Optimized Product Description"]
    ):
        # 入力フィールドのセクション
        with gr.Row():
            with gr.Column():
                # 製品カテゴリ入力用のテキストボックス
                product_category = gr.Textbox(
                    label=config.lang_store[config.language]["Product Category"],
                    placeholder=config.lang_store[config.language][
                        "Enter the product category"
                    ],
                )
                # ブランド名入力用のテキストボックス
                brand_name = gr.Textbox(
                    label=config.lang_store[config.language]["Brand Name"],
                    placeholder=config.lang_store[config.language][
                        "Enter the brand name"
                    ],
                )
                # 使用説明入力用のテキストボックス
                usage_description = gr.Textbox(
                    label=config.lang_store[config.language]["Usage Description"],
                    placeholder=config.lang_store[config.language][
                        "Enter the usage description"
                    ],
                )
                # ターゲット顧客入力用のテキストボックス
                target_customer = gr.Textbox(
                    label=config.lang_store[config.language]["Target Customer"],
                    placeholder=config.lang_store[config.language][
                        "Enter the target customer"
                    ],
                )
            with gr.Column():
                # 画像アップロードとプレビューセクション
                image_preview = gr.Gallery(
                    label=config.lang_store[config.language]["Uploaded Images"],
                    show_label=False,
                    elem_id="image_preview",
                )
                with gr.Row():
                    # 画像アップロードボタン
                    image_upload = gr.UploadButton(
                        config.lang_store[config.language][
                            "Upload Product Image (Optional)"
                        ],
                        file_types=["image", "video"],
                        file_count="multiple",
                        scale=1
                    )
                with gr.Row():
                    # 商品説明生成ボタン
                    generate_button = gr.Button(
                        config.lang_store[config.language]["Generate Product Description"],
                        scale=4
                    )
                    # クリアボタン
                    clear_button_soe = gr.Button(
                        config.lang_store[config.language].get("Clear", "Clear"),
                        scale=1
                    )

        # 生成された商品説明表示セクション
        with gr.Row():
            product_description = gr.Textbox(
                label=config.lang_store[config.language][
                    "Generated Product Description"
                ],
                lines=10,
                interactive=False,
            )
        
        # イベントハンドラを登録
        # 商品説明生成ボタンがクリックされたときの処理
        generate_button.click(
            component_manager.soeprompt.generate_description,
            inputs=[
                product_category,
                brand_name,
                usage_description,
                target_customer,
                image_upload,
            ],
            outputs=product_description,
        )
        # 画像アップロードコンポーネントにファイルがアップロードされたときの処理
        image_upload.upload(
            lambda images: images, inputs=image_upload, outputs=image_preview
        )
        # クリアボタンがクリックされたときの処理
        clear_button_soe.click(
            clear_soe_tab,
            inputs=[],
            outputs=[
                product_category,
                brand_name,
                usage_description,
                target_customer,
                image_preview, # image_uploadはクリアできないためプレビューをクリア
                product_description,
            ],
        )

def clear_soe_tab() -> Tuple[str, str, str, str, None, str]:
    """
    SOE最適化タブのすべての入力フィールドと出力フィールドをクリアします。

    Returns:
        Tuple[str, str, str, str, None, str]: クリアされたフィールドの空文字列タプル、
                                             None（画像プレビュー用）、空文字列（商品説明用）。
    """
    return "", "", "", "", None, ""
