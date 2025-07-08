import gradio as gr
from typing import Tuple

def create_soe_tab(component_manager, config):
    """SOE最適化商品説明タブのUIを作成し、イベントハンドラを登録します。"""
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
                with gr.Row():
                    image_upload = gr.UploadButton(
                        config.lang_store[config.language][
                            "Upload Product Image (Optional)"
                        ],
                        file_types=["image", "video"],
                        file_count="multiple",
                        scale=1
                    )
                with gr.Row():
                    generate_button = gr.Button(
                        config.lang_store[config.language]["Generate Product Description"],
                        scale=4
                    )
                    clear_button_soe = gr.Button(
                        config.lang_store[config.language].get("Clear", "Clear"),
                        scale=1
                    )

        with gr.Row():
            product_description = gr.Textbox(
                label=config.lang_store[config.language][
                    "Generated Product Description"
                ],
                lines=10,
                interactive=False,
            )
        
        # イベントハンドラを登録
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
        image_upload.upload(
            lambda images: images, inputs=image_upload, outputs=image_preview
        )
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
    """SOE最適化タブの入出力をクリアします。"""
    return "", "", "", "", None, ""
