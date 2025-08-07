from typing import Tuple

import gradio as gr

from src.application.soe_prompt import SOEPrompt
from src.ui.utils import notify_error


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
                        scale=1,
                    )
                with gr.Row():
                    # 商品説明生成ボタン
                    generate_button = gr.Button(
                        config.lang_store[config.language][
                            "Generate Product Description"
                        ],
                        scale=4,
                    )
                    # クリアボタン
                    clear_button_soe = gr.Button(
                        config.lang_store[config.language].get("Clear", "Clear"),
                        scale=1,
                    )

        # 生成された商品説明表示セクション（標準化）
        with gr.Row():
            product_description = gr.Textbox(
                label=config.lang_store[config.language][
                    "Generated Product Description"
                ],
                lines=16,  # 標準化
                interactive=False,
                show_copy_button=True,
            )

        # イベントハンドラを登録
        # 入力検証付きラッパ
        def _generate_wrapper(category, brand, usage, target, images):
            if not (category and category.strip()):
                notify_error("製品カテゴリを入力してください。")
                return ""
            if not (brand and brand.strip()):
                notify_error("ブランド名を入力してください。")
                return ""
            if not (usage and usage.strip()):
                notify_error("使用説明を入力してください。")
                return ""
            if not (target and target.strip()):
                notify_error("ターゲット顧客を入力してください。")
                return ""
            # 画像/動画は任意。選択がある場合の軽微チェック（拡張子）
            try:
                files = images or []
                # gr.UploadButton(file_count="multiple") は list で来る場合がある
                if not isinstance(files, list):
                    files = [files]
                allowed_ext = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov")
                for f in files:
                    fname = getattr(f, "name", None) or getattr(f, "orig_name", None) or ""
                    if isinstance(f, dict):
                        fname = f.get("name") or f.get("orig_name") or ""
                    if fname and not fname.lower().endswith(allowed_ext):
                        notify_error("アップロード可能な拡張子は画像/動画のみです。")
                        return ""
            except Exception:
                # 取得不能時はサーバ側に委譲
                pass

            try:
                return component_manager.get(SOEPrompt).generate_description(
                    category, brand, usage, target, images
                )
            except Exception as e:
                notify_error(f"商品説明の生成中にエラーが発生しました: {e}")
                return ""

        # 商品説明生成ボタンがクリックされたときの処理
        generate_button.click(
            _generate_wrapper,
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
                image_preview,  # image_uploadはクリアできないためプレビューをクリア
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
