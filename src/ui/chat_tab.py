from typing import Any, Dict, List

import gradio as gr

from src.chat import ChatService
from src.ui.utils import notify_error, notify_info, set_interactive


def create_chat_tab(config: Any):
    lang_store = config.lang_store
    language = config.language
    chat_service = ChatService()
    initial_temperature = 1.0

    # ChatService は OpenAIクライアントを self.client に保持する実装へ変更済み。
    if not getattr(chat_service, "client", None):
        gr.Warning("GROQ_API_KEY is not set. Chat tab will not work.")

    default_system_prompt = lang_store[language].get(
        "Chat Default System Prompt",
        "Act as a helpful assistant. Think step by step. Reply in fluent Japanese.",
    )

    with gr.Tab(lang_store[language].get("Chat", "Chat")):
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label=lang_store[language].get("Chatbot", "Chatbot"),
                    type="messages",
                    height=None,
                    elem_classes=["chatbot-output"],
                    elem_id="chatbot",
                )
                with gr.Row(elem_id="chat-input-row"):
                    msg = gr.Textbox(
                        label=lang_store[language].get("Your Message", "Your Message"),
                        lines=3,
                        max_lines=15,
                        scale=7,
                    )
                    submit_button = gr.Button(
                        value=lang_store[language].get("Send", "Send"), scale=1
                    )
            with gr.Column(scale=1):
                gr.Textbox(
                    label=lang_store[language].get("Model Name", "Model Name"),
                    value=chat_service.model,
                    interactive=False,
                )
                system_prompt = gr.Textbox(
                    label=lang_store[language].get("System Prompt", "System Prompt"),
                    value=default_system_prompt,
                    lines=5,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=initial_temperature,
                    step=0.1,
                    label=lang_store[language].get("Temperature", "Temperature"),
                )
                clear = gr.ClearButton(value=lang_store[language].get("Clear", "Clear"))

        def respond(
            message: str,
            chat_history: List[Dict[str, str]],
            system_prompt_text: str,
            temp_value: float,
        ):
            # 入力検証
            if not message or not message.strip():
                notify_error("メッセージを入力してください。")
                yield gr.update(), chat_history
                return

            # 送信中UI: 入力・ボタンを無効化（submit_buttonとmsgは外側のイベントで更新）
            notify_info("送信中…")
            if not isinstance(chat_history, list):
                chat_history = []

            chat_history.append({"role": "user", "content": message})

            messages_to_send = []
            if system_prompt_text:
                messages_to_send.append(
                    {"role": "system", "content": system_prompt_text}
                )
            messages_to_send.extend(chat_history)

            chat_history.append({"role": "assistant", "content": ""})

            stream = chat_service.chat_completion_stream(
                messages=messages_to_send,
                temperature=float(temp_value),
            )

            try:
                for event in stream:
                    etype = event.get("type")
                    if etype == "content":
                        token = event.get("value", "")
                        chat_history[-1]["content"] += token
                        # 出力更新のみ
                        yield "", chat_history
                    elif etype == "tool_calls":
                        calls_text = "\n".join(
                            [
                                f"[tool] {c.get('function', {}).get('name','')}({c.get('function', {}).get('arguments','')})"
                                for c in event.get("value", [])
                            ]
                        )
                        if calls_text:
                            chat_history[-1]["content"] += "\n" + calls_text
                            yield "", chat_history
                    elif etype == "error":
                        err = event.get("value", "未知のエラー")
                        chat_history[-1]["content"] = f"[エラー] {err}"
                        yield "", chat_history
            except Exception as e:
                chat_history[-1]["content"] = f"[不明なエラー] {str(e)}"
                yield "", chat_history

        gr.HTML(
            """
            <style>
              #chatbot { min-height: 360px; font-size: 0.875rem !important; line-height: 1.5 !important; }
              #chatbot .message, #chatbot .bubble, #chatbot .prose, #chatbot .markdown, #chatbot p, #chatbot li, #chatbot code, #chatbot pre, #chatbot span { font-size: 0.875rem !important; line-height: 1.5 !important; }
              #chatbot .wrap, #chatbot > div, #chatbot > div > div { background: transparent !important; }
              #chatbot, #chatbot .gr-panel, #chatbot .gr-chatbot { background-color: var(--block-background-fill, #1e2430) !important; border: 1px solid var(--border-color-primary, #3a4150) !important; box-shadow: none !important; }
              #chat-input-row { margin-top: 0 !important; padding-top: 6px; }
              #chat-input-row textarea { min-height: 72px; }
            </style>
            """
        )
        clear.add([msg, chatbot])

        # `gr.State` is removed as it's no longer needed for complex state management
        # 送信中disable/完了でenableのため、submit_button と msg の interactive を追加出力
        msg.submit(
            respond,
            [msg, chatbot, system_prompt, temperature],
            [msg, chatbot],
        )
        submit_button.click(
            respond,
            [msg, chatbot, system_prompt, temperature],
            [msg, chatbot],
        )
