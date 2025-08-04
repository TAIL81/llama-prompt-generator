from typing import Any, Generator, List

import gradio as gr

from src.chat import ChatService


def create_chat_tab(config: Any):
    lang_store = config.lang_store
    language = config.language
    chat_service = ChatService()
    chat_service_state = gr.State(
        {
            "api_key": chat_service.api_key,
            "api_base": chat_service.api_base,
            "model": chat_service.model,
            "api_version": chat_service.api_version,
            # clientは除外
        }
    )

    if not chat_service.client:
        gr.Warning("OPENAI_API_KEY is not set. Chat tab will not work.")

    # 多言語対応を考慮し、configからデフォルトのシステムプロンプトを取得
    default_system_prompt = lang_store[language].get(
        "Chat Default System Prompt",
        "Act as a helpful assistant. Think step by step. Reply in fluent Japanese.",
    )

    with gr.Tab(lang_store[language].get("Chat", "Chat")):
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label=lang_store[language].get("Chatbot", "Chatbot")
                )
                msg = gr.Textbox(
                    label=lang_store[language].get("Your Message", "Your Message")
                )
            with gr.Column(scale=1):
                system_prompt = gr.Textbox(
                    label=lang_store[language].get("System Prompt", "System Prompt"),
                    value=default_system_prompt,
                    lines=5,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label=lang_store[language].get("Temperature", "Temperature"),
                )
                clear = gr.ClearButton(value=lang_store[language].get("Clear", "Clear"))

        def respond(
            message, chat_history, system_prompt_text, temp_value, chat_service_instance
        ):
            bot_message_generator = chat_service_instance.chat_completion_stream(
                message,
                chat_history,
                system_prompt_text,
                temp_value,
            )
            chat_history.append([message, ""])
            for token in bot_message_generator:
                chat_history[-1][1] += token
                yield "", chat_history

        clear.add([msg, chatbot])
        msg.submit(
            respond,
            [msg, chatbot, system_prompt, temperature, chat_service_state],
            [msg, chatbot],
        )
