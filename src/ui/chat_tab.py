from typing import Any, Dict, Generator, List

import gradio as gr

from src.chat import ChatService


def create_chat_tab(config: Any):
    lang_store = config.lang_store
    language = config.language
    # ChatService は状態に入れず、外部で保持する
    chat_service = ChatService()
    # gr.State には deepcopy 可能な軽量 dict のみを保持（messages形式の履歴と設定）
    chat_state = gr.State({"messages": [], "temperature": 0.7, "system_prompt": ""})

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
                    label=lang_store[language].get("Chatbot", "Chatbot"),
                    type="messages",
                    height=400,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label=lang_store[language].get("Your Message", "Your Message"),
                        lines=2,  # 初期表示の行数を設定
                        max_lines=15,  # 最大の高さをこの行数分に制限
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
                    value=0.7,
                    step=0.1,
                    label=lang_store[language].get("Temperature", "Temperature"),
                )
                clear = gr.ClearButton(value=lang_store[language].get("Clear", "Clear"))

        def respond(message, chat_history, system_prompt_text, temp_value, state: Dict):
            # messages形式へ統一
            if not isinstance(chat_history, list):
                chat_history = []
            # state に現在の設定を保持（deepcopy可能なプリミティブのみ）
            state["temperature"] = float(temp_value)
            state["system_prompt"] = system_prompt_text or ""

            # ユーザーメッセージ追加
            chat_history.append({"role": "user", "content": message})

            # ChatService へ渡すための旧形式に一時変換（[user, assistant] ペア）
            history_pairs: List[List[str]] = []
            tmp_user: str = ""
            for m in chat_history:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    # ユーザー開始
                    tmp_user = content
                    # まだアシスタントがないので空で入れておく
                    history_pairs.append([tmp_user, ""])
                elif role == "assistant":
                    # 直前のユーザーに紐づくアシスタント応答を設定
                    if history_pairs:
                        history_pairs[-1][1] = content

            # 生成中のアシスタントメッセージを追加
            chat_history.append({"role": "assistant", "content": ""})

            # ストリーミング開始
            stream = chat_service.chat_completion_stream(
                message=message,
                history=history_pairs,
                system_prompt=state["system_prompt"],
                temperature=state["temperature"],
            )

            try:
                for event in stream:
                    etype = event.get("type")
                    if etype == "content":
                        token = event.get("value", "")
                        chat_history[-1]["content"] += token
                        yield "", chat_history
                    elif etype == "tool_calls":
                        # ツール呼び出しイベントの表示は簡易に追記
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
                        chat_history[-1]["content"] += f"\n[エラー] {err}"
                        yield "", chat_history
            except Exception as e:
                # 予期せぬ例外時もUIが壊れないように追記して返す
                chat_history[-1]["content"] += f"\n[不明なエラー] {str(e)}"
                yield "", chat_history

        clear.add([msg, chatbot])
        msg.submit(
            respond,
            [msg, chatbot, system_prompt, temperature, chat_state],
            [msg, chatbot],
        )
        submit_button.click(
            respond,
            [msg, chatbot, system_prompt, temperature, chat_state],
            [msg, chatbot],
        )
