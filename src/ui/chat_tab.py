from typing import Any, Dict, Generator, List

import gradio as gr

from src.chat import ChatService


def create_chat_tab(config: Any):
    lang_store = config.lang_store
    language = config.language
    # ChatService は状態に入れず、外部で保持する
    chat_service = ChatService()
    # 温度の初期値を一元管理
    initial_temperature = 1.0
    # gr.State には deepcopy 可能な軽量 dict のみを保持（messages形式の履歴と設定）
    chat_state = gr.State(
        {"messages": [], "temperature": initial_temperature, "system_prompt": ""}
    )

    if not chat_service.client:
        gr.Warning("OPENAI_API_KEY is not set. Chat tab will not work.")

    # 多言語対応を考慮し、configからデフォルトのシステムプロンプTシャツを取得
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
                # 入力行を独立させて下詰め固定・十分な高さを確保
                with gr.Row(elem_id="chat-input-row"):
                    msg = gr.Textbox(
                        label=lang_store[language].get("Your Message", "Your Message"),
                        lines=3,  # デフォルトの見た目の高さ
                        max_lines=15,  # 伸び上限
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

        # 入力行の潰れ対策CSSを注入（このタブに限定）
        gr.HTML(
            """
            <style>
              /* 左カラム内の配置: Chatbot を上、入力行を下に固定 */
              #chatbot {
                min-height: 360px;
                /* 要望: 14px相当 */
                font-size: 0.875rem !important; /* 14px (16px基準) */
                line-height: 1.5 !important;
              }
              /* 吹き出し内テキストも統一して14px相当に */
              #chatbot .message,
              #chatbot .bubble,
              #chatbot .prose,
              #chatbot .markdown,
              #chatbot p,
              #chatbot li,
              #chatbot code,
              #chatbot pre,
              #chatbot span {
                font-size: 0.875rem !important; /* 14px */
                line-height: 1.5 !important;
              }
              /* Chatbot 内部の上下で背景色が二分されるのを防ぐ（softテーマの段差補正） */
              #chatbot .wrap,
              #chatbot > div {
                background: transparent !important;
              }
              #chatbot > div > div {
                background: transparent !important;
              }
              /* Chatbot本体の背景とボーダーを明確化
                 - 背景は従来のカード色（--block-background-fill）
                 - 外枠は強めのボーダー色にする */
              #chatbot,
              #chatbot .gr-panel,
              #chatbot .gr-chatbot {
                background-color: var(--block-background-fill, #1e2430) !important;
                border: 1px solid var(--border-color-primary, #3a4150) !important;
                box-shadow: none !important;
              }
              /* 入力行の余白と最低高さ */
              #chat-input-row {
                margin-top: 0 !important;
                padding-top: 6px;
              }
              #chat-input-row textarea {
                min-height: 72px; /* 約3行分 */
              }
            </style>
            """
        )
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
