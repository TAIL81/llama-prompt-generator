import re
from typing import Any

import gradio as gr


def strip_wrapping_tags(text: str) -> str:
    """
    UI 出力直前に、<rewritten>...</rewritten> または
    表記揺れの <rerwited>...</rerwited> を最外周で 1 度だけ除去する。

    - タグは大文字小文字を区別しない
    - タグの前後・内部の余分な空白/改行は適度にトリム
    - タグが無ければそのまま返す
    """
    if not isinstance(text, str):
        return text

    pattern = re.compile(
        r"^\s*<(rewritten|rerwited)>\s*([\s\S]*?)\s*</\1>\s*$", re.IGNORECASE
    )
    m = pattern.match(text)
    if m:
        # 中身を返す。外側の無駄な空白も削る
        return m.group(2).strip()
    return text


# ========== UI 通知/更新ユーティリティ（P0） ==========
def notify_error(message: str) -> None:
    """
    ユーザー向けのエラーメッセージを統一表示（日本語）。
    """
    gr.Warning(message)


def notify_info(message: str) -> None:
    """
    ユーザー向けの情報メッセージを統一表示（日本語）。
    """
    gr.Info(message)


def clear_value(visible: bool = True):
    """
    値をクリアし、必要に応じて可視性も設定するための共通ヘルパ。
    """
    return gr.update(value="", visible=visible)


def set_visible(visible: bool = True):
    """
    可視性を切り替えるための共通ヘルパ。
    """
    return gr.update(visible=visible)


def set_interactive(interactive: bool = True):
    """
    コンポーネントの編集可否を切り替えるための共通ヘルパ。
    """
    return gr.update(interactive=interactive)


def get_lang(config: Any, key: str, fallback: str) -> str:
    """
    言語ストアから安全に文言を取得。キーが無い場合はfallbackを返す。
    """
    try:
        return config.lang_store[config.language].get(key, fallback)
    except Exception:
        return fallback
