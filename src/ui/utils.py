import re


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
