# 共通定数・文言キー定義（UI用）

# モデル選択肢（evaluation_tabなどで利用）
OPENAI_MODEL_CHOICES = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1-0528:free",
]

GROQ_MODEL_CHOICES = [
    "compound-beta-mini",
    "compound-beta",
]

# 共通ラベルキー（必要に応じて拡張）
LABEL_KEYS = {
    "chat": "Chat",
    "chatbot": "Chatbot",
    "your_message": "Your Message",
    "send": "Send",
    "model_name": "Model Name",
    "system_prompt": "System Prompt",
    "temperature": "Temperature",
    "clear": "Clear",
    "generate_prompt": "Generate Prompt",
    "revised_prompt": "Revised Prompt",
    "opt_level": "Optimize Level",
    "prompt_template_generated": "Prompt Template Generated",
    "prompt_translation": "Prompt Translation",
    "prompt_evaluation": "Prompt Evaluation",
    "meta_prompt": "Meta Prompt",
    "soe_description": "SOE-Optimized Product Description",
    "task": "Task",
    "variables": "Variables",
    "evaluate_effect": "Evaluate the Prompt Effect",
    "iterate_prompt": "Iterate the Prompt",
    "execute_prompt": "Execute prompt",
    "replace_result": "Replace Result",
    "choose_openai": "Choose OpenAI Model",
    "choose_groq": "Choose Groq Model",
}

# タブ説明（Markdown）
TAB_DESCRIPTIONS = {
    "translation": "- 入力: 元のプロンプト\n- 生成: 一回/複数回\n- 注意: 空入力の場合はエラーを表示します。",
    "evaluation": "- 手順: 変数置換 → 実行 → 自動評価/改善\n- 注意: 置換結果が空のまま実行するとエラー表示。",
    "calibration": "- 入力: タスク/元プロンプト/（任意）CSV\n- 注意: CSV拡張子は .csv を推奨。",
    "soe": "- 入力: カテゴリ/ブランド/使用説明/ターゲット\n- 画像動画は任意。拡張子制限あり。",
    "metaprompt": "- 入力: タスク/変数\n- APE実行はメタプロンプト出力生成後に行ってください。",
    "chat": "- 入力: メッセージ\n- 送信中にInfoを表示します。",
}
