import ast
import json
import logging
import logging.handlers
import operator
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import gradio as gr
from dotenv import load_dotenv

from ape import APE
from application.soe_prompt import SOEPrompt
from calibration import CalibrationPrompt
from metaprompt import MetaPrompt
from optimize import Alignment
from translate import GuideBased


# 設定管理クラスの導入
class AppConfig:  # AppConfig クラス
    def __init__(self):
        self.language: str = "ja"
        self.lang_store: Dict[str, Any] = {}
        self.load_env()
        self.safe_load_translations()
        # ログレベルをAppConfigで管理
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO")  # os.environ.getを使用

    def load_env(self):
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)
        # 言語設定を読み込む (既存の機能)
        self.language = os.environ.get("LANGUAGE", "ja")
        logging.info(f"環境変数から言語設定を読み込みました: {self.language}")

    def safe_load_translations(self):
        translations_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translations.json")
        try:
            with open(translations_path, "r", encoding="utf-8") as f:
                self.lang_store = json.load(f)
            logging.info(
                f"翻訳ファイルを正常に読み込みました: {translations_path}. lang_storeのキー: {self.lang_store.keys()}"
            )  # ロギング
        except FileNotFoundError:
            logging.error(f"翻訳ファイルが見つかりません: {translations_path}")
            self.lang_store = {}
        except json.JSONDecodeError as e:
            logging.error(f"翻訳ファイルの形式が不正です: {translations_path}. エラー: {e}")
            self.lang_store = {}
        except Exception as e:  # その他の例外をキャッチ
            logging.error(f"翻訳ファイルの読み込み中に予期せぬエラーが発生しました: {translations_path}. エラー: {e}")
            self.lang_store = {}


# AppConfigのインスタンスを作成
config = AppConfig()
language = config.language
lang_store = config.lang_store


def setup_logging():
    # AppConfigからログレベルを取得
    log_level = config.log_level
    # RotatingFileHandlerを使用してログファイルを管理
    file_handler = logging.handlers.RotatingFileHandler("app.log", maxBytes=10000, backupCount=5, encoding="utf-8")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[file_handler, logging.StreamHandler()],
    )


setup_logging()


# コンポーネントの遅延初期化を管理するクラス
class ComponentManager:  # ComponentManager クラス
    def __init__(self, config: AppConfig):
        self._config = config
        self._components: Dict[str, Any] = {
            "ape": None,
            "rewrite": None,
            "alignment": None,
            "metaprompt": None,
            "soeprompt": None,
            "calibration": None,
        }

    def __getattr__(self, name: str) -> Any:
        if name in self._components:
            if self._components[name] is None:
                self._initialize_component(name)
            return self._components[name]
        raise AttributeError(f"Component {name} not found")

    def _initialize_component(self, name: str) -> None:
        logging.info(f"Initializing component: {name}")
        if name == "ape":
            self._components[name] = APE()
        elif name == "rewrite":
            self._components[name] = GuideBased()
        elif name == "alignment":
            self._components[name] = Alignment(lang_store=self._config.lang_store, language=self._config.language)
        elif name == "metaprompt":
            self._components[name] = MetaPrompt()  # MetaPrompt のインスタンス化
        elif name == "soeprompt":
            self._components[name] = SOEPrompt()
        elif name == "calibration":
            self._components[name] = CalibrationPrompt()
        else:
            raise ValueError(f"Unknown component: {name}")


# コンポーネントマネージャーのインスタンスを作成
component_manager = ComponentManager(config)

# 各コンポーネントへのアクセスは component_manager.ape のように変更
# 例: ape = component_manager.ape
# ただし、既存のコードの変更を最小限にするため、ここでは直接変数に割り当てます
ape = component_manager.ape
rewrite = component_manager.rewrite
alignment = component_manager.alignment
metaprompt = component_manager.metaprompt
soeprompt = component_manager.soeprompt
calibration = component_manager.calibration


# generate_prompt関数の分割と型ヒントの追加
def create_single_textbox(value: str) -> List[gr.Textbox]:
    return [
        cast(
            gr.Textbox,
            gr.Textbox(
                label=lang_store[language]["Prompt Template Generated"],
                value=value,
                lines=3,
                show_copy_button=True,
                interactive=False,
            ),
        )  # 閉じ括弧を追加
    ] + [
        gr.Textbox(visible=False)
    ] * 2  # テキストボックスのリストを返す


def create_multiple_textboxes(candidates: List[str], judge_result: int) -> List[gr.Textbox]:
    textboxes = []
    for i in range(3):
        is_best = "Y" if judge_result == i else "N"
        textboxes.append(
            cast(
                gr.Textbox,
                gr.Textbox(
                    label=f"{lang_store[language]['Prompt Template Generated']} #{i+1} {is_best}",
                    value=candidates[i],
                    lines=3,
                    show_copy_button=True,
                    visible=True,
                    interactive=False,
                ),
            )  # 閉じ括弧を追加
        )
    return textboxes


def generate_single_prompt(original_prompt: str) -> List[gr.Textbox]:
    """一回生成モード"""
    result = rewrite(original_prompt)
    return create_single_textbox(result)


def generate_multiple_prompts_async(original_prompt: str) -> List[str]:
    """複数回生成モード (非同期実行用)"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(rewrite, original_prompt) for _ in range(3)]
        candidates = [future.result() for future in futures]
    return candidates


def generate_prompt(original_prompt: str, level: str) -> List[gr.Textbox]:
    """
    元のプロンプトと最適化レベルに基づいてプロンプトを生成します。

    Args:
        original_prompt (str): 元のプロンプト。
        level (str): 最適化レベル ("One-time Generation" または "Multiple-time Generation")。

    Returns:
        list: Gradioテキストボックスコンポーネントのリスト。
    """
    if level == "One-time Generation":  # 一回生成モードの場合
        return generate_single_prompt(original_prompt)
    elif level == "Multiple-time Generation":
        candidates = generate_multiple_prompts_async(original_prompt)
        judge_result = rewrite.judge(candidates)
        return create_multiple_textboxes(candidates, judge_result)
    return []  # デフォルトの戻り値


# セキュリティ改善: 制限付きコード実行クラス
class SafeCodeExecutor:
    ALLOWED_NODES = (
        ast.Expression,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.Set,
        ast.Attribute,
        ast.Subscript,
        ast.Index,
        ast.Slice,
    )
    ALLOWED_FUNCTIONS = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "range": range,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "all": all,
        "any": any,
        "getattr": getattr,  # 属性アクセスを許可
        "hasattr": hasattr,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "print": print,  # デバッグ用にprintを許可
    }
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Not: operator.not_,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: operator.contains,
        ast.NotIn: lambda a, b: not operator.contains(a, b),
    }

    def execute_safe_code(self, code_str: str, context: Dict[str, Any]) -> Any:
        try:
            tree = ast.parse(code_str, mode="eval")  # コードをASTにパース

            for node in ast.walk(tree):
                if not isinstance(node, self.ALLOWED_NODES):
                    # 許可されていないノードタイプが含まれているかチェック
                    raise ValueError(f"許可されていないASTノードタイプが含まれています: {type(node).__name__}")
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name) or node.func.id not in self.ALLOWED_FUNCTIONS:
                        raise ValueError(
                            f"許可されていない関数呼び出しが含まれています: {node.func.id if isinstance(node.func, ast.Name) else 'unknown'}"
                        )
                if isinstance(
                    node,
                    (
                        ast.Import,
                        ast.ImportFrom,
                        ast.Lambda,
                        ast.GeneratorExp,
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                        ast.AsyncFunctionDef,
                        ast.Await,
                        ast.Yield,
                        ast.YieldFrom,
                        ast.Starred,
                        ast.AnnAssign,
                        ast.AugAssign,
                        ast.For,
                        ast.AsyncFor,
                        ast.While,
                        ast.If,
                        ast.With,
                        ast.AsyncWith,
                        ast.Raise,
                        ast.Try,
                        ast.Assert,
                        ast.Delete,
                        ast.Pass,
                        ast.Break,
                        ast.Continue,
                        ast.Global,
                        ast.Nonlocal,
                        ast.ClassDef,
                        ast.FunctionDef,
                    ),
                ):
                    raise ValueError(f"許可されていない操作が含まれています: {type(node).__name__}")
                if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                    op_type = type(node.op)
                    if op_type not in self.ALLOWED_OPERATORS:
                        raise ValueError(f"許可されていない演算子が含まれています: {op_type.__name__}")

            # 実行コンテキストを制限
            safe_globals = {"__builtins__": self.ALLOWED_FUNCTIONS}
            safe_globals.update(context)

            return eval(compile(tree, "<string>", "eval"), safe_globals, safe_globals)
        except Exception as e:
            logging.error(f"コード実行エラー: {e}")
            return None


# SafeCodeExecutor のインスタンスを作成
safe_code_executor = SafeCodeExecutor()


# calibration.optimize 関数内で postprocess_code の実行を safe_code_executor を使うように変更する必要がある
# これは calibration.py ファイルの変更が必要になるため、ここでは app.py の変更のみに留める
# ただし、app.py の calibration.optimize の呼び出し部分で postprocess_code が渡されているため、
# calibration.py 側で SafeCodeExecutor を使用するように修正が必要であることをメモしておく。
# 現時点では app.py の変更のみに集中する。
def ape_prompt(original_prompt, user_data):
    logging.debug("ape_prompt function was called!")
    logging.debug(f"original_prompt = {original_prompt}")
    logging.debug(f"user_data = {user_data}")
    """
    APE (Automatic Prompt Engineering) を使用してプロンプトを生成します。

    Args:
        original_prompt (str): 元のプロンプト。
        user_data (str): JSON形式のユーザーデータ。

    Returns:
        list: Gradioテキストボックスコンポーネントのリスト。
    """
    try:
        parsed_user_data = json.loads(user_data)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding user_data in ape_prompt: {e}. User data was: '{user_data}'")
        # エラーが発生した場合、空の辞書を返すか、適切なエラー処理を行う
        # ここでは、エラーメッセージを表示するために空のテキストボックスを返す
        return [
            gr.Textbox(
                label="Error",  # この行に "label=" が追加されました
                value=f"Error processing user data: {e}. Please ensure it's valid JSON.",
                lines=3,
                show_copy_button=True,
                interactive=False,
            )
        ] + [gr.Textbox(visible=False)] * 2

    result = ape(original_prompt, 1, parsed_user_data)
    return [
        gr.Textbox(
            label="Prompt Generated",
            value=result["prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
    ] + [
        gr.Textbox(visible=False)
    ] * 2  # 他のタブとの互換性のための非表示テキストボックス


# メタプロンプト出力に対してAPEを実行する関数
def run_ape_on_metaprompt_output(metaprompt_template, metaprompt_variables_str):
    """
    メタプロンプトで生成されたテンプレートと変数文字列を元にAPEを実行します。

    Args:
        metaprompt_template (str): メタプロンプトによって生成されたプロンプトテンプレート。
        metaprompt_variables_str (str): 改行区切りの変数名文字列 (例: "VAR1\nVAR2")。
                                         metaprompt.py の修正により、変数名の中身のみが含まれます。

    Returns:
        tuple: (APEによって生成された新しいプロンプトテンプレート, 元の変数文字列)
    """
    variable_names = [var for var in metaprompt_variables_str.split("\n") if var]
    demo_data = {}
    for var_name in variable_names:
        # APEはプロンプト内の {{VAR_NAME}} 形式の変数を期待し、demo_dataのキーもそれに合わせる
        placeholder_key_for_ape = "{{{{var_name}}}}"  # 例: {{CUSTOMER_COMPLAINT}
        demo_data[placeholder_key_for_ape] = f"dummy_{var_name.lower()}"  # APE評価用のダミーデータ

    # APEを実行 (epoch=1は固定)
    result_dict = ape(initial_prompt=metaprompt_template, epoch=1, demo_data=demo_data)
    return result_dict["prompt"], metaprompt_variables_str  # variables_result は変更しない


# メタプロンプト生成ボタンクリック時のラッパー関数
def metaprompt_wrapper(task: str, variables_str: str) -> Tuple[str, str, str, str]:
    """
    metapromptを実行し、APE結果表示用のテキストボックスをクリアするための値を返すラッパー。
    """
    prompt, new_vars = metaprompt(task, variables_str)
    # メタプロンプトの出力と、APE側をクリアするための空文字列を返す
    # Gradioの出力コンポーネント数に合わせて4つの値を返す
    return prompt, new_vars, "", ""


# Gradioインターフェースを定義
# lang_storeが設定された後でGradioインターフェースを定義
with gr.Blocks(title=config.lang_store[config.language]["Automatic Prompt Engineering"], theme="soft") as demo:
    gr.Markdown(f"# {config.lang_store[config.language]['Automatic Prompt Engineering']}")

    # 「メタプロンプト」タブ
    with gr.Tab(config.lang_store[config.language]["Meta Prompt"]):
        original_task = gr.Textbox(
            label=config.lang_store[config.language]["Task"],
            lines=3,
            info=config.lang_store[config.language]["Please input your task"],
            placeholder=config.lang_store[config.language]["Draft an email responding to a customer complaint"],
        )
        variables = gr.Textbox(
            label=config.lang_store[config.language]["Variables"],
            info=config.lang_store[config.language]["Please input your variables, one variable per line"],
            lines=5,
            placeholder=config.lang_store[config.language]["CUSTOMER_COMPLAINT\nCOMPANY_NAME"],
        )
        with gr.Column(scale=2):
            with gr.Row():
                metaprompt_button = gr.Button(config.lang_store[config.language]["Generate Prompt"], scale=1)
                ape_on_metaprompt_button = gr.Button(
                    config.lang_store[config.language]["APE on MetaPrompt Output"], scale=1
                )

        with gr.Row():
            with gr.Column():
                prompt_result_meta = gr.Textbox(
                    label=config.lang_store[config.language]["MetaPrompt Output: Prompt Template"],
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                )
                variables_result_meta = gr.Textbox(
                    label=config.lang_store[config.language]["MetaPrompt Output: Variables"],
                    lines=5,
                    show_copy_button=True,
                    interactive=False,
                )
            with gr.Column():
                prompt_result_ape = gr.Textbox(
                    label=config.lang_store[config.language]["APE Output: Prompt Template"],
                    lines=30,
                    show_copy_button=True,
                    interactive=False,
                    scale=1,
                )
                variables_result_ape = gr.Textbox(
                    label=config.lang_store[config.language]["APE Output: Variables"],
                    lines=5,
                    show_copy_button=True,
                    interactive=False,
                    scale=1,
                )

        metaprompt_button.click(
            metaprompt_wrapper,
            inputs=[original_task, variables],
            outputs=[prompt_result_meta, variables_result_meta, prompt_result_ape, variables_result_ape],
        )

        ape_on_metaprompt_button.click(
            run_ape_on_metaprompt_output,
            inputs=[prompt_result_meta, variables_result_meta],
            outputs=[prompt_result_ape, variables_result_ape],
        )
    # 「プロンプト翻訳」タブ (実際にはプロンプトの書き換え・改善機能)
    with gr.Tab(config.lang_store[config.language]["Prompt Translation"]):
        original_prompt = gr.Textbox(
            label=config.lang_store[config.language]["Please input your original prompt"],
            lines=3,
            placeholder=config.lang_store[config.language][
                'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
            ],
        )
        gr.Markdown(r"Use {\{xxx\}} to express custom variable, e.g. {\{document\}}")
        with gr.Row():
            with gr.Column(scale=2):
                level = gr.Radio(
                    ["One-time Generation", "Multiple-time Generation"],
                    label=config.lang_store[config.language]["Optimize Level"],
                    value="One-time Generation",
                )
                b1 = gr.Button(config.lang_store[config.language]["Generate Prompt"])
                textboxes = []
                for i in range(3):
                    t = gr.Textbox(
                        label=config.lang_store[config.language]["Prompt Template Generated"],
                        elem_id="textbox_id",
                        lines=3,
                        show_copy_button=True,
                        interactive=False,
                        visible=False if i > 0 else True,
                    )
                    textboxes.append(t)
                b1.click(generate_prompt, inputs=[original_prompt, level], outputs=textboxes)

    # プロンプト評価タブの定義
    with gr.Tab(config.lang_store[config.language]["Prompt Evaluation"]):
        with gr.Row():
            user_prompt_original = gr.Textbox(
                label=config.lang_store[config.language]["Please input your original prompt"], lines=3
            )
            kv_input_original = gr.Textbox(
                label=config.lang_store[config.language]["[Optional]Input the template variable need to be replaced"],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            user_prompt_original_replaced = gr.Textbox(
                label=config.lang_store[config.language]["Replace Result"], lines=3, interactive=False
            )
            user_prompt_eval = gr.Textbox(
                label=config.lang_store[config.language]["Please input the prompt need to be evaluate"], lines=3
            )  # 評価用プロンプト入力
            kv_input_eval = gr.Textbox(
                label=config.lang_store[config.language]["[Optional]Input the template variable need to be replaced"],
                placeholder="Ref format: key1:value1;key2:value2",
                lines=3,
            )
            user_prompt_eval_replaced = gr.Textbox(
                label=config.lang_store[config.language]["Replace Result"], lines=3, interactive=False
            )

        # 変数置換ボタンの定義
        with gr.Row():
            insert_button_original = gr.Button(
                config.lang_store[config.language]["Replace Variables in Original Prompt"]
            )
            insert_button_original.click(
                alignment.insert_kv,
                inputs=[user_prompt_original, kv_input_original],
                outputs=user_prompt_original_replaced,
            )

            insert_button_revise = gr.Button(config.lang_store[config.language]["Replace Variables in Revised Prompt"])
            insert_button_revise.click(
                alignment.insert_kv,
                inputs=[user_prompt_eval, kv_input_eval],  # 評価用プロンプトと変数を入力
                outputs=user_prompt_eval_replaced,
            )

        # モデル選択と実行ボタンの定義
        with gr.Row():
            openrouter_model_dropdown = gr.Dropdown(
                label=config.lang_store[config.language].get("Choose OpenRouter Model", "Choose OpenRouter Model"),
                choices=[
                    "deepseek/deepseek-chat-v3-0324:free",
                    "deepseek/deepseek-r1-0528:free",
                ],
                value="deepseek/deepseek-chat-v3-0324:free",
            )
            groq_model_dropdown = gr.Dropdown(
                label=config.lang_store[config.language].get("Choose Groq Model", "Choose Groq Model"),
                choices=[
                    "compound-beta-mini",
                    "compound-beta",
                ],
                value="compound-beta-mini",
            )

            invoke_button = gr.Button(config.lang_store[config.language]["Execute prompt"])

        # モデル実行結果表示エリア
        with gr.Row():
            openrouter_output = gr.Textbox(
                label=config.lang_store[config.language]["元のプロンプトの出力 (OpenRouter)"],
                lines=3,
                interactive=False,
                show_copy_button=True,
            )
            groq_output = gr.Textbox(
                label=config.lang_store[config.language]["評価が必要なプロンプトの出力 (Groq)"],
                lines=3,
                interactive=False,
                show_copy_button=True,
            )

        # プロンプト実行イベント
        invoke_button.click(
            alignment.invoke_prompt,
            inputs=[
                user_prompt_original_replaced,
                user_prompt_eval_replaced,
                user_prompt_original,
                user_prompt_eval,
                openrouter_model_dropdown,
                groq_model_dropdown,
            ],
            outputs=[openrouter_output, groq_output],
        )

        # フィードバックと評価、改善プロンプト生成エリア
        with gr.Row():
            feedback_input = gr.Textbox(
                label=config.lang_store[config.language]["Evaluate the Prompt Effect"],
                placeholder=config.lang_store[config.language]["Input your feedback manually or by model"],
                lines=3,
                show_copy_button=True,
            )
            eval_model_dropdown = gr.Dropdown(
                label=config.lang_store[config.language]["Choose the Evaluation Model"],
                choices=[
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "meta-llama/llama-4-maverick-17b-128e-instruct",
                ],
                value="meta-llama/llama-4-scout-17b-16e-instruct",
            )
            # 自動評価ボタン
            evaluate_button = gr.Button(config.lang_store[config.language]["Auto-evaluate the Prompt Effect"])
            evaluate_button.click(
                alignment.evaluate_response,
                inputs=[openrouter_output, groq_output, eval_model_dropdown],
                outputs=[feedback_input],
            )

            # プロンプト改善ボタン
            revise_button = gr.Button(config.lang_store[config.language]["Iterate the Prompt"])  # 改善ボタン
            revised_prompt_output = gr.Textbox(  # 改善後プロンプト表示
                label=config.lang_store[config.language]["Revised Prompt"],
                lines=3,
                interactive=False,
                show_copy_button=True,
            )
            revise_button.click(
                alignment.generate_revised_prompt,
                inputs=[feedback_input, user_prompt_eval, openrouter_output, groq_output, eval_model_dropdown],
                outputs=revised_prompt_output,
            )

    # 「SOE最適化商品説明」タブ
    with gr.Tab(config.lang_store[config.language]["SOE-Optimized Product Description"]):
        with gr.Row():
            with gr.Column():
                product_category = gr.Textbox(
                    label=config.lang_store[config.language]["Product Category"],
                    placeholder=config.lang_store[config.language]["Enter the product category"],
                )
                brand_name = gr.Textbox(
                    label=config.lang_store[config.language]["Brand Name"],
                    placeholder=config.lang_store[config.language]["Enter the brand name"],
                )
                usage_description = gr.Textbox(
                    label=config.lang_store[config.language]["Usage Description"],
                    placeholder=config.lang_store[config.language]["Enter the usage description"],
                )
                target_customer = gr.Textbox(
                    label=config.lang_store[config.language]["Target Customer"],
                    placeholder=config.lang_store[config.language]["Enter the target customer"],
                )
            with gr.Column():
                # 画像アップロードとプレビュー
                image_preview = gr.Gallery(
                    label=config.lang_store[config.language]["Uploaded Images"],
                    show_label=False,
                    elem_id="image_preview",
                )
                image_upload = gr.UploadButton(
                    config.lang_store[config.language]["Upload Product Image (Optional)"],
                    file_types=["image", "video"],
                    file_count="multiple",
                )
                generate_button = gr.Button(config.lang_store[config.language]["Generate Product Description"])

        with gr.Row():
            product_description = gr.Textbox(
                label=config.lang_store[config.language]["Generated Product Description"], lines=10, interactive=False
            )
        # 商品説明生成イベント
        generate_button.click(
            soeprompt.generate_description,
            inputs=[product_category, brand_name, usage_description, target_customer, image_upload],
            outputs=product_description,
        )
        image_upload.upload(lambda images: images, inputs=image_upload, outputs=image_preview)

    # 「プロンプトキャリブレーション」タブ
    with gr.Tab(config.lang_store[config.language]["Prompt Calibration"]):  # 例: タブ名の変更（必要に応じて）
        default_code = """
def postprocess(llm_output):
    return llm_output
""".strip()
        with gr.Row():
            with gr.Column(scale=2):  # カラムの定義
                calibration_task = gr.Textbox(
                    label=config.lang_store[config.language]["Please input your task"], lines=3
                )
                calibration_prompt_original = gr.Textbox(
                    label=config.lang_store[config.language]["Please input your original prompt"],
                    lines=5,
                    placeholder=config.lang_store[config.language][
                        'Summarize the text delimited by triple quotes.\n\n"""{{insert text here}}"""'
                    ],
                )
            with gr.Column(scale=2):  # カラムの定義
                postprocess_code = gr.Textbox(
                    label=config.lang_store[config.language]["Please input your postprocess code"],
                    lines=3,
                    value=default_code,
                )  # 例: ラベルの変更（より明確に）
                dataset_file = gr.File(file_types=["csv"], type="binary")
        with gr.Row():  # 行の定義
            calibration_task_type = gr.Radio(
                ["classification"], value="classification", label=config.lang_store[config.language]["Task type"]
            )  # タスクタイプ選択
            steps_num = gr.Slider(1, 5, value=1, step=1, label=config.lang_store[config.language]["Epoch"])
        # 例: 不要なラベルの削除（またはより具体的なラベルに変更）
        #  calibration_prompt = gr.Textbox(label=config.lang_store[config.language]["Revised Prompt"],
        #                                  lines=3, show_copy_button=True, interactive=False)
        calibration_optimization = gr.Button(config.lang_store[config.language]["Optimization based on prediction"])
        calibration_prompt = gr.Textbox(
            label=config.lang_store[config.language]["Revised Prompt"],
            lines=3,
            show_copy_button=True,
            interactive=False,
        )
        calibration_optimization.click(
            calibration.optimize,
            inputs=[calibration_task, calibration_prompt_original, dataset_file, postprocess_code, steps_num],
            outputs=calibration_prompt,
        )


def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    time.sleep(2)  # タスク完了のための猶予時間（必要に応じて調整）
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    demo.launch()
