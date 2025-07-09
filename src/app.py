import ast
import json
import logging
import operator
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor # 並行処理のためのThreadPoolExecutorをインポート
from pathlib import Path # ファイルパスを扱うためのPathクラスをインポート
from typing import Any, Callable, Dict, List, Tuple, Type, Union, cast # 型ヒントのための各種型をインポート

import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from ape import APE
from application.soe_prompt import SOEPrompt
from calibration import CalibrationPrompt
from metaprompt import MetaPrompt
from optimize import Alignment
from translate import GuideBased
from ui.calibration_tab import create_calibration_tab
from ui.evaluation_tab import create_evaluation_tab
from ui.metaprompt_tab import create_metaprompt_tab
from ui.soe_tab import create_soe_tab
from ui.translation_tab import create_translation_tab


# 設定管理クラスの導入
class AppConfig(BaseSettings):
    """アプリケーション設定を管理するクラス。

    設定は環境変数と翻訳ファイルから読み込まれます。
    """

    language: str = Field(default="ja", validation_alias="LANGUAGE")
    TRANSLATIONS_PATH: str = Field(
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "translations.json"
        )
    )
    lang_store: Dict[str, Any] = {}

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.lang_store = self._load_translations()

    def _load_translations(self) -> Dict[str, Any]:
        """翻訳ファイルを読み込みます。"""
        try:
            with open(self.TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"翻訳ファイルが見つかりません: {self.TRANSLATIONS_PATH}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(
                f"翻訳ファイルの形式が不正です: {self.TRANSLATIONS_PATH}. エラー: {e}"
            )
            return {}
        except Exception as e:
            logging.error(
                f"翻訳ファイルの読み込み中に予期せぬエラーが発生しました: {self.TRANSLATIONS_PATH}. エラー: {e}"
            )
            return {}

    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        env_file_encoding = "utf-8"


# AppConfig のインスタンスを作成
config = AppConfig()
language = config.language
lang_store = config.lang_store


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


setup_logging()


# コンポーネントの遅延初期化を管理するクラス
class ComponentManager:
    """
    アプリケーションコンポーネントの初期化を遅延させるクラス。

    必要な時にコンポーネントを初期化し、リソースを効率的に利用します。
    """

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
        """存在しない属性へのアクセスを処理し、コンポーネントを初期化して返します。"""
        if name in self._components:
            if self._components[name] is None:
                self._initialize_component(name)
            return self._components[name]
        raise AttributeError(f"Component {name} not found")

    def _initialize_component(self, name: str) -> None:  # コンポーネントの初期化を行う
        logging.info(f"Initializing component: {name}")
        if name == "ape":
            self._components[name] = APE()  # APEコンポーネントの初期化
        elif name == "rewrite":
            self._components[name] = GuideBased()  # GuideBasedコンポーネントの初期化
        elif name == "alignment":
            self._components[name] = Alignment(
                lang_store=self._config.lang_store, language=self._config.language
            )
        elif name == "metaprompt":  # メタプロンプトコンポーネントの初期化
            self._components[name] = MetaPrompt()  # MetaPrompt のインスタンス化
        elif name == "soeprompt":
            self._components[name] = SOEPrompt()
        elif name == "calibration":
            self._components[name] = CalibrationPrompt()
        else:
            raise ValueError(f"Unknown component: {name}")


# コンポーネントマネージャーのインスタンスを作成
component_manager = ComponentManager(config)

# 各コンポーネントへのアクセスは component_manager インスタンス経由で行います。
# 例: component_manager.ape

from safe_executor import SafeCodeExecutor

# SafeCodeExecutor のインスタンスを作成
safe_code_executor = SafeCodeExecutor()


def ape_prompt(original_prompt: str, user_data: str) -> List[gr.Textbox]:
    """APE (Automatic Prompt Engineering) を使用してプロンプトを生成します。

    Args:
        original_prompt (str): 元の��ロンプト。
        user_data (str): JSON形式のユーザーデータ。
    Returns:
        List[gr.Textbox]: Gradioテキストボックスコンポーネントのリスト。生成されたプロンプトを含む。
    """
    logging.debug("ape_prompt function was called!")
    logging.debug(f"original_prompt = {original_prompt}")
    logging.debug(f"user_data = {user_data}")

    try:
        parsed_user_data = json.loads(user_data)
    except json.JSONDecodeError as e:
        logging.error(
            f"Error decoding user_data in ape_prompt: {e}. User data was: '{user_data}'"
        )
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

    result = component_manager.ape(original_prompt, 1, parsed_user_data)
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
    ] * 2  # 他のタブの出力コンポーネント数と合わせるため、3つの出力のうち2つは非表示にする


# Gradioインターフェースを定義
with gr.Blocks(
    title=config.lang_store[config.language]["Automatic Prompt Engineering"],
    theme="soft",
) as demo:
    clear_button_label = config.lang_store[config.language].get("Clear", "Clear")
    gr.Markdown(
        f"# {config.lang_store[config.language]['Automatic Prompt Engineering']}"
    )

    create_metaprompt_tab(component_manager, config)
    create_translation_tab(component_manager, config)
    create_evaluation_tab(component_manager, config)
    create_soe_tab(component_manager, config)
    create_calibration_tab(component_manager, config)


# シグナルハンドラ
def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    demo.close()  # Gradioサーバーを安全に停止


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    demo.launch()
