import ast
import json
import logging
import operator
import os
import signal
import sys  # sysモジュールを追加
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union, cast

import gradio as gr
import psutil  # psutilをファイル先頭に追加
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# 外部モジュールのインポート
from src.ape import APE
from src.application.soe_prompt import SOEPrompt
from src.calibration import CalibrationPrompt
from src.metaprompt import MetaPrompt
from src.optimize import Alignment
from src.translate import GuideBased

# UIタブ作成関数のインポート
from src.ui.calibration_tab import create_calibration_tab
from src.ui.evaluation_tab import create_evaluation_tab
from src.ui.metaprompt_tab import create_metaprompt_tab
from src.ui.soe_tab import create_soe_tab
from src.ui.translation_tab import create_translation_tab
from src.ui.chat_tab import create_chat_tab


# 設定管理クラスの導入
class AppConfig(BaseSettings):
    """
    アプリケーション設定を管理するクラス。

    設定は環境変数と翻訳ファイルから読み込まれます。
    """

    # アプリケーションの言語設定（デフォルトは日本語）
    language: str = Field(default="ja", validation_alias="LANGUAGE")
    # 翻訳ファイルのパス（デフォルトはスクリプトと同じディレクトリのtranslations.json）
    TRANSLATIONS_PATH: str = Field(
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "translations.json"
        )
    )
    # 翻訳データを格納する辞書
    lang_store: Dict[str, Any] = {}

    def __init__(self, **data: Any):
        """
        AppConfigのコンストラクタ。
        翻訳ファイルを読み込み、lang_storeに格納します。
        """
        super().__init__(**data)
        self.lang_store = self._load_translations()

    def _load_translations(self) -> Dict[str, Any]:
        """
        翻訳ファイルを読み込みます。

        Returns:
            Dict[str, Any]: 読み込まれた翻訳データ。エラーが発生した場合は空の辞書。
        """
        try:
            with open(self.TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # 翻訳ファイルが見つからない場合のエラーログ
            logging.error(f"翻訳ファイルが見つかりません: {self.TRANSLATIONS_PATH}")
            return {}
        except json.JSONDecodeError as e:
            # 翻訳ファイルのJSON形式が不正な場合のエラーログ
            logging.error(
                f"翻訳ファイルの形式が不正です: {self.TRANSLATIONS_PATH}. エラー: {e}"
            )
            return {}
        except Exception as e:
            # その他の予期せぬエラーが発生した場合のログ
            logging.error(
                f"翻訳ファイルの読み込み中に予期せぬエラーが発生しました: {self.TRANSLATIONS_PATH}. エラー: {e}"
            )
            return {}

    class Config:
        # .envファイルから環境変数を読み込む設定
        env_file = Path(__file__).parent.parent / ".env"
        env_file_encoding = "utf-8"


# AppConfig のインスタンスを作成し、アプリケーション全体で設定を共有
config = AppConfig()
language = config.language
lang_store = config.lang_store


def setup_logging():
    """
    アプリケーションのロギングを設定します。
    INFOレベル以上のメッセージをコンソールに出力します。
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


# ロギングのセットアップを実行
setup_logging()


# コンポーネントの遅延初期化を管理するクラス
class ComponentManager:
    """
    アプリケーションコンポーネントの初期化を遅延させるクラス。

    必要な時にコンポーネントを初期化し、リソースを効率的に利用します。
    これにより、アプリケーションの起動時間を短縮し、不要なリソースの消費を防ぎます。
    """

    def __init__(self, config: AppConfig):
        """
        ComponentManagerのコンストラクタ。

        Args:
            config (AppConfig): アプリケーションの設定オブジェクト。
        """
        self._config = config
        self._instances: Dict[Type, Any] = {}
        self._initializers: Dict[Type, Callable[[], Any]] = {
            APE: lambda: APE(),
            GuideBased: lambda: GuideBased(),
            Alignment: lambda: Alignment(
                lang_store=self._config.lang_store, language=self._config.language
            ),
            MetaPrompt: lambda: MetaPrompt(),
            SOEPrompt: lambda: SOEPrompt(),
            CalibrationPrompt: lambda: CalibrationPrompt(),
        }

    def get(self, component_type: Type[Any]) -> Any:
        """
        指定された型のコンポーネントインスタンスを取得します。
        インスタンスがまだ作成されていない場合は、初期化してから返します。

        Args:
            component_type (Type[Any]): 取得するコンポーネントのクラス。

        Returns:
            Any: コンポーネントのインスタンス。
        """
        if component_type not in self._instances:
            if component_type in self._initializers:
                logging.info(f"Initializing component: {component_type.__name__}")
                self._instances[component_type] = self._initializers[component_type]()
            else:
                raise TypeError(f"Unknown component type: {component_type.__name__}")
        return self._instances[component_type]


# コンポーネントマネージャーのインスタンスを作成
# これにより、各タブが必要なコンポーネントにアクセスできるようになります。
component_manager = ComponentManager(config)

# SafeCodeExecutor のインポートとインスタンス化
# コードの安全な実行を管理するためのコンポーネント
from src.safe_executor import SafeCodeExecutor

safe_code_executor = SafeCodeExecutor()


def ape_prompt(original_prompt: str, user_data: str) -> List[gr.Textbox]:
    """
    APE (Automatic Prompt Engineering) を使用してプロンプトを生成します。
    この関数は、Gradio UIから呼び出されることを想定しています。

    Args:
        original_prompt (str): 元のプロンプトテンプレート。
        user_data (str): JSON形式のユーザーデータ文字列。
                         このデータはプロンプトのデモデータとして使用されます。

    Returns:
        List[gr.Textbox]: Gradioテキストボックスコンポーネントのリスト。
                          生成されたプロンプト、またはエラーメッセージを含むテキストボックスが含まれます。
                          他のタブの出力コンポーネント数と合わせるため、3つの出力のうち2つは非表示になります。
    """
    logging.debug("ape_prompt function was called!")
    logging.debug(f"original_prompt = {original_prompt}")
    logging.debug(f"user_data = {user_data}")

    try:
        # ユーザーデータをJSON文字列からPython辞書にデコード
        parsed_user_data = json.loads(user_data)
    except json.JSONDecodeError as e:
        # JSONデコードエラーが発生した場合の処理
        logging.error(
            f"Error decoding user_data in ape_prompt: {e}. User data was: '{user_data}'"
        )
        # エラーメッセージを表示するテキストボックスを返す
        return [
            gr.Textbox(
                label="Error",
                value=f"Error processing user data: {e}. Please ensure it's valid JSON.",
                lines=3,
                show_copy_button=True,
                interactive=False,
            )
        ] + [
            gr.Textbox(visible=False)
        ] * 2  # 他の出力コンポーネントと数を合わせる

    # APEコンポーネントを使用してプロンプトを生成
    ape_instance = component_manager.get(APE)
    result = ape_instance(original_prompt, 1, parsed_user_data)
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
valid_themes = ["default", "soft", "glass"]
theme_name = "soft" if "soft" in valid_themes else "default"

with gr.Blocks(
    title=config.lang_store[config.language]["Automatic Prompt Engineering"],
    theme=theme_name,
) as demo:
    # クリアボタンのラベルを取得
    clear_button_label = config.lang_store[config.language].get("Clear", "Clear")
    # アプリケーションのタイトルをMarkdownで表示
    gr.Markdown(
        f"# {config.lang_store[config.language]['Automatic Prompt Engineering']}"
    )

    # 各UIタブの作成関数を呼び出し、Gradioインターフェースにタブを追加
    create_metaprompt_tab(component_manager, config)
    create_translation_tab(component_manager, config)
    create_evaluation_tab(component_manager, config)
    create_soe_tab(component_manager, config)
    create_calibration_tab(component_manager, config)
    create_chat_tab()


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """指定されたPIDの子プロセスをすべて終了させる"""
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            print(f"子プロセス {child.pid} を終了します")
            child.send_signal(sig)
    except psutil.NoSuchProcess:
        return  # 親プロセスがすでに存在しない場合は何もしない


# シグナルハンドラ
def signal_handler(sig, frame):
    """シグナルハンドラでグレースフルシャットダウンを開始"""
    print("\nシャットダウンシグナルを受信しました。サーバーを終了します...")
    # Gradioのサーバーを閉じる
    demo.close()
    # このスクリプトから起動した子プロセスを終了
    kill_child_processes(os.getpid())
    sys.exit(0)


# SIGINT (Ctrl+C) と SIGTERM シグナルにハンドラを登録
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# スクリプトが直接実行された場合にGradioアプリケーションを起動
if __name__ == "__main__":
    load_dotenv()  # .envファイルから環境変数をロード
    try:
        # Gradioアプリを起動し、サーバーインスタンスを取得
        app, local_url, share_url = demo.launch(ssr_mode=True, share=True)
    except Exception as e:
        logging.error(f"サーバー起動エラー: {e}")
        # エラー発生時に子プロセスをクリーンアップ
        kill_child_processes(os.getpid())
        raise
