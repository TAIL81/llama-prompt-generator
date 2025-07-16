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
from ape import APE
from application.soe_prompt import SOEPrompt
from calibration import CalibrationPrompt
from metaprompt import MetaPrompt
from optimize import Alignment
from translate import GuideBased

# UIタブ作成関数のインポート
from ui.calibration_tab import create_calibration_tab
from ui.evaluation_tab import create_evaluation_tab
from ui.metaprompt_tab import create_metaprompt_tab
from ui.soe_tab import create_soe_tab
from ui.translation_tab import create_translation_tab


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
        # 各コンポーネントのインスタンスをNoneで初期化
        self._components: Dict[str, Any] = {
            "ape": None,
            "rewrite": None,
            "alignment": None,
            "metaprompt": None,
            "soeprompt": None,
            "calibration": None,
        }

    def __getattr__(self, name: str) -> Any:
        """
        存在しない属性へのアクセスを処理し、コンポーネントを初期化して返します。
        これにより、`component_manager.ape`のようにアクセスすると、必要に応じてAPEコンポーネントが初期化されます。

        Args:
            name (str): アクセスされた属性の名前（コンポーネント名）。

        Returns:
            Any: 初期化されたコンポーネントのインスタンス。

        Raises:
            AttributeError: 指定されたコンポーネント名が見つからない場合。
        """
        if name in self._components:
            # コンポーネントがまだ初期化されていない場合、初期化を実行
            if self._components[name] is None:
                self._initialize_component(name)
            return self._components[name]
        # 存在しない属性へのアクセスの場合、AttributeErrorを発生
        raise AttributeError(f"Component {name} not found")

    def _initialize_component(self, name: str) -> None:
        """
        指定された名前のコンポーネントを初期化します。

        Args:
            name (str): 初期化するコンポーネントの名前。

        Raises:
            ValueError: 未知のコンポーネント名が指定された場合。
        """
        logging.info(f"Initializing component: {name}")
        if name == "ape":
            self._components[name] = APE()  # APEコンポーネントの初期化
        elif name == "rewrite":
            self._components[name] = GuideBased()  # GuideBasedコンポーネントの初期化
        elif name == "alignment":
            self._components[name] = Alignment(
                lang_store=self._config.lang_store, language=self._config.language
            )
        elif name == "metaprompt":
            self._components[name] = MetaPrompt()  # MetaPrompt のインスタンス化
        elif name == "soeprompt":
            self._components[name] = SOEPrompt()
        elif name == "calibration":
            self._components[name] = CalibrationPrompt()
        else:
            raise ValueError(f"Unknown component: {name}")


# コンポーネントマネージャーのインスタンスを作成
# これにより、各タブが必要なコンポーネントにアクセスできるようになります。
component_manager = ComponentManager(config)

# SafeCodeExecutor のインポートとインスタンス化
# コードの安全な実行を管理するためのコンポーネント
from safe_executor import SafeCodeExecutor

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


def kill_port_process(port: int):
    """指定ポートを使用しているプロセスを強制終了"""
    for proc in psutil.process_iter(["pid", "name", "connections"]):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    print(f"ポート{port}のプロセスを終了します (PID: {proc.pid})")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


# シグナルハンドラ
def signal_handler(sig, frame):
    """シグナルハンドラでグレースフルシャットダウンを開始"""
    print("\nシャットダウンシグナルを受信しました。サーバーを終了します...")
    demo.close()  # Gradioサーバーを安全に停止
    kill_port_process(7860)  # ポートを解放
    sys.exit(0)


# SIGINT (Ctrl+C) と SIGTERM シグナルにハンドラを登録
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# スクリプトが直接実行された場合にGradioアプリケーションを起動
if __name__ == "__main__":
    load_dotenv()  # .envファイルから環境変数をロード
    try:
        demo.launch()  # Gradioアプリケーションを起動
    except Exception as e:
        logging.error(f"サーバー起動エラー: {e}")
        kill_port_process(7860)  # エラー時もポートを解放
        raise
