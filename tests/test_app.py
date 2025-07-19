import os
import signal
import sys
from unittest.mock import MagicMock, call, patch

import pytest

# ルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.app as app_module  # モジュールエイリアスの作成

# テスト対象モジュール
from src.ape import APE  # APEクラスの明示的インポートを追加
from src.app import AppConfig, ComponentManager, demo, setup_logging
from src.metaprompt import MetaPrompt # MetaPromptをインポート


@pytest.fixture
def mock_env(monkeypatch):
    """環境変数をモック"""
    monkeypatch.setenv("LANGUAGE", "ja")
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key")


def test_app_config_loads_translations(mock_env):
    """設定クラスが翻訳ファイルを正しく読み込む"""
    config = AppConfig()
    assert config.language == "ja"
    assert "Automatic Prompt Engineering" in config.lang_store["ja"]
    assert isinstance(config.lang_store, dict)


def test_app_config_missing_translations(mock_env, tmp_path):
    """翻訳ファイルがない場合のエラーハンドリング"""
    # 一時的な不正な翻訳ファイルを作成
    invalid_path = tmp_path / "invalid.json"
    with open(invalid_path, "w") as f:
        f.write("{ invalid json }")

    # 不正なパスで設定を初期化
    with patch("src.app.AppConfig.Config.env_file", new=None), patch(
        "src.app.AppConfig.TRANSLATIONS_PATH", str(invalid_path)
    ):
        config = AppConfig()
        assert config.lang_store == {}


def test_component_manager_lazy_initialization():
    """コンポーネントの遅延初期化が正しく動作する"""
    mock_config = MagicMock()
    manager = ComponentManager(mock_config)

    # 初回アクセスで初期化
    ape_instance = manager.get(APE)
    assert ape_instance is not None

    # 2回目は同じインスタンスを返す
    assert manager.get(APE) is ape_instance


def test_gradio_ui_setup():
    """Gradio UIが正しくセットアップされる"""
    with patch("src.app.gr.Blocks") as mock_blocks, patch(
        "src.app.create_metaprompt_tab"
    ), patch("src.app.create_translation_tab"), patch(
        "src.app.create_evaluation_tab"
    ), patch(
        "src.app.create_soe_tab"
    ), patch(
        "src.app.create_calibration_tab"
    ):
        # ComponentManagerのmetaprompt属性を明示的に初期化
        manager = ComponentManager(MagicMock())
        manager.get(APE)  # APEコンポーネントを初期化
        manager.get(MetaPrompt)  # MetaPromptコンポーネントを初期化

        # UI構築処理を実行
        with demo:
            pass

        # 検証
        mock_blocks.assert_called_once_with(
            title="自動プロンプトエンジニアリング", theme="soft"
        )
        assert mock_blocks().Markdown.called
        assert mock_blocks().__enter__().launch.called


def test_signal_handling_registration():
    """シグナルハンドラが正しく登録される"""
    with patch("src.app.signal.signal") as mock_signal:
        # モジュールの再読み込みでシグナルハンドラ登録をトリガー
        import importlib

        importlib.reload(app_module)  # エイリアスを使用した参照

        # 検証
        expected_calls = [
            call(signal.SIGINT, app_module.signal_handler),
            call(signal.SIGTERM, app_module.signal_handler),
        ]
        mock_signal.assert_has_calls(expected_calls, any_order=True)


def test_process_cleanup():
    """プロセスクリーンアップが正しく動作する"""
    mock_process = MagicMock()
    mock_process.children.return_value = [MagicMock(pid=123), MagicMock(pid=456)]

    with patch("src.app.psutil.Process", return_value=mock_process), patch(
        "src.app.sys.exit"
    ) as mock_exit:

        # シグナルハンドラを実行
        app_module.signal_handler(signal.SIGINT, None)  # エイリアスを使用した参照

        # 検証
        assert mock_process.children.called
        assert mock_process.children()[0].send_signal.called_with(signal.SIGTERM)
        assert mock_process.children()[1].send_signal.called_with(signal.SIGTERM)
        mock_exit.assert_called_once_with(0)


def test_main_execution():
    """メイン実行が正しく動作する"""
    with patch("src.app.load_dotenv"), patch(
        "src.app.demo.launch"
    ) as mock_launch, patch("sys.exit") as mock_exit, patch(
        "src.app.kill_child_processes"
    ):

        # メインモジュール実行をシミュレート
        app_module.__name__ = "__main__"
        app_module.demo.launch()  # エイリアスを使用した参照

        # 検証
        mock_launch.assert_called_once()
        mock_exit.assert_not_called()  # 正常実行時はexitしない


if __name__ == "__main__":
    pytest.main()
