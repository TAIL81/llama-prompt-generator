import sys
import os
import pytest
import json
from unittest.mock import MagicMock, patch

# ルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rater import Rater, GroqConfig
from groq import APIError, RateLimitError


@pytest.fixture
def rater_instance(mocker):
    """Raterインスタンスとモッククライアントをセットアップ"""
    # Groqクライアントと環境変数をモック
    mocker.patch("src.rater.os.getenv", return_value="fake-api-key")
    mock_groq_client = MagicMock()
    mocker.patch("src.rater.Groq", return_value=mock_groq_client)

    instance = Rater()
    instance.groq_client = mock_groq_client  # テスト用にモッククライアントを注入
    return instance


def test_groq_config_defaults():
    """GroqConfigのデフォルト値が正しいことを検証"""
    config = GroqConfig()
    assert config.get_output_model == "meta-llama/llama-4-maverick-17b-128e-instruct"
    assert config.rater_model == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert config.max_tokens_get_output == 1024
    assert config.max_tokens_rater == 8192
    assert config.temperature_get_output == 0.7
    assert config.temperature_rater == 0.7


def test_rater_call_success(rater_instance):
    """候補プロンプトの評価が正常に動作するケース"""
    # モック設定
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "生成された出力"
    rater_instance.groq_client.chat.completions.create.return_value = mock_response

    # テストデータ
    initial_prompt = "初期プロンプト"
    candidates = [{"prompt": "候補1"}, {"prompt": "候補2"}]
    demo_data = {"key": "value"}

    # テスト実行
    result = rater_instance(initial_prompt, candidates, demo_data)

    # 検証
    assert result == 0  # 最初の候補が選択される
    for c in candidates:
        assert "output" in c
        assert c["output"] == "生成された出力"


def test_rater_json_parse_fallback(rater_instance):
    """不正なJSONレスポンスからのフォールバックを検証"""
    # モック設定（不正なJSONレスポンス）
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"Preferred": "Response 1"'
    rater_instance.groq_client.chat.completions.create.return_value = mock_response

    # テストデータ
    initial_prompt = "プロンプト"
    candidates = [
        {"prompt": "候補1", "output": "出力1"},
        {"prompt": "候補2", "output": "出力2"},
    ]

    # テスト実行
    result = rater_instance.rater(initial_prompt, candidates)

    # 検証（フォールバックで最初の候補が選択される）
    assert result == 0


def test_rater_rate_limit_retry(rater_instance, mocker):
    """レートリミットエラー時のリトライを検証"""
    # sleepをモック
    mocker.patch("time.sleep")

    # モック設定（最初はエラー、次に成功）
    mock_success = MagicMock()
    mock_success.choices[0].message.content = json.dumps({"Preferred": "Response 1"})
    rater_instance.groq_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limit exceeded"),
        mock_success,
    ]

    # テストデータ
    initial_prompt = "プロンプト"
    candidates = [
        {"prompt": "候補1", "output": "出力1"},
        {"prompt": "候補2", "output": "出力2"},
    ]

    # テスト実行
    result = rater_instance.rater(initial_prompt, candidates)

    # 検証
    assert result == 0
    assert rater_instance.groq_client.chat.completions.create.call_count == 2


def test_rater_api_error(rater_instance):
    """APIエラー時の挙動を検証"""
    # モック設定（APIエラー）
    rater_instance.groq_client.chat.completions.create.side_effect = APIError(
        "Test error"
    )

    # テストデータ
    initial_prompt = "プロンプト"
    candidates = [
        {"prompt": "候補1", "output": "出力1"},
        {"prompt": "候補2", "output": "出力2"},
    ]

    # テスト実行
    result = rater_instance.rater(initial_prompt, candidates)

    # 検証（エラー時はNoneを返す）
    assert result is None


def test_rater_missing_output(rater_instance):
    """出力生成が失敗した場合の挙動を検証"""
    # モック設定（出力生成失敗）
    rater_instance.groq_client.chat.completions.create.return_value = None

    # テストデータ
    initial_prompt = "プロンプト"
    candidates = [{"prompt": "候補1"}, {"prompt": "候補2"}]
    demo_data = {"key": "value"}

    # テスト実行
    result = rater_instance(initial_prompt, candidates, demo_data)

    # 検証（出力がない候補は評価できない）
    assert result is None


if __name__ == "__main__":
    pytest.main()
