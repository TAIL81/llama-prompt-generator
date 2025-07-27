import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import APIError

from src.ape import APE, GroqConfig


@pytest.fixture
def ape_instance(mocker):
    """APEインスタンスとモッククライアントをセットアップ"""
    # Groqクライアントと環境変数をモック
    mocker.patch("src.ape.os.getenv", return_value="fake-api-key")
    mock_groq_client = MagicMock()
    mocker.patch("src.ape.Groq", return_value=mock_groq_client)

    instance = APE()
    instance.groq_client = mock_groq_client  # テスト用にモッククライアントを注入
    return instance


def test_groq_config_defaults():
    """GroqConfigのデフォルト値が正しいことを検証"""
    config = GroqConfig()
    assert config.rewrite_model == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert config.max_tokens == 8192
    assert config.temperature == 0.7


def test_ape_call_success(ape_instance):
    """APEのメイン処理が正常に動作するケース"""
    # モック設定
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "書き換えられたプロンプト"
    ape_instance.groq_client.chat.completions.create.return_value = mock_response

    # テスト実行
    initial_prompt = "初期プロンプト"
    demo_data = {"key": "value"}
    result = ape_instance(initial_prompt, 1, demo_data)

    # 検証
    assert "prompt" in result
    assert result["prompt"] == "書き換えられたプロンプト"
    assert ape_instance.groq_client.chat.completions.create.call_count >= 2


def test_ape_rewrite_api_error(ape_instance, mocker):
    """APIエラー時のrewriteメソッドの挙動を検証"""
    # APIエラーをシミュレート
    ape_instance.groq_client.chat.completions.create.side_effect = APIError(
        "Test error"
    )

    # テスト実行
    result = ape_instance.rewrite("テストプロンプト")

    # 検証
    assert result is None


def test_ape_generate_more_success(ape_instance):
    """generate_moreが既存プロンプトから新しい候補を生成"""
    # モック設定
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "新しいプロンプト候補"
    ape_instance.groq_client.chat.completions.create.return_value = mock_response

    # テスト実行
    result = ape_instance.generate_more("初期プロンプト", "既存の良い例")

    # 検証
    assert result == "新しいプロンプト候補"


def test_ape_call_empty_demo_data(ape_instance):
    """デモデータが空の場合のエラーハンドリングを検証"""
    with pytest.raises(ValueError, match="デモデータが提供されていません"):
        ape_instance("プロンプト", 1, {})


def test_ape_call_filtering_failure(ape_instance, caplog):
    """カスタマイズ変数が不足した候補のフィルタリングを検証"""
    # モック設定（変数を含まないプロンプトを返す）
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "変数なしプロンプト"
    ape_instance.groq_client.chat.completions.create.return_value = mock_response

    # テスト実行
    demo_data = {"required_var": "value"}
    result = ape_instance("プロンプト", 1, demo_data)

    # 検証
    assert "error" in result
    assert "必要な変数" in result["error"]
    assert "フィルタリング" in caplog.text


if __name__ == "__main__":
    pytest.main()
