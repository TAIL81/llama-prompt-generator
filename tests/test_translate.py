import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from groq import APIError, RateLimitError

# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.translate import GuideBased


@pytest.fixture
def guide_based_instance(mocker):
    """
    GuideBasedクラスのインスタンスを生成し、Groqクライアントをモック化するフィクスチャ。
    """
    # GroqクライアントのコンストラクタとAPIキーの読み込みをモック化
    mocker.patch("src.translate.os.getenv", return_value="fake-api-key")
    mock_groq_client = MagicMock()
    mocker.patch("src.translate.Groq", return_value=mock_groq_client)

    # GuideBasedのインスタンスを生成
    instance = GuideBased()
    # テスト内でモッククライアントにアクセスできるように、インスタンスに直接設定
    instance.groq_client = mock_groq_client
    return instance


# --- detect_lang メソッドのテスト ---


@pytest.mark.parametrize(
    "lang_code",
    [
        "ja",
        "en",
        "ch",
    ],
)
def test_detect_lang_success(guide_based_instance, lang_code):
    """言語検出が成功するケースをテストします。"""
    # APIからの正常な応答をシミュレート
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({"lang": lang_code})
    guide_based_instance.groq_client.chat.completions.create.return_value = (
        mock_response
    )

    result = guide_based_instance.detect_lang("some prompt")
    assert result == lang_code


def test_detect_lang_invalid_json(guide_based_instance):
    """APIが不正なJSONを返した場合のテスト。"""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"lang": "ja"'  # 不正なJSON
    guide_based_instance.groq_client.chat.completions.create.return_value = (
        mock_response
    )

    result = guide_based_instance.detect_lang("some prompt")
    assert result == ""


def test_detect_lang_api_error(guide_based_instance):
    """APIエラーが発生した場合のテスト。"""
    guide_based_instance.groq_client.chat.completions.create.side_effect = APIError(
        "Test API Error", request=None, body=None
    )

    result = guide_based_instance.detect_lang("some prompt")
    assert result == ""


def test_detect_lang_rate_limit_error(guide_based_instance, mocker):
    """レート制限エラーからのリトライをテストします。"""
    mocker.patch("time.sleep")  # time.sleepをモック化して待機時間をなくす
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({"lang": "en"})

    # 最初の呼び出しでエラーを発生させ、2回目で成功させる
    guide_based_instance.groq_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limit exceeded", request=None, body=None),
        mock_response,
    ]

    result = guide_based_instance.detect_lang("some prompt")
    assert result == "en"
    assert guide_based_instance.groq_client.chat.completions.create.call_count == 2


# --- __call__ メソッドのテスト ---


def test_call_success(guide_based_instance, mocker):
    """プロンプトの書き換えが成功するケースをテストします。"""
    # detect_langメソッドをモック化して、常に'ja'を返すようにする
    mocker.patch.object(guide_based_instance, "detect_lang", return_value="ja")

    # APIからの正常な応答をシミュレート
    rewritten_prompt = "これは書き換えられたプロンプトです。"
    # src/translate.pyの実装に合わせて <instruction> タグを使用し、タグ除去ロジックをテストします
    mock_response = MagicMock()
    mock_response.choices[0].message.content = f"<instruction>{rewritten_prompt}</instruction>"
    guide_based_instance.groq_client.chat.completions.create.return_value = (
        mock_response
    )

    result = guide_based_instance("初期プロンプト")
    assert result == rewritten_prompt
    guide_based_instance.detect_lang.assert_called_once_with("初期プロンプト")


def test_call_empty_prompt(guide_based_instance):
    """空のプロンプトが渡された場合にValueErrorが発生することをテストします。"""
    with pytest.raises(ValueError, match="初期プロンプトが空です"):
        guide_based_instance("")


def test_call_api_error(guide_based_instance, mocker):
    """APIエラーが発生した場合に空文字列を返すことをテストします。"""
    mocker.patch.object(guide_based_instance, "detect_lang", return_value="en")
    guide_based_instance.groq_client.chat.completions.create.side_effect = APIError(
        "Test API Error", request=None, body=None
    )

    result = guide_based_instance("some prompt")
    assert result == ""


def test_call_rate_limit_error(guide_based_instance, mocker):
    """__call__でレート制限エラーからのリトライをテストします。"""
    mocker.patch("time.sleep")
    mocker.patch.object(guide_based_instance, "detect_lang", return_value="ja")

    rewritten_prompt = "これは書き換えられたプロンプトです。"
    mock_response = MagicMock()
    mock_response.choices[0].message.content = f"<instruction>{rewritten_prompt}</instruction>"

    guide_based_instance.groq_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limit exceeded", request=None, body=None),
        mock_response,
    ]

    result = guide_based_instance("初期プロンプト")
    assert result == rewritten_prompt
    assert guide_based_instance.groq_client.chat.completions.create.call_count == 2


# --- judge メソッドのテスト ---


def test_judge_success(guide_based_instance):
    """プロンプト候補の評価が成功するケースをテストします。"""
    candidates = ["prompt 1", "prompt 2", "prompt 3"]
    mock_response = MagicMock()
    # LLMが2番目の候補を選択したと仮定
    mock_response.choices[0].message.content = json.dumps(
        {"Preferred": "Instruction 2"}
    )
    guide_based_instance.groq_client.chat.completions.create.return_value = (
        mock_response
    )

    result = guide_based_instance.judge(candidates)
    assert result == 1  # 0-based index


def test_judge_malformed_json_with_fallback(guide_based_instance):
    """不正なJSONでも正規表現でフォールバックできるケースをテストします。"""
    candidates = ["prompt 1", "prompt 2"]
    # JSONの前後に余計なテキストが付与されているケース
    raw_content = 'Here is the result: ```json\n{"Preferred": "Instruction 1"}\n```'
    mock_response = MagicMock()
    mock_response.choices[0].message.content = raw_content
    guide_based_instance.groq_client.chat.completions.create.return_value = (
        mock_response
    )

    result = guide_based_instance.judge(candidates)
    assert result == 0


def test_judge_api_error(guide_based_instance):
    """評価中にAPIエラーが発生した場合のテスト。"""
    candidates = ["prompt 1", "prompt 2"]
    guide_based_instance.groq_client.chat.completions.create.side_effect = APIError(
        "Test API Error", request=None, body=None
    )

    result = guide_based_instance.judge(candidates)
    assert result is None


def test_judge_rate_limit_error(guide_based_instance, mocker):
    """judgeでレート制限エラーからのリトライをテストします。"""
    mocker.patch("time.sleep")
    candidates = ["prompt 1", "prompt 2"]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({"Preferred": "Instruction 1"})

    # 最初の呼び出しでエラーを発生させ、2回目で成功させる
    guide_based_instance.groq_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limit exceeded", request=None, body=None),
        mock_response,
    ]

    result = guide_based_instance.judge(candidates)
    assert result == 0
    assert guide_based_instance.groq_client.chat.completions.create.call_count == 2
