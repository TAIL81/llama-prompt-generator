# テストスイート説明

## 🚀 テスト実行方法

```bash

# すべてのテストを実行
pytest tests/

# カバレッジ付きで実行（レポート生成）
pytest --cov=src --cov-report=html tests/

# 特定のテストファイルを指定
pytest tests/test_ape.py -v
```

## 📁 テストファイル構成

| ファイル名 | テスト対象 | 主要テストケース |
|-----------|-----------|------------------|
| `test_ape.py` | `ape.py` | プロンプト生成ロジック・API統合 |
| `test_rater.py` | `rater.py` | プロンプト評価・レートリミット処理 |
| `test_app.py` | `app.py` | アプリ設定・UIセットアップ |
| `test_metaprompt.py` | `metaprompt.py` | 基本プロンプト生成機能 |
| `test_translate.py` | `translate.py` | 言語検出・プロンプト書き換え |

## 🔍 各テスト詳細

### `test_ape.py`

- **主要テストケース**
  - Groq API設定のデフォルト値検証
  - プロンプト生成成功時の挙動
  - APIエラー時のフォールバック処理
  - デモデータ不足時の例外発生

### `test_rater.py`

- **主要テストケース**
  - プロンプト評価ロジックの正常動作
  - レートリミット時の指数バックオフリトライ
  - 不正JSONレスポンスの回復処理
  - 出力生成失敗時のハンドリング

### `test_app.py`

- **主要テストケース**
  - 翻訳ファイルの読み込み検証
  - コンポーネント遅延初期化の動作
  - Gradro UIのセットアップ確認
  - SIGINT/SIGTERMシグナルハンドリング

### `test_metaprompt.py`

- **主要テストケース**
  - 基本的なプロンプト生成機能
  - 変数展開の検証
  - 出力フォーマットの整合性チェック

### `test_translate.py`

- **主要テストケース**
  - 言語自動検出機能
  - プロンプト書き換えロジック
  - レートリミットエラーからの回復
  - XMLタグ除去機能のテスト

## ⚙️ テスト実行時の注意点

1. 外部API呼び出しはすべてモック化されています
2. テスト実行には`pytest`と`pytest-cov`が必要です
3. テストデータは各テストファイル内で動的に生成されます
4. カバレッジレポートは`htmlcov/`ディレクトリに生成されます

```bash
# 必要な依存関係をインストール
pip install pytest pytest-cov

# Windowsでのテスト実行方法
set PYTHONPATH=src
pytest tests/
```

## PowerShellでのテスト実行方法

```powershell
$env:PYTHONPATH = "src"
pytest tests/
```
