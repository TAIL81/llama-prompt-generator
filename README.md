# プロンプトジェネレーター

プロンプトジェネレーターは、タスクに基づいて最適なAIプロンプトを自動生成するPythonアプリケーションです。Groq API および OpenRouter を活用し、Llama4 モデルを用いて効果的で構造化されたプロンプトを作成します。

## 主な機能

- タスクベースのプロンプト生成
- **テンプレート生成**: タスク記述と初期変数ヒントに基づいて、プロンプトテンプレートを生成
- **動的変数抽出**: テンプレートから変数を自動検出し、入力フィールドを動的に構築
- **リアルタイムバリデーション**: 変数名の形式（大文字英数字とアンダースコア）を検証し、必要に応じて変換
- **変数入力と最終プロンプト生成**: 入力された変数値をもとに最終的なAI回答を生成
- エンコードされた入力のサポート（UTF-8）
- 不適切な変数（浮遊変数）の検出と自動修正
- REST API による外部アプリケーション連携
- フロントエンド付き（HTML+JS）

## 必要要件

- Python 3.x
- Groq API キー
- OpenRouter API キー（OpenAI 互換）
- `requirements.txt` に記載された依存パッケージのインストール

## セットアップ手順

### 1. 環境変数の設定

```bash
# Windows PowerShell の場合
$env:GROQ_API_KEY="your-groq-api-key"
$env:OPENROUTER_API_KEY="your-openrouter-api-key"

# Linux/macOS の場合
export GROQ_API_KEY="your-groq-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

### 2. 仮想環境の作成（推奨）

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### フロントエンドの使用

1. アプリケーションを起動：

```bash
python app.py
```

2. フロントエンドをブラウザで開く：

```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

3. 操作手順：
   - タスクを入力（例：「Acme Dynamics社のカスタマーサクセス担当として丁寧に回答せよ」）
   - 「プロンプト生成」ボタンをクリック
   - 変数（FAQ、QUESTION など）に具体的な値を入力
   - 「最終回答を生成」でLlama4モデルによる出力を確認

### コマンドラインからの使用例

```bash
python main.py "数学の問題を解く" "difficulty,topic"
```

## Web API の利用

アプリケーションは REST API としても利用可能です。

### エンドポイント一覧

#### GET /
アプリケーションの状態確認用

**レスポンス例**
```json
{
  "status": "running",
  "service": "Prompt Generator API",
  "version": "1.0.0"
}
```

#### POST /generate_template
プロンプトテンプレート生成

**リクエスト例**
```json
{
  "task": "数学の問題を解く",
  "variables": ["difficulty", "topic"]
}
```

**レスポンス例**
```json
{
  "template": "生成されたプロンプトテンプレート",
  "variables": ["DIFFICULTY", "TOPIC"],
  "unused_variables": []
}
```

#### POST /execute_template
プロンプトを基にAI出力を生成

**リクエスト例**
```json
{
  "template": "生成済みテンプレート",
  "variables": {
    "DIFFICULTY": "中級",
    "TOPIC": "三角関数"
  }
}
```

**レスポンス例**
```json
{
  "final_output": "生成された最終回答"
}
```

## プロジェクト構成

- `main.py`: CLI ベースのテンプレート生成スクリプト
- `app.py`: REST API のサーバー実装
- `index.html`: フロントエンドの画面
- `app.js`: 入力・変数処理およびAPI連携
- `styles.css`: フロントエンドのスタイル
- `metaprompt.txt`: プロンプトテンプレートのテンプレート
- `remove_floating_variables_prompt.txt`: 浮遊変数を除去する補助プロンプト
- `requirements.txt`: 必要なパッケージのリスト

## 実装の要点

### プロンプト生成フロー

1. タスクと変数を受け取る
2. テンプレートエンジンを用いてプロンプトを作成
3. 変数の正当性を検証
4. Llama4 モデルを介してAI回答を生成

### 変数処理の特徴

- AWS 形式（{$VAR}）による動的変数の検出
- 小文字の自動大文字化と形式バリデーション
- 未使用・不正な変数の警告・除去
- 入力値のチェックと自動整形

### エラーハンドリング

- API 呼び出し時の通信エラー
- 入力形式の検証とエラーメッセージ
- テンプレート読み込み失敗時の処理

## 制限事項

- Groq および OpenRouter API の使用制限・料金体系に準拠
- 入力は UTF-8 エンコードで送信される必要あり

## ライセンス

このプロジェクトは [MITライセンス](LICENSE) の下で提供されています。

## コントリビューション

不具合報告や改善提案は GitHub Issue を通じて歓迎します。

---

より詳細な情報は、プロジェクトのドキュメントや Wiki をご参照ください。
