import ast
import io
import json
import operator
import os
import pathlib
import re
import time
from pathlib import Path
from typing import Any, Callable, Union

import gradio as gr
import pandas as pd
from dotenv import load_dotenv  # .envファイルから環境変数を読み込むために使用
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from sklearn.metrics import confusion_matrix  # 混同行列の計算に使用

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from app import SafeCodeExecutor

# スクリプト (calibration.py) が置かれているディレクトリの絶対パス
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# プロジェクトのルートディレクトリ (srcディレクトリの親)
_PROJECT_ROOT_DIR = os.path.dirname(_CURRENT_SCRIPT_DIR)

# promptディレクトリへのパス
_PROMPT_DIR = os.path.join(_CURRENT_SCRIPT_DIR, "prompt")

# tempディレクトリへのパス (src/temp)
_TEMP_DIR_PATH = os.path.join(_CURRENT_SCRIPT_DIR, "temp")

# 各プロンプトファイルへの絶対パス
_ERROR_ANALYSIS_PROMPT_PATH = os.path.join(_PROMPT_DIR, "error_analysis_classification.prompt")
_STEP_PROMPT_PATH = os.path.join(_PROMPT_DIR, "step_prompt_classification.prompt")
_PROMPT_GUIDE_SHORT_PATH = os.path.join(_PROMPT_DIR, "prompt_guide_short.prompt")
# 各プロンプトファイルを読み込みます
with open(_ERROR_ANALYSIS_PROMPT_PATH, encoding="utf-8") as f:
    error_analysis_prompt = f.read()
with open(_STEP_PROMPT_PATH, encoding="utf-8") as f:
    step_prompt = f.read()
with open(_PROMPT_GUIDE_SHORT_PATH, encoding="utf-8") as f:
    prompt_guide_short = f.read()


# プロンプトキャリブレーションを行うクラス
class CalibrationPrompt:
    def __init__(self) -> None:
        # metaprompt.txt への絶対パス (srcディレクトリ内にあると仮定)
        _METAPROMPT_PATH = os.path.join(_CURRENT_SCRIPT_DIR, "metaprompt.txt")
        with open(_METAPROMPT_PATH, encoding="utf-8") as f:
            self.metaprompt = f.read()
        # Groq APIキーを環境変数から取得し、クライアントを初期化します
        groq_api_key = os.getenv("GROQ_API_KEY")
        # tempディレクトリを作成 (存在しない場合のみ)
        os.makedirs(_TEMP_DIR_PATH, exist_ok=True)
        self.groq_client = Groq(api_key=groq_api_key)
        self.safe_code_executor = SafeCodeExecutor()  # SafeCodeExecutorのインスタンスを作成

    def invoke_model(self, prompt: str, model: str = "scout") -> str:
        """
        指定されたプロンプトとモデルを使用してGroq API経由でモデルを呼び出します。

        Args:
            prompt (str): モデルに送信するプロンプト。
            model (str, optional): 使用するモデルのエイリアス。現在は 'scout' のみサポート。
                                   デフォルトは 'scout'。

        Returns:
            str: モデルからの応答メッセージ。
        """

        model_id = "meta-llama/llama-4-scout-17b-16e-instruct"
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
        # Groq APIを呼び出し、チャット補完を生成します
        completion = self.groq_client.chat.completions.create(
            model=model_id,
            messages=messages, 
            max_completion_tokens=8192,
        )
        message = completion.choices[0].message.content or ""
        return message
    
    def get_output(
        self,
        prompt: str,
        dataset: Union[bytes, pd.DataFrame],
        postprocess_code: str,
        return_df: bool = False,
    ) -> Union[pd.DataFrame, gr.DownloadButton]:
        """
        データセット内の各行に対してプロンプトを実行し、後処理を適用して結果を取得します。

        Args:
            prompt (str): 実行するプロンプトテンプレート。変数プレースホルダを含むことができます。
            dataset (bytes or pd.DataFrame): CSV形式のデータセットまたはPandas DataFrame。
            postprocess_code (str): ユーザー定義の後処理Pythonコード文字列。
                                    'postprocess(llm_output)'という関数を定義する必要があります。
            return_df (bool, optional): 結果をDataFrameとして返すか、ダウンロードボタンとして返すか。
                                        デフォルトは False (ダウンロードボタン)。

        Returns:
            pd.DataFrame or gr.DownloadButton: return_dfの値に応じた結果。
        """
        if isinstance(dataset, bytes):
            data_io = io.BytesIO(dataset)
            df = pd.read_csv(data_io)
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise TypeError("dataset must be bytes or pd.DataFrame")

        # postprocess_codeからpostprocess関数を動的に取得 (安全性を強化)
        def get_postprocess_func(code_str: str) -> Callable[[Any], Any]:
            try:
                # コードをASTにパース
                tree = ast.parse(code_str, mode='exec')

                # 悪意のある可能性のある操作をチェック
                for node in ast.walk(tree):
                    # 許可するノードタイプを定義
                    allowed_node_types = self.safe_code_executor.ALLOWED_NODES + (
                        ast.FunctionDef, ast.Return, ast.Module, ast.arguments, ast.arg
                    )
                    if not isinstance(node, allowed_node_types):
                        raise ValueError(f"許可されていないASTノードタイプが含まれています: {type(node).__name__}")

                    # 危険な組み込み関数の呼び出しをチェック
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id not in self.safe_code_executor.ALLOWED_FUNCTIONS:
                            raise ValueError(f"許可されていない関数呼び出しが含まれています: {node.func.id}")
                    
                    # import文を禁止
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        raise ValueError("Import statements are not allowed in postprocess code.")

            except (SyntaxError, ValueError) as e:
                raise ValueError(f"後処理コードの検証に失敗しました: {e}")


            local_exec_context: dict[str, Any] = {}
            try:
                # 検証済みコードの実行
                exec(code_str, {"__builtins__": self.safe_code_executor.ALLOWED_FUNCTIONS}, local_exec_context)
                postprocess_func = local_exec_context.get("postprocess")
                if not callable(postprocess_func):
                    raise ValueError("postprocess関数がコード内で定義されていないか、無効です。")
                return postprocess_func
            except Exception as e:
                raise ValueError(f"後処理コードの実行中にエラーが発生しました: {e}")

        # postprocess関数を取得し、callableであることを確認
        postprocess_func = get_postprocess_func(postprocess_code)

        results = []
        for row_idx, row in df.iterrows():
            # データセットの各行から変数を抽出（'label'列を除く）
            variables = {}
            for key, value in dict(row).items():
                if key == "label":
                    continue
                variables[key] = value

            # プロンプトに変数を適用
            formatted_prompt = prompt.format(**variables)
            # モデルを呼び出して予測を取得
            predict = self.invoke_model(formatted_prompt)
            # 安全なコンテキストで後処理関数を実行
            safe_context = {"llm_output": predict, "postprocess": postprocess_func}
            result = self.safe_code_executor.execute_safe_code("postprocess(llm_output)", safe_context)

            results.append(result)

        df["predict"] = results
        # 結果をDataFrameとして返すか、ダウンロードボタンとして返すかを決定
        if return_df:
            return df
        timestr = time.strftime("%Y%m%d-%H%M%S")
        temp_file_path = os.path.join(_TEMP_DIR_PATH, f"predict_{timestr}.csv")
        df.to_csv(temp_file_path, index=False)
        return gr.DownloadButton(
            label=f"Download predict result (predict_{timestr}.csv)", value=pathlib.Path(temp_file_path), visible=True
        )

    def optimize(
        self,
        task_description: str,
        prompt: str,
        dataset: Union[bytes, pd.DataFrame],
        postprocess_code: str,
        step_num: int = 3,
    ) -> str:
        """
        指定されたステップ数だけプロンプトの最適化処理を繰り返します。

        Args:
            task_description (str): 最適化対象のタスクの説明。
            prompt (str): 初期プロンプト。
            dataset (bytes or pd.DataFrame): 評価に使用するデータセット。
            postprocess_code (str): 後処理コード。
            step_num (int, optional): 最適化のステップ（エポック）数。デフォルトは 3。

        Returns:
            str: 最適化された最終的なプロンプト。
        """
        if isinstance(dataset, bytes):
            data_io = io.BytesIO(dataset)
            df = pd.read_csv(data_io)
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise TypeError("dataset must be bytes or pd.DataFrame")

        # 初期プロンプトで出力を取得し、それを現在のデータセットとして使用
        output = self.get_output(prompt, df, postprocess_code, return_df=True)
        if not isinstance(output, pd.DataFrame):
            raise TypeError("Expected a DataFrame")
        dataset_with_predictions = output

        history: list[dict[str, Any]] = []
        for _ in range(step_num):
            # 最適化の1ステップを実行
            step_result = self.step(task_description, prompt, dataset_with_predictions, postprocess_code, history)
            prompt = step_result["cur_prompt"]
            dataset_with_predictions = step_result["dataset"]
            history = step_result["history"]
        return prompt.strip()

    def step(
        self,
        task_description: str,
        prompt: str,
        dataset: pd.DataFrame,
        postprocess_code: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # 1回の最適化ステップを実行します。エラー分析、履歴の追加、新しいプロンプトの提案を行います。
        num_errors = 5
        mean_score = self.eval_score(dataset)
        errors = self.extract_errors(dataset)
        large_error_to_str = self.large_error_to_str(errors, num_errors)
        history = self.add_history(prompt, dataset, task_description, history, mean_score, errors)
        prompt_input = {
            "original_instruction": history[-1]["prompt"].strip(),
            "task_description": task_description,
            "error_analysis": history[-1]["analysis"],
            "failure_cases": large_error_to_str,
        }
        prompt_input["labels"] = json.dumps([str(label) for label in list(dataset["label"].unique())])
        # 新しいプロンプトを提案するモデルを呼び出します
        prompt_suggestion = self.invoke_model(step_prompt.format(**prompt_input), model="scout")
        pattern = r"<new_prompt>(.*?)</new_prompt>"
        match = re.search(pattern, prompt_suggestion, re.DOTALL)
        if not match:
            raise ValueError("Could not find <new_prompt> in the model's output.")
        cur_prompt = match.group(1)
        # 新しいプロンプトで出力を取得
        output = self.get_output(cur_prompt, dataset.copy(), postprocess_code, return_df=True)
        if not isinstance(output, pd.DataFrame):
            raise TypeError("Expected a DataFrame")
        cur_dataset_with_predictions = output
        score = self.eval_score(cur_dataset_with_predictions)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        temp_file_path = os.path.join(_TEMP_DIR_PATH, f"predict_{timestr}.csv")
        cur_dataset_with_predictions.to_csv(temp_file_path, index=False)
        return {
            "cur_prompt": cur_prompt,
            "score": score,
            "explanation": prompt_suggestion,
            "dataset": cur_dataset_with_predictions,
            "history": history,
        }

    def get_eval_function(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        # 評価関数を生成します。この例では、ラベルと予測が一致するかどうかでスコアを付けます。
        def set_function_from_iterrow(func: Callable[[pd.Series], bool]) -> Callable[[pd.DataFrame], pd.DataFrame]:
            def wrapper(dataset: pd.DataFrame) -> pd.DataFrame:
                dataset["score"] = dataset.apply(func, axis=1)
                return dataset

            return wrapper

        return set_function_from_iterrow(lambda record: record["label"] == record["predict"])

    def sample_to_text(self, sample: dict[str, Any], num_errors_per_label: int = 0, is_score: bool = True) -> str:
        """
        Return a string that organize the information of from the step run for the meta-prompt
        :param sample: The eval information for specific step
        :param num_errors_per_label: The max number of large errors per class that will appear in the meta-prompt
        :param is_score: If True, add the score information to the meta-prompt
        :return: A string that contains the information of the step run
        """
        if is_score:
            return f"<example>\n<prompt_score>\n{sample['score']:.2f}\n</prompt_score>\n<prompt>\n{sample['prompt']}\n</prompt>\n<example>\n"
        else:
            return f"####\n##Prompt:\n{sample['prompt']}\n{self.large_error_to_str(sample['errors'], num_errors_per_label)}####\n "

    def large_error_to_str(self, error_df: pd.DataFrame, num_large_errors_per_label: int) -> str:
        """
        Return a string that contains the large errors
        :param error_df: A dataframe contains all the mislabeled samples
        :param num_large_errors_per_label: The (maximum) number of large errors per label
        :return: A string that contains the large errors that is used in the meta-prompt
        """
        if error_df.empty:
            return ""
        required_columns = error_df.columns.tolist()  # エラー分析に必要な列
        label_schema = error_df["label"].unique()
        error_res_df_list = []
        txt_res = ""
        for label in label_schema:
            cur_df = error_df[error_df["label"] == label]
            cur_df = cur_df.sample(frac=1.0, random_state=42)[:num_large_errors_per_label]
            # 各ラベルごとに指定された数のエラーサンプルをランダムに抽出
            error_res_df_list.append(cur_df[required_columns])
        if len(error_res_df_list) > 0:
            error_res_df = pd.concat(error_res_df_list, ignore_index=True)
            error_res_df = error_res_df.sample(frac=1.0, random_state=42)
            # 抽出されたエラーサンプルをテキスト形式に変換
            for _, row in error_res_df.iterrows():
                label = row.label
                prediction = row.predict
                Sample = ""
                for k, v in dict(row).items():
                    if k in ("label", "predict", "score"):
                        continue
                    Sample += f"{k}: {v}\n"
                Sample = Sample.strip()
                txt_res += (
                    f"<Sample>\n{Sample}\n</Sample>\n<Prediction>\n{prediction}\n</Prediction>\n<GT>\n{label}\n</GT>\n"
                )
        return txt_res.strip()

    def eval_score(self, dataset: pd.DataFrame) -> float:
        # データセットの平均スコアを計算します。
        score_func = self.get_eval_function()
        dataset = score_func(dataset)
        mean_score = dataset["score"].mean()
        return float(mean_score)

    def extract_errors(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the errors from the dataset
        :return: records that contains the errors
        """
        df = dataset
        # スコアが0.5未満のレコードをエラーとして抽出します
        err_df = df[df["score"] < 0.5].copy()
        err_df.sort_values(by=["score"], inplace=True)
        return err_df

    def add_history(
        self,
        prompt: str,
        dataset: pd.DataFrame,
        task_description: str,
        history: list[dict[str, Any]],
        mean_score: float,
        errors: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        現在のステップの情報を履歴に追加します。エラー分析も行います。

        Args:
            prompt (str): 現在のプロンプト。
            dataset (pd.DataFrame): 現在のデータセット（予測結果を含む）。
            task_description (str): タスクの説明。
            history (list): これまでの履歴のリスト。
            mean_score (float): 現在の平均スコア。
            errors (pd.DataFrame): 現在のエラーデータフレーム。
        """
        num_errors = 5
        # エラー情報を文字列に変換
        large_error_to_str = self.large_error_to_str(errors, num_errors)
        prompt_input = {
            "task_description": task_description,
            "accuracy": mean_score,
            "prompt": prompt,
            "failure_cases": large_error_to_str,
        }
        label_schema = sorted(list(dataset["label"].unique()))
        # 混同行列を計算
        conf_matrix = confusion_matrix(dataset["label"], dataset["predict"], labels=label_schema)
        conf_text = f"Confusion matrix columns:{label_schema} the matrix data:"
        for i, row in enumerate(conf_matrix):
            conf_text += f"\n{label_schema[i]}: {row}"
        prompt_input["confusion_matrix"] = conf_text
        # エラー分析プロンプトを実行
        analysis = self.invoke_model(error_analysis_prompt.format(**prompt_input), model="scout")
        pattern = r"<analysis>(.*?)</analysis>"
        match = re.search(pattern, analysis, re.DOTALL)
        if not match:
            raise ValueError("Could not find <analysis> in the model's output.")
        analysis_text = match.group(1).strip()
        # 現在の情報を履歴に追加
        history.append(
            {
                "prompt": prompt,
                "score": mean_score,
                "errors": errors,
                "confusion_matrix": conf_matrix,
                "analysis": analysis_text,
            }
        )
        return history