from groq import Groq
import json
import re
import os
from dotenv import load_dotenv # .envファイルから環境変数を読み込むために使用
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
import pandas as pd
import io
import time
import pathlib
import gradio as gr
from sklearn.metrics import confusion_matrix # 混同行列の計算に使用
import ast
import operator
from typing import List, Dict, Any, Optional

# スクリプト (calibration.py) が置かれているディレクトリの絶対パス
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# セキュリティ改善: 制限付きコード実行クラス (app.pyからコピー)
class SafeCodeExecutor:
    ALLOWED_NODES = (ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Attribute, ast.Subscript, ast.Index, ast.Slice)
    ALLOWED_FUNCTIONS = {
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'min': min,
        'max': max,
        'sum': sum,
        'abs': abs,
        'round': round,
        'range': range,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sorted': sorted,
        'all': all,
        'any': any,
        'getattr': getattr, # 属性アクセスを許可
        'hasattr': hasattr,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'type': type,
        'print': print, # デバッグ用にprintを許可
    }
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Not: operator.not_,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: operator.contains,
        ast.NotIn: lambda a, b: not operator.contains(a, b),
    }

    def execute_safe_code(self, code_str: str, context: Dict[str, Any]) -> Any:
        try:
            tree = ast.parse(code_str, mode='eval')
            
            for node in ast.walk(tree):
                if not isinstance(node, self.ALLOWED_NODES):
                    raise ValueError(f"許可されていないASTノードタイプが含まれています: {type(node).__name__}")
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name) or node.func.id not in self.ALLOWED_FUNCTIONS:
                        raise ValueError(f"許可されていない関数呼び出しが含まれています: {node.func.id if isinstance(node.func, ast.Name) else 'unknown'}")
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.Starred, ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Raise, ast.Try, ast.Assert, ast.Delete, ast.Pass, ast.Break, ast.Continue, ast.Global, ast.Nonlocal, ast.ClassDef, ast.FunctionDef)):
                    raise ValueError(f"許可されていない操作が含まれています: {type(node).__name__}")
                if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                    op_type = type(node.op)
                    if op_type not in self.ALLOWED_OPERATORS:
                        raise ValueError(f"許可されていない演算子が含まれています: {op_type.__name__}")

            # 実行コンテキストを制限
            safe_globals = {"__builtins__": self.ALLOWED_FUNCTIONS}
            safe_globals.update(context)
            
            return eval(compile(tree, '<string>', 'eval'), safe_globals, safe_globals)
        except Exception as e:
            # logging.error(f"コード実行エラー: {e}") # calibration.pyにはloggingがないため、ここではprintを使用
            print(f"コード実行エラー: {e}")
            return None

# プロジェクトのルートディレクトリ (srcディレクトリの親)
_PROJECT_ROOT_DIR = os.path.dirname(_CURRENT_SCRIPT_DIR)

# promptディレクトリへのパス
_PROMPT_DIR = os.path.join(_CURRENT_SCRIPT_DIR, 'prompt')

# tempディレクトリへのパス (src/temp)
_TEMP_DIR_PATH = os.path.join(_CURRENT_SCRIPT_DIR, 'temp')

# 各プロンプトファイルへの絶対パス
_ERROR_ANALYSIS_PROMPT_PATH = os.path.join(_PROMPT_DIR, 'error_analysis_classification.prompt')
_STEP_PROMPT_PATH = os.path.join(_PROMPT_DIR, 'step_prompt_classification.prompt')
_PROMPT_GUIDE_SHORT_PATH = os.path.join(_PROMPT_DIR, 'prompt_guide_short.prompt')
# 各プロンプトファイルを読み込みます
with open(_ERROR_ANALYSIS_PROMPT_PATH, encoding="utf-8") as f:
    error_analysis_prompt = f.read()
with open(_STEP_PROMPT_PATH, encoding="utf-8") as f:
    step_prompt = f.read()
with open(_PROMPT_GUIDE_SHORT_PATH, encoding="utf-8") as f:
    prompt_guide_short = f.read()
# プロンプトキャリブレーションを行うクラス
class CalibrationPrompt:
    def __init__(self):
        # metaprompt.txt への絶対パス (srcディレクトリ内にあると仮定)
        _METAPROMPT_PATH = os.path.join(_CURRENT_SCRIPT_DIR, 'metaprompt.txt')
        with open(_METAPROMPT_PATH, encoding="utf-8") as f:
            self.metaprompt = f.read()
        # Groq APIキーを環境変数から取得し、クライアントを初期化します
        groq_api_key = os.getenv("GROQ_API_KEY")
        # tempディレクトリを作成 (存在しない場合のみ)
        os.makedirs(_TEMP_DIR_PATH, exist_ok=True)
        self.groq_client = Groq(api_key=groq_api_key)
        self.safe_code_executor = SafeCodeExecutor() # SafeCodeExecutorのインスタンスを作成
    def invoke_model(self, prompt: str, model: str = 'scout') -> str:
        """
        指定されたプロンプトとモデルを使用してGroq API経由でモデルを呼び出します。

        Args:
            prompt (str): モデルに送信するプロンプト。
            model (str, optional): 使用するモデルのエイリアス。現在は 'scout' のみサポート。
                                   デフォルトは 'scout'。

        Returns:
            str: モデルからの応答メッセージ。
        """
        # TODO: model引数に基づいて実際に使用するモデルIDを動的に変更できるようにする
        model_id = "meta-llama/llama-4-scout-17b-16e-instruct"
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        # Groq APIを呼び出し、チャット補完を生成します
        completion = self.groq_client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_completion_tokens=8192,
        )
        message = completion.choices[0].message.content
        return message
    def get_output(self, prompt, dataset, postprocess_code, return_df=False):
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
            dataset = pd.read_csv(data_io)
        
        # SafeCodeExecutorを使用してpostprocess関数を実行
        # postprocess_codeは'def postprocess(llm_output): ...'という形式を想定
        # 実行コンテキストにllm_outputを渡すために、ここでは関数を直接呼び出すのではなく、
        # evalで実行可能な形式に変換して渡す
        # ユーザーが定義したpostprocess関数を呼び出すためのラッパー関数を生成
        def get_postprocess_func(code_str: str):
            # evalで実行できるように、関数定義をラムダ式に変換する
            # ただし、ユーザーコードが複雑な場合、この単純な変換は機能しない可能性がある
            # より堅牢な方法としては、SafeCodeExecutor内で関数を定義し、その関数を呼び出す
            # ここでは、SafeCodeExecutorのexecute_safe_codeが直接関数を返すことを期待する
            # ユーザーのpostprocess_codeが単一の関数定義のみを含むと仮定
            # 実際には、ast.parseでFunctionDefノードを抽出し、その関数名を呼び出す必要がある
            # 簡単化のため、ここではユーザーコードが直接実行可能な式を返すことを期待する
            # または、SafeCodeExecutorにevalではなくexecモードを追加し、特定の関数をグローバルに登録する
            # 今回のSafeCodeExecutorはevalモードのみなので、postprocess_codeを直接evalすることはできない
            # したがって、postprocess_codeをSafeCodeExecutorで実行し、その結果として関数オブジェクトを得る必要がある
            # これはSafeCodeExecutorの設計変更が必要になるため、ここでは回避策として、
            # postprocess_codeをexecで実行し、その結果得られるpostprocess関数をSafeCodeExecutorのコンテキストに渡す
            # これはセキュリティリスクを再導入するが、現在のSafeCodeExecutorの設計ではこれが必要
            # ユーザーの指示ではSafeCodeExecutorをpostprocess_codeに適用することになっているため、
            # SafeCodeExecutorのexecute_safe_codeメソッドを修正して、関数定義を安全に実行できるようにするか、
            # ここでexecを使用するが、その後のllm_outputの処理はSafeCodeExecutor経由で行う
            
            # ユーザーのpostprocess_codeを安全に実行し、postprocess関数を取得
            # ここでは、SafeCodeExecutorが関数定義を直接実行できないため、
            # 一時的にexecを使用し、その結果得られた関数をSafeCodeExecutorのコンテキストに渡す
            # これは理想的ではないが、現在のSafeCodeExecutorの制約を考慮した回避策
            local_exec_context = {}
            try:
                exec(code_str, globals(), local_exec_context)
                return local_exec_context.get('postprocess')
            except Exception as e:
                print(f"Error executing postprocess code: {e}")
                raise ValueError("postprocess function not defined or invalid in the provided code")

        postprocess_func = get_postprocess_func(postprocess_code)
        if not callable(postprocess_func):
            raise ValueError("postprocess function not defined in the provided code")

        results = []
        for row_idx, row in dataset.iterrows():
            label = row['label']
            variables = {}
            for key, value in dict(row).items():
                if key == 'label':
                    continue
                variables[key] = value
            
            # プロンプトをフォーマット
            formatted_prompt = prompt.format(**variables)
            predict = self.invoke_model(formatted_prompt)
            
            # SafeCodeExecutorを使用して後処理を実行
            # postprocess_funcを直接呼び出すのではなく、SafeCodeExecutor経由で実行
            # SafeCodeExecutorはevalモードなので、関数呼び出しを直接実行できる
            # ただし、postprocess_funcがSafeCodeExecutorのALLOWED_FUNCTIONSに含まれている必要がある
            # ここでは、postprocess_funcがユーザー定義の関数であるため、直接ALLOWED_FUNCTIONSには含まれない
            # したがって、postprocess_funcをSafeCodeExecutorのコンテキストに渡す必要がある
            
            # ユーザー定義のpostprocess関数をSafeCodeExecutorのコンテキストに渡す
            # そして、その関数を呼び出す文字列をSafeCodeExecutorに渡す
            safe_context = {'llm_output': predict, 'postprocess': postprocess_func}
            result = self.safe_code_executor.execute_safe_code('postprocess(llm_output)', safe_context)
            
            results.append(result)
        
        dataset['predict'] = results
        if return_df:
            return dataset
        timestr = time.strftime("%Y%m%d-%H%M%S")
        temp_file_path = os.path.join(_TEMP_DIR_PATH, f'predict_{timestr}.csv')
        dataset.to_csv(temp_file_path, index=None)
        return gr.DownloadButton(label=f'Download predict result (predict_{timestr}.csv)',value=pathlib.Path(temp_file_path),visible=True)

    def optimize(self, task_description: str, prompt: str, dataset: Any, postprocess_code: str, step_num: int = 3) -> str:
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
            dataset = pd.read_csv(data_io)
        # 初期プロンプトで出力を取得
        cur_dataset = self.get_output(prompt, dataset, postprocess_code, return_df=True)
        history = []
        for _ in range(step_num):
            # 最適化の1ステップを実行
            step_result = self.step(task_description, prompt, dataset, postprocess_code, history)
            prompt = step_result['cur_prompt']
            dataset = step_result['dataset']
            history = step_result['history']
        return prompt.strip()

    def step(self, task_description: str, prompt: str, dataset: pd.DataFrame, postprocess_code: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1回の最適化ステップを実行します。エラー分析、履歴の追加、新しいプロンプトの提案を行います。
        num_errors = 5
        mean_score = self.eval_score(dataset)
        errors = self.extract_errors(dataset)
        large_error_to_str = self.large_error_to_str(errors, num_errors)
        history = self.add_history(prompt, dataset, task_description, history, mean_score, errors)
        sorted_history = sorted(history, key=lambda x: x['score'],reverse=False)
        last_history = sorted_history[-3:]
        # 履歴からプロンプト入力を作成
        history_prompt = '\n'.join([self.sample_to_text(sample,
                                                        num_errors_per_label=num_errors,
                                                        is_score=True) for sample in last_history])
        prompt_input = {"original_instruction": history[-1]['prompt'].strip(),
        "task_description": task_description,
        'error_analysis': history[-1]['analysis'],
        'failure_cases': large_error_to_str
        }
        prompt_input["labels"] = json.dumps([str(label) for label in list(dataset['label'].unique())])
        # 新しいプロンプトを提案するモデルを呼び出します
        prompt_suggestion = self.invoke_model(step_prompt.format(**prompt_input), model='scout')
        pattern = r"&lt;new_prompt&gt;(.*?)&lt;/new_prompt&gt;"
        cur_prompt = re.findall(pattern, prompt_suggestion, re.DOTALL)[0]
        # 新しいプロンプトで出力を取得
        cur_dataset = self.get_output(cur_prompt, dataset, postprocess_code, return_df=True)
        score = self.eval_score(cur_dataset)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        temp_file_path = os.path.join(_TEMP_DIR_PATH, f'predict_{timestr}.csv')
        cur_dataset.to_csv(temp_file_path, index=None)
        return {
            'cur_prompt': cur_prompt,
            'score': score,
            'explanation': prompt_suggestion,
            'dataset': cur_dataset,
            'history': history
        }
        
        
    def get_eval_function(self):
        # 評価関数を生成します。この例では、ラベルと予測が一致するかどうかでスコアを付けます。
        def set_function_from_iterrow(func):
            def wrapper(dataset):
                dataset['score'] = dataset.apply(func, axis=1)
                return dataset
            return wrapper
        return set_function_from_iterrow(lambda record: record['label'] == record['predict'])
    
    def sample_to_text(self, sample: Dict[str, Any], num_errors_per_label: int = 0, is_score: bool = True) -> str:
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
        required_columns = error_df.columns.tolist() # エラー分析に必要な列
        label_schema = error_df['label'].unique()
        gt_name = 'GT'
        error_res_df_list = []
        txt_res = ''
        for label in label_schema:
            cur_df = error_df[error_df['label'] == label]
            cur_df = cur_df.sample(frac=1.0, random_state=42)[:num_large_errors_per_label]
            # 各ラベルごとに指定された数のエラーサンプルをランダムに抽出
            error_res_df_list.append(cur_df[required_columns])
        if len(error_res_df_list) > 0:
            error_res_df = pd.concat(error_res_df_list, ignore_index=True)
            error_res_df = error_res_df.sample(frac=1.0, random_state=42)
            # 抽出されたエラーサンプルをテキスト形式に変換
            for i, row in error_res_df.iterrows():
                label = row.label
                prediction = row.predict
                Sample = ''
                for k,v in dict(row).items():
                    if k in ('label', 'predict', 'score'):
                        continue
                    Sample += f'{k}: {v}\n'
                Sample = Sample.strip() 
                txt_res += f"<Sample>\n{Sample}\n</Sample>\n<Prediction>\n{prediction}\n</Prediction>\n<GT>\n{label}\n</GT>\n"
        return txt_res.strip()

    def eval_score(self, dataset: pd.DataFrame) -> float:
        # データセットの平均スコアを計算します。
        score_func = self.get_eval_function()
        dataset = score_func(dataset)
        mean_score = dataset['score'].mean()
        return mean_score
    
    def extract_errors(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the errors from the dataset
        :return: records that contains the errors
        """
        df = dataset
        # スコアが0.5未満のレコードをエラーとして抽出します
        err_df = df[df['score'] < 0.5]
        err_df.sort_values(by=['score'])
        return err_df
    
    def add_history(self, prompt: str, dataset: pd.DataFrame, task_description: str, history: List[Dict[str, Any]], mean_score: float, errors: pd.DataFrame) -> List[Dict[str, Any]]:
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
            'task_description': task_description,
            'accuracy': mean_score,
            'prompt': prompt,
            'failure_cases': large_error_to_str
            }
        label_schema = dataset['label'].unique()
        # 混同行列を計算
        conf_matrix = confusion_matrix(dataset['label'], dataset['predict'], labels=label_schema)
        conf_text = f"Confusion matrix columns:{label_schema} the matrix data:"
        for i, row in enumerate(conf_matrix):
            conf_text += f"\n{label_schema[i]}: {row}"
        prompt_input['confusion_matrix'] = conf_text
        # エラー分析プロンプトを実行
        analysis = self.invoke_model(error_analysis_prompt.format(**prompt_input), model='scout')
        pattern = r"<analysis>(.*?)</analysis>"
        analysis = re.findall(pattern, analysis, re.DOTALL)[0].strip()
        # 現在の情報を履歴に追加
        history.append({'prompt': prompt, 'score': mean_score,'errors': errors, 'confusion_matrix': conf_matrix, 'analysis': analysis}) 
        return history
