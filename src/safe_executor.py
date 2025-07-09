import ast
import logging
import operator
from typing import Any, Callable, Dict, Type, Union


class SafeCodeExecutor:
    """
    安全なPythonコード実行環境を提供するクラス。

    許可されたASTノード、関数、演算子のみを含むコードを実行します。
    これにより、悪意のあるコードの実行を防ぎ、サンドボックス化された環境を提供します。
    """

    # 許可されたASTノードのタプル。
    # これらは、コードが実行時に構築できる構文要素を制限します。
    ALLOWED_NODES = (
        ast.Expression, # 式ノード (例: `1 + 2`, `"hello"`)
        ast.Call, # 関数呼び出しノード (例: `len("abc")`)
        ast.Name, # 変数名ノード (例: `x`, `my_var`)
        ast.Load, # 変数のロード操作 (変数の値を取得する操作)
        ast.Constant, # 定数ノード (例: `1`, `"string"`, `True`, `None`)
        ast.Tuple, # タプルリテラル (例: `(1, 2)`)
        ast.List, # リストリテラル (例: `[1, 2]`)
        ast.Dict, # 辞書リテラル (例: `{"a": 1}`)
        ast.Set, # セットリテラル (例: `{1, 2}`)
        ast.Attribute, # 属性アクセス (例: `obj.attr`)
        ast.Subscript, # サブスクリプトアクセス (例: `list[0]`, `dict["key"]`)
        ast.Index, # インデックス指定 (サブスクリプト内で使用)
        ast.Slice, # スライス指定 (例: `list[1:5]`)
    )

    # 許可された組み込み関数の辞書。
    # キーは関数名、値は実際の関数オブジェクトです。
    ALLOWED_FUNCTIONS = {
        "len": len, # オブジェクトの長さを返す
        "str": str, # オブジェクトを文字列に変換する
        "int": int, # オブジェクトを整数に変換する
        "float": float, # オブジェクトを浮動小数点数に変換する
        "bool": bool, # オブジェクトをブール値に変換する
        "list": list, # イテラブルをリストに変換する
        "dict": dict, # キーワード引数から辞書を作成する
        "set": set, # イテラブルをセットに変換する
        "tuple": tuple, # イテラブルをタプルに変換する
        "min": min, # 最小値を返す
        "max": max, # 最大値を返す
        "sum": sum, # 合計を計算する
        "abs": abs, # 絶対値を返す
        "round": round, # 四捨五入する
        "range": range, # 数値のシーケンスを生成する
        "zip": zip, # 複数のイテラブルを結合する
        "map": map, # 関数をイテラブルの各要素に適用する
        "filter": filter, # 条件を満たす要素をフィルタリングする
        "sorted": sorted, # ソートされたリストを返す
        "all": all, # すべての要素が真であるかチェックする
        "any": any, # いずれかの要素が真であるかチェックする
        "getattr": getattr, # オブジェクトの属性値を取得する
        "hasattr": hasattr, # オブジェクトが指定された属性を持つかチェックする
        "isinstance": isinstance, # オブジェクトが指定されたクラスのインスタンスであるかチェックする
        "issubclass": issubclass, # クラスが別のクラスのサブクラスであるかチェックする
        "type": type, # オブジェクトの型を返す
        "print": print, # 値を出力する (デバッグ用)
    }

    # 許可された演算子の型ヒント。
    # astモジュールの演算子ノード型をUnionで結合します。
    OperatorType = Union[
        Type[ast.Add], # 加算 (+)
        Type[ast.Sub], # 減算 (-)
        Type[ast.Mult], # 乗算 (*)
        Type[ast.Div], # 除算 (/)
        Type[ast.FloorDiv], # 切り捨て除算 (//)
        Type[ast.Mod], # 剰余 (%)
        Type[ast.Pow], # べき乗 (**)
        Type[ast.LShift], # 左シフト (<<)
        Type[ast.RShift], # 右シフト (>>)
        Type[ast.BitOr], # ビットOR (|)
        Type[ast.BitXor], # ビットXOR (^)
        Type[ast.BitAnd], # ビットAND (&)
        Type[ast.USub], # 単項減算 (-)
        Type[ast.UAdd], # 単項加算 (+)
        Type[ast.Not], # 論理NOT (not)
        Type[ast.Eq], # 等しい (==)
        Type[ast.NotEq], # 等しくない (!=)
        Type[ast.Lt], # より小さい (<)
        Type[ast.LtE], # 以下 (<=)
        Type[ast.Gt], # より大きい (>)
        Type[ast.GtE], # 以上 (>=)
        Type[ast.Is], # 同一 (is)
        Type[ast.IsNot], # 同一でない (is not)
        Type[ast.In], # メンバーシップ (in)
        Type[ast.NotIn], # 非メンバーシップ (not in)
    ]

    # 許可された演算子とその対応する`operator`モジュールの関数の辞書。
    # これにより、ASTノードの演算子を実際のPython演算子にマッピングします。
    ALLOWED_OPERATORS: Dict[OperatorType, Callable[..., Any]] = {
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
        ast.NotIn: lambda a, b: not operator.contains(a, b), # `not in` のカスタム実装
    }

    def execute_safe_code(self, code_str: str, context: Dict[str, Any]) -> Any:
        """
        与えられたコード文字列を、指定されたコンテキスト内で安全に実行します。
        コードはASTを走査して検証され、許可された構文と関数のみが実行されます。

        Args:
            code_str (str): 実行するPythonコードの文字列。
            context (Dict[str, Any]): コード実行時のコンテキスト（変数など）。
                                       この辞書内の変数は、実行されるコードからアクセスできます。

        Returns:
            Any: コードの実行結果。エラーが発生した場合はNoneを返します。

        Raises:
            ValueError: 許可されていないASTノード、関数、または操作がコードに含まれている場合。
        """
        try:
            # コード文字列をAST (抽象構文木) にパースします。
            # mode="eval" は、単一の式をパースすることを示します。
            tree = ast.parse(code_str, mode="eval")

            # ASTをウォークし、各ノードを検証します。
            for node in ast.walk(tree):
                # 許可されていないASTノードタイプが含まれていないかチェック
                if not isinstance(node, self.ALLOWED_NODES):
                    raise ValueError(
                        f"許可されていないASTノードタイプが含まれています: {type(node).__name__}"
                    )
                
                # 関数呼び出しノードの場合のチェック
                if isinstance(node, ast.Call):
                    # 呼び出される関数がast.Name型であり、かつ許可された関数リストに含まれているかチェック
                    if (
                        not isinstance(node.func, ast.Name)
                        or node.func.id not in self.ALLOWED_FUNCTIONS
                    ):
                        raise ValueError(
                            f"許可されていない関数呼び出しが含まれています: {node.func.id if isinstance(node.func, ast.Name) else 'unknown'}"
                        )
                
                # 明示的に禁止されているASTノードタイプが含まれていないかチェック
                # これらは、ファイルシステムアクセス、動的なコード生成、クラス/関数定義など、
                # セキュリティリスクとなる可能性のある操作です。
                if isinstance(
                    node,
                    (
                        ast.Import, # import文
                        ast.ImportFrom, # from ... import ... 文
                        ast.Lambda, # ラムダ式 (関数定義)
                        ast.GeneratorExp, # ジェネレータ式
                        ast.ListComp, # リスト内包表記
                        ast.SetComp, # セット内包表記
                        ast.DictComp, # 辞書内包表記
                        ast.AsyncFunctionDef, # 非同期関数定義
                        ast.Await, # await式
                        ast.Yield, # yield式
                        ast.YieldFrom, # yield from式
                        ast.Starred, # *args, **kwargs のようなアンパック演算子
                        ast.AnnAssign, # 型アノテーション付き代入
                        ast.AugAssign, # 複合代入 (例: `x += 1`)
                        ast.For, # forループ
                        ast.AsyncFor, # 非同期forループ
                        ast.While, # whileループ
                        ast.If, # if文
                        ast.With, # with文
                        ast.AsyncWith, # 非同期with文
                        ast.Raise, # 例外発生
                        ast.Try, # try-except文
                        ast.Assert, # assert文
                        ast.Delete, # del文
                        ast.Pass, # pass文
                        ast.Break, # break文
                        ast.Continue, # continue文
                        ast.Global, # global宣言
                        ast.Nonlocal, # nonlocal宣言
                        ast.ClassDef, # クラス定義
                        ast.FunctionDef, # 関数定義
                    ),
                ):
                    raise ValueError(
                        f"許可されていない操作が含まれています: {type(node).__name__}"
                    )
                
                # 二項演算子または単項演算子の場合のチェック
                if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                    op_type: Type[ast.AST] = type(node.op) # 演算子のタイプを取得
                    # 演算子が許可された演算子リストに含まれているかチェック
                    if op_type not in self.ALLOWED_OPERATORS:
                        raise ValueError(
                            f"許可されていない演算子が含まれています: {op_type.__name__}"
                        )
                # 比較演算子の場合のチェック (例: `a == b`, `x > y`)
                elif isinstance(node, ast.Compare):
                    for op in node.ops: # 複数の比較演算子がある場合 (例: `1 < x < 10`)
                        op_type = type(op)
                        # 各比較演算子が許可された演算子リストに含まれているかチェック
                        if op_type not in self.ALLOWED_OPERATORS:
                            raise ValueError(
                                f"許可されていない演算子が含まれています: {op_type.__name__}"
                            )

            # eval関数に渡すグローバル名前空間を構築します。
            # __builtins__ を許可された関数のみに制限することで、危険な組み込み関数へのアクセスを防ぎます。
            safe_globals = {"__builtins__": self.ALLOWED_FUNCTIONS}
            # ユーザーが提供したコンテキストをグローバル名前空間に追加します。
            safe_globals.update(context)

            # 安全な環境でコードを実行します。
            # compile: ASTをコードオブジェクトにコンパイルします。
            # eval: コンパイルされたコードオブジェクトを実行します。
            #       第2引数と第3引数に同じ辞書を渡すことで、グローバルとローカルの名前空間を同じにします。
            return eval(compile(tree, "<string>", "eval"), safe_globals, safe_globals)
        except Exception as e:
            # コード実行中に発生したすべての例外をキャッチし、ログに記録します。
            logging.error(f"コード実行エラー: {e}")
            return None # エラー発生時はNoneを返す
