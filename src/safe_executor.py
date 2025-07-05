import ast
import logging
import operator
from typing import Any, Callable, Dict, Type, Union

class SafeCodeExecutor:
    """安全なPythonコード実行環境を提供するクラス。

    許可されたASTノード、関数、演算子のみを含むコードを実行します。
    これにより、悪意のあるコードの実行を防ぎます。
    """

    ALLOWED_NODES = (
        ast.Expression,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.Set,
        ast.Attribute,
        ast.Subscript,
        ast.Index,
        ast.Slice,
    )
    ALLOWED_FUNCTIONS = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "range": range,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "all": all,
        "any": any,
        "getattr": getattr,
        "hasattr": hasattr,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "print": print,
    }
    OperatorType = Union[
        Type[ast.Add], Type[ast.Sub], Type[ast.Mult], Type[ast.Div],
        Type[ast.FloorDiv], Type[ast.Mod], Type[ast.Pow], Type[ast.LShift],
        Type[ast.RShift], Type[ast.BitOr], Type[ast.BitXor], Type[ast.BitAnd],
        Type[ast.USub], Type[ast.UAdd], Type[ast.Not], Type[ast.Eq],
        Type[ast.NotEq], Type[ast.Lt], Type[ast.LtE], Type[ast.Gt],
        Type[ast.GtE], Type[ast.Is], Type[ast.IsNot], Type[ast.In],
        Type[ast.NotIn]
    ]
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
        ast.NotIn: lambda a, b: not operator.contains(a, b),
    }

    def execute_safe_code(self, code_str: str, context: Dict[str, Any]) -> Any:
        """
        与えられたコード文字列を、指定されたコンテキスト内で安全に実行します。

        Args:
            code_str (str): 実行するPythonコードの文字列。
            context (Dict[str, Any]): コード実行時のコンテキスト（変数など）。
        Returns: コードの実行結果。
        """
        try:
            tree = ast.parse(code_str, mode="eval")

            for node in ast.walk(tree):
                if not isinstance(node, self.ALLOWED_NODES):
                    raise ValueError(f"許可されていないASTノードタイプが含まれています: {type(node).__name__}")
                if isinstance(node, ast.Call):
                    if (
                        not isinstance(node.func, ast.Name) or node.func.id not in self.ALLOWED_FUNCTIONS
                    ):
                        raise ValueError(
                            f"許可されていない関数呼び出しが含まれています: {node.func.id if isinstance(node.func, ast.Name) else 'unknown'}"
                        )
                if isinstance(
                    node,
                    (
                        ast.Import,
                        ast.ImportFrom,
                        ast.Lambda,
                        ast.GeneratorExp,
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                        ast.AsyncFunctionDef,
                        ast.Await,
                        ast.Yield,
                        ast.YieldFrom,
                        ast.Starred,
                        ast.AnnAssign,
                        ast.AugAssign,
                        ast.For,
                        ast.AsyncFor,
                        ast.While,
                        ast.If,
                        ast.With,
                        ast.AsyncWith,
                        ast.Raise,
                        ast.Try,
                        ast.Assert,
                        ast.Delete,
                        ast.Pass,
                        ast.Break,
                        ast.Continue,
                        ast.Global,
                        ast.Nonlocal,
                        ast.ClassDef,
                        ast.FunctionDef,
                    ),
                ):
                    raise ValueError(f"許可されていない操作が含まれています: {type(node).__name__}")
                if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                    op_type: Type[ast.AST] = type(node.op)
                    if op_type not in self.ALLOWED_OPERATORS:
                        raise ValueError(f"許可されていない演算子が含まれています: {op_type.__name__}")
                elif isinstance(node, ast.Compare):
                    for op in node.ops:
                        op_type = type(op)
                        if op_type not in self.ALLOWED_OPERATORS:
                            raise ValueError(f"許可されていない演算子が含まれています: {op_type.__name__}")

            safe_globals = {"__builtins__": self.ALLOWED_FUNCTIONS}
            safe_globals.update(context)

            return eval(compile(tree, "<string>", "eval"), safe_globals, safe_globals)
        except Exception as e:
            logging.error(f"コード実行エラー: {e}")
            return None
