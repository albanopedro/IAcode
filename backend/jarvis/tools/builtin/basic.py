"""Safe tools: a calculator (no eval) and the current date and time."""

from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime
from typing import Any

from jarvis.tools.base import Param, Policy, Risk, Tool, ToolContext, ToolError, ToolResult

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "log",
        "log10",
        "log2",
        "exp",
        "floor",
        "ceil",
        "radians",
        "degrees",
        "hypot",
        "gcd",
    )
} | {"abs": abs, "round": round, "min": min, "max": max}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
MAX_EXPONENT = 1000
MAX_FACTORIAL = 500
MAX_DIGITS = 400


def safe_eval(expression: str) -> float | int:
    """Evaluate arithmetic without ``eval``: only numbers, operators and math functions."""
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    if not re.search(r"[a-z]", expression, re.IGNORECASE):
        # Brazilian decimal comma ("1,5 * 2"); with function calls a comma separates args.
        expression = re.sub(r"(\d),(\d)", r"\1.\2", expression)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"expressão inválida: {expression}") from exc

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ToolError("expoente grande demais")
            try:
                return _BINARY[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ToolError("divisão por zero") from exc
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [walk(a) for a in node.args]
            if node.func.id == "factorial":
                if len(args) != 1 or int(args[0]) != args[0] or not 0 <= args[0] <= MAX_FACTORIAL:
                    raise ToolError(f"fatorial só de inteiros entre 0 e {MAX_FACTORIAL}")
                return math.factorial(int(args[0]))
            if node.func.id in _FUNCTIONS:
                try:
                    return _FUNCTIONS[node.func.id](*args)
                except (ValueError, TypeError) as exc:
                    raise ToolError(f"{node.func.id}: {exc}") from exc
        raise ToolError("só números, + - * / // % ^, parênteses e funções matemáticas")

    result = walk(tree)
    if isinstance(result, int) and len(str(abs(result))) > MAX_DIGITS:
        raise ToolError("resultado grande demais")
    return result


class CalculatorTool(Tool):
    name = "calculator"
    title = "Calculadora"
    description = (
        "Calcula expressões matemáticas exatas (use para qualquer conta). "
        "Aceita + - * / // % ^, parênteses, sqrt, sin, cos, tan, log, log10, exp, factorial, "
        "pi, e."
    )
    params = (Param("expression", "a expressão, por exemplo 'sqrt(2) * 15 ^ 2'"),)
    risk = Risk.SAFE
    default_policy = Policy.ALLOW

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        expression = self.arg(args, "expression", max_len=300)
        result = safe_eval(expression)
        if isinstance(result, float):
            result = round(result, 12)
            if result.is_integer() and abs(result) < 1e15:
                result = int(result)
        return ToolResult(f"{expression} = {result}")


WEEKDAYS = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]


class ClockTool(Tool):
    name = "current_time"
    title = "Data e hora"
    description = "Informa a data, o dia da semana e a hora atuais do computador do usuário."
    risk = Risk.SAFE
    default_policy = Policy.ALLOW

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        now = datetime.now().astimezone()
        return ToolResult(
            f"{WEEKDAYS[now.weekday()]}, {now:%d/%m/%Y}, {now:%H:%M} "
            f"(fuso {now.tzname()}, UTC{now:%z})"
        )
