"""Code-cell REPL that runs INSIDE the sandbox. Standard library + pandas only;
it must not import anything from the `app` package.

    python -u driver.py <dataset.parquet> <marker>

Loads the dataset into `df`, then reads one JSON request per stdin line:
    {"id": 3, "code": "df.shape", "timeout": 60}
and answers each with one stdout line, prefixed with <marker>:
    <marker>{"id": 3, "ok": true, "stdout": "(1338, 7)\\n", "error": null, ...}

Cell output is captured, so the only lines on the real stdout are protocol
lines. State persists across cells, like a notebook. As in Jupyter, a cell
whose last statement is an expression prints that expression's value.
"""
import ast
import contextlib
import io
import json
import os
import signal
import sys
import time
import traceback

OUTPUT_LIMIT = 20_000
ERROR_LIMIT = 4_000

# The protocol reads its own duplicate of stdin, and cells get an empty one:
# otherwise input() in a cell would swallow protocol lines, and the builtin
# exit() (which closes sys.stdin) would end the session.
_protocol_in = os.fdopen(os.dup(sys.stdin.fileno()), "r", encoding="utf-8")
sys.stdin = io.StringIO("")
_protocol_out = sys.stdout


class CellTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise CellTimeout()


def _emit(marker: str, payload: dict) -> None:
    _protocol_out.write(marker + json.dumps(payload) + "\n")
    _protocol_out.flush()


def _run_cell(code: str, namespace: dict) -> None:
    tree = ast.parse(code, filename="<cell>", mode="exec")
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<cell>", "exec"), namespace)
    if last_expr is not None:
        value = eval(compile(last_expr, "<cell>", "eval"), namespace)
        if value is not None:
            print(repr(value))


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters truncated] ...\n" + text[-half:], True


def main() -> None:
    data_path, marker = sys.argv[1], sys.argv[2]
    namespace: dict = {"__name__": "__main__"}
    try:
        exec(
            "import pandas as pd\n"
            "import numpy as np\n"
            "pd.set_option('display.width', 200)\n"
            "pd.set_option('display.max_columns', 60)\n"
            "pd.set_option('display.max_rows', 60)\n"
            f"df = pd.read_parquet({data_path!r})\n",
            namespace,
        )
    except BaseException:
        _emit(marker, {"id": 0, "ok": False, "stdout": "", "error": traceback.format_exc()[-ERROR_LIMIT:]})
        return
    rows, cols = namespace["df"].shape
    _emit(marker, {"id": 0, "ok": True, "stdout": f"df loaded: {rows} rows x {cols} columns\n", "error": None})

    has_alarm = hasattr(signal, "SIGALRM")  # absent on Windows (local dev); the host enforces its own deadline
    if has_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)

    for line in _protocol_in:
        if not line.strip():
            continue
        request = json.loads(line)
        timeout = int(request.get("timeout") or 60)
        buf = io.StringIO()
        error = None
        started = time.monotonic()
        if has_alarm:
            signal.alarm(timeout)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                _run_cell(request["code"], namespace)
        except CellTimeout:
            error = f"TimeoutError: cell exceeded {timeout}s and was interrupted. Variables defined before this cell are intact."
        except BaseException:  # includes SystemExit from exit(): the session must survive it
            error = traceback.format_exc()[-ERROR_LIMIT:]
        finally:
            if has_alarm:
                signal.alarm(0)
        stdout, truncated = _clip(buf.getvalue(), OUTPUT_LIMIT)
        _emit(marker, {
            "id": request["id"],
            "ok": error is None,
            "stdout": stdout,
            "error": error,
            "truncated": truncated,
            "seconds": round(time.monotonic() - started, 3),
        })


if __name__ == "__main__":
    main()
