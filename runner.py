"""Runs one submission against a problem's tests.

Loaded twice: by worker.js inside Pyodide (the browser), and by tools/build.py
under CPython to verify every problem before it ships.
"""
import contextlib
import io
import traceback

USER_FILE = "<你的代码>"
TEST_FILE = "<测试>"
MAX_STDOUT = 4000


def _describe(exc, test_source=None):
    """One short message: what failed, and on which line of the user's code."""
    frames = traceback.extract_tb(exc.__traceback__)
    user_line = next((f.lineno for f in reversed(frames) if f.filename == USER_FILE), None)

    if isinstance(exc, AssertionError):
        if str(exc):
            return str(exc)
        if test_source is not None:
            line = next((f.lineno for f in reversed(frames) if f.filename == TEST_FILE), None)
            if line:
                return "没通过：" + test_source.splitlines()[line - 1].strip()
        return "断言没通过"

    if isinstance(exc, SyntaxError) and exc.filename == USER_FILE:
        return f"SyntaxError: {exc.msg}（你的代码第 {exc.lineno} 行）"

    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return f"{text}（你的代码第 {user_line} 行）" if user_line else text


def run(user_code, setup, tests):
    out = io.StringIO()
    result = {"error": None, "tests": [], "stdout": ""}
    namespace = {"__name__": "__main__"}

    try:
        with contextlib.redirect_stdout(out):
            exec(compile(user_code, USER_FILE, "exec"), namespace)
    except BaseException as exc:  # noqa: BLE001 - user code may raise anything
        result["error"] = _describe(exc)
        result["stdout"] = out.getvalue()[:MAX_STDOUT]
        return result

    for test in tests:
        source = (setup + "\n" if setup else "") + test["code"]
        scope = dict(namespace)
        try:
            import numpy
            scope.setdefault("np", numpy)
        except ImportError:
            pass
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(source, TEST_FILE, "exec"), scope)
            result["tests"].append({"name": test["name"], "ok": True, "msg": ""})
        except BaseException as exc:  # noqa: BLE001
            result["tests"].append({"name": test["name"], "ok": False, "msg": _describe(exc, source)})

    result["stdout"] = out.getvalue()[:MAX_STDOUT]
    return result
