"""Verify every problem, then write ../problems.js.

    python3 tools/build.py

For each problem the reference solution must pass every test and the starter
code must not. Nothing is written if any problem fails that check.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import problems_1_3  # noqa: E402
import problems_4_6  # noqa: E402
from runner import run  # noqa: E402

PROBLEMS = problems_1_3.PROBLEMS + problems_4_6.PROBLEMS
REQUIRED = {"id", "stage", "level", "title", "statement", "starter", "solution", "setup", "tests"}


def main():
    failures = []
    seen = set()
    for p in PROBLEMS:
        assert set(p) == REQUIRED, (p.get("id"), set(p) ^ REQUIRED)
        assert p["id"] not in seen, p["id"]
        seen.add(p["id"])
        assert p["level"] in ("易", "中", "难"), p["id"]
        tests = [{"name": name, "code": code} for name, code in p["tests"]]

        solved = run(p["solution"], p["setup"], tests)
        bad = [t for t in solved["tests"] if not t["ok"]]
        if solved["error"] or bad:
            failures.append((p["id"], "参考答案没通过", solved["error"] or bad))

        blank = run(p["starter"], p["setup"], tests)
        if not blank["error"] and all(t["ok"] for t in blank["tests"]):
            failures.append((p["id"], "起始代码也能通过全部测试", None))

        print(f'{p["id"]:20} {len(tests)} 个测试  答案 {"通过" if not (solved["error"] or bad) else "失败"}'
              f'  起始代码挂 {sum(not t["ok"] for t in blank["tests"])} 个')

    if failures:
        print("\n没有写出 problems.js：")
        for f in failures:
            print("  ", f)
        sys.exit(1)

    out = [
        {
            "id": p["id"], "stage": p["stage"], "level": p["level"], "title": p["title"],
            "statement": p["statement"].strip(), "starter": p["starter"], "solution": p["solution"],
            "setup": p["setup"], "tests": [{"name": n, "code": c} for n, c in p["tests"]],
        }
        for p in PROBLEMS
    ]
    path = os.path.join(ROOT, "problems.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("// 由 tools/build.py 生成，不要手改。题目源文件在 tools/problems_*.py。\n")
        f.write("window.PROBLEMS = " + json.dumps(out, ensure_ascii=False, indent=1) + ";\n")
    per_stage = {s: sum(p["stage"] == s for p in PROBLEMS) for s in range(1, 7)}
    print(f"\n写出 {len(out)} 道题到 problems.js，各阶段：{per_stage}")


if __name__ == "__main__":
    main()
