# int.youngerwu.com

一个贯穿项目的路线图和练习页：从零训一个小 GPT，做推理优化，再接上 Agent harness。

静态站，GitHub Pages 直接从 `main` 分支根目录发布，push 即上线。

## 文件

| 文件 | 作用 |
|---|---|
| `index.html` | 路线图：六个阶段，每个阶段有要点、练习题列表、自测问答 |
| `solve.html` | 做题页：左边题面，右边编辑器，测试在浏览器里跑 |
| `worker.js` | 在 Web Worker 里加载 Pyodide 和 NumPy，执行提交的代码 |
| `runner.py` | 跑测试的 Python 代码，浏览器和 `tools/build.py` 共用 |
| `problems.js` | 题目数据，由 `tools/build.py` 生成，不要手改 |
| `tools/problems_*.py` | 题目源文件：题面、起始代码、参考答案、测试 |

## 加题或改题

改 `tools/problems_1_3.py` 或 `tools/problems_4_6.py`，然后：

```bash
python3 tools/build.py
```

脚本会逐题检查「参考答案通过全部测试、起始代码通不过」，都满足才写出 `problems.js`。需要本机有 NumPy。

本地预览要起一个服务器（直接双击打开 `solve.html` 时浏览器不让加载 worker）：

```bash
python3 -m http.server 8765
```

## 更新阶段进度

改 `index.html` 里对应阶段的 `data-status`：`todo` 未开始，`doing` 进行中，`done` 已验证。
这和做题记录是两回事：做题记录存在访客各自浏览器的 localStorage 里。

想挂笔记或代码链接，在该阶段的 `<p class="state">` 里加 `<a href="...">笔记</a>`。

## 其他

两个页面都带 `<meta name="robots" content="noindex">`，搜索引擎不会收录；想被搜到就删掉那一行。

做题页从 CDN 加载 Pyodide（jsdelivr）和 CodeMirror（cdnjs），版本号写在 `worker.js` 和 `solve.html` 里。
