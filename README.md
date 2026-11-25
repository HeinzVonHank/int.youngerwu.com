# int.youngerwu.com

一个贯穿项目的路线图：从零训一个小 GPT，做推理优化，再接上 Agent harness。

单页静态站，GitHub Pages 直接从 `main` 分支根目录发布。

## 更新进度

改 `index.html` 里对应阶段的 `data-status`：

- `todo` 未开始
- `doing` 进行中
- `done` 已验证

想挂笔记或代码链接，在该阶段的 `<p class="state">` 里加 `<a href="...">笔记</a>`。

页面带 `<meta name="robots" content="noindex">`，搜索引擎不会收录；想被搜到就删掉那一行。
