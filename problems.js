// 由 tools/build.py 生成，不要手改。题目源文件在 tools/problems_*.py。
window.PROBLEMS = [
 {
  "id": "s1-tokenizer",
  "stage": 1,
  "level": "易",
  "title": "字符级 tokenizer",
  "statement": "<p>把文本变成整数 id，再变回来。实现三个函数：</p>\n<ul>\n  <li><code>build_vocab(text)</code>：词表是 <code>text</code> 里出现过的字符，按 <code>sorted</code> 的顺序从 0 编号。返回 <code>(stoi, itos)</code>，两个 dict，分别是字符到 id 和 id 到字符。</li>\n  <li><code>encode(text, stoi)</code>：返回 id 的 list。</li>\n  <li><code>decode(ids, itos)</code>：返回字符串。</li>\n</ul>\n<h3>示例</h3>\n<pre><code>stoi, itos = build_vocab(\"hello\")\nstoi                    # {'e': 0, 'h': 1, 'l': 2, 'o': 3}\nencode(\"hello\", stoi)   # [1, 0, 2, 2, 3]\ndecode([1, 0, 2, 2, 3], itos)   # \"hello\"</code></pre>",
  "starter": "def build_vocab(text):\n    # 返回 (stoi, itos)\n    pass\n\n\ndef encode(text, stoi):\n    pass\n\n\ndef decode(ids, itos):\n    pass\n",
  "solution": "def build_vocab(text):\n    chars = sorted(set(text))\n    stoi = {c: i for i, c in enumerate(chars)}\n    itos = {i: c for c, i in stoi.items()}\n    return stoi, itos\n\n\ndef encode(text, stoi):\n    return [stoi[c] for c in text]\n\n\ndef decode(ids, itos):\n    return \"\".join(itos[i] for i in ids)\n",
  "setup": "",
  "tests": [
   {
    "name": "词表按字符排序，从 0 编号",
    "code": "stoi, itos = build_vocab(\"hello\")\nassert stoi == {\"e\": 0, \"h\": 1, \"l\": 2, \"o\": 3}\nassert itos == {0: \"e\", 1: \"h\", 2: \"l\", 3: \"o\"}"
   },
   {
    "name": "encode 返回 id 的 list",
    "code": "stoi, itos = build_vocab(\"hello\")\nassert encode(\"hello\", stoi) == [1, 0, 2, 2, 3]"
   },
   {
    "name": "encode 再 decode 得到原文",
    "code": "s = \"to be, or not to be\"\nstoi, itos = build_vocab(s)\nassert decode(encode(s, stoi), itos) == s"
   },
   {
    "name": "空格和中文也是字符",
    "code": "s = \"小 GPT 小\"\nstoi, itos = build_vocab(s)\nassert len(stoi) == 5\nassert decode(encode(s, stoi), itos) == s"
   }
  ]
 },
 {
  "id": "s1-embedding",
  "stage": 1,
  "level": "易",
  "title": "Token 和位置 embedding",
  "statement": "<p>Embedding 是查表。实现 <code>embed(ids, tok_emb, pos_emb)</code>：</p>\n<ul>\n  <li><code>ids</code>：整数数组，形状 <code>(B, T)</code>。</li>\n  <li><code>tok_emb</code>：形状 <code>(V, C)</code>，第 i 行是 token i 的向量。</li>\n  <li><code>pos_emb</code>：形状 <code>(max_T, C)</code>，第 t 行是位置 t 的向量，保证 <code>T ≤ max_T</code>。</li>\n</ul>\n<p>返回形状 <code>(B, T, C)</code> 的数组：每个位置是它的 token 向量加上它的位置向量。</p>\n<h3>示例</h3>\n<pre><code>tok_emb = [[1, 1], [2, 2], [3, 3]]      # V = 3, C = 2\npos_emb = [[10, 0], [20, 0]]            # max_T = 2\nembed([[2, 0]], tok_emb, pos_emb)       # [[[13, 3], [21, 1]]]</code></pre>",
  "starter": "import numpy as np\n\n\ndef embed(ids, tok_emb, pos_emb):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef embed(ids, tok_emb, pos_emb):\n    ids = np.asarray(ids)\n    T = ids.shape[1]\n    return np.asarray(tok_emb)[ids] + np.asarray(pos_emb)[:T]\n",
  "setup": "tok = np.arange(12, dtype=float).reshape(4, 3)\npos = np.array([[100.0, 0, 0], [200.0, 0, 0], [300.0, 0, 0]])\nids = np.array([[0, 2], [3, 3]])",
  "tests": [
   {
    "name": "输出形状是 (B, T, C)",
    "code": "assert embed(ids, tok, pos).shape == (2, 2, 3)"
   },
   {
    "name": "token 向量加位置向量",
    "code": "out = embed(ids, tok, pos)\nassert np.allclose(out[0, 0], [100, 1, 2])\nassert np.allclose(out[0, 1], [206, 7, 8])\nassert np.allclose(out[1, 0], [109, 10, 11])"
   },
   {
    "name": "同一个 token 在不同位置得到不同向量",
    "code": "out = embed(ids, tok, pos)\nassert not np.allclose(out[1, 0], out[1, 1])\nassert np.allclose(out[1, 1] - out[1, 0], pos[1] - pos[0])"
   },
   {
    "name": "T 等于 max_T 时也能用",
    "code": "out = embed(np.array([[1, 1, 1]]), tok, pos)\nassert out.shape == (1, 3, 3)\nassert np.allclose(out[0, 2], tok[1] + pos[2])"
   }
  ]
 },
 {
  "id": "s1-attention",
  "stage": 1,
  "level": "中",
  "title": "Causal self-attention",
  "statement": "<p>实现单头的 causal self-attention：<code>causal_attention(x, Wq, Wk, Wv)</code>。</p>\n<ul>\n  <li><code>x</code>：形状 <code>(B, T, C)</code>。</li>\n  <li><code>Wq</code>、<code>Wk</code>、<code>Wv</code>：形状 <code>(C, D)</code>。<code>q = x @ Wq</code>，k 和 v 同理。</li>\n</ul>\n<p>步骤：</p>\n<ol>\n  <li>分数 <code>scores = q @ kᵀ / √D</code>，形状 <code>(B, T, T)</code>。</li>\n  <li>位置 t 只能看位置 ≤ t：把 <code>scores[:, t, s]</code> 里 <code>s &gt; t</code> 的部分设成 −∞。</li>\n  <li>对最后一维做 softmax，得到权重 <code>w</code>。</li>\n  <li>输出 <code>out = w @ v</code>，形状 <code>(B, T, D)</code>。</li>\n</ol>\n<p>返回 <code>(out, w)</code>。</p>\n<h3>示例</h3>\n<pre><code># Wq 和 Wk 全为 0 时所有分数都是 0，\n# 位置 t 对前 t+1 个位置的权重都是 1 / (t+1)：\nw[0]  # [[1,   0,   0  ],\n      #  [1/2, 1/2, 0  ],\n      #  [1/3, 1/3, 1/3]]</code></pre>",
  "starter": "import numpy as np\n\n\ndef causal_attention(x, Wq, Wk, Wv):\n    # 返回 (out, w)\n    pass\n",
  "solution": "import numpy as np\n\n\ndef causal_attention(x, Wq, Wk, Wv):\n    T = x.shape[1]\n    q, k, v = x @ Wq, x @ Wk, x @ Wv\n    scores = q @ k.transpose(0, 2, 1) / np.sqrt(k.shape[-1])\n    future = np.triu(np.ones((T, T), dtype=bool), k=1)\n    scores = np.where(future, -np.inf, scores)\n    scores = scores - scores.max(-1, keepdims=True)      # 数值稳定\n    w = np.exp(scores)\n    w = w / w.sum(-1, keepdims=True)\n    return w @ v, w\n",
  "setup": "rng = np.random.default_rng(0)\nx = rng.normal(size=(2, 5, 4))\nWq, Wk, Wv = (rng.normal(size=(4, 3)) for _ in range(3))",
  "tests": [
   {
    "name": "输出和权重的形状",
    "code": "out, w = causal_attention(x, Wq, Wk, Wv)\nassert out.shape == (2, 5, 3)\nassert w.shape == (2, 5, 5)"
   },
   {
    "name": "每一行权重的和是 1",
    "code": "out, w = causal_attention(x, Wq, Wk, Wv)\nassert np.allclose(w.sum(-1), 1)"
   },
   {
    "name": "未来位置的权重是 0",
    "code": "out, w = causal_attention(x, Wq, Wk, Wv)\nassert np.allclose(np.triu(w[0], k=1), 0), \"上三角应该全是 0\"\nassert np.allclose(w[:, 0, 0], 1), \"位置 0 只能看自己\""
   },
   {
    "name": "分数全相等时是前缀平均",
    "code": "Z = np.zeros((4, 4))\nout, w = causal_attention(x, Z, Z, np.eye(4))\nfor t in range(5):\n    assert np.allclose(w[0, t, : t + 1], 1 / (t + 1))\n    assert np.allclose(out[0, t], x[0, : t + 1].mean(0))"
   },
   {
    "name": "分数除以了 √D",
    "code": "x2 = np.array([[[1.0, 0, 0, 0], [2.0, 0, 0, 0]]])\nout, w = causal_attention(x2, np.eye(4), np.eye(4), np.eye(4))\nassert np.allclose(w[0, 1], [0.26894142, 0.73105858]), \"位置 1 的分数应该是 [1, 2]，不是 [2, 4]\""
   },
   {
    "name": "改后面的输入，前面的输出不变",
    "code": "out, _ = causal_attention(x, Wq, Wk, Wv)\nx2 = x.copy()\nx2[:, 3:] += 1.0\nout2, _ = causal_attention(x2, Wq, Wk, Wv)\nassert np.allclose(out[:, :3], out2[:, :3])\nassert not np.allclose(out[:, 3:], out2[:, 3:])"
   }
  ]
 },
 {
  "id": "s1-block",
  "stage": 1,
  "level": "中",
  "title": "Pre-norm block：归一化和残差",
  "statement": "<p>实现两个函数。</p>\n<p><code>layer_norm(x, gamma, beta, eps=1e-5)</code>：在最后一维上归一化。</p>\n<pre><code>y = gamma * (x - mean) / sqrt(var + eps) + beta</code></pre>\n<p><code>mean</code> 和 <code>var</code> 都只在最后一维上算，所以每个位置各自归一化，互不影响。</p>\n<p><code>block(x, attn, mlp, g1, b1, g2, b2)</code>：一个 pre-norm 的 Transformer block。<code>attn</code> 和 <code>mlp</code> 是传进来的函数，输入输出形状相同。</p>\n<pre><code>x = x + attn(layer_norm(x, g1, b1))\nx = x + mlp(layer_norm(x, g2, b2))</code></pre>\n<p>注意两点：归一化在子层前面；残差加的是没归一化的 <code>x</code>。</p>\n<h3>示例</h3>\n<pre><code>layer_norm(np.array([[1., 2., 3.]]), 1, 0)   # [[-1.2247, 0, 1.2247]]</code></pre>",
  "starter": "import numpy as np\n\n\ndef layer_norm(x, gamma, beta, eps=1e-5):\n    pass\n\n\ndef block(x, attn, mlp, g1, b1, g2, b2):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef layer_norm(x, gamma, beta, eps=1e-5):\n    mean = x.mean(-1, keepdims=True)\n    var = x.var(-1, keepdims=True)\n    return gamma * (x - mean) / np.sqrt(var + eps) + beta\n\n\ndef block(x, attn, mlp, g1, b1, g2, b2):\n    x = x + attn(layer_norm(x, g1, b1))\n    x = x + mlp(layer_norm(x, g2, b2))\n    return x\n",
  "setup": "rng = np.random.default_rng(0)\nx = rng.normal(size=(2, 3, 8)) * 3 + 1\none, zero = np.ones(8), np.zeros(8)\ndef _ln(a):\n    return (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)",
  "tests": [
   {
    "name": "layer_norm 之后最后一维均值 0、方差 1",
    "code": "y = layer_norm(x, one, zero)\nassert y.shape == x.shape\nassert np.allclose(y.mean(-1), 0, atol=1e-6)\nassert np.allclose(y.var(-1), 1, atol=1e-3)"
   },
   {
    "name": "gamma 缩放，beta 平移",
    "code": "y = layer_norm(x, one, zero)\nassert np.allclose(layer_norm(x, 2 * one, 3 * one), 2 * y + 3)"
   },
   {
    "name": "每个位置各自归一化",
    "code": "y = layer_norm(x, one, zero)\nx2 = x.copy()\nx2[0, 0] += 10\nassert np.allclose(layer_norm(x2, one, zero)[0, 1], y[0, 1])"
   },
   {
    "name": "子层输出 0 时，block 原样返回 x",
    "code": "out = block(x, lambda h: 0 * h, lambda h: 0 * h, one, zero, one, zero)\nassert np.allclose(out, x)"
   },
   {
    "name": "归一化在子层前面，残差加的是原来的 x",
    "code": "out = block(x, lambda h: h, lambda h: 2 * h, one, zero, one, zero)\nx1 = x + _ln(x)\nassert np.allclose(out, x1 + 2 * _ln(x1), atol=1e-6)"
   },
   {
    "name": "两次归一化各用各的参数",
    "code": "out = block(x, lambda h: h, lambda h: h, one, zero, 3 * one, one)\nx1 = x + _ln(x)\nassert np.allclose(out, x1 + 3 * _ln(x1) + 1, atol=1e-6)"
   }
  ]
 },
 {
  "id": "s1-causal-check",
  "stage": 1,
  "level": "中",
  "title": "验：改未来 token，前面的 logits 不变",
  "statement": "<p>这是第 1 步的验收。给你一个模型的 <code>forward</code>，判断它是不是 causal 的。</p>\n<p>实现 <code>check_causal(forward, vocab_size, T=12, t=6, seed=0)</code>：</p>\n<ol>\n  <li>用 <code>np.random.default_rng(seed)</code> 生成一条形状 <code>(1, T)</code> 的随机 token 序列 <code>x</code>。</li>\n  <li>复制一份 <code>x2</code>，把位置 <code>t</code> 及之后的<strong>每个</strong> token 都换成另一个 token。</li>\n  <li>分别算 <code>forward(x)</code> 和 <code>forward(x2)</code>，形状都是 <code>(1, T, V)</code>。</li>\n  <li>位置 <code>&lt; t</code> 的 logits 完全相同，<strong>并且</strong>位置 <code>≥ t</code> 的 logits 有变化，才返回 <code>True</code>。</li>\n</ol>\n<p>第二个条件不能省。一个根本不看输入的模型，改什么它的输出都不变，只查第一个条件会把它放过去。</p>\n<h3>示例</h3>\n<pre><code>check_causal(causal_model, vocab_size=11)     # True\ncheck_causal(leaky_model, vocab_size=11)      # False：前面的位置读到了后面\ncheck_causal(constant_model, vocab_size=11)   # False：模型没在看输入</code></pre>",
  "starter": "import numpy as np\n\n\ndef check_causal(forward, vocab_size, T=12, t=6, seed=0):\n    # forward(ids) -> logits，ids 形状 (1, T)，logits 形状 (1, T, V)\n    pass\n",
  "solution": "import numpy as np\n\n\ndef check_causal(forward, vocab_size, T=12, t=6, seed=0):\n    rng = np.random.default_rng(seed)\n    x = rng.integers(vocab_size, size=(1, T))\n    x2 = x.copy()\n    x2[:, t:] = (x[:, t:] + 1) % vocab_size          # 每个都换成不同的 token\n    a, b = forward(x), forward(x2)\n    past_same = np.allclose(a[:, :t], b[:, :t], atol=1e-6)\n    future_changed = not np.allclose(a[:, t:], b[:, t:], atol=1e-6)\n    return bool(past_same and future_changed)\n",
  "setup": "_rng = np.random.default_rng(1)\nV = 11\n_E = _rng.normal(size=(V, 8))\n_W = _rng.normal(size=(8, V))\n\ndef causal(ids):                       # 每个位置是它之前所有 token 的平均\n    h = np.cumsum(_E[ids], axis=1) / np.arange(1, ids.shape[1] + 1)[None, :, None]\n    return h @ _W\n\ndef leaky(ids):                        # 每个位置都读了整条序列\n    h = _E[ids]\n    return (h.mean(1, keepdims=True) + 0 * h) @ _W\n\ndef blind(ids):                        # 完全不看输入\n    return np.zeros(ids.shape + (V,))\n\ndef sneaky(ids):                       # 只有位置 0 偷看了最后一个 token\n    out = causal(ids).copy()\n    out[:, 0] += _E[ids[:, -1]] @ _W\n    return out",
  "tests": [
   {
    "name": "causal 的模型通过",
    "code": "assert check_causal(causal, V)"
   },
   {
    "name": "每个位置都读整条序列的模型不通过",
    "code": "assert not check_causal(leaky, V)"
   },
   {
    "name": "不看输入的模型不通过",
    "code": "assert not check_causal(blind, V), \"只查了「前面不变」，没查「后面有变化」\""
   },
   {
    "name": "只漏了一个位置也要查出来",
    "code": "assert not check_causal(sneaky, V)"
   },
   {
    "name": "t 取别的值也成立",
    "code": "assert check_causal(causal, V, T=10, t=1)\nassert check_causal(causal, V, T=10, t=9)\nassert not check_causal(leaky, V, T=10, t=9)"
   }
  ]
 },
 {
  "id": "s2-batches",
  "stage": 2,
  "level": "易",
  "title": "切分数据，构造 next-token 样本",
  "statement": "<p>训练语言模型之前要做两件事：留出一部分数据，再把剩下的切成 (输入, 目标) 对。</p>\n<p><code>split(ids, val_frac)</code>：把 token 序列按顺序切成两段，返回 <code>(train, val)</code>。留出集是<strong>最后</strong> <code>int(len(ids) * val_frac)</code> 个 token，不打乱。</p>\n<p><code>make_examples(ids, T)</code>：返回 <code>(x, y)</code>，两个形状 <code>(N, T)</code> 的数组，<code>N = len(ids) − T</code>。</p>\n<pre><code>x[i] = ids[i     : i + T]\ny[i] = ids[i + 1 : i + T + 1]</code></pre>\n<p><code>y</code> 就是 <code>x</code> 右移一位：<code>y[i][j]</code> 是 <code>x[i][j]</code> 的下一个 token。序列太短（<code>N ≤ 0</code>）时返回两个形状 <code>(0, T)</code> 的数组。</p>\n<h3>示例</h3>\n<pre><code>make_examples([0, 1, 2, 3, 4], T=3)\n# x = [[0, 1, 2], [1, 2, 3]]\n# y = [[1, 2, 3], [2, 3, 4]]</code></pre>",
  "starter": "import numpy as np\n\n\ndef split(ids, val_frac):\n    # 返回 (train, val)\n    pass\n\n\ndef make_examples(ids, T):\n    # 返回 (x, y)，形状都是 (N, T)\n    pass\n",
  "solution": "import numpy as np\n\n\ndef split(ids, val_frac):\n    n_val = int(len(ids) * val_frac)\n    cut = len(ids) - n_val\n    return ids[:cut], ids[cut:]\n\n\ndef make_examples(ids, T):\n    ids = np.asarray(ids)\n    n = len(ids) - T\n    if n <= 0:\n        empty = np.zeros((0, T), dtype=int)\n        return empty, empty.copy()\n    x = np.stack([ids[i : i + T] for i in range(n)])\n    y = np.stack([ids[i + 1 : i + T + 1] for i in range(n)])\n    return x, y\n",
  "setup": "",
  "tests": [
   {
    "name": "留出集是最后一段，顺序不变",
    "code": "train, val = split(list(range(10)), 0.2)\nassert list(train) == [0, 1, 2, 3, 4, 5, 6, 7]\nassert list(val) == [8, 9]"
   },
   {
    "name": "val_frac 为 0 时留出集为空",
    "code": "train, val = split(list(range(10)), 0.0)\nassert len(train) == 10 and len(val) == 0"
   },
   {
    "name": "x 和 y 的形状与内容",
    "code": "x, y = make_examples(np.arange(6), 3)\nassert np.array_equal(x, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])\nassert np.array_equal(y, [[1, 2, 3], [2, 3, 4], [3, 4, 5]])"
   },
   {
    "name": "y 是 x 右移一位",
    "code": "ids = np.random.default_rng(0).integers(50, size=40)\nx, y = make_examples(ids, 8)\nassert x.shape == y.shape == (32, 8)\nassert np.array_equal(x[:, 1:], y[:, :-1])"
   },
   {
    "name": "序列太短时返回空数组",
    "code": "x, y = make_examples(np.arange(3), 3)\nassert np.asarray(x).shape == (0, 3) and np.asarray(y).shape == (0, 3)"
   },
   {
    "name": "训练样本里不出现留出集的 token",
    "code": "train, val = split(np.arange(20), 0.25)\nx, y = make_examples(train, 4)\nassert y.max() == 14 and np.asarray(val).min() == 15"
   }
  ]
 },
 {
  "id": "s2-cross-entropy",
  "stage": 2,
  "level": "易",
  "title": "交叉熵，以及初始 loss",
  "statement": "<p>Next-token loss 就是每个位置上一次 V 选 1 的分类。实现 <code>cross_entropy(logits, targets)</code>：</p>\n<ul>\n  <li><code>logits</code>：形状 <code>(N, V)</code>。</li>\n  <li><code>targets</code>：形状 <code>(N,)</code>，每个是正确类别的下标。</li>\n</ul>\n<p>返回所有样本的平均负对数似然：</p>\n<pre><code>loss = −mean( log softmax(logits)[i, targets[i]] )</code></pre>\n<p>要求数值稳定：logits 里出现 1000 这样的数也不能得到 <code>inf</code> 或 <code>nan</code>。常用办法是先减去每行的最大值。</p>\n<h3>示例</h3>\n<pre><code>cross_entropy(np.zeros((5, 7)), [0, 1, 2, 3, 4])   # ln 7 ≈ 1.9459</code></pre>\n<p>这也是随机初始化的模型应该给出的 loss：预测接近均匀时，loss 接近 <code>ln V</code>。</p>",
  "starter": "import numpy as np\n\n\ndef cross_entropy(logits, targets):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef cross_entropy(logits, targets):\n    logits = np.asarray(logits, dtype=float)\n    targets = np.asarray(targets)\n    z = logits - logits.max(axis=1, keepdims=True)\n    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))\n    return float(-logp[np.arange(len(targets)), targets].mean())\n",
  "setup": "",
  "tests": [
   {
    "name": "均匀预测时等于 ln V",
    "code": "loss = cross_entropy(np.zeros((5, 7)), np.array([0, 1, 2, 3, 4]))\nassert abs(loss - np.log(7)) < 1e-9"
   },
   {
    "name": "一个算得出来的例子",
    "code": "loss = cross_entropy(np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([0, 1]))\nassert abs(loss - 0.2200949) < 1e-6"
   },
   {
    "name": "是平均，不是求和",
    "code": "one = cross_entropy(np.array([[1.0, 2.0, 3.0]]), np.array([0]))\nmany = cross_entropy(np.array([[1.0, 2.0, 3.0]] * 4), np.array([0, 0, 0, 0]))\nassert abs(one - many) < 1e-9"
   },
   {
    "name": "很大的 logits 也不溢出",
    "code": "loss = cross_entropy(np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.array([0, 1]))\nassert np.isfinite(loss) and loss < 1e-6"
   },
   {
    "name": "自信地答错，loss 很大但有限",
    "code": "loss = cross_entropy(np.array([[0.0, 1000.0]]), np.array([0]))\nassert np.isfinite(loss) and abs(loss - 1000) < 1e-6"
   }
  ]
 },
 {
  "id": "s2-adamw",
  "stage": 2,
  "level": "中",
  "title": "AdamW 的一步",
  "statement": "<p>实现 <code>adamw_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01)</code>，返回更新后的 <code>(w, m, v)</code>。<code>t</code> 从 1 开始计数。</p>\n<pre><code>m = b1 * m + (1 − b1) * g\nv = b2 * v + (1 − b2) * g²\nm̂ = m / (1 − b1ᵗ)\nv̂ = v / (1 − b2ᵗ)\nw = w − lr * ( m̂ / (√v̂ + eps) + wd * w )</code></pre>\n<p>最后一行是 AdamW 和「Adam 加 L2」的区别：<code>wd * w</code> 直接加在更新量上，不进 <code>g</code>，所以不会被 <code>√v̂</code> 缩放。</p>\n<h3>示例</h3>\n<pre><code># 第一步、m 和 v 从 0 开始时，m̂ = g，v̂ = g²，更新量约等于 lr * sign(g)\nadamw_step(w=[1., -2.], g=[0.5, -3.], m=0, v=0, t=1, lr=0.1, wd=0)\n# w ≈ [0.9, -1.9]</code></pre>",
  "starter": "import numpy as np\n\n\ndef adamw_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):\n    # 返回 (w, m, v)\n    pass\n",
  "solution": "import numpy as np\n\n\ndef adamw_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):\n    m = b1 * m + (1 - b1) * g\n    v = b2 * v + (1 - b2) * g * g\n    m_hat = m / (1 - b1 ** t)\n    v_hat = v / (1 - b2 ** t)\n    w = w - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * w)\n    return w, m, v\n",
  "setup": "Z = np.zeros(2)",
  "tests": [
   {
    "name": "第一步的更新量约等于 lr * sign(g)",
    "code": "w, m, v = adamw_step(np.array([1.0, -2.0]), np.array([0.5, -3.0]), Z, Z, t=1, lr=0.1, wd=0.0)\nassert np.allclose(w, [0.9, -1.9], atol=1e-6)"
   },
   {
    "name": "m 和 v 是梯度的滑动平均",
    "code": "w, m, v = adamw_step(np.array([1.0, -2.0]), np.array([0.5, -3.0]), Z, Z, t=1, lr=0.1, wd=0.0)\nassert np.allclose(m, [0.05, -0.3])\nassert np.allclose(v, [0.00025, 0.009])"
   },
   {
    "name": "梯度为 0 时只剩下衰减",
    "code": "w, m, v = adamw_step(np.array([2.0, -4.0]), Z, Z, Z, t=1, lr=0.1, wd=0.5)\nassert np.allclose(w, [1.9, -3.8])\nassert np.allclose(m, 0) and np.allclose(v, 0)"
   },
   {
    "name": "衰减不经过 √v̂ 的缩放",
    "code": "w, m, v = adamw_step(np.array([10.0]), np.array([1.0]), np.zeros(1), np.zeros(1), t=1, lr=0.1, wd=0.1)\nassert np.allclose(w, [9.8], atol=1e-6), \"得到 9.9 说明把 wd * w 加进了梯度，那是 L2，不是 AdamW\""
   },
   {
    "name": "第二步用到了偏差修正",
    "code": "w, m, v = adamw_step(np.array([0.0]), np.array([1.0]), np.zeros(1), np.zeros(1), t=1, lr=0.1, wd=0.0)\nw, m, v = adamw_step(w, np.array([3.0]), m, v, t=2, lr=0.1, wd=0.0)\nassert np.allclose(m, [0.39]) and np.allclose(v, [0.009999])\nassert np.allclose(w, [-0.1917781], atol=1e-6)"
   }
  ]
 },
 {
  "id": "s2-lr-schedule",
  "stage": 2,
  "level": "易",
  "title": "Warmup 加 cosine 的学习率",
  "statement": "<p>实现 <code>lr_at(step, max_lr, warmup, total, min_lr=0.0)</code>，返回第 <code>step</code> 步（从 0 开始）用的学习率。</p>\n<ul>\n  <li><strong>Warmup</strong>（<code>step &lt; warmup</code>）：线性上升，<code>max_lr * (step + 1) / warmup</code>。最后一步 warmup 正好到 <code>max_lr</code>。</li>\n  <li><strong>Cosine</strong>（<code>warmup ≤ step &lt; total</code>）：\n    <pre><code>progress = (step − warmup) / (total − warmup)\nlr = min_lr + 0.5 * (max_lr − min_lr) * (1 + cos(π * progress))</code></pre>\n  </li>\n  <li><strong>结束之后</strong>（<code>step ≥ total</code>）：<code>min_lr</code>。</li>\n</ul>\n<h3>示例</h3>\n<pre><code>lr_at(0,  max_lr=1.0, warmup=10, total=100)   # 0.1\nlr_at(9,  max_lr=1.0, warmup=10, total=100)   # 1.0\nlr_at(55, max_lr=1.0, warmup=10, total=100)   # 0.5，cosine 段的正中间</code></pre>",
  "starter": "import math\n\n\ndef lr_at(step, max_lr, warmup, total, min_lr=0.0):\n    pass\n",
  "solution": "import math\n\n\ndef lr_at(step, max_lr, warmup, total, min_lr=0.0):\n    if step < warmup:\n        return max_lr * (step + 1) / warmup\n    if step >= total:\n        return min_lr\n    progress = (step - warmup) / (total - warmup)\n    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))\n",
  "setup": "",
  "tests": [
   {
    "name": "warmup 线性上升",
    "code": "assert abs(lr_at(0, 1.0, 10, 100) - 0.1) < 1e-12\nassert abs(lr_at(4, 1.0, 10, 100) - 0.5) < 1e-12\nassert abs(lr_at(9, 1.0, 10, 100) - 1.0) < 1e-12"
   },
   {
    "name": "cosine 段从 max_lr 开始",
    "code": "assert abs(lr_at(10, 1.0, 10, 100) - 1.0) < 1e-12"
   },
   {
    "name": "cosine 段的中点是两端的平均",
    "code": "assert abs(lr_at(55, 1.0, 10, 100) - 0.5) < 1e-12\nassert abs(lr_at(55, 1.0, 10, 100, min_lr=0.1) - 0.55) < 1e-12"
   },
   {
    "name": "结束之后停在 min_lr",
    "code": "assert abs(lr_at(100, 1.0, 10, 100, min_lr=0.1) - 0.1) < 1e-12\nassert abs(lr_at(5000, 1.0, 10, 100, min_lr=0.1) - 0.1) < 1e-12"
   },
   {
    "name": "warmup 之后只降不升",
    "code": "lrs = [lr_at(s, 3e-4, 10, 100, min_lr=1e-5) for s in range(10, 101)]\nassert all(a >= b for a, b in zip(lrs, lrs[1:]))\nassert abs(lrs[0] - 3e-4) < 1e-12 and abs(lrs[-1] - 1e-5) < 1e-12"
   }
  ]
 },
 {
  "id": "s2-overfit",
  "stage": 2,
  "level": "难",
  "title": "验：过拟合一个 batch",
  "statement": "<p>这是第 2 步的验收：手写一个最小的训练循环，让模型把一个 batch 背下来。</p>\n<p>模型是 bigram：一张 <code>(V, V)</code> 的表 <code>W</code>，输入 token <code>a</code> 时，下一个 token 的 logits 就是 <code>W[a]</code>。</p>\n<p>实现 <code>train(x, y, V, steps=300, lr=1.0, seed=0)</code>，返回 <code>(W, losses)</code>：</p>\n<ol>\n  <li>初始化 <code>W = np.random.default_rng(seed).normal(0, 0.01, size=(V, V))</code>。</li>\n  <li>把 <code>x</code>、<code>y</code> 拉平成长度 N 的一维数组。每一步：\n    <ul>\n      <li><code>logits = W[x]</code>，形状 <code>(N, V)</code>，算平均交叉熵，把它记进 <code>losses</code>（更新之前的 loss）。</li>\n      <li>对 logits 的梯度是 <code>(softmax − onehot(y)) / N</code>。</li>\n      <li>把这些梯度按行加回 <code>W</code> 对应的行：同一个 token 在 batch 里出现多次时梯度要累加，用 <code>np.add.at(dW, x, dlogits)</code>。</li>\n      <li><code>W −= lr * dW</code>。</li>\n    </ul>\n  </li>\n</ol>\n<p><code>losses</code> 是长度为 <code>steps</code> 的 list。</p>\n<h3>要看到的三件事</h3>\n<ul>\n  <li>第一个 loss 接近 <code>ln V</code>。</li>\n  <li>每个 token 只有一种续写时，loss 降到接近 0。</li>\n  <li>同一个 token 有两种续写、各占一半时，loss 停在 <code>ln 2</code>，降不下去。这不是 bug。</li>\n</ul>",
  "starter": "import numpy as np\n\n\ndef train(x, y, V, steps=300, lr=1.0, seed=0):\n    # 返回 (W, losses)\n    pass\n",
  "solution": "import numpy as np\n\n\ndef train(x, y, V, steps=300, lr=1.0, seed=0):\n    W = np.random.default_rng(seed).normal(0, 0.01, size=(V, V))\n    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()\n    N = len(x)\n    rows = np.arange(N)\n    losses = []\n    for _ in range(steps):\n        logits = W[x]\n        z = logits - logits.max(axis=1, keepdims=True)\n        p = np.exp(z)\n        p /= p.sum(axis=1, keepdims=True)\n        losses.append(float(-np.log(p[rows, y]).mean()))\n\n        dlogits = p\n        dlogits[rows, y] -= 1\n        dlogits /= N\n        dW = np.zeros_like(W)\n        np.add.at(dW, x, dlogits)\n        W -= lr * dW\n    return W, losses\n",
  "setup": "V = 8\nseq = np.tile(np.arange(V), 5)                 # 0 1 2 ... 7 0 1 2 ...\nx = seq[:-1].reshape(3, 13)\ny = seq[1:].reshape(3, 13)                     # 每个 token 的下一个都是 (token + 1) % V\n\ndef _ce(logits, t):\n    z = logits - logits.max(1, keepdims=True)\n    logp = z - np.log(np.exp(z).sum(1, keepdims=True))\n    return float(-logp[np.arange(len(t)), t].mean())",
  "tests": [
   {
    "name": "返回值的形状",
    "code": "W, losses = train(x, y, V, steps=20)\nassert W.shape == (V, V)\nassert len(losses) == 20"
   },
   {
    "name": "第一个 loss 接近 ln V",
    "code": "W, losses = train(x, y, V, steps=5)\nassert abs(losses[0] - np.log(V)) < 0.05, f\"第一个 loss 是 {losses[0]:.3f}，应接近 {np.log(V):.3f}\""
   },
   {
    "name": "能把这个 batch 背下来",
    "code": "W, losses = train(x, y, V, steps=300, lr=10.0)\nassert losses[-1] < 0.05, f\"最后的 loss 是 {losses[-1]:.3f}\"\nassert losses[-1] < losses[0]"
   },
   {
    "name": "学到的是 token → 下一个 token",
    "code": "W, losses = train(x, y, V, steps=300, lr=10.0)\nassert np.array_equal(W.argmax(1), (np.arange(V) + 1) % V)"
   },
   {
    "name": "一个前缀两种续写时，loss 停在 ln 2",
    "code": "W, losses = train(np.array([[0, 0, 0, 0]]), np.array([[1, 2, 1, 2]]), 3, steps=300, lr=1.0)\nassert abs(losses[-1] - np.log(2)) < 0.02, f\"最后的 loss 是 {losses[-1]:.3f}，应接近 0.693\""
   },
   {
    "name": "背下来不等于会泛化",
    "code": "W, losses = train(x, y, V, steps=300, lr=10.0)\nheld_x = np.arange(V)\nheld_y = (np.arange(V) + 2) % V                # 留出数据的规律不一样\nassert _ce(W[held_x], held_y) > losses[-1] + 3"
   }
  ]
 },
 {
  "id": "s3-lora",
  "stage": 3,
  "level": "易",
  "title": "LoRA：forward、合并、参数量",
  "statement": "<p>LoRA 冻结原来的权重 <code>W</code>，只训练一个低秩的增量 <code>ΔW = (alpha / r) · B · A</code>。</p>\n<ul>\n  <li><code>W</code>：形状 <code>(d_out, d_in)</code>。</li>\n  <li><code>A</code>：形状 <code>(r, d_in)</code>。<code>B</code>：形状 <code>(d_out, r)</code>。</li>\n  <li><code>x</code>：形状 <code>(N, d_in)</code>。</li>\n</ul>\n<p>实现三个函数：</p>\n<ul>\n  <li><code>lora_forward(x, W, A, B, alpha)</code>：<code>x @ Wᵀ + (alpha / r) * x @ Aᵀ @ Bᵀ</code>，形状 <code>(N, d_out)</code>。</li>\n  <li><code>merge(W, A, B, alpha)</code>：返回合并后的权重 <code>W + (alpha / r) * B @ A</code>，不要改动传进来的 <code>W</code>。</li>\n  <li><code>trainable_params(d_in, d_out, r)</code>：A 和 B 一共多少个参数。</li>\n</ul>\n<h3>示例</h3>\n<pre><code>trainable_params(4096, 4096, 8)   # 65536，原来的层有 16,777,216 个</code></pre>",
  "starter": "import numpy as np\n\n\ndef lora_forward(x, W, A, B, alpha):\n    pass\n\n\ndef merge(W, A, B, alpha):\n    pass\n\n\ndef trainable_params(d_in, d_out, r):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef lora_forward(x, W, A, B, alpha):\n    r = A.shape[0]\n    return x @ W.T + (alpha / r) * (x @ A.T @ B.T)\n\n\ndef merge(W, A, B, alpha):\n    r = A.shape[0]\n    return W + (alpha / r) * (B @ A)\n\n\ndef trainable_params(d_in, d_out, r):\n    return r * (d_in + d_out)\n",
  "setup": "rng = np.random.default_rng(0)\nx = rng.normal(size=(5, 6))\nW = rng.normal(size=(4, 6))\nA = rng.normal(size=(2, 6))\nB = rng.normal(size=(4, 2))",
  "tests": [
   {
    "name": "B 为 0 时输出和原模型一样",
    "code": "assert np.allclose(lora_forward(x, W, A, np.zeros((4, 2)), alpha=16), x @ W.T)"
   },
   {
    "name": "一个算得出来的例子",
    "code": "out = lora_forward(np.array([[1.0, 2.0]]), np.eye(2), np.array([[1.0, 1.0]]), np.array([[1.0], [2.0]]), alpha=2)\nassert np.allclose(out, [[7.0, 14.0]])"
   },
   {
    "name": "缩放是 alpha / r",
    "code": "base = x @ W.T\nd1 = lora_forward(x, W, A, B, alpha=2) - base\nd2 = lora_forward(x, W, A, B, alpha=8) - base\nassert np.allclose(d2, 4 * d1)\nassert np.allclose(d1, x @ A.T @ B.T)          # r = 2，alpha = 2，缩放是 1"
   },
   {
    "name": "合并之后是同一个函数",
    "code": "assert np.allclose(x @ merge(W, A, B, 16).T, lora_forward(x, W, A, B, 16))"
   },
   {
    "name": "merge 不改原来的 W",
    "code": "W0 = W.copy()\nmerge(W, A, B, 16)\nassert np.array_equal(W, W0)"
   },
   {
    "name": "参数量",
    "code": "assert trainable_params(4096, 4096, 8) == 65536\nassert trainable_params(768, 3072, 4) == 15360"
   }
  ]
 },
 {
  "id": "s3-sft-mask",
  "stage": 3,
  "level": "中",
  "title": "SFT：只在回答上算 loss",
  "statement": "<p>SFT 的一条样本是「prompt + 回答」拼成的一条序列，loss 只算在回答的 token 上。</p>\n<p>设 prompt 有 P 个 token，回答有 R 个。输入是序列去掉最后一个 token，目标是序列去掉第一个，长度都是 <code>P + R − 1</code>。</p>\n<p><code>build_mask(prompt_len, response_len)</code>：返回长度 <code>P + R − 1</code> 的 0/1 数组，<code>mask[i] = 1</code> 当且仅当第 i 个<strong>目标</strong>是回答里的 token。</p>\n<p><code>sft_loss(logits, targets, loss_mask)</code>：</p>\n<ul>\n  <li><code>logits</code> 形状 <code>(T, V)</code>，<code>targets</code> 形状 <code>(T,)</code>，<code>loss_mask</code> 形状 <code>(T,)</code>。</li>\n  <li>返回 mask 为 1 的位置上交叉熵的平均。mask 全为 0 时返回 <code>0.0</code>。</li>\n</ul>\n<h3>示例</h3>\n<pre><code># 序列: p0 p1 p2 r0 r1      P = 3, R = 2\n# 输入: p0 p1 p2 r0\n# 目标: p1 p2 r0 r1\nbuild_mask(3, 2)   # [0, 0, 1, 1]</code></pre>\n<p>注意第三个位置：输入是 prompt 的最后一个 token，目标已经是回答的第一个 token，要算 loss。</p>",
  "starter": "import numpy as np\n\n\ndef build_mask(prompt_len, response_len):\n    pass\n\n\ndef sft_loss(logits, targets, loss_mask):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef build_mask(prompt_len, response_len):\n    return np.array([0] * (prompt_len - 1) + [1] * response_len)\n\n\ndef sft_loss(logits, targets, loss_mask):\n    keep = np.asarray(loss_mask).astype(bool)\n    if not keep.any():\n        return 0.0\n    logits = np.asarray(logits, dtype=float)\n    targets = np.asarray(targets)\n    z = logits - logits.max(axis=1, keepdims=True)\n    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))\n    nll = -logp[np.arange(len(targets)), targets]\n    return float(nll[keep].mean())\n",
  "setup": "",
  "tests": [
   {
    "name": "mask 的长度和内容",
    "code": "assert list(build_mask(3, 2)) == [0, 0, 1, 1]\nassert len(build_mask(10, 5)) == 14 and int(np.sum(build_mask(10, 5))) == 5"
   },
   {
    "name": "prompt 只有一个 token 时全是 1",
    "code": "assert list(build_mask(1, 3)) == [1, 1, 1]"
   },
   {
    "name": "只平均 mask 为 1 的位置",
    "code": "logits = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]])\nloss = sft_loss(logits, np.array([0, 0, 1]), np.array([0, 1, 1]))\nassert abs(loss - 0.3499311) < 1e-6"
   },
   {
    "name": "prompt 位置的 logits 不影响 loss",
    "code": "rng = np.random.default_rng(0)\nlogits = rng.normal(size=(6, 5))\ntargets = rng.integers(5, size=6)\nmask = build_mask(4, 3)\na = sft_loss(logits, targets, mask)\nlogits2 = logits.copy()\nlogits2[:3] += rng.normal(size=(3, 5)) * 10\nassert abs(sft_loss(logits2, targets, mask) - a) < 1e-9"
   },
   {
    "name": "mask 全为 0 时返回 0.0",
    "code": "assert sft_loss(np.zeros((3, 4)), np.array([0, 1, 2]), np.zeros(3)) == 0.0"
   }
  ]
 },
 {
  "id": "s3-dpo",
  "stage": 3,
  "level": "中",
  "title": "DPO loss",
  "statement": "<p>偏好数据的每一条是：一个 prompt、一个好的回答 y⁺、一个差的回答 y⁻。DPO 让当前策略 π 相对参考模型 π_ref 更偏向 y⁺。</p>\n<p>实现 <code>dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1)</code>。四个输入都是形状 <code>(N,)</code> 的数组，是整条回答的对数概率：</p>\n<ul>\n  <li><code>pi_pos[i] = log π(y⁺ | x)</code>，<code>pi_neg[i] = log π(y⁻ | x)</code></li>\n  <li><code>ref_pos[i] = log π_ref(y⁺ | x)</code>，<code>ref_neg[i] = log π_ref(y⁻ | x)</code></li>\n</ul>\n<pre><code>z = beta * ( (pi_pos − ref_pos) − (pi_neg − ref_neg) )\nloss = mean( −log σ(z) )</code></pre>\n<p>要求数值稳定：<code>z</code> 是 ±2000 时也不能出现 <code>inf</code> 或 <code>nan</code>。提示：<code>−log σ(z) = log(1 + e^(−z))</code>，可以用 <code>np.logaddexp(0, −z)</code>。</p>\n<h3>示例</h3>\n<pre><code># 策略和参考模型一样时 z = 0，loss = ln 2\ndpo_loss([-5.], [-7.], [-5.], [-7.])   # 0.6931</code></pre>",
  "starter": "import numpy as np\n\n\ndef dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1):\n    pi_pos, pi_neg = np.asarray(pi_pos, float), np.asarray(pi_neg, float)\n    ref_pos, ref_neg = np.asarray(ref_pos, float), np.asarray(ref_neg, float)\n    z = beta * ((pi_pos - ref_pos) - (pi_neg - ref_neg))\n    return float(np.logaddexp(0, -z).mean())\n",
  "setup": "",
  "tests": [
   {
    "name": "策略等于参考模型时是 ln 2",
    "code": "loss = dpo_loss(np.array([-5.0, -3.0]), np.array([-7.0, -1.0]), np.array([-5.0, -3.0]), np.array([-7.0, -1.0]))\nassert abs(loss - np.log(2)) < 1e-9"
   },
   {
    "name": "一个算得出来的例子",
    "code": "assert abs(dpo_loss([-10.0], [-12.0], [-11.0], [-11.0], beta=1.0) - 0.1269280) < 1e-6\nassert abs(dpo_loss([-10.0], [-12.0], [-11.0], [-11.0], beta=0.5) - 0.3132617) < 1e-6"
   },
   {
    "name": "好回答的概率升高，loss 下降",
    "code": "a = dpo_loss([-10.0], [-10.0], [-10.0], [-10.0])\nb = dpo_loss([-8.0], [-10.0], [-10.0], [-10.0])\nc = dpo_loss([-10.0], [-8.0], [-10.0], [-10.0])\nassert b < a < c"
   },
   {
    "name": "只看相对参考模型的变化",
    "code": "a = dpo_loss([-10.0], [-12.0], [-11.0], [-11.0], beta=1.0)\nb = dpo_loss([-110.0], [-12.0], [-111.0], [-11.0], beta=1.0)\nassert abs(a - b) < 1e-9"
   },
   {
    "name": "对一个 batch 取平均",
    "code": "loss = dpo_loss([-10.0, -5.0], [-12.0, -5.0], [-11.0, -5.0], [-11.0, -5.0], beta=1.0)\nassert abs(loss - (0.1269280 + np.log(2)) / 2) < 1e-6"
   },
   {
    "name": "z 很大或很小时不溢出",
    "code": "big = dpo_loss([0.0], [-2000.0], [0.0], [0.0], beta=1.0)\nsmall = dpo_loss([-2000.0], [0.0], [0.0], [0.0], beta=1.0)\nassert np.isfinite(big) and big < 1e-6\nassert np.isfinite(small) and abs(small - 2000) < 1e-6"
   }
  ]
 },
 {
  "id": "s3-policy-gradient",
  "stage": 3,
  "level": "中",
  "title": "策略梯度的目标，对照 DPO",
  "statement": "<p>RL 的目标是最大化期望奖励。和 DPO 对照着看：DPO 需要成对的回答和一个参考模型，RL 需要从当前策略采样，再用奖励给每个样本加权。</p>\n<p><code>pg_loss(logp, rewards, baseline=None)</code>：REINFORCE 的 surrogate loss。</p>\n<pre><code>loss = −mean( (rewards − baseline) * logp )</code></pre>\n<p><code>logp[i]</code> 是第 i 个采样到的回答的对数概率。<code>baseline</code> 没给时用 <code>rewards</code> 的平均值。</p>\n<p><code>pg_grad_logits(logits, action, reward, baseline=0.0)</code>：策略是对 <code>logits</code>（形状 <code>(K,)</code>）做 softmax，采样到了 <code>action</code>。返回这一个样本的 loss 对 <code>logits</code> 的梯度，形状 <code>(K,)</code>：</p>\n<pre><code>grad = (reward − baseline) * (softmax(logits) − onehot(action))</code></pre>\n<h3>示例</h3>\n<pre><code>pg_loss(logp=[-1., -2.], rewards=[1., 3.])   # baseline = 2，loss = 0.5</code></pre>\n<p>奖励全都一样时，减掉 baseline 之后每个样本的权重都是 0：没有哪个回答比别的好，就没有学习信号。</p>",
  "starter": "import numpy as np\n\n\ndef pg_loss(logp, rewards, baseline=None):\n    pass\n\n\ndef pg_grad_logits(logits, action, reward, baseline=0.0):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef pg_loss(logp, rewards, baseline=None):\n    logp = np.asarray(logp, dtype=float)\n    rewards = np.asarray(rewards, dtype=float)\n    if baseline is None:\n        baseline = rewards.mean()\n    return float(-((rewards - baseline) * logp).mean())\n\n\ndef pg_grad_logits(logits, action, reward, baseline=0.0):\n    logits = np.asarray(logits, dtype=float)\n    p = np.exp(logits - logits.max())\n    p /= p.sum()\n    p[action] -= 1\n    return (reward - baseline) * p\n",
  "setup": "def _logsoftmax(l):\n    z = l - l.max()\n    return z - np.log(np.exp(z).sum())",
  "tests": [
   {
    "name": "默认 baseline 是奖励的平均",
    "code": "assert abs(pg_loss([-1.0, -2.0], [1.0, 3.0]) - 0.5) < 1e-9"
   },
   {
    "name": "可以指定 baseline",
    "code": "assert abs(pg_loss([-1.0, -2.0], [1.0, 3.0], baseline=0.0) - 3.5) < 1e-9"
   },
   {
    "name": "奖励全都一样时没有信号",
    "code": "assert abs(pg_loss([-1.0, -2.0, -3.0], [5.0, 5.0, 5.0])) < 1e-9"
   },
   {
    "name": "梯度和数值微分一致",
    "code": "logits = np.array([0.2, -1.0, 0.7, 0.1])\na, r, b = 2, 1.5, 0.4\ng = pg_grad_logits(logits, a, r, b)\nnum = np.zeros(4)\nfor i in range(4):\n    d = np.zeros(4); d[i] = 1e-6\n    f = lambda l: -(r - b) * _logsoftmax(l)[a]\n    num[i] = (f(logits + d) - f(logits - d)) / 2e-6\nassert np.asarray(g).shape == (4,)\nassert np.allclose(g, num, atol=1e-6)"
   },
   {
    "name": "奖励高于 baseline，这个动作的概率上升",
    "code": "logits = np.array([0.0, 0.0, 0.0])\ng = pg_grad_logits(logits, action=1, reward=2.0, baseline=1.0)\nnew = logits - 0.5 * g\nassert np.exp(_logsoftmax(new))[1] > 1 / 3"
   },
   {
    "name": "奖励低于 baseline，这个动作的概率下降",
    "code": "logits = np.array([0.0, 0.0, 0.0])\ng = pg_grad_logits(logits, action=1, reward=0.0, baseline=1.0)\nnew = logits - 0.5 * g\nassert np.exp(_logsoftmax(new))[1] < 1 / 3"
   }
  ]
 },
 {
  "id": "s3-paired",
  "stage": 3,
  "level": "易",
  "title": "验：同一组留出 prompts，训练前后配对比较",
  "statement": "<p>这是第 3 步的验收。训练前后用同一组没训练过的 prompts 各生成一次，逐条打分，再配对比较。</p>\n<p>实现 <code>compare(before, after, tol=1e-9)</code>。<code>before[i]</code> 和 <code>after[i]</code> 是第 i 个 prompt 在训练前后的分数。返回一个 dict：</p>\n<ul>\n  <li><code>\"wins\"</code>：<code>after − before &gt; tol</code> 的条数。</li>\n  <li><code>\"losses\"</code>：<code>after − before &lt; −tol</code> 的条数。</li>\n  <li><code>\"ties\"</code>：其余的条数。</li>\n  <li><code>\"mean_delta\"</code>：<code>after − before</code> 的平均值。</li>\n</ul>\n<p>两个列表长度不一样时，说明前后用的不是同一组 prompts，抛出 <code>ValueError</code>。</p>\n<h3>示例</h3>\n<pre><code>compare([0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.4, 1.0])\n# {\"wins\": 1, \"losses\": 3, \"ties\": 0, \"mean_delta\": 0.05}</code></pre>\n<p>平均分涨了，但四条里有三条变差了。只看平均分会漏掉这件事。</p>",
  "starter": "def compare(before, after, tol=1e-9):\n    pass\n",
  "solution": "def compare(before, after, tol=1e-9):\n    if len(before) != len(after):\n        raise ValueError(\"before 和 after 的条数不一致，不是同一组 prompts\")\n    deltas = [a - b for a, b in zip(after, before)]\n    return {\n        \"wins\": sum(d > tol for d in deltas),\n        \"losses\": sum(d < -tol for d in deltas),\n        \"ties\": sum(abs(d) <= tol for d in deltas),\n        \"mean_delta\": sum(deltas) / len(deltas) if deltas else 0.0,\n    }\n",
  "setup": "",
  "tests": [
   {
    "name": "数出变好、变差和持平的条数",
    "code": "r = compare([0.2, 0.5, 0.9, 0.4], [0.6, 0.5, 0.7, 0.8])\nassert (r[\"wins\"], r[\"losses\"], r[\"ties\"]) == (2, 1, 1)\nassert abs(r[\"mean_delta\"] - 0.15) < 1e-9"
   },
   {
    "name": "平均分涨了，多数却变差了",
    "code": "r = compare([0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.4, 1.0])\nassert r[\"wins\"] == 1 and r[\"losses\"] == 3\nassert r[\"mean_delta\"] > 0"
   },
   {
    "name": "三个数加起来等于条数",
    "code": "import random\nrandom.seed(0)\nb = [random.random() for _ in range(50)]\na = [v if i % 5 == 0 else random.random() for i, v in enumerate(b)]\nr = compare(b, a)\nassert r[\"wins\"] + r[\"losses\"] + r[\"ties\"] == 50\nassert r[\"ties\"] == 10"
   },
   {
    "name": "长度不一致时抛出 ValueError",
    "code": "try:\n    compare([0.1, 0.2], [0.3])\nexcept ValueError:\n    pass\nelse:\n    assert False, \"长度不一致时应该抛出 ValueError\""
   }
  ]
 },
 {
  "id": "s4-sampling",
  "stage": 4,
  "level": "中",
  "title": "Temperature、top-k、top-p",
  "statement": "<p>实现 <code>sampling_probs(logits, temperature=1.0, top_k=0, top_p=1.0)</code>：返回真正用来抽样的概率分布，形状和 <code>logits</code> 一样。按这个顺序处理：</p>\n<ol>\n  <li><strong>Temperature</strong>：<code>softmax(logits / temperature)</code>。<code>temperature == 0</code> 时直接返回 one-hot：最大的那个是 1，其余是 0。</li>\n  <li><strong>Top-k</strong>（<code>top_k &gt; 0</code> 时）：只保留概率最高的 k 个，其余置 0，再归一化。</li>\n  <li><strong>Top-p</strong>（<code>top_p &lt; 1</code> 时）：按概率从大到小排，保留累计概率<strong>刚好达到</strong> <code>top_p</code> 的最短前缀（至少一个），其余置 0，再归一化。</li>\n</ol>\n<h3>示例</h3>\n<pre><code>p = [0.5, 0.3, 0.2]\n# top_p = 0.6：0.5 不够，0.5 + 0.3 = 0.8 够了，保留前两个\nsampling_probs(np.log(p), top_p=0.6)   # [0.625, 0.375, 0]</code></pre>",
  "starter": "import numpy as np\n\n\ndef sampling_probs(logits, temperature=1.0, top_k=0, top_p=1.0):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef sampling_probs(logits, temperature=1.0, top_k=0, top_p=1.0):\n    logits = np.asarray(logits, dtype=float)\n    if temperature == 0:\n        p = np.zeros_like(logits)\n        p[np.argmax(logits)] = 1.0\n        return p\n\n    z = logits / temperature\n    p = np.exp(z - z.max())\n    p /= p.sum()\n    order = np.argsort(-p)                    # 概率从大到小的下标\n\n    if top_k > 0:\n        p[order[top_k:]] = 0.0\n        p /= p.sum()\n\n    if top_p < 1.0:\n        cum = np.cumsum(p[order])\n        n = int(np.searchsorted(cum, top_p - 1e-12)) + 1\n        p[order[n:]] = 0.0\n        p /= p.sum()\n    return p\n",
  "setup": "",
  "tests": [
   {
    "name": "默认参数就是 softmax",
    "code": "p = sampling_probs(np.array([1.0, 2.0, 3.0]))\nassert np.allclose(p, [0.09003057, 0.24472847, 0.66524096])"
   },
   {
    "name": "temperature 为 0 时是 one-hot",
    "code": "p = sampling_probs(np.array([1.0, 3.0, 2.0]), temperature=0)\nassert np.allclose(p, [0, 1, 0])"
   },
   {
    "name": "温度越高分布越平",
    "code": "l = np.array([1.0, 2.0, 3.0])\nassert sampling_probs(l, temperature=0.5).max() > sampling_probs(l).max() > sampling_probs(l, temperature=2.0).max()"
   },
   {
    "name": "top-k 只留 k 个，并重新归一化",
    "code": "p = sampling_probs(np.array([1.0, 2.0, 3.0, 0.0]), top_k=2)\nassert np.allclose(p, [0, 0.26894142, 0.73105858, 0])"
   },
   {
    "name": "top-p 保留刚好达到 p 的最短前缀",
    "code": "p = sampling_probs(np.log([0.5, 0.3, 0.2]), top_p=0.6)\nassert np.allclose(p, [0.625, 0.375, 0])"
   },
   {
    "name": "top-p 很小时至少留一个",
    "code": "p = sampling_probs(np.log([0.2, 0.5, 0.3]), top_p=0.01)\nassert np.allclose(p, [0, 1, 0])"
   },
   {
    "name": "先 top-k，再在剩下的分布上做 top-p",
    "code": "p = sampling_probs(np.log([0.4, 0.3, 0.2, 0.1]), top_k=3, top_p=0.5)\nassert np.allclose(p, [4 / 7, 3 / 7, 0, 0])"
   },
   {
    "name": "结果的和是 1",
    "code": "l = np.random.default_rng(0).normal(size=50)\np = sampling_probs(l, temperature=0.7, top_k=20, top_p=0.9)\nassert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()"
   }
  ]
 },
 {
  "id": "s4-kv-cache",
  "stage": 4,
  "level": "中",
  "title": "验：KV cache 的一步，和整段重算一致",
  "statement": "<p>这是第 4 步的验收。<code>attend_full</code> 已经写好：它拿到整段输入，每次都从头算一遍 causal attention。</p>\n<p>你来写带 cache 的版本：<code>attend_step(x_new, Wq, Wk, Wv, cache)</code>，一次只处理一个新 token。</p>\n<ul>\n  <li><code>x_new</code>：形状 <code>(C,)</code>，新来的这个 token 的向量。</li>\n  <li><code>cache</code>：<code>{\"K\": ..., \"V\": ...}</code>，两个形状 <code>(t, D)</code> 的数组，是之前 t 个 token 的 key 和 value。<code>t</code> 可以是 0。</li>\n</ul>\n<p>步骤：</p>\n<ol>\n  <li>只为新 token 算 <code>q</code>、<code>k</code>、<code>v</code>。</li>\n  <li>把 <code>k</code>、<code>v</code> 接到 cache 后面，得到 <code>(t+1, D)</code>。</li>\n  <li>用 <code>q</code> 和全部 <code>t+1</code> 个 key 算分数（除以 <code>√D</code>），softmax，再对 value 加权求和。</li>\n</ol>\n<p>返回 <code>(out, new_cache)</code>，<code>out</code> 形状 <code>(D,)</code>。不要原地修改传进来的 <code>cache</code>。</p>\n<h3>要成立的事</h3>\n<pre><code>把 X 的每一行依次喂给 attend_step，\n第 t 次的 out 应该等于 attend_full(X, ...)[t]</code></pre>\n<p>这里不需要 mask：cache 里只有过去的 token，新 token 本来就看不到未来。</p>",
  "starter": "import numpy as np\n\n\ndef attend_full(X, Wq, Wk, Wv):\n    \"\"\"整段重算。X: (T, C)，返回 (T, D)。已经写好，用来对照。\"\"\"\n    T = X.shape[0]\n    Q, K, V = X @ Wq, X @ Wk, X @ Wv\n    S = Q @ K.T / np.sqrt(K.shape[1])\n    S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)\n    P = np.exp(S - S.max(axis=1, keepdims=True))\n    P /= P.sum(axis=1, keepdims=True)\n    return P @ V\n\n\ndef attend_step(x_new, Wq, Wk, Wv, cache):\n    # 返回 (out, new_cache)\n    pass\n",
  "solution": "import numpy as np\n\n\ndef attend_full(X, Wq, Wk, Wv):\n    \"\"\"整段重算。X: (T, C)，返回 (T, D)。已经写好，用来对照。\"\"\"\n    T = X.shape[0]\n    Q, K, V = X @ Wq, X @ Wk, X @ Wv\n    S = Q @ K.T / np.sqrt(K.shape[1])\n    S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)\n    P = np.exp(S - S.max(axis=1, keepdims=True))\n    P /= P.sum(axis=1, keepdims=True)\n    return P @ V\n\n\ndef attend_step(x_new, Wq, Wk, Wv, cache):\n    q, k, v = x_new @ Wq, x_new @ Wk, x_new @ Wv\n    K = np.vstack([cache[\"K\"], k[None, :]])          # 新数组，不动原来的 cache\n    V = np.vstack([cache[\"V\"], v[None, :]])\n    s = K @ q / np.sqrt(K.shape[1])\n    p = np.exp(s - s.max())\n    p /= p.sum()\n    return p @ V, {\"K\": K, \"V\": V}\n",
  "setup": "rng = np.random.default_rng(0)\nC, D, T = 6, 4, 7\nX = rng.normal(size=(T, C))\nWq, Wk, Wv = (rng.normal(size=(C, D)) for _ in range(3))\n\ndef _full(X):\n    n = X.shape[0]\n    Q, K, V = X @ Wq, X @ Wk, X @ Wv\n    S = Q @ K.T / np.sqrt(D)\n    S = np.where(np.triu(np.ones((n, n), dtype=bool), k=1), -np.inf, S)\n    P = np.exp(S - S.max(axis=1, keepdims=True))\n    P /= P.sum(axis=1, keepdims=True)\n    return P @ V\n\ndef _empty():\n    return {\"K\": np.zeros((0, D)), \"V\": np.zeros((0, D))}",
  "tests": [
   {
    "name": "cache 为空时，输出就是新 token 自己的 value",
    "code": "out, cache = attend_step(X[0], Wq, Wk, Wv, _empty())\nassert np.asarray(out).shape == (D,)\nassert np.allclose(out, X[0] @ Wv)"
   },
   {
    "name": "逐个 token 喂进去，每一步都和整段重算一致",
    "code": "ref = _full(X)\ncache = _empty()\nfor t in range(T):\n    out, cache = attend_step(X[t], Wq, Wk, Wv, cache)\n    assert np.allclose(out, ref[t], atol=1e-8), f\"第 {t} 步对不上\""
   },
   {
    "name": "cache 每步多一行",
    "code": "cache = _empty()\nfor t in range(T):\n    out, cache = attend_step(X[t], Wq, Wk, Wv, cache)\n    assert cache[\"K\"].shape == (t + 1, D) and cache[\"V\"].shape == (t + 1, D)"
   },
   {
    "name": "cache 里存的是每个 token 的 key 和 value",
    "code": "cache = _empty()\nfor t in range(T):\n    out, cache = attend_step(X[t], Wq, Wk, Wv, cache)\nassert np.allclose(cache[\"K\"], X @ Wk)\nassert np.allclose(cache[\"V\"], X @ Wv)"
   },
   {
    "name": "不原地修改传进来的 cache",
    "code": "old = _empty()\nout, new = attend_step(X[0], Wq, Wk, Wv, old)\nassert old[\"K\"].shape == (0, D) and old[\"V\"].shape == (0, D)\nout2, newer = attend_step(X[1], Wq, Wk, Wv, new)\nassert new[\"K\"].shape == (1, D)"
   }
  ]
 },
 {
  "id": "s4-kv-size",
  "stage": 4,
  "level": "易",
  "title": "KV cache 占多少显存",
  "statement": "<p>每一层、每个头、每个位置都要存一个 key 和一个 value，各 <code>head_dim</code> 个数。</p>\n<p><code>kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, batch=1, bytes_per_value=2)</code>：返回 KV cache 的字节数。<code>bytes_per_value</code> 是每个数占的字节：fp16 是 2，fp32 是 4。</p>\n<pre><code>2 × n_layers × batch × seq_len × n_heads × head_dim × bytes_per_value</code></pre>\n<p><code>max_context(budget_bytes, n_layers, n_heads, head_dim, batch=1, bytes_per_value=2)</code>：给定显存预算，返回放得下的最长 <code>seq_len</code>（整数，向下取整）。</p>\n<h3>示例</h3>\n<pre><code># 12 层、12 头、每头 64 维、fp16、1024 个 token\nkv_cache_bytes(12, 12, 64, 1024)   # 37,748,736 字节，约 36 MB</code></pre>",
  "starter": "def kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, batch=1, bytes_per_value=2):\n    pass\n\n\ndef max_context(budget_bytes, n_layers, n_heads, head_dim, batch=1, bytes_per_value=2):\n    pass\n",
  "solution": "def kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, batch=1, bytes_per_value=2):\n    return 2 * n_layers * batch * seq_len * n_heads * head_dim * bytes_per_value\n\n\ndef max_context(budget_bytes, n_layers, n_heads, head_dim, batch=1, bytes_per_value=2):\n    per_token = 2 * n_layers * batch * n_heads * head_dim * bytes_per_value\n    return budget_bytes // per_token\n",
  "setup": "",
  "tests": [
   {
    "name": "示例里的数",
    "code": "assert kv_cache_bytes(12, 12, 64, 1024) == 37748736"
   },
   {
    "name": "随序列长度和 batch 线性增长",
    "code": "base = kv_cache_bytes(12, 12, 64, 1024)\nassert kv_cache_bytes(12, 12, 64, 2048) == 2 * base\nassert kv_cache_bytes(12, 12, 64, 1024, batch=8) == 8 * base"
   },
   {
    "name": "fp32 是 fp16 的两倍",
    "code": "assert kv_cache_bytes(12, 12, 64, 1024, bytes_per_value=4) == 2 * kv_cache_bytes(12, 12, 64, 1024)"
   },
   {
    "name": "max_context 是反过来算",
    "code": "assert max_context(37748736, 12, 12, 64) == 1024\nassert max_context(37748735, 12, 12, 64) == 1023\nassert max_context(37748736, 12, 12, 64, batch=4) == 256"
   },
   {
    "name": "返回整数",
    "code": "assert isinstance(max_context(10 ** 9, 24, 16, 64), int)\nassert kv_cache_bytes(24, 16, 64, max_context(10 ** 9, 24, 16, 64)) <= 10 ** 9"
   }
  ]
 },
 {
  "id": "s4-latency",
  "stage": 4,
  "level": "易",
  "title": "从时间戳算 TTFT 和每 token 延迟",
  "statement": "<p>一次流式生成留下一串时间戳。实现 <code>latency_stats(t_request, token_times)</code>：</p>\n<ul>\n  <li><code>t_request</code>：请求发出的时刻（秒）。</li>\n  <li><code>token_times</code>：每个输出 token 到达的时刻，至少有一个，按时间递增。</li>\n</ul>\n<p>返回一个 dict：</p>\n<ul>\n  <li><code>\"ttft\"</code>：首 token 延迟，<code>token_times[0] − t_request</code>。主要由 prefill 决定。</li>\n  <li><code>\"itl\"</code>：相邻两个 token 的平均间隔。主要由 decode 决定。只有一个 token 时是 <code>0.0</code>。</li>\n  <li><code>\"tps\"</code>：decode 阶段每秒出几个 token，<code>(n − 1) / (最后一个 − 第一个)</code>。只有一个 token 时是 <code>0.0</code>。</li>\n  <li><code>\"total\"</code>：<code>token_times[-1] − t_request</code>。</li>\n</ul>\n<h3>示例</h3>\n<pre><code>latency_stats(10.0, [10.5, 10.6, 10.7, 10.9])\n# ttft = 0.5, itl ≈ 0.1333, tps = 7.5, total = 0.9</code></pre>",
  "starter": "def latency_stats(t_request, token_times):\n    pass\n",
  "solution": "def latency_stats(t_request, token_times):\n    n = len(token_times)\n    first, last = token_times[0], token_times[-1]\n    gaps = [b - a for a, b in zip(token_times, token_times[1:])]\n    return {\n        \"ttft\": first - t_request,\n        \"itl\": sum(gaps) / len(gaps) if gaps else 0.0,\n        \"tps\": (n - 1) / (last - first) if n > 1 else 0.0,\n        \"total\": last - t_request,\n    }\n",
  "setup": "",
  "tests": [
   {
    "name": "示例里的数",
    "code": "r = latency_stats(10.0, [10.5, 10.6, 10.7, 10.9])\nassert abs(r[\"ttft\"] - 0.5) < 1e-9\nassert abs(r[\"itl\"] - 0.4 / 3) < 1e-9\nassert abs(r[\"tps\"] - 7.5) < 1e-9\nassert abs(r[\"total\"] - 0.9) < 1e-9"
   },
   {
    "name": "只有一个 token",
    "code": "r = latency_stats(0.0, [2.0])\nassert abs(r[\"ttft\"] - 2.0) < 1e-9 and abs(r[\"total\"] - 2.0) < 1e-9\nassert r[\"itl\"] == 0.0 and r[\"tps\"] == 0.0"
   },
   {
    "name": "prompt 变长只影响 TTFT",
    "code": "a = latency_stats(10.0, [10.5, 10.6, 10.7, 10.9])\nb = latency_stats(10.0, [11.5, 11.6, 11.7, 11.9])\nassert abs(b[\"ttft\"] - a[\"ttft\"] - 1.0) < 1e-9\nassert abs(b[\"itl\"] - a[\"itl\"]) < 1e-9 and abs(b[\"tps\"] - a[\"tps\"]) < 1e-9"
   },
   {
    "name": "tps 是 itl 的倒数",
    "code": "r = latency_stats(0.0, [1.0, 1.25, 1.5, 1.75, 2.0])\nassert abs(r[\"tps\"] * r[\"itl\"] - 1) < 1e-9"
   }
  ]
 },
 {
  "id": "s4-quant",
  "stage": 4,
  "level": "易",
  "title": "对称 int8 量化",
  "statement": "<p>把 float32 的权重存成 int8，显存和每步要读的字节数都变成四分之一。</p>\n<p><code>quantize(w)</code>：返回 <code>(q, scale)</code>。</p>\n<pre><code>scale = max(|w|) / 127\nq = round(w / scale)，截断到 [−127, 127]，类型是 int8</code></pre>\n<p><code>w</code> 全为 0 时 <code>scale</code> 取 <code>1.0</code>。</p>\n<p><code>dequantize(q, scale)</code>：返回 <code>q * scale</code>，类型是 float32。</p>\n<h3>示例</h3>\n<pre><code>q, scale = quantize(np.array([-0.6, 0.2, 1.0], dtype=np.float32))\nq        # [-76, 25, 127]\nscale    # 1 / 127 ≈ 0.007874</code></pre>\n<p>量化之后再还原，每个数的误差不超过半个 <code>scale</code>。</p>",
  "starter": "import numpy as np\n\n\ndef quantize(w):\n    # 返回 (q, scale)\n    pass\n\n\ndef dequantize(q, scale):\n    pass\n",
  "solution": "import numpy as np\n\n\ndef quantize(w):\n    w = np.asarray(w, dtype=np.float32)\n    biggest = float(np.abs(w).max()) if w.size else 0.0\n    scale = biggest / 127 if biggest > 0 else 1.0\n    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)\n    return q, scale\n\n\ndef dequantize(q, scale):\n    return q.astype(np.float32) * np.float32(scale)\n",
  "setup": "w = np.random.default_rng(0).normal(size=(64, 32)).astype(np.float32)",
  "tests": [
   {
    "name": "示例里的数",
    "code": "q, scale = quantize(np.array([-0.6, 0.2, 1.0], dtype=np.float32))\nassert list(q) == [-76, 25, 127]\nassert abs(scale - 1 / 127) < 1e-7"
   },
   {
    "name": "q 是 int8，范围在 ±127 以内",
    "code": "q, scale = quantize(w)\nassert q.dtype == np.int8\nassert q.min() >= -127 and q.max() <= 127\nassert np.abs(q).max() == 127"
   },
   {
    "name": "还原之后误差不超过半个 scale",
    "code": "q, scale = quantize(w)\nback = dequantize(q, scale)\nassert back.dtype == np.float32\nassert np.abs(back - w).max() <= scale / 2 + 1e-6"
   },
   {
    "name": "全为 0 的输入",
    "code": "q, scale = quantize(np.zeros(5, dtype=np.float32))\nassert scale == 1.0 and not q.any()"
   },
   {
    "name": "占的字节是原来的四分之一",
    "code": "q, scale = quantize(w)\nassert q.nbytes * 4 == w.nbytes"
   },
   {
    "name": "正负对称",
    "code": "q1, s1 = quantize(w)\nq2, s2 = quantize(-w)\nassert abs(s1 - s2) < 1e-12 and np.array_equal(q2, -q1)"
   }
  ]
 },
 {
  "id": "s5-tool-call",
  "stage": 5,
  "level": "中",
  "title": "解析并校验工具调用",
  "statement": "<p>模型输出的是一段文本。执行之前，harness 要先确认它是一个合法的工具调用。</p>\n<p>实现 <code>parse_call(raw, tools)</code>：</p>\n<ul>\n  <li><code>raw</code>：模型输出的字符串，约定是 JSON：<code>{\"tool\": \"名字\", \"args\": {...}}</code>。没有 <code>\"args\"</code> 时当作 <code>{}</code>。</li>\n  <li><code>tools</code>：<code>{工具名: {\"params\": {参数名: 类型名}}}</code>，类型名是 <code>\"str\"</code>、<code>\"int\"</code>、<code>\"float\"</code>、<code>\"bool\"</code> 之一。所有参数都必填。</li>\n</ul>\n<p>合法时返回 <code>({\"tool\": 名字, \"args\": 参数}, None)</code>。不合法时返回 <code>(None, 错误信息)</code>，错误信息以下面某个前缀开头，并带上出问题的工具名或参数名。按这个顺序检查，遇到第一个问题就返回：</p>\n<ol>\n  <li><code>invalid_json</code>：不是合法 JSON；或者不是对象；或者没有字符串类型的 <code>\"tool\"</code>；或者 <code>\"args\"</code> 不是对象。</li>\n  <li><code>unknown_tool</code>：工具名不在 <code>tools</code> 里。</li>\n  <li><code>missing_arg</code>：少了必填参数。</li>\n  <li><code>unexpected_arg</code>：多了没定义的参数。</li>\n  <li><code>wrong_type</code>：参数类型不对。</li>\n</ol>\n<p>类型规则：<code>\"float\"</code> 接受整数和小数；<code>True</code> / <code>False</code> 只能给 <code>\"bool\"</code>，不能当 <code>\"int\"</code> 或 <code>\"float\"</code> 用。</p>\n<p>这个函数<strong>永远不抛异常</strong>：错误信息是要回给模型看的。</p>\n<h3>示例</h3>\n<pre><code>tools = {\"click\": {\"params\": {\"selector\": \"str\"}}}\nparse_call('{\"tool\": \"click\", \"args\": {\"selector\": \"#buy\"}}', tools)\n# ({\"tool\": \"click\", \"args\": {\"selector\": \"#buy\"}}, None)\nparse_call('{\"tool\": \"click\", \"args\": {}}', tools)\n# (None, \"missing_arg: selector\")</code></pre>",
  "starter": "import json\n\n\ndef parse_call(raw, tools):\n    # 返回 (call, None) 或 (None, error)\n    pass\n",
  "solution": "import json\n\nTYPES = {\"str\": str, \"int\": int, \"float\": (int, float), \"bool\": bool}\n\n\ndef parse_call(raw, tools):\n    try:\n        obj = json.loads(raw)\n    except Exception:\n        return None, \"invalid_json: 不是合法的 JSON\"\n    if not isinstance(obj, dict) or not isinstance(obj.get(\"tool\"), str):\n        return None, 'invalid_json: 需要 {\"tool\": \"...\", \"args\": {...}}'\n    name, args = obj[\"tool\"], obj.get(\"args\", {})\n    if not isinstance(args, dict):\n        return None, \"invalid_json: args 必须是对象\"\n\n    if name not in tools:\n        return None, f\"unknown_tool: {name}\"\n    params = tools[name][\"params\"]\n    for p in params:\n        if p not in args:\n            return None, f\"missing_arg: {p}\"\n    for a in args:\n        if a not in params:\n            return None, f\"unexpected_arg: {a}\"\n    for p, type_name in params.items():\n        value = args[p]\n        is_bool = isinstance(value, bool)          # bool 是 int 的子类，要单独挡住\n        if (is_bool and type_name != \"bool\") or not isinstance(value, TYPES[type_name]):\n            return None, f\"wrong_type: {p}\"\n    return {\"tool\": name, \"args\": args}, None\n",
  "setup": "TOOLS = {\n    \"click\": {\"params\": {\"selector\": \"str\"}},\n    \"type\": {\"params\": {\"selector\": \"str\", \"text\": \"str\"}},\n    \"scroll\": {\"params\": {\"pixels\": \"int\"}},\n    \"wait\": {\"params\": {\"seconds\": \"float\"}},\n    \"set_checkbox\": {\"params\": {\"selector\": \"str\", \"checked\": \"bool\"}},\n    \"screenshot\": {\"params\": {}},\n}",
  "tests": [
   {
    "name": "合法的调用",
    "code": "call, err = parse_call('{\"tool\": \"click\", \"args\": {\"selector\": \"#buy\"}}', TOOLS)\nassert err is None\nassert call == {\"tool\": \"click\", \"args\": {\"selector\": \"#buy\"}}"
   },
   {
    "name": "没有 args 时当作空对象",
    "code": "call, err = parse_call('{\"tool\": \"screenshot\"}', TOOLS)\nassert err is None and call == {\"tool\": \"screenshot\", \"args\": {}}"
   },
   {
    "name": "不是 JSON",
    "code": "for raw in [\"我来点击购买按钮\", \"\", \"{tool: click}\"]:\n    call, err = parse_call(raw, TOOLS)\n    assert call is None and err.startswith(\"invalid_json\"), raw"
   },
   {
    "name": "是 JSON 但形状不对",
    "code": "for raw in [\"[1, 2]\", \"null\", '{\"args\": {}}', '{\"tool\": 3}', '{\"tool\": \"click\", \"args\": [1]}']:\n    call, err = parse_call(raw, TOOLS)\n    assert call is None and err.startswith(\"invalid_json\"), raw"
   },
   {
    "name": "未知工具",
    "code": "call, err = parse_call('{\"tool\": \"buy_now\", \"args\": {}}', TOOLS)\nassert call is None and err.startswith(\"unknown_tool\") and \"buy_now\" in err"
   },
   {
    "name": "少了参数",
    "code": "call, err = parse_call('{\"tool\": \"type\", \"args\": {\"selector\": \"#q\"}}', TOOLS)\nassert call is None and err.startswith(\"missing_arg\") and \"text\" in err"
   },
   {
    "name": "多了参数",
    "code": "call, err = parse_call('{\"tool\": \"click\", \"args\": {\"selector\": \"#buy\", \"force\": true}}', TOOLS)\nassert call is None and err.startswith(\"unexpected_arg\") and \"force\" in err"
   },
   {
    "name": "类型不对",
    "code": "call, err = parse_call('{\"tool\": \"scroll\", \"args\": {\"pixels\": \"300\"}}', TOOLS)\nassert call is None and err.startswith(\"wrong_type\") and \"pixels\" in err"
   },
   {
    "name": "true 不能当整数用",
    "code": "call, err = parse_call('{\"tool\": \"scroll\", \"args\": {\"pixels\": true}}', TOOLS)\nassert call is None and err.startswith(\"wrong_type\")\ncall, err = parse_call('{\"tool\": \"set_checkbox\", \"args\": {\"selector\": \"#a\", \"checked\": true}}', TOOLS)\nassert err is None"
   },
   {
    "name": "float 参数接受整数",
    "code": "call, err = parse_call('{\"tool\": \"wait\", \"args\": {\"seconds\": 2}}', TOOLS)\nassert err is None and call[\"args\"][\"seconds\"] == 2\ncall, err = parse_call('{\"tool\": \"wait\", \"args\": {\"seconds\": 0.5}}', TOOLS)\nassert err is None"
   }
  ]
 },
 {
  "id": "s5-retry",
  "stage": 5,
  "level": "中",
  "title": "超时和重试",
  "statement": "<p>工具调用会偶尔超时。实现 <code>call_with_retry(fn, max_attempts=3, base_delay=1.0, sleep=time.sleep, retryable=(TimeoutError,))</code>：</p>\n<ul>\n  <li>调用 <code>fn()</code>，成功就返回它的结果。</li>\n  <li>抛出 <code>retryable</code> 里的异常时：如果还有剩余次数，先 <code>sleep(base_delay * 2 ** (已失败次数 − 1))</code>，再试一次。等待时间依次是 1、2、4…倍的 <code>base_delay</code>。</li>\n  <li>第 <code>max_attempts</code> 次还失败，把这次的异常抛出去，之后<strong>不要</strong>再 sleep。</li>\n  <li>其他异常说明不是暂时性的问题，立刻抛出，不重试。</li>\n</ul>\n<p><code>sleep</code> 是传进来的，测试里会换成一个只记录时长的函数，所以不会真的等。</p>\n<h3>示例</h3>\n<pre><code># fn 前两次超时，第三次返回 \"ok\"\ncall_with_retry(fn)          # 返回 \"ok\"，中间 sleep(1.0)、sleep(2.0)</code></pre>",
  "starter": "import time\n\n\ndef call_with_retry(fn, max_attempts=3, base_delay=1.0, sleep=time.sleep, retryable=(TimeoutError,)):\n    pass\n",
  "solution": "import time\n\n\ndef call_with_retry(fn, max_attempts=3, base_delay=1.0, sleep=time.sleep, retryable=(TimeoutError,)):\n    for attempt in range(1, max_attempts + 1):\n        try:\n            return fn()\n        except retryable:\n            if attempt == max_attempts:\n                raise\n            sleep(base_delay * 2 ** (attempt - 1))\n",
  "setup": "def flaky(*outcomes):\n    \"\"\"按顺序给出结果：异常类就抛出，其他值就返回。\"\"\"\n    it = iter(outcomes)\n    calls = []\n    def fn():\n        calls.append(1)\n        o = next(it)\n        if isinstance(o, type) and issubclass(o, BaseException):\n            raise o(\"模拟的失败\")\n        return o\n    fn.calls = calls\n    return fn",
  "tests": [
   {
    "name": "第一次就成功，不 sleep",
    "code": "delays = []\nfn = flaky(\"ok\")\nassert call_with_retry(fn, sleep=delays.append) == \"ok\"\nassert len(fn.calls) == 1 and delays == []"
   },
   {
    "name": "超时两次后成功，等待时间翻倍",
    "code": "delays = []\nfn = flaky(TimeoutError, TimeoutError, \"ok\")\nassert call_with_retry(fn, sleep=delays.append) == \"ok\"\nassert len(fn.calls) == 3 and delays == [1.0, 2.0]"
   },
   {
    "name": "base_delay 可以调",
    "code": "delays = []\nfn = flaky(TimeoutError, TimeoutError, TimeoutError, \"ok\")\nassert call_with_retry(fn, max_attempts=5, base_delay=0.5, sleep=delays.append) == \"ok\"\nassert delays == [0.5, 1.0, 2.0]"
   },
   {
    "name": "次数用完后抛出，最后一次之后不 sleep",
    "code": "delays = []\nfn = flaky(TimeoutError, TimeoutError, TimeoutError, \"ok\")\ntry:\n    call_with_retry(fn, max_attempts=3, sleep=delays.append)\nexcept TimeoutError:\n    pass\nelse:\n    assert False, \"三次都超时，应该抛出 TimeoutError\"\nassert len(fn.calls) == 3 and delays == [1.0, 2.0]"
   },
   {
    "name": "不可重试的异常立刻抛出",
    "code": "delays = []\nfn = flaky(ValueError, \"ok\")\ntry:\n    call_with_retry(fn, sleep=delays.append)\nexcept ValueError:\n    pass\nelse:\n    assert False, \"ValueError 不该被吞掉\"\nassert len(fn.calls) == 1 and delays == []"
   },
   {
    "name": "retryable 可以指定多种异常",
    "code": "delays = []\nfn = flaky(ConnectionError, TimeoutError, 42)\nassert call_with_retry(fn, sleep=delays.append, retryable=(TimeoutError, ConnectionError)) == 42\nassert len(fn.calls) == 3"
   }
  ]
 },
 {
  "id": "s5-trim",
  "stage": 5,
  "level": "易",
  "title": "把对话历史裁进预算",
  "statement": "<p>Agent 每走一步，上下文就长一截。放不下时要丢掉旧的。</p>\n<p>实现 <code>trim(messages, budget, count=len)</code>：</p>\n<ul>\n  <li><code>messages</code>：<code>{\"role\": ..., \"content\": ...}</code> 的 list，按时间顺序。</li>\n  <li>一条消息的长度是 <code>count(message[\"content\"])</code>，默认按字符数算。</li>\n  <li>第一条如果是 <code>\"system\"</code>，一定保留，它的长度也算在预算里。</li>\n  <li>其余的从<strong>最新</strong>的往回看，一条条加进来，直到下一条放不下为止。遇到放不下的就停，不要跳过它去拿更早的：历史中间不能有洞。</li>\n</ul>\n<p>返回保留下来的消息，顺序和原来一样。不要修改传进来的 list。</p>\n<h3>示例</h3>\n<pre><code>msgs = [{\"role\": \"system\", \"content\": \"ab\"},\n        {\"role\": \"user\", \"content\": \"cccc\"},\n        {\"role\": \"assistant\", \"content\": \"dd\"},\n        {\"role\": \"user\", \"content\": \"e\"}]\ntrim(msgs, budget=5)   # 保留 system、\"dd\"、\"e\"：2 + 2 + 1 = 5</code></pre>",
  "starter": "def trim(messages, budget, count=len):\n    pass\n",
  "solution": "def trim(messages, budget, count=len):\n    if not messages:\n        return []\n    head = [messages[0]] if messages[0][\"role\"] == \"system\" else []\n    used = sum(count(m[\"content\"]) for m in head)\n    kept = []\n    for m in reversed(messages[len(head):]):\n        size = count(m[\"content\"])\n        if used + size > budget:\n            break                                  # 放不下就停，不往更早的跳\n        kept.append(m)\n        used += size\n    return head + kept[::-1]\n",
  "setup": "def M(role, content):\n    return {\"role\": role, \"content\": content}\nSYS = M(\"system\", \"ab\")\nmsgs = [SYS, M(\"user\", \"cccc\"), M(\"assistant\", \"dd\"), M(\"user\", \"e\")]",
  "tests": [
   {
    "name": "放得下就原样返回",
    "code": "assert trim(msgs, 100) == msgs"
   },
   {
    "name": "先丢最旧的",
    "code": "assert trim(msgs, 5) == [SYS, M(\"assistant\", \"dd\"), M(\"user\", \"e\")]"
   },
   {
    "name": "system 一定保留",
    "code": "assert trim(msgs, 2) == [SYS]\nassert trim(msgs, 3) == [SYS, M(\"user\", \"e\")]"
   },
   {
    "name": "遇到放不下的就停，中间不留洞",
    "code": "m = [SYS, M(\"user\", \"aaaa\"), M(\"assistant\", \"b\" * 10), M(\"user\", \"cc\")]\nassert trim(m, 9) == [SYS, M(\"user\", \"cc\")], \"不能跳过放不下的那条去拿更早的\""
   },
   {
    "name": "没有 system 消息",
    "code": "m = [M(\"user\", \"aaa\"), M(\"assistant\", \"bb\"), M(\"user\", \"c\")]\nassert trim(m, 3) == [M(\"assistant\", \"bb\"), M(\"user\", \"c\")]"
   },
   {
    "name": "可以换计数方式",
    "code": "m = [SYS, M(\"user\", \"one two three\"), M(\"assistant\", \"four five\"), M(\"user\", \"six\")]\nwords = lambda s: len(s.split())\nassert trim(m, 4, count=words) == [SYS, M(\"assistant\", \"four five\"), M(\"user\", \"six\")]"
   },
   {
    "name": "不修改传进来的 list",
    "code": "before = list(msgs)\ntrim(msgs, 3)\nassert msgs == before and len(msgs) == 4\nassert trim([], 10) == []"
   }
  ]
 },
 {
  "id": "s5-agent-loop",
  "stage": 5,
  "level": "难",
  "title": "验：跑一轮任务，把每一步记下来",
  "statement": "<p>这是第 5 步的验收：写 agent 的主循环，并且每一步留一条记录。</p>\n<p>实现 <code>run_episode(model, env, tools, max_steps=10)</code>，返回日志：一个 list，每步一个 dict。</p>\n<ul>\n  <li><code>env.reset()</code> 返回初始观察；<code>env.observe()</code> 返回当前观察。观察是字符串。</li>\n  <li><code>model(history, obs)</code> 返回模型的原始输出（字符串），约定是 JSON：<code>{\"tool\": \"名字\", \"args\": {...}}</code>，没有 <code>\"args\"</code> 时当作 <code>{}</code>。</li>\n  <li><code>tools[名字](env, **args)</code> 执行动作，返回一个字符串。</li>\n</ul>\n<p>每一步：</p>\n<ol>\n  <li>调用 <code>model(history, obs)</code> 得到 <code>raw</code>。</li>\n  <li>解析并执行，得到这一步的记录：\n<pre><code>{\"step\": 步号（从 0 开始）, \"obs\": 模型这一步看到的观察, \"raw\": 模型的原始输出,\n \"action\": 解析出的 dict（解析失败时是 None）,\n \"result\": 字符串, \"error\": None 或错误名}</code></pre>\n    <ul>\n      <li>解析失败（不是 JSON、不是对象、没有 <code>\"tool\"</code>）：<code>error = \"invalid_json\"</code>。</li>\n      <li>工具名是 <code>\"done\"</code>：<code>result = \"done\"</code>，记录这一步后结束。</li>\n      <li>工具名不在 <code>tools</code> 里：<code>error = \"unknown_tool\"</code>。</li>\n      <li>工具抛异常：<code>error</code> 是异常的类名，比如 <code>\"TimeoutError\"</code>。</li>\n      <li>有错误时 <code>result = \"error: \" + error</code>；否则 <code>result</code> 是工具的返回值。</li>\n    </ul>\n  </li>\n  <li>把 <code>(obs, raw, result)</code> 追加到 <code>history</code>，这样模型下一步能看到刚才发生了什么。</li>\n  <li>如果没结束，<code>obs = env.observe()</code>，继续。最多走 <code>max_steps</code> 步。</li>\n</ol>\n<p>任何一步出错都<strong>不能让循环崩掉</strong>：记下来，把错误交给模型，继续走。</p>",
  "starter": "import json\n\n\ndef run_episode(model, env, tools, max_steps=10):\n    # 返回日志：每步一个 dict\n    pass\n",
  "solution": "import json\n\n\ndef run_episode(model, env, tools, max_steps=10):\n    obs = env.reset()\n    history, log = [], []\n    for step in range(max_steps):\n        raw = model(history, obs)\n        rec = {\"step\": step, \"obs\": obs, \"raw\": raw, \"action\": None, \"result\": None, \"error\": None}\n        done = False\n\n        try:\n            action = json.loads(raw)\n            if not isinstance(action, dict) or \"tool\" not in action:\n                raise ValueError(\"不是工具调用\")\n        except Exception:\n            rec[\"error\"] = \"invalid_json\"\n        else:\n            rec[\"action\"] = action\n            name = action[\"tool\"]\n            if name == \"done\":\n                rec[\"result\"] = \"done\"\n                done = True\n            elif name not in tools:\n                rec[\"error\"] = \"unknown_tool\"\n            else:\n                try:\n                    rec[\"result\"] = tools[name](env, **action.get(\"args\", {}))\n                except Exception as exc:\n                    rec[\"error\"] = type(exc).__name__\n\n        if rec[\"error\"]:\n            rec[\"result\"] = \"error: \" + rec[\"error\"]\n        log.append(rec)\n        history.append((obs, raw, rec[\"result\"]))\n        if done:\n            break\n        obs = env.observe()\n    return log\n",
  "setup": "class Env:\n    def reset(self):\n        self.n = 0\n        return \"n=0\"\n    def observe(self):\n        return f\"n={self.n}\"\n\ndef add(env, by):\n    env.n += by\n    return \"ok\"\n\ndef boom(env):\n    raise TimeoutError(\"页面没加载出来\")\n\nTOOLS = {\"add\": add, \"boom\": boom}\nADD2 = '{\"tool\": \"add\", \"args\": {\"by\": 2}}'\nADD3 = '{\"tool\": \"add\", \"args\": {\"by\": 3}}'\nDONE = '{\"tool\": \"done\"}'\n\ndef scripted(*outputs):\n    \"\"\"按顺序输出，并记下每次被调用时看到了什么。\"\"\"\n    it = iter(outputs)\n    seen = []\n    def model(history, obs):\n        seen.append({\"n_history\": len(history), \"obs\": obs, \"last\": history[-1] if history else None})\n        return next(it)\n    model.seen = seen\n    return model",
  "tests": [
   {
    "name": "正常跑完一个任务",
    "code": "env = Env()\nlog = run_episode(scripted(ADD2, ADD3, DONE), env, TOOLS)\nassert env.n == 5\nassert [r[\"step\"] for r in log] == [0, 1, 2]\nassert log[-1][\"result\"] == \"done\""
   },
   {
    "name": "每条记录的字段齐全",
    "code": "log = run_episode(scripted(ADD2, ADD3, DONE), Env(), TOOLS)\nfor r in log:\n    assert set(r) == {\"step\", \"obs\", \"raw\", \"action\", \"result\", \"error\"}, set(r)\nassert [r[\"obs\"] for r in log] == [\"n=0\", \"n=2\", \"n=5\"], \"obs 是模型在这一步看到的观察\"\nassert log[0][\"raw\"] == ADD2\nassert log[0][\"action\"] == {\"tool\": \"add\", \"args\": {\"by\": 2}}\nassert log[0][\"result\"] == \"ok\" and log[0][\"error\"] is None"
   },
   {
    "name": "解析失败时记下来，继续走",
    "code": "env = Env()\nlog = run_episode(scripted(\"我想加 2\", ADD2, DONE), env, TOOLS)\nassert len(log) == 3 and env.n == 2\nassert log[0][\"error\"] == \"invalid_json\" and log[0][\"action\"] is None\nassert log[0][\"result\"] == \"error: invalid_json\""
   },
   {
    "name": "未知工具",
    "code": "log = run_episode(scripted('{\"tool\": \"multiply\", \"args\": {\"by\": 2}}', DONE), Env(), TOOLS)\nassert log[0][\"error\"] == \"unknown_tool\" and log[0][\"result\"] == \"error: unknown_tool\"\nassert log[0][\"action\"] == {\"tool\": \"multiply\", \"args\": {\"by\": 2}}"
   },
   {
    "name": "工具抛异常时循环不崩",
    "code": "env = Env()\nlog = run_episode(scripted('{\"tool\": \"boom\"}', ADD3, DONE), env, TOOLS)\nassert log[0][\"error\"] == \"TimeoutError\" and log[0][\"result\"] == \"error: TimeoutError\"\nassert len(log) == 3 and env.n == 3"
   },
   {
    "name": "参数不对也算工具执行出错",
    "code": "log = run_episode(scripted('{\"tool\": \"add\", \"args\": {\"amount\": 2}}', DONE), Env(), TOOLS)\nassert log[0][\"error\"] == \"TypeError\""
   },
   {
    "name": "模型下一步能看到上一步的结果",
    "code": "m = scripted(\"胡说\", ADD2, DONE)\nrun_episode(m, Env(), TOOLS)\nassert [s[\"n_history\"] for s in m.seen] == [0, 1, 2]\nassert m.seen[1][\"last\"] == (\"n=0\", \"胡说\", \"error: invalid_json\")\nassert m.seen[2][\"last\"] == (\"n=0\", ADD2, \"ok\")\nassert m.seen[2][\"obs\"] == \"n=2\""
   },
   {
    "name": "最多走 max_steps 步",
    "code": "env = Env()\nlog = run_episode(scripted(*[ADD2] * 20), env, TOOLS, max_steps=4)\nassert len(log) == 4 and env.n == 8"
   },
   {
    "name": "done 之后不再调用模型",
    "code": "m = scripted(DONE, ADD2, ADD2)\nenv = Env()\nlog = run_episode(m, env, TOOLS)\nassert len(log) == 1 and len(m.seen) == 1 and env.n == 0"
   }
  ]
 },
 {
  "id": "s6-checker",
  "stage": 6,
  "level": "易",
  "title": "独立检查器",
  "statement": "<p>任务有没有完成，由检查器读环境的最终状态来判定，不听 agent 自己怎么说。</p>\n<p><code>check(final_state, expected)</code>：</p>\n<ul>\n  <li><code>final_state</code>：任务结束时环境的状态，一个可以嵌套的 dict。</li>\n  <li><code>expected</code>：<code>{路径: 期望的值}</code>。路径用点号一层层往下取，比如 <code>\"cart.total\"</code> 指 <code>final_state[\"cart\"][\"total\"]</code>。</li>\n</ul>\n<p>返回 <code>(passed, reasons)</code>。<code>reasons</code> 是字符串的 list，每个不满足的条件一条，里面要带上那条路径。路径不存在也算不满足。全部满足时是 <code>(True, [])</code>。<code>final_state</code> 里多出来的字段不影响结果。</p>\n<p><code>score(run, expected)</code>：<code>run</code> 是 <code>{\"claimed_done\": bool, \"final_state\": {...}}</code>。返回任务是否成功，<strong>只</strong>取决于 <code>check</code> 的结果。</p>\n<h3>示例</h3>\n<pre><code>state = {\"cart\": {\"total\": 30, \"items\": [\"书\"]}, \"page\": \"checkout\"}\ncheck(state, {\"cart.total\": 30, \"page\": \"done\"})\n# (False, [\"page: 期望 'done'，实际 'checkout'\"])</code></pre>",
  "starter": "def check(final_state, expected):\n    # 返回 (passed, reasons)\n    pass\n\n\ndef score(run, expected):\n    pass\n",
  "solution": "MISSING = object()\n\n\ndef lookup(state, path):\n    current = state\n    for part in path.split(\".\"):\n        if not isinstance(current, dict) or part not in current:\n            return MISSING\n        current = current[part]\n    return current\n\n\ndef check(final_state, expected):\n    reasons = []\n    for path, want in expected.items():\n        got = lookup(final_state, path)\n        if got is MISSING:\n            reasons.append(f\"{path}: 期望 {want!r}，实际不存在\")\n        elif got != want:\n            reasons.append(f\"{path}: 期望 {want!r}，实际 {got!r}\")\n    return len(reasons) == 0, reasons\n\n\ndef score(run, expected):\n    passed, _ = check(run[\"final_state\"], expected)\n    return passed\n",
  "setup": "state = {\"cart\": {\"total\": 30, \"items\": [\"书\"]}, \"page\": \"checkout\", \"user\": {\"address\": {\"city\": \"匹兹堡\"}}}",
  "tests": [
   {
    "name": "全部满足",
    "code": "assert check(state, {\"page\": \"checkout\", \"cart.total\": 30}) == (True, [])"
   },
   {
    "name": "一个条件不满足",
    "code": "passed, reasons = check(state, {\"cart.total\": 30, \"page\": \"done\"})\nassert passed is False\nassert len(reasons) == 1 and \"page\" in reasons[0]"
   },
   {
    "name": "每个不满足的条件各一条原因",
    "code": "passed, reasons = check(state, {\"cart.total\": 99, \"page\": \"done\", \"cart.items\": [\"书\"]})\nassert passed is False and len(reasons) == 2\nassert any(\"cart.total\" in r for r in reasons) and any(\"page\" in r for r in reasons)"
   },
   {
    "name": "多层路径",
    "code": "assert check(state, {\"user.address.city\": \"匹兹堡\"}) == (True, [])\nassert check(state, {\"user.address.city\": \"纽约\"})[0] is False"
   },
   {
    "name": "路径不存在算不满足，不报错",
    "code": "for path in [\"order_id\", \"cart.coupon\", \"page.title\", \"user.address.zip.code\"]:\n    passed, reasons = check(state, {path: 1})\n    assert passed is False and len(reasons) == 1 and path in reasons[0], path"
   },
   {
    "name": "状态里多出来的字段不影响",
    "code": "assert check(state, {\"page\": \"checkout\"})[0] is True\nassert check(state, {}) == (True, [])"
   },
   {
    "name": "agent 说完成了，状态不对，算失败",
    "code": "run = {\"claimed_done\": True, \"final_state\": {\"page\": \"checkout\"}}\nassert score(run, {\"page\": \"done\"}) is False"
   },
   {
    "name": "agent 没说完成，状态对了，算成功",
    "code": "run = {\"claimed_done\": False, \"final_state\": {\"page\": \"done\"}}\nassert score(run, {\"page\": \"done\"}) is True"
   }
  ]
 },
 {
  "id": "s6-passk",
  "stage": 6,
  "level": "易",
  "title": "pass@k 和 pass^k",
  "statement": "<p>同一个任务跑 k 次，结果不一定每次都一样。两个指标回答两个不同的问题。</p>\n<p><code>trials</code> 是一个 list，每个任务一行，每行是 k 个 <code>bool</code>：这个任务的 k 次运行各自成没成功。实现三个函数，都返回 0 到 1 之间的小数：</p>\n<ul>\n  <li><code>success_rate(trials)</code>：所有运行里成功的比例。</li>\n  <li><code>pass_at_k(trials)</code>：k 次里<strong>至少一次</strong>成功的任务比例。量的是能力上限。</li>\n  <li><code>pass_hat_k(trials)</code>：k 次<strong>全部</strong>成功的任务比例。量的是可靠性。</li>\n</ul>\n<h3>示例</h3>\n<pre><code>trials = [[True,  True,  True ],\n          [True,  False, False],\n          [False, False, False],\n          [True,  True,  False]]\nsuccess_rate(trials)   # 0.5\npass_at_k(trials)      # 0.75\npass_hat_k(trials)     # 0.25</code></pre>\n<p>同一组运行，说成「75% 的任务能做」和「25% 的任务靠得住」都对。报告结果时要说清用的是哪个。</p>",
  "starter": "def success_rate(trials):\n    pass\n\n\ndef pass_at_k(trials):\n    pass\n\n\ndef pass_hat_k(trials):\n    pass\n",
  "solution": "def success_rate(trials):\n    flat = [t for task in trials for t in task]\n    return sum(flat) / len(flat)\n\n\ndef pass_at_k(trials):\n    return sum(any(task) for task in trials) / len(trials)\n\n\ndef pass_hat_k(trials):\n    return sum(all(task) for task in trials) / len(trials)\n",
  "setup": "T, F = True, False\ntrials = [[T, T, T], [T, F, F], [F, F, F], [T, T, F]]",
  "tests": [
   {
    "name": "示例里的数",
    "code": "assert abs(success_rate(trials) - 0.5) < 1e-9\nassert abs(pass_at_k(trials) - 0.75) < 1e-9\nassert abs(pass_hat_k(trials) - 0.25) < 1e-9"
   },
   {
    "name": "k = 1 时三个数相等",
    "code": "one = [[T], [F], [T], [T]]\nassert success_rate(one) == pass_at_k(one) == pass_hat_k(one) == 0.75"
   },
   {
    "name": "pass^k ≤ 成功率 ≤ pass@k",
    "code": "import random\nrandom.seed(0)\nfor _ in range(20):\n    tr = [[random.random() < 0.6 for _ in range(4)] for _ in range(15)]\n    assert pass_hat_k(tr) <= success_rate(tr) + 1e-12 <= pass_at_k(tr) + 2e-12"
   },
   {
    "name": "全部成功和全部失败",
    "code": "assert pass_hat_k([[T, T], [T, T]]) == 1.0 and pass_at_k([[T, T], [T, T]]) == 1.0\nassert pass_hat_k([[F, F], [F, F]]) == 0.0 and pass_at_k([[F, F], [F, F]]) == 0.0"
   },
   {
    "name": "成功率一样，可靠性可以差很多",
    "code": "steady = [[T, T], [T, T], [F, F], [F, F]]      # 一半任务稳定成功\nflaky = [[T, F], [F, T], [T, F], [F, T]]       # 每个任务都时好时坏\nassert success_rate(steady) == success_rate(flaky) == 0.5\nassert pass_hat_k(steady) == 0.5 and pass_hat_k(flaky) == 0.0\nassert pass_at_k(steady) == 0.5 and pass_at_k(flaky) == 1.0"
   }
  ]
 },
 {
  "id": "s6-mcnemar",
  "stage": 6,
  "level": "中",
  "title": "两版 harness，配对比较",
  "statement": "<p>固定模型，两版 harness 跑同一批任务。结果是配对的：对每个任务，你知道 A 过没过、B 过没过。</p>\n<p>两版都过或都没过的任务，分不出谁好。差别全在只有一版通过的那些任务上。</p>\n<p>实现 <code>paired(pass_a, pass_b)</code>，两个等长的 <code>bool</code> list。返回一个 dict：</p>\n<ul>\n  <li><code>\"only_a\"</code>：只有 A 通过的任务数，记作 b。</li>\n  <li><code>\"only_b\"</code>：只有 B 通过的任务数，记作 c。</li>\n  <li><code>\"p_value\"</code>：McNemar 精确检验的双侧 p 值。如果两版一样好，这 <code>n = b + c</code> 个任务每个都是一半一半：\n<pre><code>p = min(1, 2 × Σ_{i=0}^{min(b, c)} C(n, i) / 2ⁿ)</code></pre>\n    <code>n = 0</code> 时 <code>p = 1.0</code>。</li>\n</ul>\n<p>两个 list 长度不同时抛出 <code>ValueError</code>。</p>\n<h3>示例</h3>\n<pre><code># 50 个任务：A 过 31 个（62%），B 过 33 个（66%）\n# 其中只有 A 过的 3 个，只有 B 过的 5 个\npaired(pass_a, pass_b)   # {\"only_a\": 3, \"only_b\": 5, \"p_value\": 0.7266}</code></pre>\n<p>p 值 0.73：这 4 个点的差距，在两版一样好的情况下也很容易出现。</p>",
  "starter": "from math import comb\n\n\ndef paired(pass_a, pass_b):\n    pass\n",
  "solution": "from math import comb\n\n\ndef paired(pass_a, pass_b):\n    if len(pass_a) != len(pass_b):\n        raise ValueError(\"两版的任务数不一致\")\n    only_a = sum(1 for a, b in zip(pass_a, pass_b) if a and not b)\n    only_b = sum(1 for a, b in zip(pass_a, pass_b) if b and not a)\n    n = only_a + only_b\n    if n == 0:\n        p = 1.0\n    else:\n        tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))\n        p = min(1.0, 2 * tail / 2 ** n)\n    return {\"only_a\": only_a, \"only_b\": only_b, \"p_value\": p}\n",
  "setup": "def mk(both, only_a, only_b, neither):\n    a = [True] * both + [True] * only_a + [False] * only_b + [False] * neither\n    b = [True] * both + [False] * only_a + [True] * only_b + [False] * neither\n    return a, b",
  "tests": [
   {
    "name": "62% 对 66%，50 个任务",
    "code": "a, b = mk(28, 3, 5, 14)\nassert sum(a) == 31 and sum(b) == 33 and len(a) == 50\nr = paired(a, b)\nassert r[\"only_a\"] == 3 and r[\"only_b\"] == 5\nassert abs(r[\"p_value\"] - 0.7265625) < 1e-9"
   },
   {
    "name": "一边倒的差别",
    "code": "r = paired(*mk(20, 0, 10, 20))\nassert r[\"only_a\"] == 0 and r[\"only_b\"] == 10\nassert abs(r[\"p_value\"] - 0.001953125) < 1e-12"
   },
   {
    "name": "没有不一致的任务时 p 是 1",
    "code": "r = paired(*mk(30, 0, 0, 20))\nassert r == {\"only_a\": 0, \"only_b\": 0, \"p_value\": 1.0}"
   },
   {
    "name": "p 值不超过 1",
    "code": "r = paired(*mk(0, 2, 2, 0))\nassert r[\"p_value\"] == 1.0"
   },
   {
    "name": "A 和 B 对调，p 值不变",
    "code": "a, b = mk(10, 1, 7, 5)\nr1, r2 = paired(a, b), paired(b, a)\nassert r1[\"only_a\"] == r2[\"only_b\"] == 1 and r1[\"only_b\"] == r2[\"only_a\"] == 7\nassert abs(r1[\"p_value\"] - r2[\"p_value\"]) < 1e-12\nassert abs(r1[\"p_value\"] - 0.0703125) < 1e-9"
   },
   {
    "name": "总成功率一样，配对结果可以不一样",
    "code": "same = paired(*mk(30, 0, 0, 20))\nswapped = paired(*mk(20, 10, 10, 10))\nassert same[\"only_a\"] == 0 and swapped[\"only_a\"] == 10 and swapped[\"only_b\"] == 10"
   },
   {
    "name": "任务数不一致时抛出 ValueError",
    "code": "try:\n    paired([True, False], [True])\nexcept ValueError:\n    pass\nelse:\n    assert False, \"长度不一致时应该抛出 ValueError\""
   }
  ]
 },
 {
  "id": "s6-triage",
  "stage": 6,
  "level": "中",
  "title": "验：把失败分成规划、工具执行、评分",
  "statement": "<p>这是第 6 步的验收。读完日志之后，每一次运行有这些人工确认过的信息：</p>\n<pre><code>run = {\n  \"checker_passed\": bool,     # 检查器判的结果\n  \"truly_done\": bool,         # 人看过最终状态：任务到底完成没有\n  \"steps\": [                  # 每一步\n    {\"intended_ok\": bool,     # 模型想做的这个动作，方向对不对\n     \"executed_ok\": bool},    # 这个动作有没有按意图执行成功\n  ],\n}</code></pre>\n<p>实现 <code>triage(run)</code>，返回四个字符串之一。按顺序判断，前面的优先：</p>\n<ol>\n  <li><code>\"grading\"</code>：检查器判的和实际情况不一致（做完了被判失败，或没做完被判成功）。</li>\n  <li><code>\"success\"</code>：确实完成了，检查器也判了通过。</li>\n  <li><code>\"tool\"</code>：有某一步方向是对的（<code>intended_ok</code>），却没执行成功（<code>executed_ok</code> 为假）。</li>\n  <li><code>\"planning\"</code>：其余的失败。动作要么都执行成功了但任务没完成，要么失败的动作本身方向就不对。</li>\n</ol>\n<p>再实现 <code>table(runs)</code>：返回 <code>{类别: 次数}</code> 的 dict，只包含出现过的类别。</p>\n<p>顺序是有道理的。评分排第一：检查器有错，其他的数字都不可信。工具执行排在规划前面：动作没落地时，还看不出计划本身行不行。</p>",
  "starter": "def triage(run):\n    pass\n\n\ndef table(runs):\n    pass\n",
  "solution": "from collections import Counter\n\n\ndef triage(run):\n    if run[\"checker_passed\"] != run[\"truly_done\"]:\n        return \"grading\"\n    if run[\"truly_done\"]:\n        return \"success\"\n    if any(s[\"intended_ok\"] and not s[\"executed_ok\"] for s in run[\"steps\"]):\n        return \"tool\"\n    return \"planning\"\n\n\ndef table(runs):\n    return dict(Counter(triage(r) for r in runs))\n",
  "setup": "def S(intended_ok, executed_ok):\n    return {\"intended_ok\": intended_ok, \"executed_ok\": executed_ok}\n\ndef R(checker_passed, truly_done, *steps):\n    return {\"checker_passed\": checker_passed, \"truly_done\": truly_done, \"steps\": list(steps)}\n\nGOOD, BAD_PLAN, BAD_EXEC, BAD_BOTH = S(True, True), S(False, True), S(True, False), S(False, False)",
  "tests": [
   {
    "name": "做完了，也判对了",
    "code": "assert triage(R(True, True, GOOD, GOOD)) == \"success\""
   },
   {
    "name": "做完了却被判失败",
    "code": "assert triage(R(False, True, GOOD, GOOD)) == \"grading\""
   },
   {
    "name": "没做完却被判通过",
    "code": "assert triage(R(True, False, GOOD, BAD_PLAN)) == \"grading\""
   },
   {
    "name": "方向对，动作没落地",
    "code": "assert triage(R(False, False, GOOD, BAD_EXEC, GOOD)) == \"tool\""
   },
   {
    "name": "动作都执行了，任务没完成",
    "code": "assert triage(R(False, False, GOOD, GOOD)) == \"planning\"\nassert triage(R(False, False, GOOD, BAD_PLAN)) == \"planning\"\nassert triage(R(False, False)) == \"planning\""
   },
   {
    "name": "方向本来就错的动作失败了，不算工具问题",
    "code": "assert triage(R(False, False, GOOD, BAD_BOTH)) == \"planning\""
   },
   {
    "name": "评分优先于工具，工具优先于规划",
    "code": "assert triage(R(True, False, BAD_EXEC)) == \"grading\"\nassert triage(R(False, False, BAD_PLAN, BAD_EXEC)) == \"tool\""
   },
   {
    "name": "汇总成表",
    "code": "runs = [R(True, True, GOOD), R(True, True, GOOD), R(False, True, GOOD),\n        R(False, False, BAD_EXEC), R(False, False, GOOD), R(False, False, BAD_PLAN)]\nassert table(runs) == {\"success\": 2, \"grading\": 1, \"tool\": 1, \"planning\": 2}\nassert table([R(True, True, GOOD)]) == {\"success\": 1}"
   }
  ]
 }
];
