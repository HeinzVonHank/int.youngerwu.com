"""Problems for stages 1-3. Each entry: id, stage, level, title, statement (HTML),
starter, solution, optional setup (prepended to every test), tests [(name, code)]."""

PROBLEMS = []


def P(**kw):
    kw.setdefault("setup", "")
    PROBLEMS.append(kw)


# ───────────────────────── 1 Transformer ─────────────────────────

P(
    id="s1-tokenizer", stage=1, level="易", title="字符级 tokenizer",
    statement="""
<p>把文本变成整数 id，再变回来。实现三个函数：</p>
<ul>
  <li><code>build_vocab(text)</code>：词表是 <code>text</code> 里出现过的字符，按 <code>sorted</code> 的顺序从 0 编号。返回 <code>(stoi, itos)</code>，两个 dict，分别是字符到 id 和 id 到字符。</li>
  <li><code>encode(text, stoi)</code>：返回 id 的 list。</li>
  <li><code>decode(ids, itos)</code>：返回字符串。</li>
</ul>
<h3>示例</h3>
<pre><code>stoi, itos = build_vocab("hello")
stoi                    # {'e': 0, 'h': 1, 'l': 2, 'o': 3}
encode("hello", stoi)   # [1, 0, 2, 2, 3]
decode([1, 0, 2, 2, 3], itos)   # "hello"</code></pre>
""",
    starter='''def build_vocab(text):
    # 返回 (stoi, itos)
    pass


def encode(text, stoi):
    pass


def decode(ids, itos):
    pass
''',
    solution='''def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)
''',
    tests=[
        ("词表按字符排序，从 0 编号", '''stoi, itos = build_vocab("hello")
assert stoi == {"e": 0, "h": 1, "l": 2, "o": 3}
assert itos == {0: "e", 1: "h", 2: "l", 3: "o"}'''),
        ("encode 返回 id 的 list", '''stoi, itos = build_vocab("hello")
assert encode("hello", stoi) == [1, 0, 2, 2, 3]'''),
        ("encode 再 decode 得到原文", '''s = "to be, or not to be"
stoi, itos = build_vocab(s)
assert decode(encode(s, stoi), itos) == s'''),
        ("空格和中文也是字符", '''s = "小 GPT 小"
stoi, itos = build_vocab(s)
assert len(stoi) == 5
assert decode(encode(s, stoi), itos) == s'''),
    ],
)

P(
    id="s1-embedding", stage=1, level="易", title="Token 和位置 embedding",
    statement="""
<p>Embedding 是查表。实现 <code>embed(ids, tok_emb, pos_emb)</code>：</p>
<ul>
  <li><code>ids</code>：整数数组，形状 <code>(B, T)</code>。</li>
  <li><code>tok_emb</code>：形状 <code>(V, C)</code>，第 i 行是 token i 的向量。</li>
  <li><code>pos_emb</code>：形状 <code>(max_T, C)</code>，第 t 行是位置 t 的向量，保证 <code>T ≤ max_T</code>。</li>
</ul>
<p>返回形状 <code>(B, T, C)</code> 的数组：每个位置是它的 token 向量加上它的位置向量。</p>
<h3>示例</h3>
<pre><code>tok_emb = [[1, 1], [2, 2], [3, 3]]      # V = 3, C = 2
pos_emb = [[10, 0], [20, 0]]            # max_T = 2
embed([[2, 0]], tok_emb, pos_emb)       # [[[13, 3], [21, 1]]]</code></pre>
""",
    starter='''import numpy as np


def embed(ids, tok_emb, pos_emb):
    pass
''',
    solution='''import numpy as np


def embed(ids, tok_emb, pos_emb):
    ids = np.asarray(ids)
    T = ids.shape[1]
    return np.asarray(tok_emb)[ids] + np.asarray(pos_emb)[:T]
''',
    setup='''tok = np.arange(12, dtype=float).reshape(4, 3)
pos = np.array([[100.0, 0, 0], [200.0, 0, 0], [300.0, 0, 0]])
ids = np.array([[0, 2], [3, 3]])''',
    tests=[
        ("输出形状是 (B, T, C)", '''assert embed(ids, tok, pos).shape == (2, 2, 3)'''),
        ("token 向量加位置向量", '''out = embed(ids, tok, pos)
assert np.allclose(out[0, 0], [100, 1, 2])
assert np.allclose(out[0, 1], [206, 7, 8])
assert np.allclose(out[1, 0], [109, 10, 11])'''),
        ("同一个 token 在不同位置得到不同向量", '''out = embed(ids, tok, pos)
assert not np.allclose(out[1, 0], out[1, 1])
assert np.allclose(out[1, 1] - out[1, 0], pos[1] - pos[0])'''),
        ("T 等于 max_T 时也能用", '''out = embed(np.array([[1, 1, 1]]), tok, pos)
assert out.shape == (1, 3, 3)
assert np.allclose(out[0, 2], tok[1] + pos[2])'''),
    ],
)

P(
    id="s1-attention", stage=1, level="中", title="Causal self-attention",
    statement="""
<p>实现单头的 causal self-attention：<code>causal_attention(x, Wq, Wk, Wv)</code>。</p>
<ul>
  <li><code>x</code>：形状 <code>(B, T, C)</code>。</li>
  <li><code>Wq</code>、<code>Wk</code>、<code>Wv</code>：形状 <code>(C, D)</code>。<code>q = x @ Wq</code>，k 和 v 同理。</li>
</ul>
<p>步骤：</p>
<ol>
  <li>分数 <code>scores = q @ kᵀ / √D</code>，形状 <code>(B, T, T)</code>。</li>
  <li>位置 t 只能看位置 ≤ t：把 <code>scores[:, t, s]</code> 里 <code>s &gt; t</code> 的部分设成 −∞。</li>
  <li>对最后一维做 softmax，得到权重 <code>w</code>。</li>
  <li>输出 <code>out = w @ v</code>，形状 <code>(B, T, D)</code>。</li>
</ol>
<p>返回 <code>(out, w)</code>。</p>
<h3>示例</h3>
<pre><code># Wq 和 Wk 全为 0 时所有分数都是 0，
# 位置 t 对前 t+1 个位置的权重都是 1 / (t+1)：
w[0]  # [[1,   0,   0  ],
      #  [1/2, 1/2, 0  ],
      #  [1/3, 1/3, 1/3]]</code></pre>
""",
    starter='''import numpy as np


def causal_attention(x, Wq, Wk, Wv):
    # 返回 (out, w)
    pass
''',
    solution='''import numpy as np


def causal_attention(x, Wq, Wk, Wv):
    T = x.shape[1]
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(k.shape[-1])
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    scores = scores - scores.max(-1, keepdims=True)      # 数值稳定
    w = np.exp(scores)
    w = w / w.sum(-1, keepdims=True)
    return w @ v, w
''',
    setup='''rng = np.random.default_rng(0)
x = rng.normal(size=(2, 5, 4))
Wq, Wk, Wv = (rng.normal(size=(4, 3)) for _ in range(3))''',
    tests=[
        ("输出和权重的形状", '''out, w = causal_attention(x, Wq, Wk, Wv)
assert out.shape == (2, 5, 3)
assert w.shape == (2, 5, 5)'''),
        ("每一行权重的和是 1", '''out, w = causal_attention(x, Wq, Wk, Wv)
assert np.allclose(w.sum(-1), 1)'''),
        ("未来位置的权重是 0", '''out, w = causal_attention(x, Wq, Wk, Wv)
assert np.allclose(np.triu(w[0], k=1), 0), "上三角应该全是 0"
assert np.allclose(w[:, 0, 0], 1), "位置 0 只能看自己"'''),
        ("分数全相等时是前缀平均", '''Z = np.zeros((4, 4))
out, w = causal_attention(x, Z, Z, np.eye(4))
for t in range(5):
    assert np.allclose(w[0, t, : t + 1], 1 / (t + 1))
    assert np.allclose(out[0, t], x[0, : t + 1].mean(0))'''),
        ("分数除以了 √D", '''x2 = np.array([[[1.0, 0, 0, 0], [2.0, 0, 0, 0]]])
out, w = causal_attention(x2, np.eye(4), np.eye(4), np.eye(4))
assert np.allclose(w[0, 1], [0.26894142, 0.73105858]), "位置 1 的分数应该是 [1, 2]，不是 [2, 4]"'''),
        ("改后面的输入，前面的输出不变", '''out, _ = causal_attention(x, Wq, Wk, Wv)
x2 = x.copy()
x2[:, 3:] += 1.0
out2, _ = causal_attention(x2, Wq, Wk, Wv)
assert np.allclose(out[:, :3], out2[:, :3])
assert not np.allclose(out[:, 3:], out2[:, 3:])'''),
    ],
)

P(
    id="s1-block", stage=1, level="中", title="Pre-norm block：归一化和残差",
    statement="""
<p>实现两个函数。</p>
<p><code>layer_norm(x, gamma, beta, eps=1e-5)</code>：在最后一维上归一化。</p>
<pre><code>y = gamma * (x - mean) / sqrt(var + eps) + beta</code></pre>
<p><code>mean</code> 和 <code>var</code> 都只在最后一维上算，所以每个位置各自归一化，互不影响。</p>
<p><code>block(x, attn, mlp, g1, b1, g2, b2)</code>：一个 pre-norm 的 Transformer block。<code>attn</code> 和 <code>mlp</code> 是传进来的函数，输入输出形状相同。</p>
<pre><code>x = x + attn(layer_norm(x, g1, b1))
x = x + mlp(layer_norm(x, g2, b2))</code></pre>
<p>注意两点：归一化在子层前面；残差加的是没归一化的 <code>x</code>。</p>
<h3>示例</h3>
<pre><code>layer_norm(np.array([[1., 2., 3.]]), 1, 0)   # [[-1.2247, 0, 1.2247]]</code></pre>
""",
    starter='''import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    pass


def block(x, attn, mlp, g1, b1, g2, b2):
    pass
''',
    solution='''import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def block(x, attn, mlp, g1, b1, g2, b2):
    x = x + attn(layer_norm(x, g1, b1))
    x = x + mlp(layer_norm(x, g2, b2))
    return x
''',
    setup='''rng = np.random.default_rng(0)
x = rng.normal(size=(2, 3, 8)) * 3 + 1
one, zero = np.ones(8), np.zeros(8)
def _ln(a):
    return (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)''',
    tests=[
        ("layer_norm 之后最后一维均值 0、方差 1", '''y = layer_norm(x, one, zero)
assert y.shape == x.shape
assert np.allclose(y.mean(-1), 0, atol=1e-6)
assert np.allclose(y.var(-1), 1, atol=1e-3)'''),
        ("gamma 缩放，beta 平移", '''y = layer_norm(x, one, zero)
assert np.allclose(layer_norm(x, 2 * one, 3 * one), 2 * y + 3)'''),
        ("每个位置各自归一化", '''y = layer_norm(x, one, zero)
x2 = x.copy()
x2[0, 0] += 10
assert np.allclose(layer_norm(x2, one, zero)[0, 1], y[0, 1])'''),
        ("子层输出 0 时，block 原样返回 x", '''out = block(x, lambda h: 0 * h, lambda h: 0 * h, one, zero, one, zero)
assert np.allclose(out, x)'''),
        ("归一化在子层前面，残差加的是原来的 x", '''out = block(x, lambda h: h, lambda h: 2 * h, one, zero, one, zero)
x1 = x + _ln(x)
assert np.allclose(out, x1 + 2 * _ln(x1), atol=1e-6)'''),
        ("两次归一化各用各的参数", '''out = block(x, lambda h: h, lambda h: h, one, zero, 3 * one, one)
x1 = x + _ln(x)
assert np.allclose(out, x1 + 3 * _ln(x1) + 1, atol=1e-6)'''),
    ],
)

P(
    id="s1-causal-check", stage=1, level="中", title="验：改未来 token，前面的 logits 不变",
    statement="""
<p>这是第 1 步的验收。给你一个模型的 <code>forward</code>，判断它是不是 causal 的。</p>
<p>实现 <code>check_causal(forward, vocab_size, T=12, t=6, seed=0)</code>：</p>
<ol>
  <li>用 <code>np.random.default_rng(seed)</code> 生成一条形状 <code>(1, T)</code> 的随机 token 序列 <code>x</code>。</li>
  <li>复制一份 <code>x2</code>，把位置 <code>t</code> 及之后的<strong>每个</strong> token 都换成另一个 token。</li>
  <li>分别算 <code>forward(x)</code> 和 <code>forward(x2)</code>，形状都是 <code>(1, T, V)</code>。</li>
  <li>位置 <code>&lt; t</code> 的 logits 完全相同，<strong>并且</strong>位置 <code>≥ t</code> 的 logits 有变化，才返回 <code>True</code>。</li>
</ol>
<p>第二个条件不能省。一个根本不看输入的模型，改什么它的输出都不变，只查第一个条件会把它放过去。</p>
<h3>示例</h3>
<pre><code>check_causal(causal_model, vocab_size=11)     # True
check_causal(leaky_model, vocab_size=11)      # False：前面的位置读到了后面
check_causal(constant_model, vocab_size=11)   # False：模型没在看输入</code></pre>
""",
    starter='''import numpy as np


def check_causal(forward, vocab_size, T=12, t=6, seed=0):
    # forward(ids) -> logits，ids 形状 (1, T)，logits 形状 (1, T, V)
    pass
''',
    solution='''import numpy as np


def check_causal(forward, vocab_size, T=12, t=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(vocab_size, size=(1, T))
    x2 = x.copy()
    x2[:, t:] = (x[:, t:] + 1) % vocab_size          # 每个都换成不同的 token
    a, b = forward(x), forward(x2)
    past_same = np.allclose(a[:, :t], b[:, :t], atol=1e-6)
    future_changed = not np.allclose(a[:, t:], b[:, t:], atol=1e-6)
    return bool(past_same and future_changed)
''',
    setup='''_rng = np.random.default_rng(1)
V = 11
_E = _rng.normal(size=(V, 8))
_W = _rng.normal(size=(8, V))

def causal(ids):                       # 每个位置是它之前所有 token 的平均
    h = np.cumsum(_E[ids], axis=1) / np.arange(1, ids.shape[1] + 1)[None, :, None]
    return h @ _W

def leaky(ids):                        # 每个位置都读了整条序列
    h = _E[ids]
    return (h.mean(1, keepdims=True) + 0 * h) @ _W

def blind(ids):                        # 完全不看输入
    return np.zeros(ids.shape + (V,))

def sneaky(ids):                       # 只有位置 0 偷看了最后一个 token
    out = causal(ids).copy()
    out[:, 0] += _E[ids[:, -1]] @ _W
    return out''',
    tests=[
        ("causal 的模型通过", '''assert check_causal(causal, V)'''),
        ("每个位置都读整条序列的模型不通过", '''assert not check_causal(leaky, V)'''),
        ("不看输入的模型不通过", '''assert not check_causal(blind, V), "只查了「前面不变」，没查「后面有变化」"'''),
        ("只漏了一个位置也要查出来", '''assert not check_causal(sneaky, V)'''),
        ("t 取别的值也成立", '''assert check_causal(causal, V, T=10, t=1)
assert check_causal(causal, V, T=10, t=9)
assert not check_causal(leaky, V, T=10, t=9)'''),
    ],
)

# ───────────────────────── 2 预训练 ─────────────────────────

P(
    id="s2-batches", stage=2, level="易", title="切分数据，构造 next-token 样本",
    statement="""
<p>训练语言模型之前要做两件事：留出一部分数据，再把剩下的切成 (输入, 目标) 对。</p>
<p><code>split(ids, val_frac)</code>：把 token 序列按顺序切成两段，返回 <code>(train, val)</code>。留出集是<strong>最后</strong> <code>int(len(ids) * val_frac)</code> 个 token，不打乱。</p>
<p><code>make_examples(ids, T)</code>：返回 <code>(x, y)</code>，两个形状 <code>(N, T)</code> 的数组，<code>N = len(ids) − T</code>。</p>
<pre><code>x[i] = ids[i     : i + T]
y[i] = ids[i + 1 : i + T + 1]</code></pre>
<p><code>y</code> 就是 <code>x</code> 右移一位：<code>y[i][j]</code> 是 <code>x[i][j]</code> 的下一个 token。序列太短（<code>N ≤ 0</code>）时返回两个形状 <code>(0, T)</code> 的数组。</p>
<h3>示例</h3>
<pre><code>make_examples([0, 1, 2, 3, 4], T=3)
# x = [[0, 1, 2], [1, 2, 3]]
# y = [[1, 2, 3], [2, 3, 4]]</code></pre>
""",
    starter='''import numpy as np


def split(ids, val_frac):
    # 返回 (train, val)
    pass


def make_examples(ids, T):
    # 返回 (x, y)，形状都是 (N, T)
    pass
''',
    solution='''import numpy as np


def split(ids, val_frac):
    n_val = int(len(ids) * val_frac)
    cut = len(ids) - n_val
    return ids[:cut], ids[cut:]


def make_examples(ids, T):
    ids = np.asarray(ids)
    n = len(ids) - T
    if n <= 0:
        empty = np.zeros((0, T), dtype=int)
        return empty, empty.copy()
    x = np.stack([ids[i : i + T] for i in range(n)])
    y = np.stack([ids[i + 1 : i + T + 1] for i in range(n)])
    return x, y
''',
    tests=[
        ("留出集是最后一段，顺序不变", '''train, val = split(list(range(10)), 0.2)
assert list(train) == [0, 1, 2, 3, 4, 5, 6, 7]
assert list(val) == [8, 9]'''),
        ("val_frac 为 0 时留出集为空", '''train, val = split(list(range(10)), 0.0)
assert len(train) == 10 and len(val) == 0'''),
        ("x 和 y 的形状与内容", '''x, y = make_examples(np.arange(6), 3)
assert np.array_equal(x, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
assert np.array_equal(y, [[1, 2, 3], [2, 3, 4], [3, 4, 5]])'''),
        ("y 是 x 右移一位", '''ids = np.random.default_rng(0).integers(50, size=40)
x, y = make_examples(ids, 8)
assert x.shape == y.shape == (32, 8)
assert np.array_equal(x[:, 1:], y[:, :-1])'''),
        ("序列太短时返回空数组", '''x, y = make_examples(np.arange(3), 3)
assert np.asarray(x).shape == (0, 3) and np.asarray(y).shape == (0, 3)'''),
        ("训练样本里不出现留出集的 token", '''train, val = split(np.arange(20), 0.25)
x, y = make_examples(train, 4)
assert y.max() == 14 and np.asarray(val).min() == 15'''),
    ],
)

P(
    id="s2-cross-entropy", stage=2, level="易", title="交叉熵，以及初始 loss",
    statement="""
<p>Next-token loss 就是每个位置上一次 V 选 1 的分类。实现 <code>cross_entropy(logits, targets)</code>：</p>
<ul>
  <li><code>logits</code>：形状 <code>(N, V)</code>。</li>
  <li><code>targets</code>：形状 <code>(N,)</code>，每个是正确类别的下标。</li>
</ul>
<p>返回所有样本的平均负对数似然：</p>
<pre><code>loss = −mean( log softmax(logits)[i, targets[i]] )</code></pre>
<p>要求数值稳定：logits 里出现 1000 这样的数也不能得到 <code>inf</code> 或 <code>nan</code>。常用办法是先减去每行的最大值。</p>
<h3>示例</h3>
<pre><code>cross_entropy(np.zeros((5, 7)), [0, 1, 2, 3, 4])   # ln 7 ≈ 1.9459</code></pre>
<p>这也是随机初始化的模型应该给出的 loss：预测接近均匀时，loss 接近 <code>ln V</code>。</p>
""",
    starter='''import numpy as np


def cross_entropy(logits, targets):
    pass
''',
    solution='''import numpy as np


def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets)
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(targets)), targets].mean())
''',
    tests=[
        ("均匀预测时等于 ln V", '''loss = cross_entropy(np.zeros((5, 7)), np.array([0, 1, 2, 3, 4]))
assert abs(loss - np.log(7)) < 1e-9'''),
        ("一个算得出来的例子", '''loss = cross_entropy(np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([0, 1]))
assert abs(loss - 0.2200949) < 1e-6'''),
        ("是平均，不是求和", '''one = cross_entropy(np.array([[1.0, 2.0, 3.0]]), np.array([0]))
many = cross_entropy(np.array([[1.0, 2.0, 3.0]] * 4), np.array([0, 0, 0, 0]))
assert abs(one - many) < 1e-9'''),
        ("很大的 logits 也不溢出", '''loss = cross_entropy(np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.array([0, 1]))
assert np.isfinite(loss) and loss < 1e-6'''),
        ("自信地答错，loss 很大但有限", '''loss = cross_entropy(np.array([[0.0, 1000.0]]), np.array([0]))
assert np.isfinite(loss) and abs(loss - 1000) < 1e-6'''),
    ],
)

P(
    id="s2-adamw", stage=2, level="中", title="AdamW 的一步",
    statement="""
<p>实现 <code>adamw_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01)</code>，返回更新后的 <code>(w, m, v)</code>。<code>t</code> 从 1 开始计数。</p>
<pre><code>m = b1 * m + (1 − b1) * g
v = b2 * v + (1 − b2) * g²
m̂ = m / (1 − b1ᵗ)
v̂ = v / (1 − b2ᵗ)
w = w − lr * ( m̂ / (√v̂ + eps) + wd * w )</code></pre>
<p>最后一行是 AdamW 和「Adam 加 L2」的区别：<code>wd * w</code> 直接加在更新量上，不进 <code>g</code>，所以不会被 <code>√v̂</code> 缩放。</p>
<h3>示例</h3>
<pre><code># 第一步、m 和 v 从 0 开始时，m̂ = g，v̂ = g²，更新量约等于 lr * sign(g)
adamw_step(w=[1., -2.], g=[0.5, -3.], m=0, v=0, t=1, lr=0.1, wd=0)
# w ≈ [0.9, -1.9]</code></pre>
""",
    starter='''import numpy as np


def adamw_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    # 返回 (w, m, v)
    pass
''',
    solution='''import numpy as np


def adamw_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    w = w - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * w)
    return w, m, v
''',
    setup='''Z = np.zeros(2)''',
    tests=[
        ("第一步的更新量约等于 lr * sign(g)", '''w, m, v = adamw_step(np.array([1.0, -2.0]), np.array([0.5, -3.0]), Z, Z, t=1, lr=0.1, wd=0.0)
assert np.allclose(w, [0.9, -1.9], atol=1e-6)'''),
        ("m 和 v 是梯度的滑动平均", '''w, m, v = adamw_step(np.array([1.0, -2.0]), np.array([0.5, -3.0]), Z, Z, t=1, lr=0.1, wd=0.0)
assert np.allclose(m, [0.05, -0.3])
assert np.allclose(v, [0.00025, 0.009])'''),
        ("梯度为 0 时只剩下衰减", '''w, m, v = adamw_step(np.array([2.0, -4.0]), Z, Z, Z, t=1, lr=0.1, wd=0.5)
assert np.allclose(w, [1.9, -3.8])
assert np.allclose(m, 0) and np.allclose(v, 0)'''),
        ("衰减不经过 √v̂ 的缩放", '''w, m, v = adamw_step(np.array([10.0]), np.array([1.0]), np.zeros(1), np.zeros(1), t=1, lr=0.1, wd=0.1)
assert np.allclose(w, [9.8], atol=1e-6), "得到 9.9 说明把 wd * w 加进了梯度，那是 L2，不是 AdamW"'''),
        ("第二步用到了偏差修正", '''w, m, v = adamw_step(np.array([0.0]), np.array([1.0]), np.zeros(1), np.zeros(1), t=1, lr=0.1, wd=0.0)
w, m, v = adamw_step(w, np.array([3.0]), m, v, t=2, lr=0.1, wd=0.0)
assert np.allclose(m, [0.39]) and np.allclose(v, [0.009999])
assert np.allclose(w, [-0.1917781], atol=1e-6)'''),
    ],
)

P(
    id="s2-lr-schedule", stage=2, level="易", title="Warmup 加 cosine 的学习率",
    statement="""
<p>实现 <code>lr_at(step, max_lr, warmup, total, min_lr=0.0)</code>，返回第 <code>step</code> 步（从 0 开始）用的学习率。</p>
<ul>
  <li><strong>Warmup</strong>（<code>step &lt; warmup</code>）：线性上升，<code>max_lr * (step + 1) / warmup</code>。最后一步 warmup 正好到 <code>max_lr</code>。</li>
  <li><strong>Cosine</strong>（<code>warmup ≤ step &lt; total</code>）：
    <pre><code>progress = (step − warmup) / (total − warmup)
lr = min_lr + 0.5 * (max_lr − min_lr) * (1 + cos(π * progress))</code></pre>
  </li>
  <li><strong>结束之后</strong>（<code>step ≥ total</code>）：<code>min_lr</code>。</li>
</ul>
<h3>示例</h3>
<pre><code>lr_at(0,  max_lr=1.0, warmup=10, total=100)   # 0.1
lr_at(9,  max_lr=1.0, warmup=10, total=100)   # 1.0
lr_at(55, max_lr=1.0, warmup=10, total=100)   # 0.5，cosine 段的正中间</code></pre>
""",
    starter='''import math


def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    pass
''',
    solution='''import math


def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
''',
    tests=[
        ("warmup 线性上升", '''assert abs(lr_at(0, 1.0, 10, 100) - 0.1) < 1e-12
assert abs(lr_at(4, 1.0, 10, 100) - 0.5) < 1e-12
assert abs(lr_at(9, 1.0, 10, 100) - 1.0) < 1e-12'''),
        ("cosine 段从 max_lr 开始", '''assert abs(lr_at(10, 1.0, 10, 100) - 1.0) < 1e-12'''),
        ("cosine 段的中点是两端的平均", '''assert abs(lr_at(55, 1.0, 10, 100) - 0.5) < 1e-12
assert abs(lr_at(55, 1.0, 10, 100, min_lr=0.1) - 0.55) < 1e-12'''),
        ("结束之后停在 min_lr", '''assert abs(lr_at(100, 1.0, 10, 100, min_lr=0.1) - 0.1) < 1e-12
assert abs(lr_at(5000, 1.0, 10, 100, min_lr=0.1) - 0.1) < 1e-12'''),
        ("warmup 之后只降不升", '''lrs = [lr_at(s, 3e-4, 10, 100, min_lr=1e-5) for s in range(10, 101)]
assert all(a >= b for a, b in zip(lrs, lrs[1:]))
assert abs(lrs[0] - 3e-4) < 1e-12 and abs(lrs[-1] - 1e-5) < 1e-12'''),
    ],
)

P(
    id="s2-overfit", stage=2, level="难", title="验：过拟合一个 batch",
    statement="""
<p>这是第 2 步的验收：手写一个最小的训练循环，让模型把一个 batch 背下来。</p>
<p>模型是 bigram：一张 <code>(V, V)</code> 的表 <code>W</code>，输入 token <code>a</code> 时，下一个 token 的 logits 就是 <code>W[a]</code>。</p>
<p>实现 <code>train(x, y, V, steps=300, lr=1.0, seed=0)</code>，返回 <code>(W, losses)</code>：</p>
<ol>
  <li>初始化 <code>W = np.random.default_rng(seed).normal(0, 0.01, size=(V, V))</code>。</li>
  <li>把 <code>x</code>、<code>y</code> 拉平成长度 N 的一维数组。每一步：
    <ul>
      <li><code>logits = W[x]</code>，形状 <code>(N, V)</code>，算平均交叉熵，把它记进 <code>losses</code>（更新之前的 loss）。</li>
      <li>对 logits 的梯度是 <code>(softmax − onehot(y)) / N</code>。</li>
      <li>把这些梯度按行加回 <code>W</code> 对应的行：同一个 token 在 batch 里出现多次时梯度要累加，用 <code>np.add.at(dW, x, dlogits)</code>。</li>
      <li><code>W −= lr * dW</code>。</li>
    </ul>
  </li>
</ol>
<p><code>losses</code> 是长度为 <code>steps</code> 的 list。</p>
<h3>要看到的三件事</h3>
<ul>
  <li>第一个 loss 接近 <code>ln V</code>。</li>
  <li>每个 token 只有一种续写时，loss 降到接近 0。</li>
  <li>同一个 token 有两种续写、各占一半时，loss 停在 <code>ln 2</code>，降不下去。这不是 bug。</li>
</ul>
""",
    starter='''import numpy as np


def train(x, y, V, steps=300, lr=1.0, seed=0):
    # 返回 (W, losses)
    pass
''',
    solution='''import numpy as np


def train(x, y, V, steps=300, lr=1.0, seed=0):
    W = np.random.default_rng(seed).normal(0, 0.01, size=(V, V))
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    N = len(x)
    rows = np.arange(N)
    losses = []
    for _ in range(steps):
        logits = W[x]
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        losses.append(float(-np.log(p[rows, y]).mean()))

        dlogits = p
        dlogits[rows, y] -= 1
        dlogits /= N
        dW = np.zeros_like(W)
        np.add.at(dW, x, dlogits)
        W -= lr * dW
    return W, losses
''',
    setup='''V = 8
seq = np.tile(np.arange(V), 5)                 # 0 1 2 ... 7 0 1 2 ...
x = seq[:-1].reshape(3, 13)
y = seq[1:].reshape(3, 13)                     # 每个 token 的下一个都是 (token + 1) % V

def _ce(logits, t):
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(t)), t].mean())''',
    tests=[
        ("返回值的形状", '''W, losses = train(x, y, V, steps=20)
assert W.shape == (V, V)
assert len(losses) == 20'''),
        ("第一个 loss 接近 ln V", '''W, losses = train(x, y, V, steps=5)
assert abs(losses[0] - np.log(V)) < 0.05, f"第一个 loss 是 {losses[0]:.3f}，应接近 {np.log(V):.3f}"'''),
        ("能把这个 batch 背下来", '''W, losses = train(x, y, V, steps=300, lr=10.0)
assert losses[-1] < 0.05, f"最后的 loss 是 {losses[-1]:.3f}"
assert losses[-1] < losses[0]'''),
        ("学到的是 token → 下一个 token", '''W, losses = train(x, y, V, steps=300, lr=10.0)
assert np.array_equal(W.argmax(1), (np.arange(V) + 1) % V)'''),
        ("一个前缀两种续写时，loss 停在 ln 2", '''W, losses = train(np.array([[0, 0, 0, 0]]), np.array([[1, 2, 1, 2]]), 3, steps=300, lr=1.0)
assert abs(losses[-1] - np.log(2)) < 0.02, f"最后的 loss 是 {losses[-1]:.3f}，应接近 0.693"'''),
        ("背下来不等于会泛化", '''W, losses = train(x, y, V, steps=300, lr=10.0)
held_x = np.arange(V)
held_y = (np.arange(V) + 2) % V                # 留出数据的规律不一样
assert _ce(W[held_x], held_y) > losses[-1] + 3'''),
    ],
)

# ───────────────────────── 3 后训练 ─────────────────────────

P(
    id="s3-lora", stage=3, level="易", title="LoRA：forward、合并、参数量",
    statement="""
<p>LoRA 冻结原来的权重 <code>W</code>，只训练一个低秩的增量 <code>ΔW = (alpha / r) · B · A</code>。</p>
<ul>
  <li><code>W</code>：形状 <code>(d_out, d_in)</code>。</li>
  <li><code>A</code>：形状 <code>(r, d_in)</code>。<code>B</code>：形状 <code>(d_out, r)</code>。</li>
  <li><code>x</code>：形状 <code>(N, d_in)</code>。</li>
</ul>
<p>实现三个函数：</p>
<ul>
  <li><code>lora_forward(x, W, A, B, alpha)</code>：<code>x @ Wᵀ + (alpha / r) * x @ Aᵀ @ Bᵀ</code>，形状 <code>(N, d_out)</code>。</li>
  <li><code>merge(W, A, B, alpha)</code>：返回合并后的权重 <code>W + (alpha / r) * B @ A</code>，不要改动传进来的 <code>W</code>。</li>
  <li><code>trainable_params(d_in, d_out, r)</code>：A 和 B 一共多少个参数。</li>
</ul>
<h3>示例</h3>
<pre><code>trainable_params(4096, 4096, 8)   # 65536，原来的层有 16,777,216 个</code></pre>
""",
    starter='''import numpy as np


def lora_forward(x, W, A, B, alpha):
    pass


def merge(W, A, B, alpha):
    pass


def trainable_params(d_in, d_out, r):
    pass
''',
    solution='''import numpy as np


def lora_forward(x, W, A, B, alpha):
    r = A.shape[0]
    return x @ W.T + (alpha / r) * (x @ A.T @ B.T)


def merge(W, A, B, alpha):
    r = A.shape[0]
    return W + (alpha / r) * (B @ A)


def trainable_params(d_in, d_out, r):
    return r * (d_in + d_out)
''',
    setup='''rng = np.random.default_rng(0)
x = rng.normal(size=(5, 6))
W = rng.normal(size=(4, 6))
A = rng.normal(size=(2, 6))
B = rng.normal(size=(4, 2))''',
    tests=[
        ("B 为 0 时输出和原模型一样", '''assert np.allclose(lora_forward(x, W, A, np.zeros((4, 2)), alpha=16), x @ W.T)'''),
        ("一个算得出来的例子", '''out = lora_forward(np.array([[1.0, 2.0]]), np.eye(2), np.array([[1.0, 1.0]]), np.array([[1.0], [2.0]]), alpha=2)
assert np.allclose(out, [[7.0, 14.0]])'''),
        ("缩放是 alpha / r", '''base = x @ W.T
d1 = lora_forward(x, W, A, B, alpha=2) - base
d2 = lora_forward(x, W, A, B, alpha=8) - base
assert np.allclose(d2, 4 * d1)
assert np.allclose(d1, x @ A.T @ B.T)          # r = 2，alpha = 2，缩放是 1'''),
        ("合并之后是同一个函数", '''assert np.allclose(x @ merge(W, A, B, 16).T, lora_forward(x, W, A, B, 16))'''),
        ("merge 不改原来的 W", '''W0 = W.copy()
merge(W, A, B, 16)
assert np.array_equal(W, W0)'''),
        ("参数量", '''assert trainable_params(4096, 4096, 8) == 65536
assert trainable_params(768, 3072, 4) == 15360'''),
    ],
)

P(
    id="s3-sft-mask", stage=3, level="中", title="SFT：只在回答上算 loss",
    statement="""
<p>SFT 的一条样本是「prompt + 回答」拼成的一条序列，loss 只算在回答的 token 上。</p>
<p>设 prompt 有 P 个 token，回答有 R 个。输入是序列去掉最后一个 token，目标是序列去掉第一个，长度都是 <code>P + R − 1</code>。</p>
<p><code>build_mask(prompt_len, response_len)</code>：返回长度 <code>P + R − 1</code> 的 0/1 数组，<code>mask[i] = 1</code> 当且仅当第 i 个<strong>目标</strong>是回答里的 token。</p>
<p><code>sft_loss(logits, targets, loss_mask)</code>：</p>
<ul>
  <li><code>logits</code> 形状 <code>(T, V)</code>，<code>targets</code> 形状 <code>(T,)</code>，<code>loss_mask</code> 形状 <code>(T,)</code>。</li>
  <li>返回 mask 为 1 的位置上交叉熵的平均。mask 全为 0 时返回 <code>0.0</code>。</li>
</ul>
<h3>示例</h3>
<pre><code># 序列: p0 p1 p2 r0 r1      P = 3, R = 2
# 输入: p0 p1 p2 r0
# 目标: p1 p2 r0 r1
build_mask(3, 2)   # [0, 0, 1, 1]</code></pre>
<p>注意第三个位置：输入是 prompt 的最后一个 token，目标已经是回答的第一个 token，要算 loss。</p>
""",
    starter='''import numpy as np


def build_mask(prompt_len, response_len):
    pass


def sft_loss(logits, targets, loss_mask):
    pass
''',
    solution='''import numpy as np


def build_mask(prompt_len, response_len):
    return np.array([0] * (prompt_len - 1) + [1] * response_len)


def sft_loss(logits, targets, loss_mask):
    keep = np.asarray(loss_mask).astype(bool)
    if not keep.any():
        return 0.0
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets)
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    nll = -logp[np.arange(len(targets)), targets]
    return float(nll[keep].mean())
''',
    tests=[
        ("mask 的长度和内容", '''assert list(build_mask(3, 2)) == [0, 0, 1, 1]
assert len(build_mask(10, 5)) == 14 and int(np.sum(build_mask(10, 5))) == 5'''),
        ("prompt 只有一个 token 时全是 1", '''assert list(build_mask(1, 3)) == [1, 1, 1]'''),
        ("只平均 mask 为 1 的位置", '''logits = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]])
loss = sft_loss(logits, np.array([0, 0, 1]), np.array([0, 1, 1]))
assert abs(loss - 0.3499311) < 1e-6'''),
        ("prompt 位置的 logits 不影响 loss", '''rng = np.random.default_rng(0)
logits = rng.normal(size=(6, 5))
targets = rng.integers(5, size=6)
mask = build_mask(4, 3)
a = sft_loss(logits, targets, mask)
logits2 = logits.copy()
logits2[:3] += rng.normal(size=(3, 5)) * 10
assert abs(sft_loss(logits2, targets, mask) - a) < 1e-9'''),
        ("mask 全为 0 时返回 0.0", '''assert sft_loss(np.zeros((3, 4)), np.array([0, 1, 2]), np.zeros(3)) == 0.0'''),
    ],
)

P(
    id="s3-dpo", stage=3, level="中", title="DPO loss",
    statement="""
<p>偏好数据的每一条是：一个 prompt、一个好的回答 y⁺、一个差的回答 y⁻。DPO 让当前策略 π 相对参考模型 π_ref 更偏向 y⁺。</p>
<p>实现 <code>dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1)</code>。四个输入都是形状 <code>(N,)</code> 的数组，是整条回答的对数概率：</p>
<ul>
  <li><code>pi_pos[i] = log π(y⁺ | x)</code>，<code>pi_neg[i] = log π(y⁻ | x)</code></li>
  <li><code>ref_pos[i] = log π_ref(y⁺ | x)</code>，<code>ref_neg[i] = log π_ref(y⁻ | x)</code></li>
</ul>
<pre><code>z = beta * ( (pi_pos − ref_pos) − (pi_neg − ref_neg) )
loss = mean( −log σ(z) )</code></pre>
<p>要求数值稳定：<code>z</code> 是 ±2000 时也不能出现 <code>inf</code> 或 <code>nan</code>。提示：<code>−log σ(z) = log(1 + e^(−z))</code>，可以用 <code>np.logaddexp(0, −z)</code>。</p>
<h3>示例</h3>
<pre><code># 策略和参考模型一样时 z = 0，loss = ln 2
dpo_loss([-5.], [-7.], [-5.], [-7.])   # 0.6931</code></pre>
""",
    starter='''import numpy as np


def dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1):
    pass
''',
    solution='''import numpy as np


def dpo_loss(pi_pos, pi_neg, ref_pos, ref_neg, beta=0.1):
    pi_pos, pi_neg = np.asarray(pi_pos, float), np.asarray(pi_neg, float)
    ref_pos, ref_neg = np.asarray(ref_pos, float), np.asarray(ref_neg, float)
    z = beta * ((pi_pos - ref_pos) - (pi_neg - ref_neg))
    return float(np.logaddexp(0, -z).mean())
''',
    tests=[
        ("策略等于参考模型时是 ln 2", '''loss = dpo_loss(np.array([-5.0, -3.0]), np.array([-7.0, -1.0]), np.array([-5.0, -3.0]), np.array([-7.0, -1.0]))
assert abs(loss - np.log(2)) < 1e-9'''),
        ("一个算得出来的例子", '''assert abs(dpo_loss([-10.0], [-12.0], [-11.0], [-11.0], beta=1.0) - 0.1269280) < 1e-6
assert abs(dpo_loss([-10.0], [-12.0], [-11.0], [-11.0], beta=0.5) - 0.3132617) < 1e-6'''),
        ("好回答的概率升高，loss 下降", '''a = dpo_loss([-10.0], [-10.0], [-10.0], [-10.0])
b = dpo_loss([-8.0], [-10.0], [-10.0], [-10.0])
c = dpo_loss([-10.0], [-8.0], [-10.0], [-10.0])
assert b < a < c'''),
        ("只看相对参考模型的变化", '''a = dpo_loss([-10.0], [-12.0], [-11.0], [-11.0], beta=1.0)
b = dpo_loss([-110.0], [-12.0], [-111.0], [-11.0], beta=1.0)
assert abs(a - b) < 1e-9'''),
        ("对一个 batch 取平均", '''loss = dpo_loss([-10.0, -5.0], [-12.0, -5.0], [-11.0, -5.0], [-11.0, -5.0], beta=1.0)
assert abs(loss - (0.1269280 + np.log(2)) / 2) < 1e-6'''),
        ("z 很大或很小时不溢出", '''big = dpo_loss([0.0], [-2000.0], [0.0], [0.0], beta=1.0)
small = dpo_loss([-2000.0], [0.0], [0.0], [0.0], beta=1.0)
assert np.isfinite(big) and big < 1e-6
assert np.isfinite(small) and abs(small - 2000) < 1e-6'''),
    ],
)

P(
    id="s3-policy-gradient", stage=3, level="中", title="策略梯度的目标，对照 DPO",
    statement="""
<p>RL 的目标是最大化期望奖励。和 DPO 对照着看：DPO 需要成对的回答和一个参考模型，RL 需要从当前策略采样，再用奖励给每个样本加权。</p>
<p><code>pg_loss(logp, rewards, baseline=None)</code>：REINFORCE 的 surrogate loss。</p>
<pre><code>loss = −mean( (rewards − baseline) * logp )</code></pre>
<p><code>logp[i]</code> 是第 i 个采样到的回答的对数概率。<code>baseline</code> 没给时用 <code>rewards</code> 的平均值。</p>
<p><code>pg_grad_logits(logits, action, reward, baseline=0.0)</code>：策略是对 <code>logits</code>（形状 <code>(K,)</code>）做 softmax，采样到了 <code>action</code>。返回这一个样本的 loss 对 <code>logits</code> 的梯度，形状 <code>(K,)</code>：</p>
<pre><code>grad = (reward − baseline) * (softmax(logits) − onehot(action))</code></pre>
<h3>示例</h3>
<pre><code>pg_loss(logp=[-1., -2.], rewards=[1., 3.])   # baseline = 2，loss = 0.5</code></pre>
<p>奖励全都一样时，减掉 baseline 之后每个样本的权重都是 0：没有哪个回答比别的好，就没有学习信号。</p>
""",
    starter='''import numpy as np


def pg_loss(logp, rewards, baseline=None):
    pass


def pg_grad_logits(logits, action, reward, baseline=0.0):
    pass
''',
    solution='''import numpy as np


def pg_loss(logp, rewards, baseline=None):
    logp = np.asarray(logp, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    if baseline is None:
        baseline = rewards.mean()
    return float(-((rewards - baseline) * logp).mean())


def pg_grad_logits(logits, action, reward, baseline=0.0):
    logits = np.asarray(logits, dtype=float)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    p[action] -= 1
    return (reward - baseline) * p
''',
    setup='''def _logsoftmax(l):
    z = l - l.max()
    return z - np.log(np.exp(z).sum())''',
    tests=[
        ("默认 baseline 是奖励的平均", '''assert abs(pg_loss([-1.0, -2.0], [1.0, 3.0]) - 0.5) < 1e-9'''),
        ("可以指定 baseline", '''assert abs(pg_loss([-1.0, -2.0], [1.0, 3.0], baseline=0.0) - 3.5) < 1e-9'''),
        ("奖励全都一样时没有信号", '''assert abs(pg_loss([-1.0, -2.0, -3.0], [5.0, 5.0, 5.0])) < 1e-9'''),
        ("梯度和数值微分一致", '''logits = np.array([0.2, -1.0, 0.7, 0.1])
a, r, b = 2, 1.5, 0.4
g = pg_grad_logits(logits, a, r, b)
num = np.zeros(4)
for i in range(4):
    d = np.zeros(4); d[i] = 1e-6
    f = lambda l: -(r - b) * _logsoftmax(l)[a]
    num[i] = (f(logits + d) - f(logits - d)) / 2e-6
assert np.asarray(g).shape == (4,)
assert np.allclose(g, num, atol=1e-6)'''),
        ("奖励高于 baseline，这个动作的概率上升", '''logits = np.array([0.0, 0.0, 0.0])
g = pg_grad_logits(logits, action=1, reward=2.0, baseline=1.0)
new = logits - 0.5 * g
assert np.exp(_logsoftmax(new))[1] > 1 / 3'''),
        ("奖励低于 baseline，这个动作的概率下降", '''logits = np.array([0.0, 0.0, 0.0])
g = pg_grad_logits(logits, action=1, reward=0.0, baseline=1.0)
new = logits - 0.5 * g
assert np.exp(_logsoftmax(new))[1] < 1 / 3'''),
    ],
)

P(
    id="s3-paired", stage=3, level="易", title="验：同一组留出 prompts，训练前后配对比较",
    statement="""
<p>这是第 3 步的验收。训练前后用同一组没训练过的 prompts 各生成一次，逐条打分，再配对比较。</p>
<p>实现 <code>compare(before, after, tol=1e-9)</code>。<code>before[i]</code> 和 <code>after[i]</code> 是第 i 个 prompt 在训练前后的分数。返回一个 dict：</p>
<ul>
  <li><code>"wins"</code>：<code>after − before &gt; tol</code> 的条数。</li>
  <li><code>"losses"</code>：<code>after − before &lt; −tol</code> 的条数。</li>
  <li><code>"ties"</code>：其余的条数。</li>
  <li><code>"mean_delta"</code>：<code>after − before</code> 的平均值。</li>
</ul>
<p>两个列表长度不一样时，说明前后用的不是同一组 prompts，抛出 <code>ValueError</code>。</p>
<h3>示例</h3>
<pre><code>compare([0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.4, 1.0])
# {"wins": 1, "losses": 3, "ties": 0, "mean_delta": 0.05}</code></pre>
<p>平均分涨了，但四条里有三条变差了。只看平均分会漏掉这件事。</p>
""",
    starter='''def compare(before, after, tol=1e-9):
    pass
''',
    solution='''def compare(before, after, tol=1e-9):
    if len(before) != len(after):
        raise ValueError("before 和 after 的条数不一致，不是同一组 prompts")
    deltas = [a - b for a, b in zip(after, before)]
    return {
        "wins": sum(d > tol for d in deltas),
        "losses": sum(d < -tol for d in deltas),
        "ties": sum(abs(d) <= tol for d in deltas),
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
    }
''',
    tests=[
        ("数出变好、变差和持平的条数", '''r = compare([0.2, 0.5, 0.9, 0.4], [0.6, 0.5, 0.7, 0.8])
assert (r["wins"], r["losses"], r["ties"]) == (2, 1, 1)
assert abs(r["mean_delta"] - 0.15) < 1e-9'''),
        ("平均分涨了，多数却变差了", '''r = compare([0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.4, 1.0])
assert r["wins"] == 1 and r["losses"] == 3
assert r["mean_delta"] > 0'''),
        ("三个数加起来等于条数", '''import random
random.seed(0)
b = [random.random() for _ in range(50)]
a = [v if i % 5 == 0 else random.random() for i, v in enumerate(b)]
r = compare(b, a)
assert r["wins"] + r["losses"] + r["ties"] == 50
assert r["ties"] == 10'''),
        ("长度不一致时抛出 ValueError", '''try:
    compare([0.1, 0.2], [0.3])
except ValueError:
    pass
else:
    assert False, "长度不一致时应该抛出 ValueError"'''),
    ],
)
