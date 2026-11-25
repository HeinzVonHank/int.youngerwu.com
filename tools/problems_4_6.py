"""Problems for stages 4-6. Same format as problems_1_3.py."""

PROBLEMS = []


def P(**kw):
    kw.setdefault("setup", "")
    PROBLEMS.append(kw)


# ───────────────────────── 4 推理与 Infra ─────────────────────────

P(
    id="s4-sampling", stage=4, level="中", title="Temperature、top-k、top-p",
    statement="""
<p>实现 <code>sampling_probs(logits, temperature=1.0, top_k=0, top_p=1.0)</code>：返回真正用来抽样的概率分布，形状和 <code>logits</code> 一样。按这个顺序处理：</p>
<ol>
  <li><strong>Temperature</strong>：<code>softmax(logits / temperature)</code>。<code>temperature == 0</code> 时直接返回 one-hot：最大的那个是 1，其余是 0。</li>
  <li><strong>Top-k</strong>（<code>top_k &gt; 0</code> 时）：只保留概率最高的 k 个，其余置 0，再归一化。</li>
  <li><strong>Top-p</strong>（<code>top_p &lt; 1</code> 时）：按概率从大到小排，保留累计概率<strong>刚好达到</strong> <code>top_p</code> 的最短前缀（至少一个），其余置 0，再归一化。</li>
</ol>
<h3>示例</h3>
<pre><code>p = [0.5, 0.3, 0.2]
# top_p = 0.6：0.5 不够，0.5 + 0.3 = 0.8 够了，保留前两个
sampling_probs(np.log(p), top_p=0.6)   # [0.625, 0.375, 0]</code></pre>
""",
    starter='''import numpy as np


def sampling_probs(logits, temperature=1.0, top_k=0, top_p=1.0):
    pass
''',
    solution='''import numpy as np


def sampling_probs(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = np.asarray(logits, dtype=float)
    if temperature == 0:
        p = np.zeros_like(logits)
        p[np.argmax(logits)] = 1.0
        return p

    z = logits / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p)                    # 概率从大到小的下标

    if top_k > 0:
        p[order[top_k:]] = 0.0
        p /= p.sum()

    if top_p < 1.0:
        cum = np.cumsum(p[order])
        n = int(np.searchsorted(cum, top_p - 1e-12)) + 1
        p[order[n:]] = 0.0
        p /= p.sum()
    return p
''',
    tests=[
        ("默认参数就是 softmax", '''p = sampling_probs(np.array([1.0, 2.0, 3.0]))
assert np.allclose(p, [0.09003057, 0.24472847, 0.66524096])'''),
        ("temperature 为 0 时是 one-hot", '''p = sampling_probs(np.array([1.0, 3.0, 2.0]), temperature=0)
assert np.allclose(p, [0, 1, 0])'''),
        ("温度越高分布越平", '''l = np.array([1.0, 2.0, 3.0])
assert sampling_probs(l, temperature=0.5).max() > sampling_probs(l).max() > sampling_probs(l, temperature=2.0).max()'''),
        ("top-k 只留 k 个，并重新归一化", '''p = sampling_probs(np.array([1.0, 2.0, 3.0, 0.0]), top_k=2)
assert np.allclose(p, [0, 0.26894142, 0.73105858, 0])'''),
        ("top-p 保留刚好达到 p 的最短前缀", '''p = sampling_probs(np.log([0.5, 0.3, 0.2]), top_p=0.6)
assert np.allclose(p, [0.625, 0.375, 0])'''),
        ("top-p 很小时至少留一个", '''p = sampling_probs(np.log([0.2, 0.5, 0.3]), top_p=0.01)
assert np.allclose(p, [0, 1, 0])'''),
        ("先 top-k，再在剩下的分布上做 top-p", '''p = sampling_probs(np.log([0.4, 0.3, 0.2, 0.1]), top_k=3, top_p=0.5)
assert np.allclose(p, [4 / 7, 3 / 7, 0, 0])'''),
        ("结果的和是 1", '''l = np.random.default_rng(0).normal(size=50)
p = sampling_probs(l, temperature=0.7, top_k=20, top_p=0.9)
assert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()'''),
    ],
)

P(
    id="s4-kv-cache", stage=4, level="中", title="验：KV cache 的一步，和整段重算一致",
    statement="""
<p>这是第 4 步的验收。<code>attend_full</code> 已经写好：它拿到整段输入，每次都从头算一遍 causal attention。</p>
<p>你来写带 cache 的版本：<code>attend_step(x_new, Wq, Wk, Wv, cache)</code>，一次只处理一个新 token。</p>
<ul>
  <li><code>x_new</code>：形状 <code>(C,)</code>，新来的这个 token 的向量。</li>
  <li><code>cache</code>：<code>{"K": ..., "V": ...}</code>，两个形状 <code>(t, D)</code> 的数组，是之前 t 个 token 的 key 和 value。<code>t</code> 可以是 0。</li>
</ul>
<p>步骤：</p>
<ol>
  <li>只为新 token 算 <code>q</code>、<code>k</code>、<code>v</code>。</li>
  <li>把 <code>k</code>、<code>v</code> 接到 cache 后面，得到 <code>(t+1, D)</code>。</li>
  <li>用 <code>q</code> 和全部 <code>t+1</code> 个 key 算分数（除以 <code>√D</code>），softmax，再对 value 加权求和。</li>
</ol>
<p>返回 <code>(out, new_cache)</code>，<code>out</code> 形状 <code>(D,)</code>。不要原地修改传进来的 <code>cache</code>。</p>
<h3>要成立的事</h3>
<pre><code>把 X 的每一行依次喂给 attend_step，
第 t 次的 out 应该等于 attend_full(X, ...)[t]</code></pre>
<p>这里不需要 mask：cache 里只有过去的 token，新 token 本来就看不到未来。</p>
""",
    starter='''import numpy as np


def attend_full(X, Wq, Wk, Wv):
    """整段重算。X: (T, C)，返回 (T, D)。已经写好，用来对照。"""
    T = X.shape[0]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(K.shape[1])
    S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V


def attend_step(x_new, Wq, Wk, Wv, cache):
    # 返回 (out, new_cache)
    pass
''',
    solution='''import numpy as np


def attend_full(X, Wq, Wk, Wv):
    """整段重算。X: (T, C)，返回 (T, D)。已经写好，用来对照。"""
    T = X.shape[0]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(K.shape[1])
    S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V


def attend_step(x_new, Wq, Wk, Wv, cache):
    q, k, v = x_new @ Wq, x_new @ Wk, x_new @ Wv
    K = np.vstack([cache["K"], k[None, :]])          # 新数组，不动原来的 cache
    V = np.vstack([cache["V"], v[None, :]])
    s = K @ q / np.sqrt(K.shape[1])
    p = np.exp(s - s.max())
    p /= p.sum()
    return p @ V, {"K": K, "V": V}
''',
    setup='''rng = np.random.default_rng(0)
C, D, T = 6, 4, 7
X = rng.normal(size=(T, C))
Wq, Wk, Wv = (rng.normal(size=(C, D)) for _ in range(3))

def _full(X):
    n = X.shape[0]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(D)
    S = np.where(np.triu(np.ones((n, n), dtype=bool), k=1), -np.inf, S)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V

def _empty():
    return {"K": np.zeros((0, D)), "V": np.zeros((0, D))}''',
    tests=[
        ("cache 为空时，输出就是新 token 自己的 value", '''out, cache = attend_step(X[0], Wq, Wk, Wv, _empty())
assert np.asarray(out).shape == (D,)
assert np.allclose(out, X[0] @ Wv)'''),
        ("逐个 token 喂进去，每一步都和整段重算一致", '''ref = _full(X)
cache = _empty()
for t in range(T):
    out, cache = attend_step(X[t], Wq, Wk, Wv, cache)
    assert np.allclose(out, ref[t], atol=1e-8), f"第 {t} 步对不上"'''),
        ("cache 每步多一行", '''cache = _empty()
for t in range(T):
    out, cache = attend_step(X[t], Wq, Wk, Wv, cache)
    assert cache["K"].shape == (t + 1, D) and cache["V"].shape == (t + 1, D)'''),
        ("cache 里存的是每个 token 的 key 和 value", '''cache = _empty()
for t in range(T):
    out, cache = attend_step(X[t], Wq, Wk, Wv, cache)
assert np.allclose(cache["K"], X @ Wk)
assert np.allclose(cache["V"], X @ Wv)'''),
        ("不原地修改传进来的 cache", '''old = _empty()
out, new = attend_step(X[0], Wq, Wk, Wv, old)
assert old["K"].shape == (0, D) and old["V"].shape == (0, D)
out2, newer = attend_step(X[1], Wq, Wk, Wv, new)
assert new["K"].shape == (1, D)'''),
    ],
)

P(
    id="s4-kv-size", stage=4, level="易", title="KV cache 占多少显存",
    statement="""
<p>每一层、每个头、每个位置都要存一个 key 和一个 value，各 <code>head_dim</code> 个数。</p>
<p><code>kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, batch=1, bytes_per_value=2)</code>：返回 KV cache 的字节数。<code>bytes_per_value</code> 是每个数占的字节：fp16 是 2，fp32 是 4。</p>
<pre><code>2 × n_layers × batch × seq_len × n_heads × head_dim × bytes_per_value</code></pre>
<p><code>max_context(budget_bytes, n_layers, n_heads, head_dim, batch=1, bytes_per_value=2)</code>：给定显存预算，返回放得下的最长 <code>seq_len</code>（整数，向下取整）。</p>
<h3>示例</h3>
<pre><code># 12 层、12 头、每头 64 维、fp16、1024 个 token
kv_cache_bytes(12, 12, 64, 1024)   # 37,748,736 字节，约 36 MB</code></pre>
""",
    starter='''def kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, batch=1, bytes_per_value=2):
    pass


def max_context(budget_bytes, n_layers, n_heads, head_dim, batch=1, bytes_per_value=2):
    pass
''',
    solution='''def kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, batch=1, bytes_per_value=2):
    return 2 * n_layers * batch * seq_len * n_heads * head_dim * bytes_per_value


def max_context(budget_bytes, n_layers, n_heads, head_dim, batch=1, bytes_per_value=2):
    per_token = 2 * n_layers * batch * n_heads * head_dim * bytes_per_value
    return budget_bytes // per_token
''',
    tests=[
        ("示例里的数", '''assert kv_cache_bytes(12, 12, 64, 1024) == 37748736'''),
        ("随序列长度和 batch 线性增长", '''base = kv_cache_bytes(12, 12, 64, 1024)
assert kv_cache_bytes(12, 12, 64, 2048) == 2 * base
assert kv_cache_bytes(12, 12, 64, 1024, batch=8) == 8 * base'''),
        ("fp32 是 fp16 的两倍", '''assert kv_cache_bytes(12, 12, 64, 1024, bytes_per_value=4) == 2 * kv_cache_bytes(12, 12, 64, 1024)'''),
        ("max_context 是反过来算", '''assert max_context(37748736, 12, 12, 64) == 1024
assert max_context(37748735, 12, 12, 64) == 1023
assert max_context(37748736, 12, 12, 64, batch=4) == 256'''),
        ("返回整数", '''assert isinstance(max_context(10 ** 9, 24, 16, 64), int)
assert kv_cache_bytes(24, 16, 64, max_context(10 ** 9, 24, 16, 64)) <= 10 ** 9'''),
    ],
)

P(
    id="s4-latency", stage=4, level="易", title="从时间戳算 TTFT 和每 token 延迟",
    statement="""
<p>一次流式生成留下一串时间戳。实现 <code>latency_stats(t_request, token_times)</code>：</p>
<ul>
  <li><code>t_request</code>：请求发出的时刻（秒）。</li>
  <li><code>token_times</code>：每个输出 token 到达的时刻，至少有一个，按时间递增。</li>
</ul>
<p>返回一个 dict：</p>
<ul>
  <li><code>"ttft"</code>：首 token 延迟，<code>token_times[0] − t_request</code>。主要由 prefill 决定。</li>
  <li><code>"itl"</code>：相邻两个 token 的平均间隔。主要由 decode 决定。只有一个 token 时是 <code>0.0</code>。</li>
  <li><code>"tps"</code>：decode 阶段每秒出几个 token，<code>(n − 1) / (最后一个 − 第一个)</code>。只有一个 token 时是 <code>0.0</code>。</li>
  <li><code>"total"</code>：<code>token_times[-1] − t_request</code>。</li>
</ul>
<h3>示例</h3>
<pre><code>latency_stats(10.0, [10.5, 10.6, 10.7, 10.9])
# ttft = 0.5, itl ≈ 0.1333, tps = 7.5, total = 0.9</code></pre>
""",
    starter='''def latency_stats(t_request, token_times):
    pass
''',
    solution='''def latency_stats(t_request, token_times):
    n = len(token_times)
    first, last = token_times[0], token_times[-1]
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    return {
        "ttft": first - t_request,
        "itl": sum(gaps) / len(gaps) if gaps else 0.0,
        "tps": (n - 1) / (last - first) if n > 1 else 0.0,
        "total": last - t_request,
    }
''',
    tests=[
        ("示例里的数", '''r = latency_stats(10.0, [10.5, 10.6, 10.7, 10.9])
assert abs(r["ttft"] - 0.5) < 1e-9
assert abs(r["itl"] - 0.4 / 3) < 1e-9
assert abs(r["tps"] - 7.5) < 1e-9
assert abs(r["total"] - 0.9) < 1e-9'''),
        ("只有一个 token", '''r = latency_stats(0.0, [2.0])
assert abs(r["ttft"] - 2.0) < 1e-9 and abs(r["total"] - 2.0) < 1e-9
assert r["itl"] == 0.0 and r["tps"] == 0.0'''),
        ("prompt 变长只影响 TTFT", '''a = latency_stats(10.0, [10.5, 10.6, 10.7, 10.9])
b = latency_stats(10.0, [11.5, 11.6, 11.7, 11.9])
assert abs(b["ttft"] - a["ttft"] - 1.0) < 1e-9
assert abs(b["itl"] - a["itl"]) < 1e-9 and abs(b["tps"] - a["tps"]) < 1e-9'''),
        ("tps 是 itl 的倒数", '''r = latency_stats(0.0, [1.0, 1.25, 1.5, 1.75, 2.0])
assert abs(r["tps"] * r["itl"] - 1) < 1e-9'''),
    ],
)

P(
    id="s4-quant", stage=4, level="易", title="对称 int8 量化",
    statement="""
<p>把 float32 的权重存成 int8，显存和每步要读的字节数都变成四分之一。</p>
<p><code>quantize(w)</code>：返回 <code>(q, scale)</code>。</p>
<pre><code>scale = max(|w|) / 127
q = round(w / scale)，截断到 [−127, 127]，类型是 int8</code></pre>
<p><code>w</code> 全为 0 时 <code>scale</code> 取 <code>1.0</code>。</p>
<p><code>dequantize(q, scale)</code>：返回 <code>q * scale</code>，类型是 float32。</p>
<h3>示例</h3>
<pre><code>q, scale = quantize(np.array([-0.6, 0.2, 1.0], dtype=np.float32))
q        # [-76, 25, 127]
scale    # 1 / 127 ≈ 0.007874</code></pre>
<p>量化之后再还原，每个数的误差不超过半个 <code>scale</code>。</p>
""",
    starter='''import numpy as np


def quantize(w):
    # 返回 (q, scale)
    pass


def dequantize(q, scale):
    pass
''',
    solution='''import numpy as np


def quantize(w):
    w = np.asarray(w, dtype=np.float32)
    biggest = float(np.abs(w).max()) if w.size else 0.0
    scale = biggest / 127 if biggest > 0 else 1.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q, scale):
    return q.astype(np.float32) * np.float32(scale)
''',
    setup='''w = np.random.default_rng(0).normal(size=(64, 32)).astype(np.float32)''',
    tests=[
        ("示例里的数", '''q, scale = quantize(np.array([-0.6, 0.2, 1.0], dtype=np.float32))
assert list(q) == [-76, 25, 127]
assert abs(scale - 1 / 127) < 1e-7'''),
        ("q 是 int8，范围在 ±127 以内", '''q, scale = quantize(w)
assert q.dtype == np.int8
assert q.min() >= -127 and q.max() <= 127
assert np.abs(q).max() == 127'''),
        ("还原之后误差不超过半个 scale", '''q, scale = quantize(w)
back = dequantize(q, scale)
assert back.dtype == np.float32
assert np.abs(back - w).max() <= scale / 2 + 1e-6'''),
        ("全为 0 的输入", '''q, scale = quantize(np.zeros(5, dtype=np.float32))
assert scale == 1.0 and not q.any()'''),
        ("占的字节是原来的四分之一", '''q, scale = quantize(w)
assert q.nbytes * 4 == w.nbytes'''),
        ("正负对称", '''q1, s1 = quantize(w)
q2, s2 = quantize(-w)
assert abs(s1 - s2) < 1e-12 and np.array_equal(q2, -q1)'''),
    ],
)

# ───────────────────────── 5 Agent / CUA ─────────────────────────

P(
    id="s5-tool-call", stage=5, level="中", title="解析并校验工具调用",
    statement="""
<p>模型输出的是一段文本。执行之前，harness 要先确认它是一个合法的工具调用。</p>
<p>实现 <code>parse_call(raw, tools)</code>：</p>
<ul>
  <li><code>raw</code>：模型输出的字符串，约定是 JSON：<code>{"tool": "名字", "args": {...}}</code>。没有 <code>"args"</code> 时当作 <code>{}</code>。</li>
  <li><code>tools</code>：<code>{工具名: {"params": {参数名: 类型名}}}</code>，类型名是 <code>"str"</code>、<code>"int"</code>、<code>"float"</code>、<code>"bool"</code> 之一。所有参数都必填。</li>
</ul>
<p>合法时返回 <code>({"tool": 名字, "args": 参数}, None)</code>。不合法时返回 <code>(None, 错误信息)</code>，错误信息以下面某个前缀开头，并带上出问题的工具名或参数名。按这个顺序检查，遇到第一个问题就返回：</p>
<ol>
  <li><code>invalid_json</code>：不是合法 JSON；或者不是对象；或者没有字符串类型的 <code>"tool"</code>；或者 <code>"args"</code> 不是对象。</li>
  <li><code>unknown_tool</code>：工具名不在 <code>tools</code> 里。</li>
  <li><code>missing_arg</code>：少了必填参数。</li>
  <li><code>unexpected_arg</code>：多了没定义的参数。</li>
  <li><code>wrong_type</code>：参数类型不对。</li>
</ol>
<p>类型规则：<code>"float"</code> 接受整数和小数；<code>True</code> / <code>False</code> 只能给 <code>"bool"</code>，不能当 <code>"int"</code> 或 <code>"float"</code> 用。</p>
<p>这个函数<strong>永远不抛异常</strong>：错误信息是要回给模型看的。</p>
<h3>示例</h3>
<pre><code>tools = {"click": {"params": {"selector": "str"}}}
parse_call('{"tool": "click", "args": {"selector": "#buy"}}', tools)
# ({"tool": "click", "args": {"selector": "#buy"}}, None)
parse_call('{"tool": "click", "args": {}}', tools)
# (None, "missing_arg: selector")</code></pre>
""",
    starter='''import json


def parse_call(raw, tools):
    # 返回 (call, None) 或 (None, error)
    pass
''',
    solution='''import json

TYPES = {"str": str, "int": int, "float": (int, float), "bool": bool}


def parse_call(raw, tools):
    try:
        obj = json.loads(raw)
    except Exception:
        return None, "invalid_json: 不是合法的 JSON"
    if not isinstance(obj, dict) or not isinstance(obj.get("tool"), str):
        return None, 'invalid_json: 需要 {"tool": "...", "args": {...}}'
    name, args = obj["tool"], obj.get("args", {})
    if not isinstance(args, dict):
        return None, "invalid_json: args 必须是对象"

    if name not in tools:
        return None, f"unknown_tool: {name}"
    params = tools[name]["params"]
    for p in params:
        if p not in args:
            return None, f"missing_arg: {p}"
    for a in args:
        if a not in params:
            return None, f"unexpected_arg: {a}"
    for p, type_name in params.items():
        value = args[p]
        is_bool = isinstance(value, bool)          # bool 是 int 的子类，要单独挡住
        if (is_bool and type_name != "bool") or not isinstance(value, TYPES[type_name]):
            return None, f"wrong_type: {p}"
    return {"tool": name, "args": args}, None
''',
    setup='''TOOLS = {
    "click": {"params": {"selector": "str"}},
    "type": {"params": {"selector": "str", "text": "str"}},
    "scroll": {"params": {"pixels": "int"}},
    "wait": {"params": {"seconds": "float"}},
    "set_checkbox": {"params": {"selector": "str", "checked": "bool"}},
    "screenshot": {"params": {}},
}''',
    tests=[
        ("合法的调用", '''call, err = parse_call('{"tool": "click", "args": {"selector": "#buy"}}', TOOLS)
assert err is None
assert call == {"tool": "click", "args": {"selector": "#buy"}}'''),
        ("没有 args 时当作空对象", '''call, err = parse_call('{"tool": "screenshot"}', TOOLS)
assert err is None and call == {"tool": "screenshot", "args": {}}'''),
        ("不是 JSON", '''for raw in ["我来点击购买按钮", "", "{tool: click}"]:
    call, err = parse_call(raw, TOOLS)
    assert call is None and err.startswith("invalid_json"), raw'''),
        ("是 JSON 但形状不对", '''for raw in ["[1, 2]", "null", '{"args": {}}', '{"tool": 3}', '{"tool": "click", "args": [1]}']:
    call, err = parse_call(raw, TOOLS)
    assert call is None and err.startswith("invalid_json"), raw'''),
        ("未知工具", '''call, err = parse_call('{"tool": "buy_now", "args": {}}', TOOLS)
assert call is None and err.startswith("unknown_tool") and "buy_now" in err'''),
        ("少了参数", '''call, err = parse_call('{"tool": "type", "args": {"selector": "#q"}}', TOOLS)
assert call is None and err.startswith("missing_arg") and "text" in err'''),
        ("多了参数", '''call, err = parse_call('{"tool": "click", "args": {"selector": "#buy", "force": true}}', TOOLS)
assert call is None and err.startswith("unexpected_arg") and "force" in err'''),
        ("类型不对", '''call, err = parse_call('{"tool": "scroll", "args": {"pixels": "300"}}', TOOLS)
assert call is None and err.startswith("wrong_type") and "pixels" in err'''),
        ("true 不能当整数用", '''call, err = parse_call('{"tool": "scroll", "args": {"pixels": true}}', TOOLS)
assert call is None and err.startswith("wrong_type")
call, err = parse_call('{"tool": "set_checkbox", "args": {"selector": "#a", "checked": true}}', TOOLS)
assert err is None'''),
        ("float 参数接受整数", '''call, err = parse_call('{"tool": "wait", "args": {"seconds": 2}}', TOOLS)
assert err is None and call["args"]["seconds"] == 2
call, err = parse_call('{"tool": "wait", "args": {"seconds": 0.5}}', TOOLS)
assert err is None'''),
    ],
)

P(
    id="s5-retry", stage=5, level="中", title="超时和重试",
    statement="""
<p>工具调用会偶尔超时。实现 <code>call_with_retry(fn, max_attempts=3, base_delay=1.0, sleep=time.sleep, retryable=(TimeoutError,))</code>：</p>
<ul>
  <li>调用 <code>fn()</code>，成功就返回它的结果。</li>
  <li>抛出 <code>retryable</code> 里的异常时：如果还有剩余次数，先 <code>sleep(base_delay * 2 ** (已失败次数 − 1))</code>，再试一次。等待时间依次是 1、2、4…倍的 <code>base_delay</code>。</li>
  <li>第 <code>max_attempts</code> 次还失败，把这次的异常抛出去，之后<strong>不要</strong>再 sleep。</li>
  <li>其他异常说明不是暂时性的问题，立刻抛出，不重试。</li>
</ul>
<p><code>sleep</code> 是传进来的，测试里会换成一个只记录时长的函数，所以不会真的等。</p>
<h3>示例</h3>
<pre><code># fn 前两次超时，第三次返回 "ok"
call_with_retry(fn)          # 返回 "ok"，中间 sleep(1.0)、sleep(2.0)</code></pre>
""",
    starter='''import time


def call_with_retry(fn, max_attempts=3, base_delay=1.0, sleep=time.sleep, retryable=(TimeoutError,)):
    pass
''',
    solution='''import time


def call_with_retry(fn, max_attempts=3, base_delay=1.0, sleep=time.sleep, retryable=(TimeoutError,)):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retryable:
            if attempt == max_attempts:
                raise
            sleep(base_delay * 2 ** (attempt - 1))
''',
    setup='''def flaky(*outcomes):
    """按顺序给出结果：异常类就抛出，其他值就返回。"""
    it = iter(outcomes)
    calls = []
    def fn():
        calls.append(1)
        o = next(it)
        if isinstance(o, type) and issubclass(o, BaseException):
            raise o("模拟的失败")
        return o
    fn.calls = calls
    return fn''',
    tests=[
        ("第一次就成功，不 sleep", '''delays = []
fn = flaky("ok")
assert call_with_retry(fn, sleep=delays.append) == "ok"
assert len(fn.calls) == 1 and delays == []'''),
        ("超时两次后成功，等待时间翻倍", '''delays = []
fn = flaky(TimeoutError, TimeoutError, "ok")
assert call_with_retry(fn, sleep=delays.append) == "ok"
assert len(fn.calls) == 3 and delays == [1.0, 2.0]'''),
        ("base_delay 可以调", '''delays = []
fn = flaky(TimeoutError, TimeoutError, TimeoutError, "ok")
assert call_with_retry(fn, max_attempts=5, base_delay=0.5, sleep=delays.append) == "ok"
assert delays == [0.5, 1.0, 2.0]'''),
        ("次数用完后抛出，最后一次之后不 sleep", '''delays = []
fn = flaky(TimeoutError, TimeoutError, TimeoutError, "ok")
try:
    call_with_retry(fn, max_attempts=3, sleep=delays.append)
except TimeoutError:
    pass
else:
    assert False, "三次都超时，应该抛出 TimeoutError"
assert len(fn.calls) == 3 and delays == [1.0, 2.0]'''),
        ("不可重试的异常立刻抛出", '''delays = []
fn = flaky(ValueError, "ok")
try:
    call_with_retry(fn, sleep=delays.append)
except ValueError:
    pass
else:
    assert False, "ValueError 不该被吞掉"
assert len(fn.calls) == 1 and delays == []'''),
        ("retryable 可以指定多种异常", '''delays = []
fn = flaky(ConnectionError, TimeoutError, 42)
assert call_with_retry(fn, sleep=delays.append, retryable=(TimeoutError, ConnectionError)) == 42
assert len(fn.calls) == 3'''),
    ],
)

P(
    id="s5-trim", stage=5, level="易", title="把对话历史裁进预算",
    statement="""
<p>Agent 每走一步，上下文就长一截。放不下时要丢掉旧的。</p>
<p>实现 <code>trim(messages, budget, count=len)</code>：</p>
<ul>
  <li><code>messages</code>：<code>{"role": ..., "content": ...}</code> 的 list，按时间顺序。</li>
  <li>一条消息的长度是 <code>count(message["content"])</code>，默认按字符数算。</li>
  <li>第一条如果是 <code>"system"</code>，一定保留，它的长度也算在预算里。</li>
  <li>其余的从<strong>最新</strong>的往回看，一条条加进来，直到下一条放不下为止。遇到放不下的就停，不要跳过它去拿更早的：历史中间不能有洞。</li>
</ul>
<p>返回保留下来的消息，顺序和原来一样。不要修改传进来的 list。</p>
<h3>示例</h3>
<pre><code>msgs = [{"role": "system", "content": "ab"},
        {"role": "user", "content": "cccc"},
        {"role": "assistant", "content": "dd"},
        {"role": "user", "content": "e"}]
trim(msgs, budget=5)   # 保留 system、"dd"、"e"：2 + 2 + 1 = 5</code></pre>
""",
    starter='''def trim(messages, budget, count=len):
    pass
''',
    solution='''def trim(messages, budget, count=len):
    if not messages:
        return []
    head = [messages[0]] if messages[0]["role"] == "system" else []
    used = sum(count(m["content"]) for m in head)
    kept = []
    for m in reversed(messages[len(head):]):
        size = count(m["content"])
        if used + size > budget:
            break                                  # 放不下就停，不往更早的跳
        kept.append(m)
        used += size
    return head + kept[::-1]
''',
    setup='''def M(role, content):
    return {"role": role, "content": content}
SYS = M("system", "ab")
msgs = [SYS, M("user", "cccc"), M("assistant", "dd"), M("user", "e")]''',
    tests=[
        ("放得下就原样返回", '''assert trim(msgs, 100) == msgs'''),
        ("先丢最旧的", '''assert trim(msgs, 5) == [SYS, M("assistant", "dd"), M("user", "e")]'''),
        ("system 一定保留", '''assert trim(msgs, 2) == [SYS]
assert trim(msgs, 3) == [SYS, M("user", "e")]'''),
        ("遇到放不下的就停，中间不留洞", '''m = [SYS, M("user", "aaaa"), M("assistant", "b" * 10), M("user", "cc")]
assert trim(m, 9) == [SYS, M("user", "cc")], "不能跳过放不下的那条去拿更早的"'''),
        ("没有 system 消息", '''m = [M("user", "aaa"), M("assistant", "bb"), M("user", "c")]
assert trim(m, 3) == [M("assistant", "bb"), M("user", "c")]'''),
        ("可以换计数方式", '''m = [SYS, M("user", "one two three"), M("assistant", "four five"), M("user", "six")]
words = lambda s: len(s.split())
assert trim(m, 4, count=words) == [SYS, M("assistant", "four five"), M("user", "six")]'''),
        ("不修改传进来的 list", '''before = list(msgs)
trim(msgs, 3)
assert msgs == before and len(msgs) == 4
assert trim([], 10) == []'''),
    ],
)

P(
    id="s5-agent-loop", stage=5, level="难", title="验：跑一轮任务，把每一步记下来",
    statement="""
<p>这是第 5 步的验收：写 agent 的主循环，并且每一步留一条记录。</p>
<p>实现 <code>run_episode(model, env, tools, max_steps=10)</code>，返回日志：一个 list，每步一个 dict。</p>
<ul>
  <li><code>env.reset()</code> 返回初始观察；<code>env.observe()</code> 返回当前观察。观察是字符串。</li>
  <li><code>model(history, obs)</code> 返回模型的原始输出（字符串），约定是 JSON：<code>{"tool": "名字", "args": {...}}</code>，没有 <code>"args"</code> 时当作 <code>{}</code>。</li>
  <li><code>tools[名字](env, **args)</code> 执行动作，返回一个字符串。</li>
</ul>
<p>每一步：</p>
<ol>
  <li>调用 <code>model(history, obs)</code> 得到 <code>raw</code>。</li>
  <li>解析并执行，得到这一步的记录：
<pre><code>{"step": 步号（从 0 开始）, "obs": 模型这一步看到的观察, "raw": 模型的原始输出,
 "action": 解析出的 dict（解析失败时是 None）,
 "result": 字符串, "error": None 或错误名}</code></pre>
    <ul>
      <li>解析失败（不是 JSON、不是对象、没有 <code>"tool"</code>）：<code>error = "invalid_json"</code>。</li>
      <li>工具名是 <code>"done"</code>：<code>result = "done"</code>，记录这一步后结束。</li>
      <li>工具名不在 <code>tools</code> 里：<code>error = "unknown_tool"</code>。</li>
      <li>工具抛异常：<code>error</code> 是异常的类名，比如 <code>"TimeoutError"</code>。</li>
      <li>有错误时 <code>result = "error: " + error</code>；否则 <code>result</code> 是工具的返回值。</li>
    </ul>
  </li>
  <li>把 <code>(obs, raw, result)</code> 追加到 <code>history</code>，这样模型下一步能看到刚才发生了什么。</li>
  <li>如果没结束，<code>obs = env.observe()</code>，继续。最多走 <code>max_steps</code> 步。</li>
</ol>
<p>任何一步出错都<strong>不能让循环崩掉</strong>：记下来，把错误交给模型，继续走。</p>
""",
    starter='''import json


def run_episode(model, env, tools, max_steps=10):
    # 返回日志：每步一个 dict
    pass
''',
    solution='''import json


def run_episode(model, env, tools, max_steps=10):
    obs = env.reset()
    history, log = [], []
    for step in range(max_steps):
        raw = model(history, obs)
        rec = {"step": step, "obs": obs, "raw": raw, "action": None, "result": None, "error": None}
        done = False

        try:
            action = json.loads(raw)
            if not isinstance(action, dict) or "tool" not in action:
                raise ValueError("不是工具调用")
        except Exception:
            rec["error"] = "invalid_json"
        else:
            rec["action"] = action
            name = action["tool"]
            if name == "done":
                rec["result"] = "done"
                done = True
            elif name not in tools:
                rec["error"] = "unknown_tool"
            else:
                try:
                    rec["result"] = tools[name](env, **action.get("args", {}))
                except Exception as exc:
                    rec["error"] = type(exc).__name__

        if rec["error"]:
            rec["result"] = "error: " + rec["error"]
        log.append(rec)
        history.append((obs, raw, rec["result"]))
        if done:
            break
        obs = env.observe()
    return log
''',
    setup='''class Env:
    def reset(self):
        self.n = 0
        return "n=0"
    def observe(self):
        return f"n={self.n}"

def add(env, by):
    env.n += by
    return "ok"

def boom(env):
    raise TimeoutError("页面没加载出来")

TOOLS = {"add": add, "boom": boom}
ADD2 = '{"tool": "add", "args": {"by": 2}}'
ADD3 = '{"tool": "add", "args": {"by": 3}}'
DONE = '{"tool": "done"}'

def scripted(*outputs):
    """按顺序输出，并记下每次被调用时看到了什么。"""
    it = iter(outputs)
    seen = []
    def model(history, obs):
        seen.append({"n_history": len(history), "obs": obs, "last": history[-1] if history else None})
        return next(it)
    model.seen = seen
    return model''',
    tests=[
        ("正常跑完一个任务", '''env = Env()
log = run_episode(scripted(ADD2, ADD3, DONE), env, TOOLS)
assert env.n == 5
assert [r["step"] for r in log] == [0, 1, 2]
assert log[-1]["result"] == "done"'''),
        ("每条记录的字段齐全", '''log = run_episode(scripted(ADD2, ADD3, DONE), Env(), TOOLS)
for r in log:
    assert set(r) == {"step", "obs", "raw", "action", "result", "error"}, set(r)
assert [r["obs"] for r in log] == ["n=0", "n=2", "n=5"], "obs 是模型在这一步看到的观察"
assert log[0]["raw"] == ADD2
assert log[0]["action"] == {"tool": "add", "args": {"by": 2}}
assert log[0]["result"] == "ok" and log[0]["error"] is None'''),
        ("解析失败时记下来，继续走", '''env = Env()
log = run_episode(scripted("我想加 2", ADD2, DONE), env, TOOLS)
assert len(log) == 3 and env.n == 2
assert log[0]["error"] == "invalid_json" and log[0]["action"] is None
assert log[0]["result"] == "error: invalid_json"'''),
        ("未知工具", '''log = run_episode(scripted('{"tool": "multiply", "args": {"by": 2}}', DONE), Env(), TOOLS)
assert log[0]["error"] == "unknown_tool" and log[0]["result"] == "error: unknown_tool"
assert log[0]["action"] == {"tool": "multiply", "args": {"by": 2}}'''),
        ("工具抛异常时循环不崩", '''env = Env()
log = run_episode(scripted('{"tool": "boom"}', ADD3, DONE), env, TOOLS)
assert log[0]["error"] == "TimeoutError" and log[0]["result"] == "error: TimeoutError"
assert len(log) == 3 and env.n == 3'''),
        ("参数不对也算工具执行出错", '''log = run_episode(scripted('{"tool": "add", "args": {"amount": 2}}', DONE), Env(), TOOLS)
assert log[0]["error"] == "TypeError"'''),
        ("模型下一步能看到上一步的结果", '''m = scripted("胡说", ADD2, DONE)
run_episode(m, Env(), TOOLS)
assert [s["n_history"] for s in m.seen] == [0, 1, 2]
assert m.seen[1]["last"] == ("n=0", "胡说", "error: invalid_json")
assert m.seen[2]["last"] == ("n=0", ADD2, "ok")
assert m.seen[2]["obs"] == "n=2"'''),
        ("最多走 max_steps 步", '''env = Env()
log = run_episode(scripted(*[ADD2] * 20), env, TOOLS, max_steps=4)
assert len(log) == 4 and env.n == 8'''),
        ("done 之后不再调用模型", '''m = scripted(DONE, ADD2, ADD2)
env = Env()
log = run_episode(m, env, TOOLS)
assert len(log) == 1 and len(m.seen) == 1 and env.n == 0'''),
    ],
)

# ───────────────────────── 6 Evals 与面试 ─────────────────────────

P(
    id="s6-checker", stage=6, level="易", title="独立检查器",
    statement="""
<p>任务有没有完成，由检查器读环境的最终状态来判定，不听 agent 自己怎么说。</p>
<p><code>check(final_state, expected)</code>：</p>
<ul>
  <li><code>final_state</code>：任务结束时环境的状态，一个可以嵌套的 dict。</li>
  <li><code>expected</code>：<code>{路径: 期望的值}</code>。路径用点号一层层往下取，比如 <code>"cart.total"</code> 指 <code>final_state["cart"]["total"]</code>。</li>
</ul>
<p>返回 <code>(passed, reasons)</code>。<code>reasons</code> 是字符串的 list，每个不满足的条件一条，里面要带上那条路径。路径不存在也算不满足。全部满足时是 <code>(True, [])</code>。<code>final_state</code> 里多出来的字段不影响结果。</p>
<p><code>score(run, expected)</code>：<code>run</code> 是 <code>{"claimed_done": bool, "final_state": {...}}</code>。返回任务是否成功，<strong>只</strong>取决于 <code>check</code> 的结果。</p>
<h3>示例</h3>
<pre><code>state = {"cart": {"total": 30, "items": ["书"]}, "page": "checkout"}
check(state, {"cart.total": 30, "page": "done"})
# (False, ["page: 期望 'done'，实际 'checkout'"])</code></pre>
""",
    starter='''def check(final_state, expected):
    # 返回 (passed, reasons)
    pass


def score(run, expected):
    pass
''',
    solution='''MISSING = object()


def lookup(state, path):
    current = state
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current


def check(final_state, expected):
    reasons = []
    for path, want in expected.items():
        got = lookup(final_state, path)
        if got is MISSING:
            reasons.append(f"{path}: 期望 {want!r}，实际不存在")
        elif got != want:
            reasons.append(f"{path}: 期望 {want!r}，实际 {got!r}")
    return len(reasons) == 0, reasons


def score(run, expected):
    passed, _ = check(run["final_state"], expected)
    return passed
''',
    setup='''state = {"cart": {"total": 30, "items": ["书"]}, "page": "checkout", "user": {"address": {"city": "匹兹堡"}}}''',
    tests=[
        ("全部满足", '''assert check(state, {"page": "checkout", "cart.total": 30}) == (True, [])'''),
        ("一个条件不满足", '''passed, reasons = check(state, {"cart.total": 30, "page": "done"})
assert passed is False
assert len(reasons) == 1 and "page" in reasons[0]'''),
        ("每个不满足的条件各一条原因", '''passed, reasons = check(state, {"cart.total": 99, "page": "done", "cart.items": ["书"]})
assert passed is False and len(reasons) == 2
assert any("cart.total" in r for r in reasons) and any("page" in r for r in reasons)'''),
        ("多层路径", '''assert check(state, {"user.address.city": "匹兹堡"}) == (True, [])
assert check(state, {"user.address.city": "纽约"})[0] is False'''),
        ("路径不存在算不满足，不报错", '''for path in ["order_id", "cart.coupon", "page.title", "user.address.zip.code"]:
    passed, reasons = check(state, {path: 1})
    assert passed is False and len(reasons) == 1 and path in reasons[0], path'''),
        ("状态里多出来的字段不影响", '''assert check(state, {"page": "checkout"})[0] is True
assert check(state, {}) == (True, [])'''),
        ("agent 说完成了，状态不对，算失败", '''run = {"claimed_done": True, "final_state": {"page": "checkout"}}
assert score(run, {"page": "done"}) is False'''),
        ("agent 没说完成，状态对了，算成功", '''run = {"claimed_done": False, "final_state": {"page": "done"}}
assert score(run, {"page": "done"}) is True'''),
    ],
)

P(
    id="s6-passk", stage=6, level="易", title="pass@k 和 pass^k",
    statement="""
<p>同一个任务跑 k 次，结果不一定每次都一样。两个指标回答两个不同的问题。</p>
<p><code>trials</code> 是一个 list，每个任务一行，每行是 k 个 <code>bool</code>：这个任务的 k 次运行各自成没成功。实现三个函数，都返回 0 到 1 之间的小数：</p>
<ul>
  <li><code>success_rate(trials)</code>：所有运行里成功的比例。</li>
  <li><code>pass_at_k(trials)</code>：k 次里<strong>至少一次</strong>成功的任务比例。量的是能力上限。</li>
  <li><code>pass_hat_k(trials)</code>：k 次<strong>全部</strong>成功的任务比例。量的是可靠性。</li>
</ul>
<h3>示例</h3>
<pre><code>trials = [[True,  True,  True ],
          [True,  False, False],
          [False, False, False],
          [True,  True,  False]]
success_rate(trials)   # 0.5
pass_at_k(trials)      # 0.75
pass_hat_k(trials)     # 0.25</code></pre>
<p>同一组运行，说成「75% 的任务能做」和「25% 的任务靠得住」都对。报告结果时要说清用的是哪个。</p>
""",
    starter='''def success_rate(trials):
    pass


def pass_at_k(trials):
    pass


def pass_hat_k(trials):
    pass
''',
    solution='''def success_rate(trials):
    flat = [t for task in trials for t in task]
    return sum(flat) / len(flat)


def pass_at_k(trials):
    return sum(any(task) for task in trials) / len(trials)


def pass_hat_k(trials):
    return sum(all(task) for task in trials) / len(trials)
''',
    setup='''T, F = True, False
trials = [[T, T, T], [T, F, F], [F, F, F], [T, T, F]]''',
    tests=[
        ("示例里的数", '''assert abs(success_rate(trials) - 0.5) < 1e-9
assert abs(pass_at_k(trials) - 0.75) < 1e-9
assert abs(pass_hat_k(trials) - 0.25) < 1e-9'''),
        ("k = 1 时三个数相等", '''one = [[T], [F], [T], [T]]
assert success_rate(one) == pass_at_k(one) == pass_hat_k(one) == 0.75'''),
        ("pass^k ≤ 成功率 ≤ pass@k", '''import random
random.seed(0)
for _ in range(20):
    tr = [[random.random() < 0.6 for _ in range(4)] for _ in range(15)]
    assert pass_hat_k(tr) <= success_rate(tr) + 1e-12 <= pass_at_k(tr) + 2e-12'''),
        ("全部成功和全部失败", '''assert pass_hat_k([[T, T], [T, T]]) == 1.0 and pass_at_k([[T, T], [T, T]]) == 1.0
assert pass_hat_k([[F, F], [F, F]]) == 0.0 and pass_at_k([[F, F], [F, F]]) == 0.0'''),
        ("成功率一样，可靠性可以差很多", '''steady = [[T, T], [T, T], [F, F], [F, F]]      # 一半任务稳定成功
flaky = [[T, F], [F, T], [T, F], [F, T]]       # 每个任务都时好时坏
assert success_rate(steady) == success_rate(flaky) == 0.5
assert pass_hat_k(steady) == 0.5 and pass_hat_k(flaky) == 0.0
assert pass_at_k(steady) == 0.5 and pass_at_k(flaky) == 1.0'''),
    ],
)

P(
    id="s6-mcnemar", stage=6, level="中", title="两版 harness，配对比较",
    statement="""
<p>固定模型，两版 harness 跑同一批任务。结果是配对的：对每个任务，你知道 A 过没过、B 过没过。</p>
<p>两版都过或都没过的任务，分不出谁好。差别全在只有一版通过的那些任务上。</p>
<p>实现 <code>paired(pass_a, pass_b)</code>，两个等长的 <code>bool</code> list。返回一个 dict：</p>
<ul>
  <li><code>"only_a"</code>：只有 A 通过的任务数，记作 b。</li>
  <li><code>"only_b"</code>：只有 B 通过的任务数，记作 c。</li>
  <li><code>"p_value"</code>：McNemar 精确检验的双侧 p 值。如果两版一样好，这 <code>n = b + c</code> 个任务每个都是一半一半：
<pre><code>p = min(1, 2 × Σ_{i=0}^{min(b, c)} C(n, i) / 2ⁿ)</code></pre>
    <code>n = 0</code> 时 <code>p = 1.0</code>。</li>
</ul>
<p>两个 list 长度不同时抛出 <code>ValueError</code>。</p>
<h3>示例</h3>
<pre><code># 50 个任务：A 过 31 个（62%），B 过 33 个（66%）
# 其中只有 A 过的 3 个，只有 B 过的 5 个
paired(pass_a, pass_b)   # {"only_a": 3, "only_b": 5, "p_value": 0.7266}</code></pre>
<p>p 值 0.73：这 4 个点的差距，在两版一样好的情况下也很容易出现。</p>
""",
    starter='''from math import comb


def paired(pass_a, pass_b):
    pass
''',
    solution='''from math import comb


def paired(pass_a, pass_b):
    if len(pass_a) != len(pass_b):
        raise ValueError("两版的任务数不一致")
    only_a = sum(1 for a, b in zip(pass_a, pass_b) if a and not b)
    only_b = sum(1 for a, b in zip(pass_a, pass_b) if b and not a)
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
        p = min(1.0, 2 * tail / 2 ** n)
    return {"only_a": only_a, "only_b": only_b, "p_value": p}
''',
    setup='''def mk(both, only_a, only_b, neither):
    a = [True] * both + [True] * only_a + [False] * only_b + [False] * neither
    b = [True] * both + [False] * only_a + [True] * only_b + [False] * neither
    return a, b''',
    tests=[
        ("62% 对 66%，50 个任务", '''a, b = mk(28, 3, 5, 14)
assert sum(a) == 31 and sum(b) == 33 and len(a) == 50
r = paired(a, b)
assert r["only_a"] == 3 and r["only_b"] == 5
assert abs(r["p_value"] - 0.7265625) < 1e-9'''),
        ("一边倒的差别", '''r = paired(*mk(20, 0, 10, 20))
assert r["only_a"] == 0 and r["only_b"] == 10
assert abs(r["p_value"] - 0.001953125) < 1e-12'''),
        ("没有不一致的任务时 p 是 1", '''r = paired(*mk(30, 0, 0, 20))
assert r == {"only_a": 0, "only_b": 0, "p_value": 1.0}'''),
        ("p 值不超过 1", '''r = paired(*mk(0, 2, 2, 0))
assert r["p_value"] == 1.0'''),
        ("A 和 B 对调，p 值不变", '''a, b = mk(10, 1, 7, 5)
r1, r2 = paired(a, b), paired(b, a)
assert r1["only_a"] == r2["only_b"] == 1 and r1["only_b"] == r2["only_a"] == 7
assert abs(r1["p_value"] - r2["p_value"]) < 1e-12
assert abs(r1["p_value"] - 0.0703125) < 1e-9'''),
        ("总成功率一样，配对结果可以不一样", '''same = paired(*mk(30, 0, 0, 20))
swapped = paired(*mk(20, 10, 10, 10))
assert same["only_a"] == 0 and swapped["only_a"] == 10 and swapped["only_b"] == 10'''),
        ("任务数不一致时抛出 ValueError", '''try:
    paired([True, False], [True])
except ValueError:
    pass
else:
    assert False, "长度不一致时应该抛出 ValueError"'''),
    ],
)

P(
    id="s6-triage", stage=6, level="中", title="验：把失败分成规划、工具执行、评分",
    statement="""
<p>这是第 6 步的验收。读完日志之后，每一次运行有这些人工确认过的信息：</p>
<pre><code>run = {
  "checker_passed": bool,     # 检查器判的结果
  "truly_done": bool,         # 人看过最终状态：任务到底完成没有
  "steps": [                  # 每一步
    {"intended_ok": bool,     # 模型想做的这个动作，方向对不对
     "executed_ok": bool},    # 这个动作有没有按意图执行成功
  ],
}</code></pre>
<p>实现 <code>triage(run)</code>，返回四个字符串之一。按顺序判断，前面的优先：</p>
<ol>
  <li><code>"grading"</code>：检查器判的和实际情况不一致（做完了被判失败，或没做完被判成功）。</li>
  <li><code>"success"</code>：确实完成了，检查器也判了通过。</li>
  <li><code>"tool"</code>：有某一步方向是对的（<code>intended_ok</code>），却没执行成功（<code>executed_ok</code> 为假）。</li>
  <li><code>"planning"</code>：其余的失败。动作要么都执行成功了但任务没完成，要么失败的动作本身方向就不对。</li>
</ol>
<p>再实现 <code>table(runs)</code>：返回 <code>{类别: 次数}</code> 的 dict，只包含出现过的类别。</p>
<p>顺序是有道理的。评分排第一：检查器有错，其他的数字都不可信。工具执行排在规划前面：动作没落地时，还看不出计划本身行不行。</p>
""",
    starter='''def triage(run):
    pass


def table(runs):
    pass
''',
    solution='''from collections import Counter


def triage(run):
    if run["checker_passed"] != run["truly_done"]:
        return "grading"
    if run["truly_done"]:
        return "success"
    if any(s["intended_ok"] and not s["executed_ok"] for s in run["steps"]):
        return "tool"
    return "planning"


def table(runs):
    return dict(Counter(triage(r) for r in runs))
''',
    setup='''def S(intended_ok, executed_ok):
    return {"intended_ok": intended_ok, "executed_ok": executed_ok}

def R(checker_passed, truly_done, *steps):
    return {"checker_passed": checker_passed, "truly_done": truly_done, "steps": list(steps)}

GOOD, BAD_PLAN, BAD_EXEC, BAD_BOTH = S(True, True), S(False, True), S(True, False), S(False, False)''',
    tests=[
        ("做完了，也判对了", '''assert triage(R(True, True, GOOD, GOOD)) == "success"'''),
        ("做完了却被判失败", '''assert triage(R(False, True, GOOD, GOOD)) == "grading"'''),
        ("没做完却被判通过", '''assert triage(R(True, False, GOOD, BAD_PLAN)) == "grading"'''),
        ("方向对，动作没落地", '''assert triage(R(False, False, GOOD, BAD_EXEC, GOOD)) == "tool"'''),
        ("动作都执行了，任务没完成", '''assert triage(R(False, False, GOOD, GOOD)) == "planning"
assert triage(R(False, False, GOOD, BAD_PLAN)) == "planning"
assert triage(R(False, False)) == "planning"'''),
        ("方向本来就错的动作失败了，不算工具问题", '''assert triage(R(False, False, GOOD, BAD_BOTH)) == "planning"'''),
        ("评分优先于工具，工具优先于规划", '''assert triage(R(True, False, BAD_EXEC)) == "grading"
assert triage(R(False, False, BAD_PLAN, BAD_EXEC)) == "tool"'''),
        ("汇总成表", '''runs = [R(True, True, GOOD), R(True, True, GOOD), R(False, True, GOOD),
        R(False, False, BAD_EXEC), R(False, False, GOOD), R(False, False, BAD_PLAN)]
assert table(runs) == {"success": 2, "grading": 1, "tool": 1, "planning": 2}
assert table([R(True, True, GOOD)]) == {"success": 1}'''),
    ],
)
