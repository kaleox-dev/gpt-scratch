"""Verification suite for kv_cache.py.

Checks that KV-cached generation is mathematically identical to the naive
full-recompute generation in gpt.py, then measures timing behavior.

Run:  python test_kv_cache.py [--smoke]
(--smoke additionally runs a tiny 12-iteration end-to-end training pass on a
temp copy of kv_cache.py.)
"""
import os
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_ns(path, device=None):
    """Exec a script's definitions (everything before its training block)
    into a fresh namespace so classes can be built without running training."""
    src = open(path, 'r', encoding='utf-8').read()
    prefix = src.split('model = GPTLanguageModel()')[0]
    assert 'class GPTLanguageModel' in prefix, f'unexpected layout in {path}'
    ns = {'__name__': 'loaded_' + os.path.basename(path)}
    exec(compile(prefix, path, 'exec'), ns)
    if device is not None:
        ns['device'] = device  # forward() reads this global at call time
    return ns


def greedy_naive(model, idx, n, bs):
    for _ in range(n):
        logits, _ = model(idx[:, -bs:])
        idx = torch.cat([idx, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    return idx


def greedy_cached(model, idx, n, bs):
    for blk in model.blocks:
        for h in blk.sa.heads:
            h.reset_cache()
    n = min(n, bs - idx.shape[1])
    for i in range(n):
        x, off = (idx, 0) if i == 0 else (idx[:, -1:], idx.shape[1] - 1)
        logits, _ = model(x, cache=True, pos_offset=off)
        idx = torch.cat([idx, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    return idx


def sample_naive(model, idx, n, bs):
    for _ in range(n):
        logits, _ = model(idx[:, -bs:])
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx


def segment_time(model, cached, start, length, bs):
    """Generate `length` tokens; return time spent on steps [start, start+length)."""
    if cached:
        for blk in model.blocks:
            for h in blk.sa.heads:
                h.reset_cache()
    idx = torch.zeros((1, 1), dtype=torch.long)
    t0 = time.perf_counter()
    for i in range(start + length):
        if cached:
            x, off = (idx, 0) if i == 0 else (idx[:, -1:], idx.shape[1] - 1)
            logits, _ = model(x, cache=True, pos_offset=off)
        else:
            logits, _ = model(idx[:, -bs:])
        idx = torch.cat([idx, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        if i == start - 1:
            t0 = time.perf_counter()  # restart clock at the segment boundary
    return time.perf_counter() - t0


# ---------------- phase 1: exact equivalence (CPU, float64) ----------------
nsA = load_ns(os.path.join(HERE, 'gpt.py'), device='cpu')
nsB = load_ns(os.path.join(HERE, 'kv_cache.py'), device='cpu')
V, BS = nsA['vocab_size'], nsA['block_size']

torch.manual_seed(1337)
mA = nsA['GPTLanguageModel']().double().eval()
torch.manual_seed(1337)
mB = nsB['GPTLanguageModel']().double().eval()

for (n1, p1), (n2, p2) in zip(mA.named_parameters(), mB.named_parameters()):
    assert n1 == n2 and torch.equal(p1, p2), f'weight mismatch at {n1}'
print(f'[1] identical weights ({sum(p.numel() for p in mA.parameters())/1e6:.2f}M params): OK')

g = torch.Generator().manual_seed(7)
prompt = torch.randint(0, V, (1, 32), generator=g)
with torch.no_grad():
    la, _ = mA(prompt)
    lb, _ = mB(prompt, cache=True, pos_offset=0)
    assert torch.allclose(la, lb, rtol=1e-8, atol=1e-8), 'prefill logits differ'
    nxt = la[:, -1].argmax(-1, keepdim=True)
    la2, _ = mA(torch.cat([prompt, nxt], dim=1))
    lb2, _ = mB(nxt, cache=True, pos_offset=prompt.shape[1])
    assert torch.allclose(la2[:, -1], lb2[:, -1], rtol=1e-8, atol=1e-8), 'decode logits differ'
print(f'[2] cached step == naive full recompute (max logit diff '
      f'{(la2[:, -1] - lb2[:, -1]).abs().max():.2e}): OK')

with torch.no_grad():
    assert torch.equal(greedy_naive(mA, prompt.clone(), 64, BS),
                       greedy_cached(mB, prompt.clone(), 64, BS))
    p2 = torch.randint(0, V, (2, 24), generator=g)
    assert torch.equal(greedy_naive(mA, p2.clone(), 64, BS),
                       greedy_cached(mB, p2.clone(), 64, BS))
print('[3] greedy generation token-identical (B=1 and B=2): OK')

torch.manual_seed(123)
sa = sample_naive(mA, prompt[:, :1].clone(), 80, BS)
torch.manual_seed(123)
sb = mB.generate(prompt[:, :1].clone(), 80)
assert torch.equal(sa, sb), 'stochastic generation diverges'
print('[4] stochastic generation (multinomial, same seed) token-identical: OK')

out = mB.generate(prompt[:, :1].clone(), 10**9)
lens = {h.cache_len for blk in mB.blocks for h in blk.sa.heads}
assert out.shape[1] == BS, f'cap broken: len={out.shape[1]}'
assert lens == {out.shape[1] - 1}, f'cache_len inconsistent: {lens}'
print(f"[5] cap: huge request -> {out.shape[1] - 1} new tokens; all 36 heads at "
      f"cache_len={lens.pop()} (cache lags sequence by 1: final token is never fed back): OK")

# ---------------- phase 2: timing on the machine's real device ----------------
nsC = load_ns(os.path.join(HERE, 'kv_cache.py'))
nsD = load_ns(os.path.join(HERE, 'gpt.py'))
dev = nsC['device']
torch.manual_seed(1337)
mC = nsC['GPTLanguageModel']().to(dev).eval()
torch.manual_seed(1337)
mD = nsD['GPTLanguageModel']().to(dev).eval()

N = 150
ctx = torch.zeros((1, 1), dtype=torch.long, device=dev)
label = 'cuda' if dev == 'cuda' else f'cpu ({torch.get_num_threads()} threads)'
if dev == 'cuda':
    label += f' [{torch.cuda.get_device_name(0)}]'
with torch.no_grad():
    t0 = time.perf_counter()
    _ = greedy_naive(mD, ctx.clone(), N, BS)
    if dev == 'cuda':
        torch.cuda.synchronize()
    t_naive = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = greedy_cached(mC, ctx.clone(), N, BS)
    if dev == 'cuda':
        torch.cuda.synchronize()
    t_cached = time.perf_counter() - t0
print(f'[6] timing, {N} tokens on {label}:')
print(f'    naive  (full recompute): {t_naive*1000:8.1f} ms total, {t_naive/N*1000:6.2f} ms/token')
print(f'    cached (kv cache):       {t_cached*1000:8.1f} ms total, {t_cached/N*1000:6.2f} ms/token')
print(f'    speedup: {t_naive / t_cached:.1f}x')

with torch.no_grad():
    t_early_a = segment_time(mA, False, 0, 50, BS)
    t_late_a = segment_time(mA, False, 150, 50, BS)
    t_early_b = segment_time(mB, True, 0, 50, BS)
    t_late_b = segment_time(mB, True, 150, 50, BS)
print('[7] cost growth with context length (CPU):')
print(f'    naive : early {t_early_a/50*1000:6.2f} ms/token | late {t_late_a/50*1000:6.2f} '
      f'ms/token | growth x{t_late_a/t_early_a:.2f}')
print(f'    cached: early {t_early_b/50*1000:6.2f} ms/token | late {t_late_b/50*1000:6.2f} '
      f'ms/token | growth x{t_late_b/t_early_b:.2f}')

# ---------------- optional: end-to-end smoke train ----------------
if '--smoke' in sys.argv:
    src = open('kv_cache.py', 'r', encoding='utf-8').read()
    for old, new in [('max_iters = 5000', 'max_iters = 12'),
                     ('eval_interval = 500', 'eval_interval = 4'),
                     ('eval_iters = 200', 'eval_iters = 2'),
                     ('batch_size = 64', 'batch_size = 8')]:
        assert old in src, old
        src = src.replace(old, new, 1)
    fd, tmp = tempfile.mkstemp(suffix='.py', dir=tempfile.gettempdir())
    with os.fdopen(fd, 'w') as f:
        f.write(src)
    r = subprocess.run([sys.executable, tmp], cwd=HERE, capture_output=True,
                       text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    losses = [ln for ln in r.stdout.splitlines() if 'loss' in ln]
    assert len(losses) >= 3, losses
    print('[8] smoke train (12 iters, real script path): OK')
    for ln in losses:
        print('    ' + ln)

print('\nALL CHECKS PASSED')
