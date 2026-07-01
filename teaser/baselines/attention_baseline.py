#!/usr/bin/env python3
# =============================================================================
# attention_baseline.py
# A clean, standalone STANDARD-TRANSFORMER baseline for the CM-Galaxy teaser.
#
# This file is ONLY the attention baseline: a standard decoder-only Transformer
#   (token embedding + RoPE + causal attention + RMSNorm + SwiGLU + KV-cache).
# It contains NOTHING of CM-Galaxy's own architecture. Its purpose is
# transparency and reproducibility: it lets anyone confirm the baseline is a
# standard, non-weakened Transformer, and measure attention's own perplexity,
# long-gap induction recall, and decode memory / throughput.
#
# It is configured to match the exact baseline used in the CM-Galaxy comparison:
#   d_model=256, n_layers=8, n_heads=4, SwiGLU hidden = round(8/3*d) to mult 64,
#   gpt2 vocab (50257), UNTIED embeddings  ->  ~32.2M parameters.
# Attention uses F.scaled_dot_product_attention (FlashAttention / memory-efficient
# kernel on GPU) -- the fair, memory-efficient baseline used for the efficiency
# (memory/throughput) measurements. (Techniques used -- RoPE, RMSNorm, SwiGLU,
# KV-cache -- are standard and openly used across permissively-licensed LLMs.)
#
# License: MIT.
#
# Usage
#   SMOKE=1 python3 attention_baseline.py    # ~1 min, tiny + synthetic data, CPU-ok (correctness check)
#   python3 attention_baseline.py            # full: trains on WikiText-103
#                                            #   (pip install datasets tiktoken; GPU recommended)
# Env knobs: DModel NLayers NHeads Vocab LR BS Curriculum EvalGaps IndSpan DecodeSeq Device Seed
# =============================================================================

import os, math, time
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("SMOKE", "0") == "1"
def _i(k, d): return int(os.environ.get(k, d))
def _f(k, d): return float(os.environ.get(k, d))


class Cfg:
    # architecture (matches the compared baseline)
    d_model  = _i("DModel",  64 if SMOKE else 256)
    n_layers = _i("NLayers",  4 if SMOKE else 8)
    n_heads  = _i("NHeads",   2 if SMOKE else 4)
    vocab    = 512            # smoke default; full run overwrites with the gpt2 tokenizer size
    dropout  = _f("Dropout", 0.0)
    # optimisation
    lr       = _f("LR", 3e-4)
    bs       = _i("BS", 4 if SMOKE else 8)
    seed     = _i("Seed", 0)
    # Optional length curriculum: train on shorter sequences first, then longer.
    # This is a STANDARD, published training technique -- not specific to this work
    # (curriculum learning: Bengio et al., ICML 2009; sequence-length curricula are
    # common in long-context training, e.g. Shortformer, Press et al., 2021).
    # The schedule below is an illustrative default ("seq:steps" stages); the exact
    # step counts are not significant -- adjust freely, or set a single stage.
    _cur = os.environ.get("Curriculum", "32:4,64:4" if SMOKE else "512:1000,1024:1000,2048:1000")
    curriculum = [(int(a), int(b)) for a, b in (s.split(":") for s in _cur.split(","))]
    max_seq  = max(s for s, _ in curriculum)
    # runtime
    device   = os.environ.get("Device", "cuda" if torch.cuda.is_available() else "cpu")
    amp      = (device == "cuda")     # bf16 autocast on GPU (matches the reference)


C = Cfg()
torch.manual_seed(C.seed)
DEV = torch.device(C.device)
def amp_ctx():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if C.amp else nullcontext()


# ------------------------------- model ---------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def build_rope(seqlen, head_dim, device, base=10000.0):
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    ang = torch.outer(torch.arange(seqlen, device=device).float(), inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1)
    sin = torch.cat([ang.sin(), ang.sin()], -1)
    return cos, sin


def apply_rope(x, cos, sin):            # x: (B,H,T,D) ; cos/sin: (T,D)
    D = x.shape[-1]; half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class Block(nn.Module):
    """Standard transformer block: RMSNorm -> RoPE causal attention (SDPA) -> RMSNorm -> SwiGLU."""
    def __init__(self, c):
        super().__init__()
        d = c.d_model; self.nh = c.n_heads; self.hd = d // c.n_heads
        self.n1 = RMSNorm(d); self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.n2 = RMSNorm(d)
        hidden = int(((8 / 3) * d + 63) // 64 * 64)                       # SwiGLU sizing (matches reference)
        self.w1 = nn.Linear(d, hidden, bias=False); self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.pdrop = c.dropout

    def _qkv(self, xn):
        b, n, d = xn.shape
        q, k, v = self.qkv(xn).chunk(3, -1)
        sh = lambda t: t.view(b, n, self.nh, self.hd).transpose(1, 2)
        return sh(q), sh(k), sh(v)

    def _swiglu(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def forward(self, x, cos, sin, past=None, pos0=0):
        b, n, d = x.shape
        q, k, v = self._qkv(self.n1(x))
        cq, sq = cos[pos0:pos0 + n], sin[pos0:pos0 + n]
        q = apply_rope(q, cq, sq); k = apply_rope(k, cq, sq)
        new_past = None
        if past is not None:
            pk, pv = past
            if pk is not None: k = torch.cat([pk, k], 2); v = torch.cat([pv, v], 2)
            new_past = (k, v)
        causal = (q.shape[2] == k.shape[2])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                           dropout_p=self.pdrop if self.training else 0.0)
        y = y.transpose(1, 2).reshape(b, n, d)
        x = x + self.o(y)
        return x + self._swiglu(self.n2(x)), new_past


class TransformerLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)     # UNTIED (matches reference)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("o.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, past=None, pos0=0):
        x = self.emb(idx)
        cos, sin = build_rope(self.c.max_seq if past is None else pos0 + idx.shape[1] + 1,
                              self.c.d_model // self.c.n_heads, x.device)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        np_ = [] if past is not None else None
        for i, blk in enumerate(self.blocks):
            x, nkv = blk(x, cos, sin, (past[i] if past is not None else None), pos0)
            if np_ is not None: np_.append(nkv)
        return self.head(self.norm(x)), np_

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------- data ----------------------------------------
def synth_pool(n, vocab, device):
    return torch.randint(0, vocab, (n,), device=device)

class RealData:
    def __init__(self):
        import tiktoken
        from datasets import load_dataset
        self.enc = tiktoken.get_encoding("gpt2")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        ids, budget = [], 40_000_000
        for row in ds:
            if row["text"]: ids.extend(self.enc.encode_ordinary(row["text"]))
            if len(ids) >= budget: break
        self.data = torch.tensor(ids[:budget], dtype=torch.long)
        self.vocab = self.enc.n_vocab
    def pool(self, device): return self.data.to(device)


def make_batch(pool, bs, seq, device):
    ix = torch.randint(0, pool.numel() - seq - 1, (bs,), device=device)
    x = torch.stack([pool[i:i + seq] for i in ix])
    y = torch.stack([pool[i + 1:i + 1 + seq] for i in ix])
    return x, y


# ------------------------------- eval ----------------------------------------
@torch.no_grad()
def eval_ppl(model, pool, seq, iters=20):
    model.eval(); tot = 0.0
    for _ in range(iters):
        x, y = make_batch(pool, C.bs, seq, DEV)
        with amp_ctx():
            logits, _ = model(x)
        tot += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    return math.exp(tot / iters)


@torch.no_grad()
def eval_induction(model, pool, gaps, span, batches, bs):
    """Real-token induction: seq = [A, filler(gap), A] with A a `span`-token block;
    predict the SECOND A block. Accuracy over the second-A positions (matches the
    CM-Galaxy comparison harness)."""
    model.eval(); n = pool.numel(); out = {}
    for g in gaps:
        if span + g + span > model.c.max_seq:
            out[g] = float("nan"); continue
        c = t = 0
        for _ in range(batches):
            seqs = []
            for _b in range(bs):
                ao = torch.randint(0, n - span, (1,)).item(); A = pool[ao:ao + span]
                if g > 0:
                    fo = torch.randint(0, n - g, (1,)).item(); seqs.append(torch.cat([A, pool[fo:fo + g], A]))
                else:
                    seqs.append(torch.cat([A, A]))
            x = torch.stack(seqs).to(DEV)
            with amp_ctx():
                pred = model(x[:, :-1])[0].argmax(-1)
            tgt = x[:, 1:]; lo, hi = span + g - 1, span + g + span - 2
            c += (pred[:, lo:hi + 1] == tgt[:, lo:hi + 1]).sum().item(); t += bs * (hi - lo + 1)
        out[g] = round(100 * c / max(t, 1), 1)
    return out


@torch.no_grad()
def bench_decode(model, seq_len, warmup=2, steps=20):
    model.eval(); is_cuda = (DEV.type == "cuda")
    if is_cuda: torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    plen = min(seq_len, model.c.max_seq - steps - 2)
    prompt = torch.randint(0, model.c.vocab, (1, plen), device=DEV)
    past = [None] * model.c.n_layers
    with amp_ctx():
        logits, past = model(prompt, past=past, pos0=0)
    pos = plen; cur = logits[:, -1:, :].argmax(-1)
    for _ in range(warmup):
        with amp_ctx(): logits, past = model(cur, past=past, pos0=pos)
        pos += 1; cur = logits[:, -1:, :].argmax(-1)
    if is_cuda: torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        with amp_ctx(): logits, past = model(cur, past=past, pos0=pos)
        pos += 1; cur = logits[:, -1:, :].argmax(-1)
    if is_cuda: torch.cuda.synchronize()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6) if is_cuda else float("nan")
    return (steps / dt if dt > 0 else float("inf")), peak


def causality_check(model):
    model.eval(); T = min(48, model.c.max_seq)
    x = torch.randint(0, model.c.vocab, (2, T), device=DEV)
    with torch.no_grad(), amp_ctx():
        l1, _ = model(x); x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 7) % model.c.vocab; l2, _ = model(x2)
    diff = (l1[:, :-1].float() - l2[:, :-1].float()).abs().max().item()
    print(f"[causality] max |Δlogits| at earlier positions = {diff:.2e} -> {'OK (causal)' if diff < 1e-2 else 'FAIL'}")
    assert diff < 1e-2, "causality check failed"


# ------------------------------- main ----------------------------------------
def main():
    print(f"== attention_baseline ==  SMOKE={SMOKE}  device={C.device}  amp={C.amp}")
    if SMOKE:
        C.vocab = 512
        pool = synth_pool(200_000, C.vocab, DEV)
    else:
        try:
            rd = RealData(); C.vocab = rd.vocab; pool = rd.pool(DEV)
        except Exception as e:
            raise SystemExit(f"Full run needs `datasets` + `tiktoken`: pip install datasets tiktoken\n({e})")

    model = TransformerLM(C).to(DEV)
    print(f"params: {model.num_params()/1e6:.2f}M   "
          f"(d_model={C.d_model}, layers={C.n_layers}, heads={C.n_heads}, "
          f"swiglu={int(((8/3)*C.d_model+63)//64*64)}, vocab={C.vocab}, untied)")
    causality_check(model)

    opt = torch.optim.AdamW(model.parameters(), lr=C.lr)
    model.train(); t0 = time.time(); step = 0
    for seq, steps in C.curriculum:                         # length curriculum: short -> long
        print(f"  -- stage: seq {seq} for {steps} steps --")
        for _ in range(steps):
            step += 1
            x, y = make_batch(pool, C.bs, seq, DEV)
            with amp_ctx():
                logits, _ = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % max(1, steps // 3) == 0:
                print(f"     step {step}  seq{seq}  loss {loss.item():.3f}  ({time.time()-t0:.1f}s)")

    ev_seq = C.curriculum[-1][0]
    print(f"[ppl @seq{ev_seq}]  {eval_ppl(model, pool, ev_seq, iters=10 if SMOKE else 30):.2f}")

    gaps = [int(g) for g in os.environ.get("EvalGaps", "0,8" if SMOKE else "0,512,1024,2048,4096,8192").split(",")]
    span = _i("IndSpan", 8 if SMOKE else 64)
    rec = eval_induction(model, pool, gaps, span, batches=2 if SMOKE else 40, bs=4 if SMOKE else 16)
    print("[induction recall %]  " + "  ".join(f"gap{g}={rec[g]}" for g in gaps))

    dseq = _i("DecodeSeq", 64 if SMOKE else 2048)
    tps, mb = bench_decode(model, dseq)
    print(f"[decode @seq{dseq}]  {tps:.1f} tok/s   peak {f'{mb:.0f} MB' if mb == mb else 'n/a (CPU)'}")
    print("done.")


if __name__ == "__main__":
    main()
