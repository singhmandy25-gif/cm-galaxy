CM-Galaxy
A sub-quadratic, attention-free language-model backbone for cheap long-context modeling.
Independent research by Mandeep Singh (ORCID 0009-0003-7176-2395) · speedgaplabs@gmail.com
Teaser (v0) — concept, measured results, and honest limits. The architecture recipe, training details, source code, and checkpoints are not released at this stage.
What it is
CM-Galaxy (the CM denotes the author's Continuous Medium framework[^1]) is a small (~25M-parameter) language-model backbone built to make long-context modeling cheap — in both training and inference. It is attention-free and sub-quadratic (O(n log n)): it combines a sub-quadratic token mixer (long-convolution / FFT family) with a gated delta-rule memory block (in the GatedDeltaNet family). The central design choice is a constant-size decode state — there is no key/value cache that grows with sequence length.
Rather than comparing each new token against every past token — attention's token-by-token lookup — it folds the sequence into an associative memory and a convolutional mixer, carrying forward a running, compressed representation of the whole context. With no growing pile of past tokens to look up, the cost of each new token stays flat.
It is not a general-purpose chat assistant and is not positioned against frontier LLMs. It is an efficiency-focused backbone.
The problem it targets — and why patches don't solve it
In a standard transformer, long-context cost is dominated by attention: the key/value cache grows linearly with context length, and attention compute grows quadratically — in training and at inference. The model must look at every previous token, for every new token.
Most current responses work around this rather than fixing it:
RAG bolts an external retriever onto a frozen attention model. Useful, but it doesn't change attention's per-token cost — it just feeds it fewer tokens.
Mixture-of-experts (MoE) sparsifies the feed-forward layers. It cuts compute per token, but the attention KV-cache still grows with context. (MoE is in fact orthogonal to CM-Galaxy: MoE sparsifies the feed-forward, CM-Galaxy replaces the token mixer — so the two could be combined.)
These are patches on top of attention. CM-Galaxy instead addresses the cost at the architecture level: where full token-to-token attention isn't needed, it uses a constant-state memory instead, so the cost of each additional token of context stays flat.
What has been numerically demonstrated
All results below are from a single ~25M-parameter model (trained on WikiText-103) and a standard attention baseline, measured directly. These are small-scale figures — the contribution is the architecture and its efficiency, not absolute quality — but they are real, and mutually consistent.
It learns language and recalls well (it works):
Clean associative recall ≈ attention. On MQAR (multi-query associative recall) in the independent Zoology harness, the memory block matches attention across sequence lengths 64–512: ~99.5–99.9% (versus attention's ~99.9–100%), in a fair same-scaffold comparison, robust over 3 seeds, learned within a few epochs. At the hardest length (512) it is ~99.5% with slightly more run-to-run variance — very close, not identical.
Long-gap recall holds up better than attention. On a real-token induction task (recall a token after a long filler gap; same harness, both models trained to ≤2048 context and evaluated with memory-efficient (FlashAttention) attention, so the long gaps are an extrapolation), CM-Galaxy's recall stays roughly flat as the gap grows, while the attention baseline's declines steadily:
Gap (tokens)
CM-Galaxy
Attention
0
38.6
33.6
512
29.6
26.6
2048
25.9
25.4
4096
23.6
21.1
8192
24.0
18.4
16384
22.7
16.4
From about 4k onward CM-Galaxy's recall is the higher of the two, and the margin widens with distance — at a 16k gap, 22.7% vs 16.4%. (Caveats: absolute numbers are modest at this scale; the attention baseline has ~30% more parameters, ~32M vs ~25M; perplexity stays stable across gaps, and a separate run training CM-Galaxy directly at 8k reproduces the flat recall — so this is genuine capacity, not extrapolation drift.)
Healthy language modeling at 8k. Trained directly at 8k context (via a length curriculum, not merely run there), the 25M model reaches ~37.5 PPL with ~18.1% LAMBADA (last-word accuracy) — genuine next-word competence at this small scale, well above chance.
It stays cheap as context and batch grow (it survives):
Flat decode memory. Peak decode memory stays flat at ~312 MB as the sequence grows from 8k to 8.4M tokens. An attention KV-cache grows linearly into the tens of GB over the same range — about 30× less at 1M tokens and ~235× less at 8.4M. (Below ~30k tokens the attention cache is smaller — CM-Galaxy carries a fixed overhead; beyond that crossover, its flat memory wins by a widening margin.)
Throughput at the memory limit. At high batch sizes, CM-Galaxy sustains roughly 92× (at 8k) and 159× (at 16k) more tokens/sec/GPU than the attention baseline. The baseline runs out of memory at batch ~512–1024; CM-Galaxy reaches batch 8192 with flat memory.
The picture these form. The recall, PPL and LAMBADA numbers show the model genuinely works — it learns language, matches attention on clean recall, and holds up better than attention on long-gap recall. The memory and throughput numbers show it survives at scale — its cost stays flat exactly where attention's cost explodes. Together: a working sub-quadratic backbone whose efficiency advantage is real and grows with context and batch.
How it was measured. The speed and memory figures were collected on a single NVIDIA A100-80GB, with CM-Galaxy and the baseline run together in the same session — so the ratios are a fair same-hardware comparison (absolute tokens/sec are hardware-specific and are not the claim). The baseline is a standard transformer using SDPA / FlashAttention, not a weakened one. These speed and memory measurements use randomly-initialized weights (they characterize cost only, not output quality). The out-of-memory point is on the 80 GB card; smaller GPUs reach it sooner.
Honest limits
Stated as plainly as the results, because they bound what the results mean.
Absolute long-range recall is modest — the cost of a fixed-size state. Although CM-Galaxy's long-gap recall holds up better than the attention baseline (above), in absolute terms it is only ~24–27% at the longest 8k gaps. It degrades gracefully with distance rather than collapsing (the dominant failure mode is interference between competing items, not a failure to store them), but ~24–27% is far from the near-perfect recall one would ultimately want. Raising that absolute level is the central open problem.
Recall is verified to 8k (evaluated to 16k). 32k / 128k / 1M are untested. Memory stays flat out to 8.4M tokens, but whether useful recall survives at very long context is still open.
The memory and throughput measurements used random weights. They characterize speed and memory only — not output quality.
Single-stream latency favors attention. Per token, at batch 1, the attention baseline is about 1.4–1.5× faster. CM-Galaxy's advantage is total throughput at the memory limit (a batched-serving win), not single-stream latency.
Scale and SSM comparison are untested. Larger models (70M / 125M and beyond) have not been trained, and no controlled comparison has been run against Mamba, RWKV, or other state-space / linear-attention architectures — so CM-Galaxy should not be read as outperforming them; it is characterized on its own terms.
No Indic-language modeling has been done — Indic is an intended future application, not a current result (see below).
Where it goes next
The limits above are the roadmap, not a dead end. Each step needs more research, and likely collaboration and compute:
Raising absolute long-range recall — and why the flat memory matters here. The central open problem is lifting recall in absolute terms and extending it past 8k. The flat-memory result is the key enabler: because the state is so cheap (~312 MB where attention needs tens of GB), there is large headroom to spend. The memory can be allowed to grow in a controlled, paged way — holding more context while staying far below attention's cost. This direction is being actively explored elsewhere too (e.g. "Memory Caching," arXiv:2602.24281); a genuine contribution here would have to go beyond simply growing the state — being selective about what is retained, or keeping decode cost from growing in step with it. Whether a paged memory actually maintains recall at long context is the central hypothesis still to be tested.
Indic long-context modeling. One notably underserved application — to be pursued once the architecture is matured and scaled. The natural first step is tokenization suited to Indic morphology. But the efficiency is general: cheap long-context modeling helps wherever contexts are long — documents, retrieved passages, code, dialogue — so Indic is one opportunity among several, not the only target.
Scale. Validating the architecture at 70M / 125M and beyond, and benchmarking directly against other sub-quadratic families (Mamba, RWKV, gated delta-rule models).
Agentic / parallel-rollout workloads (speculative). The large batched-throughput advantage suits settings that run many parallel, long-context rollouts at once — agentic loops and RL training among them. This is untested, and such workloads also lean on the long-context recall that remains the open problem, so it is a direction to explore, not a claim.
None of these are solved yet — they are open research problems, set out plainly so that what is demonstrated today and what remains to be shown are never confused. The honest claim is narrow and real: a working, well-characterized sub-quadratic backbone with a measured efficiency advantage, and a clear, evidence-driven path for the parts that remain open.
Positioning
The building blocks come from known families — long-convolution / FFT-style mixers, and gated delta-rule (GatedDeltaNet-style) memory; the constant-size state is shared with state-space / linear-recurrent models (Mamba, RWKV). The contribution here is the specific attention-free combination and its rigorous, honest characterization at small scale: a clean, well-measured sub-quadratic data point, with the constant-state-decode property worked through end to end — including where it still falls short.
It is deliberately not claimed to be a "transformer killer," and not claimed to be (or to beat) a state-space model (no SSM comparison has been done). It targets a niche that is currently largely empty: an indigenous, sub-quadratic, attention-free backbone for long-context modeling, at a time when prominent Indic models are built largely on transformer + mixture-of-experts.
Status & disclosure
This is a teaser. It shares the concept, the measured results, and the limits — what it does, not how it is built. The architecture recipe, training curriculum, model source code, and checkpoints are not part of this release. A standard-transformer baseline used for the fair comparison can be shared separately for independent verification.
Paper (DOI): to be added — Zenodo
Citation: to be added with the DOI
Collaboration
This work was designed, built, and validated solo and independently, on rented cloud GPUs. Research collaboration is welcome (non-commercial at this stage). If you work on efficient long-context modeling, Indic-language modeling, or sub-quadratic architectures, and would like to discuss this or help validate it further, please get in touch.
Contact: Mandeep Singh — speedgaplabs@gmail.com
© 2026 Mandeep Singh. Licensed under CC BY 4.0 — you may share and adapt this document with attribution. This license covers this document only; it does not grant rights to any unreleased source code, model weights, or architecture details.
[^1]: Continuous Medium (CM) is a physics framework previously developed by the author (26+ papers on Zenodo; ORCID 0009-0003-7176-2395), and several of CM-Galaxy's design choices were initially drawn from it. The physics motivated the search space, not the claims: each component is kept only where controlled ablation earns it, and several physics-motivated modules were tested and dropped. All results in this document rest on standard benchmarks and ablations.
