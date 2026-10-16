# Qwen3.8-27B: fp8-mtp4-draftvocab

**What it is**
- Weights: `Qwen/Qwen3.8-27B-FP8` (Qwen's official FP8 checkpoint), FP8 KV cache, FlashInfer attention, prefix caching. Context up to 131,072 tokens (the results below were measured with an 81,920-token limit; the limit was raised later so spark-arena's 100k-token tests fit).
- Speculative decoding with the model's built-in MTP head, 4 draft tokens per step.
- **Reduced-vocabulary draft head** (chisel-draftvocab; [how it works](../../docs/reduced-vocabulary-draft-head.md)). The MTP head proposes draft tokens through the model's full 248,320 × 5,120 BF16 output layer, 2.5 GB read once per draft token. The plugin gives the draft steps a 0.5 GB head over the 50,858 tokens that cover about 98% of generated text, mapped back to the full vocabulary. The verify step still uses the full output layer, so the served tokens are exactly those the model would produce at that draft length; only the cost of drafting drops. Same-session measurement with 3 draft tokens: 59.9 → 49.0 ms per token, outputs identical on 100 of 100 prompts.

**Speed** (2026-10-10, DGX Spark, driver 580.178.04)
- Decode: median 43.5 ms per output token (23.1 tok/s), p90 57.8 ms, over 25 prompts (chat, code, JSON, long story), 256 output tokens, three repeats, one user. By prompt type: JSON 32.3 ms, code 43.5, long story 43.4, chat 49.8.
- Time to first token: median 316 ms on the same prompts.
- GPU clock makes no difference: 43.5 ms with the clock capped at 2,000 MHz, 43.7 ms uncapped (2,333 to 2,411 MHz, up to 90 W and 79 °C). Decode here is limited by memory bandwidth, so a cap that keeps the box cooler costs nothing.
- Details: [`results/qwen3.8-27b/fp8-mtp4-draftvocab/2026-10-10/`](../../results/qwen3.8-27b/fp8-mtp4-draftvocab/2026-10-10/).

**Quality against BF16** (Qwen/Qwen3.8-27B, BF16 weights and KV cache)

| Task | BF16 | This recipe | Recovery | Difference [95% CI] |
|---|---|---|---|---|
| gsm8k (5-shot) | 90.07 | 90.45 | 100.4% | +0.38 [−0.57, +1.33] |
| ifeval (prompt-level strict) | 80.96 | 82.99 | 102.5% | +2.03 [−0.34, +4.40] |
| humaneval (pass@1, chat) | 95.12 | 93.90 | 98.7% | −1.22 [−3.61, +1.17] |
| LongBench NarrativeQA (F1, full stories up to 68k tokens) | 34.83 | 35.35 | 101.5% | +0.52 [−1.98, +3.02] |

The bar: every task keeps at least 97% of the BF16 score, no task is clearly worse (95% interval of the paired difference entirely below zero), and mean answer length stays within 90–110% of BF16's. All four tasks pass. An FP8 model does not reproduce BF16's text word for word under greedy decoding (neither does BF16 across batch shapes), so the comparison is on task scores, the convention of MLPerf and of published quantized checkpoints. Scored on the recipe's own container with lm-evaluation-harness through the chat template, thinking off, greedy, full tasks; method and per-question answers: [`quality/`](../../results/qwen3.8-27b/fp8-mtp4-draftvocab/2026-10-10/quality/).

**Run it**

```
sparkrun registry add https://github.com/yuanlin2004/chisel-spark-recipes
sparkrun run @chisel/fp8-mtp4-draftvocab --tp 1 --hosts localhost
```

or without sparkrun: `bash recipes/qwen3.8-27b/fp8-mtp4-draftvocab.docker.sh`. The recipe ([fp8-mtp4-draftvocab.yaml](fp8-mtp4-draftvocab.yaml)) is stock vLLM 0.31.0 (the official `vllm/vllm-openai:v0.31.0` image) with one small plugin, [chisel-draftvocab](../../plugin/), installed at start. The server is OpenAI-compatible on port 8000.

**On spark-arena**: [sub1791769550627](https://spark-arena.com/benchmark/sub1791769550627) (28.3 tok/s on the ranking test) and [sub1791836777805](https://spark-arena.com/benchmark/sub1791836777805) (27.2 tok/s), the fastest single-Spark FP8 entries for Qwen3.8-27B. 1 user, no context: 16.2 and 16.5 tok/s.

[All Qwen3.8-27B recipes](README.md) · [All recipes](../../README.md)
