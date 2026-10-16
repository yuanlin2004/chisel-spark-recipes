# Qwen3.8-27B: chisel-nvfp4-dflash2

**What it is**
- Weights: `unsloth/Qwen3.8-27B-NVFP4` (4-bit weights on the GB10's FP4 tensor cores). Context up to 262,144 tokens, thinking on with medium effort, up to 4 concurrent requests.
- Speculative decoding with DFlash2, 10 draft tokens per step, using **Chisel's NVFP4 drafter** ([yuanlin2004/Qwen3.8-27B-Chisel-DFlash2-NVFP4-drafter](https://huggingface.co/yuanlin2004/Qwen3.8-27B-Chisel-DFlash2-NVFP4-drafter)): z-lab's DFlash2 drafter with its linear layers quantized to NVFP4, 1.4 GB instead of 3.8 GB.
- The linear-attention recurrent state is cached in BF16 instead of FP32 (`--mamba-ssm-cache-dtype bfloat16`), which halves the state written per draft position.
- Builds on Luis Poveda's single-Spark recipe on spark-arena (unsloth NVFP4 + DFlash2, 8 drafts). On our box, under the same cap, that recipe measured 22.8 ms per token on our benchmark; this one 16.5 ms.

**Speed** (2026-10-14, DGX Spark, driver 580.178.04, GPU clock capped at 2,000 MHz)
- Decode: median 16.44 and 16.54 ms per output token in two fresh-start sessions (about 60.5 tok/s), p90 30.9 to 31.1 ms, over the same 25 prompts as fp8-mtp4-draftvocab, 256 output tokens, three repeats each, one user. By prompt type: JSON 14.2 ms, code 15.5, chat 28.9, long story 30.5. 4.34 draft tokens accepted per step.
- Time to first token: median 250 ms on the same prompts.
- llama-benchy 0.4.0 with spark-arena's settings, run locally (not arena scores): 51.6 tok/s total with 2 users at 16k context (10 runs over two sessions; single runs 42 to 61), 30.7 with 1 user at 16k, 38.5 with 1 user and no context (20 runs). Prefill: 2,106 tok/s on a 2,048-token prompt, about 1,120 at 16k context. Single runs vary a lot because each run continues a different random slice of a novel, and draft acceptance follows the text.
- Details: [`results/qwen3.8-27b/chisel-nvfp4-dflash2/2026-10-14/`](../../results/qwen3.8-27b/chisel-nvfp4-dflash2/2026-10-14/).

**Quality against BF16** (Qwen/Qwen3.8-27B, BF16 weights and KV cache)

| Task | BF16 | This recipe | Recovery | Difference [95% CI] | Answer length vs BF16 |
|---|---|---|---|---|---|
| gsm8k (5-shot) | 90.07 | 89.46 | 99.3% | −0.61 [−1.90, +0.69] | 1.002 |
| ifeval (prompt-level strict) | 80.96 | 84.10 | 103.9% | +3.14 [+0.57, +5.72] | 1.024 |
| humaneval (pass@1, chat) | 95.12 | 94.51 | 99.4% | −0.61 [−3.78, +2.56] | 1.018 |
| LongBench NarrativeQA (F1, full stories) | 34.83 | 34.92 | 100.3% | +0.09 [−1.92, +2.10] | 0.938 |

All four tasks pass the bar in the [model page](README.md#quality-bar). Scored with thinking off; method and per-question answers: [`quality/`](../../results/qwen3.8-27b/chisel-nvfp4-dflash2/2026-10-14/quality/).

**Run it**

```
sparkrun registry add https://github.com/yuanlin2004/chisel-spark-recipes
sparkrun run @chisel/chisel-nvfp4-dflash2 --tp 1 --hosts localhost
```

The recipe ([chisel-nvfp4-dflash2.yaml](chisel-nvfp4-dflash2.yaml)) runs the spark-arena vLLM nightly image as is, pinned by digest. The server is OpenAI-compatible on port 8000.

**On spark-arena**: [sub1792022796220](https://spark-arena.com/benchmark/sub1792022796220), 51.6 tok/s on the leaderboard's ranking test (128-token generation at 16k context, 2 users); 1 user: 39.1 tok/s and 2,132 tok/s prefill with no context, 24.2 tok/s at 100k context. Awaiting the maintainers' approval.

[All Qwen3.8-27B recipes](README.md) · [All recipes](../../README.md)
