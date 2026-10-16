# Qwen3.8-27B: chisel-nvfp4-dflash2-replay

**What it is**
- [chisel-nvfp4-dflash2](chisel-nvfp4-dflash2.md) (unsloth's NVFP4 checkpoint, DFlash2 with 10 draft tokens, Chisel's NVFP4 drafter, linear-attention state in BF16) plus two lossless passes from Chisel's vLLM plugin, [chisel-passes](mods/chisel-passes/), installed at start:
- **GDN state replay.** During verification of the draft tokens, Qwen3.8's linear-attention (GDN) layers normally store the full recurrent state after every draft token. The plugin saves only the small per-token inputs, and once the number of accepted tokens is known it replays them and stores the accepted state once. The stored state is bit-identical to vLLM's own.
- **Reduced draft vocabulary.** The drafter's output head covers the 100,032 most-used tokens instead of all 248,320, mapped back to the full vocabulary, and is kept in FP8. The verify step keeps the full output layer, so the served tokens are the target model's own.

**Speed** (2026-10-15, DGX Spark, driver 580.178.04, GPU clock capped at 2,000 MHz)
- Decode: median 15.35 ms per output token in two fresh-start sessions (65.1 tok/s), p90 28.3 and 29.1 ms, over the same 25 prompts as the other recipes, 256 output tokens, three repeats each, one user. By prompt type: JSON 13.7 to 14.1 ms, code 14.2 to 14.3, chat 26.4 to 27.2, long story 28.3 to 28.8. 4.32 draft tokens accepted per step.
- Time to first token: median 241 ms on the same prompts.
- llama-benchy 0.4.0 with spark-arena's settings, run locally (not arena scores): 58.0 tok/s total with 2 users at 16k context (10 runs over two sessions; single runs 52 to 69), 30.0 with 1 user at 16k. Prefill about 1,150 tok/s at 16k context.
- Through this recipe and sparkrun: 15.38 ms median on the same benchmark (one repeat).
- Details: [`results/qwen3.8-27b/chisel-nvfp4-dflash2-replay/2026-10-15/`](../../results/qwen3.8-27b/chisel-nvfp4-dflash2-replay/2026-10-15/).

**Quality against BF16** (Qwen/Qwen3.8-27B, BF16 weights and KV cache)

| Task | BF16 | This recipe | Recovery | Difference [95% CI] | Answer length vs BF16 |
|---|---|---|---|---|---|
| gsm8k (5-shot) | 90.07 | 89.46 | 99.3% | −0.61 [−1.81, +0.60] | 1.009 |
| ifeval (prompt-level strict) | 80.96 | 81.89 | 101.1% | +0.92 [−1.56, +3.41] | 1.001 |
| humaneval (pass@1, chat) | 95.12 | 94.51 | 99.4% | −0.61 [−2.68, +1.46] | 1.007 |
| LongBench NarrativeQA (F1, full stories) | 34.83 | 35.24 | 101.2% | +0.40 [−1.64, +2.45] | 0.962 |

All four tasks pass the bar in the [model page](README.md#quality-bar). Scored with thinking off; method and per-question answers: [`quality/`](../../results/qwen3.8-27b/chisel-nvfp4-dflash2-replay/2026-10-15/quality/).

**Run it**

```
sparkrun registry add https://github.com/yuanlin2004/chisel-spark-recipes
sparkrun run @chisel/chisel-nvfp4-dflash2-replay --tp 1 --hosts localhost
```

The recipe ([chisel-nvfp4-dflash2-replay.yaml](chisel-nvfp4-dflash2-replay.yaml)) runs the pinned spark-arena vLLM nightly image and installs the chisel-passes plugin from [mods/chisel-passes/](mods/chisel-passes/) at start. The server is OpenAI-compatible on port 8000.

**On spark-arena**: submission pending.

[All Qwen3.8-27B recipes](README.md) · [All recipes](../../README.md)
