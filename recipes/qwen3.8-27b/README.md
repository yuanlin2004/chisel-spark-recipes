# Qwen3.8-27B on one DGX Spark

| Recipe | Decode, one user | Time to first token | Quality vs BF16 | Runtime |
|---|---|---|---|---|
| [chisel-nvfp4-dflash2-replay](chisel-nvfp4-dflash2-replay.md) | **65.1 tok/s** (15.35 ms/token median, 28.3–29.1 ms p90) | 241 ms median | 99.3–101.2% on 4 tasks | vLLM nightly (0.31.1rc1.dev380) + chisel-passes |
| [chisel-nvfp4-dflash2](chisel-nvfp4-dflash2.md) | **60.5 tok/s** (16.5 ms/token median, 31.0 ms p90) | 250 ms median | 99.3–103.9% on 4 tasks | vLLM nightly (0.31.1rc1.dev380), no plugin |
| [fp8-mtp4-draftvocab](fp8-mtp4-draftvocab.md) | **23.1 tok/s** (43.5 ms/token median, 57.8 ms p90) | 316 ms median | 98.7–102.5% on 4 tasks | vLLM 0.31.0 + chisel-draftvocab |

For comparison, on the same box with the same benchmark, stock vLLM serves Qwen3.8-27B-FP8 at 8.0 tok/s without speculation and 16.7 tok/s with its built-in MTP head (3 draft tokens).

Each recipe page has what it is, its speed and quality against BF16, its spark-arena entries, and the command to run it. Raw files: [`results/qwen3.8-27b/`](../../results/qwen3.8-27b/). BF16 reference scores: [`results/qwen3.8-27b/bf16-reference/`](../../results/qwen3.8-27b/bf16-reference/).

## Quality bar

The bar: every task keeps at least 97% of the BF16 score, no task is clearly worse (95% interval of the paired difference entirely below zero), and mean answer length stays within 90–110% of BF16's. Scored with lm-evaluation-harness through the chat template, thinking off, greedy, full tasks, on gsm8k, ifeval, humaneval and LongBench NarrativeQA.

Discussion: [NVIDIA Developer Forums thread](https://forums.developer.nvidia.com/t/qwen3-8-27b-fp8-on-one-spark-23-1-tok-s-single-user-quality-measured-against-bf16-plus-a-question-about-dflash2-on-long-contexts/385526).

[All recipes](../../README.md)
