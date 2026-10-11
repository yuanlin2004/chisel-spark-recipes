<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/chisel-dark.svg">
    <img src="logo/chisel-light.svg" alt="Chisel logo" width="360">
  </picture>
</p>

# chisel-spark-recipes

Tuned LLM serving recipes for one NVIDIA DGX Spark (GB10). Each one is published with its measured single-user decode speed and its task scores against the model's own BF16 weights.

## Results

| Model | Recipe | Decode, one user | Time to first token | Quality vs BF16 | Runtime |
|---|---|---|---|---|---|
| Qwen3.8-27B | [fp8-mtp4-draftvocab](recipes/qwen3.8-27b/fp8-mtp4-draftvocab.yaml) | **23.1 tok/s** (43.5 ms/token median, 57.8 ms p90) | 316 ms median | 98.7–102.5% on 4 tasks | vLLM 0.31.0 + chisel-draftvocab |

For comparison, on the same box with the same benchmark, stock vLLM serves Qwen3.8-27B-FP8 at 8.0 tok/s without speculation and 16.7 tok/s with its built-in MTP head (3 draft tokens).

Every row links to its recipe. Every result has a folder under [`results/`](results/) with the raw benchmark and quality files, the versions used, and how to reproduce them. A published recipe and its results never change; a better result is a new recipe and a new row.

## Run it

With [sparkrun](https://github.com/spark-arena/sparkrun):

```
sparkrun registry add https://github.com/yuanlin2004/chisel-spark-recipes
sparkrun run @chisel/qwen3.8-27b/fp8-mtp4-draftvocab
```

With plain docker:

```
bash recipes/qwen3.8-27b/fp8-mtp4-draftvocab.docker.sh
```

Either way the server is OpenAI-compatible on port 8000. The recipe is stock vLLM 0.31.0 (the official `vllm/vllm-openai:v0.31.0` image) with one small plugin, [chisel-draftvocab](plugin/), installed at start.

## Qwen3.8-27B: fp8-mtp4-draftvocab

**What it is**
- Weights: `Qwen/Qwen3.8-27B-FP8` (Qwen's official FP8 checkpoint), FP8 KV cache, FlashInfer attention, prefix caching.
- Speculative decoding with the model's built-in MTP head, 4 draft tokens per step.
- **Reduced-vocabulary draft head** (chisel-draftvocab; [how it works](docs/reduced-vocabulary-draft-head.md)). The MTP head proposes draft tokens through the model's full 248,320 × 5,120 BF16 output layer, 2.5 GB read once per draft token. The plugin gives the draft steps a 0.5 GB head over the 50,858 tokens that cover about 98% of generated text, mapped back to the full vocabulary. The verify step still uses the full output layer, so the served tokens are exactly those the model would produce at that draft length; only the cost of drafting drops. Same-session measurement with 3 draft tokens: 59.9 → 49.0 ms per token, outputs identical on 100 of 100 prompts.

**Speed** (2026-10-10, DGX Spark, driver 580.178.04)
- Decode: median 43.5 ms per output token (23.1 tok/s), p90 57.8 ms, over 25 prompts (chat, code, JSON, long story), 256 output tokens, three repeats, one user. By prompt type: JSON 32.3 ms, code 43.5, long story 43.4, chat 49.8.
- Time to first token: median 316 ms on the same prompts.
- GPU clock makes no difference: 43.5 ms with the clock capped at 2,000 MHz, 43.7 ms uncapped (2,333 to 2,411 MHz, up to 90 W and 79 °C). Decode here is limited by memory bandwidth, so a cap that keeps the box cooler costs nothing.
- Details: [`results/qwen3.8-27b/fp8-mtp4-draftvocab/2026-10-10/`](results/qwen3.8-27b/fp8-mtp4-draftvocab/2026-10-10/).

**Quality against BF16** (Qwen/Qwen3.8-27B, BF16 weights and KV cache)

| Task | BF16 | This recipe | Recovery | Difference [95% CI] |
|---|---|---|---|---|
| gsm8k (5-shot) | 90.07 | 90.45 | 100.4% | +0.38 [−0.57, +1.33] |
| ifeval (prompt-level strict) | 80.96 | 82.99 | 102.5% | +2.03 [−0.34, +4.40] |
| humaneval (pass@1, chat) | 95.12 | 93.90 | 98.7% | −1.22 [−3.61, +1.17] |
| LongBench NarrativeQA (F1, full stories up to 68k tokens) | 34.83 | 35.35 | 101.5% | +0.52 [−1.98, +3.02] |

The bar: every task keeps at least 97% of the BF16 score, no task is clearly worse (95% interval of the paired difference entirely below zero), and mean answer length stays within 90–110% of BF16's. All four tasks pass. An FP8 model does not reproduce BF16's text word for word under greedy decoding (neither does BF16 across batch shapes), so the comparison is on task scores, the convention of MLPerf and of published quantized checkpoints. Scored on the recipe's own container with lm-evaluation-harness through the chat template, thinking off, greedy, full tasks; method and per-question answers: [`quality/`](results/qwen3.8-27b/fp8-mtp4-draftvocab/2026-10-10/quality/).

## Repository layout

```
recipes/<model>/<recipe>.yaml        sparkrun recipe (and <recipe>.docker.sh, the same without sparkrun)
plugin/                              chisel-draftvocab, a vLLM general plugin; passes are off unless a recipe turns them on
results/<model>/<recipe>/<date>/     benchmark/, quality/, outputs/, ENVIRONMENT.md, METHOD.md
results/<model>/bf16-reference/      BF16 task scores every recipe for that model is compared against
prompts/                             the prompt sets the benchmarks use, with their sources
tools/                               the benchmark, output-check and quality-check scripts
```

## License

Apache-2.0. Model weights are under their own licences (Qwen3.8-27B: Apache-2.0).
