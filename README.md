<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/chisel-recipe-dark.svg">
    <img src="logo/chisel-recipe-light.svg" alt="Chisel Recipe logo" width="360">
  </picture>
</p>

# chisel-spark-recipes

Tuned LLM serving recipes for one NVIDIA DGX Spark (GB10). Each one is published with its measured single-user decode speed and its task scores against the model's own BF16 weights.

These recipes come from [Chisel](https://github.com/yuanlin2004/chisel), which automates building the inference stack for a given model on a given machine.

## Results

| Model | Recipe | Decode, one user | Time to first token | Quality vs BF16 | Runtime |
|---|---|---|---|---|---|
| Qwen3.8-27B | [chisel-nvfp4-dflash2](recipes/qwen3.8-27b/chisel-nvfp4-dflash2.yaml) | **60.5 tok/s** (16.5 ms/token median, 31.0 ms p90) | 250 ms median | 99.3–103.9% on 4 tasks | vLLM nightly (0.31.1rc1.dev380), no plugin |
| Qwen3.8-27B | [fp8-mtp4-draftvocab](recipes/qwen3.8-27b/fp8-mtp4-draftvocab.yaml) | **23.1 tok/s** (43.5 ms/token median, 57.8 ms p90) | 316 ms median | 98.7–102.5% on 4 tasks | vLLM 0.31.0 + chisel-draftvocab |

For comparison, on the same box with the same benchmark, stock vLLM serves Qwen3.8-27B-FP8 at 8.0 tok/s without speculation and 16.7 tok/s with its built-in MTP head (3 draft tokens).

Every row links to its recipe. Every result has a folder under [`results/`](results/) with the raw benchmark and quality files, the versions used, and how to reproduce them. A published recipe and its results never change; a better result is a new recipe and a new row.

On [spark-arena](https://spark-arena.com), whose leaderboard ranks by 128-token generation at 16k context with 2 users: chisel-nvfp4-dflash2 is [sub1792022796220](https://spark-arena.com/benchmark/sub1792022796220), 51.6 tok/s on that test (1 user: 39.1 tok/s and 2,132 tok/s prefill with no context; 24.2 tok/s at 100k context), awaiting the maintainers' approval. fp8-mtp4-draftvocab is listed as [sub1791769550627](https://spark-arena.com/benchmark/sub1791769550627) (28.3 tok/s) and [sub1791836777805](https://spark-arena.com/benchmark/sub1791836777805) (27.2 tok/s), the fastest single-Spark FP8 entries for Qwen3.8-27B.

Discussion: [NVIDIA Developer Forums thread](https://forums.developer.nvidia.com/t/qwen3-8-27b-fp8-on-one-spark-23-1-tok-s-single-user-quality-measured-against-bf16-plus-a-question-about-dflash2-on-long-contexts/385526).

## Run it

With [sparkrun](https://github.com/spark-arena/sparkrun):

```
sparkrun registry add https://github.com/yuanlin2004/chisel-spark-recipes
sparkrun run @chisel/chisel-nvfp4-dflash2 --tp 1 --hosts localhost   # fastest
sparkrun run @chisel/fp8-mtp4-draftvocab --tp 1 --hosts localhost    # FP8, vLLM 0.31.0
```

With plain docker (fp8-mtp4-draftvocab):

```
bash recipes/qwen3.8-27b/fp8-mtp4-draftvocab.docker.sh
```

Either way the server is OpenAI-compatible on port 8000. chisel-nvfp4-dflash2 runs the spark-arena vLLM nightly image as is. fp8-mtp4-draftvocab is stock vLLM 0.31.0 (the official `vllm/vllm-openai:v0.31.0` image) with one small plugin, [chisel-draftvocab](plugin/), installed at start.

## Qwen3.8-27B: chisel-nvfp4-dflash2

**What it is**
- Weights: `unsloth/Qwen3.8-27B-NVFP4` (4-bit weights on the GB10's FP4 tensor cores). Context up to 262,144 tokens, thinking on with medium effort, up to 4 concurrent requests.
- Speculative decoding with DFlash2, 10 draft tokens per step, using **Chisel's NVFP4 drafter** ([yuanlin2004/Qwen3.8-27B-Chisel-DFlash2-NVFP4-drafter](https://huggingface.co/yuanlin2004/Qwen3.8-27B-Chisel-DFlash2-NVFP4-drafter)): z-lab's DFlash2 drafter with its linear layers quantized to NVFP4, 1.4 GB instead of 3.8 GB.
- The linear-attention recurrent state is cached in BF16 instead of FP32 (`--mamba-ssm-cache-dtype bfloat16`), which halves the state written per draft position.
- Builds on Luis Poveda's single-Spark recipe on spark-arena (unsloth NVFP4 + DFlash2, 8 drafts). On our box, under the same cap, that recipe measured 22.8 ms per token on our benchmark; this one 16.5 ms.

**Speed** (2026-10-14, DGX Spark, driver 580.178.04, GPU clock capped at 2,000 MHz)
- Decode: median 16.44 and 16.54 ms per output token in two fresh-start sessions (about 60.5 tok/s), p90 30.9 to 31.1 ms, over the same 25 prompts as fp8-mtp4-draftvocab, 256 output tokens, three repeats each, one user. By prompt type: JSON 14.2 ms, code 15.5, chat 28.9, long story 30.5. 4.34 draft tokens accepted per step.
- Time to first token: median 250 ms on the same prompts.
- llama-benchy 0.4.0 with spark-arena's settings, run locally (not arena scores): 51.6 tok/s total with 2 users at 16k context (10 runs over two sessions; single runs 42 to 61), 30.7 with 1 user at 16k, 38.5 with 1 user and no context (20 runs). Prefill: 2,106 tok/s on a 2,048-token prompt, about 1,120 at 16k context. Single runs vary a lot because each run continues a different random slice of a novel, and draft acceptance follows the text.
- Details: [`results/qwen3.8-27b/chisel-nvfp4-dflash2/2026-10-14/`](results/qwen3.8-27b/chisel-nvfp4-dflash2/2026-10-14/).

**Quality against BF16** (Qwen/Qwen3.8-27B, BF16 weights and KV cache)

| Task | BF16 | This recipe | Recovery | Difference [95% CI] | Answer length vs BF16 |
|---|---|---|---|---|---|
| gsm8k (5-shot) | 90.07 | 89.46 | 99.3% | −0.61 [−1.90, +0.69] | 1.002 |
| ifeval (prompt-level strict) | 80.96 | 84.10 | 103.9% | +3.14 [+0.57, +5.72] | 1.024 |
| humaneval (pass@1, chat) | 95.12 | 94.51 | 99.4% | −0.61 [−3.78, +2.56] | 1.018 |
| LongBench NarrativeQA (F1, full stories) | 34.83 | 34.92 | 100.3% | +0.09 [−1.92, +2.10] | 0.938 |

Same bar and method as below; all four tasks pass. Scored with thinking off; method and per-question answers: [`quality/`](results/qwen3.8-27b/chisel-nvfp4-dflash2/2026-10-14/quality/).

## Qwen3.8-27B: fp8-mtp4-draftvocab

**What it is**
- Weights: `Qwen/Qwen3.8-27B-FP8` (Qwen's official FP8 checkpoint), FP8 KV cache, FlashInfer attention, prefix caching. Context up to 131,072 tokens (the results below were measured with an 81,920-token limit; the limit was raised later so spark-arena's 100k-token tests fit).
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

Apache-2.0. Model weights are under their own licences (Qwen3.8-27B, unsloth's NVFP4 checkpoint, z-lab's DFlash2 drafter and Chisel's NVFP4 drafter: all Apache-2.0).
