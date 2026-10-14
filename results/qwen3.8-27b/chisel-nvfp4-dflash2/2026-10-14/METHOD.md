# Method

## Speed (benchmark/)
- Our benchmark: `tools/bench_decode.py`, streaming `/v1/chat/completions`, thinking on with medium effort (the recipe's default), greedy, `max_tokens` 256, one request at a time, on `prompts/qwen3.8-27b-test25.jsonl` (9 chat, 9 code, 6 JSON-schema, 1 long story of about 34k tokens). Decode latency per token = (time of the last streamed token − time of the first) / (output tokens − 1). Two fresh-start sessions (s4x, s2), each 3 repeats of 25 requests: bench-s4x-r1..3.json, bench-s2-r1..3.json.
- spark-arena's test shape, run locally (not arena scores): llama-benchy 0.4.0 with the spark-arena-v2 settings (pp 2048, tg 128, prefix caching on). llama-benchy-d0-*.json: no context, 1 user, 10 runs per session. llama-benchy-d16384-*.json: 16,384 tokens of context, 1 and 2 users, 5 runs per session. The leaderboard ranks by tg128 at 16k context with 2 users; that number is the total across both users.
- The GPU monitor logs (gpu-*.csv) show the clock held at the 2,000 MHz cap with no gaps.

## Quality (quality/)
- lm-evaluation-harness 0.4.13 through `tools/quality_eval.py --chat`, chat template, greedy, full tasks, the server started with `--default-chat-template-kwargs '{"enable_thinking": false}'` (quality/server-quality.cmd); all other flags as the recipe.
- Tasks and bar as for fp8-mtp4-draftvocab (see its METHOD.md): gsm8k, ifeval, humaneval_chat, longbench_narrativeqa; at least 97% of BF16 on every task, no task clearly worse, mean answer length within 90–110% of BF16.
- Compared with `tools/compare_quality.py` against `results/qwen3.8-27b/bf16-reference/2026-10-09/quality/` (compare-vs-bf16.json). Per-question answers: quality/samples/.
- Reproduce the comparison: `python tools/compare_quality.py --reference results/qwen3.8-27b/bf16-reference/2026-10-09/quality/scores.json --candidate results/qwen3.8-27b/chisel-nvfp4-dflash2/2026-10-14/quality/scores.json`
