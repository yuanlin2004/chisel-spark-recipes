# Method

## Speed (benchmark/)
- Client: `tools/bench_decode.py`, streaming `/v1/chat/completions` (each prompt is one user message through the model's chat template; thinking on, the model's default), greedy, `max_tokens` 256, one request at a time.
- Prompts: `prompts/qwen3.8-27b-test25.jsonl` (9 chat, 9 code, 6 JSON-schema, 1 long story of about 34k tokens).
- Per request, decode latency per token = (time of the last streamed token − time of the first) / (output tokens − 1); time to first token is measured separately. Reported: median and 90th percentile across requests, and the median per prompt type.
- 5 warm-up requests, then 3 repeats of 25 requests (bench-r1..3.json). The GPU monitor log (gpu.csv) must show no clock drop below 95% of the cap and no gaps.
- Reproduce: start the recipe, then
  `python tools/bench_decode.py --url http://localhost:8000 --model qwen3.8-27b --prompts prompts/qwen3.8-27b-test25.jsonl --n 25 --max-tokens 256 --report-acceptance --out bench.json`

## Quality (quality/)
- lm-evaluation-harness 0.4.13 through `tools/quality_eval.py --chat`: `local-chat-completions`, chat template, few-shot examples as turns, greedy, full tasks, 16 concurrent requests; the server started with `--default-chat-template-kwargs '{"enable_thinking": false}'`.
- Tasks: gsm8k (5-shot, flexible-extract), ifeval (prompt-level strict), humaneval_chat (`tools/lm_eval_tasks/humaneval_chat`: the problem as a chat message, the answer's first code block tested, pass@1; lm-eval's humaneval_instruct needs an assistant prefill the chat API does not honour), longbench_narrativeqa (full stories, up to 81,920 tokens of context).
- Compared question by question with `tools/compare_quality.py` against `results/qwen3.8-27b/bf16-reference/2026-10-09/quality/`. Bar: recovery (recipe score / BF16 score) at least 97% on every task, no task with its 95% interval on the paired difference entirely below zero, mean answer length within 90–110% of BF16.
- Measured on the recipe's own container (`recipes/qwen3.8-27b/fp8-mtp4-draftvocab.docker.sh` with `EXTRA_ARGS="--default-chat-template-kwargs '{\"enable_thinking\": false}'"`; server-quality.log).
- outputs/reference100-container.jsonl holds the container's greedy outputs (256 tokens, thinking on) on `prompts/qwen3.8-27b-reference100.jsonl`, for anyone checking that their build behaves the same (`tools/check_outputs.py compare`). Expect partial rather than exact agreement across different CUDA library builds: greedy decoding is sensitive to kernel and library details.
- Reproduce the comparison from the stored files: `python tools/compare_quality.py --reference results/qwen3.8-27b/bf16-reference/2026-10-09/quality/scores.json --candidate results/qwen3.8-27b/fp8-mtp4-draftvocab/2026-10-10/quality/scores.json`
