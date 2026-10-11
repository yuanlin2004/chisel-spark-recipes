# Prompt sets

Both files are built from public datasets by Chisel's prompt builder (seed 0) and are used unchanged here.

- `qwen3.8-27b-test25.jsonl`: the 25 test prompts every speed number in this repository is measured on (chat 9, code 9, structured 6, long 1). One JSON object per line: `prompt`, `class`, `max_tokens`, `source`.
- `qwen3.8-27b-reference100.jsonl`: 100 test prompts used to check that a recipe's outputs match the configuration it claims to be equivalent to.

Sources, by class:
- chat: HuggingFaceH4/ultrachat_200k (test_sft)
- code: openai/openai_humaneval (test), google-research-datasets/mbpp (test)
- structured: NousResearch/json-mode-eval (train)
- long: LongBench NarrativeQA (zai-org/LongBench, test), full stories
