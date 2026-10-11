# chisel-draftvocab

A vLLM general plugin (entry point `vllm.general_plugins: chisel`). It installs hooks at start and does nothing unless a pass is turned on in `--additional-config`.

## Reduced-vocabulary draft head
`--additional-config '{"chisel_draft_vocab": "qwen3.8-27b"}'` (a token list bundled in `chisel_draftvocab/data/`, or a path to a file of token ids, one per line or space-separated).

After vLLM loads the speculative drafter, the pass replaces the drafter's output head with one holding only the listed token rows, and maps draft ids back to target ids (`draft_id_to_target_id`, the EAGLE-3 convention). The target model's verify step keeps its full output layer, so served tokens are unchanged; a token outside the list is simply never drafted, which can only lower acceptance. The server log shows `chisel draft vocab: draft head reduced to N of V tokens` when it is active.

Tested with vLLM 0.31.0 (V1 proposer and V2 model runner MTP speculator), single GPU.
