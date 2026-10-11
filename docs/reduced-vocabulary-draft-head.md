# The reduced-vocabulary draft head

This is the one change chisel-draftvocab makes to stock vLLM in the `fp8-mtp4-draftvocab` recipe. It makes speculative decoding cheaper without changing a single served token.

## 1. Background: where a decode step's time goes

On a DGX Spark, single-user decode is limited by memory bandwidth. The GPU has to stream the model's weights from memory for every forward pass, and at about 250 GB/s the bytes read decide the time.

Qwen3.8-27B-FP8 reads about **27.9 GB per forward pass**. That covers all 64 layers plus the output layer ("lm_head"), and at 249 GB/s it is about 112 ms. Plain decoding produces one token per pass, so it can't go faster than that.

Speculative decoding produces several tokens per pass:

1. **Draft.** A small, cheap predictor guesses the next *k* tokens one at a time. Qwen3.8 ships its own predictor, the MTP head: a single extra transformer layer trained to predict the token after next.
2. **Verify.** The full model runs once over all *k* guesses together. Because decode is bandwidth-bound, checking 4 positions costs about the same as checking 1.
3. **Accept.** The model keeps the guesses that match what it would have chosen itself, and adds one more token of its own. With greedy decoding a guess is kept only if it equals the model's own top choice. So the output is exactly what the model would produce token by token. Speculation changes only how many tokens each pass yields, not which tokens.

## 2. The hidden cost: the draft step's output layer

Each draft step has to turn the MTP layer's hidden state into a token, and it does that through the model's output layer:

- The output layer is a 248,320 × 5,120 matrix, one row per vocabulary token. It stays in **BF16** even in the FP8 checkpoint, so it is **2.54 GB**.
- The MTP layer itself is only 0.48 GB in the FP8 checkpoint.
- So each draft token reads 0.48 GB of MTP layer and **2.54 GB of output layer**. The output layer is 84% of the draft step's cost.

Per decode step with 3 draft tokens:

| Part | GB read |
|---|---|
| Verify pass (whole model, output layer once) | 27.9 |
| 3 draft steps: MTP layer | 3 × 0.48 = 1.4 |
| 3 draft steps: output layer | **3 × 2.54 = 7.6** |
| **Total** | **36.9** |

A fifth of every step went into re-reading the output layer just to propose guesses. The step breakdown confirmed this on the INT4 arm, where a BF16 matrix-vector kernel of exactly that shape was 22% of GPU time.

## 3. The idea

A guess only has to be *good enough to be accepted*. It doesn't have to be chosen from the whole vocabulary:

- Text is dominated by a fraction of the vocabulary. 248,320 tokens cover dozens of languages, scripts and rare fragments, but the tokens actually generated for English chat, code and JSON come from a much smaller set.
- If the draft head can only propose tokens from that set, it is rarely wrong *because of* the restriction. When the right next token is outside the set, the guess is simply wrong, the verify step rejects it, and the model writes the right token itself. We lose a bit of acceptance, never correctness.
- The verify step still uses the full 248,320-row output layer, so the model's own choice is always computed over the full vocabulary.

So the draft steps get their own small output layer: the rows of the real output layer for the chosen tokens, copied once at start-up.

## 4. Choosing the tokens

`qwen3.8-27b.txt` lists **50,858 tokens**, 20% of the vocabulary. Their rows take **0.52 GB** instead of 2.54. The list is built from text that is *not* in the benchmark:

- every token that appears in the calibration and tuning prompt sets (chat, code, JSON and long stories from public datasets);
- every token in BF16 Qwen3.8-27B's answers to the four quality tasks;
- every token id below 32,768 (byte tokens and the most common merges sit there in Qwen's tokenizer);
- all special and control tokens.

Coverage was checked on the held-out test outputs: the list contains **97.9%** of the tokens BF16 actually generates there (chat 97.7%, code 97.9%, JSON 97.9%, long stories 99.3%). That is the share of positions where the reduced head can still propose the right token.

## 5. How it works inside vLLM

chisel-draftvocab is a vLLM *general plugin*, which vLLM loads in every process at start-up. It does nothing unless the server is started with `--additional-config '{"chisel_draft_vocab": "qwen3.8-27b"}'`. With that setting:

1. vLLM loads the target model and the MTP drafter as usual. The drafter's output layer is shared with the target model's, so it is the same 2.54 GB tensor.
2. Right after that, the plugin builds a new output layer for the drafter alone, with 50,858 rows. They are gathered from the shared tensor by token id and copied once (0.52 GB of extra memory). It then points the drafter at the new layer. The target model keeps the original.
3. It attaches a mapping from draft index to real token id (`draft_id_to_target_id`). When the drafter picks row *i* of its small layer, the token proposed is `ids[i]`. This is the same convention vLLM already uses for EAGLE-3 drafters with their own small vocabularies, so the rest of vLLM's speculation code handles it without changes.
4. If any code path asks the drafter for full logits instead of its top token, the plugin scatters the 50,858 scores into a full-size vector with −∞ everywhere else, so nothing downstream sees a shape change.

From then on every draft step multiplies by a 0.52 GB matrix instead of a 2.54 GB one. CUDA graphs capture the smaller layer like any other weight. The server log confirms it: `chisel draft vocab: draft head reduced to 50858 of 248320 tokens (2.54 GB to 0.52 GB)`.

## 6. What it buys

Bytes per decode step, FP8 + MTP:

| Draft tokens | Stock | With reduced head | Saving |
|---|---|---|---|
| 3 | 27.9 + 3 × 3.02 = 36.9 GB | 27.9 + 3 × 1.00 = 30.9 GB | 16% |
| 4 | 27.9 + 4 × 3.02 = 40.0 GB | 27.9 + 4 × 1.00 = 31.9 GB | 20% |

Measured on one DGX Spark, the same session, 25 prompts, 256 tokens, one user:

| | Decode per token (median) | Accepted drafts per step |
|---|---|---|
| FP8 + MTP, 3 drafts, stock | 59.9 ms | 2.02 |
| FP8 + MTP, 3 drafts, reduced head | 49.0 ms (−18%) | 1.96 |
| FP8 + MTP, 4 drafts, reduced head | **43.4 ms** | 2.31 |

- **Output unchanged.** With 3 drafts, the reduced head gave word-for-word the same output as stock FP8 + MTP on 100 of 100 reference prompts.
- **Acceptance cost is small.** It dropped 3% (2.02 to 1.96), close to the 2% of positions the list doesn't cover.
- **It makes more drafts worth it.** Stock, each extra draft token cost 3.0 GB, so the 4th guess rarely paid for itself. With the reduced head an extra draft costs 1.0 GB, so 4 drafts beat 3 by another 11%. That is why the recipe uses 4.

## 7. Limits

- **Model-specific list.** The token list is for Qwen3.8's tokenizer and for English chat, code and JSON. Other languages would want a list built from their own text. Outputs stay correct either way; only speed would suffer.
- **Single GPU.** The plugin skips itself under tensor parallelism; it was tested only on one GPU.
- **Drafters that use the target's output layer.** It applies to drafters that propose through the target model's output layer: MTP heads, and EAGLE-style or block drafters that share it. A drafter with its own small vocabulary already has this property.
- **Not a new idea in general.** EAGLE-3 trains its drafter on a reduced vocabulary, and FR-Spec (frequency-ranked speculative sampling) proposes the same restriction. What's new here is applying it, without retraining, to a model's built-in MTP head in stock vLLM, measuring it on a DGX Spark, and checking that served outputs don't change.
