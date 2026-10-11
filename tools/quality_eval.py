#!/usr/bin/env python3
"""Run a task-accuracy set against a running OpenAI-compatible server with lm-evaluation-harness,
and save the scores and the per-question results in the run folder. Use it for the format choice
in a goal's quality bar; compare two runs with tools/compare_quality.py.

  .venv/bin/python tools/quality_eval.py --url http://localhost:8000 --model <served name> \
      --tokenizer <hf repo or path> --tasks gsm8k,ifeval,humaneval,longbench_narrativeqa \
      --out runs/<run>/quality-<config>.json

Each task runs as its own lm-eval call, so a failure in one task cannot lose another's score; a task
whose result file already exists in the samples folder is skipped, so a rerun resumes. Per-question
results go to runs/<run>/quality-<config>-samples/<task>.jsonl (needed for paired differences and
answer flips). Generative tasks only: the completions endpoint does not give prompt log-probabilities
on every server. Use --chat for a chat model (the owner's choice for Qwen3.8 on 2026-10-09: raw few-shot
prompts let it slip into its thinking mode and get cut off). Code tasks (humaneval, mbpp) execute the model's code on this machine; they run only
with --allow-code-execution.
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile

CODE_TASKS = ("humaneval", "mbpp")
# Local task variants (tools/lm_eval_tasks), e.g. humaneval_stop4: humaneval with the four stop strings vLLM accepts.
TASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_eval_tasks")


def run_task(a, task, samples_dir):
    lm_eval = os.path.join(os.path.dirname(sys.executable), "lm_eval")
    env = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        if a.chat:  # through the model's chat template, as a served chat model is used
            cmd = [lm_eval, "--model", "local-chat-completions",
                   "--model_args", f"model={a.model},base_url={a.url.rstrip('/')}/v1/chat/completions,"
                                   f"num_concurrent={a.concurrency},timeout={a.timeout}",
                   "--apply_chat_template", "--fewshot_as_multiturn"]
        else:
            cmd = [lm_eval, "--model", "local-completions",
                   "--model_args", f"model={a.model},base_url={a.url.rstrip('/')}/v1/completions,tokenizer={a.tokenizer},"
                                   f"num_concurrent={a.concurrency},tokenized_requests=False,timeout={a.timeout},max_length={a.max_length}"]
        cmd += ["--tasks", task, "--output_path", tmp, "--log_samples", "--include_path", TASK_DIR]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if task.startswith(CODE_TASKS):
            if not a.allow_code_execution:
                sys.exit(f"{task} executes generated code; pass --allow-code-execution to run it")
            cmd += ["--confirm_run_unsafe_code"]
            env["HF_ALLOW_CODE_EVAL"] = "1"
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)
        results = None
        for f in glob.glob(os.path.join(tmp, "**", "results*.json"), recursive=True):
            results = json.load(open(f))
        if results is None:
            sys.exit(f"lm_eval wrote no results file for {task}")
        for f in glob.glob(os.path.join(tmp, "**", "samples_*.jsonl"), recursive=True):
            name = os.path.basename(f).split("_20")[0].replace("samples_", "")  # samples_<task>_<timestamp>.jsonl
            shutil.copy(f, os.path.join(samples_dir, f"{name}.jsonl"))
        json.dump(results, open(os.path.join(samples_dir, f"{task}.results.json"), "w"), indent=1)
        return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True); p.add_argument("--model", required=True); p.add_argument("--tokenizer", required=True)
    p.add_argument("--tasks", default="gsm8k,ifeval"); p.add_argument("--limit", type=int, default=0, help="0 = full task")
    p.add_argument("--concurrency", type=int, default=4); p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=int, default=3600,
                   help="seconds per request; lm-eval's default of 300 is too short for long generations on a slow format")
    p.add_argument("--max-length", type=int, default=81920,
                   help="context length lm-eval may use; its default of 2,048 silently cuts long-context prompts (set to the server's max-model-len)")
    p.add_argument("--chat", action="store_true",
                   help="send each question through the chat template (lm-eval local-chat-completions, few-shot as turns). "
                        "Serve with --default-chat-template-kwargs to fix the thinking mode for every request")
    p.add_argument("--allow-code-execution", action="store_true",
                   help="allow code tasks, which run the model's generated code on this machine")
    a = p.parse_args()
    samples_dir = os.path.splitext(a.out)[0] + "-samples"
    os.makedirs(samples_dir, exist_ok=True)
    merged = {"results": {}, "versions": {}, "configs": {}}
    for task in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        done = os.path.join(samples_dir, f"{task}.results.json")
        r = json.load(open(done)) if os.path.exists(done) else run_task(a, task, samples_dir)
        for k in merged:
            merged[k].update(r.get(k) or {})
    out = {"url": a.url, "model": a.model, "tasks": a.tasks, "limit": a.limit, "samples": samples_dir,
           "results": merged["results"], "versions": merged["versions"]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["results"], indent=1))


if __name__ == "__main__":
    main()
