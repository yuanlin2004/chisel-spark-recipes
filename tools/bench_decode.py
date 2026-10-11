#!/usr/bin/env python3
"""Measure time to first token and inter-token latency against an OpenAI-compatible
streaming endpoint (stock vLLM for the baseline, or Chisel's engine).

Standard library only. Prompts file: JSON lines with a "prompt" field (optionally "class").
"""
import argparse, json, statistics, sys, threading, time, urllib.request


def stream(url, model, prompt, max_tokens, temperature, api="chat"):
    # api "chat": the prompt is one user message, formatted by the model's chat template (what a served
    # chat model sees). api "completions": the raw prompt text, no template.
    req_body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
    if api == "chat":
        req_body["messages"] = [{"role": "user", "content": prompt}]
        path = "/v1/chat/completions"
    else:
        req_body["prompt"] = prompt
        path = "/v1/completions"
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(req_body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    arrivals, usage = [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            ch = (ev.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if ch.get("text") or delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                arrivals.append(time.perf_counter())
    tokens = (usage or {}).get("completion_tokens") or len(arrivals)
    return t0, arrivals, tokens


def scrape_metrics(url):
    """Speculative-decoding counters from the server's Prometheus /metrics (vLLM), summed over labels."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def run(a, prompts):
    results, empty, lock = [], [], threading.Lock()
    queue = list(prompts)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                p = queue.pop(0)
            t0, arr, tokens = stream(a.url, a.model, p["prompt"], a.max_tokens, a.temperature, a.api)
            if len(arr) < 2:  # nothing to time; counted in the summary, not silently dropped
                with lock:
                    empty.append({"class": p.get("class", "unlabeled"), "tokens": tokens})
                continue
            # Per output token latency: decode time spread over tokens after the first.
            # Chunks can carry several tokens under speculation, so divide by tokens, not chunks.
            decode = arr[-1] - arr[0]
            itl = decode / max(1, tokens - 1)
            gaps = [b - c for b, c in zip(arr[1:], arr[:-1])]
            with lock:
                results.append({"class": p.get("class", "unlabeled"), "ttft_s": arr[0] - t0, "tokens": tokens,
                                "decode_s": decode, "itl_s": itl, "chunks": len(arr), "chunk_gaps_s": gaps,
                                "end": arr[-1], "start": t0})

    threads = [threading.Thread(target=worker) for _ in range(a.users)]
    for t in threads: t.start()
    for t in threads: t.join()
    run.empty = empty
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True); p.add_argument("--n", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3); p.add_argument("--users", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=512); p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--report-acceptance", action="store_true",
                   help="also report tokens per streamed chunk per class (a proxy; prefer accepted_draft_tokens_per_step from the server counters)")
    p.add_argument("--api", choices=["chat", "completions"], default="chat",
                   help="chat: send each prompt as a user message through the chat template (default)")
    p.add_argument("--out")
    a = p.parse_args()
    prompts = [json.loads(l) for l in open(a.prompts) if l.strip()]
    if a.warmup:
        run(argparse.Namespace(**{**vars(a), "users": 1}), prompts[: a.warmup])
    if a.n == 0:
        return
    picked = (prompts * (a.n // len(prompts) + 1))[: a.n]
    before = scrape_metrics(a.url)
    t_start = time.perf_counter()
    res = run(a, picked)
    wall = time.perf_counter() - t_start
    itl = [r["itl_s"] * 1000 for r in res]
    summary = {
        "url": a.url, "model": a.model, "users": a.users, "requests": len(res), "max_tokens": a.max_tokens,
        "ttft_ms_median": statistics.median(r["ttft_s"] * 1000 for r in res),
        "itl_ms_median": statistics.median(itl), "itl_ms_p90": pct(itl, 0.9),
        "tok_s_per_user_median": statistics.median(r["tokens"] / r["decode_s"] for r in res if r["decode_s"] > 0),
        "tok_s_total": sum(r["tokens"] for r in res) / wall, "api": a.api,
        "untimed_requests": len(run.empty), "untimed_by_class": {c: sum(1 for e in run.empty if e["class"] == c) for c in {e["class"] for e in run.empty}},
        "itl_ms_median_by_class": {c: statistics.median(r["itl_s"] * 1000 for r in res if r["class"] == c) for c in {r["class"] for r in res}},
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    after = scrape_metrics(a.url)
    if after:
        delta = {k: after[k] - before.get(k, 0.0) for k in after}
        summary["spec_decode_counters"] = delta
        acc = next((v for k, v in delta.items() if "accepted_tokens" in k and not k.endswith("_created")), None)
        drafts = next((v for k, v in delta.items() if k.endswith("num_drafts_total") or k.endswith("num_drafts")), None)
        if acc is not None and drafts:
            summary["accepted_draft_tokens_per_step"] = acc / drafts  # committed tokens per step = this + 1
    if a.report_acceptance:
        by = {}
        for r in res:
            by.setdefault(r["class"], []).append(r["tokens"] / max(1, r["chunks"]))
        summary["tokens_per_chunk_by_class"] = {k: statistics.mean(v) for k, v in by.items()}
    print(json.dumps(summary, indent=2))
    if a.out:
        json.dump({"summary": summary, "requests": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
