#!/usr/bin/env python3
"""Record greedy reference outputs, or compare a configuration's outputs against them.

record:  store outputs (normally from the BF16 configuration).
compare: report exact-match share, mean matching-prefix length (in characters) and the output length ratio.
Exact match is the test for lossless changes; record the chosen format's own outputs and compare against
those. For a lossy format against BF16 it is reported, not a gate (tools/compare_quality.py and
tools/logit_distance.py judge the format).
Standard library only. Prompts file: JSON lines with a "prompt" field.
"""
import argparse, json, os, urllib.request


def complete(url, model, prompt, max_tokens, api="chat"):
    # api "chat" sends the prompt as one user message through the chat template, as bench_decode.py does.
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    if api == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
        path = "/v1/chat/completions"
    else:
        body["prompt"] = prompt
        path = "/v1/completions"
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        ch = json.load(r)["choices"][0]
        if api == "chat":
            m = ch["message"]
            return (m.get("reasoning_content") or m.get("reasoning") or "") + (m.get("content") or "")
        return ch["text"]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="cmd", required=True)
    r = s.add_parser("record"); r.add_argument("--prompts", required=True)
    c = s.add_parser("compare"); c.add_argument("--reference", required=True)
    for q in (r, c):
        q.add_argument("--url", required=True); q.add_argument("--model", required=True)
        q.add_argument("--max-tokens", type=int, default=256); q.add_argument("--out", required=True)
        q.add_argument("--api", choices=["chat", "completions"], default="chat")
    a = p.parse_args()
    if a.cmd == "record":
        with open(a.out, "w") as fh:
            for line in open(a.prompts):
                if line.strip():
                    pr = json.loads(line)["prompt"]
                    fh.write(json.dumps({"prompt": pr, "output": complete(a.url, a.model, pr, a.max_tokens, a.api)}) + "\n")
        print(f"recorded {a.out}")
        return
    refs = [json.loads(l) for l in open(a.reference) if l.strip()]
    rows = []
    for ref in refs:
        out = complete(a.url, a.model, ref["prompt"], a.max_tokens, a.api)
        rows.append({"exact": out == ref["output"], "prefix_chars": common_prefix(out, ref["output"]),
                     "ref_chars": len(ref["output"]), "out_chars": len(out), "first_diff": None if out == ref["output"] else out[:200]})
    summary = {"prompts": len(rows), "exact_match_share": sum(r["exact"] for r in rows) / max(1, len(rows)),
               "mean_prefix_share": sum(r["prefix_chars"] / max(1, r["ref_chars"]) for r in rows) / max(1, len(rows)),
               "length_ratio": sum(r["out_chars"] for r in rows) / max(1, sum(r["ref_chars"] for r in rows))}
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
