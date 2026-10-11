#!/usr/bin/env python3
"""Compare a candidate's task scores with a reference's (normally BF16), question by question.

  .venv/bin/python tools/compare_quality.py --reference runs/<run>/quality-bf16.json \
      --candidate runs/<run>/quality-<config>.json --margin 1.0 --out runs/<run>/quality-compare-<config>.json

Reads the per-question samples that tools/quality_eval.py saves. For each task it reports both scores,
the paired difference (candidate minus reference, in points) with a 95% confidence interval, the
number of answer flips (right to wrong and wrong to right) for pass/fail metrics, and the ratio of
mean answer length. A task passes the non-inferiority test when the interval's lower end is above
minus --margin points (reported only). The gate (Chisel's bar, owner's choice 2026-10-09) is recovery: every
task keeps at least --min-recovery of the reference score, no task is clearly worse (the whole interval below
zero), and the mean answer length stays within --length-range of the reference's. Standard library only.
"""
import argparse, gzip, json, math, os, statistics

# Metric and filter used per task (the first listed in lm-eval's report for that task).
METRIC = {"gsm8k": ("exact_match", "flexible-extract"), "ifeval": ("prompt_level_strict_acc", "none"),
          "humaneval": ("pass@1", "create_test"), "humaneval_stop4": ("pass@1", "create_test"),
          "humaneval_instruct": ("pass@1", "create_test"), "humaneval_chat": ("pass@1", "create_test"), "mbpp": ("pass_at_1", None),
          "longbench_narrativeqa": ("qa_f1_score", "none")}


def value(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return {"true": 1.0, "false": 0.0}.get(v.lower(), None) if not v.replace(".", "", 1).isdigit() else float(v)
    if isinstance(v, list) and v:  # per-instruction list: share satisfied
        vals = [value(x) for x in v]
        return sum(vals) / len(vals)
    return None


def load(samples_dir, task):
    metric, flt = METRIC.get(task, (None, None))
    rows = {}
    path = os.path.join(samples_dir, f"{task}.jsonl")
    fh = open(path) if os.path.exists(path) else gzip.open(path + ".gz", "rt")
    for line in fh:
        r = json.loads(line)
        if flt and r.get("filter") not in (flt, None):
            continue
        m = metric or next(k for k in r.get("metrics", []))
        resp = r.get("resps")
        text = resp[0][0] if isinstance(resp, list) and resp and isinstance(resp[0], list) else str(resp)
        rows[str(r["doc_id"])] = (value(r.get(m)), len(text))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True); p.add_argument("--candidate", required=True)
    p.add_argument("--margin", type=float, default=1.0, help="allowed loss in points for the non-inferiority test")
    p.add_argument("--min-recovery", type=float, default=0.97, help="minimum candidate score / reference score per task")
    p.add_argument("--length-range", type=float, nargs=2, default=[0.9, 1.1], help="allowed answer length ratio")
    p.add_argument("--out")
    a = p.parse_args()
    ref, cand = json.load(open(a.reference)), json.load(open(a.candidate))
    tasks = [t for t in ref["tasks"].split(",") if t in cand["tasks"].split(",")]
    res = {"reference": a.reference, "candidate": a.candidate, "margin_points": a.margin, "tasks": {}}
    for t in tasks:
        r, c = load(ref["samples"], t), load(cand["samples"], t)
        ids = sorted(set(r) & set(c), key=lambda x: int(x) if x.isdigit() else x)
        ids = [i for i in ids if r[i][0] is not None and c[i][0] is not None]
        d = [100 * (c[i][0] - r[i][0]) for i in ids]
        n = len(d)
        mean = statistics.fmean(d) if n else float("nan")
        se = statistics.stdev(d) / math.sqrt(n) if n > 1 else float("nan")
        binary = all(r[i][0] in (0.0, 1.0) and c[i][0] in (0.0, 1.0) for i in ids)
        out = {"n": n, "reference_score": 100 * statistics.fmean(r[i][0] for i in ids),
               "candidate_score": 100 * statistics.fmean(c[i][0] for i in ids),
               "diff_points": mean, "ci95_points": [mean - 1.96 * se, mean + 1.96 * se],
               "non_inferior": (mean - 1.96 * se) > -a.margin,
               "length_ratio": statistics.fmean(c[i][1] for i in ids) / max(1e-9, statistics.fmean(r[i][1] for i in ids))}
        if binary:
            out["flips_right_to_wrong"] = sum(1 for i in ids if r[i][0] == 1 and c[i][0] == 0)
            out["flips_wrong_to_right"] = sum(1 for i in ids if r[i][0] == 0 and c[i][0] == 1)
        out["recovery"] = out["candidate_score"] / out["reference_score"] if out["reference_score"] else None
        out["clearly_worse"] = out["ci95_points"][1] < 0
        out["length_ok"] = a.length_range[0] <= out["length_ratio"] <= a.length_range[1]
        out["passes"] = (out["recovery"] is not None and out["recovery"] >= a.min_recovery
                         and not out["clearly_worse"] and out["length_ok"])
        res["tasks"][t] = out
    res["all_non_inferior"] = all(v["non_inferior"] for v in res["tasks"].values())
    res["bar"] = {"min_recovery": a.min_recovery, "length_range": a.length_range, "no_task_clearly_worse": True}
    res["passes"] = all(v["passes"] for v in res["tasks"].values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
