import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


# --- Chisel addition: programs from a chat answer (humaneval_chat) ---
import re as _re


def build_predictions_chat(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Take the first fenced code block of a chat answer. If it defines the entry point, it is the
    program (keep the prompt's imports in front); otherwise it is the body, appended to the prompt."""
    out = []
    for resp, doc in zip(resps, docs):
        progs = []
        for r in resp:
            m = _re.search(r"```(?:python|py)?\s*\n(.*?)(```|$)", r, _re.S)
            code = m.group(1) if m else r
            if f"def {doc['entry_point']}" in code:
                header = "\n".join(l for l in doc["prompt"].splitlines() if l.startswith(("import ", "from ")))
                progs.append(header + "\n" + code)
            else:
                progs.append(doc["prompt"] + code)
        out.append(progs)
    return out
