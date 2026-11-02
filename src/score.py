"""Parse reader replies and score them against reference labels.

Answer reading rule (same for every reader): after whitespace and leading quotes or asterisks are removed, the reply
must start with one of the answer strings, case-insensitive; for the timing question "3 to 5 years" etc. without the
leading "in" are accepted (added after the pilot, recorded in the technical appendix). Replies that start otherwise
are counted as format failures, not guessed.

Usage: python src/score.py --replies results/demo_timing_v4.jsonl --labels data/synthetic_labels.csv --question timing
       python src/score.py --replies results/demo_stage.jsonl --labels data/synthetic_labels.csv --question stage
Labels file: columns id, timing, stage (answer strings).
"""
import argparse
import csv
import json
import re
from collections import Counter

TIMING = ["already in commercial use", "within 2 years", "in 3 to 5 years", "in 6 to 10 years",
          "more than 10 years from now", "never or not foreseeable", "no information is provided"]
STAGE = ["research only", "pilots or trials", "paid early deployments", "regular commercial use",
         "no information is provided"]
NO = "no information is provided"
YEARS = {"already in commercial use": 0, "within 2 years": 1, "in 3 to 5 years": 4, "in 6 to 10 years": 8,
         "more than 10 years from now": 15}


def norm(s):
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


def parse(reply, options):
    aliases = {o[3:]: o for o in options if o.startswith("in ")}
    r = norm(reply).lstrip("*\"' ").lower()
    for o in sorted(list(options) + list(aliases), key=len, reverse=True):
        if r.startswith(o):
            return aliases.get(o, o)
    return None


def kappa(a, b, cats, weights=None):
    """Cohen's kappa; with weights = disagreement weights (e.g. |i - j| for linear weighting)."""
    n = len(a)
    if n == 0:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    w = weights or [[0 if i == j else 1 for j in range(len(cats))] for i in range(len(cats))]
    obs = sum(w[idx[x]][idx[y]] for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    exp = sum(w[i][j] * pa[ci] * pb[cj] for i, ci in enumerate(cats) for j, cj in enumerate(cats)) / n / n
    return None if exp == 0 else round(1 - obs / exp, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--question", choices=["timing", "stage"], required=True)
    a = ap.parse_args()
    opts = TIMING if a.question == "timing" else STAGE
    lab = {r["id"]: r[a.question] for r in csv.DictReader(open(a.labels))}
    rep = {}
    for line in open(a.replies):
        j = json.loads(line)
        if not j["is_error"]:
            rep[j["id"]] = parse(j["reply"], opts)
    ids = [i for i in lab if i in rep]
    x = [lab[i] for i in ids]
    y = [rep[i] if rep[i] is not None else NO for i in ids]
    info = lambda c: c != NO
    out = {"question": a.question, "n": len(ids), "format_ok": sum(rep[i] is not None for i in ids),
           "agree": sum(p == q for p, q in zip(x, y)), "kappa": kappa(x, y, opts),
           "reader_informative": sum(map(info, y)),
           "reader_informative_and_label_informative": sum(info(p) and info(q) for p, q in zip(x, y)),
           "label_informative": sum(map(info, x))}
    ordered = opts[:5] if a.question == "timing" else opts[:4]
    both = [(p, q) for p, q in zip(x, y) if p in ordered and q in ordered]
    if both:
        k = len(ordered)
        out["both_ordered_n"] = len(both)
        out["weighted_kappa_linear"] = kappa([p for p, _ in both], [q for _, q in both], ordered,
                                             [[abs(i - j) for j in range(k)] for i in range(k)])
    out["disagreements"] = [{"id": i, "label": p, "reader": q} for i, p, q in zip(ids, x, y) if p != q]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
