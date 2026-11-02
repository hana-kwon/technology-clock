"""Run the reader: one call per passage, the prompt followed by the passage (prompt structure of Jha, Qian, Weber, and
Yang, 2024, NBER WP 32161: passage, fixed question, closed answer list, one-sentence explanation).

Uses the Claude Code command-line client (`claude -p`) from an empty working directory, with no tools and no settings
files, so that only the prompt and the passage reach the model. Model id is pinned (an alias resolved to an older
model on 2026-11-01). Output: one JSON line per passage with the raw reply, the model ids reported by the client, and
the reported cost.

Usage: python src/read_claude.py --passages data/synthetic_passages.jsonl --prompt prompts/timing_v4.txt \
           --out results/demo_timing_v4.jsonl [--model claude-sonnet-5-5] [--workers 4]
Passages need fields: id, call_year, text. Finished ids in --out are skipped (resumable).
"""
import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EMPTY = Path(tempfile.mkdtemp(prefix="reader_"))


def ask(prompt, text, call_year, model):
    msg = f"{prompt.replace('{call_year}', str(call_year))}\n\n{text}"
    cmd = ["claude", "-p", "--model", model, "--setting-sources", "", "--tools", "", "--output-format", "json",
           "--system-prompt", "Answer the user's request."]
    p = None
    for attempt in range(3):
        p = subprocess.run(cmd, input=msg, capture_output=True, text=True, cwd=EMPTY, timeout=300)
        try:
            j = json.loads(p.stdout)
            if not j.get("is_error"):
                return j
        except json.JSONDecodeError:
            pass
        time.sleep(5 * (attempt + 1))
    return {"is_error": True, "result": (p.stdout[-500:] + p.stderr[-500:]) if p else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    prompt = Path(a.prompt).read_text().strip()
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in out.open() if not json.loads(l)["is_error"]} if out.exists() else set()
    rows = [json.loads(l) for l in open(a.passages)]
    rows = [r for r in rows if r["id"] not in done]

    def one(r):
        j = ask(prompt, r["text"], r["call_year"], a.model)
        return {"id": r["id"], "prompt": Path(a.prompt).name, "model": a.model,
                "model_ids": list((j.get("modelUsage") or {}).keys()), "is_error": bool(j.get("is_error")),
                "reply": j.get("result", ""), "cost_usd": j.get("total_cost_usd"),
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        for res in ex.map(one, rows):
            f.write(json.dumps(res) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
