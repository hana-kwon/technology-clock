"""Split an earnings-call transcript body into passages that fit a 1,792-token reader context.

Rule used in the study: paragraphs are added to a passage until their token counts sum to 1,150 GPT-2 tokens; a
paragraph longer than that is cut at sentence boundaries. The separators between joined paragraphs are not counted, so
a passage can slightly exceed 1,150 tokens (largest in the study: 1,203). The rule is kept as run so that the study's
passages are reproduced exactly. Instruction plus passage stayed within the 1,792-token context of ChronoGPT-Instruct
(largest prompt 1,564 tokens), so every reader sees the same passages.

Usage: python src/passages.py transcript.txt > passages.jsonl
The input is plain text with paragraphs separated by blank lines (licensed transcripts are not included here).
"""
import json
import re
import sys

from transformers import GPT2TokenizerFast

MAX_TOK = 1150
TOK = GPT2TokenizerFast.from_pretrained("gpt2")


def ntok(s):
    return len(TOK.encode(s))


def split_long(p):
    out, cur = [], ""
    for s in re.split(r"(?<=[.?!])\s+", p):
        if cur and ntok(cur + " " + s) > MAX_TOK:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def passages(text):
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    out, cur, cur_t = [], [], 0
    for p in (p for p in paras if p):
        for q in (split_long(p) if ntok(p) > MAX_TOK else [p]):
            t = ntok(q)
            if cur and cur_t + t > MAX_TOK:
                out.append("\n\n".join(cur))
                cur, cur_t = [], 0
            cur.append(q)
            cur_t += t
    if cur:
        out.append("\n\n".join(cur))
    return out


if __name__ == "__main__":
    for i, p in enumerate(passages(open(sys.argv[1]).read())):
        print(json.dumps({"passage": i, "n_tok": ntok(p), "text": p}))
