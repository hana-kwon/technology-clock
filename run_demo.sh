#!/bin/bash
# Reproduce the prototype on synthetic passages (no licensed text needed).
# Requires: Python 3.9+, the Claude Code CLI (`claude`) logged in. Cost about 0.16 USD for 36 calls (2026-11-01).
# Step 1 reads 12 synthetic passages with three prompts; step 2 scores the replies against the intended labels.
# To rescore the saved replies without calling a model, skip step 1 (results/demo_*.jsonl are included).
set -e
cd "$(dirname "$0")"
if [ "$1" != "--score-only" ]; then
  for pr in timing_v1_template timing_v4 stage; do
    python3 src/read_claude.py --passages data/synthetic_passages.jsonl --prompt prompts/$pr.txt \
      --out results/demo_$pr.jsonl --workers 6
  done
fi
for pr in timing_v1_template timing_v4; do
  python3 src/score.py --replies results/demo_$pr.jsonl --labels data/synthetic_labels.csv --question timing
done
python3 src/score.py --replies results/demo_stage.jsonl --labels data/synthetic_labels.csv --question stage
