# The Technology Clock: prototype

Prototype for reading, from earnings-call passages, (1) the stage of use of a technology that a speaker describes as already under way and (2) any stated time until regular commercial use. The first application is quantum computing. This repository accompanies a proposal to the Organization Science special issue "AI-Enabled Frontiers in Organizational Science" (Stage 1, November 2026).

The study's transcripts are licensed (S&P Capital IQ) and are not included. Everything here runs on synthetic passages; the results from the licensed sample are reported as aggregate numbers only.

## Contents

| Path | What it is |
|---|---|
| `prompts/timing_v1_template.txt` | Timing question in the prompt structure of Jha, Qian, Weber, and Yang (2024, NBER WP 32161), only the question changed |
| `prompts/timing_v4.txt` | Timing question with three rules added after the pilot (sales or installations are not commercial use; a time range only if a time is stated; encryption-breaking dates are a different event) and the call year |
| `prompts/stage_v1.txt`, `prompts/stage_v2.txt` | Stage question: research only, pilots or trials, paid early deployments, regular commercial use, or no information. v2 adds that passing mentions and equipment supply are not a stage of use (revised after comparing v1 with model coders) |
| `prompts/qday_v1.txt` | Separate question on when public-key encryption could be broken (not used in the main measures) |
| `src/passages.py` | Splits a transcript into passages (the rule used in the study) |
| `src/read_claude.py` | Reader: one model call per passage, prompt followed by the passage |
| `src/score.py` | Reads answers and scores them against reference labels (agreement, Cohen's kappa, linearly weighted kappa on ordered categories) |
| `data/synthetic_passages.jsonl` | 12 synthetic passages from fictional companies, each built around a case that caused errors in the pilot |
| `data/synthetic_labels.csv` | Intended answers under the coding guide, with the rule that decides each case |
| `results/demo_*.jsonl`, `results/demo_score_*.json` | Reader replies and scores for the synthetic passages (run 2026-11-01) |
| `results/pilot_summary_261101.json` | Aggregate results from the licensed sample |
| `docs/coding_guide.md` | Human coding guide (draft) for the same questions |
| `chronogpt/` | Scripts for the three tests of time-locked models (ChronoGPT), as run on a GPU cluster |
| `run_demo.sh` | Runs the reader on the synthetic passages and scores it; `--score-only` rescores the saved replies |

## Run

```
bash run_demo.sh --score-only     # rescore saved replies, no model calls
bash run_demo.sh                  # call the reader again (Claude Code CLI, about 0.16 USD)
```

Requirements: Python 3.9 or later; for new reader calls, the Claude Code command-line client logged in; for `src/passages.py`, the `transformers` package (GPT-2 tokenizer).

Saved demo result (synthetic passages, 2026-11-01): template prompt 10 of 12 agree with the intended labels, revised prompt 12 of 12, stage question v1 11 of 12 and v2 12 of 12. The synthetic passages and their labels were written by the authors who wrote the rules, so the demo shows that the pipeline runs and reproduces the two template errors seen in the pilot; it is not a validation.

## Results from the licensed sample (aggregate)

See `results/pilot_summary_261101.json`. In short:

- 1,222 calls (2011 to October 2026), 17,645 passages, 2,247 passages that mention quantum computing.
- Template prompt on 230 pilot passages: the same answer on a second reading for 223 of 230; no fabricated quotations; a check of 40 informative answers against the source text found 16 of 39 correct (judged by a Claude session reading the source text, not by humans).
- Holdout of 150 new passages, 83 of them coded blind by a panel of three other models (Claude Opus 5.5, Fable 5.1, Haiku 5.5; majority of three; unanimous on 80 timing and 66 stage answers): template prompt 42 of 80 (kappa 0.24), revised prompt 74 of 81 (kappa 0.76; of its 19 timing answers the panel gave one to 14 and the same answer to 13). The revised prompt was written after reading the pilot; the labels are model labels, not human coding, which has not started; 95 of the 150 passages come from companies that also appear in the pilot.
- Stage question: the first version read passing mentions as research and reported a stage in 79 percent of holdout passages. The revised version reports one in 47 percent (timing information: 13 percent) and matches the panel on 71 of 82 (kappa 0.81), 47 of 48 within one stage when both assign a stage. It was revised after other coders' labels on many of the same passages had been seen.
- A coder from another developer (Codex) agreed with the panel on 35 of 40 timing and 36 of 40 stage answers.
- No post-date terms (for example "ChatGPT" in a call before November 30, 2022) in the 2,210 Claude replies checked on 2026-11-01.
- Time-locked models (ChronoGPT): the year-filling test and the sentence-contrast test failed criteria fixed in advance; the instruction-tuned reader followed the answer format in 0 of 61 passages for the 2024 vintage, with other vintages still running.

## Notes on reproducibility

- Reader model `claude-sonnet-5-5`, pinned by full id (the alias `sonnet` resolved to an older model on 2026-11-01). The command-line client sometimes adds calls to other models for its own purposes; these appear in `model_ids`, and every reply in the study includes the pinned model. A later version will call the model API directly and will repeat the reading with an open-weight model to report how much agreement is lost.
- Temperature cannot be set through the command-line client; agreement across two readings is reported instead.
- ChronoGPT weights are on Hugging Face (`manelalab/chrono-gpt-v1-YYYYMMDD`, `manelalab/chrono-gpt-instruct-v1-YYYYMMDD`, MIT license); `chronogpt/vendor_chronogpt_instruct_manelalab.py` is byte-identical to `ChronoGPT_instruct.py` in `manelalab/chrono-gpt-instruct-v1-20241231` (checked 2026-11-01), and `chronogpt/vendor_chronogpt_inference_manelalab.py` is `ChronoGPT_inference.py` from `manelalab/chrono-gpt-v1-20081231` with a provenance header added.
- Licensed transcripts can be rebuilt by a researcher with S&P Capital IQ access by searching transcripts for "quantum computing" and applying `src/passages.py`; whether the passage identifiers and call dates used in the study can be shared depends on the data license, which has not yet been confirmed.

## License

Code is under the MIT License (`LICENSE`). Data and documents written for this project (synthetic passages, labels, coding guide, aggregate results) are under CC BY 4.0. The vendor files in `chronogpt/` keep their own MIT license.
