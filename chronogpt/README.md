# Tests of time-locked language models

Scripts for the ChronoGPT tests reported in the technical appendix, kept as they were run on a GPU cluster except that private paths were replaced. They download public weights from Hugging Face. `83_*` and `86_*` score the outputs; `83_*` also needs the licensed passages and the private project layout named in its header, so it is included for inspection rather than for rerunning here.

| File | Test |
|---|---|
| `70_contrast_clock_grid_261031.py`, `70_job_one_grid_261031.sh` | Sentence contrast across yearly vintages (372 sentences; no licensed text, can be rerun) |
| `86_contrast_clock_eval_261101.py` | Scores the sentence contrast with the rules fixed before the run |
| `77_reader_pilot_chronogpt_grid_261101.py`, `77_job_one_grid_261101.sh` | Instruction-tuned reader on transcript passages |
| `83_reader_pilot_chronogpt_eval_261101.py` | Scores the instruction-tuned reader |
| `vendor_chronogpt_*.py` | Inference code from the model repositories (MIT license); provenance in the main README |
