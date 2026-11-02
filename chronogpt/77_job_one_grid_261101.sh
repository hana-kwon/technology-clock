#!/bin/bash
# ============================================================
# START DATE   : 2026-11-01
# DESCRIPTION  : One LSF job = one ChronoGPT-Instruct vintage for the reader pilot (protocol 75, step R5): download
#                the weights (7.4 GB), run 77_reader_pilot_chronogpt_grid_261101.py, delete the weights. Grid home had
#                11 GB free on 2026-11-01, so only one vintage may sit on disk. Called by submit.sh, not by hand.
#                Arguments: vintage (YYYYMMDD), mode (realtime | all), prompt files (space-separated, in input/).
#                2026-11-01: realtime reads prompt v1 (Jha et al. template) and v4 (rules added after the Claude pilot);
#                all-mode reads v4 only. Weights are downloaded once per job and used for every prompt.
# ============================================================
V=$1
MODE=${2:-realtime}
# ★ IMPORTANT: jobs 3304656-3304671 were submitted at 2026-11-01 01:09 EDT with the earlier submit.sh (no prompt
# argument, -W 3:00 realtime / 8:00 all). Defaults by mode keep those jobs inside their time limits:
# realtime = v1 + v4 (at most 61 chunks x 2), all = v4 only (230 chunks).
if [ "$MODE" = "all" ]; then DEF="75_reader_prompt_quantum_261101_v4.txt"; else DEF="75_reader_prompt_quantum_261101.txt 75_reader_prompt_quantum_261101_v4.txt"; fi
PROMPTS=${3:-$DEF}
RUN=${RUN:-~/reader_pilot}
PY=${PY:-python3}
W=$RUN/weights/$V
export HF_HOME=$RUN/hf_home
trap 'rm -rf "$W" "$HF_HOME"' EXIT
echo "=== $(date) vintage $V mode $MODE on $(hostname)"
FREE_GB=$(df -Pk ~ | awk 'NR==2 {print int($4/1048576)}')
echo "free GB in home: $FREE_GB"
# ★ IMPORTANT: never start a 7.4 GB download without room for it (spec 70 jobs use the same home folder)
if [ "$FREE_GB" -lt 8 ]; then echo "STOP: less than 8 GB free"; exit 3; fi
$PY -c "from huggingface_hub import hf_hub_download as d; [d('manelalab/chrono-gpt-instruct-v1-$V', f, local_dir='$W') for f in ('config.json','pytorch_model.bin')]" \
  || { echo "DOWNLOAD FAILED (no internet on this node?)"; exit 2; }
ls -l $W
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
cd $RUN/source || exit 4
# ★ IMPORTANT: spec 70's job script ended with exit 0 even when python died (CUDA OOM), so LSF reported
# "Successfully completed" for vintages with no output. Carry the python exit code to LSF instead.
RC=0
for PR in $PROMPTS; do
  echo "--- prompt $PR"
  # ★ IMPORTANT (2026-11-01 14:0x): spec 70's 2024 vintage died with "RuntimeError: CUDA error: out of memory" while the
  # weights were moved to the shared V100. That is a CUDA-level error, not torch.OutOfMemoryError, and the in-process
  # retry in the python file may not recover from it. So a failed run that mentions "out of memory" is started again
  # in a fresh process (up to 6 tries, 5 minutes apart). The output file is resumable: finished chunks are skipped.
  for TRY in 1 2 3 4 5 6; do
    ERRF=$RUN/logs/${V}_${MODE}_${PR%.txt}_try$TRY.err
    $PY 77_reader_pilot_chronogpt_grid_261101.py --vintage $V --mode $MODE --run_dir $RUN --weights_dir $W --prompt $PR 2> "$ERRF"
    PRC=$?
    cat "$ERRF" >&2
    [ $PRC -eq 0 ] && break
    if [ $TRY -lt 6 ] && grep -qi "out of memory" "$ERRF"; then
      echo "OOM exit $PRC (try $TRY of 6), new process in 300 s"
      sleep 300
    else
      break
    fi
  done
  [ $PRC -ne 0 ] && { RC=$PRC; echo "FAILED: prompt $PR exit $PRC"; }
done
echo "=== $(date) finished $V $MODE (exit $RC)"
exit $RC
