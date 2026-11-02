#!/bin/bash
# ============================================================
# START DATE   : 2026-10-31
# DESCRIPTION  : One LSF job = one ChronoGPT vintage for spec 70: download the weights (7.4 GB), run
#                70_contrast_clock_grid_261031.py (372 sentences), delete the weights. Grid home had 11 GB free on
#                2026-10-31, so only one vintage may sit on disk. Called by submit.sh; not meant to be run by hand.
# ============================================================
V=$1
RUN=${RUN:-~/contrast_clock}
PY=${PY:-python3}
W=$RUN/weights/$V
export HF_HOME=$RUN/hf_home
trap 'rm -rf "$W" "$HF_HOME"' EXIT
echo "=== $(date) vintage $V on $(hostname)"
if [ -e "$RUN/output/70_contrast_clock_$V.json" ]; then echo "output exists, skipping"; exit 0; fi
FREE_GB=$(df -Pk ~ | awk 'NR==2 {print int($4/1048576)}')
echo "free GB in home: $FREE_GB"
# ★ IMPORTANT: never start a 7.4 GB download without room for it
if [ "$FREE_GB" -lt 8 ]; then echo "STOP: less than 8 GB free"; exit 3; fi
$PY -c "from huggingface_hub import hf_hub_download as d; [d('manelalab/chrono-gpt-v1-$V', f, local_dir='$W') for f in ('config.json','pytorch_model.bin')]" \
  || { echo "DOWNLOAD FAILED (no internet on this node?)"; exit 2; }
ls -l $W
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
cd $RUN/source || exit 4
# ★ IMPORTANT (2026-11-01 14:1x): this script used to end with exit 0 even when python died, so LSF showed
# "Successfully completed" for the 4 vintages lost to CUDA out-of-memory (2003, 2004, 2005, 2024). Now a failed run
# that mentions "out of memory" starts again in a fresh process (up to 6 tries, 5 minutes apart; a CUDA-level OOM
# may not be recoverable inside one process), and the python exit code goes to LSF.
for TRY in 1 2 3 4 5 6; do
  ERRF=$RUN/logs/${V}_try$TRY.err
  $PY 70_contrast_clock_grid_261031.py --vintage $V --weights_dir $W --tok_dir $RUN/tokenizer --out_dir $RUN/output 2> "$ERRF"
  PRC=$?
  cat "$ERRF" >&2
  [ $PRC -eq 0 ] && break
  if [ $TRY -lt 6 ] && grep -qi "out of memory" "$ERRF"; then
    echo "OOM exit $PRC (try $TRY of 6), new process in 300 s"
    sleep 300
  else
    echo "FAILED: exit $PRC"
    break
  fi
done
echo "=== $(date) finished $V (exit $PRC)"
exit $PRC
