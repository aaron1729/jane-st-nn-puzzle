#!/usr/bin/env bash
set -euo pipefail

# Runs multiple Sinkhorn jobs in parallel batches until time budget expires.
# Writes rolling summaries and a done flag for external sync/monitor loops.

HOURS=8
BASE_SEED=3000
PREFIX="overnight"
PARALLEL=4
SLEEP_BETWEEN_BATCHES=2
STOP_ON_HASH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours) HOURS="$2"; shift 2 ;;
    --base-seed) BASE_SEED="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --sleep-between-batches) SLEEP_BETWEEN_BATCHES="$2"; shift 2 ;;
    --stop-on-hash) STOP_ON_HASH="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")"

START_TS=$(date +%s)
END_TS=$((START_TS + HOURS * 3600))

SUMMARY_CSV="${PREFIX}_summary.csv"
STATUS_TXT="${PREFIX}_status.txt"
DONE_FLAG="${PREFIX}_DONE.flag"
BEST_JSON="${PREFIX}_best.json"
RUN_LOG="${PREFIX}_driver.log"

rm -f "$DONE_FLAG"
echo "batch,job,name,seed,best_discrete,final_discrete,soft_full,hash_match" > "$SUMMARY_CSV"
{
  echo "prefix=$PREFIX"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hours=$HOURS"
  echo "parallel=$PARALLEL"
  echo "base_seed=$BASE_SEED"
} > "$STATUS_TXT"

touch "$RUN_LOG"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting driver" | tee -a "$RUN_LOG"

pick_config() {
  local slot="$1"
  case "$slot" in
    1)
      echo "--steps 900 --lr 0.03 --tau-start 1.0 --tau-end 0.030 --sinkhorn-iters 30 --lambda-ent 0.0003 --train-subsample 4096"
      ;;
    2)
      echo "--steps 900 --lr 0.03 --tau-start 1.0 --tau-end 0.028 --sinkhorn-iters 30 --lambda-ent 0.0003 --train-subsample 4096"
      ;;
    3)
      echo "--steps 700 --lr 0.02 --tau-start 1.0 --tau-end 0.030 --sinkhorn-iters 30 --lambda-ent 0.0003 --train-subsample 0"
      ;;
    4)
      echo "--steps 700 --lr 0.015 --tau-start 1.0 --tau-end 0.025 --sinkhorn-iters 30 --lambda-ent 0.0001 --train-subsample 0"
      ;;
    *)
      # Fallback for PARALLEL > 4: repeat strongest configuration family.
      echo "--steps 900 --lr 0.03 --tau-start 1.0 --tau-end 0.030 --sinkhorn-iters 30 --lambda-ent 0.0003 --train-subsample 4096"
      ;;
  esac
}

BATCH=0
while (( $(date +%s) < END_TS )); do
  BATCH=$((BATCH + 1))
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] batch=$BATCH start" | tee -a "$RUN_LOG"

  PIDS=()
  NAMES=()
  SEEDS=()

  for JOB in $(seq 1 "$PARALLEL"); do
    NAME="${PREFIX}_b$(printf "%03d" "$BATCH")_j${JOB}"
    SEED=$((BASE_SEED + BATCH * 100 + JOB))
    CFG=$(pick_config "$JOB")

    CMD="python3 solve_sinkhorn.py --device cuda --eval-every 25 --save-json ${NAME}.json --seed ${SEED} ${CFG}"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start ${NAME} seed=${SEED}" | tee -a "$RUN_LOG"
    nohup bash -lc "$CMD" > "${NAME}.log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$NAME")
    SEEDS+=("$SEED")
  done

  for PID in "${PIDS[@]}"; do
    wait "$PID"
  done

  python3 - "$SUMMARY_CSV" "$STATUS_TXT" "$BEST_JSON" "$STOP_ON_HASH" "$BATCH" "${NAMES[@]}" -- "${SEEDS[@]}" <<'PY'
import glob
import json
import os
import shutil
import sys
from datetime import datetime, timezone

summary_csv = sys.argv[1]
status_txt = sys.argv[2]
best_json = sys.argv[3]
stop_on_hash = int(sys.argv[4])
batch = int(sys.argv[5])

sep = sys.argv.index("--")
names = sys.argv[6:sep]
seeds = [int(x) for x in sys.argv[sep + 1 :]]

rows = []
hash_found = False

for job_idx, (name, seed) in enumerate(zip(names, seeds), start=1):
    path = f"{name}.json"
    best = final = soft = ""
    hash_match = False
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        best = d.get("best_seen", {}).get("discrete_mse", "")
        final = d.get("final", {}).get("discrete_mse", "")
        soft = d.get("final", {}).get("soft_mse_full", "")
        hash_match = bool(d.get("final", {}).get("hash_match", False))
    rows.append((batch, job_idx, name, seed, best, final, soft, hash_match))
    hash_found = hash_found or hash_match

with open(summary_csv, "a") as f:
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

best_file = None
best_val = float("inf")
for path in sorted(glob.glob("*_b*_j*.json")):
    try:
        with open(path) as f:
            d = json.load(f)
        v = d.get("best_seen", {}).get("discrete_mse", None)
        if v is None:
            continue
        v = float(v)
        if v < best_val:
            best_val = v
            best_file = path
    except Exception:
        continue

if best_file:
    shutil.copyfile(best_file, best_json)

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
with open(status_txt, "w") as f:
    f.write(f"updated_utc={now}\n")
    f.write(f"batch={batch}\n")
    f.write(f"best_file={best_file}\n")
    f.write(f"best_discrete={best_val if best_file else ''}\n")
    f.write(f"hash_found={hash_found}\n")

if hash_found and stop_on_hash:
    with open("STOP_EARLY.flag", "w") as f:
        f.write("hash matched\n")
PY

  if [[ -f STOP_EARLY.flag ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] hash found, stopping early" | tee -a "$RUN_LOG"
    break
  fi

  if (( $(date +%s) >= END_TS )); then
    break
  fi
  sleep "$SLEEP_BETWEEN_BATCHES"
done

echo "done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_TXT"
touch "$DONE_FLAG"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] driver complete" | tee -a "$RUN_LOG"
