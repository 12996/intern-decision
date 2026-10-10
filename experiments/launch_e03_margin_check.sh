#!/bin/bash
# Bounded released-vs-HF800 margin check. Does not train and does not use GPU 2.
set -euo pipefail
ROOT=/home/zhaopu/intern-decision
OUT=$ROOT/outputs/e03-check-20261002
PY=$ROOT/.venv-xt310/bin/python
REPO=$ROOT/Intern-Decision
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
if ! flock -n 9; then
  echo "REFUSE_CHECK_LOCK"
  exit 6
fi
if pgrep -af 'e03_margin_score.py' | grep -v -e pgrep -e 'grep ' >/dev/null; then
  echo "REFUSE_SCORE_ALREADY_RUNNING"
  exit 8
fi
pick_gpu() {
  "$PY" - <<'PY'
import subprocess
apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory", "--format=csv,noheader"], text=True)
gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"], text=True)
best = None
for line in gpus.splitlines():
    if not line.strip():
        continue
    index, bus, used, free, total = [part.strip() for part in line.split(",")]
    if index == "2":
        print(f"GPU 2 EXCLUDED free={free}")
        continue
    free_i = int(free)
    total_i = int(total)
    used_i = int(used)
    print(f"GPU {index} FREE_MIB {free_i} USED_MIB {used_i}")
    if free_i >= 18000 and (best is None or free_i > best[0]):
        best = (free_i, index, used_i, total_i)
if best is None:
    raise SystemExit("REFUSE_NO_INFERENCE_GPU floor=18000")
free_i, index, used_i, total_i = best
allow = total_i - used_i - 768
fraction = allow / total_i
print(f"SELECTED_GPU {index}")
print(f"MEM_FRACTION {fraction:.4f}")
print(f"ALLOW_MIB {allow}")
PY
}
eval "$(pick_gpu | awk '/^SELECTED_GPU /{print "GPU="$2} /^MEM_FRACTION /{print "MEM_FRACTION="$2}')"
if [ "${GPU:-}" = "2" ] || [ -z "${GPU:-}" ]; then
  echo "REFUSE_GPU"
  exit 2
fi
export INTERN_DECISION_REPO=$REPO
export MEDIA_ROOT=$ROOT/frames/id_pilot_frames_8
export TOKENIZER_CHECKPOINT=$ROOT/models/Intern-Decision-2B
export FROZEN_PAIRS=$OUT/frozen_pairs.json
export FORWARD_LEDGER=$OUT/forwards.jsonl
export CUDA_VISIBLE_DEVICES=$GPU
export MEM_FRACTION
export MAX_PIXELS=16777216
export PYTHONPATH=$REPO/runtime/xtuner:$REPO:$REPO/third_party/xtuner
export XTUNER_HF_IMPL=1
export XTUNER_USE_FA3=0
export XTUNER_GC_ENABLE=1
export XTUNER_ACTIVATION_OFFLOAD=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DEADLINE_EPOCH=${DEADLINE_EPOCH:-$(( $(date +%s) + 3600 ))}
export MAX_FORWARDS=28
echo "DEADLINE_EPOCH $DEADLINE_EPOCH"
echo "SELECTED_VISIBLE_GPU $GPU"
score_one() {
  local name="$1"
  local checkpoint="$2"
  local attempt=1
  while [ "$attempt" -le 2 ]; do
    local now
    now=$(date +%s)
    if [ "$now" -ge "$DEADLINE_EPOCH" ]; then
      echo "BUDGET_STOP before $name"
      return 75
    fi
    local used
    used=$("$PY" -c 'import json,pathlib,sys
path=pathlib.Path(sys.argv[1])
n=0
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        n+=1
print(n)' "$FORWARD_LEDGER")
    if [ "$used" -ge 28 ]; then
      echo "BUDGET_STOP forwards $used"
      return 75
    fi
    eval "$(pick_gpu | awk '/^SELECTED_GPU /{print "GPU="$2} /^MEM_FRACTION /{print "MEM_FRACTION="$2}')"
    export CUDA_VISIBLE_DEVICES=$GPU
    export MEM_FRACTION
    local port
    port=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
    echo "SCORE_BEGIN $name attempt=$attempt gpu=$GPU fraction=$MEM_FRACTION port=$port"
    set +e
    "$PY" -m torch.distributed.run \
      --nproc-per-node=1 --nnodes=1 --node-rank=0 \
      --master-addr=127.0.0.1 --master-port="$port" \
      "$ROOT/e03_margin_score.py" \
      >> "$OUT/score-$name.log" 2>&1
    local code=$?
    set -e
    echo "SCORE_EXIT $name attempt=$attempt code=$code"
    local rows
    rows=$("$PY" -c 'import json,os,pathlib,sys
path=pathlib.Path(sys.argv[1]); name=sys.argv[2]; n=0
if path.exists():
    lines=path.read_text(encoding="utf-8").splitlines()
    for i,line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item=json.loads(line)
        except json.JSONDecodeError:
            if i==len(lines)-1:
                continue
            raise
        if item.get("model")==name and item.get("control")=="matched":
            n+=1
print(n)' "$OUT/scores.jsonl" "$name")
    if [ "${CONTROL_REVERSE:-0}" = "1" ]; then
      local reverses
      reverses=$("$PY" -c 'import json,pathlib,sys
path=pathlib.Path(sys.argv[1]); name=sys.argv[2]; n=0
if path.exists():
    lines=path.read_text(encoding="utf-8").splitlines()
    for i,line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item=json.loads(line)
        except json.JSONDecodeError:
            if i==len(lines)-1:
                continue
            raise
        if item.get("model")==name and item.get("control")=="reversed_frames":
            n+=1
print(n)' "$OUT/scores.jsonl" "$name")
      if [ "$reverses" -ge 4 ]; then
        return 0
      fi
    elif [ "$rows" -ge 12 ]; then
      return 0
    fi
    if [ "$code" -eq 75 ]; then
      return 75
    fi
    if ! grep -q "CUDA out of memory" "$OUT/score-$name.log"; then
      return "$code"
    fi
    attempt=$((attempt + 1))
  done
  return 1
}
export PARTIAL=$OUT/scores.jsonl
export STATUS_PATH=$OUT/score-status.jsonl
export MODEL_NAME=released
export CHECKPOINT=$ROOT/models/Intern-Decision-2B
score_one released "$CHECKPOINT"
export MODEL_NAME=hf800
export CHECKPOINT=$ROOT/runs/charades-id2b-e03-20260929-224918/20260929225053/hf-800
score_one hf800 "$CHECKPOINT"
export RELEASED_PARTIAL=$OUT/scores.jsonl
export HF800_PARTIAL=$OUT/scores.jsonl
export OUTPUT=$OUT/summary.json
"$PY" "$ROOT/e03_margin_summarize.py" | tee "$OUT/summary-status.txt"
if grep -q '"controls_needed": true' "$OUT/summary.json"; then
  now=$(date +%s)
  if [ "$now" -lt "$DEADLINE_EPOCH" ]; then
    export CONTROL_REVERSE=1
    export MODEL_NAME=released
    export CHECKPOINT=$ROOT/models/Intern-Decision-2B
    echo "REVERSE_CONTROLS_BEGIN"
    score_one released "$CHECKPOINT" || true
    unset CONTROL_REVERSE
    "$PY" "$ROOT/e03_margin_summarize.py" | tee "$OUT/summary-status.txt"
  fi
fi
echo "CHECK_DONE"
