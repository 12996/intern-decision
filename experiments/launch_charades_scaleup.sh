#!/bin/bash
# E03 Charades run: 1000 optimizer steps from the released model.
# HF checkpoints every HF_INTERVAL steps. Official DCP checkpoints,
# including optimizer, scheduler, dataloader and trainer step, every
# DCP_INTERVAL steps. GPU 2 is reserved.
set -euo pipefail
ROOT=/home/zhaopu/intern-decision
REPO=$ROOT/Intern-Decision
PY=$ROOT/.venv-xt310/bin/python
AUDIT=${AUDIT_JSON:-$ROOT/outputs/fwdbwd_largest.json}
TRAIN_JSONL=${TRAIN_JSONL:-$ROOT/data/full-20260929/train.jsonl}
MEDIA_ROOT=${MEDIA_ROOT:-$ROOT/frames/id_pilot_frames_8}
mkdir -p "$ROOT/runs"
exec 9>"$ROOT/runs/.charades-pilot.lock"
if ! flock -n 9; then
  echo "REFUSE_LAUNCH_LOCK"
  exit 6
fi
TRAINER_MATCHES=$(pgrep -af 'xtuner.v1.train.cli.sft' | grep -v -e pgrep -e 'grep ' || true)
if [ -n "$TRAINER_MATCHES" ]; then
  echo "REFUSE_TRAINER_ALREADY_RUNNING"
  echo "$TRAINER_MATCHES"
  exit 8
fi
AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
echo "MEM_AVAILABLE_KB $AVAIL_KB"
if [ "$AVAIL_KB" -lt $((32 * 1024 * 1024)) ]; then
  echo "REFUSE_LOW_RAM $AVAIL_KB"
  exit 7
fi
DISK_KB=$(df -Pk /home | awk 'NR==2 {print $4}')
if [ -n "${RESUME_DISK_NEED_KB:-}" ]; then
  # Remaining-checkpoint budget from the E03 resume launcher.
  # The fresh-run 260GiB estimate stays in force when this variable is unset.
  NEED_KB=$RESUME_DISK_NEED_KB
  echo "RESUME_DISK_GATE NEED_KB $NEED_KB"
else
  # 10 HF shards at 5 GiB, 4 DCP states at 40 GiB, plus 50 GiB headroom.
  NEED_KB=$(( (50 + 50 + 160) * 1024 * 1024 ))
fi
echo "DISK_AVAIL_KB $DISK_KB NEED_KB $NEED_KB"
if [ "$DISK_KB" -lt "$NEED_KB" ]; then
  echo "REFUSE_DISK_HEADROOM avail=$DISK_KB need=$NEED_KB"
  exit 9
fi
if [ ! -f "$AUDIT" ] || [ ! -f "$TRAIN_JSONL" ]; then
  echo "REFUSE_MISSING_INPUT audit=$AUDIT train=$TRAIN_JSONL"
  exit 3
fi
read -r LOSS_OK ADAM_OK STEP_OK VARIANT PEAK_RESERVED < <("$PY" - "$AUDIT" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(
    "yes" if payload.get("loss_finite") is True else "no",
    "yes" if payload.get("adam_states_fit") is True else "no",
    "yes" if payload.get("optimizer_step") is True else "no",
    payload.get("adam_variant") or "missing",
    int(payload["peak_reserved_bytes"]),
)
PY
)
if [ "$LOSS_OK" != "yes" ] || [ "$ADAM_OK" != "yes" ] || [ "$STEP_OK" != "yes" ]; then
  echo "REFUSE_AUDIT loss_finite=$LOSS_OK adam_states_fit=$ADAM_OK optimizer_step=$STEP_OK variant=$VARIANT"
  exit 4
fi
if [ "$VARIANT" = "swap_adamw" ]; then
  SWAP_OPTIMIZER=1
elif [ "$VARIANT" = "adamw" ]; then
  SWAP_OPTIMIZER=0
else
  echo "REFUSE_UNKNOWN_ADAM_VARIANT $VARIANT"
  exit 4
fi
echo "ADAM_VARIANT $VARIANT SWAP_OPTIMIZER $SWAP_OPTIMIZER"
MARGIN_MIB=${REQUIRED_MARGIN_MIB:-1024}
REQUIRED=$((PEAK_RESERVED / 1024 / 1024 + MARGIN_MIB))
echo "AUDIT_JSON $AUDIT"
echo "AUDIT_PEAK_RESERVED_BYTES $PEAK_RESERVED"
echo "REQUIRED_MARGIN_MIB $MARGIN_MIB"
AUTO_EXCLUDE=$("$PY" - <<'PY'
import subprocess
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name", "--format=csv,noheader"],
    text=True,
)
gpus = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
    text=True,
)
index_of = {}
for line in gpus.splitlines():
    if not line.strip():
        continue
    index, bus = [part.strip() for part in line.split(",", 1)]
    index_of[bus.lower()] = index
excluded = []
for line in apps.splitlines():
    if "diagnose_loss_contrast.py" not in line:
        continue
    bus, pid, _name = [part.strip() for part in line.split(",", 2)]
    index = index_of.get(bus.lower())
    if index is not None:
        excluded.append(index)
print(",".join(excluded))
PY
)
EXCLUDED_GPUS="${EXCLUDED_GPUS:+$EXCLUDED_GPUS,}$AUTO_EXCLUDE"
echo "EXCLUDED_GPUS ${EXCLUDED_GPUS:-none}"
BEST=""
BEST_FREE=0
for GPU in 0 1 3 4 5 6 7; do
  case ",${EXCLUDED_GPUS:-}," in
    *",$GPU,"*) echo "GPU $GPU EXCLUDED"; continue ;;
  esac
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
  FREE=${FREE// /}
  echo "GPU $GPU FREE_MIB $FREE REQUIRED_MIB $REQUIRED"
  if [ "$FREE" -ge "$REQUIRED" ] && [ "$FREE" -gt "$BEST_FREE" ]; then
    BEST=$GPU
    BEST_FREE=$FREE
  fi
done
if [ -z "$BEST" ]; then
  echo "REFUSE_NO_PERMITTED_GPU required=$REQUIRED reserved=2"
  exit 5
fi
echo "SELECTED_GPU $BEST FREE_MIB $BEST_FREE"
cp "$ROOT/scaleup_observe.py" "$REPO/scaleup_observe.py"
"$PY" "$ROOT/patch_train_log_txt.py" "$REPO"
"$PY" "$ROOT/patch_pilot_budget.py" "$REPO"
"$PY" "$ROOT/patch_checkpoint_interval.py" "$REPO/configs/training/qwen35.py"
"$PY" "$ROOT/patch_scaleup_metadata.py" "$REPO/src/inputs/masked.py"
"$PY" - "$REPO/configs/training/qwen35.py" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
updated = text.replace("import pilot_step_audit\n", "").replace("pilot_step_audit.install()\n", "")
if updated != text:
    path.write_text(updated, encoding="utf-8")
    print(f"REMOVED_PILOT_STEP_AUDIT_HOOK {path}")
else:
    print(f"NO_PILOT_STEP_AUDIT_HOOK {path}")
if "scaleup_observe.install()" not in updated:
    raise SystemExit("scaleup observer hook missing")
if "pilot_step_audit.install()" in updated:
    raise SystemExit("pilot step audit would double-log optimizer steps")
PY
if [ "${AUTO_RESUME:-0}" = "1" ]; then
  if [ -z "${WORK_DIR:-}" ] || [ ! -d "$WORK_DIR" ]; then
    echo "REFUSE_RESUME_WORK_DIR ${WORK_DIR:-}"
    exit 2
  fi
else
  STAMP=$(date +%Y%m%d-%H%M%S)
  WORK_DIR=${WORK_DIR:-$ROOT/runs/charades-id2b-e03-$STAMP}
  if [ -e "$WORK_DIR" ]; then
    echo "REFUSE_EXISTING_WORK_DIR $WORK_DIR"
    exit 2
  fi
fi
export MODEL_PATH=${MODEL_PATH:-$ROOT/models/Intern-Decision-2B}
export DATA_PATH=$TRAIN_JSONL
export MEDIA_ROOT
export WORK_DIR
export PYTHON_BIN=$PY
export XTUNER_PATH=$REPO/third_party/xtuner
export PYTHONPATH=$REPO/runtime/xtuner:$REPO:$XTUNER_PATH${PYTHONPATH:+:$PYTHONPATH}
export NPROC_PER_NODE=1
export GLOBAL_BATCH_SIZE=1
export TOTAL_EPOCH=1
export TOTAL_STEP=${TOTAL_STEP:-1000}
export SWAP_OPTIMIZER
export LR=2e-6
export WEIGHT_DECAY=0.01
export WARMUP_RATIO=0.03
export LR_MIN=2e-7
export LOSS_MODE=chunk
export LOSS_CHUNK_SIZE=1024
export LOSS_REDUCTION=token
export SAMPLE_MAX_LENGTH=${SAMPLE_MAX_LENGTH:-2064}
export PACK_MAX_LENGTH=${PACK_MAX_LENGTH:-$SAMPLE_MAX_LENGTH}
export MAX_PIXELS=${MAX_PIXELS:-16777216}
export DATALOADER_NUM_WORKERS=0
export PACK_WORKERS=1
export MODEL_COMPILE=0
export RECOMPUTE_RATIO=1.0
export VISION_RECOMPUTE_RATIO=0.0
export CUDA_VISIBLE_DEVICES=$BEST
export XTUNER_HF_IMPL=1
export XTUNER_USE_FA3=0
export XTUNER_GC_ENABLE=1
export XTUNER_ACTIVATION_OFFLOAD=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_INTERVAL=${HF_INTERVAL:-100}
export DCP_INTERVAL=${DCP_INTERVAL:-250}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-$HF_INTERVAL}
export AUTO_RESUME=${AUTO_RESUME:-0}
cd "$REPO"
echo "LAUNCH_WORK_DIR $WORK_DIR"
echo "LAUNCH_GPU $BEST"
echo "LAUNCH_TOTAL_STEP $TOTAL_STEP"
echo "LAUNCH_HF_INTERVAL $HF_INTERVAL"
echo "LAUNCH_DCP_INTERVAL $DCP_INTERVAL"
bash scripts/train.sh
