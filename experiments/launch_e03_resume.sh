#!/bin/bash
# Resume E03 from the official DCP at step 750 into a new attempt directory.
# Full language fine-tuning, frozen vision/projector, SwapAdamW. Not LoRA.
# Horizon stays 1000. This attempt writes only hf-1000 and ckpt-step-1000.
set -euo pipefail
ROOT=/home/zhaopu/intern-decision
PY=$ROOT/.venv-xt310/bin/python
OLD_ROOT=$ROOT/runs/charades-id2b-e03-20260929-224918
OLD_EXP=$OLD_ROOT/20260929225053
SOURCE_CKPT=$OLD_EXP/checkpoints/ckpt-step-750
HISTORY=$OLD_ROOT/scaleup_steps.jsonl
AUDIT=$ROOT/outputs/fwdbwd_wss3r_2066.json
TRAIN_JSONL=$ROOT/data/full-20260929/train.jsonl
EXPECTED_SHA=a0e8ed904d294decda2495065dab98d02ad34bbf79b43a7ac310897cd33bdde7
STAMP=$(date +%Y%m%d-%H%M%S)
WORK_DIR=$ROOT/runs/charades-id2b-e03-resume750-$STAMP
LOG=$ROOT/outputs/e03-resume750-$STAMP.log
ATTEMPT=$ROOT/outputs/e03-resume750-$STAMP-attempt.json
BUDGET=$ROOT/outputs/e03-resume750-$STAMP-budget.json
POINTER=$ROOT/outputs/e03-resume-current.path
MONITOR_STATUS=$ROOT/outputs/e03-resume750-$STAMP-monitor.json
MONITOR_LOG=$ROOT/outputs/e03-resume750-$STAMP-monitor.log
mkdir -p "$ROOT/outputs"
# Declared RESUME_LOG must receive the launcher stream when the process is detached.
exec > >(tee -a "$LOG") 2>&1
cat > "$POINTER" <<EOF
stamp=$STAMP
work_dir=$WORK_DIR
log=$LOG
attempt=$ATTEMPT
budget=$BUDGET
monitor_status=$MONITOR_STATUS
monitor_log=$MONITOR_LOG
launcher_pid=$$
started_epoch=$(date +%s)
EOF
echo "RESUME_STAMP $STAMP"
echo "RESUME_WORK_DIR $WORK_DIR"
echo "RESUME_LOG $LOG"
echo "RESUME_LAUNCHER_PID $$"

if [ -e "$WORK_DIR" ]; then
  echo "REFUSE_EXISTING_WORK_DIR $WORK_DIR"
  exit 2
fi
if pgrep -af 'xtuner.v1.train.cli.sft' | grep -v -e pgrep -e 'grep ' >/dev/null; then
  echo "REFUSE_TRAINER_ALREADY_RUNNING"
  pgrep -af 'xtuner.v1.train.cli.sft' | grep -v -e pgrep -e 'grep ' || true
  exit 8
fi

"$PY" "$ROOT/patch_e03_explicit_checkpoint.py" "$ROOT/Intern-Decision/configs/training/qwen35.py"
cp "$ROOT/scaleup_observe.py" "$ROOT/Intern-Decision/scaleup_observe.py"

export SOURCE_CKPT HISTORY TRAIN_JSONL EXPECTED_SHA OLD_EXP BUDGET
# Diagnostics stay on stderr. Stdout is one integer kilobyte count.
NEED_KB=$("$PY" - <<'PY'
import json, os, sys
from pathlib import Path
source = Path(os.environ["SOURCE_CKPT"])
hf = Path(os.environ["OLD_EXP"]) / "hf-800"
state = json.loads((source / "train_state.json").read_text(encoding="utf-8"))
if int(state["cur_step"]) != 750:
    raise SystemExit(f"REFUSE_TRAIN_STATE {state}")
for name in ("dataloader", "lr_scheduler", "rng-rank0.pt", "weights"):
    path = source / name
    if not path.exists() or path.is_symlink():
        raise SystemExit(f"REFUSE_CHECKPOINT_PART {path}")
if not (source / "weights").is_dir():
    raise SystemExit(f"REFUSE_NOT_DCP {source}")
import hashlib
hasher = hashlib.sha256()
with open(os.environ["TRAIN_JSONL"], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
digest = hasher.hexdigest()
if digest != os.environ["EXPECTED_SHA"]:
    raise SystemExit(f"REFUSE_DATA_HASH {digest}")
sched = __import__("torch").load(source / "lr_scheduler", map_location="cpu", weights_only=False)
if not isinstance(sched, dict):
    raise SystemExit(f"REFUSE_SCHEDULER_TYPE {type(sched).__name__}")
milestones = [int(item) for item in list(sched.get("_milestones") or [])]
last_epoch = int(sched.get("last_epoch"))
children = [child for child in (sched.get("_schedulers") or []) if isinstance(child, dict)]
tmax = [int(child["T_max"]) for child in children if "T_max" in child]
# Saved scheduler is the original 30-step warmup plus 970-step cosine, at step 750.
if milestones != [30] or last_epoch != 750 or 970 not in tmax:
    raise SystemExit(
        f"REFUSE_SCHEDULER_HORIZON milestones={milestones} last_epoch={last_epoch} T_max={tmax}"
    )
print(
    f"SCHEDULER_HORIZON milestones={milestones} last_epoch={last_epoch} T_max={tmax}",
    file=sys.stderr,
    flush=True,
)

def du(path):
    total = 0
    if path.is_file():
        return path.stat().st_size
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            item = Path(dirpath) / name
            if not item.is_symlink():
                total += item.stat().st_size
    return total

hf_bytes = du(hf)
dcp_bytes = du(source)
if hf_bytes <= 0 or dcp_bytes <= 0:
    raise SystemExit(f"REFUSE_CHECKPOINT_SIZE hf={hf_bytes} dcp={dcp_bytes}")
reserve = 50 * 1024 ** 3
uncertainty = 1024 ** 3
# One final HF, one final DCP, one HF-sized transient copy, and 1GiB uncertainty.
writes = hf_bytes + dcp_bytes + hf_bytes + uncertainty
need = writes + reserve
avail = os.statvfs("/home").f_bavail * os.statvfs("/home").f_frsize
print(
    f"RESUME_BUDGET hf_bytes={hf_bytes} dcp_bytes={dcp_bytes} writes={writes} reserve={reserve} avail={avail} need={need}",
    file=sys.stderr,
    flush=True,
)
if avail < need:
    raise SystemExit(f"REFUSE_RESUME_DISK avail={avail} need={need}")
payload = {
    "hf_bytes": hf_bytes,
    "dcp_bytes": dcp_bytes,
    "writes": writes,
    "reserve": reserve,
    "uncertainty": uncertainty,
    "need": need,
    "available": avail,
    "scheduler": {"milestones": milestones, "last_epoch": last_epoch, "T_max": tmax},
}
Path(os.environ["BUDGET"]).write_text(json.dumps(payload) + "\n", encoding="utf-8")
print((need + 1023) // 1024)
PY
)
if [[ ! "$NEED_KB" =~ ^[0-9]+$ ]]; then
  echo "REFUSE_NEED_KB_NOT_INTEGER"
  printf 'NEED_KB_RAW=%q\n' "$NEED_KB"
  exit 9
fi
echo "RESUME_DISK_NEED_KB $NEED_KB"

# Official Trainer._maybe_save treats checkpoint_interval as:
#   None: DCP disabled
#   -1: save only at total_step
#   N > 0: save when step % N == 0 or step == total_step
# qwen35.py assigns checkpoint_interval from DCP_INTERVAL after the
# CHECKPOINT_INTERVAL constructor read. 250 is that positive trigger.
# From step 750 the only new multiple through 1000 is step 1000.
# HF_INTERVAL=1000 saves the HF export on that same step. Zero or an empty
# DCP_INTERVAL becomes None in the patch and would silently disable DCP.
export WORK_DIR
export AUTO_RESUME=0
export LOAD_CHECKPOINT=$SOURCE_CKPT
export SCALEUP_HISTORY_JSONL=$HISTORY
export SCALEUP_HISTORY_MAX_STEP=750
export SKIP_HF_STEPS=800,900
export TOTAL_STEP=1000
export HF_INTERVAL=1000
export DCP_INTERVAL=250
export CHECKPOINT_INTERVAL=250
export SAMPLE_MAX_LENGTH=2066
export PACK_MAX_LENGTH=2066
export AUDIT_JSON=$AUDIT
export TRAIN_JSONL
export MEDIA_ROOT=$ROOT/frames/id_pilot_frames_8
export MODEL_PATH=$ROOT/models/Intern-Decision-2B
export EXCLUDED_GPUS=2
export RESUME_DISK_NEED_KB=$NEED_KB
export REQUIRED_MARGIN_MIB=1024
python3 - "$ATTEMPT" "$STAMP" "$WORK_DIR" "$LOG" "$SOURCE_CKPT" "$BUDGET" <<'PY'
import json, sys
from pathlib import Path
budget = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8"))
payload = {
    "stamp": sys.argv[2],
    "work_dir": sys.argv[3],
    "log": sys.argv[4],
    "source_checkpoint": sys.argv[5],
    "started_epoch": 0,
    "launcher_pid": 0,
    "total_step": 1000,
    "resume_from_step": 750,
    "max_new_updates": 250,
    "wall_clock_limit_s": 12 * 3600,
    "method": "full_language_finetune_frozen_vision_projector_swap_adamw",
    "seed": 42,
    "lr": 2e-6,
    "weight_decay": 0.01,
    "warmup_steps": 30,
    "cosine_steps": 970,
    "hf_interval": 1000,
    "dcp_interval": 250,
    "checkpoint_interval": 250,
    "checkpoint_interval_trigger": "positive modulo; None would disable DCP",
    "skip_hf_steps": [800, 900],
    "history_max_step": 750,
    "hf_bytes": budget["hf_bytes"],
    "dcp_bytes": budget["dcp_bytes"],
    "best_model": False,
    "best_model_reason": "no retrieval-validated selection; cross-entropy does not make the final weights best",
    "retention_policy": {
        "prospective": True,
        "cleanup_authorized": False,
        "keep_existing_hf_steps": [100, 200, 300, 400, 500, 600, 700, 800],
        "keep_source_dcp": sys.argv[5],
        "source_dcp_retained_until": "final ckpt-step-1000 has weights, dataloader, lr_scheduler, train_state.json and rng-rank0.pt",
        "this_attempt_saves": ["hf-1000", "ckpt-step-1000"],
        "does_not_save": ["hf-800", "hf-900"],
        "automatic_historical_deletion": False,
    },
    "checkpoint_on_signal": False,
    "signal_note": "Official trainer.py has no SIGTERM handler. A stop does not write a new checkpoint.",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("ATTEMPT_WRITTEN", sys.argv[1], flush=True)
PY
python3 - "$ATTEMPT" "$$" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["launcher_pid"] = int(sys.argv[2])
payload["started_epoch"] = int(__import__("time").time())
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
export E03_ATTEMPT=$ATTEMPT
E03_HF_BYTES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["hf_bytes"])' "$BUDGET")
E03_DCP_BYTES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["dcp_bytes"])' "$BUDGET")
export E03_HF_BYTES E03_DCP_BYTES
export E03_MONITOR_STATUS=$MONITOR_STATUS
# Separate session: this monitor must survive launcher signals and an SSH hangup.
setsid "$PY" -u "$ROOT/e03_resume_monitor.py" >> "$MONITOR_LOG" 2>&1 < /dev/null &
MONITOR_PID=$!
echo "$MONITOR_PID" > "$ROOT/outputs/e03-resume750-$STAMP-monitor.pid"
python3 - "$ATTEMPT" "$MONITOR_PID" "$MONITOR_STATUS" "$MONITOR_LOG" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["monitor_pid"] = int(sys.argv[2])
payload["monitor_status"] = sys.argv[3]
payload["monitor_log"] = sys.argv[4]
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
echo "RESUME_MONITOR_PID $MONITOR_PID"
echo "RESUME_MONITOR_STATUS $MONITOR_STATUS"
echo "LAUNCHING $WORK_DIR from $SOURCE_CKPT"
echo "LAUNCH_INTERVALS HF_INTERVAL=$HF_INTERVAL DCP_INTERVAL=$DCP_INTERVAL CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL SKIP_HF_STEPS=$SKIP_HF_STEPS"
exec bash "$ROOT/launch_charades_scaleup.sh"
