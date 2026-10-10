#!/usr/bin/env python3
"""Select one official DCP directory without turning auto-resume on.

auto_resume replaces checkpoint_path with the last entry in the work
directory's own .xtuner. A fresh attempt has no such entry, so the load
has to name ckpt-step-750 and keep optimizer, scheduler, data, and RNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = """_total_step = os.environ.get("TOTAL_STEP", "").strip()
if _total_step:
    trainer.total_epoch = None
    trainer.total_step = int(_total_step)
"""
BLOCK = ANCHOR + """
_load_checkpoint = os.environ.get("LOAD_CHECKPOINT", "").strip()
if _load_checkpoint:
    from xtuner.v1.train.trainer import LoadCheckpointConfig

    trainer.auto_resume = False
    trainer.resume_cfg = None
    trainer.load_checkpoint_cfg = LoadCheckpointConfig(
        checkpoint_path=_load_checkpoint,
        load_optimizer_states=True,
        load_optimizer_args=True,
        load_dataset=True,
        load_scheduler=True,
    )
    print(f"LOAD_CHECKPOINT_EXPLICIT {_load_checkpoint}", flush=True)
"""
MARKER = "LOAD_CHECKPOINT_EXPLICIT"


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"ALREADY_PATCHED {path}")
        return
    if ANCHOR not in text:
        raise SystemExit(f"total_step anchor missing in {path}")
    path.write_text(text.replace(ANCHOR, BLOCK, 1), encoding="utf-8")
    print(f"PATCHED {path}")


if __name__ == "__main__":
    main()
