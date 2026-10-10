"""Isolated100-update fork; import the actual official adaptation configuration."""

import os
from pathlib import Path
import runpy

namespace = runpy.run_path(str(Path(os.environ["INTERN_DECISION_REPO"]) / "configs/training/qwen35.py"))
trainer = namespace["trainer"]
trainer.total_epoch = None
trainer.total_step = 100
trainer.seed = 42
trainer.checkpoint_interval = 25
trainer.checkpoint_maxkeep = None
trainer.snapshot_interval = None
trainer.hf_interval = None
trainer.async_checkpoint = False
trainer.async_hf_export = False
trainer.auto_resume = False
trainer.resume_cfg = None

import quick_train_hooks

quick_train_hooks.install()
