"""Score one frozen E03 diagnostic list with one checkpoint.

Each finished pair is appended to the partial JSONL and fsync'd before the
next forward. A retried process skips ids already present. This does not
train and does not rewrite the frozen list.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/home/zhaopu/intern-decision")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("INTERN_DECISION_REPO", str(ROOT / "Intern-Decision"))
os.environ.setdefault("MEDIA_ROOT", str(ROOT / "frames/id_pilot_frames_8"))
os.chdir(os.environ["INTERN_DECISION_REPO"])

from diagnose_loss_contrast import load_model, row_of, score_rows
from transformers import AutoTokenizer


def valid_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        records.append(item)
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    deadline = int(os.environ["DEADLINE_EPOCH"])
    max_forwards = int(os.environ["MAX_FORWARDS"])
    frozen = json.loads(Path(os.environ["FROZEN_PAIRS"]).read_text(encoding="utf-8"))
    partial = Path(os.environ["PARTIAL"])
    ledger = Path(os.environ["FORWARD_LEDGER"])
    model_name = os.environ["MODEL_NAME"]
    checkpoint = os.environ["CHECKPOINT"]
    status_path = Path(os.environ["STATUS_PATH"])
    done = {record["id"] for record in valid_records(partial) if record.get("model") == model_name}
    forwards = len(valid_records(ledger))
    pairs = [pair for pair in frozen["pairs"] if pair["id"] not in done]
    if os.environ.get("CONTROL_REVERSE") == "1":
        pairs = []
        for pair in frozen["pairs"]:
            if pair["pair_type"] != "positive":
                continue
            control_id = f"{pair['cap_id']}#reverse"
            if control_id in done:
                continue
            copied = dict(pair)
            copied["id"] = control_id
            copied["pair_type"] = "positive_reversed_frames"
            copied["control"] = "reversed_frames"
            copied["label"] = "yes"
            pairs.append(copied)
    if time.time() > deadline:
        raise SystemExit("BUDGET_STOP before load")
    if forwards >= max_forwards:
        raise SystemExit(f"BUDGET_STOP forwards {forwards} already at {max_forwards}")
    if not pairs:
        print(json.dumps({"model": model_name, "scored": 0, "reason": "already_complete"}), flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_CHECKPOINT"], local_files_only=True)
    model = load_model(checkpoint)
    try:
        for pair in pairs:
            if time.time() > deadline or forwards >= max_forwards:
                append_jsonl(
                    status_path,
                    {"model": model_name, "event": "budget_stop", "forwards": forwards, "remaining_ids": [item["id"] for item in pairs]},
                )
                raise SystemExit(75)
            if pair["control"] == "reversed_frames":
                row = row_of(pair["id"], pair["query"], pair["video_id"], pair["label"], pair["pair_type"], "matched")
                row["images"] = [f"{pair['video_id']}/frame_{index:02d}.png" for index in range(7, -1, -1)]
                row["control"] = "reversed_frames"
            else:
                row = row_of(
                    pair["id"], pair["query"], pair["video_id"], pair["label"], pair["pair_type"], pair["control"]
                )
            started = time.time()
            records = score_rows(model, tokenizer, [row], checkpoint)
            if len(records) != 1:
                raise SystemExit(f"expected one record for {pair['id']}")
            record = records[0]
            record.update(
                {
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "backend": "xtuner_fsdp_from_hf_nograd",
                    "cap_id": pair["cap_id"],
                    "frame_count": pair["frame_count"],
                    "library_rank": pair.get("library_rank"),
                    "control": pair["control"],
                    "mask_check_passed": record.get("label_in_model_input") is False and record.get("supervised_targets") == 1,
                    "elapsed_s": round(time.time() - started, 3),
                    "max_memory_reserved_bytes": int(__import__("torch").cuda.max_memory_reserved()),
                }
            )
            append_jsonl(partial, record)
            append_jsonl(ledger, {"id": record["id"], "model": model_name, "at": time.time()})
            forwards += 1
            print(
                json.dumps(
                    {
                        "flushed": record["id"],
                        "model": model_name,
                        "ce": record["full_vocabulary_ce"],
                        "margin": record["yes_no_margin_logit_yes_minus_no"],
                        "forwards": forwards,
                    }
                ),
                flush=True,
            )
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            return
        append_jsonl(
            status_path,
            {"model": model_name, "event": "score_error", "error": "".join(traceback.format_exception_only(type(exc), exc)).strip()},
        )
        raise
    finally:
        del model


if __name__ == "__main__":
    main()
