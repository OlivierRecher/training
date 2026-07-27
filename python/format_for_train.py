"""Convert the timestamp-preserving split CSV into a training-ready JSONL.

Input rows: `userId;nRecord;cellId;ts;cellId;ts;...` (as produced by
split_sample_for_training.py, timestamps kept). `ts` is seconds since local
midnight and follows the cell_id it belongs to (see CLAUDE.md); `hour =
ts // 3600` becomes the hour-of-day signal `train_cellid_llm.ipynb` uses for
its transition-window weighting.

Each user is serialised into:
    {"conversations": [...],          # cell_id-only text, for readability/back-compat
     "user_id": "43",
     "cells": ["BSOHSL2", "BSOHSL2", ...],
     "hours": [3, 4, ...]}

`--n-train`/`--n-test` give an exact split: the first `n_train` lines go to
`<stem>_train.jsonl`, the next `n_test` to `<stem>_test.jsonl`.

Usage:
    python python/format_for_train.py cleaned.csv --n-train N --n-test M [--out DIR]
"""
import argparse
import csv
import json
from pathlib import Path

from cellid_encoding import event_hour_from_seconds

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "dataset_for_training"

DELIMITER = ";"

SYSTEM_PROMPT = ("You are an AI that generates a user's cell-visit trajectory. "
                 "Output format: User <USER_ID> | <N_RECORDS> events | "
                 "<CELL_ID> <CELL_ID> ...")
USER_PROMPT = "Here is the sequence for this user."


def parse_records(records):
    """Split a flat [cellId, ts, cellId, ts, ...] list into (cells, hours),
    dropping a stray unpaired trailing field if present."""
    pairs = len(records) // 2 * 2
    cells = records[0:pairs:2]
    hours = [event_hour_from_seconds(ts) for ts in records[1:pairs:2]]
    return cells, hours


def row_to_conversation(user_id, n_record, cells, hours):
    """Serialise one user's trajectory into a ChatML `conversations` object,
    plus the structured `cells`/`hours` fields the notebook reads directly."""
    trajectory = " ".join(cells)
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": f"User {user_id} | {n_record} events | {trajectory}"},
        ],
        "user_id": user_id,
        "cells": cells,
        "hours": hours,
    }


def convert(src, out_dir, n_train, n_test):
    """Convert `src` to two JSONL files using an exact `n_train`/`n_test`
    split (capped to however many lines are available). The first `n_train`
    lines go to `<stem>_train.jsonl`, the next `n_test` to `<stem>_test.jsonl`.
    Returns the two paths with their line counts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    with src.open(newline="") as fin:
        reader = csv.reader(fin, delimiter=DELIMITER)
        next(reader, None)  # skip the header row
        for fields in reader:
            if not fields:
                continue
            user_id = fields[0]
            n_record = fields[1] if len(fields) > 1 else ""
            cells, hours = parse_records(fields[2:])
            lines.append(json.dumps(row_to_conversation(user_id, n_record, cells, hours),
                                    ensure_ascii=False))

    if n_train + n_test > len(lines):
        print(f"warning: requested {n_train} train + {n_test} test "
              f"({n_train + n_test}) but only {len(lines)} lines available; "
              "capping to what's available")
    cut = min(n_train, len(lines))
    test_end = cut + n_test
    train_dest = out_dir / (src.stem + "_train.jsonl")
    test_dest = out_dir / (src.stem + "_test.jsonl")
    train_lines = lines[:cut]
    test_lines = lines[cut:test_end]
    train_dest.write_text("\n".join(train_lines) + "\n" if train_lines else "")
    test_dest.write_text("\n".join(test_lines) + "\n" if test_lines else "")

    return train_dest, len(train_lines), test_dest, len(test_lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path,
                        help="cleaned CSV to convert (e.g. 400_100_sample_users.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: data/dataset_for_training)")
    parser.add_argument("--n-train", type=int, required=True,
                        help="exact number of train lines to keep")
    parser.add_argument("--n-test", type=int, required=True,
                        help="exact number of test lines to keep")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source file not found: {args.source}")

    train_dest, n_train, test_dest, n_test = convert(args.source, args.out,
                                                      args.n_train, args.n_test)
    print(f"formatted -> {train_dest} ({n_train} lines)")
    print(f"formatted -> {test_dest} ({n_test} lines)")


if __name__ == "__main__":
    main()
