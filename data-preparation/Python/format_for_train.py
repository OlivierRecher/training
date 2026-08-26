"""Convert a cleaned sample CSV into an Unsloth-ready training set.

Unsloth trains a *language model*, so it expects a chat/instruction schema —
not the hundreds of `cellId` columns the cleaned CSV carries. Feeding the wide
CSV directly makes Unsloth read the whole `;`-separated header as ONE column
name and fail with "Could not auto-detect column roles".

**Intentionally an unconditioned generative task, not a supervised
prediction task**: the user turn is a fixed prompt (no per-example input), and
the assistant turn carries the user's *entire* cellId trajectory in one go.
The model is trained to generate a full, plausible trajectory from scratch —
not to predict a continuation from a given prefix. (An earlier version of this
script explored a prefix/continuation supervised framing; it was reverted in
favour of this simpler unconditioned-generation framing.)

This script serialises each user's trajectory into a single string, wraps it
as the assistant turn of an OpenAI/ChatML `conversations` list, and writes one
JSON object per line (JSONL):

    {"conversations": [
        {"role": "system", "content": "You are an AI that generates user event sequences..."},
        {"role": "user", "content": "Here is the sequence for this user."},
        {"role": "assistant", "content": "User 43 | 20 events | BSOHSL2 ..."}
    ]}

Timestamps carry no information for training and were already dropped by
split_sample_for_training.py, so each trajectory is just its cellId
sequence — shorter sequences mean fewer tokens per example, which speeds up
training.

`--n-train`/`--n-test` give an exact split (no randomness in the split size):
the first `n_train` lines go to `<stem>_train.jsonl`, the next `n_test` to
`<stem>_test.jsonl` — used by `make data-train`, where the sampling step
already picked exactly `n_train + n_test` users. `--all` instead converts the
whole input into a single `<stem>.jsonl` — for sources that are *already* one
split (e.g. `400_100_sample_users_train_padded.csv`, or the per-split CSVs
`split_sample_for_training.py --separate` writes). JSONL sidesteps every
delimiter/ragged-row issue. Load them in Unsloth with:

    from datasets import load_dataset
    ds = load_dataset("json", data_files={
        "train": "400_100_sample_users_train.jsonl",
        "test":  "400_100_sample_users_test.jsonl",
    })

Input is the cleaned CSV produced by split_sample_for_training.py
(`;`-separated, with the userId/nRecord/cellId header). Source path is a CLI
argument so any file with that structure works.

Usage:
    python Python/format_for_train.py cleaned.csv --n-train N --n-test M [--out DIR]
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "Database" / "dataset_for_training"

DELIMITER = ";"

SYSTEM_PROMPT = ("You are an AI that generates a user's cell-visit trajectory. "
                 "Output format: User <USER_ID> | <N_RECORDS> events | "
                 "<CELL_ID> <CELL_ID> ...")
USER_PROMPT = "Here is the sequence for this user."


def row_to_text(user_id: str, n_record: str, records: list[str]) -> str:
    """Serialise one user's cellId trajectory into a single string."""
    trajectory = " ".join(records)
    return f"User {user_id} | {n_record} events | {trajectory}"


def read_conversations(src: Path) -> list[str]:
    """Read the cleaned CSV `src` (header + one user per row) and return one
    serialised JSON conversation string per user."""
    lines = []
    with src.open(newline="") as fin:
        reader = csv.reader(fin, delimiter=DELIMITER)
        next(reader, None)  # skip the header row
        for fields in reader:
            if not fields:
                continue
            user_id = fields[0]
            n_record = fields[1] if len(fields) > 1 else ""
            records = fields[2:]
            sequence = row_to_text(user_id, n_record, records)
            conversation = {
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                    {"role": "assistant", "content": sequence},
                ]
            }
            lines.append(json.dumps(conversation, ensure_ascii=False))
    return lines


def write_jsonl(lines: list[str], dest: Path) -> Path:
    dest.write_text("\n".join(lines) + "\n" if lines else "")
    return dest


def convert_all(src: Path, out_dir: Path) -> tuple[Path, int]:
    """Convert every row of `src` into a single `<stem>.jsonl` — for inputs that
    are already one split (e.g. `..._train_padded.csv`), so no cutting is needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = read_conversations(src)
    return write_jsonl(lines, out_dir / (src.stem + ".jsonl")), len(lines)


def convert(src: Path, out_dir: Path, n_train: int, n_test: int) -> tuple[Path, int, Path, int]:
    """Convert `src` to two JSONL files using an exact `n_train`/`n_test` split
    (capped to however many lines are available). The first `n_train` lines go
    to `<stem>_train.jsonl`, the next `n_test` to `<stem>_test.jsonl`. Returns
    the two paths with their line counts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = read_conversations(src)

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
    write_jsonl(train_lines, train_dest)
    write_jsonl(test_lines, test_dest)

    return train_dest, len(train_lines), test_dest, len(test_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path,
                        help="cleaned CSV to convert (e.g. 400_100_sample_users.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: Database/dataset_for_training)")
    parser.add_argument("--n-train", type=int,
                        help="exact number of train lines to keep (omit with --all)")
    parser.add_argument("--n-test", type=int,
                        help="exact number of test lines to keep (omit with --all)")
    parser.add_argument("--all", action="store_true",
                        help="convert the whole file into a single <stem>.jsonl "
                             "(for inputs that are already one split)")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source file not found: {args.source}")
    if args.all:
        if args.n_train is not None or args.n_test is not None:
            parser.error("--all cannot be combined with --n-train/--n-test")
    elif args.n_train is None or args.n_test is None:
        parser.error("--n-train and --n-test are required unless --all is given")

    if args.all:
        dest, n_lines = convert_all(args.source, args.out)
        print(f"formatted -> {dest} ({n_lines} lines)")
        return

    train_dest, n_train, test_dest, n_test = convert(args.source, args.out,
                                                      args.n_train, args.n_test)
    print(f"formatted -> {train_dest} ({n_train} lines)")
    print(f"formatted -> {test_dest} ({n_test} lines)")


if __name__ == "__main__":
    main()
