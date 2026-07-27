"""Clean the sample_for_training day-file, then keep a random sample of users
split into an exact train/test count -- with NO filtering of any kind.

The raw day-file is `;`-separated. Columns 0-6 are metadata we drop except
`userId` (0). Column 7 is `nRecord`. From column 8 the row alternates
`cellId, timestamp, cellId, timestamp, ...` -- the timestamp (seconds since
local midnight, see CLAUDE.md) follows the cell_id it belongs to, and is KEPT
here (unlike the Stage_2A pipeline this is adapted from, which dropped it):
`format_for_train.py` turns it into an hour-of-day signal.

The train/test assignment is **stratified by `nRecord`** (sequence length):
users are grouped into quantile bins of `nRecord` and each bin is split
train/test in the same proportion, via `sklearn.model_selection.train_test_split`
-- a plain positional split can otherwise produce train/test sets with very
different sequence-length distributions if row order in the source file
correlates with `nRecord`.

The sampled file gets a header row and is written to
`data/dataset_for_training/` as `{n_train}_{n_test}_sample_users.csv`, with
all train rows first followed by all test rows -- `format_for_train.py`
relies on this ordering to recover the two splits by position.

Usage:
    python python/split_sample_for_training.py [source.csv] [--out DIR] \
        [--n-train 400] [--n-test 100] [--seed 42]
"""
import argparse
import csv
import random
import statistics
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split

DEFAULT_SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "data" / "sample_for_training" / "2014-03-12_sample_for_training.csv"
DEFAULT_OUT = ROOT / "data" / "dataset_for_training"

DEFAULT_N_TRAIN = 400
DEFAULT_N_TEST = 100

DELIMITER = ";"


def parse_line(fields):
    """Keep col 0 (userId), col 7 (nRecord), and every (cellId, timestamp)
    pair from col 8 onward -- both halves kept, unlike the timestamp-dropping
    original. Trailing empty fields (from a trailing ';') are dropped."""
    kept = [fields[0]] + fields[7:8] + fields[8:]
    while kept and kept[-1] == "":
        kept.pop()
    return kept


def build_header(n_events):
    """['userId', 'nRecord', 'cellId', 'ts', 'cellId', 'ts', ...]."""
    return ["userId", "nRecord"] + ["cellId", "ts"] * n_events


def sample_rows(src, n, rng):
    """Parse every row of `src` (no filtering) and return a random sample of
    at most `n` of them (parse_line output). If `src` has fewer than `n`
    rows, all of them are returned."""
    kept = []
    with src.open(newline="") as f:
        for fields in csv.reader(f, delimiter=DELIMITER):
            if not fields:
                continue
            kept.append(parse_line(fields))

    if len(kept) <= n:
        return kept
    return rng.sample(kept, n)


def quantile_bins(n_records, max_bins=10):
    """Group `n_records` into up to `max_bins` quantile bins, for use as a
    stratification label."""
    n_bins = max(2, min(max_bins, len(n_records) // 20, len(set(n_records))))
    return list(pd.qcut(n_records, q=n_bins, labels=False, duplicates="drop"))


def stratified_train_test_split(rows, n_train, n_test, seed):
    """Split `rows` into train/test, stratified by each row's `nRecord`
    (index 1). If fewer rows are available than `n_train + n_test`, all rows
    are kept and the counts are scaled down proportionally."""
    n_records = [int(row[1]) for row in rows]
    bins = quantile_bins(n_records)

    total_requested = n_train + n_test
    if len(rows) < total_requested:
        n_test_actual = round(len(rows) * n_test / total_requested)
        n_train_actual = len(rows) - n_test_actual
    else:
        n_train_actual, n_test_actual = n_train, n_test

    return train_test_split(rows, train_size=n_train_actual, test_size=n_test_actual,
                            stratify=bins, random_state=seed)


def log_length_balance(train_rows, test_rows):
    """Print train/test sequence-length (nRecord) mean/median and a
    Mann-Whitney U test, to check the stratified split balanced them."""
    train_lens = [int(row[1]) for row in train_rows]
    test_lens = [int(row[1]) for row in test_rows]
    print(f"train length: mean={statistics.mean(train_lens):.1f} "
          f"median={statistics.median(train_lens):.1f} (n={len(train_lens)})")
    print(f"test length:  mean={statistics.mean(test_lens):.1f} "
          f"median={statistics.median(test_lens):.1f} (n={len(test_lens)})")
    u_stat, p_value = mannwhitneyu(train_lens, test_lens)
    verdict = "no significant difference" if p_value > 0.05 else "SIGNIFICANT difference"
    print(f"Mann-Whitney U={u_stat:.1f} p={p_value:.4g} ({verdict} at alpha=0.05)")


def split_sample(src, out_dir, n_train, n_test, rng, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{n_train}_{n_test}_sample_users.csv"

    pool = sample_rows(src, n_train + n_test, rng)
    train_rows, test_rows = stratified_train_test_split(pool, n_train, n_test, seed)
    rows = train_rows + test_rows

    widest_fields = max((len(row) - 2 for row in rows), default=0)
    header = build_header(widest_fields // 2)

    with dest.open("w", newline="") as fout:
        writer = csv.writer(fout, delimiter=DELIMITER)
        writer.writerow(header)
        writer.writerows(rows)

    return dest, train_rows, test_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SRC,
                        help="raw day-file to split (default: 2014-03-12_sample_for_training.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: data/dataset_for_training)")
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN,
                        help=f"number of train users to keep (default: {DEFAULT_N_TRAIN})")
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST,
                        help=f"number of test users to keep (default: {DEFAULT_N_TEST})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed for the random sample and stratified split (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source file not found: {args.source}")

    rng = random.Random(args.seed)
    dest, train_rows, test_rows = split_sample(args.source, args.out, args.n_train, args.n_test,
                                               rng, args.seed)
    kept = len(train_rows) + len(test_rows)
    n_requested = args.n_train + args.n_test
    if kept < n_requested:
        print(f"warning: only {kept} users available in the source file "
              f"(requested {n_requested} = {args.n_train} train + {args.n_test} test)")
    print(f"sampled -> {dest} ({len(train_rows)} train + {len(test_rows)} test)")
    log_length_balance(train_rows, test_rows)


if __name__ == "__main__":
    main()
