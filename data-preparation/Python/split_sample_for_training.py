"""Clean the sample_for_training day-file, then keep a random sample of users
split into an exact train/test count — with NO filtering of any kind.

The raw day-file is `;`-separated. Columns 1-6 are metadata we drop. Columns
8, 10, 12, ... hold cellId; the interleaved columns 9, 11, 13, ... hold a
timestamp that carries no information for training and is dropped. Every user
in the source file is eligible; the only operation is picking a random sample
of `--n-train + --n-test` of them.

The train/test assignment is **stratified by `nRecord`** (sequence length):
users are grouped into quantile bins of `nRecord` and each bin is split
train/test in the same proportion, via `sklearn.model_selection.train_test_split`.
Without this, a plain positional split (first N rows = train, rest = test) can
produce train/test sets with significantly different sequence-length
distributions purely from the order rows happen to appear in the source file —
this happened in practice on `2014-03-12_sample_for_training.csv`, whose row
order correlates with `nRecord` (Mann-Whitney p≈3e-5 between train/test length
before this fix).

The sampled file gets a header row and is written to
`Database/dataset_for_training/` as `{n_train}_{n_test}_sample_users.csv`,
with all train rows first followed by all test rows — `format_for_train.py`
relies on this ordering to recover the two splits by position.

`--separate` writes one CSV per split instead
(`{stem}_{n_train}_train.csv` / `{stem}_{n_test}_test.csv`), so no positional
convention is needed downstream. `--train-subset N` (repeatable) additionally
writes `{stem}_{N}_train.csv`, a stratified subset of the train rows — for
comparing the effect of training-set size on the same held-out test set. Every
subset is drawn from the train rows only, so the test users never leak in.

Usage:
    python Python/split_sample_for_training.py [source.csv] [--out DIR] \
        [--n-train 400] [--n-test 100] [--seed 42] \
        [--separate] [--train-subset N ...]
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
DEFAULT_SRC = ROOT / "Database" / "sample_for_training" / "2014-03-12_sample_for_training.csv"
DEFAULT_OUT = ROOT / "Database" / "dataset_for_training"

DEFAULT_N_TRAIN = 400
DEFAULT_N_TEST = 100

DELIMITER = ";"


def parse_line(fields: list[str]) -> list[str]:
    """Keep col 0 (userId), col 7 (nRecord), and the cellId half of the
    interleaved (cellId, timestamp) records — timestamps are dropped.

    Trailing empty fields (from a trailing ';') are dropped so the row is not
    padded with blanks.
    """
    kept = [fields[0]] + fields[7:8] + fields[8::2]
    while kept and kept[-1] == "":
        kept.pop()
    return kept


def build_header(n_record_cols: int) -> list[str]:
    """['userId', 'nRecord', 'cellId', 'cellId', ...]."""
    return ["userId", "nRecord"] + ["cellId"] * n_record_cols


def sample_rows(src: Path, n: int, rng: random.Random) -> list[list[str]]:
    """Parse every row of `src` (no filtering) and return a random sample of
    at most `n` of them. Each returned row is `[userId, nRecord, <cellId...>]`
    (parse_line output). If `src` has fewer than `n` rows, all of them are
    returned.
    """
    kept = []
    with src.open(newline="") as f:
        for fields in csv.reader(f, delimiter=DELIMITER):
            if not fields:
                continue
            kept.append(parse_line(fields))

    if len(kept) <= n:
        return kept
    return rng.sample(kept, n)


def quantile_bins(n_records: list[int], max_bins: int = 10) -> list[int]:
    """Group `n_records` into up to `max_bins` quantile bins, for use as a
    stratification label. Falls back to fewer bins when there aren't enough
    rows or distinct values to fill `max_bins` (at least ~20 rows/bin).
    """
    n_bins = max(2, min(max_bins, len(n_records) // 20, len(set(n_records))))
    return list(pd.qcut(n_records, q=n_bins, labels=False, duplicates="drop"))


def stratified_train_test_split(rows: list[list[str]], n_train: int, n_test: int,
                                seed: int) -> tuple[list[list[str]], list[list[str]]]:
    """Split `rows` into train/test, stratified by each row's `nRecord` (index
    1) so both splits have comparable sequence-length distributions. If fewer
    rows are available than `n_train + n_test`, all rows are kept and the
    train/test counts are scaled down proportionally.
    """
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


def stratified_subset(rows: list[list[str]], n: int, seed: int) -> list[list[str]]:
    """Return `n` of `rows`, stratified by `nRecord` (index 1) so the subset keeps
    the sequence-length distribution of the full set. All rows if `n` >= len(rows).
    """
    if n >= len(rows):
        return rows
    bins = quantile_bins([int(row[1]) for row in rows])
    subset, _ = train_test_split(rows, train_size=n, stratify=bins, random_state=seed)
    return subset


def write_rows(rows: list[list[str]], dest: Path) -> Path:
    """Write `rows` to `dest` with a header sized to the widest row."""
    widest = max((len(row) - 2 for row in rows), default=0)
    with dest.open("w", newline="") as fout:
        writer = csv.writer(fout, delimiter=DELIMITER)
        writer.writerow(build_header(widest))
        writer.writerows(rows)
    return dest


def log_length_balance(train_rows: list[list[str]], test_rows: list[list[str]]) -> None:
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


def split_sample(src: Path, out_dir: Path, n_train: int, n_test: int,
                 rng: random.Random, seed: int) -> tuple[Path, list[list[str]], list[list[str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{n_train}_{n_test}_sample_users.csv"

    pool = sample_rows(src, n_train + n_test, rng)
    train_rows, test_rows = stratified_train_test_split(pool, n_train, n_test, seed)
    rows = train_rows + test_rows
    write_rows(rows, dest)

    return dest, train_rows, test_rows


def split_sample_separate(src: Path, out_dir: Path, n_train: int, n_test: int,
                          rng: random.Random, seed: int,
                          train_subsets: list[int]) -> tuple[list[Path], list[list[str]], list[list[str]]]:
    """Same split as `split_sample`, but written as one CSV per split plus one
    per requested train subset. Returns the written paths and the train/test rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    pool = sample_rows(src, n_train + n_test, rng)
    train_rows, test_rows = stratified_train_test_split(pool, n_train, n_test, seed)

    written = [
        write_rows(train_rows, out_dir / f"{stem}_{len(train_rows)}_train.csv"),
        write_rows(test_rows, out_dir / f"{stem}_{len(test_rows)}_test.csv"),
    ]
    for n in train_subsets:
        subset = stratified_subset(train_rows, n, seed)
        if len(subset) < n:
            print(f"warning: train subset of {n} requested but only "
                  f"{len(subset)} train rows available")
        written.append(write_rows(subset, out_dir / f"{stem}_{len(subset)}_train.csv"))

    return written, train_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SRC,
                        help="raw vektory CSV to split (default: 2014-03-12_sample_for_training.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: Database/dataset_for_training)")
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN,
                        help=f"number of train users to keep (default: {DEFAULT_N_TRAIN})")
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST,
                        help=f"number of test users to keep (default: {DEFAULT_N_TEST})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed for the random sample and stratified split (default: {DEFAULT_SEED})")
    parser.add_argument("--separate", action="store_true",
                        help="write one CSV per split ({stem}_{n}_train.csv / _test.csv) "
                             "instead of a single combined file")
    parser.add_argument("--train-subset", type=int, action="append", default=[],
                        metavar="N",
                        help="also write a stratified subset of N train users "
                             "(repeatable; implies --separate)")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source file not found: {args.source}")

    rng = random.Random(args.seed)
    if args.separate or args.train_subset:
        written, train_rows, test_rows = split_sample_separate(
            args.source, args.out, args.n_train, args.n_test, rng, args.seed,
            args.train_subset)
    else:
        dest, train_rows, test_rows = split_sample(args.source, args.out, args.n_train,
                                                   args.n_test, rng, args.seed)
        written = [dest]

    kept = len(train_rows) + len(test_rows)
    n_requested = args.n_train + args.n_test
    if kept < n_requested:
        print(f"warning: only {kept} users available in the source file "
              f"(requested {n_requested} = {args.n_train} train + {args.n_test} test)")
    for path in written:
        print(f"sampled -> {path}")
    print(f"split: {len(train_rows)} train + {len(test_rows)} test")
    log_length_balance(train_rows, test_rows)


if __name__ == "__main__":
    main()
