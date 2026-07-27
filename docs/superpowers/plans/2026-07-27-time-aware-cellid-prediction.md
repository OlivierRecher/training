# Time-aware cell_id prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `train_cellid_llm.ipynb` aware of the hour-of-day of each `cell_id` event, and use that to push training to reason (rather than trivially repeat the last `cell_id`) inside configurable "transition window" hours — backed by a new real-data pipeline (`make data-train`) that keeps timestamps instead of dropping them.

**Architecture:** A new `python/cellid_encoding.py` module holds all pure, torch-free logic (hour tokens, transition-window membership, per-position loss weight, event chunking, event↔token-position mapping) shared by two new data-prep scripts (`python/split_sample_for_training.py`, `python/format_for_train.py`, adapted from `/Users/olivierrecher/Documents/IMT/SIO/work/Stage_2A`) and by `train_cellid_llm.ipynb`. The notebook interleaves a dedicated `<Hxx>` hour token before each `cell_id` token (hour always given, never predicted), and a custom `Trainer` subclass applies a configurable per-position loss weight to events whose hour falls in a transition window.

**Tech Stack:** Python 3.11, pandas/scikit-learn/scipy (data split), pytest (unit tests), transformers/peft/torch (notebook, already in requirements.txt).

## Global Constraints

- Time granularity is **hour-of-day (0–23) only** — no minute/second-level feature (decided with the user: 24 dedicated tokens are learnable; finer precision either explodes the token vocabulary or the sequence length against `max_seq_len=512`/`chunk_len=128`).
- `CONFIG["transition_windows"]` (list of `(start_hour, end_hour, weight)`, half-open `[start, end)`) is the **single source of truth**, lives in the notebook's `CONFIG` cell — not a separate config file — so changing the windows never requires regenerating data.
- Default transition windows: `[(4, 6, 2.0), (18, 20, 2.0)]`.
- All new Python scripts live under a top-level `python/` folder (per explicit user request), importable via `sys.path` (no package/`__init__.py`), and use `data/` (not `Database/`) as the data root.
- Existing synthetic JSONL files (`400_users_train.jsonl`, etc., no `hours` field) **must keep working unchanged** — hour-awareness only activates when a JSONL provides `hours`.
- Timestamp = seconds since local midnight, and it **follows** the `cell_id` it belongs to (`…;cell_id;ts;cell_id;ts;…`), per `CLAUDE.md`.
- Reference implementation to adapt from: `/Users/olivierrecher/Documents/IMT/SIO/work/Stage_2A/Makefile`, `Python/split_sample_for_training.py`, `Python/format_for_train.py` — same behavior, except timestamps are **kept**, not dropped.

---

### Task 1: Add data-prep and test dependencies

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `pandas`, `scikit-learn`, `scipy`, `pytest` importable in `.venv` for later tasks.

- [ ] **Step 1: Add the new dependencies**

Add these lines to the end of `requirements.txt`:

```
pandas
scikit-learn
scipy
pytest>=7.0
```

- [ ] **Step 2: Install them**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: `pandas`, `scikit-learn`, `scipy`, `pytest` installed with no errors (or already satisfied if present from prior torch/transformers install chain).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add pandas/scikit-learn/scipy/pytest for the data-train pipeline"
```

---

### Task 2: `python/cellid_encoding.py` — shared time-aware encoding logic

**Files:**
- Create: `python/cellid_encoding.py`
- Create: `pytest.ini`
- Test: `tests/test_cellid_encoding.py`

**Interfaces:**
- Produces (used by Tasks 4, 7–14):
  - `event_hour_from_seconds(ts_seconds: int) -> int`
  - `hour_token(hour: int) -> str`
  - `HOUR_VOCAB: list[str]` (24 entries, `"<H00>"`…`"<H23>"`)
  - `in_transition_window(hour: int, transition_windows: list[tuple[int,int,float]]) -> bool`
  - `event_weight(hour: int | None, transition_windows, base_weight: float = 1.0) -> float`
  - `split_index(n_events: int, context_fraction: float) -> int`
  - `make_chunks(n_events: int, chunk_len: int, chunk_stride: int) -> list[tuple[int,int,int]]`
  - `event_token_positions(n_events: int, has_hours: bool) -> list[int]`
  - `encode_events(cells: list[str], hours: list[int] | None, cell_to_id: dict, hour_to_id: dict, prefix_ids: list[int], transition_windows, n_masked_cells: int = 0, base_weight: float = 1.0) -> tuple[list[int], list[int], list[float]]`

- [ ] **Step 1: Create `pytest.ini` so tests can import scripts under `python/`**

```ini
[pytest]
pythonpath = python
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_cellid_encoding.py`:

```python
from cellid_encoding import (
    event_hour_from_seconds, hour_token, HOUR_VOCAB, in_transition_window,
    event_weight, split_index, make_chunks, event_token_positions, encode_events,
)


def test_event_hour_from_seconds():
    assert event_hour_from_seconds(0) == 0
    assert event_hour_from_seconds(3599) == 0
    assert event_hour_from_seconds(3600) == 1
    assert event_hour_from_seconds(86399) == 23


def test_hour_token_and_vocab():
    assert hour_token(0) == "<H00>"
    assert hour_token(23) == "<H23>"
    assert HOUR_VOCAB[0] == "<H00>"
    assert HOUR_VOCAB[-1] == "<H23>"
    assert len(HOUR_VOCAB) == 24


def test_in_transition_window_bounds_are_half_open():
    windows = [(4, 6, 2.0), (18, 20, 2.0)]
    assert in_transition_window(4, windows) is True
    assert in_transition_window(5, windows) is True
    assert in_transition_window(6, windows) is False   # end excluded
    assert in_transition_window(19, windows) is True
    assert in_transition_window(20, windows) is False
    assert in_transition_window(12, windows) is False


def test_event_weight_uses_window_weight_or_base():
    windows = [(4, 6, 2.0), (18, 20, 3.0)]
    assert event_weight(5, windows) == 2.0
    assert event_weight(19, windows) == 3.0
    assert event_weight(12, windows) == 1.0
    assert event_weight(12, windows, base_weight=0.5) == 0.5
    assert event_weight(None, windows) == 1.0   # hour-less data never weighted


def test_split_index_normal_case():
    assert split_index(10, 0.8) == 8


def test_split_index_clamped_to_valid_range():
    assert split_index(1, 0.8) == 1   # inherited edge-case: len-1 events still needs >=1 target
    assert split_index(5, 0.99) == 4  # never returns n_events itself


def test_make_chunks_single_window_when_short():
    assert make_chunks(5, chunk_len=10, chunk_stride=5) == [(0, 5, 0)]


def test_make_chunks_multiple_windows_with_overlap():
    # 10 events, chunk_len=4, stride=2: windows [0,4) ctx0, [2,6) ctx2, [4,8) ctx2, [6,10) ctx2
    assert make_chunks(10, chunk_len=4, chunk_stride=2) == [
        (0, 4, 0), (2, 6, 2), (4, 8, 2), (6, 10, 2),
    ]


def test_event_token_positions_with_hours():
    assert event_token_positions(3, has_hours=True) == [1, 3, 5]


def test_event_token_positions_without_hours():
    assert event_token_positions(3, has_hours=False) == [0, 1, 2]


def _vocab_maps():
    cell_to_id = {"A": 100, "B": 101}
    hour_to_id = {h: 200 + i for i, h in enumerate(HOUR_VOCAB)}
    return cell_to_id, hour_to_id


def test_encode_events_interleaves_hour_and_cell_tokens():
    cell_to_id, hour_to_id = _vocab_maps()
    windows = [(4, 6, 2.0)]
    ids, labels, weights = encode_events(
        cells=["A", "B"], hours=[5, 10], cell_to_id=cell_to_id, hour_to_id=hour_to_id,
        prefix_ids=[1], transition_windows=windows)
    assert ids == [1, hour_to_id[hour_token(5)], 100, hour_to_id[hour_token(10)], 101]
    assert labels == [-100, -100, 100, -100, 101]
    assert weights == [1.0, 1.0, 2.0, 1.0, 1.0]   # event 0 (hour 5) is in the window, event 1 is not


def test_encode_events_masks_first_n_cells():
    cell_to_id, hour_to_id = _vocab_maps()
    ids, labels, weights = encode_events(
        cells=["A", "B"], hours=[5, 5], cell_to_id=cell_to_id, hour_to_id=hour_to_id,
        prefix_ids=[], transition_windows=[(4, 6, 2.0)], n_masked_cells=1)
    assert labels == [-100, -100, -100, 101]   # first cell (A) masked despite being in-window
    assert weights == [1.0, 1.0, 1.0, 2.0]


def test_encode_events_without_hours_is_cell_only():
    cell_to_id, hour_to_id = _vocab_maps()
    ids, labels, weights = encode_events(
        cells=["A", "B"], hours=None, cell_to_id=cell_to_id, hour_to_id=hour_to_id,
        prefix_ids=[9], transition_windows=[(4, 6, 2.0)])
    assert ids == [9, 100, 101]
    assert labels == [-100, 100, 101]
    assert weights == [1.0, 1.0, 1.0]   # no hour info -> always base_weight
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cellid_encoding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cellid_encoding'`

- [ ] **Step 4: Write the implementation**

Create `python/cellid_encoding.py`:

```python
"""Time-aware event encoding shared by the data-prep scripts and the training
notebook.

An "event" is one (cell_id, hour) pair from a user's daily trajectory. Hour is
the hour-of-day (0-23) derived from the raw timestamp (seconds since local
midnight, per CLAUDE.md) attached to that cell_id -- the timestamp follows the
cell_id it belongs to.
"""


def event_hour_from_seconds(ts_seconds):
    """Hour of day (0-23) from a timestamp in seconds since local midnight."""
    return (int(ts_seconds) // 3600) % 24


def hour_token(hour):
    """Dedicated tokenizer token for an hour-of-day, e.g. hour_token(6) == '<H06>'."""
    return f"<H{hour:02d}>"


HOUR_VOCAB = [hour_token(h) for h in range(24)]


def in_transition_window(hour, transition_windows):
    """True if `hour` falls in any (start_hour, end_hour, weight) window.
    Bounds are half-open [start_hour, end_hour) -- end excluded."""
    return any(start <= hour < end for start, end, _weight in transition_windows)


def event_weight(hour, transition_windows, base_weight=1.0):
    """Loss weight for an event at `hour`: the configured window's weight if
    `hour` falls inside one of `transition_windows`, else `base_weight`.
    `hour=None` (hour-less legacy data) always returns `base_weight`."""
    if hour is None:
        return base_weight
    for start, end, weight in transition_windows:
        if start <= hour < end:
            return weight
    return base_weight


def split_index(n_events, context_fraction):
    """Cutoff index for a context/target split over `n_events` events: at
    least 1, at most n_events - 1."""
    return max(1, min(n_events - 1, round(n_events * context_fraction)))


def make_chunks(n_events, chunk_len, chunk_stride):
    """Sliding-window (start, end, n_context_events) triples over event
    indices [0, n_events). `n_context_events` is how many of the window's
    leading events are overlap from the previous window (already trained on,
    excluded from the loss the second time). Windows are at most `chunk_len`
    events, advancing by `chunk_stride` each time."""
    if n_events <= chunk_len:
        return [(0, n_events, 0)]
    out, start = [], 0
    while start < n_events:
        end = min(start + chunk_len, n_events)
        ctx = 0 if start == 0 else chunk_len - chunk_stride
        if end - start > ctx:
            out.append((start, end, ctx))
        if start + chunk_len >= n_events:
            break
        start += chunk_stride
    return out


def event_token_positions(n_events, has_hours):
    """Index (relative to the first token *after* the prompt prefix) of each
    event's cell_id token in the flat sequence built by `encode_events`. When
    `has_hours` is True, each event is 2 tokens (hour then cell_id), so
    cell_id tokens sit at 1, 3, 5, ...; when False, each event is 1 token
    (cell_id only), so they sit at 0, 1, 2, ..."""
    step = 2 if has_hours else 1
    offset = 1 if has_hours else 0
    return [offset + i * step for i in range(n_events)]


def encode_events(cells, hours, cell_to_id, hour_to_id, prefix_ids,
                  transition_windows, n_masked_cells=0, base_weight=1.0):
    """Build (input_ids, labels, weights) for one sequence of events.

    cells: list[str] cell_id per event.
    hours: list[int] | None. If None, cell tokens only (legacy/no hour info) --
        no interleaved hour tokens, every unmasked position gets `base_weight`.
    cell_to_id / hour_to_id: dict[str, int] tokenizer vocab lookups.
    prefix_ids: list[int] tokens prepended before any event (e.g. a fixed
        prompt) -- always masked (label -100, weight base_weight).
    n_masked_cells: the first N events' cell labels are masked (-100) --
        used for chunk overlap, so a repeated context window isn't trained on
        twice.
    Returns three lists of equal length (one entry per token): input_ids,
    labels, weights.
    """
    ids = list(prefix_ids)
    labels = [-100] * len(prefix_ids)
    weights = [base_weight] * len(prefix_ids)
    for i, cell in enumerate(cells):
        hour = hours[i] if hours is not None else None
        if hour is not None:
            ids.append(hour_to_id[hour_token(hour)])
            labels.append(-100)
            weights.append(base_weight)
        ids.append(cell_to_id[cell])
        if i < n_masked_cells:
            labels.append(-100)
            weights.append(base_weight)
        else:
            labels.append(cell_to_id[cell])
            weights.append(event_weight(hour, transition_windows, base_weight))
    return ids, labels, weights
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cellid_encoding.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add python/cellid_encoding.py pytest.ini tests/test_cellid_encoding.py
git commit -m "feat: add shared time-aware event encoding module"
```

---

### Task 3: `python/split_sample_for_training.py` — sample + stratified split, timestamps kept

**Files:**
- Create: `python/split_sample_for_training.py`
- Test: `tests/test_split_sample_for_training.py`

**Interfaces:**
- Consumes: nothing from Task 2 (pure CSV/pandas/sklearn logic).
- Produces (used by Task 5's Makefile, and read by Task 4's `format_for_train.py`):
  - CLI: `python python/split_sample_for_training.py [source.csv] [--out DIR] [--n-train N] [--n-test M] [--seed S]`
  - Output CSV at `<out>/{n_train}_{n_test}_sample_users.csv`, header `userId;nRecord;cellId;ts;cellId;ts;...`, train rows first then test rows.
  - `parse_line(fields: list[str]) -> list[str]`
  - `build_header(n_events: int) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_split_sample_for_training.py`:

```python
import csv
from pathlib import Path

from split_sample_for_training import parse_line, build_header, split_sample
import random


def test_parse_line_keeps_userid_nrecord_and_all_cell_ts_pairs():
    fields = ["1234", "22", "M", "36452", "CETR", "", "", "3",
              "BKVDOU3", "23822", "BKVVIV2", "24105", "BKVSTN4", "24154"]
    assert parse_line(fields) == [
        "1234", "3", "BKVDOU3", "23822", "BKVVIV2", "24105", "BKVSTN4", "24154",
    ]


def test_parse_line_strips_trailing_empty_fields():
    fields = ["1234", "22", "M", "", "CETR", "", "", "1", "BKVDOU3", "23822", ""]
    assert parse_line(fields) == ["1234", "1", "BKVDOU3", "23822"]


def test_build_header_alternates_cellid_ts():
    assert build_header(3) == ["userId", "nRecord", "cellId", "ts", "cellId", "ts", "cellId", "ts"]


def _write_fake_source(path, n_users=12):
    rng = random.Random(0)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        for uid in range(n_users):
            n_events = rng.randint(2, 6)
            row = [str(1000 + uid), str(20 + uid), "M", "1", "CETR", "", "", str(n_events)]
            for _ in range(n_events):
                row += [f"CELL{rng.randint(0, 3)}", str(rng.randint(0, 86399))]
            writer.writerow(row)


def test_split_sample_writes_train_then_test_with_ts_preserved(tmp_path):
    src = tmp_path / "fake_source.csv"
    _write_fake_source(src, n_users=12)
    out_dir = tmp_path / "out"
    rng = random.Random(42)

    dest, train_rows, test_rows = split_sample(src, out_dir, n_train=8, n_test=4, rng=rng, seed=42)

    assert dest.exists()
    assert len(train_rows) == 8
    assert len(test_rows) == 4

    with dest.open(newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        assert header[:4] == ["userId", "nRecord", "cellId", "ts"]
        rows = list(reader)
    assert len(rows) == 12
    # every kept row must still have a numeric timestamp right after its first cellId
    first_data_row = rows[0]
    assert first_data_row[3].isdigit()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_split_sample_for_training.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'split_sample_for_training'`

- [ ] **Step 3: Write the implementation**

Create `python/split_sample_for_training.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_split_sample_for_training.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/split_sample_for_training.py tests/test_split_sample_for_training.py
git commit -m "feat: add timestamp-preserving user sampling script"
```

---

### Task 4: `python/format_for_train.py` — JSONL with cells + hours

**Files:**
- Create: `python/format_for_train.py`
- Test: `tests/test_format_for_train.py`

**Interfaces:**
- Consumes: `event_hour_from_seconds` from Task 2's `cellid_encoding`.
- Produces (used by Task 5's Makefile, and read by the notebook in Task 7):
  - CLI: `python python/format_for_train.py cleaned.csv --n-train N --n-test M [--out DIR]`
  - Output JSONL lines: `{"conversations": [...], "user_id": str, "cells": list[str], "hours": list[int]}`
  - `parse_records(records: list[str]) -> tuple[list[str], list[int]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_format_for_train.py`:

```python
import json
from pathlib import Path

from format_for_train import parse_records, row_to_conversation, convert


def test_parse_records_splits_cells_and_computes_hours():
    records = ["BKVDOU3", "23822", "BKVVIV2", "24105"]
    cells, hours = parse_records(records)
    assert cells == ["BKVDOU3", "BKVVIV2"]
    assert hours == [23822 // 3600, 24105 // 3600]


def test_parse_records_drops_unpaired_trailing_field():
    records = ["BKVDOU3", "23822", "BKVVIV2"]   # stray trailing cellId with no ts
    cells, hours = parse_records(records)
    assert cells == ["BKVDOU3"]
    assert hours == [23822 // 3600]


def test_row_to_conversation_has_readable_text_and_structured_fields():
    conv = row_to_conversation("43", "2", ["A", "B"], [3, 4])
    assert conv["user_id"] == "43"
    assert conv["cells"] == ["A", "B"]
    assert conv["hours"] == [3, 4]
    text = conv["conversations"][-1]["content"]
    assert text == "User 43 | 2 events | A B"


def test_convert_splits_exact_train_test_counts(tmp_path):
    src = tmp_path / "cleaned.csv"
    lines = ["userId;nRecord;cellId;ts;cellId;ts"]
    for uid in range(6):
        lines.append(f"{uid};2;CELL{uid};3600;CELL{uid};7200")
    src.write_text("\n".join(lines) + "\n")

    train_dest, n_train, test_dest, n_test = convert(src, tmp_path / "out", n_train=4, n_test=2)

    assert n_train == 4
    assert n_test == 2
    train_objs = [json.loads(l) for l in train_dest.read_text().splitlines()]
    test_objs = [json.loads(l) for l in test_dest.read_text().splitlines()]
    assert len(train_objs) == 4
    assert len(test_objs) == 2
    assert train_objs[0]["hours"] == [1, 2]   # 3600s -> hour 1, 7200s -> hour 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_format_for_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'format_for_train'`

- [ ] **Step 3: Write the implementation**

Create `python/format_for_train.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_format_for_train.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/format_for_train.py tests/test_format_for_train.py
git commit -m "feat: add hour-aware JSONL formatting script"
```

---

### Task 5: `Makefile` — `data-train` target

**Files:**
- Create: `Makefile`

**Interfaces:**
- Consumes: `python/split_sample_for_training.py`, `python/format_for_train.py` (Tasks 3–4).
- Produces: `data/dataset_for_training/{N_TRAIN}_{N_TEST}_sample_users_{train,test}.jsonl`, used manually to feed the notebook's `CONFIG["train_file"]`/`CONFIG["test_file"]`.

- [ ] **Step 1: Create the Makefile**

Create `Makefile` at the repo root:

```makefile
# Makefile for the training project -- data preparation pipeline
# Quick usage:
#   make data-train    -> build the timestamp-aware N_TRAIN/N_TEST sample dataset
#                         (override counts/source with N_TRAIN=.../N_TEST=.../SRC=...)
#   make clean         -> remove generated dataset files and Python caches
#   make help          -> show help
#
# Requires the virtual environment created by `bash setup_venv.sh` -- run that
# first if `.venv` does not exist yet.

VENV_PYTHON := .venv/bin/python
PYTHON_DIR  := python
DATA_DIR    := data/dataset_for_training

# Number of train/test users to sample for the `data-train` target
# (override: make data-train N_TRAIN=1000 N_TEST=200)
N_TRAIN := 400
N_TEST  := 100
# Sample day-file to split (unfiltered) for the `data-train` target
# (override: make data-train SRC=data/sample_for_training/2014-03-13_sample_for_training.csv)
SRC := data/sample_for_training/2014-03-12_sample_for_training.csv

.DEFAULT_GOAL := help

$(VENV_PYTHON):
	@echo "ERREUR: .venv introuvable -- lancez d'abord: bash setup_venv.sh" >&2
	@exit 1

# --- Sampled training dataset (N_TRAIN + N_TEST users), timestamps kept -----
# Splits the sample_for_training day-file into an EXACT train/test count with
# NO filtering at all -- userId/nRecord/cellId/timestamp cleaning (timestamps
# KEPT this time, turned into an hour-of-day signal by format_for_train.py), a
# random pick of N_TRAIN + N_TEST users stratified by nRecord, then JSONL
# formatting.
#   make data-train                          -> 400 train + 100 test (default)
#   make data-train N_TRAIN=1000 N_TEST=200  -> 1000 train + 200 test
#   make data-train SRC=path/to/other.csv    -> split a different day-file
#   -> data/dataset_for_training/{N_TRAIN}_{N_TEST}_sample_users_train.jsonl
#   -> data/dataset_for_training/{N_TRAIN}_{N_TEST}_sample_users_test.jsonl
.PHONY: data-train
data-train: $(VENV_PYTHON)  ## Build the train/test sample dataset + format; N_TRAIN=400/N_TEST=100 by default
	@echo ">> Splitting $(N_TRAIN) train + $(N_TEST) test users (no filters) from $(SRC)"
	$(VENV_PYTHON) $(PYTHON_DIR)/split_sample_for_training.py $(SRC) --n-train $(N_TRAIN) --n-test $(N_TEST)
	$(VENV_PYTHON) $(PYTHON_DIR)/format_for_train.py $(DATA_DIR)/$(N_TRAIN)_$(N_TEST)_sample_users.csv --n-train $(N_TRAIN) --n-test $(N_TEST)

.PHONY: clean
clean:              ## Remove generated dataset files and Python caches
	@echo ">> Removing generated datasets and caches"
	rm -rf $(DATA_DIR)
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

.PHONY: help
help:               ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
```

- [ ] **Step 2: Run it against the real sample data**

Run: `make data-train N_TRAIN=8 N_TEST=4`
Expected: prints the split summary (train/test length balance + Mann-Whitney line) and `formatted -> ... (8 lines)` / `formatted -> ... (4 lines)`. Confirm the files exist:

Run: `ls data/dataset_for_training/`
Expected: `8_4_sample_users.csv`, `8_4_sample_users_train.jsonl`, `8_4_sample_users_test.jsonl`

Run: `python3 -c "import json; d=json.loads(open('data/dataset_for_training/8_4_sample_users_train.jsonl').readline()); print(d['cells'][:3], d['hours'][:3])"`
Expected: prints a short list of cell_id strings alongside a matching list of small integers (0–23) — confirms `hours` really is hour-of-day, not raw seconds.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add make data-train pipeline (timestamps preserved as hours)"
```

---

### Task 6: Notebook `CONFIG` cell — transition windows + module path

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `3538fcec` (the `CONFIGURATION` cell)

**Interfaces:**
- Produces: `CONFIG["transition_windows"]` (used by Tasks 7, 9, 11–13), `python/` on `sys.path` (used by every later notebook task's `from cellid_encoding import ...`).

- [ ] **Step 1: Replace the cell**

The notebook must already have been Read in this conversation (`Read train_cellid_llm.ipynb`) before calling `NotebookEdit`. Use `NotebookEdit` with `notebook_path=<repo>/train_cellid_llm.ipynb`, `cell_id="3538fcec"`, `edit_mode="replace"`, and this `new_source`:

```python
# ============================== CONFIGURATION ==============================
from pathlib import Path
import sys

sys.path.insert(0, str(Path("python")))   # python/cellid_encoding.py -- upload it too on Colab

N_USERS = 500   # nombre d'utilisateurs du jeu d'entraînement : changez uniquement cette valeur
                # (500, 100, ou toute autre valeur — generate_sample_users.py régénère
                # automatiquement les fichiers manquants, voir la cellule PRÉFLIGHT)

CONFIG = {
    # Modèle de base — décommentez celui voulu :
    # "model_name": "mistralai/Mistral-7B-v0.1",      # 7B : GPU (Colab) obligatoire, chargé en 4-bit (QLoRA) ; licence à accepter sur huggingface.co
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",       # 0.5B : léger, fonctionne aussi en local (MPS)

    # Jeu de données : dérivé automatiquement de N_USERS ci-dessus.
    # Pour utiliser un jeu réel produit par `make data-train`, pointez ces deux
    # clés directement vers ses fichiers, ex. :
    #   "train_file": "data/dataset_for_training/400_100_sample_users_train.jsonl",
    #   "test_file":  "data/dataset_for_training/400_100_sample_users_test.jsonl",
    "train_file": f"{N_USERS}_users_train.jsonl",
    "test_file": f"{N_USERS}_users_test.jsonl",
    "output_dir": "outputs",
    "seed": 42,

    # Token Hugging Face (optionnel : Qwen2.5 est public). Ordre de priorité au login :
    # variable d'env HF_TOKEN > secret Colab "HF_TOKEN" > la valeur ci-dessous.
    # ⚠️ Si vous partagez ce notebook, videz cette valeur et utilisez les secrets Colab.
    "hf_token": "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",

    # --- Arrêt de l'entraînement ---
    "early_stop_patience": 4,       # epochs sans amélioration avant arrêt — NE PAS augmenter :
                                    # le pic arrive tôt (~epoch 9), au-delà le modèle sur-apprend
    "max_epochs": 20,               # plafond dur (le scheduler cosine est calé dessus)

    # --- LoRA / optimisation (réglés anti-overfit : dataset très petit) ---
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "learning_rate": 1.5e-4,
    "weight_decay": 0.01,
    "max_seq_len": 512,          # longueur max à l'évaluation (séquence complète)
    "chunk_len": 128,            # fenêtres d'entraînement courtes -> mémoire réduite (en nb d'événements)
    "chunk_stride": 64,
    # Fraction de la séquence de chaque utilisateur donnée en contexte : le modèle
    # n'est entraîné QUE sur ce préfixe, et doit prédire la suite (1 - context_fraction)
    # jamais vue à l'entraînement. Appliqué à TOUS les utilisateurs (train ET test),
    # donc la validation couvre tous les comportements plutôt qu'un échantillon aléatoire
    # d'utilisateurs. Modifiable ici uniquement.
    "context_fraction": 0.8,
    "prefix_text": "Events:",

    # Créneaux de transition home <-> activité : heures où l'utilisateur a
    # statistiquement plus de chances de changer de cell_id. Liste de
    # (heure_debut, heure_fin, poids_loss) -- bornes [heure_debut, heure_fin[
    # (fin exclue). Le poids remplace la valeur par défaut (1.0) dans la loss
    # pondérée pour pousser le modèle à raisonner plutôt qu'à recopier le
    # dernier cell_id dans ces créneaux. Modifiable ici uniquement -- aucune
    # regénération de données nécessaire. Extension future possible : un
    # dict {user_id: [...]} avec repli sur ce défaut global (non implémenté).
    "transition_windows": [(4, 6, 2.0), (18, 20, 2.0)],

    # --- Guards (protection machine) ---
    "min_free_ram_gb": 2.0,         # RAM libre minimale (préflight + pendant l'entraînement)
    "max_ram_used_fraction": 0.90,  # arrêt propre si la RAM système dépasse ce taux
    "min_free_disk_gb": 5.0,        # espace disque minimal (modèle + checkpoints)
    "ram_check_every_steps": 10,
    "save_total_limit": 2,          # nb max de checkpoints conservés sur disque
    "resume_from_checkpoint": False, # True pour reprendre après une interruption
}
```

- [ ] **Step 2: Manual verification**

Run: `.venv/bin/jupyter execute --output /tmp/cell_check.ipynb <(python3 -c "
import json
nb = json.load(open('train_cellid_llm.ipynb'))
nb['cells'] = [c for c in nb['cells'] if c.get('id') == '3538fcec']
json.dump(nb, open('/tmp/one_cell.ipynb', 'w'))
")`

Simpler: just run `.venv/bin/python -c "exec(open('/dev/stdin').read())" <<'EOF'
import sys, json
nb = json.load(open('train_cellid_llm.ipynb'))
src = ''.join(c['source'] for c in nb['cells'] if c.get('id') == '3538fcec')
exec(src)
print('transition_windows =', CONFIG['transition_windows'])
print('python on sys.path:', str(Path('python')) in sys.path or any('python' in p for p in sys.path))
EOF`
Expected: `transition_windows = [(4, 6, 2.0), (18, 20, 2.0)]` and confirmation `python` is on `sys.path`, no exceptions.

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "feat(notebook): add transition_windows config and python/ module path"
```

---

### Task 7: Notebook data-loading cell — hour-aware `load_users` + baseline breakdown

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `877bd015` (`CHARGEMENT DES DONNÉES + BASELINES`)

**Interfaces:**
- Consumes: `split_index`, `in_transition_window` from `cellid_encoding` (Task 2); `CONFIG["transition_windows"]` (Task 6).
- Produces: `users_train`, `users_test` as `list[(user_id, cells, hours_or_None)]` (used by every later notebook task); `VOCAB`, `HAS_HOURS`, `baseline_persistence(users, context_fraction=None)` (unchanged float-returning signature — still used as-is by Tasks 12/13), `baseline_persistence_breakdown(users, context_fraction, transition_windows)`.

- [ ] **Step 1: Replace the cell**

Use `NotebookEdit` with `cell_id="877bd015"`, `edit_mode="replace"`:

```python
# ==================== CHARGEMENT DES DONNÉES + BASELINES ====================
import json
from collections import Counter
from cellid_encoding import split_index, in_transition_window

def load_users(path):
    """Parse le JSONL -> [(user_id, [cell_id, ...], [heure, ...] | None), ...].
    Si une ligne porte les champs structurés "cells"/"hours" (nouveau format,
    voir python/format_for_train.py), ils sont utilisés directement. Sinon
    (anciens jeux synthétiques), on retombe sur le texte "assistant" -- heures
    = None -> pas d'heure interleavée, comportement inchangé."""
    users = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "cells" in row:
                uid = row.get("user_id", "?")
                cells = row["cells"]
                hours = row.get("hours")
            else:
                content = row["conversations"][-1]["content"]
                head, _, seq = content.split("|", 2)
                uid = head.split()[1]
                cells = seq.split()
                hours = None
            users.append((uid, cells, hours))
    return users

users_train = load_users(CONFIG["train_file"])
users_test  = load_users(CONFIG["test_file"])

VOCAB = sorted({c for _, cells, _ in users_train + users_test for c in cells})
train_vocab = {c for _, cells, _ in users_train for c in cells}
lens = [len(cells) for _, cells, _ in users_train + users_test]
print(f"{len(users_train)} utilisateurs train | {len(users_test)} test | vocabulaire = {len(VOCAB)} cell_id")
print(f"Longueur des séquences : min={min(lens)} max={max(lens)} moy={sum(lens)/len(lens):.0f}")
print(f"cell_id du test absents du train : {len(set(VOCAB) - train_vocab)}")

HAS_HOURS = any(hours is not None for _, _, hours in users_train + users_test)
print(f"Heures disponibles dans les données : {'oui' if HAS_HOURS else 'non (mode heure désactivé)'}")

def split_index_for(cells, context_fraction):
    return split_index(len(cells), context_fraction)

def baseline_persistence(users, context_fraction=None):
    """Prédire « même cell_id que le précédent » — la référence à battre.
    Si context_fraction est fourni, ne compte que les positions de la cible (les
    (1 - context_fraction) derniers cell_id de chaque utilisateur), pour rester comparable
    à evaluate_cell_accuracy(..., context_fraction=...)."""
    tot = ok = 0
    for _, s, _ in users:
        min_j = split_index_for(s, context_fraction) if context_fraction is not None else 1
        for i in range(1, len(s)):
            if i < min_j:
                continue
            tot += 1
            ok += s[i] == s[i - 1]
    return ok / tot if tot else float("nan")

def baseline_persistence_breakdown(users, context_fraction, transition_windows):
    """Comme baseline_persistence, mais rapporte l'accuracy séparément dans /
    hors créneaux de transition (utilisateurs sans heures ignorés dans le
    calcul du breakdown, mais toujours comptés dans "overall")."""
    tot = ok = 0
    tot_in = ok_in = tot_out = ok_out = 0
    for _, s, hours in users:
        min_j = split_index_for(s, context_fraction)
        for i in range(1, len(s)):
            if i < min_j:
                continue
            tot += 1
            hit = s[i] == s[i - 1]
            ok += hit
            if hours is not None:
                if in_transition_window(hours[i], transition_windows):
                    tot_in += 1; ok_in += hit
                else:
                    tot_out += 1; ok_out += hit
    return {
        "overall": ok / tot if tot else float("nan"),
        "in_window": ok_in / tot_in if tot_in else float("nan"),
        "out_window": ok_out / tot_out if tot_out else float("nan"),
    }

BASELINE_TEST = baseline_persistence(users_test, CONFIG["context_fraction"])
print(f"Baseline persistance — test (sur la cible {1 - CONFIG['context_fraction']:.0%}) : "
      f"{BASELINE_TEST:.1%}")
if HAS_HOURS:
    bd = baseline_persistence_breakdown(users_test, CONFIG["context_fraction"], CONFIG["transition_windows"])
    print(f"  dans créneaux de transition : {bd['in_window']:.1%}  |  hors créneaux : {bd['out_window']:.1%}")
```

- [ ] **Step 2: Manual verification**

This cell depends on `CONFIG`/`load_users` from Task 6 and real data files. Full verification happens in Task 16's end-to-end run. As a quick syntax/logic check now:

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0, 'python')
from cellid_encoding import split_index, in_transition_window
assert split_index(10, 0.8) == 8
assert in_transition_window(5, [(4,6,2.0)]) is True
print('helpers OK')
"`
Expected: `helpers OK`

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "feat(notebook): hour-aware load_users + transition-window baseline breakdown"
```

---

### Task 8: Notebook tokenizer/model cell — dedicated hour tokens

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `bc1e1e76` (`TOKENIZER ÉTENDU + MODÈLE + LoRA`)

**Interfaces:**
- Consumes: `HOUR_VOCAB` from `cellid_encoding` (Task 2), `HAS_HOURS` (Task 7).
- Produces: `HOUR_TOKEN_IDS` (used by Task 9), `trainable_tokens` now covering both cell and hour embeddings.

- [ ] **Step 1: Replace the cell**

Use `NotebookEdit` with `cell_id="bc1e1e76"`, `edit_mode="replace"`:

```python
# ==================== TOKENIZER ÉTENDU + MODÈLE + LoRA ====================
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model
from cellid_encoding import HOUR_VOCAB

set_seed(CONFIG["seed"])

# Un modèle >= 7B ne tient pas en local : GPU obligatoire, chargé en 4-bit (QLoRA)
BIG_MODEL = any(t in CONFIG["model_name"] for t in ("7B", "8B", "13B", "70B"))
if BIG_MODEL and DEVICE != "cuda":
    raise RuntimeError(
        f"{CONFIG['model_name']} nécessite un GPU (Colab). En local, repassez sur "
        "Qwen/Qwen2.5-0.5B-Instruct dans la cellule CONFIG."
    )

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
if tokenizer.pad_token is None:          # Mistral n'a pas de pad token
    tokenizer.pad_token = tokenizer.eos_token
n_added = tokenizer.add_tokens(VOCAB)   # 1 cell_id = 1 token dédié
print(f"{n_added} tokens cell_id ajoutés au tokenizer")
n_added_hours = tokenizer.add_tokens(HOUR_VOCAB) if HAS_HOURS else 0
if HAS_HOURS:
    print(f"{n_added_hours} tokens heure ajoutés au tokenizer")

if BIG_MODEL:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE)
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"],
                                                 quantization_config=bnb, device_map={"": 0})
else:
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"], dtype=DTYPE)

# Redimensionnement AVANT prepare_model_for_kbit_training, init simple (mean_resizing
# provoque des erreurs CUDA sur modèles quantifiés/fp16)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
if BIG_MODEL:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False

CELL_TOKEN_IDS = tokenizer.convert_tokens_to_ids(VOCAB)
HOUR_TOKEN_IDS = tokenizer.convert_tokens_to_ids(HOUR_VOCAB) if HAS_HOURS else []

# Lignes d'embedding des nouveaux tokens à entraîner (cell_id ET heure). Si les
# embeddings d'entrée et de sortie ne sont PAS liés (cas de Mistral), il faut
# aussi entraîner les lignes du lm_head, sinon les logits des nouveaux tokens
# restent figés à leur init aléatoire.
NEW_TOKEN_IDS = CELL_TOKEN_IDS + HOUR_TOKEN_IDS
tied = getattr(model.config, "tie_word_embeddings", False)
trainable_tokens = ({"embed_tokens": NEW_TOKEN_IDS} if tied
                    else {"embed_tokens": NEW_TOKEN_IDS, "lm_head": NEW_TOKEN_IDS})

lora_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    trainable_token_indices=trainable_tokens,
)
model = get_peft_model(model, lora_cfg)
if not BIG_MODEL:               # le modèle 4-bit est déjà placé sur le GPU par device_map
    model.to(DEVICE)
model.print_trainable_parameters()
```

- [ ] **Step 2: Manual verification**

Covered by Task 16's end-to-end run (this cell needs a real tokenizer/model download, too heavy to verify standalone). Sanity-check the diff introduces no syntax errors:

Run: `.venv/bin/python -c "
import json
nb = json.load(open('train_cellid_llm.ipynb'))
src = next(c['source'] for c in nb['cells'] if c.get('id') == 'bc1e1e76')
compile(src, 'cell', 'exec')
print('syntax OK')
"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "feat(notebook): add dedicated hour tokens to the tokenizer/model"
```

---

### Task 9: Notebook datasets cell — event-based encoding via `cellid_encoding`

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `c380dc72` (`DATASETS`)

**Interfaces:**
- Consumes: `make_chunks`, `encode_events` from `cellid_encoding` (Task 2); `split_index` (Task 7's import); `VOCAB`, `CELL_TOKEN_IDS`, `HOUR_VOCAB`, `HOUR_TOKEN_IDS`, `HAS_HOURS` (Tasks 7–8).
- Produces: `CELL_TO_ID`, `HOUR_TO_ID` (used by Tasks 11, 14); `encode_sequence(cells, hours, n_masked_cells=0) -> dict` with keys `input_ids`, `labels`, `weights`, `attention_mask` (used by Tasks 10, 11); `train_ds` (used by Tasks 10, 12); `collate(batch) -> dict` including a `"weights"` tensor (used by Task 12).

- [ ] **Step 1: Replace the cell**

Use `NotebookEdit` with `cell_id="c380dc72"`, `edit_mode="replace"`:

```python
# ==================== DATASETS (causal, loss uniquement sur les cell_id) ====================
import random
from torch.utils.data import Dataset
from cellid_encoding import make_chunks, encode_events

rng = random.Random(CONFIG["seed"])

# Split contexte/cible PAR UTILISATEUR : chaque utilisateur contribue son préfixe
# (context_fraction) à l'entraînement, le reste sert de cible de validation (jamais entraînée).
context_users = []
for uid, cells, hours in users_train:
    cut = split_index(len(cells), CONFIG["context_fraction"])
    context_users.append((uid, cells[:cut], hours[:cut] if hours is not None else None))
print(f"Contexte d'entraînement : {len(context_users)} utilisateurs, "
      f"{CONFIG['context_fraction']:.0%} de chaque séquence (le reste sert de cible de validation)")

PREFIX_IDS = tokenizer(CONFIG["prefix_text"], add_special_tokens=False).input_ids
CELL_TO_ID = {c: i for c, i in zip(VOCAB, CELL_TOKEN_IDS)}
HOUR_TO_ID = {h: i for h, i in zip(HOUR_VOCAB, HOUR_TOKEN_IDS)} if HAS_HOURS else {}

def encode_sequence(cells, hours, n_masked_cells=0):
    """prefix + (heure, cell_id) interleavés (ou cell_id seuls si hours est None) ;
    labels -100 sur le préfixe, les heures, et les n_masked_cells premiers cell_id."""
    ids, labels, weights = encode_events(
        cells, hours, CELL_TO_ID, HOUR_TO_ID, PREFIX_IDS,
        CONFIG["transition_windows"], n_masked_cells=n_masked_cells)
    L = CONFIG["max_seq_len"]
    ids, labels, weights = ids[:L], labels[:L], weights[:L]
    return {"input_ids": ids, "labels": labels, "weights": weights, "attention_mask": [1] * len(ids)}

def make_chunks_for(cells, chunk_len, chunk_stride):
    return make_chunks(len(cells), chunk_len, chunk_stride)

class CellDataset(Dataset):
    def __init__(self, users):
        self.items = []
        for _, cells, hours in users:
            for start, end, ctx in make_chunks_for(cells, CONFIG["chunk_len"], CONFIG["chunk_stride"]):
                chunk_hours = hours[start:end] if hours is not None else None
                self.items.append(encode_sequence(cells[start:end], chunk_hours, n_masked_cells=ctx))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    pad = tokenizer.pad_token_id
    return {
        "input_ids":      torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] + [0] * (L - len(b["attention_mask"])) for b in batch]),
        "labels":         torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch]),
        "weights":        torch.tensor([b["weights"] + [1.0] * (L - len(b["weights"])) for b in batch]),
    }

train_ds = CellDataset(context_users)
print(f"{len(train_ds)} fenêtres d'entraînement (chunk={CONFIG['chunk_len']}, stride={CONFIG['chunk_stride']})")
```

- [ ] **Step 2: Manual verification**

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0, 'python')
from cellid_encoding import make_chunks, encode_events, HOUR_VOCAB
cell_to_id = {'A': 100, 'B': 101}
hour_to_id = {h: 200 + i for i, h in enumerate(HOUR_VOCAB)}
ids, labels, weights = encode_events(['A','B'], [5, 19], cell_to_id, hour_to_id, [1], [(4,6,2.0),(18,20,2.0)])
assert weights == [1.0, 1.0, 2.0, 1.0, 2.0], weights
print('encode_events wiring OK', ids, labels, weights)
"`
Expected: `encode_events wiring OK ...` with the asserted weights (both events fall in a transition window: hour 5 in [4,6), hour 19 in [18,20)).

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "feat(notebook): build training dataset via shared event encoding"
```

---

### Task 10: Notebook sanity-check cell — strip `weights` before the raw forward call

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `143c9f6c` (`SANITY CHECK AVANT ENTRAÎNEMENT`)

**Interfaces:**
- Consumes: `train_ds`, `collate` (Task 9). `collate`'s output now includes a `"weights"` key that `model(**batch)` cannot accept directly — this cell calls the model directly (not through `Trainer`), so it must drop that key itself.

- [ ] **Step 1: Replace the cell**

Use `NotebookEdit` with `cell_id="143c9f6c"`, `edit_mode="replace"`:

```python
# ==================== SANITY CHECK AVANT ENTRAÎNEMENT ====================
# Détecte à l'avance la cause classique du crash CUDA « device-side assert » :
# des ids de tokens qui dépassent la taille des tables embed_tokens / lm_head.
n_embed = model.get_input_embeddings().weight.shape[0]
out_emb = model.get_output_embeddings()
n_out = out_emb.weight.shape[0] if out_emb is not None else n_embed
max_id = max(max(CELL_TOKEN_IDS), max(PREFIX_IDS), *(HOUR_TOKEN_IDS or [0]))
print(f"tokenizer: {len(tokenizer)} | embed_tokens: {n_embed} | lm_head: {n_out} | id max utilisé: {max_id}")
assert max_id < n_embed and max_id < n_out, (
    "Ids de tokens hors des tables d'embedding : le resize_token_embeddings n'a pas été appliqué "
    "correctement. Relancez la cellule du modèle (Runtime > Restart sur Colab si besoin).")

# Toute valeur de label négative doit être EXACTEMENT -100 (ignore_index par défaut de la loss
# PyTorch) : une autre valeur négative (ex. -500) provoque un cryptique « IndexError: Target -500
# is out of bounds » au premier forward, loin de sa cause réelle. On le détecte ici avec un
# message clair, avant l'entraînement.
bad_labels = {l for item in (train_ds[0], train_ds[1]) for l in item["labels"] if l < 0 and l != -100}
assert not bad_labels, (
    f"Labels invalides détectés {bad_labels} — seule la valeur -100 masque la loss "
    "(vérifiez encode_sequence() dans la cellule DATASETS).")

# Un pas forward/backward sur un vrai batch : toute erreur apparaît ICI avec un message clair,
# plutôt qu'en plein entraînement de façon asynchrone. "weights" est retiré du batch : ce n'est
# pas un argument de model.forward(), seul WeightedLossTrainer.compute_loss (cellule ENTRAÎNEMENT)
# sait le consommer.
batch = collate([train_ds[0], train_ds[1]])
model_inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "weights"}
loss = model(**model_inputs).loss
loss.backward()
model.zero_grad()
print(f"forward/backward OK — loss initiale : {float(loss.detach()):.2f}")
```

- [ ] **Step 2: Manual verification**

Covered by Task 16's end-to-end run (needs the real model loaded). Confirm no syntax error:

Run: `.venv/bin/python -c "
import json
nb = json.load(open('train_cellid_llm.ipynb'))
src = next(c['source'] for c in nb['cells'] if c.get('id') == '143c9f6c')
compile(src, 'cell', 'exec')
print('syntax OK')
"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "fix(notebook): strip weights key before the direct sanity-check forward call"
```

---

### Task 11: Notebook evaluation cell — in/out transition-window accuracy

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `5fb3064c` (`ÉVALUATION : accuracy top-k`)

**Interfaces:**
- Consumes: `event_token_positions`, `in_transition_window` from `cellid_encoding` (Task 2); `CELL_TO_ID` (Task 9); `encode_sequence(cells, hours)` (Task 9, now 2-arg).
- Produces: `evaluate_cell_accuracy(model, users, ks=(1,3,5), context_fraction=None, transition_windows=None) -> dict` with existing keys (`top{k}`, `top1_seen`, `n`) unchanged, plus new keys `top1_seen_in_window`, `top1_seen_out_window` when `transition_windows` is passed and hours are available (used by Task 13; existing callers in Tasks 8/12's `AccuracyStopCallback` keep working since they don't pass `transition_windows` and only read the pre-existing keys).

- [ ] **Step 1: Replace the cell**

Use `NotebookEdit` with `cell_id="5fb3064c"`, `edit_mode="replace"`:

```python
# ==================== ÉVALUATION : accuracy top-k sur le prochain cell_id ====================
from cellid_encoding import event_token_positions, in_transition_window

CELL_IDS_T = torch.tensor(CELL_TOKEN_IDS)
CELL_POS = {tid: i for i, tid in enumerate(CELL_TOKEN_IDS)}   # id de token -> indice dans VOCAB

@torch.no_grad()
def evaluate_cell_accuracy(model, users, ks=(1, 3, 5), context_fraction=None, transition_windows=None):
    """Teacher forcing : pour chaque position i>=1, prédit le cell i depuis le contexte 0..i-1
    (toujours le VRAI historique, heures comprises si disponibles).
    Si context_fraction est fourni, ne compte dans l'accuracy que les positions de la CIBLE.
    Si transition_windows est fourni (et que les heures existent), rapporte en plus
    top1_seen_in_window / top1_seen_out_window (même métrique pilote que top1_seen, mais
    calculée séparément dans / hors créneaux de transition).
    Deux variantes top-k : classique (argmax sur les 264 cell_id) et top1_seen (argmax
    restreint aux cellules DÉJÀ VUES dans le préfixe de l'utilisateur)."""
    was_training = model.training
    model.eval()
    cell_ids_dev = CELL_IDS_T.to(DEVICE)
    kmax = max(ks)
    hits = {k: 0 for k in ks}
    hits_seen = 0
    total = 0
    hits_seen_in = tot_in = hits_seen_out = tot_out = 0
    n_prefix = len(PREFIX_IDS)
    for _, cells, hours in users:
        if len(cells) < 2:
            continue
        min_j = split_index(len(cells), context_fraction) if context_fraction is not None else 1
        enc = encode_sequence(cells, hours)
        ids = torch.tensor([enc["input_ids"]], device=DEVICE)
        logits = model(input_ids=ids).logits[0][:, cell_ids_dev].float().cpu()
        positions = event_token_positions(len(cells), has_hours=hours is not None)
        seen = torch.zeros(len(CELL_TOKEN_IDS), dtype=torch.bool)
        for j in range(len(cells)):
            token_pos = n_prefix + positions[j]
            if token_pos >= len(enc["input_ids"]):
                break   # séquence tronquée par max_seq_len : rien de plus à évaluer
            target_pos = CELL_POS[CELL_TO_ID[cells[j]]]
            if j == 0:
                seen[target_pos] = True
                continue
            if j >= min_j:
                scores = logits[token_pos - 1]
                top = scores.topk(kmax).indices.tolist()
                for k in ks:
                    hits[k] += int(target_pos in top[:k])
                masked = scores.clone()
                masked[~seen] = float("-inf")
                hit_seen = int(int(masked.argmax()) == target_pos)
                hits_seen += hit_seen
                total += 1
                if transition_windows is not None and hours is not None:
                    if in_transition_window(hours[j], transition_windows):
                        tot_in += 1; hits_seen_in += hit_seen
                    else:
                        tot_out += 1; hits_seen_out += hit_seen
            seen[target_pos] = True
    if was_training:
        model.train()
    if total == 0:
        result = {**{f"top{k}": float("nan") for k in ks}, "top1_seen": float("nan"), "n": 0}
    else:
        result = {**{f"top{k}": hits[k] / total for k in ks}, "top1_seen": hits_seen / total, "n": total}
    if transition_windows is not None:
        result["top1_seen_in_window"] = hits_seen_in / tot_in if tot_in else float("nan")
        result["top1_seen_out_window"] = hits_seen_out / tot_out if tot_out else float("nan")
    return result
```

- [ ] **Step 2: Manual verification**

Covered by Task 16 (needs the real model/tokenizer). Confirm no syntax error:

Run: `.venv/bin/python -c "
import json
nb = json.load(open('train_cellid_llm.ipynb'))
src = next(c['source'] for c in nb['cells'] if c.get('id') == '5fb3064c')
compile(src, 'cell', 'exec')
print('syntax OK')
"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "feat(notebook): report accuracy split by transition window"
```

---

### Task 12: Notebook training cell — weighted-loss `Trainer`

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `400843a2` (`ENTRAÎNEMENT`)

**Interfaces:**
- Consumes: `train_ds`, `collate` (Task 9, `collate` output includes `"weights"`).
- Produces: `trainer` built from `WeightedLossTrainer` instead of `Trainer` (no interface change for later cells — they only read `trainer`, `acc_cb`, `ram_cb`, unchanged).

- [ ] **Step 1: Replace the cell**

Use `NotebookEdit` with `cell_id="400843a2"`, `edit_mode="replace"`:

```python
# ==================== ENTRAÎNEMENT ====================
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F

class WeightedLossTrainer(Trainer):
    """Trainer standard, sauf que la cross-entropy est pondérée par position :
    les cell_id dont l'heure tombe dans un créneau de transition (voir
    CONFIG["transition_windows"]) comptent plus fort dans la loss, pour
    pousser le modèle à raisonner plutôt que recopier le cell_id précédent
    sur ces positions. Sans heures (weights == 1.0 partout), équivalent à la
    loss non pondérée standard."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        mask = (shift_labels.view(-1) != -100).float()
        weighted = losses * shift_weights.reshape(-1) * mask
        loss = weighted.sum() / mask.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss

if DEVICE == "cuda":
    batch_size, grad_accum = (4, 2) if BIG_MODEL else (8, 1)
elif DEVICE == "mps":
    batch_size, grad_accum = 4, 2
else:
    batch_size, grad_accum = 1, 8

targs = TrainingArguments(
    output_dir=str(CKPT_DIR),
    num_train_epochs=CONFIG["max_epochs"],
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=CONFIG["learning_rate"],
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=20,
    weight_decay=CONFIG["weight_decay"],
    save_strategy="epoch",
    save_total_limit=CONFIG["save_total_limit"],   # guard disque : max 2 checkpoints
    bf16=(DEVICE == "cuda" and DTYPE == torch.bfloat16),
    fp16=(DEVICE == "cuda" and DTYPE == torch.float16),
    report_to=[],
    seed=CONFIG["seed"],
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

ram_cb = RamGuardCallback()
acc_cb = AccuracyStopCallback(model)
trainer = WeightedLossTrainer(model=model, args=targs, train_dataset=train_ds,
                              data_collator=collate, callbacks=[ram_cb, acc_cb])

has_ckpt = CKPT_DIR.exists() and any(CKPT_DIR.glob("checkpoint-*"))
resume = CONFIG["resume_from_checkpoint"] and has_ckpt
trainer.train(resume_from_checkpoint=True if resume else None)

stop_reason = ("guard RAM déclenché (reprendre avec resume_from_checkpoint=True)" if ram_cb.triggered
               else acc_cb.reason or "plafond d'epochs atteint")
print(f"\nEntraînement terminé — raison de l'arrêt : {stop_reason}")
print(f"Meilleure accuracy validation : {acc_cb.best:.1%} (modèle sauvegardé dans {BEST_DIR})")
```

- [ ] **Step 2: Manual verification**

Isolate and unit-test the weighting math without a real model, using plain tensors:

Run: `.venv/bin/python -c "
import torch
import torch.nn.functional as F

logits = torch.randn(1, 4, 5)        # batch=1, seq_len=4, vocab=5
labels = torch.tensor([[-100, 2, -100, 3]])
weights = torch.tensor([[1.0, 1.0, 1.0, 2.0]])

shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
shift_weights = weights[..., 1:].contiguous()
losses = F.cross_entropy(shift_logits.view(-1, 5), shift_labels.view(-1), ignore_index=-100, reduction='none')
mask = (shift_labels.view(-1) != -100).float()
weighted = losses * shift_weights.reshape(-1) * mask
loss = weighted.sum() / mask.sum().clamp(min=1)
assert mask.sum().item() == 2      # only 2 real labels among the 3 shifted positions
assert loss.item() > 0
print('weighted loss math OK, loss =', loss.item())
"`
Expected: `weighted loss math OK, loss = <some positive float>`, no exception.

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "feat(notebook): weighted-loss Trainer for transition-window emphasis"
```

---

### Task 13: Notebook final-evaluation cell — report the breakdown

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `56616448` (`ÉVALUATION FINALE SUR LE TEST`)

**Interfaces:**
- Consumes: `evaluate_cell_accuracy(..., transition_windows=...)` (Task 11), `HAS_HOURS` (Task 7).

- [ ] **Step 1: Replace the cell**

Use `NotebookEdit` with `cell_id="56616448"`, `edit_mode="replace"`:

```python
# ==================== ÉVALUATION FINALE SUR LE TEST (utilisateurs jamais vus) ====================
# On évalue le MEILLEUR modèle (pic de validation), pas celui de la dernière epoch
if acc_cb.best_params is not None:
    model.load_state_dict(acc_cb.best_params, strict=False)
    print(f"Meilleur modèle rechargé pour l'évaluation (val top1_seen = {acc_cb.best:.1%})\n")

# Même protocole que la validation : on donne context_fraction de la séquence de chaque
# utilisateur de test en contexte, et on évalue uniquement sa capacité à prédire la suite.
TEST_CONTEXT_FRACTIONS = [CONFIG["context_fraction"], 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]

results_by_fraction = {}
for frac in TEST_CONTEXT_FRACTIONS:
    m = evaluate_cell_accuracy(model, users_test, context_fraction=frac,
                                transition_windows=CONFIG["transition_windows"])
    baseline_frac = baseline_persistence(users_test, frac)
    results_by_fraction[frac] = (m, baseline_frac)

print("Contexte donné -> accuracy sur la suite (mêmes utilisateurs de test) :")
for frac, (m, baseline_frac) in results_by_fraction.items():
    score = max(m["top1"], m["top1_seen"])
    ref = " (référence)" if frac == CONFIG["context_fraction"] else ""
    print(f"  contexte {frac:.0%}{ref:<12} | n={m['n']:4d} | "
          f"top1={m['top1']:.1%}  top1_seen={m['top1_seen']:.1%}  "
          f"top3={m['top3']:.1%}  top5={m['top5']:.1%}  "
          f"(baseline persistance {baseline_frac:.1%}, écart {score - baseline_frac:+.1%})")
    if HAS_HOURS:
        print(f"      dans créneaux de transition : top1_seen={m['top1_seen_in_window']:.1%}  |  "
              f"hors créneaux : top1_seen={m['top1_seen_out_window']:.1%}")

# Détail par utilisateur, pour TOUTES les fractions (verbeux, mais demandé explicitement) :
for frac in TEST_CONTEXT_FRACTIONS:
    ref = " (référence)" if frac == CONFIG["context_fraction"] else ""
    print(f"\nAccuracy top-1 (répertoire) par utilisateur de test — contexte {frac:.0%}{ref} :")
    for uid, cells, hours in users_test:
        mu = evaluate_cell_accuracy(model, [(uid, cells, hours)], ks=(1,), context_fraction=frac)
        if mu["n"] == 0:   # séquence trop courte pour dégager une cible avec cette fraction
            print(f"  {uid:>10} | (pas assez d'événements pour une cible à {frac:.0%} de contexte)")
            continue
        bar = "█" * round(mu["top1_seen"] * 30)
        print(f"  {uid:>10} | {mu['top1_seen']:6.1%} sur {mu['n']:4d} prédictions | {bar}")
```

- [ ] **Step 2: Manual verification**

Covered by Task 16's end-to-end run. Confirm no syntax error:

Run: `.venv/bin/python -c "
import json
nb = json.load(open('train_cellid_llm.ipynb'))
src = next(c['source'] for c in nb['cells'] if c.get('id') == '56616448')
compile(src, 'cell', 'exec')
print('syntax OK')
"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "feat(notebook): print transition-window accuracy breakdown in final eval"
```

---

### Task 14: Notebook demo cell — hour-aware `predict_next_cells`

**Files:**
- Modify: `train_cellid_llm.ipynb`, cell `b3dba5bf` (`SAUVEGARDE FINALE + DÉMO D'INFÉRENCE`)

**Interfaces:**
- Consumes: `CELL_TO_ID`, `HOUR_TO_ID`, `HAS_HOURS` (Tasks 7–9), `hour_token` (`cellid_encoding`, Task 2), `split_index` (Task 7's import).

- [ ] **Step 1: Replace the cell**

Use `NotebookEdit` with `cell_id="b3dba5bf"`, `edit_mode="replace"`:

```python
# ==================== SAUVEGARDE FINALE + DÉMO D'INFÉRENCE ====================
from cellid_encoding import hour_token

FINAL_DIR = Path(CONFIG["output_dir"]) / "final_model"
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"Adaptateur LoRA + tokenizer sauvegardés dans {FINAL_DIR}\n")

@torch.no_grad()
def predict_next_cells(prefix_cells, prefix_hours, next_hours, top_k=3, restrict_to_seen=True):
    """Prochains cell_id les plus probables (génération greedy).
    prefix_hours : heures des cell_id du préfixe (None si HAS_HOURS est False).
    next_hours : heure de chaque pas à prédire -- connue hors-ligne ici (users_test) ;
    en production ce serait l'heure courante au moment de chaque prédiction (toujours connue).
    restrict_to_seen : ne propose que des cellules du répertoire déjà observé de l'utilisateur."""
    model.eval()
    ids = list(PREFIX_IDS)
    for i, cell in enumerate(prefix_cells):
        if HAS_HOURS:
            ids.append(HOUR_TO_ID[hour_token(prefix_hours[i])])
        ids.append(CELL_TO_ID[cell])
    cell_ids_dev = CELL_IDS_T.to(DEVICE)
    allowed = torch.tensor([c in set(prefix_cells) for c in VOCAB]) if restrict_to_seen else None
    steps = []
    for step_hour in next_hours:
        if HAS_HOURS:
            ids.append(HOUR_TO_ID[hour_token(step_hour)])
        logits = model(input_ids=torch.tensor([ids], device=DEVICE)).logits[0, -1].float()
        scores = logits[cell_ids_dev].cpu()
        if allowed is not None:
            scores[~allowed] = float("-inf")
        probs = scores.softmax(-1)
        topv, topi = probs.topk(top_k)
        steps.append([(VOCAB[i], float(v)) for v, i in zip(topv, topi)])
        ids.append(int(cell_ids_dev[topi[0]]))
    return steps

# Démo sur PLUSIEURS utilisateurs de test : chacun a un comportement différent.
# Même protocole que l'évaluation finale : on donne context_fraction de la séquence en
# préfixe, et on compare la suite prédite à la vraie suite (réservée, jamais entraînée).
N_DEMO_USERS = 5
N_STEPS = 3
for demo_user, demo_cells, demo_hours in users_test[:N_DEMO_USERS]:
    n_ctx = split_index(len(demo_cells), CONFIG["context_fraction"])
    n_steps = min(N_STEPS, len(demo_cells) - n_ctx)
    prefix_hours = demo_hours[:n_ctx] if demo_hours is not None else None
    next_hours = (demo_hours[n_ctx:n_ctx + n_steps] if demo_hours is not None
                  else [None] * n_steps)
    print(f"Utilisateur {demo_user} — préfixe ({CONFIG['context_fraction']:.0%}) : {' '.join(demo_cells[:n_ctx])}")
    preds = predict_next_cells(demo_cells[:n_ctx], prefix_hours, next_hours)
    for step, cands in enumerate(preds, 1):
        print(f"  +{step} prédit : " + "  |  ".join(f"{c} ({p:.0%})" for c, p in cands))
    print(f"  réel       : {' '.join(demo_cells[n_ctx:n_ctx + n_steps])}\n")
```

- [ ] **Step 2: Manual verification**

Covered by Task 16's end-to-end run. Confirm no syntax error:

Run: `.venv/bin/python -c "
import json
nb = json.load(open('train_cellid_llm.ipynb'))
src = next(c['source'] for c in nb['cells'] if c.get('id') == 'b3dba5bf')
compile(src, 'cell', 'exec')
print('syntax OK')
"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train_cellid_llm.ipynb
git commit -m "feat(notebook): hour-aware inference demo"
```

---

### Task 15: Documentation — `CLAUDE.md` and notebook Notes cell

**Files:**
- Modify: `CLAUDE.md`
- Modify: `train_cellid_llm.ipynb`, cell `57facbed` (`Notes`, markdown)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add the timestamp/transition-window section to `CLAUDE.md`**

Append this new section at the end of `CLAUDE.md` (after the existing `cell_id` schema section):

```markdown

## Timestamps and transition windows

Each `cell_id` in a raw day record is paired with a **timestamp in seconds
since local midnight** (`0 … 86399`), and the timestamp always **follows**
the `cell_id` it belongs to in the interleaved columns (`…; cell_id; ts;
cell_id; ts; …`). `ts // 3600` gives the hour of day (0–23).

**Transition windows** are hours where a user has a statistically higher
chance of a home↔activity `cell_id` change — by default **04h–06h** and
**18h–20h**. Training should be pushed to reason about a possible change
during these windows rather than defaulting to "repeat the last `cell_id`
seen" (an otherwise strong baseline). This is implemented as a **loss
weight** applied to predictions whose event falls inside a window (see
`CONFIG["transition_windows"]` in `train_cellid_llm.ipynb` — a list of
`(start_hour, end_hour, weight)` tuples, easily edited in one place; a
future iteration may key this by `user_id` instead of a single global
default, since different users can plausibly have different transition
hours).

The real-data pipeline (`make data-train`, see `python/split_sample_for_training.py`
and `python/format_for_train.py`) keeps this timestamp instead of dropping
it, and turns it into the hour-of-day signal above.
```

- [ ] **Step 2: Update the notebook Notes cell**

Use `NotebookEdit` with `cell_id="57facbed"`, `edit_mode="replace"`, `cell_type="markdown"`:

```markdown
## Notes

- **Changer le nombre d'utilisateurs** : modifier `N_USERS` dans la cellule CONFIG (500, 100, ou
  toute autre valeur). `train_file`/`test_file` s'adaptent automatiquement, et la cellule PRÉFLIGHT
  génère les fichiers manquants via `generate_sample_users.py` (sauf sur Colab où il faut uploader).
- **Utiliser des données réelles** : lancer `make data-train` (voir la racine du repo — override
  avec `N_TRAIN=`/`N_TEST=`/`SRC=`), puis pointer `CONFIG["train_file"]`/`CONFIG["test_file"]`
  directement vers les deux fichiers produits dans `data/dataset_for_training/`. Ces fichiers
  portent les heures réelles de chaque `cell_id` ; les jeux synthétiques (`generate_sample_users.py`)
  n'en portent pas et continuent de fonctionner à l'identique (mode heure désactivé automatiquement).
- **Créneaux de transition** : `CONFIG["transition_windows"]` (04h–06h / 18h–20h par défaut) pondère
  la loss pour pousser le modèle à raisonner plutôt qu'à recopier le dernier `cell_id` dans ces
  créneaux. Modifiable en une ligne, sans regénérer les données.
- **Changer de modèle de base** : commenter/décommenter `model_name` dans CONFIG.
  - `mistralai/Mistral-7B-v0.1` : **GPU Colab obligatoire** (chargé automatiquement en 4-bit / QLoRA). Modèle *gated* : acceptez d'abord la licence sur [sa page Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1) avec le compte du token.
  - `Qwen/Qwen2.5-0.5B-Instruct` (actif par défaut) : léger, fonctionne en local (MPS) comme sur Colab.
- **Reprise après interruption** (guard RAM ou coupure) : mettre `CONFIG["resume_from_checkpoint"] = True` et relancer les cellules — l'entraînement repart du dernier checkpoint.
- **Meilleur modèle** : le meilleur état (accuracy validation) est toujours dans `outputs/best_model` ; le modèle final est dans `outputs/final_model`.
- **Recharger le modèle sans réentraîner** :
  ```python
  from peft import PeftModel
  base = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"], dtype=DTYPE)
  tokenizer = AutoTokenizer.from_pretrained("outputs/final_model")
  base.resize_token_embeddings(len(tokenizer))
  model = PeftModel.from_pretrained(base, "outputs/final_model").to(DEVICE)
  ```
- **Colab** : uploader `train_cellid_llm.ipynb`, `python/cellid_encoding.py`, et les deux `.jsonl`
  (panneau Fichiers), runtime GPU T4. L'entraînement y prend ~2-5 min. Le `torchao` préinstallé
  (incompatible peft) est retiré automatiquement.
- **Token Hugging Face** : préférez le stocker dans un secret Colab nommé `HF_TOKEN` (icône 🔑 à gauche) ou la variable d'environnement `HF_TOKEN`, plutôt qu'en clair dans `CONFIG["hf_token"]`. **Si vous partagez ce notebook avec un token en clair, révoquez-le** sur huggingface.co/settings/tokens.
- **Ajuster si < 80 %** : augmenter `max_epochs`, passer `lora_r` à 32 / `lora_alpha` à 64, ou relancer avec un autre `seed`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md train_cellid_llm.ipynb
git commit -m "docs: document timestamps/transition windows and the real-data pipeline"
```

---

### Task 16: End-to-end smoke verification

**Files:** none created/modified (verification only, uses a throwaway copy outside the repo).

**Interfaces:** none — this task exercises every interface produced by Tasks 1–15 together.

- [ ] **Step 1: Build a tiny real dataset**

Run: `make data-train N_TRAIN=8 N_TEST=4`
Expected: `data/dataset_for_training/8_4_sample_users_train.jsonl` and `..._test.jsonl` exist (8 and 4 lines respectively), each line has non-empty `cells` and `hours` arrays of equal length.

- [ ] **Step 2: Duplicate the notebook for a fast smoke run**

Run: `cp train_cellid_llm.ipynb /tmp/smoke_train_cellid_llm.ipynb`

- [ ] **Step 3: Point the copy at the tiny real dataset and shrink the training budget**

Read `/tmp/smoke_train_cellid_llm.ipynb` (required before `NotebookEdit`), then use `NotebookEdit` on it (`notebook_path=/tmp/smoke_train_cellid_llm.ipynb`, `cell_id="3538fcec"`, `edit_mode="replace"`) with the same source as Task 6's Step 1, except these three lines changed:

```python
    "train_file": "data/dataset_for_training/8_4_sample_users_train.jsonl",
    "test_file": "data/dataset_for_training/8_4_sample_users_test.jsonl",
```

and:

```python
    "early_stop_patience": 1,
    "max_epochs": 1,
```

(replacing the corresponding lines from the Task 6 source — everything else identical).

- [ ] **Step 4: Execute the whole notebook end to end**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 --output /tmp/smoke_executed.ipynb /tmp/smoke_train_cellid_llm.ipynb`
Expected: exits with status 0 (no cell raised an exception). If a cell fails, `nbconvert` prints the failing cell's traceback — fix the corresponding task's cell and re-run this step.

- [ ] **Step 5: Confirm the time-aware code paths actually ran**

Run: `.venv/bin/jupyter nbconvert --to script --stdout /tmp/smoke_executed.ipynb > /tmp/smoke_executed.py 2>/dev/null; grep -o '"text/plain": "[^"]*"' /tmp/smoke_executed.ipynb | grep -E "tokens heure|Heures disponibles|dans créneaux de transition" | head -10`

Expected: at least these three lines appear among the matches (values may differ):
- `... tokens heure ajoutés au tokenizer` (a positive count, from Task 8)
- `Heures disponibles dans les données : oui` (from Task 7)
- `... dans créneaux de transition : top1_seen=...` (from Task 13, and/or the baseline breakdown from Task 7)

- [ ] **Step 6: Run the full pytest suite one more time**

Run: `.venv/bin/pytest -v`
Expected: all tests from Tasks 2–4 PASS.

- [ ] **Step 7: Clean up the throwaway files**

Run: `rm -f /tmp/smoke_train_cellid_llm.ipynb /tmp/smoke_executed.ipynb /tmp/smoke_executed.py`

No commit for this task — it only verifies work already committed in Tasks 1–15. If Step 4 or Step 5 uncovers a bug, fix it inside the relevant earlier task's file, re-run that task's own verification, then re-run this task from Step 1.
