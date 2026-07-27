# Time-aware cell_id prediction — design

**Date:** 2026-07-27
**Author:** Olivier Recher (with Claude)

## Context

`train_cellid_llm.ipynb` currently predicts a user's next `cell_id` from a bare
sequence of `cell_id` tokens — no notion of time of day. Real user movement has
**transition windows** (statistically higher chance of a home↔activity cell
change): by default **04h–06h** and **18h–20h**. Today the model (and the
"repeat last cell_id" baseline it must beat) has no way to know it is inside
such a window, so nothing pushes it to reason harder there instead of trivially
copying the previous `cell_id`.

There is also no existing pipeline turning the raw daily CSVs in
`data/sample_for_training/` into training data — `generate_sample_users.py`
only produces **synthetic** sequences (no real timestamps at all). This design
adds a real-data pipeline that preserves the timestamp attached to each
`cell_id`, and a training mechanism that uses it.

Reference implementation copied from and adapted from
`/Users/olivierrecher/Documents/IMT/SIO/work/Stage_2A/Makefile` +
`Python/split_sample_for_training.py` + `Python/format_for_train.py` (which
drop timestamps entirely — the key difference here is that timestamps are
**kept** and turned into an hour-of-day signal).

## Decisions already settled

- **Scope**: adapt the Stage_2A `Makefile` + its two data scripts to this
  repo's layout (`data/` instead of `Database/`, scripts under a `python/`
  folder — same idea as Stage_2A's `Python/`, just lowercase to match this
  repo's naming — `.venv` instead of `venv`), keeping timestamps this time.
- **Time granularity: hour-of-day (0–23), not minute/second.** Reasons (agreed
  with the user):
  1. Feasible as **dedicated tokens** (`<H00>`…`<H23>`, 24 of them, same
     pattern as the 264 `cell_id` tokens) — learnable embeddings. Second-level
     precision would need either an unlearnable 86 400-value vocabulary or
     ~8 extra text tokens per event (blowing the `max_seq_len=512` /
     `chunk_len=128` budget at the expense of actual `cell_id` content).
  2. Consistent with the rest of the project, which already reasons in hours /
     4-hour slots (6-bit presence code, 05h–19h activity window, hourly cell
     occupancy) — no existing use of minute/second granularity.
  - Elapsed-time-since-last-event is a *different* feature (dwell duration,
    not clock time) and is explicitly out of scope here.
- **The model sees the hour** (not just a hidden loss-weight signal): each
  event is encoded as an `<Hxx>` token immediately followed by its `cell_id`
  token. The hour token is always given (never predicted, label `-100`); only
  the `cell_id` token that follows carries the real target label. This is safe
  for both offline evaluation (hour known from the historical record) and
  production inference (the hour "now" is always known when a prediction is
  requested).
- **Transition-window config lives in the notebook's `CONFIG`**, not in a
  separate file, so it stays a one-line edit and the data files never need
  regenerating when the windows change:
  ```python
  CONFIG["transition_windows"] = [(4, 6, 2.0), (18, 20, 2.0)]  # (start_h, end_h, loss weight)
  ```
  Structure is a list now; documented as the future extension point for a
  per-`user_id` override (e.g. `{user_id: [(start,end,weight), ...]}` falling
  back to this global default) — **not implemented in this iteration**.

## 1. Data pipeline (new files, adapted from Stage_2A)

### `python/split_sample_for_training.py`

Adapted from Stage_2A's script of the same name. Differences from the
original:
- Default source: `data/sample_for_training/2014-03-12_sample_for_training.csv`
  (any file with the same structure works, passed positionally).
- Default output dir: `data/dataset_for_training/`.
- **Timestamps are kept**, not dropped: `parse_line` keeps
  `[userId, nRecord, cellId, ts, cellId, ts, ...]` (both halves of each
  interleaved pair), instead of `fields[8::2]` only.
- Same behaviour otherwise: random sample of `n_train + n_test` users,
  stratified train/test split by `nRecord` (quantile-binned) via
  `sklearn.model_selection.train_test_split`, Mann-Whitney length-balance
  check printed, written as one `;`-separated CSV
  (`{n_train}_{n_test}_sample_users.csv`) with train rows first.

### `python/format_for_train.py`

Adapted from Stage_2A's script of the same name. Differences:
- Input rows now carry `cellId, ts, cellId, ts, ...` (see above). For each
  record, `hour = int(ts) // 3600` (timestamp = seconds since local midnight,
  per `CLAUDE.md`; `ts` follows the `cell_id` it belongs to).
- Output JSONL keeps the existing human-readable `conversations` text
  (`"User <id> | <n> events | <cell> <cell> ..."`, **cell_id only**, for
  readability/back-compat with anything that reads that field) but adds two
  **top-level sibling fields** the notebook reads directly instead of
  re-parsing text:
  ```json
  {"conversations": [...],
   "user_id": "43",
   "cells": ["BSOHSL2", "BSOHSL2", "BKVDVR3", ...],
   "hours": [3, 4, 5, ...]}
  ```
- Same `--n-train`/`--n-test` exact-count split behavior (first N lines train,
  next M test).

### `Makefile` (new, repo root)

Adapted from Stage_2A's `Makefile`:
- `VENV := .venv` (this repo already creates `.venv` via `setup_venv.sh`);
  `install`/`clean`/`help` targets kept for parity but `data-train` does not
  require them to have been run through `make` specifically (works with the
  existing `.venv`).
- `data-train` target: same `N_TRAIN=400 N_TEST=100 SRC=...` override
  convention, calling the two scripts under `python/`:
  ```
  make data-train
  make data-train N_TRAIN=1000 N_TEST=200
  make data-train SRC=data/sample_for_training/2014-03-13_sample_for_training.csv
  ```
  Output: `data/dataset_for_training/{N_TRAIN}_{N_TEST}_sample_users_train.jsonl`
  and `..._test.jsonl`.
- `requirements.txt` gains `pandas`, `scikit-learn`, `scipy` (needed by the
  stratified split, mirroring Stage_2A's own requirements).

## 2. Notebook: time-aware sequence encoding

- `HOUR_VOCAB = [f"<H{h:02d}>" for h in range(24)]` added as dedicated tokens
  alongside the existing `VOCAB` (264 `cell_id`), same
  `trainable_token_indices` mechanism (embeddings for both sets trained).
- `load_users` reads the new `cells` / `hours` top-level JSON fields directly.
  **Backward compatibility**: if a JSONL file has no `hours` field (the
  existing synthetic `*_users_train/test.jsonl`), `hours` is `None` for that
  file and the sequence is encoded exactly as today (`cell_id` tokens only, no
  interleaving, no transition weighting) — old synthetic datasets keep working
  unchanged, time-awareness only activates for files that carry real hours.
- `encode_sequence`/`make_chunks` operate on a list of events;  when `hours` is
  present, each event becomes 2 tokens `[<Hxx>, cell_id]` with label `-100` on
  the hour position and the real cell label (or `-100` if in the masked
  prefix) on the cell position. `chunk_len`/`chunk_stride` in `CONFIG` keep
  their current meaning (number of *events*, not raw tokens).
- `collate` gains a `weights` tensor (parallel to `labels`, padded with `1.0`):
  `1.0` by default, `transition_windows`'s weight (e.g. `2.0`) at cell-label
  positions whose event hour falls inside a configured window.

## 3. Weighted loss

A `Trainer` subclass overrides `compute_loss`: pops `weights` from the batch,
computes per-token cross-entropy with `reduction="none"` and `ignore_index=-100`
against the model's logits, multiplies by `weights`, and takes the mean over
non-masked positions (matching the unweighted loss when all weights are `1.0`,
so nothing changes for hour-less datasets).

## 4. Evaluation: measuring the effect

`evaluate_cell_accuracy` (and `baseline_persistence`) report accuracy split
into **inside transition windows** vs. **outside**, in addition to the
existing overall top-1/top-3/top-5/top1_seen — so a run can show concretely
whether the weighted loss narrowed the gap between "copy last cell_id" and
real prediction specifically inside the transition windows it targets.

## 5. `CLAUDE.md` update

Add a section documenting: timestamps are seconds since local midnight and
follow the `cell_id` they belong to; transition windows (default 04h–06h /
18h–20h, configurable, one day a per-user override) are where a user
statistically has a higher chance of a home↔activity cell change, and where
training should be pushed to reason rather than default to repeating the last
`cell_id`.

## Out of scope for this iteration

- Per-user transition windows (noted as an extension point, not implemented).
- Minute/second-level time features or elapsed-time-since-last-event.
- Changes to `generate_sample_users.py` (synthetic data stays hour-less;
  notebook backward-compat handles that).
- Changes to the "unconditioned generation" vs. "prefix/continuation"
  framing debate from Stage_2A — this repo's notebook already does
  prefix/continuation via `context_fraction`, unaffected by this change.
