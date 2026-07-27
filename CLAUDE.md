# CLAUDE.md

## Data schema — what a `cell_id` actually is

A `cell_id` (e.g. `UKVCHE102`, `BKVPAN1`) is not an opaque location token — it encodes three distinct pieces of information:

1. **1st letter — radio technology of the cell**
   - `B` or `D` → 2G
   - `U` or `V` → 3G

2. **Middle letters — root ID of the base station (the physical antenna/site)**
   - This is the actual physical location. The same physical site can appear under multiple `cell_id` values if it has several technology generations and/or sectors.

3. **Trailing digits — zone/sector of that base station**
   - 2G: usually `1`, `2`, `3`, or `4` — but sites with more sectors can go as low as `0` or up to `5`-`9` (e.g. `BKVHRO0`, `BKVPUP9`, `BKVTES5`). Treat any single digit `0`-`9` as valid, not just `1`-`4`.
   - 3G: `<zone digit><sector 2 digits>`, zone is `1` or `2`, sector is usually `01`-`04` — but sites with more sectors per zone extend up to `09` (e.g. `UKVPUP109`, `UKVPUP209`). Treat any `<1|2><01-09>` as valid, not just `101`-`104`/`201`-`204`.

So `cell_id = <tech letter><site root><zone digits>`. Two `cell_id` strings can share the same site root while differing only in technology letter and/or zone digits — that represents the same physical location, observed through a different radio cell.

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
