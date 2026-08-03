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
weight** applied to predictions whose event falls inside a window.

Transition windows are now **keyed by behavioural group**, not by a single
global default (see below) — the iteration this file previously anticipated.
`CONFIG["transition_windows"]` still exists in `train_cellid_llm.ipynb` as a
list of `(start_hour, end_hour, weight)` tuples, but only as the fallback used
when groups are disabled (`use_groups=False`) or the dataset carries no hours.
The default `04h–06h` / `18h–20h` guess turned out to be **misplaced** on the
real data: `04h–06h` is *easier* than average for every group, so upweighting
it pushed gradient at positions the baseline already gets right.

The real-data pipeline (`make data-train`, see `python/split_sample_for_training.py`
and `python/format_for_train.py`) keeps this timestamp instead of dropping
it, and turns it into the hour-of-day signal above.

## Behavioural user groups (cross-training)

`python/user_groups.py` groups users by their **site-change rate per hour
band** — how often the *physical site* changes, as a function of time of day.
`train_cellid_llm.ipynb` prepends a dedicated `<Gk>` token to each sequence, so
users in the same group inform each other instead of being N independent
sequences. `make groups` prints the full report.

Two things matter when working on this:

1. **Movement is measured on the site root, never on the raw `cell_id`.**
   Because a `cell_id` is `<tech><root><zone>`, the same antenna appears under
   several ids. On the real data **12.4%** of consecutive-event transitions are
   "same site root, different radio/sector" — the phone re-attached to another
   cell of the same antenna without the user going anywhere. Use
   `user_groups.site_root()` before comparing two `cell_id` for movement.

2. **A group must be recoverable from a context prefix alone.** At inference
   only the user's past is available, so the notebook detects the group from the
   context prefix (`group_mode="detected"`), never from the target. Anything
   that makes groups depend on the whole sequence breaks that guarantee — the
   `"oracle"` mode exists only as a diagnostic upper bound.

The four groups found on `400_users_train.jsonl` (numbered by increasing
mobility; G2 and G3 are near mirror images in time):

| group | share | site-change (sleep/morning/day/evening) | persistence | derived windows |
|---|---|---|---|---|
| G0 | 28% | 28/32/34/38% | 56% | none — flat, nothing worth upweighting |
| G1 | 35% | 31/47/46/45% | 42% | 18-19h ×1.4, 20-21h ×1.7 |
| G2 | 14% | 30/47/**67/60**% | 20% | 13-14h ×1.7, 16-21h ×2.5 |
| G3 | 23% | 35/**67/65**/43% | 18% | 07-08h ×2.0, 10-13h ×2.3 |

The group label explains **69%** of the variance in per-user predictability on
train and **72% on test** (centroids fitted on train only). Note that sequence
length is a partial confound — `log(n)` alone explains 48% — but the label still
explains 38% of the residual after removing it, and the scale-free
morning-minus-evening shape asymmetry correlates only −0.28 with `log(n)`.
