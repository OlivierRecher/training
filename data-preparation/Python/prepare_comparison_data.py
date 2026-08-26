"""Build the per-(user, day) join between the two activity-cell definitions.

Original (continuous stay, this repo, Stage_2A) is recomputed on the fly from
Database/no_duplicate for the reference period, at the 3 merge granularities.
Relaxed (discontinuous stay, Stage3A) is read from its pre-classified CSVs.

Output: a slim cached CSV (MyResults/intermediate/comparison_activity.csv) with,
per (user, day), the assigned activity cell of each definition at each merge
mode, the number of Original activity cells (nb), the Relaxed Same_Home_Activity
flag and the Relaxed home cell. Consumed by plot_comparison_definitions.py.

Also prints the reproduced 6-column comparison table as a verification gate
(must match Presentation/screen/comparision-between-definitions.png).

Read-only on all source data. Run:  python MyPython/prepare_comparison_data.py
"""
import csv
from collections import Counter
from pathlib import Path

from utils import DAYS, get_day, get_cell_code, get_cell_code2

# ------------------------------------------------------------------ paths ----
MAIN_DIR = Path(__file__).resolve().parent.parent          # Stage_2A/
ORIGINAL_INPUT_DIR = MAIN_DIR / "Database" / "no_duplicate"

# Cross-repo dependency: the Relaxed definition lives in the Stage3A project.
STAGE3A = MAIN_DIR.parent / "Stage3A"
RELAXED_DIR = STAGE3A / "Database" / "cd_142_dataset" / "classified_dataset_merge_simple"
RELAXED_SIMPLE_CSV = RELAXED_DIR / "classified_dataset_merge_simple.csv"
RELAXED_2G3G_CSV = RELAXED_DIR / "classified_dataset_merge_2g3g.csv"

OUT_DIR = MAIN_DIR / "MyResults" / "intermediate"
OUT_CSV = OUT_DIR / "comparison_activity.csv"

# ------------------------------------------------------------- parameters ----
TARGET_PERIOD = "Home_04h10-19h50_Activity_05h00-19h00"
MORNING = 15000            # 04h10 threshold (start offset for the activity search)
ACTIVITY_START = 18000     # 05h00
ACTIVITY_END = 68400       # 19h00
STAY_THRESHOLD = 18000     # 300 minutes
GAP = 4 * 3600             # 4h disconnection cutoff


# ------------------------------------------------- Original classifier -------
def _activity_window(cells, stamps):
    """Slice the records falling in the activity window [05h00, 19h00]."""
    n = len(stamps)
    i = 0
    while i < n and stamps[i] <= MORNING:
        i += 1
    while i < n and stamps[i] < ACTIVITY_START:
        i += 1
    j = i
    while j < n and stamps[j] <= ACTIVITY_END:
        j += 1
    return cells[i:j], stamps[i:j]


def original_activity(cells, stamps):
    """Continuous-stay activity detection (verbatim logic from Stage_2A).

    Returns (activity_cell_or_'' , nb_activity_cells). A cell qualifies when the
    user stays there >= 300 min *continuously* (the dwell counter is reset when
    the user leaves the cell or a >4h gap occurs).
    """
    if len(cells) < 2:
        return "", 0
    t = Counter()
    old_cell, old_stamp = cells[0], stamps[0]
    for cur_cell, cur_ts in zip(cells[1:], stamps[1:]):
        if cur_cell != old_cell:
            if cur_ts - old_stamp < GAP:
                t[old_cell] += cur_ts - old_stamp
                old_cell, old_stamp = cur_cell, cur_ts
            else:
                t[old_cell] = 0
                old_cell, old_stamp = cur_cell, cur_ts
            if t[old_cell] < STAY_THRESHOLD:
                t[old_cell] = 0
        else:
            if cur_ts - old_stamp < GAP:
                t[old_cell] += cur_ts - old_stamp
                old_cell, old_stamp = cur_cell, cur_ts
            else:
                if t[old_cell] < STAY_THRESHOLD:
                    t[old_cell] = 0
                old_cell, old_stamp = cur_cell, cur_ts
    candidates = {c for c in t if t[c] >= STAY_THRESHOLD}
    nb = len(candidates)
    cell = ""
    for c in cells:
        if c in candidates:
            cell = c
            break
    return cell, nb


def compute_original():
    """Return {(user_id, day): (nm_cell, nm_nb, sm_cell, sm_nb, g2_cell, g2_nb)}."""
    orig = {}
    for f in sorted(ORIGINAL_INPUT_DIR.glob("*.csv")):
        day = get_day(f)
        with open(f, encoding="utf-8", newline="") as fh:
            for line in csv.reader(fh, delimiter=";"):
                uid = line[0]
                cells = [c for c in line[8::2] if c]
                stamps = [int(ts) for ts in line[9::2] if ts]
                if len(cells) == 1:
                    orig[(uid, day)] = ("", 0, "", 0, "", 0)
                    continue
                wcells, wstamps = _activity_window(cells, stamps)
                nm_cell, nm_nb = original_activity(wcells, wstamps)
                sm_cell, sm_nb = original_activity([get_cell_code(c) for c in wcells], wstamps)
                g2_cell, g2_nb = original_activity([get_cell_code2(c) for c in wcells], wstamps)
                orig[(uid, day)] = (nm_cell, nm_nb, sm_cell, sm_nb, g2_cell, g2_nb)
        print(f"  original {day} done ({len(orig)} user-days cumulated)")
    return orig


# ---------------------------------------------------- Relaxed reader ---------
def read_relaxed(path, want_merged=True):
    """Stream a Stage3A classified CSV, keeping only the reference-period rows.

    Returns {(user_id, day): (cell_activity, same_home, cell_home, merged_cell_activity)}.
    Column order: 0 user_id;1 day;3 Cell_Home;4 Cell_Activity;5 Same_Home_Activity;
    7 merged_Cell_Activity;10 period.
    """
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        header = fh.readline()  # skip
        for row in fh:
            if TARGET_PERIOD not in row:          # cheap pre-filter
                continue
            f = row.rstrip("\n").split(";")
            if f[10] != TARGET_PERIOD:
                continue
            out[(f[0], f[1])] = (f[4], f[5], f[3], f[7] if want_merged else "")
    return out


# ------------------------------------------------------------- pipeline ------
def main():
    print("Computing Original (continuous) activity cells ...")
    orig = compute_original()

    print("Reading Relaxed simple CSV ...")
    rlx_simple = read_relaxed(RELAXED_SIMPLE_CSV)
    print(f"  {len(rlx_simple)} user-days")
    print("Reading Relaxed 2g3g CSV ...")
    rlx_2g3g = read_relaxed(RELAXED_2G3G_CSV)
    print(f"  {len(rlx_2g3g)} user-days")

    # The relaxed CSV holds the full population (one row per user/day). Join on it.
    keys = rlx_simple.keys()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow([
            "day", "user_id",
            "orig_nm", "orig_nm_nb", "orig_sm", "orig_sm_nb", "orig_2g", "orig_2g_nb",
            "rlx_nm", "rlx_sm", "rlx_2g", "rlx_same_home", "rlx_home",
        ])
        for (uid, day) in keys:
            cell_act, same_home, cell_home, merged_simple = rlx_simple[(uid, day)]
            merged_2g3g = rlx_2g3g.get((uid, day), ("", "", "", ""))[3]
            nm_c, nm_nb, sm_c, sm_nb, g2_c, g2_nb = orig.get((uid, day), ("", 0, "", 0, "", 0))

            # Relaxed effective cell per merge mode (raw if present, else merged fallback)
            rlx_nm = cell_act
            rlx_sm = get_cell_code(cell_act) if cell_act else merged_simple
            rlx_2g = get_cell_code2(cell_act) if cell_act else merged_2g3g

            w.writerow([day, uid, nm_c, nm_nb, sm_c, sm_nb, g2_c, g2_nb,
                        rlx_nm, rlx_sm, rlx_2g, same_home, cell_home])

    print(f"\nWrote {OUT_CSV}")
    _verify_table(orig, rlx_simple, rlx_2g3g)


def _verify_table(orig, rlx_simple, rlx_2g3g):
    """Print the reproduced 6-column table (gate vs the presentation image)."""
    print("\n=== Reproduced comparison table (must match the screenshot) ===")
    hdr = f"{'day':12}{'O-nm':>8}{'O-sm':>8}{'O-2g':>8} | {'R-nm':>8}{'R-sm':>8}{'R-2g':>8}"
    print(hdr)
    for day in DAYS:
        o_nm = o_sm = o_2g = 0
        for (uid, d), (nmc, nmn, smc, smn, g2c, g2n) in orig.items():
            if d != day:
                continue
            o_nm += (nmn == 1)
            o_sm += (smn == 1)
            o_2g += (g2n == 1)
        r_nm = r_sm = r_2g = 0
        for (uid, d), (cell, same, chome, msimple) in rlx_simple.items():
            if d != day:
                continue
            m2 = rlx_2g3g.get((uid, d), ("", "", "", ""))[3]
            r_nm += bool(cell)
            r_sm += bool(cell) or bool(msimple)
            r_2g += bool(cell) or bool(m2)
        print(f"{day:12}{o_nm:>8}{o_sm:>8}{o_2g:>8} | {r_nm:>8}{r_sm:>8}{r_2g:>8}")


if __name__ == "__main__":
    main()
