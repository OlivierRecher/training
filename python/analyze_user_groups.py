"""Report the behavioural user groups used for cross-training.

Reads the train/test JSONL produced by `format_for_train.py`, fits the groups on
the TRAIN split only (see python/user_groups.py), and prints:

  * the profile of each group (site-change rate per hour band, persistence, size)
  * the per-group transition windows derived from the data
  * how well the group label explains per-user predictability (R^2), on train
    AND on test -- the test figure is the one that matters, since the centroids
    never saw those users
  * whether the group can be recovered from a context prefix alone, which is the
    condition for the notebook to pick the right group at inference time
  * a placement check of the per-group windows against a single global window

Usage:
    python python/analyze_user_groups.py [--train F] [--test F] [--n-groups K]
    make groups
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from user_groups import (DEFAULT_BANDS, DEFAULT_N_GROUPS, DEFAULT_SHRINKAGE,
                         describe_groups, derive_group_windows, fit_groups, site_root)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = ROOT / "data" / "dataset_for_training" / "400_users_train.jsonl"
DEFAULT_TEST = ROOT / "data" / "dataset_for_training" / "400_users_test.jsonl"
GLOBAL_WINDOWS = [(4, 6, 2.0), (18, 20, 4.0)]


def load_users(path):
    users = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "cells" in row:
                users.append((row.get("user_id", "?"), row["cells"], row.get("hours")))
            else:
                content = row["conversations"][-1]["content"]
                head, _, seq = content.split("|", 2)
                users.append((head.split()[1], seq.split(), None))
    return users


def split_index(n_events, context_fraction):
    return max(1, min(n_events - 1, round(n_events * context_fraction)))


def persistence(cells):
    return np.mean([cells[i] == cells[i - 1] for i in range(1, len(cells))]) if len(cells) > 1 else 0.0


def r2_of_labels(values, labels, n_groups):
    """Share of the variance of `values` explained by the group label."""
    values = np.asarray(values, dtype=float)
    total = ((values - values.mean()) ** 2).sum()
    if total == 0:
        return float("nan")
    within = 0.0
    for g in range(n_groups):
        sub = values[np.asarray(labels) == g]
        if len(sub):
            within += ((sub - sub.mean()) ** 2).sum()
    return 1 - within / total


def window_placement(users, labels, group_windows, global_windows):
    """Baseline persistence inside vs outside the transition windows. A window
    that isolates genuinely hard positions must be HARDER (lower persistence)
    than what sits outside it; a window at ~0 gap targets nothing."""
    out = {}
    for name, per_user in (("par groupe", True), ("global unique", False)):
        n_in = ok_in = n_out = ok_out = 0
        for (uid, cells, hours), lab in zip(users, labels):
            if hours is None:
                continue
            wins = group_windows.get(lab, []) if per_user else global_windows
            for i in range(1, len(cells)):
                hit = cells[i] == cells[i - 1]
                if any(s <= hours[i] < e for s, e, _ in wins):
                    n_in += 1; ok_in += hit
                else:
                    n_out += 1; ok_out += hit
        out[name] = {
            "in": ok_in / n_in if n_in else float("nan"),
            "out": ok_out / n_out if n_out else float("nan"),
            "n_in": n_in,
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p.add_argument("--n-groups", type=int, default=DEFAULT_N_GROUPS)
    p.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for f in (args.train, args.test):
        if not f.exists():
            p.error(f"file not found: {f} (run `make data-train` first)")

    users_train = load_users(args.train)
    users_test = load_users(args.test)
    if not any(h is not None for _u, _c, h in users_train):
        p.error(f"{args.train} carries no hours; behavioural groups need them "
                "(rebuild with `make data-train`)")

    k = args.n_groups
    model = fit_groups(users_train, n_groups=k, shrinkage=args.shrinkage, seed=args.seed)
    lab_tr = model.assign_all(users_train)
    lab_te = model.assign_all(users_test)
    windows = derive_group_windows(users_train, lab_tr, k)
    band_names = [b[0] for b in model.bands]

    # --- what fraction of "cell_id changed" is really the same physical site? ---
    same_site = changed = identical = 0
    for _u, cells, _h in users_train + users_test:
        for a, b in zip(cells, cells[1:]):
            if a == b:
                identical += 1
            elif site_root(a) == site_root(b):
                same_site += 1
            else:
                changed += 1
    tot = identical + same_site + changed
    print(f"Transitions entre événements consécutifs ({tot} au total) :")
    print(f"  cell_id identique                    : {identical / tot:6.1%}")
    print(f"  même site physique, autre techno/sect.: {same_site / tot:6.1%}  <- PAS un déplacement")
    print(f"  changement de site réel               : {changed / tot:6.1%}")

    print(f"\n{k} groupes de comportement (centroïdes ajustés sur le train uniquement) :")
    hdr = "  ".join(f"{n[:7]:>7}" for n in band_names)
    print(f"  grp   train        test  | {hdr} | persist. | long. | créneaux (heure→poids)")
    rows_tr = describe_groups(users_train, lab_tr, model, k)
    rows_te = describe_groups(users_test, lab_te, model, k)
    for r, rt in zip(rows_tr, rows_te):
        if not r["n"]:
            print(f"  G{r['group']}    (vide)")
            continue
        prof = "  ".join(f"{v:6.0%} " for v in r["profile"])
        wins = ", ".join(f"{s:02d}-{e:02d}h→×{w:.1f}" for s, e, w in windows.get(r["group"], [])) or "aucun"
        print(f"  G{r['group']}  {r['n']:4d} ({r['share']:.0%})  {rt['n']:4d} ({rt['share']:.0%}) | "
              f"{prof}| {r['persistence']:7.1%}  | {r['mean_len']:5.0f} | {wins}")

    pa_tr = [persistence(c) for _u, c, _h in users_train]
    pa_te = [persistence(c) for _u, c, _h in users_test]
    print("\nVariance de la prédictibilité par utilisateur expliquée par le groupe (R²) :")
    print(f"  train : {r2_of_labels(pa_tr, lab_tr, k):.3f}")
    print(f"  test  : {r2_of_labels(pa_te, lab_te, k):.3f}   <- utilisateurs jamais vus par les centroïdes")

    print("\nDétection du groupe depuis le seul préfixe (accord avec l'étiquette complète) :")
    for frac in (0.3, 0.5, 0.6, 0.8, 0.9):
        line = []
        for name, us, ref in (("train", users_train, lab_tr), ("test", users_test, lab_te)):
            hit = 0
            for (uid, cells, hours), g in zip(us, ref):
                cut = split_index(len(cells), frac)
                hit += model.assign(cells[:cut], hours[:cut] if hours is not None else None) == g
            line.append(f"{name} {hit / len(us):.1%}")
        print(f"  contexte {frac:.0%} : " + "  ".join(line))

    print("\nPlacement des créneaux (baseline persistance, séquence complète du test) :")
    print("  un créneau BIEN placé doit être PLUS DUR (persistance plus basse) que hors créneau")
    place = window_placement(users_test, lab_te, windows, GLOBAL_WINDOWS)
    for name, s in place.items():
        gap = s["in"] - s["out"]
        verdict = ("cible les positions difficiles" if gap < -0.02
                   else "ne cible rien de particulier" if gap < 0.02
                   else "cible des positions PLUS FACILES (mal placé)")
        print(f"  {name:<14} n_dans={s['n_in']:5d} | dans={s['in']:6.1%}  hors={s['out']:6.1%}  "
              f"écart={gap:+6.1%}  -> {verdict}")
    print(f"\n  (créneau global unique de référence : {GLOBAL_WINDOWS})")


if __name__ == "__main__":
    main()
