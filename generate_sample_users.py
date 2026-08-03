#!/usr/bin/env python3
"""Génère des fichiers <N>_users_{train,test}.jsonl synthétiques, au même format
que le jeu réel produit par `make data-train`.

Sert de **repli autonome** : le notebook peut tourner de bout en bout sans aucun
fichier de données externe. Les séquences portent donc, comme les données réelles :

  * des `cell_id` au format documenté dans CLAUDE.md -- `<lettre techno><site
    root><chiffres de zone>` -- de sorte que plusieurs `cell_id` partagent le même
    site physique (c'est ce que `user_groups.site_root()` compare) ;
  * une **heure** par événement (champ `hours`), sans laquelle le cross-training
    par groupe de comportement se désactive faute de signal horaire ;
  * quatre **archétypes de mobilité** calqués sur les groupes observés dans les
    vraies données, pour que le regroupement ait quelque chose à retrouver :
    sédentaire / régulier / actif l'après-midi / actif le matin.

Usage :
    python generate_sample_users.py --n-users 500 [--seed 42] [--out-dir DIR]
    python generate_sample_users.py --n-train 400 --n-test 100 --out-dir data/dataset_for_training
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = ("You are an AI that generates a user's cell-visit trajectory. "
                 "Output format: User <USER_ID> | <N_RECORDS> events | "
                 "<CELL_ID> <CELL_ID> ...")
USER_PROMPT = "Here is the sequence for this user."

# Archétypes : (nom, probabilité de changer de site par tranche, événements/heure).
# Les tranches sont (nuit 22-07, matin 07-12, jour 12-18, soir 18-22), dans l'ordre
# de CONFIG["group_bands"]. Les valeurs reprennent les profils mesurés sur les 400
# vrais utilisateurs (cf. python/user_groups.py) -- y compris le fait que les
# sédentaires émettent nettement plus d'événements par heure.
ARCHETYPES = [
    ("sedentaire",     (0.28, 0.32, 0.34, 0.38), 10.0, 0.30),
    ("regulier",       (0.31, 0.47, 0.46, 0.45),  9.0, 0.22),
    ("actif_apresmidi",(0.30, 0.47, 0.67, 0.60),  4.6, 0.15),
    ("actif_matin",    (0.35, 0.67, 0.65, 0.43),  4.8, 0.15),
]
BANDS = ((22, 7), (7, 12), (12, 18), (18, 22))


def band_of_hour(hour):
    for i, (start, end) in enumerate(BANDS):
        if start <= end:
            if start <= hour < end:
                return i
        elif hour >= start or hour < end:
            return i
    return 0


def load_real_vocab():
    """Vocabulaire repris des fichiers réels s'ils sont là, sinon []."""
    vocab = set()
    for path in ("100_users_train.jsonl", "100_users_test.jsonl"):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if "cells" in row:
                        vocab.update(row["cells"])
                    else:
                        content = row["conversations"][-1]["content"]
                        vocab.update(content.split("|", 2)[2].split())
        except (FileNotFoundError, KeyError, ValueError):
            pass
    return sorted(vocab)


def synthetic_vocab(n_sites=110):
    """Vocabulaire au format CLAUDE.md : pour chaque site, des cellules 2G
    (`B<root><1-4>`) et 3G (`U<root><1|2><01-04>`). Plusieurs `cell_id` par site
    physique, exactement comme dans les vraies données."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sites = {}
    for i in range(n_sites):
        root = "KV" + letters[i // 26 % 26] + letters[i % 26] + letters[(i * 7) % 26]
        cells = [f"B{root}{s}" for s in range(1, 5)]
        cells += [f"U{root}{z}{s:02d}" for z in (1, 2) for s in range(1, 5)]
        sites[root] = cells
    return sites


def sites_from_vocab(vocab):
    """Regroupe un vocabulaire plat par site physique (préfixe sans techno/zone)."""
    import re
    pat = re.compile(r"^([BDUV])([A-Z]+?)((?:[12]\d{2})|\d)$")
    sites = {}
    for cell in vocab:
        m = pat.match(cell)
        root = m.group(2) if m else cell
        sites.setdefault(root, []).append(cell)
    return sites


def make_sequence(rng, sites, archetype):
    """Une journée : répertoire de sites restreint, mobilité dépendant de l'heure.

    Un « déplacement » change de site ; sinon on reste sur place, en réémettant
    parfois une AUTRE cellule du même site (changement techno/secteur) -- c'est
    ce bruit qui rend `site_root()` indispensable côté analyse."""
    _name, move_probs, per_hour, same_site_switch = archetype
    roots = rng.sample(sorted(sites), k=rng.randint(12, 24))
    home = roots[0]
    start = rng.choices([0, 1, 6, 7, 8, 9, 10], weights=[15, 10, 12, 18, 16, 15, 14])[0]
    end = rng.choices([15, 16, 17, 18, 19, 22, 23], weights=[12, 14, 16, 12, 12, 16, 18])[0]
    if end <= start:
        end = min(23, start + 6)

    cells, hours = [], []
    current = home
    current_cell = rng.choice(sites[home])
    for hour in range(start, end + 1):
        n_events = max(1, int(rng.gauss(per_hour, per_hour * 0.35)))
        p_move = move_probs[band_of_hour(hour)]
        for _ in range(n_events):
            if rng.random() < p_move:
                # déplacement : nouveau site, donc nouvelle cellule
                current = rng.choice([r for r in roots if r != current])
                current_cell = rng.choice(sites[current])
            elif rng.random() < same_site_switch:
                # sur place, mais le téléphone se raccroche à une AUTRE cellule du
                # même site (changement techno/secteur) : le cell_id change sans
                # déplacement -- c'est ce bruit qui rend site_root() indispensable
                others = [c for c in sites[current] if c != current_cell]
                if others:
                    current_cell = rng.choice(others)
            # sinon : on ne bouge pas et on réémet EXACTEMENT la même cellule,
            # ce qui fait de « recopier le cell_id précédent » une baseline forte
            cells.append(current_cell)
            hours.append(hour)
    # borne de longueur, comme dans les vraies données (19 à 200 événements)
    if len(cells) > 200:
        cells, hours = cells[:200], hours[:200]
    while len(cells) < 19:
        cells.append(cells[-1] if cells else rng.choice(sites[home]))
        hours.append(hours[-1] if hours else start)
    return cells, hours


def make_line(user_id, cells, hours, with_hours=True):
    """Même schéma que python/format_for_train.py : champs structurés
    `cells`/`hours` (ce que le notebook lit) + le texte `conversations`."""
    row = {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant",
             "content": f"User {user_id} | {len(cells)} events | {' '.join(cells)}"},
        ],
        "user_id": str(user_id),
        "cells": cells,
    }
    if with_hours:
        row["hours"] = hours
    return json.dumps(row, ensure_ascii=False)


def generate(n_train, n_test, seed, out_dir, with_hours=True, stem=None):
    rng = random.Random(seed)
    vocab = load_real_vocab()
    if vocab:
        sites = sites_from_vocab(vocab)
    else:
        sites = synthetic_vocab()
        print("⚠️  Fichiers 100_users introuvables : vocabulaire synthétique utilisé "
              f"({len(sites)} sites au format CLAUDE.md).")

    n_total = n_train + n_test
    user_ids = rng.sample(range(19_000_000, 21_000_000), n_total)
    lines = []
    for i, uid in enumerate(user_ids):
        arch = ARCHETYPES[i % len(ARCHETYPES)]     # les 4 archétypes équirépartis
        cells, hours = make_sequence(rng, sites, arch)
        lines.append(make_line(uid, cells, hours, with_hours))
    rng.shuffle(lines)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{n_train}_users"
    written = []
    for name, chunk in ((f"{stem}_train.jsonl", lines[:n_train]),
                        (f"{stem}_test.jsonl", lines[n_train:])):
        dest = out_dir / name
        dest.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        print(f"{dest} : {len(chunk)} utilisateurs")
        written.append(dest)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-users", type=int, default=None,
                        help="nombre total d'utilisateurs (split 80/20 train/test)")
    parser.add_argument("--n-train", type=int, default=None, help="nombre exact de lignes train")
    parser.add_argument("--n-test", type=int, default=None, help="nombre exact de lignes test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="dossier de sortie (défaut : dossier courant)")
    parser.add_argument("--stem", default=None,
                        help="préfixe des fichiers (défaut : <n_train>_users)")
    parser.add_argument("--no-hours", action="store_true",
                        help="ne pas émettre le champ hours (désactive le mode heure du notebook)")
    args = parser.parse_args()

    if args.n_train is None or args.n_test is None:
        total = args.n_users if args.n_users is not None else 500
        n_train = int(total * 0.8)
        n_test = total - n_train
    else:
        n_train, n_test = args.n_train, args.n_test

    generate(n_train, n_test, args.seed, args.out_dir,
             with_hours=not args.no_hours, stem=args.stem)


if __name__ == "__main__":
    main()
