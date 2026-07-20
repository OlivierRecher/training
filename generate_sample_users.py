#!/usr/bin/env python3
"""Génère des fichiers <N>_users_{train,test}.jsonl synthétiques, au même format
que 100_users_train.jsonl (80% train / 20% test).

Usage : python generate_sample_users.py --n-users 500 [--seed 42]

Les cell_id sont repris des fichiers 100_users réels ; chaque utilisateur
synthétique a un petit répertoire de cellules et un comportement persistant
(forte probabilité de répéter la cellule précédente), comme dans les données réelles.
"""
import argparse
import json
import random

SYSTEM_PROMPT = ("You are an AI that analyzes user event sequences, DataSet start by "
                 "User <USER_ID> | <N_RECORDS> events | <CELL_ID> <CELL_ID> ...")
USER_PROMPT = "Here is the sequence for this user."


def load_real_vocab():
    vocab = set()
    for path in ("100_users_train.jsonl", "100_users_test.jsonl"):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    content = json.loads(line)["conversations"][-1]["content"]
                    vocab.update(content.split("|", 2)[2].split())
        except FileNotFoundError:
            pass
    return sorted(vocab)


def make_sequence(rng, vocab):
    """Séquence réaliste : répertoire restreint + persistance élevée."""
    repertoire = rng.sample(vocab, k=rng.choices([2, 3, 4, 5, 6, 8], weights=[20, 25, 25, 15, 10, 5])[0])
    weights = [rng.uniform(0.5, 3.0) for _ in repertoire]
    # Motif apprenable : chaque cellule a un « successeur préféré » propre à l'utilisateur
    successor = {c: rng.choice([x for x in repertoire if x != c]) for c in repertoire}
    persistence = rng.uniform(0.50, 0.80)
    length = max(10, min(460, int(rng.lognormvariate(3.8, 0.8))))
    seq = [rng.choices(repertoire, weights)[0]]
    for _ in range(length - 1):
        r = rng.random()
        if r < persistence:
            seq.append(seq[-1])
        elif r < persistence + (1 - persistence) * 0.7:
            seq.append(successor[seq[-1]])
        else:
            seq.append(rng.choices(repertoire, weights)[0])
    return seq


def make_line(user_id, seq):
    return json.dumps({"conversations": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
        {"role": "assistant", "content": f"User {user_id} | {len(seq)} events | {' '.join(seq)}"},
    ]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    vocab = load_real_vocab()
    if not vocab:
        vocab = [f"CELL{i:03d}" for i in range(264)]
        print("⚠️  Fichiers 100_users introuvables : vocabulaire synthétique utilisé.")

    user_ids = rng.sample(range(19_000_000, 21_000_000), args.n_users)
    lines = [make_line(uid, make_sequence(rng, vocab)) for uid in user_ids]

    n_train = int(args.n_users * 0.8)
    for name, chunk in ((f"{args.n_users}_users_train.jsonl", lines[:n_train]),
                        (f"{args.n_users}_users_test.jsonl", lines[n_train:])):
        with open(name, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")
        print(f"{name} : {len(chunk)} utilisateurs")


if __name__ == "__main__":
    main()
