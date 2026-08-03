"""Behavioural user groups for cross-training.

Rationale
---------
`cell_id` sequences look very different from one user to the next, but the
differences are not arbitrary: they are largely captured by *how often the user
changes physical site, as a function of the hour of day*. Two users who both
"stay put in the morning and move a lot late in the afternoon" are far more
informative about each other than two users picked at random -- which is
exactly what cross-training needs.

A group is therefore defined by a **site-change-rate profile over hour bands**:

    profile[b] = P(site root changes | the event falls in band b)

Measuring *site root* changes rather than *cell_id* changes matters. Per
CLAUDE.md a `cell_id` is `<tech letter><site root><zone digits>`, so the same
physical antenna appears under several `cell_id` values. On this dataset 12.4%
of all consecutive-event transitions are "same site root, different
radio/sector" -- i.e. the phone re-attached to another cell of the *same*
antenna without the user going anywhere. Counting those as movement would blur
the very distinction the groups are meant to capture, so `site_root()` strips
the technology letter and the zone digits before comparing.

Sparse bands are handled by shrinking each user's per-band rate toward the
global (train-set) rate for that band, with `shrinkage` pseudo-transitions of
prior weight -- a user with 3 night events does not get a wildly confident
night rate.

What the analysis found on 400_users_train.jsonl (4 bands, k=4)
---------------------------------------------------------------
Groups are numbered by increasing overall mobility, so the ordering is stable
and meaningful (G0 = most sedentary, G3 = most mobile in the morning):

    G0 "sédentaire"        28% of users  site-change 28/32/34/38% (sleep/morn/day/eve)
                           persistence 56% -- "repeat the last cell_id" is already strong
    G1 "régulier"          35%           31/47/46/45% -- flat, moderate mobility
                           persistence 42%
    G2 "actif après-midi"  14%           30/47/67/60% -- calm morning, very mobile
                           persistence 20%    from 16h to 20h (peak change rate 84%)
    G3 "actif le matin"    23%           35/67/65/43% -- very mobile 07h-13h, then
                           persistence 18%    settles down in the evening

G2 and G3 are near mirror images in time, and that shape is not an artefact of
sequence length: the (morning - evening) rate asymmetry is +0.244 for G3 vs
-0.135 for G2 while correlating only -0.28 with log(sequence length).

Why this changes the loss weighting
-----------------------------------
The single global transition window `(4, 6)` turns out to be *easier* than
average for all four groups (persistence 49-64% inside it vs 20-57% overall) --
so upweighting it was pushing gradient at positions the baseline already gets
right. And `(18, 20)` is only genuinely hard for G2; for G3 it is where the
user has just settled down for the evening (36% persistence vs 20% overall,
i.e. easier). `derive_group_windows()` replaces that one global guess with
per-group windows read off the data: the contiguous hours where a group's
`cell_id` persistence falls furthest below its own average, which is precisely
where the model has to reason instead of repeating.

This is the "keyed by group rather than a single global default" iteration that
CLAUDE.md anticipated.
"""
import re
from collections import Counter

import numpy as np

# ---------------------------------------------------------------- cell parsing
# <tech letter><site root><zone digits>; zone is 3 digits for 3G (<1|2><01-09>)
# and a single digit for 2G. See CLAUDE.md.
CELL_ID_RE = re.compile(r"^([BDUV])([A-Z]+?)((?:[12]\d{2})|\d)$")


def parse_cell_id(cell_id):
    """(tech_letter, site_root, zone_digits) or None if `cell_id` doesn't match
    the documented `<tech><root><zone>` shape."""
    m = CELL_ID_RE.match(cell_id)
    return m.groups() if m else None


def site_root(cell_id):
    """The physical-site part of a `cell_id` -- what actually has to change for
    the user to have *moved*. Falls back to the raw string if unparsable, so an
    unexpected id degrades to "its own site" instead of raising."""
    parsed = parse_cell_id(cell_id)
    return parsed[1] if parsed else cell_id


# ------------------------------------------------------------------ hour bands
# (name, start_hour, end_hour) with half-open bounds [start, end). A band whose
# start > end wraps midnight ("sleep" = 22h..07h). Four wide bands beat 5/6/8
# narrower ones on this dataset: every band keeps enough events to be estimated
# reliably, which raised the variance of per-user predictability explained by
# the group label from 0.65 (6 bands) to 0.69, and prefix-detectability from
# 85% to 86%.
DEFAULT_BANDS = [("sleep", 22, 7), ("morning", 7, 12), ("day", 12, 18), ("evening", 18, 22)]

DEFAULT_N_GROUPS = 4
DEFAULT_SHRINKAGE = 8.0


def band_of_hour(hour, bands=DEFAULT_BANDS):
    """Index of the band containing `hour`, honouring midnight-wrapping bands."""
    for i, (_name, start, end) in enumerate(bands):
        if start <= end:
            if start <= hour < end:
                return i
        elif hour >= start or hour < end:      # wraps midnight
            return i
    return 0


def band_transition_counts(cells, hours, bands=DEFAULT_BANDS):
    """(moves, transitions) per band for one user, counted on site-root changes.

    A "transition" is a consecutive pair of events; it is attributed to the band
    of the *later* event (the one being predicted). Returns two float arrays of
    length len(bands)."""
    n_bands = len(bands)
    moves = np.zeros(n_bands)
    totals = np.zeros(n_bands)
    if hours is None:
        return moves, totals
    roots = [site_root(c) for c in cells]
    for i in range(1, len(roots)):
        b = band_of_hour(hours[i], bands)
        totals[b] += 1
        moves[b] += roots[i] != roots[i - 1]
    return moves, totals


def fit_band_prior(users, bands=DEFAULT_BANDS):
    """Global per-band site-change rate over `users` -- the shrinkage target.
    Fit on the TRAIN split only, so test users never influence the geometry."""
    moves = np.zeros(len(bands))
    totals = np.zeros(len(bands))
    for _uid, cells, hours in users:
        m, t = band_transition_counts(cells, hours, bands)
        moves += m
        totals += t
    return moves / np.maximum(totals, 1.0)


def band_profile(cells, hours, prior, bands=DEFAULT_BANDS, shrinkage=DEFAULT_SHRINKAGE):
    """One user's per-band site-change rate, shrunk toward `prior`.

    `shrinkage` acts as that many pseudo-transitions already observed at the
    prior rate, so a band with few real transitions stays near the population
    rate instead of jumping to 0% or 100%.

    A band with no observed transitions *and* `shrinkage == 0` would otherwise
    be 0/0; it falls back to the prior rate for that band. Without this the
    profile would carry NaNs, every centroid distance would be NaN, and
    `GroupModel.assign` would silently put every user in group 0."""
    moves, totals = band_transition_counts(cells, hours, bands)
    numer = moves + shrinkage * prior
    denom = totals + shrinkage
    empty = denom == 0
    if empty.any():
        numer = np.where(empty, prior, numer)
        denom = np.where(empty, 1.0, denom)
    return numer / denom


# ---------------------------------------------------------------- group tokens
def group_token(group):
    """Dedicated tokenizer token for a behavioural group, e.g. '<G0>'."""
    return f"<G{group}>"


def group_vocab(n_groups=DEFAULT_N_GROUPS):
    return [group_token(g) for g in range(n_groups)]


# ------------------------------------------------------------------ the model
class GroupModel:
    """K-means over shrunk per-band site-change profiles, fitted on train users.

    Centroids are re-ordered by increasing mean site-change rate, so group 0 is
    always the most sedentary and group n-1 the most mobile whatever k-means'
    internal labelling was. Assignment is nearest centroid in Euclidean
    distance on the raw (un-standardised) profile: the bands are already
    commensurable rates in [0, 1], and standardising them made things worse --
    it inflates the sparse night band into the dominant axis (R^2 0.69 -> 0.43
    on this dataset)."""

    def __init__(self, centers, prior, bands, shrinkage, names=None):
        self.centers = np.asarray(centers, dtype=float)
        self.prior = np.asarray(prior, dtype=float)
        self.bands = list(bands)
        self.shrinkage = float(shrinkage)
        self.names = list(names) if names else [f"G{g}" for g in range(len(self.centers))]

    @property
    def n_groups(self):
        return len(self.centers)

    def profile(self, cells, hours):
        return band_profile(cells, hours, self.prior, self.bands, self.shrinkage)

    def assign(self, cells, hours):
        """Group of one user, from whatever slice of their day is passed in.

        Passing a *prefix* is what makes group detection legitimate at test
        time: the label comes only from events the model has already been
        given, never from the target it is being asked to predict."""
        if hours is None:
            return 0
        d = ((self.centers - self.profile(cells, hours)) ** 2).sum(axis=1)
        return int(np.argmin(d))

    def assign_all(self, users):
        return [self.assign(cells, hours) for _uid, cells, hours in users]


def fit_groups(users, n_groups=DEFAULT_N_GROUPS, bands=DEFAULT_BANDS,
               shrinkage=DEFAULT_SHRINKAGE, seed=42):
    """Fit a GroupModel on `users` (train split only). Requires scikit-learn."""
    from sklearn.cluster import KMeans

    prior = fit_band_prior(users, bands)
    X = np.array([band_profile(cells, hours, prior, bands, shrinkage)
                  for _uid, cells, hours in users])
    km = KMeans(n_clusters=n_groups, n_init=50, random_state=seed).fit(X)
    order = np.argsort(km.cluster_centers_.mean(axis=1))   # sedentary -> mobile
    return GroupModel(km.cluster_centers_[order], prior, bands, shrinkage)


# -------------------------------------------------- per-group loss weighting
def hour_persistence(users, labels, group, n_hours=24):
    """(same_cell_id, transitions) per hour for one group -- how often "repeat
    the previous cell_id" is correct at each hour. Uses raw `cell_id` equality,
    not site root: that is exactly the quantity the model's top-1 competes
    against."""
    same = np.zeros(n_hours)
    totals = np.zeros(n_hours)
    for (_uid, cells, hours), lab in zip(users, labels):
        if hours is None or lab != group:
            continue
        for i in range(1, len(cells)):
            h = hours[i] % n_hours
            totals[h] += 1
            same[h] += cells[i] == cells[i - 1]
    return same, totals


def derive_group_windows(users, labels, n_groups, min_support=25, min_window_events=60,
                        min_deficit=0.05, weight_gain=2.0, max_weight=4.0):
    """Per-group transition windows read off the training data.

    For each group, find the maximal runs of consecutive hours where that
    group's `cell_id` persistence sits at least `min_deficit` *below* the
    group's own average persistence -- the hours where the "repeat the last
    cell_id" baseline breaks down and the model actually has to reason.

    Hours with fewer than `min_support` observed transitions are ignored (too
    noisy to trust), and a candidate run needs `min_window_events` transitions
    in total to be kept, so a weight is never derived from a handful of events.

    The weight scales with how much harder the window is than the group's
    average::

        ratio  = group_mean_persistence / window_persistence
        weight = clip(1 + weight_gain * (ratio - 1), 1.0, max_weight)

    Returns {group: [(start_hour, end_hour, weight), ...]}. A group whose
    persistence is flat across the day legitimately gets an **empty** list: for
    G0 on this dataset "repeat the last cell_id" is uniformly strong (56%), so
    there is no hour worth upweighting and a flat weight of 1.0 is correct.
    """
    windows = {}
    for g in range(n_groups):
        same, totals = hour_persistence(users, labels, g)
        ok = totals >= min_support
        if not ok.any():
            windows[g] = []
            continue
        base = same[ok].sum() / totals[ok].sum()
        persistence = np.where(ok, same / np.maximum(totals, 1.0), np.inf)
        found = []
        h = 0
        while h < len(totals):
            if base - persistence[h] > min_deficit:
                start = h
                while h < len(totals) and base - persistence[h] > min_deficit:
                    h += 1
                n_events = totals[start:h].sum()
                if n_events >= min_window_events:
                    acc = same[start:h].sum() / n_events
                    ratio = base / max(acc, 1e-6)
                    w = float(np.clip(1.0 + weight_gain * (ratio - 1.0), 1.0, max_weight))
                    found.append((int(start), int(h), round(w, 2)))
            else:
                h += 1
        windows[g] = found
    return windows


def windows_for(group, group_windows, fallback=()):  # noqa: D401
    """Windows of `group`, or `fallback` when the group has none configured."""
    return group_windows.get(group, list(fallback))


# --------------------------------------------------------------- description
def describe_groups(users, labels, model, n_groups):
    """Per-group summary rows for printing: size, band profile, persistence."""
    rows = []
    counts = Counter(labels)
    for g in range(n_groups):
        idx = [i for i, lab in enumerate(labels) if lab == g]
        if not idx:
            rows.append({"group": g, "n": 0})
            continue
        profiles = np.array([model.profile(users[i][1], users[i][2]) for i in idx])
        pers, lens = [], []
        for i in idx:
            cells = users[i][1]
            lens.append(len(cells))
            if len(cells) > 1:
                pers.append(np.mean([cells[j] == cells[j - 1] for j in range(1, len(cells))]))
        rows.append({
            "group": g,
            "n": counts[g],
            "share": counts[g] / len(labels) if labels else 0.0,
            "profile": profiles.mean(axis=0),
            "persistence": float(np.mean(pers)) if pers else float("nan"),
            "mean_len": float(np.mean(lens)),
        })
    return rows
