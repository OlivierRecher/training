import numpy as np
import pytest

from user_groups import (
    parse_cell_id, site_root, band_of_hour, band_transition_counts, fit_band_prior,
    band_profile, group_token, group_vocab, GroupModel, fit_groups,
    hour_persistence, derive_group_windows, describe_groups,
    DEFAULT_BANDS, DEFAULT_SHRINKAGE,
)


# ------------------------------------------------------------------ cell parsing
def test_parse_cell_id_2g_and_3g():
    assert parse_cell_id("BKVCHE4") == ("B", "KVCHE", "4")
    assert parse_cell_id("UKVCHE102") == ("U", "KVCHE", "102")
    assert parse_cell_id("VSOYCH101") == ("V", "SOYCH", "101")
    assert parse_cell_id("DSOCHO3") == ("D", "SOCHO", "3")


def test_parse_cell_id_edge_sectors_from_claude_md():
    # CLAUDE.md: 2G sectors can be 0-9, 3G sectors run to 09 -- not just 1-4 / 101-104
    assert parse_cell_id("BKVHRO0") == ("B", "KVHRO", "0")
    assert parse_cell_id("BKVPUP9") == ("B", "KVPUP", "9")
    assert parse_cell_id("UKVPUP109") == ("U", "KVPUP", "109")
    assert parse_cell_id("UKVPUP209") == ("U", "KVPUP", "209")


def test_parse_cell_id_rejects_malformed():
    assert parse_cell_id("") is None
    assert parse_cell_id("XKVCHE1") is None      # unknown technology letter
    assert parse_cell_id("BKVCHE") is None       # no zone digits
    assert parse_cell_id("1KVCHE1") is None


def test_site_root_collapses_technology_and_sector():
    # The whole point: these four cell_id are the SAME physical site
    assert site_root("BKVCHE1") == "KVCHE"
    assert site_root("BKVCHE4") == "KVCHE"
    assert site_root("UKVCHE102") == "KVCHE"
    assert site_root("UKVCHE201") == "KVCHE"


def test_site_root_falls_back_to_raw_string():
    # An unparsable id degrades to "its own site" rather than raising
    assert site_root("NOT_A_CELL") == "NOT_A_CELL"


# ------------------------------------------------------------------- hour bands
def test_band_of_hour_wraps_midnight():
    # DEFAULT_BANDS: sleep 22->7 wraps, so both 23h and 3h are the sleep band
    sleep = band_of_hour(23)
    assert band_of_hour(3) == sleep
    assert band_of_hour(0) == sleep
    assert band_of_hour(6) == sleep
    assert band_of_hour(7) != sleep          # start inclusive on the next band
    assert band_of_hour(21) != sleep
    assert band_of_hour(22) == sleep


def test_band_of_hour_bounds_are_half_open():
    bands = [("a", 0, 12), ("b", 12, 24)]
    assert band_of_hour(11, bands) == 0
    assert band_of_hour(12, bands) == 1      # end excluded
    assert band_of_hour(23, bands) == 1


def test_band_transition_counts_uses_site_root_not_cell_id():
    # Same site (KVCHE) seen through 3 different cells -> 0 moves, 2 transitions
    cells = ["BKVCHE1", "UKVCHE102", "BKVCHE4"]
    hours = [8, 9, 10]
    moves, totals = band_transition_counts(cells, hours)
    assert moves.sum() == 0
    assert totals.sum() == 2


def test_band_transition_counts_counts_real_site_changes():
    cells = ["BKVCHE1", "BKVARE1", "BKVARE4"]
    moves, totals = band_transition_counts(cells, [8, 9, 10])
    assert moves.sum() == 1        # CHE -> ARE only
    assert totals.sum() == 2


def test_band_transition_counts_attributes_to_the_predicted_event():
    # transition is booked to the band of the LATER event (the one being predicted)
    bands = [("early", 0, 12), ("late", 12, 24)]
    moves, totals = band_transition_counts(["BKVCHE1", "BKVARE1"], [11, 13], bands)
    assert totals[1] == 1 and totals[0] == 0
    assert moves[1] == 1


def test_band_transition_counts_without_hours_is_empty():
    moves, totals = band_transition_counts(["BKVCHE1", "BKVARE1"], None)
    assert moves.sum() == 0 and totals.sum() == 0


# -------------------------------------------------------------------- profiles
def _users():
    # two stay-put users and two movers, all active across the day
    stay = ["BKVCHE1"] * 12
    move = [f"BKV{r}1" for r in ("AAA", "BBB", "CCC", "DDD")] * 3
    hours = [1, 3, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21]
    return [("s1", stay, hours), ("s2", stay, hours),
            ("m1", move, hours), ("m2", move, hours)]


def test_fit_band_prior_is_between_zero_and_one():
    prior = fit_band_prior(_users())
    assert prior.shape == (len(DEFAULT_BANDS),)
    assert np.all((prior >= 0) & (prior <= 1))


def test_band_profile_shrinks_toward_prior_when_data_is_thin():
    users = _users()
    prior = fit_band_prior(users)
    # A user with a single night transition should stay near the prior there
    thin = band_profile(["BKVCHE1", "BKVARE1"], [1, 2], prior, shrinkage=100.0)
    night = band_of_hour(2)
    assert abs(thin[night] - prior[night]) < 0.05


def test_band_profile_follows_the_user_with_no_shrinkage():
    users = _users()
    prior = fit_band_prior(users)
    # shrinkage=0 -> the raw rate; this user moves at every night transition
    raw = band_profile(["BKVCHE1", "BKVARE1", "BKVBER1"], [1, 2, 3], prior, shrinkage=0.0)
    assert raw[band_of_hour(2)] == pytest.approx(1.0)


def test_band_profile_separates_movers_from_stayers():
    users = _users()
    prior = fit_band_prior(users)
    stay = band_profile(users[0][1], users[0][2], prior)
    move = band_profile(users[2][1], users[2][2], prior)
    assert move.mean() > stay.mean()


# ----------------------------------------------------------------- group tokens
def test_group_token_and_vocab():
    assert group_token(0) == "<G0>"
    assert group_token(3) == "<G3>"
    assert group_vocab(4) == ["<G0>", "<G1>", "<G2>", "<G3>"]
    assert len(group_vocab(7)) == 7


# ------------------------------------------------------------------ GroupModel
def test_group_model_assign_picks_nearest_centroid():
    n_bands = len(DEFAULT_BANDS)
    centers = [np.zeros(n_bands), np.ones(n_bands)]
    prior = np.full(n_bands, 0.5)
    gm = GroupModel(centers, prior, DEFAULT_BANDS, shrinkage=0.0)
    assert gm.n_groups == 2
    # never moves -> profile all zeros -> group 0
    assert gm.assign(["BKVCHE1"] * 6, [8, 9, 10, 13, 14, 15]) == 0
    # moves every step -> profile near 1 -> group 1
    movers = ["BKVAAA1", "BKVBBB1", "BKVCCC1", "BKVDDD1", "BKVEEE1", "BKVFFF1"]
    assert gm.assign(movers, [8, 9, 10, 13, 14, 15]) == 1


def test_group_model_without_hours_returns_group_zero():
    gm = GroupModel([np.zeros(4), np.ones(4)], np.full(4, 0.5), DEFAULT_BANDS, 0.0)
    assert gm.assign(["BKVCHE1", "BKVARE1"], None) == 0


def test_fit_groups_orders_centroids_by_increasing_mobility():
    users = _users() * 6          # k-means needs more than k points
    gm = fit_groups(users, n_groups=2, seed=0)
    means = gm.centers.mean(axis=1)
    assert means[0] <= means[1], "group 0 must be the most sedentary"
    # the stay-put users land in the low group, the movers in the high one
    assert gm.assign(users[0][1], users[0][2]) == 0
    assert gm.assign(users[2][1], users[2][2]) == 1


def test_fit_groups_is_deterministic_for_a_given_seed():
    users = _users() * 6
    a = fit_groups(users, n_groups=2, seed=7)
    b = fit_groups(users, n_groups=2, seed=7)
    assert np.allclose(a.centers, b.centers)


def test_assign_all_covers_every_user():
    users = _users() * 6
    gm = fit_groups(users, n_groups=2, seed=0)
    labels = gm.assign_all(users)
    assert len(labels) == len(users)
    assert set(labels) <= {0, 1}


# ------------------------------------------------------ per-group loss windows
def test_hour_persistence_counts_only_the_requested_group():
    users = [("a", ["BKVCHE1", "BKVCHE1"], [5, 5]),
             ("b", ["BKVCHE1", "BKVARE1"], [5, 5])]
    same, totals = hour_persistence(users, [0, 1], group=0)
    assert totals[5] == 1 and same[5] == 1        # only user "a"
    same, totals = hour_persistence(users, [0, 1], group=1)
    assert totals[5] == 1 and same[5] == 0        # only user "b", which moved


def test_derive_group_windows_finds_the_hard_hours():
    # One group: persistent everywhere except 18h-19h, where it always changes.
    # 40 transitions per easy hour, comfortably above the min_support=25 floor --
    # below it the hour is discarded as too noisy and never forms a baseline.
    easy_hours = [h for h in (8, 9, 10, 11, 12, 13, 14, 15) for _ in range(40)]
    users = []
    for i, h in enumerate(easy_hours):
        users.append((f"e{i}", ["BKVCHE1", "BKVCHE1"], [h, h]))
    for i in range(80):
        users.append((f"h{i}", ["BKVCHE1", "BKVARE1"], [18, 18]))
    labels = [0] * len(users)
    wins = derive_group_windows(users, labels, n_groups=1)
    assert 0 in wins and wins[0], "the 18h deficit should produce a window"
    starts = [s for s, _e, _w in wins[0]]
    assert 18 in starts
    for _s, _e, w in wins[0]:
        assert 1.0 <= w <= 4.0


def test_derive_group_windows_returns_empty_for_a_flat_group():
    # Uniform persistence across the day -> no hour is worth upweighting
    users = []
    for i in range(400):
        h = 8 + (i % 8)
        cells = ["BKVCHE1", "BKVCHE1"] if i % 2 else ["BKVCHE1", "BKVARE1"]
        users.append((f"u{i}", cells, [h, h]))
    wins = derive_group_windows(users, [0] * len(users), n_groups=1)
    assert wins[0] == []


def test_derive_group_windows_ignores_low_support_hours():
    # A single very hard hour with almost no data must not create a window
    users = [(f"e{i}", ["BKVCHE1", "BKVCHE1"], [10, 10]) for i in range(200)]
    users += [(f"h{i}", ["BKVCHE1", "BKVARE1"], [3, 3]) for i in range(3)]
    wins = derive_group_windows(users, [0] * len(users), n_groups=1)
    assert all(s != 3 for s, _e, _w in wins[0])


def test_derive_group_windows_respects_max_weight():
    users = [(f"e{i}", ["BKVCHE1", "BKVCHE1"], [10, 10]) for i in range(300)]
    users += [(f"h{i}", ["BKVCHE1", "BKVARE1"], [18, 18]) for i in range(100)]
    wins = derive_group_windows(users, [0] * len(users), n_groups=1, max_weight=2.0)
    for _s, _e, w in wins[0]:
        assert w <= 2.0


def test_derive_group_windows_covers_every_group():
    users = _users() * 6
    labels = [i % 3 for i in range(len(users))]
    wins = derive_group_windows(users, labels, n_groups=3)
    assert set(wins) == {0, 1, 2}
    assert all(isinstance(v, list) for v in wins.values())


# ------------------------------------------------------------------ description
def test_describe_groups_reports_size_share_and_persistence():
    users = _users() * 6
    gm = fit_groups(users, n_groups=2, seed=0)
    labels = gm.assign_all(users)
    rows = describe_groups(users, labels, gm, 2)
    assert len(rows) == 2
    assert sum(r["n"] for r in rows) == len(users)
    for r in rows:
        assert 0.0 <= r["share"] <= 1.0
        assert 0.0 <= r["persistence"] <= 1.0
        assert len(r["profile"]) == len(DEFAULT_BANDS)


def test_describe_groups_handles_an_empty_group():
    users = _users() * 6
    gm = fit_groups(users, n_groups=2, seed=0)
    labels = gm.assign_all(users)
    rows = describe_groups(users, labels, gm, 3)   # group 2 has no members
    assert rows[2]["n"] == 0
