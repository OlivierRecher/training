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
