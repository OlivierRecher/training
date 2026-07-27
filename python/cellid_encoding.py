"""Time-aware event encoding used by the data-prep scripts (split_sample_for_training.py,
format_for_train.py) and covered by tests/test_cellid_encoding.py.

train_cellid_llm.ipynb does NOT import this module: it keeps its own copy of this
logic inline (Colab-only usage, must stay a single self-contained file with no
external module to upload). Keep the two copies in sync by hand if this file changes.

An "event" is one (cell_id, hour) pair from a user's daily trajectory. Hour is
the hour-of-day (0-23) derived from the raw timestamp (seconds since local
midnight, per CLAUDE.md) attached to that cell_id -- the timestamp follows the
cell_id it belongs to.
"""


def event_hour_from_seconds(ts_seconds):
    """Hour of day (0-23) from a timestamp in seconds since local midnight."""
    return (int(ts_seconds) // 3600) % 24


def hour_token(hour):
    """Dedicated tokenizer token for an hour-of-day, e.g. hour_token(6) == '<H06>'."""
    return f"<H{hour:02d}>"


HOUR_VOCAB = [hour_token(h) for h in range(24)]


def in_transition_window(hour, transition_windows):
    """True if `hour` falls in any (start_hour, end_hour, weight) window.
    Bounds are half-open [start_hour, end_hour) -- end excluded."""
    return any(start <= hour < end for start, end, _weight in transition_windows)


def event_weight(hour, transition_windows, base_weight=1.0):
    """Loss weight for an event at `hour`: the configured window's weight if
    `hour` falls inside one of `transition_windows`, else `base_weight`.
    `hour=None` (hour-less legacy data) always returns `base_weight`."""
    if hour is None:
        return base_weight
    for start, end, weight in transition_windows:
        if start <= hour < end:
            return weight
    return base_weight


def split_index(n_events, context_fraction):
    """Cutoff index for a context/target split over `n_events` events: at
    least 1, at most n_events - 1."""
    return max(1, min(n_events - 1, round(n_events * context_fraction)))


def make_chunks(n_events, chunk_len, chunk_stride):
    """Sliding-window (start, end, n_context_events) triples over event
    indices [0, n_events). `n_context_events` is how many of the window's
    leading events are overlap from the previous window (already trained on,
    excluded from the loss the second time). Windows are at most `chunk_len`
    events, advancing by `chunk_stride` each time."""
    if n_events <= chunk_len:
        return [(0, n_events, 0)]
    out, start = [], 0
    while start < n_events:
        end = min(start + chunk_len, n_events)
        ctx = 0 if start == 0 else chunk_len - chunk_stride
        if end - start > ctx:
            out.append((start, end, ctx))
        if start + chunk_len >= n_events:
            break
        start += chunk_stride
    return out


def event_token_positions(n_events, has_hours):
    """Index (relative to the first token *after* the prompt prefix) of each
    event's cell_id token in the flat sequence built by `encode_events`. When
    `has_hours` is True, each event is 2 tokens (hour then cell_id), so
    cell_id tokens sit at 1, 3, 5, ...; when False, each event is 1 token
    (cell_id only), so they sit at 0, 1, 2, ..."""
    step = 2 if has_hours else 1
    offset = 1 if has_hours else 0
    return [offset + i * step for i in range(n_events)]


def encode_events(cells, hours, cell_to_id, hour_to_id, prefix_ids,
                  transition_windows, n_masked_cells=0, base_weight=1.0):
    """Build (input_ids, labels, weights) for one sequence of events.

    cells: list[str] cell_id per event.
    hours: list[int] | None. If None, cell tokens only (legacy/no hour info) --
        no interleaved hour tokens, every unmasked position gets `base_weight`.
    cell_to_id / hour_to_id: dict[str, int] tokenizer vocab lookups.
    prefix_ids: list[int] tokens prepended before any event (e.g. a fixed
        prompt) -- always masked (label -100, weight base_weight).
    n_masked_cells: the first N events' cell labels are masked (-100) --
        used for chunk overlap, so a repeated context window isn't trained on
        twice.
    Returns three lists of equal length (one entry per token): input_ids,
    labels, weights.
    """
    ids = list(prefix_ids)
    labels = [-100] * len(prefix_ids)
    weights = [base_weight] * len(prefix_ids)
    for i, cell in enumerate(cells):
        hour = hours[i] if hours is not None else None
        if hour is not None:
            ids.append(hour_to_id[hour_token(hour)])
            labels.append(-100)
            weights.append(base_weight)
        ids.append(cell_to_id[cell])
        if i < n_masked_cells:
            labels.append(-100)
            weights.append(base_weight)
        else:
            labels.append(cell_to_id[cell])
            weights.append(event_weight(hour, transition_windows, base_weight))
    return ids, labels, weights
