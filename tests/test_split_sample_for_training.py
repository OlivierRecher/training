import csv
from pathlib import Path

from split_sample_for_training import parse_line, build_header, split_sample
import random


def test_parse_line_keeps_userid_nrecord_and_all_cell_ts_pairs():
    fields = ["1234", "22", "M", "36452", "CETR", "", "", "3",
              "BKVDOU3", "23822", "BKVVIV2", "24105", "BKVSTN4", "24154"]
    assert parse_line(fields) == [
        "1234", "3", "BKVDOU3", "23822", "BKVVIV2", "24105", "BKVSTN4", "24154",
    ]


def test_parse_line_strips_trailing_empty_fields():
    fields = ["1234", "22", "M", "", "CETR", "", "", "1", "BKVDOU3", "23822", ""]
    assert parse_line(fields) == ["1234", "1", "BKVDOU3", "23822"]


def test_build_header_alternates_cellid_ts():
    assert build_header(3) == ["userId", "nRecord", "cellId", "ts", "cellId", "ts", "cellId", "ts"]


def _write_fake_source(path, n_users=12):
    rng = random.Random(0)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        for uid in range(n_users):
            n_events = rng.randint(2, 6)
            row = [str(1000 + uid), str(20 + uid), "M", "1", "CETR", "", "", str(n_events)]
            for _ in range(n_events):
                row += [f"CELL{rng.randint(0, 3)}", str(rng.randint(0, 86399))]
            writer.writerow(row)


def test_split_sample_writes_train_then_test_with_ts_preserved(tmp_path):
    src = tmp_path / "fake_source.csv"
    _write_fake_source(src, n_users=12)
    out_dir = tmp_path / "out"
    rng = random.Random(42)

    dest, train_rows, test_rows = split_sample(src, out_dir, n_train=8, n_test=4, rng=rng, seed=42)

    assert dest.exists()
    assert len(train_rows) == 8
    assert len(test_rows) == 4

    with dest.open(newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        assert header[:4] == ["userId", "nRecord", "cellId", "ts"]
        rows = list(reader)
    assert len(rows) == 12
    # every kept row must still have a numeric timestamp right after its first cellId
    first_data_row = rows[0]
    assert first_data_row[3].isdigit()
