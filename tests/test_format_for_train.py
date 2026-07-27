import json
from pathlib import Path

from format_for_train import parse_records, row_to_conversation, convert


def test_parse_records_splits_cells_and_computes_hours():
    records = ["BKVDOU3", "23822", "BKVVIV2", "24105"]
    cells, hours = parse_records(records)
    assert cells == ["BKVDOU3", "BKVVIV2"]
    assert hours == [23822 // 3600, 24105 // 3600]


def test_parse_records_drops_unpaired_trailing_field():
    records = ["BKVDOU3", "23822", "BKVVIV2"]   # stray trailing cellId with no ts
    cells, hours = parse_records(records)
    assert cells == ["BKVDOU3"]
    assert hours == [23822 // 3600]


def test_row_to_conversation_has_readable_text_and_structured_fields():
    conv = row_to_conversation("43", "2", ["A", "B"], [3, 4])
    assert conv["user_id"] == "43"
    assert conv["cells"] == ["A", "B"]
    assert conv["hours"] == [3, 4]
    text = conv["conversations"][-1]["content"]
    assert text == "User 43 | 2 events | A B"


def test_convert_splits_exact_train_test_counts(tmp_path):
    src = tmp_path / "cleaned.csv"
    lines = ["userId;nRecord;cellId;ts;cellId;ts"]
    for uid in range(6):
        lines.append(f"{uid};2;CELL{uid};3600;CELL{uid};7200")
    src.write_text("\n".join(lines) + "\n")

    train_dest, n_train, test_dest, n_test = convert(src, tmp_path / "out", n_train=4, n_test=2)

    assert n_train == 4
    assert n_test == 2
    train_objs = [json.loads(l) for l in train_dest.read_text().splitlines()]
    test_objs = [json.loads(l) for l in test_dest.read_text().splitlines()]
    assert len(train_objs) == 4
    assert len(test_objs) == 2
    assert train_objs[0]["hours"] == [1, 2]   # 3600s -> hour 1, 7200s -> hour 2
