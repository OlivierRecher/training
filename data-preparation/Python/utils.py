"""Shared helpers for the Original vs Relaxed activity-cell comparison scripts.

`is_weekend` and `get_day` are copied from LegacyPython/utils.py; `get_cell_code`
and `get_cell_code2` from the important_cells_work classifiers. Kept self-contained
so the new Python/ pipeline does not depend on LegacyPython.
"""
import datetime
import re
from pathlib import Path

# The 15 study days, sorted (index 0 = first day). Weekends = Sat/Sun.
DAYS = [
    "2014-03-12", "2014-03-13", "2014-03-14", "2014-03-15", "2014-03-16",
    "2014-03-17", "2014-03-18", "2014-03-19", "2014-03-20", "2014-03-21",
    "2014-03-22", "2014-03-23", "2014-03-24", "2014-03-25", "2014-03-26",
]


def get_day(filepath: Path) -> str:
    """Extract the day (YYYY-MM-DD) from a day-file name (token before '_')."""
    return filepath.name.split("_")[0]


def is_weekend(day: str) -> bool:
    """True if `day` (format 'YYYY-MM-DD') falls on a Saturday or Sunday."""
    y, m, d = day.split("-")
    return datetime.date(int(y), int(m), int(d)).weekday() > 4


def get_cell_code(cell: str) -> str:
    """Simple merge: keep only the leading letters (e.g. 'BSOCHO1' -> 'BSOCHO')."""
    if cell == "":
        return cell
    return re.match(r"([a-zA-Z]+)", cell).group(1)


def get_cell_code2(cell: str) -> str:
    """2G/3G merge: leading letters minus the technology prefix ('BSOCHO1' -> 'SOCHO')."""
    if cell == "":
        return cell
    return re.match(r"([a-zA-Z]+)", cell).group(1)[1:]


def color_weekend_xticklabels(ax) -> None:
    """Color the x tick labels red when the label text is a weekend date."""
    for label in ax.get_xticklabels():
        text = label.get_text().strip()
        try:
            weekend = is_weekend(text)
        except (ValueError, IndexError):
            weekend = False
        if weekend:
            label.set_color("#ff0000")
            label.set_fontweight("bold")
