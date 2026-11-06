"""Frequency detection and CHAP ``time_period`` parsing.

CHAP encodes monthly periods as ``YYYY-MM`` and weekly periods as
``YYYY-Wnn`` (or a ``start/end`` ISO range). These helpers mirror the
conventions used elsewhere in the chap_nixtla wrapper.
"""

from __future__ import annotations

import pandas as pd


def detect_frequency(df: pd.DataFrame) -> str:
    """Return ``"MS"`` for monthly data or ``"W-MON"`` for weekly data."""
    sample = str(df["time_period"].iloc[0])
    lower = sample.lower()
    if "w" in lower:
        return "W-MON"
    if "/" in sample:
        return "W-MON"
    parts = sample.split("-")
    if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 12:
        return "W-MON"
    return "MS"


def period_to_timestamp(period: str, freq: str) -> pd.Timestamp:
    """Parse a CHAP ``time_period`` string to a pandas Timestamp."""
    s = str(period)
    if freq == "MS":
        return pd.to_datetime(s + "-01")
    if "/" in s:
        return pd.to_datetime(s.split("/")[0])
    if "w" in s.lower():
        year, week = s.lower().split("-w")
        return pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%G-W%V-%u")
    return pd.to_datetime(s)


def periods_to_timestamps(periods: pd.Series, freq: str) -> pd.Series:
    """Vectorised :func:`period_to_timestamp`: parse each distinct period once, then map.

    Panels repeat the same few hundred periods across every location, so this is
    far cheaper than ``periods.apply(period_to_timestamp)`` row by row.
    """
    as_str = periods.astype(str)
    lookup = {p: period_to_timestamp(p, freq) for p in pd.unique(as_str)}
    return as_str.map(lookup)
