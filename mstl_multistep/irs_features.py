"""Feature extraction from a raw IRS allocation column.

The raw ``irs_allocated`` column is a 0-1 coverage fraction that fires only in
the months a spray campaign is actually allocated (~2.5% of rows). Its
*protective effect*, however, persists for months after the campaign and is
known ahead of time (allocation is planned), so feeding the raw column through
the ordinary lag machinery mostly feeds the model zeros.

This module turns that sparse event series into a few **dense, contemporaneous**
signals, computed per location over the time-sorted historic(+future) panel:

- ``level``      — the raw allocation coverage this month (0-1).
- ``decay``      — protection that resets to the allocation level on a campaign
                   month and decays geometrically afterwards
                   (``d_t = max(level_t, gamma * d_{t-1})``,
                   ``gamma = 0.5 ** (1 / halflife)``). Dense and bounded in [0, 1].
- ``since``      — months since the last allocation, capped at ``since_cap``
                   (large when never sprayed).
- ``cumulative`` — running count of allocated months: a stock-of-protection /
                   program-intensity proxy.

These are returned at lag 0 (the campaign month itself) because allocation is a
known future covariate — unlike climate, we do not have to lag it to avoid
leakage. The caller merges them onto its design matrix as extra RF features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INDEX_COLS = ["time_period", "location"]

IRS_FEATURE_NAMES = (
    "level", "decay", "since", "cumulative",
    # Direction-2 feature bank (let the RF pick by importance):
    "chem_channels",                  # per-class decay channels: irs_decay_<class> (4 cols)
    "decay2", "decay8",               # decay basis at fixed half-lives 2 and 8 months
    "recent3", "recent6", "recent12", # sprayed within the last k months (binary)
    "rounds12",                       # number of campaign months in the trailing 12
)

# Per-insecticide decay half-life (months) of the residual protective effect, from the
# literature (WHO PQ / PMI residual-efficacy bioassays; approximate, surface-dependent).
# Keyed by a substring of the raw ``irs_insecticide_used`` string (matched case-insensitively).
# The ordering carbamate < pyrethroid < organophosphate < clothianidin is the robust part.
CHEM_HALFLIFE_MONTHS = {
    "bendiocarb": 2.0,     # carbamate WP — short residual (~2-4 mo, often 2 rounds/yr)
    "deltamethrin": 3.0,   # pyrethroid WG — ~3 mo
    "actellic": 5.0,       # pirimiphos-methyl 300CS (organophosphate, microencapsulated) ~4-6 mo
    "pirimiphos": 5.0,     # same active ingredient as Actellic 300CS
    "fludora": 8.0,        # Fludora Fusion (clothianidin + deltamethrin) — long, ~6-12 mo
    "clothianidin": 8.0,   # clothianidin-based (e.g. SumiShield) — long
    "sumishield": 8.0,
}


def chem_halflife(s, default: float) -> float:
    """Map a raw insecticide string to its literature decay half-life (months).

    Combination products (e.g. ``"Actellic 300CS+Fludora"``) take the *longer-lasting*
    component (max half-life). Missing / unrecognised strings fall back to ``default``.
    """
    txt = str(s).lower()
    found = [hl for kw, hl in CHEM_HALFLIFE_MONTHS.items() if kw in txt]
    return max(found) if found else float(default)


# Insecticide -> chemical class (for per-class decay channels). Same keyword matching as the
# half-life map; combos resolve to the *longer-lasting* class (the one with the max half-life).
CHEM_CLASS_OF_KEYWORD = {
    "bendiocarb": "carbamate", "deltamethrin": "pyrethroid",
    "actellic": "organophosphate", "pirimiphos": "organophosphate",
    "fludora": "clothianidin", "clothianidin": "clothianidin", "sumishield": "clothianidin",
}
CHEM_CLASSES = ("carbamate", "pyrethroid", "organophosphate", "clothianidin")


def chem_class(s) -> str | None:
    """Chemical class of an insecticide string (longer-lasting component for combos); None if unknown."""
    txt = str(s).lower()
    hits = [(CHEM_HALFLIFE_MONTHS[kw], CHEM_CLASS_OF_KEYWORD[kw])
            for kw in CHEM_CLASS_OF_KEYWORD if kw in txt]
    return max(hits)[1] if hits else None


def _decay_series(level: np.ndarray, gamma: float) -> np.ndarray:
    """``d_t = max(level_t, gamma * d_{t-1})`` — geometric decay, reset on spray."""
    out = np.zeros(len(level), dtype=float)
    prev = 0.0
    for i, lv in enumerate(level):
        prev = max(float(lv), gamma * prev)
        out[i] = prev
    return out


def _decay_series_chem(level: np.ndarray, halflife: np.ndarray) -> np.ndarray:
    """Chemical-aware decay: each campaign sets the active half-life from its insecticide.

    ``d_t = max(level_t, gamma_active * d_{t-1})`` where ``gamma_active`` is reset on a
    campaign month (``level_t > 0``) to ``0.5**(1/halflife_t)`` of that campaign's chemical,
    and held until the next campaign — so a Fludora round decays far slower than a Bendiocarb one.
    """
    out = np.zeros(len(level), dtype=float)
    prev = 0.0
    gamma = 0.5 ** (1.0 / max(float(halflife[0]) if len(halflife) else 4.0, 1e-6))
    for i, lv in enumerate(level):
        if lv > 0:
            gamma = 0.5 ** (1.0 / max(float(halflife[i]), 1e-6))
        prev = max(float(lv), gamma * prev)
        out[i] = prev
    return out


def build_irs_features(
    historic_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    column: str,
    features: list[str],
    halflife: float,
    since_cap: int = 24,
    chem_column: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(frame[INDEX_COLS + irs_cols], irs_cols)`` of engineered IRS features.

    ``frame`` covers every (time_period, location) in historic(+future). Each
    feature is computed per location over the chronologically sorted union so the
    decay/cumulative state flows correctly from history into the forecast window.
    Returns ``(empty-index frame, [])`` when the column is absent or no features
    are requested.
    """
    requested = [f for f in features if f in IRS_FEATURE_NAMES]
    base = historic_df if future_df is None else pd.concat(
        [historic_df, future_df], ignore_index=True
    )
    if not requested or column not in base.columns:
        return base[INDEX_COLS].copy(), []

    gamma = 0.5 ** (1.0 / max(float(halflife), 1e-6))
    use_chem = bool(chem_column) and chem_column in base.columns
    keep = INDEX_COLS + [column] + ([chem_column] if use_chem else [])
    src = base[keep].copy()
    src[column] = pd.to_numeric(src[column], errors="coerce").fillna(0.0)
    if use_chem:
        src["_hl"] = src[chem_column].apply(lambda s: chem_halflife(s, halflife))
        src["_cls"] = src[chem_column].apply(chem_class)
    src["_ts"] = pd.PeriodIndex(src["time_period"].astype(str), freq="M").to_timestamp()

    out_blocks = []
    for _, g in src.groupby("location", sort=False):
        g = g.sort_values("_ts").copy()
        level = g[column].to_numpy(dtype=float)
        nonzero = level > 0

        feat = {}
        if "level" in requested:
            feat["irs_level"] = level
        if "decay" in requested:
            if use_chem:
                feat["irs_decay"] = _decay_series_chem(level, g["_hl"].to_numpy(dtype=float))
            else:
                feat["irs_decay"] = _decay_series(level, gamma)
        need_since = any(f in requested for f in ("since", "recent3", "recent6", "recent12"))
        if need_since:
            since = np.empty(len(level), dtype=float)
            last = -1
            for i in range(len(level)):
                if nonzero[i]:
                    last = i
                since[i] = since_cap if last < 0 else min(i - last, since_cap)
        if "since" in requested:
            feat["irs_since"] = since
        if "cumulative" in requested:
            feat["irs_cumulative"] = np.cumsum(level)

        # --- Direction-2 bank ---
        if "chem_channels" in requested and use_chem:
            dec_chem = _decay_series_chem(level, g["_hl"].to_numpy(dtype=float))
            cls_all = g["_cls"].to_numpy(dtype=object)
            active = np.empty(len(level), dtype=object)
            cur = None
            for i in range(len(level)):
                if nonzero[i]:
                    cur = cls_all[i]
                active[i] = cur
            for klass in CHEM_CLASSES:
                feat[f"irs_decay_{klass}"] = np.where(active == klass, dec_chem, 0.0)
        if "decay2" in requested:
            feat["irs_decay2"] = _decay_series(level, 0.5 ** (1.0 / 2.0))
        if "decay8" in requested:
            feat["irs_decay8"] = _decay_series(level, 0.5 ** (1.0 / 8.0))
        for k in (3, 6, 12):
            if f"recent{k}" in requested:
                feat[f"irs_recent{k}"] = (since <= k).astype(float)
        if "rounds12" in requested:
            feat["irs_rounds12"] = (
                pd.Series(nonzero.astype(float)).rolling(12, min_periods=1).sum().to_numpy()
            )

        blk = g[INDEX_COLS].copy()
        for k, v in feat.items():
            blk[k] = v
        out_blocks.append(blk)

    frame = pd.concat(out_blocks, ignore_index=True)
    irs_cols = [c for c in frame.columns if c not in INDEX_COLS]
    return frame, irs_cols
