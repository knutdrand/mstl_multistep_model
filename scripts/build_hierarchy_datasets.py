"""Aggregate the sector dataset up to district and national levels for hierarchical reconciliation.

MinT needs independent base forecasts at every level, so we build CHAP-compatible datasets where
each "location" is a district (46) or the nation (1). Extensive quantities (cases, population) are
summed; intensive ones (climate, IRS coverage) are population-weighted means; the insecticide is the
district-month's modal non-zero product (spraying is district-coordinated, so well-defined).

Writes /Users/knutdr/Data/CH/chap_{district,national}_monthly.csv
Usage: uv run python scripts/build_hierarchy_datasets.py
"""
from __future__ import annotations
import numpy as np, pandas as pd

SRC = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
CLIM = ["rainfall_era5", "rainfall_chirps", "rainfall_iri", "mean_temperature", "max_temperature",
        "min_temperature", "dewpoint_temperature", "relative_humidity", "evi", "ndvi"]
SUMS = ["disease_cases", "population"]


def modal_chem(s):
    v = [x for x in s.astype(str) if x not in ("0", "nan", "")]
    return max(set(v), key=v.count) if v else "0"


def aggregate(df, keys, loc_name):
    out = []
    for kv, g in df.groupby(keys):
        pop = pd.to_numeric(g["population"], errors="coerce").to_numpy(float)
        w = pop / (np.nansum(pop) or 1.0)
        row = {"time_period": kv if isinstance(kv, str) else kv[-1]}
        # location id
        row["location"] = (kv if isinstance(kv, str) else kv[0]) if loc_name is None else loc_name
        for c in SUMS:
            row[c] = float(np.nansum(pd.to_numeric(g[c], errors="coerce")))
        for c in CLIM:
            x = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
            m = np.isfinite(x)
            row[c] = float(np.sum(x[m] * w[m]) / (np.sum(w[m]) or 1.0)) if m.any() else np.nan
        # IRS coverage: population-weighted mean; insecticide: modal non-zero
        ia = pd.to_numeric(g["irs_allocated"], errors="coerce").to_numpy(float)
        row["irs_allocated"] = float(np.nansum(ia * w))
        row["irs_insecticide_used"] = modal_chem(g["irs_insecticide_used"])
        out.append(row)
    r = pd.DataFrame(out)
    r["location_name"] = r["location"]
    ts = pd.PeriodIndex(r["time_period"].astype(str), freq="M")
    r["year"] = ts.year; r["month"] = ts.month
    return r.sort_values(["location", "time_period"]).reset_index(drop=True)


def main():
    df = pd.read_csv(SRC); df["location"] = df["location"].astype(str); df["district"] = df["district"].astype(str)
    dist = aggregate(df, ["district", "time_period"], None)
    dist.to_csv("/Users/knutdr/Data/CH/chap_district_monthly.csv", index=False)
    print(f"district: {dist.location.nunique()} districts x {dist.time_period.nunique()} months -> {len(dist)} rows")
    natl = aggregate(df, ["time_period"], "RW")
    natl.to_csv("/Users/knutdr/Data/CH/chap_national_monthly.csv", index=False)
    print(f"national: {natl.location.nunique()} x {natl.time_period.nunique()} months -> {len(natl)} rows")
    # sanity: district cases sum to national
    print("cases coherence (district-sum vs national):",
          np.allclose(dist.groupby("time_period")["disease_cases"].sum().values,
                      natl.set_index("time_period")["disease_cases"].values))


if __name__ == "__main__":
    main()
