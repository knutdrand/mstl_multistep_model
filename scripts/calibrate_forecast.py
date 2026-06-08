"""Apply interval calibration to a model backtest .nc (the integration entry point).

Pipeline: chap eval -> [chap aggregate-eval for district] -> calibrate_forecast.

  # sector: per-location log-space calibration (fixes per-sector coverage heterogeneity)
  uv run python scripts/calibrate_forecast.py output/arima012_bank.nc --level sector \
      --out output/arima012_bank_sector_cal.nc

  # district: per-horizon spread inflation on a bottom-up (aggregate-eval'd) district .nc
  uv run python scripts/calibrate_forecast.py output/arima012_bank_district.nc --level district \
      --out output/arima012_bank_district_cal.nc
"""
from __future__ import annotations
import argparse
import numpy as np, xarray as xr
from mstl_multistep.calibration import calibrate_sector_intervals, calibrate_district_intervals


def _cov80(ds):
    fc = ds.forecast.values; obs = ds.observed.values
    c = []
    L, T, H, _ = fc.shape
    for i in range(L):
        for t in range(T):
            y = obs[i, t]
            for h in range(H):
                s = fc[i, t, h]; s = s[np.isfinite(s)]
                if s.size == 0 or not np.isfinite(y): continue
                ql, qh = np.quantile(s, [.1, .9]); c.append(ql <= y <= qh)
    return float(np.mean(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nc")
    ap.add_argument("--level", choices=["sector", "district"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shrink-k", type=float, default=10.0)
    a = ap.parse_args()
    ds = xr.open_dataset(a.nc)
    before = _cov80(ds)
    if a.level == "sector":
        cal, lam = calibrate_sector_intervals(ds, a.shrink_k)
        print(f"sector calibration: {len(lam)} locations, lambda mean={np.mean([v.mean() for v in lam.values()]):.3f}")
    else:
        cal, lam = calibrate_district_intervals(ds)
        print(f"district calibration: per-horizon lambda={np.round(lam,3)}")
    cal.to_netcdf(a.out)
    print(f"cov@80%: {before:.3f} -> {_cov80(cal):.3f}  (nominal 0.80)   wrote {a.out}")


if __name__ == "__main__":
    main()
