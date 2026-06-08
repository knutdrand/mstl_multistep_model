"""Check predictive-interval coverage at sector vs district (bottom-up) vs district (independent).

Bottom-up aggregation sums *independently drawn* sector samples, so the aggregate spread =
sum of sector variances -- it ignores cross-sector error correlation (which we measured at
~0.27 within-district). Positive correlation means true district variance is larger, so
bottom-up intervals are too narrow -> coverage below nominal. This quantifies it.

Reports empirical coverage at the 80% (10-90) and 50% (25-75) nominal levels, plus a dispersion
ratio (mean predictive std / RMSE; <1 = under-dispersed), overall and per horizon.
"""
from __future__ import annotations
import numpy as np, xarray as xr

FILES = {
    "sector": "output/arima012_bank.nc",
    "district (bottom-up)": "output/arima012_bank_district.nc",
    "district (independent)": "output/district_base.nc",
    "national (independent)": "output/national_base.nc",
}


def cover(ds):
    fc = ds.forecast.values            # (loc, tp, h, sample)
    obs = ds.observed.values           # (loc, tp)
    L, T, H, _ = fc.shape
    res = {}
    for h in range(H):
        c80 = c50 = disp = rmse_acc = n = 0.0
        sd_acc = 0.0
        for i in range(L):
            for t in range(T):
                s = fc[i, t, h]; y = obs[i, t]
                s = s[np.isfinite(s)]
                if s.size == 0 or not np.isfinite(y):
                    continue
                q10, q25, q75, q90 = np.quantile(s, [.1, .25, .75, .9])
                c80 += (q10 <= y <= q90); c50 += (q25 <= y <= q75)
                sd_acc += s.std(); rmse_acc += (s.mean() - y) ** 2
                n += 1
        res[h + 1] = (c80 / n, c50 / n, sd_acc / n, np.sqrt(rmse_acc / n), int(n))
    return res


def main():
    print(f"{'level':<24} {'h':>2} | {'cov80':>6} {'cov50':>6} | {'meanSD':>9} {'RMSE':>9} {'SD/RMSE':>7} | n")
    print(f"{'(nominal)':<24} {'':>2} | {0.80:>6} {0.50:>6} |")
    for name, path in FILES.items():
        try:
            ds = xr.open_dataset(path)
        except Exception as e:
            print(f"{name}: missing ({e})"); continue
        r = cover(ds)
        for h, (c80, c50, sd, rmse, n) in r.items():
            print(f"{name:<24} {h:>2} | {c80:>6.3f} {c50:>6.3f} | {sd:>9.1f} {rmse:>9.1f} "
                  f"{sd/(rmse+1e-9):>7.2f} | {n}")
        # pooled
        allc80 = np.mean([r[h][0] for h in r]); allc50 = np.mean([r[h][1] for h in r])
        print(f"{name:<24} {'all':>2} | {allc80:>6.3f} {allc50:>6.3f} |")
        print()


if __name__ == "__main__":
    main()
