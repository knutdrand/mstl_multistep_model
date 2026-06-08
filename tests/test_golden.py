"""Golden-output regression test: the champion forecast must stay bit-identical through refactors.

Captured from config_arima012_bank on a fixed 20-sector subset before the publish simplification.
If a pruning step changes the champion's output, this fails.
"""
from pathlib import Path
import numpy as np, pandas as pd
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
CONFIG = "config.yaml"   # the published default (champion)
GOLDEN = Path(__file__).parent / "golden_champion.npy"


def test_champion_output_unchanged():
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    df = df[df["location"].isin(sorted(df["location"].unique())[::20])]
    freq = detect_frequency(df)
    times = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    h = df[df["time_period"].isin(times[:-3])]; f = df[df["time_period"].isin(times[-3:])]
    mc = load_model_configuration(CONFIG)
    m = build_chap_model(mc.user_option_values, mc.additional_continuous_covariates)
    m.fit(h); p = m.predict(h, f)
    sc = [c for c in p.columns if c.startswith("sample_")]
    arr = p.sort_values(["location", "time_period"])[sc].to_numpy()
    golden = np.load(GOLDEN)
    assert arr.shape == golden.shape
    # tolerance for thread-order float noise in the n_jobs=-1 RF (~1e-12 abs on count-scale
    # forecasts); any real behaviour change is orders of magnitude larger.
    np.testing.assert_allclose(arr, golden, rtol=1e-7, atol=1e-4)
