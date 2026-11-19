"""Track 05 Part 2 factor study (#5). Not part of the PR; linked from it at a fixed commit.

Rank IC of plain taker-flow imbalance (plain) and the trade-size-weighted version (weighted, Part 1)
at spans 6/24/72 against the next 1/4/24-hour close-to-close log returns, the paired difference
weighted - plain, and each signal's turnover.

    python -m research.tanmay.ic_study --data data/synthetic/train_1h.parquet   # debug
    python -m research.tanmay.ic_study                                          # full panel
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.contrib.tanmay import trade_intensity_imbalance
from src.features import cross_sectional_zscore, taker_flow_imbalance

SPANS = (6, 24, 72)
HORIZONS = (1, 4, 24)
WINDOW = 24
MIN_SYMBOLS = 5


def signals(df: pd.DataFrame, span: int) -> dict[str, pd.Series]:
    return {
        "plain": taker_flow_imbalance(df, span=span),
        "weighted": trade_intensity_imbalance(df, span=span, window=WINDOW),
    }


def forward_log_return(df: pd.DataFrame, h: int) -> pd.Series:
    """log(close[t+h] / close[t]) per symbol: the return after the signal on the close of bar t."""
    log_close = np.log(df["close"])
    return log_close.groupby(df["symbol"]).shift(-h) - log_close


def hourly_rank_ic(df: pd.DataFrame, signal: pd.Series, fwd: pd.Series) -> pd.Series:
    """Spearman correlation across symbols at each timestamp (average ranks, >= MIN_SYMBOLS)."""
    idx = pd.MultiIndex.from_arrays([df["timestamp"], df["symbol"]])
    s = pd.Series(signal.to_numpy(), index=idx).unstack()
    r = pd.Series(fwd.to_numpy(), index=idx).unstack()
    ok = s.notna() & r.notna()
    rs = s.where(ok).rank(axis=1)
    rr = r.where(ok).rank(axis=1)
    rs = rs.sub(rs.mean(axis=1), axis=0)
    rr = rr.sub(rr.mean(axis=1), axis=0)
    den = np.sqrt((rs**2).sum(axis=1) * (rr**2).sum(axis=1))
    ic = (rs * rr).sum(axis=1) / den
    return ic.where((ok.sum(axis=1) >= MIN_SYMBOLS) & (den > 1e-12)).dropna()


def newey_west_t(x: np.ndarray, lags: int) -> float:
    """t-stat of the mean with Bartlett-weighted (Newey-West) errors; lags = 0 is the plain t."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(var / n))


def two_sided_p(t: float) -> float:
    return float(2.0 * stats.norm.sf(abs(t)))


def signal_turnover(df: pd.DataFrame, signal: pd.Series) -> float:
    """Mean hourly sum |w_t - w_(t-1)| of the dollar-neutral, L1-normalised weights (no deadband)."""
    w = cross_sectional_zscore(df, signal.fillna(0.0))
    dw = (w - w.groupby(df["symbol"]).shift(1)).abs()
    return float(dw.groupby(df["timestamp"]).sum().iloc[1:].mean())


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fwd = {h: forward_log_return(df, h) for h in HORIZONS}
    rows, diffs = [], []
    for span in SPANS:
        sig = signals(df, span)
        turnover = {name: signal_turnover(df, s) for name, s in sig.items()}
        for h in HORIZONS:
            ics = {name: hourly_rank_ic(df, s, fwd[h]) for name, s in sig.items()}
            for name, ic in ics.items():
                t_nw = newey_west_t(ic.to_numpy(), h - 1)
                sub = ic.iloc[::h]
                rows.append({
                    "signal": name, "span": span, "h": h, "n_hours": len(ic),
                    "mean_ic": ic.mean(), "t": t_nw, "t_method": "plain" if h == 1 else f"NW({h - 1})",
                    "p": two_sided_p(t_nw),
                    "nonoverlap_ic": sub.mean() if h > 1 else np.nan,
                    "nonoverlap_t": newey_west_t(sub.to_numpy(), 0) if h > 1 else np.nan,
                    "turnover": turnover[name],
                })
            d = (ics["weighted"] - ics["plain"]).dropna()
            t_d = newey_west_t(d.to_numpy(), h - 1)
            diffs.append({"span": span, "h": h, "mean_diff_w_minus_p": d.mean(), "t": t_d, "p": two_sided_p(t_d)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train_1h.parquet")
    parser.add_argument("--out", default="research/tanmay/out")
    args = parser.parse_args()
    df = pd.read_parquet(args.data).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    table, diffs = run(df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ic_table.csv", index=False)
    diffs.to_csv(out / "ic_diff.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.5f}".format):
        print(f"{args.data}: {df['timestamp'].nunique()} hours x {df['symbol'].nunique()} symbols")
        print(table.to_string(index=False))
        print(diffs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())