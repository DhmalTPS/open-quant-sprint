"""Track 05 Part 2 strategy grid (#5). Not part of the PR; linked from it at a fixed commit.

Primary cell = the settings in strategies/tanmay_flow.py. The other cells are the pre-registered
deadbands and the plain-imbalance version with the same settings, reported next to it.

    python -m research.tanmay.grid --data data/synthetic/train_1h.parquet   # debug
    python -m research.tanmay.grid                                          # full panel
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.features import cross_sectional_zscore, rebalance_deadband, taker_flow_imbalance
from strategies.tanmay_flow import DEADBAND, SIGN, SPAN, build_weights

DEADBANDS = (0.005, 0.015, 0.03)
FEES = (0.0, 3.0, 5.0, 10.0)


def plain_weights(df: pd.DataFrame, span: int, deadband: float) -> pd.Series:
    signal = SIGN * taker_flow_imbalance(df, span=span)
    return rebalance_deadband(df, cross_sectional_zscore(df, signal.fillna(0.0)), buffer=deadband)


def row(name: str, df: pd.DataFrame, w: pd.Series) -> dict:
    out = {"cell": name}
    for fee in FEES:
        m = run_backtest(df, w, fee_bps=fee)
        out[f"SR {fee:g}bps"] = m.sharpe
        if fee == 5.0:
            out["CI 5bps"] = f"[{m.sharpe_ci_low:+.2f}, {m.sharpe_ci_high:+.2f}]"
            out["turnover"] = m.mean_turnover
            gross = run_backtest(df, w, fee_bps=0.0)
            # Break-even fee: gross PnL per unit of turnover, in bps (funding left in).
            out["break-even bps"] = round(gross.net_pnl_cum / max(1e-12, m.fee_cost_cum / 5.0), 2)
    out["SR lag2 5bps"] = run_backtest(df, w, fee_bps=5.0, lag_bars=2).sharpe
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train_1h.parquet")
    args = parser.parse_args()
    df = pd.read_parquet(args.data).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    rows = []
    for b in DEADBANDS:
        tag = " (primary)" if b == DEADBAND else ""
        rows.append(row(f"weighted span={SPAN} b={b}{tag}", df, build_weights(df, SPAN, b)))
    rows.append(row(f"plain span={SPAN} b={DEADBAND} (comparison)", df, plain_weights(df, SPAN, DEADBAND)))
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())