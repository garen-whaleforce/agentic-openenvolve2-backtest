#!/usr/bin/env python3
"""
run_vix_regime_backtest.py

VIX-based Regime Throttle Backtest Runner

This script implements observable, rule-based regime throttling using VIX:
- NORMAL (VIX <= 25): Full cap (12/quarter)
- RISK_OFF (25 < VIX <= 30): Reduced cap (6/quarter)
- STRESS (VIX > 30): No new positions (0/quarter)

Key requirement: VIX is read at decision time (reaction_date close),
which is BEFORE entry (T+1 open). This ensures no look-ahead bias.

Usage:
    python run_vix_regime_backtest.py \
        --signals validation_results.csv \
        --prices-folder /path/to/ohlc \
        --vix-csv /path/to/vix.csv \
        --outdir out_vix_regime \
        --tier D7_CORE

Options:
    --tier: D7_CORE (default), D6_STRICT, or ALL
    --vix-normal: VIX threshold for NORMAL regime (default: 25)
    --vix-stress: VIX threshold for STRESS regime (default: 30)
    --cap-normal: Cap per quarter in NORMAL regime (default: 12)
    --cap-riskoff: Cap per quarter in RISK_OFF regime (default: 6)
    --cost-bps: Round-trip transaction cost in basis points (default: 0)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from backtester import BacktestConfig, run_backtest, prepare_signals, _year_quarter
from metrics import compute_perf_metrics, to_dict
from price_providers import CSVPriceProvider
from calendar_utils import TradingCalendar


@dataclass
class RegimeConfig:
    """VIX-based regime thresholds and caps"""
    vix_normal_max: float = 25.0  # VIX <= this = NORMAL
    vix_stress_min: float = 30.0  # VIX > this = STRESS
    cap_normal: int = 12
    cap_riskoff: int = 6
    cap_stress: int = 0


def load_vix_data(vix_csv: Path) -> pd.Series:
    """
    Load VIX close prices from CSV.

    Expected format:
    - date column (or Date, DATE)
    - close column (or Close, CLOSE, Adj Close, etc.)

    Returns: pd.Series indexed by date with VIX close values
    """
    df = pd.read_csv(vix_csv)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Find date column
    date_col = None
    for c in ["date", "datetime", "timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"VIX CSV missing date column. Found: {df.columns.tolist()}")

    # Find close column
    close_col = None
    for c in ["close", "adj_close", "adjclose", "vix_close"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise ValueError(f"VIX CSV missing close column. Found: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col].dt.normalize()
    df = df.set_index(date_col).sort_index()

    vix = df[close_col].astype(float)
    vix.name = "vix_close"

    return vix


def get_regime(vix_value: float, config: RegimeConfig) -> str:
    """Determine regime based on VIX value"""
    if vix_value <= config.vix_normal_max:
        return "NORMAL"
    elif vix_value <= config.vix_stress_min:
        return "RISK_OFF"
    else:
        return "STRESS"


def get_cap_for_regime(regime: str, config: RegimeConfig) -> int:
    """Get cap per quarter for given regime"""
    if regime == "NORMAL":
        return config.cap_normal
    elif regime == "RISK_OFF":
        return config.cap_riskoff
    else:  # STRESS
        return config.cap_stress


def apply_vix_regime_filter(
    signals: pd.DataFrame,
    vix: pd.Series,
    regime_config: RegimeConfig,
    tier_filter: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply VIX-based regime throttling to signals.

    Args:
        signals: DataFrame with symbol, reaction_date, trade_long, tier
        vix: Series of VIX close indexed by date
        regime_config: Regime thresholds and caps
        tier_filter: "D7_CORE", "D6_STRICT", or None for all

    Returns:
        filtered_signals: DataFrame ready for backtest
        regime_log: DataFrame with regime decisions for audit
    """
    df = signals.copy()

    # Ensure reaction_date is datetime
    df["reaction_date"] = pd.to_datetime(df["reaction_date"]).dt.normalize()

    # Filter by tier if specified
    if tier_filter and tier_filter != "ALL":
        if "tier" in df.columns:
            df = df[df["tier"] == tier_filter].copy()
        elif "classification" in df.columns:
            # Map classification to tier
            tier_map = {
                "D7_CORE": "D7_CORE",
                "D6_STRICT": "D6_STRICT",
                "d7_core": "D7_CORE",
                "d6_strict": "D6_STRICT",
            }
            df = df[df["classification"].map(tier_map) == tier_filter].copy()

    # Filter to trade_long only
    if "trade_long" in df.columns:
        if df["trade_long"].dtype == object:
            df["trade_long"] = df["trade_long"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df = df[df["trade_long"] == True].copy()

    # Sort by reaction_date
    df = df.sort_values("reaction_date").reset_index(drop=True)

    # Track caps per quarter with regime awareness
    accepted = []
    regime_log = []
    cap_used: Dict[Tuple[int, int], int] = {}
    held_symbols_by_quarter: Dict[Tuple[int, int], set] = {}

    cal = TradingCalendar("XNYS")

    for _, row in df.iterrows():
        rd = pd.Timestamp(row["reaction_date"]).normalize()
        sym = str(row["symbol"]).upper()
        yq = _year_quarter(rd)

        cap_used.setdefault(yq, 0)
        held_symbols_by_quarter.setdefault(yq, set())

        # Get VIX at decision time (reaction_date close)
        # If exact date not available, look back up to 5 days
        vix_value = None
        for lookback in range(6):
            check_date = rd - pd.Timedelta(days=lookback)
            if check_date in vix.index:
                vix_value = vix.loc[check_date]
                break

        if vix_value is None:
            # Skip if no VIX data available
            regime_log.append({
                "reaction_date": rd,
                "symbol": sym,
                "vix_value": np.nan,
                "regime": "UNKNOWN",
                "cap_available": 0,
                "accepted": False,
                "reason": "No VIX data",
            })
            continue

        # Determine regime and cap
        regime = get_regime(vix_value, regime_config)
        cap_limit = get_cap_for_regime(regime, regime_config)

        # Check if we can accept this trade
        can_accept = True
        reason = "Accepted"

        if cap_used[yq] >= cap_limit:
            can_accept = False
            reason = f"Cap reached ({cap_used[yq]}/{cap_limit})"
        elif sym in held_symbols_by_quarter[yq]:
            can_accept = False
            reason = "Duplicate symbol in quarter"

        regime_log.append({
            "reaction_date": rd,
            "symbol": sym,
            "vix_value": vix_value,
            "regime": regime,
            "cap_limit": cap_limit,
            "cap_used": cap_used[yq],
            "accepted": can_accept,
            "reason": reason,
        })

        if can_accept:
            cap_used[yq] += 1
            held_symbols_by_quarter[yq].add(sym)
            accepted.append(row.to_dict())

    if not accepted:
        raise ValueError("No trades accepted after VIX regime filtering")

    filtered_df = pd.DataFrame(accepted)
    regime_log_df = pd.DataFrame(regime_log)

    return filtered_df, regime_log_df


def run_cost_sensitivity(
    signals: pd.DataFrame,
    price_provider: CSVPriceProvider,
    base_config: BacktestConfig,
    cost_levels_bps: list = [0, 10, 20, 30, 50],
) -> pd.DataFrame:
    """
    Run backtest at multiple cost levels to test sensitivity.
    """
    results = []

    for cost in cost_levels_bps:
        cfg = BacktestConfig(
            calendar_name=base_config.calendar_name,
            holding_sessions=base_config.holding_sessions,
            entry_lag_sessions=base_config.entry_lag_sessions,
            cap_entries_per_quarter=base_config.cap_entries_per_quarter,
            max_concurrent_positions=base_config.max_concurrent_positions,
            commission_bps=cost / 2,  # Half on each side
            slippage_bps=cost / 2,
            annual_rf_rate=base_config.annual_rf_rate,
        )

        try:
            nav, trades, exposure = run_backtest(signals, price_provider, cfg)
            m = compute_perf_metrics(nav, daily_rf_rate=cfg.daily_rf_rate(), trades=trades, exposure=exposure)

            results.append({
                "cost_bps": cost,
                "total_return": m.total_return,
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "max_drawdown": m.max_drawdown,
                "calmar": m.calmar,
                "win_rate": m.win_rate,
                "n_trades": m.n_trades,
                "exposure_avg": m.exposure_avg,
            })
        except Exception as e:
            print(f"Cost sensitivity at {cost}bps failed: {e}")
            results.append({
                "cost_bps": cost,
                "error": str(e),
            })

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="VIX-based Regime Throttle Backtest")
    ap.add_argument("--signals", required=True, help="Validation results CSV")
    ap.add_argument("--prices-folder", required=True, help="Folder with per-symbol OHLC CSVs")
    ap.add_argument("--vix-csv", required=True, help="VIX historical data CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--tier", default="D7_CORE", choices=["D7_CORE", "D6_STRICT", "ALL"])

    # Regime thresholds
    ap.add_argument("--vix-normal", type=float, default=25.0, help="VIX threshold for NORMAL")
    ap.add_argument("--vix-stress", type=float, default=30.0, help="VIX threshold for STRESS")
    ap.add_argument("--cap-normal", type=int, default=12, help="Cap in NORMAL regime")
    ap.add_argument("--cap-riskoff", type=int, default=6, help="Cap in RISK_OFF regime")

    # Backtest params
    ap.add_argument("--holding-sessions", type=int, default=30)
    ap.add_argument("--max-positions", type=int, default=12)
    ap.add_argument("--cost-bps", type=float, default=0.0, help="Round-trip cost in bps")

    # Options
    ap.add_argument("--cost-sensitivity", action="store_true", help="Run cost sensitivity analysis")
    ap.add_argument("--compare-hardcoded", action="store_true", help="Compare with hard-coded regime")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading signals...")
    signals = pd.read_csv(args.signals)
    print(f"  Total rows: {len(signals)}")

    print("Loading VIX data...")
    vix = load_vix_data(Path(args.vix_csv))
    print(f"  VIX data range: {vix.index.min().date()} to {vix.index.max().date()}")
    print(f"  VIX mean: {vix.mean():.1f}, max: {vix.max():.1f}")

    print("Loading price provider...")
    pp = CSVPriceProvider(folder=Path(args.prices_folder))

    # Setup regime config
    regime_config = RegimeConfig(
        vix_normal_max=args.vix_normal,
        vix_stress_min=args.vix_stress,
        cap_normal=args.cap_normal,
        cap_riskoff=args.cap_riskoff,
        cap_stress=0,
    )

    print(f"\nRegime Configuration:")
    print(f"  NORMAL (VIX <= {regime_config.vix_normal_max}): cap={regime_config.cap_normal}")
    print(f"  RISK_OFF ({regime_config.vix_normal_max} < VIX <= {regime_config.vix_stress_min}): cap={regime_config.cap_riskoff}")
    print(f"  STRESS (VIX > {regime_config.vix_stress_min}): cap={regime_config.cap_stress}")

    # Apply VIX regime filter
    print(f"\nApplying VIX regime filter (tier={args.tier})...")
    filtered_signals, regime_log = apply_vix_regime_filter(
        signals, vix, regime_config, tier_filter=args.tier
    )

    # Save regime log
    regime_log.to_csv(outdir / "regime_log.csv", index=False)
    print(f"  Accepted trades: {len(filtered_signals)}")
    print(f"  Regime log saved to {outdir}/regime_log.csv")

    # Summarize regime distribution
    if not regime_log.empty:
        regime_dist = regime_log.groupby("regime")["accepted"].agg(["count", "sum"])
        regime_dist.columns = ["total_signals", "accepted"]
        print("\n  Regime Distribution:")
        print(regime_dist.to_string())

    # Run backtest
    print("\nRunning backtest...")
    cfg = BacktestConfig(
        calendar_name="XNYS",
        holding_sessions=args.holding_sessions,
        entry_lag_sessions=1,
        cap_entries_per_quarter=999,  # Already filtered by regime
        max_concurrent_positions=args.max_positions,
        commission_bps=args.cost_bps / 2,
        slippage_bps=args.cost_bps / 2,
    )

    nav, trades, exposure = run_backtest(filtered_signals, pp, cfg)

    # Save outputs
    nav.rename("nav").to_csv(outdir / "nav.csv", index=True)
    exposure.rename("gross_exposure").to_csv(outdir / "exposure.csv", index=True)
    trades.to_csv(outdir / "trades.csv", index=False)

    # Compute metrics
    m = compute_perf_metrics(nav, daily_rf_rate=cfg.daily_rf_rate(), trades=trades, exposure=exposure)

    # Summary
    summary_lines = [
        "# VIX Regime Throttle Backtest Summary\n",
        f"\n## Configuration\n",
        f"- Tier: {args.tier}\n",
        f"- VIX Normal Threshold: {regime_config.vix_normal_max}\n",
        f"- VIX Stress Threshold: {regime_config.vix_stress_min}\n",
        f"- Cap Normal: {regime_config.cap_normal}\n",
        f"- Cap Risk-Off: {regime_config.cap_riskoff}\n",
        f"- Transaction Cost: {args.cost_bps} bps\n",
        f"\n## Performance\n",
        f"- Total Return: {m.total_return*100:.2f}%\n",
        f"- CAGR: {m.cagr*100:.2f}%\n",
        f"- Annual Vol: {m.ann_vol*100:.2f}%\n",
        f"- Sharpe: {m.sharpe:.2f}\n",
        f"- Sortino: {m.sortino:.2f}\n",
        f"- Max Drawdown: {m.max_drawdown*100:.2f}%\n",
        f"- Calmar: {m.calmar:.2f}\n",
        f"- Trades: {m.n_trades}\n",
        f"- Win Rate: {m.win_rate*100:.2f}%\n",
        f"- Profit Factor: {m.profit_factor:.2f}\n",
        f"- Avg Exposure: {m.exposure_avg:.2f}\n",
    ]

    (outdir / "summary.md").write_text("".join(summary_lines))
    (outdir / "metrics.json").write_text(json.dumps(to_dict(m), indent=2))

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"CAGR: {m.cagr*100:.2f}%")
    print(f"Sharpe: {m.sharpe:.2f}")
    print(f"Max Drawdown: {m.max_drawdown*100:.2f}%")
    print(f"Calmar: {m.calmar:.2f}")
    print(f"Trades: {m.n_trades}")
    print(f"Win Rate: {m.win_rate*100:.2f}%")
    print(f"Avg Exposure: {m.exposure_avg:.2f}")
    print("="*60)

    # Cost sensitivity analysis
    if args.cost_sensitivity:
        print("\nRunning cost sensitivity analysis...")
        cost_results = run_cost_sensitivity(filtered_signals, pp, cfg)
        cost_results.to_csv(outdir / "cost_sensitivity.csv", index=False)
        print("\nCost Sensitivity Results:")
        print(cost_results[["cost_bps", "sharpe", "cagr", "win_rate"]].to_string(index=False))

    print(f"\nOutputs written to {outdir}")


if __name__ == "__main__":
    main()
