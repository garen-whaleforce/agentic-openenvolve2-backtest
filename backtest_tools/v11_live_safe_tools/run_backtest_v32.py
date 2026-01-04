#!/usr/bin/env python3
"""
run_backtest_v32.py

CLI wrapper for backtester_v32 with leverage, stop-loss, and dynamic sizing.

Usage:
    python run_backtest_v32.py --signals signals.csv --vix vix_data.csv \\
        --gross-normal 1.25 --gross-riskoff 0.6 --gross-stress 0.0 \\
        --stop-loss 0.12 --costs 20 --borrow-rate 0.06 \\
        --output-dir out_v32
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


def compute_metrics(nav: pd.Series, rf_annual: float = 0.0) -> dict:
    """Compute performance metrics from NAV series."""
    if nav.empty or len(nav) < 2:
        return {}

    # Returns
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {}

    # Basic stats
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    n_days = len(rets)
    n_years = n_days / 252.0

    cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    ann_vol = rets.std() * np.sqrt(252)

    # Sharpe
    rf_daily = rf_annual / 252.0
    excess_rets = rets - rf_daily
    sharpe = (excess_rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0

    # Sortino
    downside = rets[rets < rf_daily]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else 0.0
    sortino = (cagr - rf_annual) / downside_std if downside_std > 0 else 0.0

    # Drawdown
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def compute_trade_metrics(trades: pd.DataFrame) -> dict:
    """Compute trade-level metrics."""
    if trades.empty:
        return {}

    n_trades = len(trades)
    wins = trades[trades["net_ret"] > 0]
    losses = trades[trades["net_ret"] <= 0]

    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = wins["net_ret"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["net_ret"].mean() if len(losses) > 0 else 0.0

    gross_profit = wins["net_ret"].sum()
    gross_loss = abs(losses["net_ret"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def main():
    parser = argparse.ArgumentParser(description="Run v3.2 backtest with leverage and stop-loss")

    # Input files
    parser.add_argument("--signals", required=True, help="Path to signals CSV")
    parser.add_argument("--vix", default="vix_data.csv", help="Path to VIX data CSV")
    parser.add_argument("--prices-folder", default="../prices", help="Path to prices folder")

    # Target gross by regime
    parser.add_argument("--gross-normal", type=float, default=1.0, help="Target gross in NORMAL regime")
    parser.add_argument("--gross-riskoff", type=float, default=0.6, help="Target gross in RISK_OFF regime")
    parser.add_argument("--gross-stress", type=float, default=0.0, help="Target gross in STRESS regime")

    # VIX thresholds
    parser.add_argument("--vix-normal", type=float, default=22.0, help="VIX threshold for NORMAL/RISK_OFF")
    parser.add_argument("--vix-stress", type=float, default=28.0, help="VIX threshold for RISK_OFF/STRESS")

    # Stop-loss
    parser.add_argument("--stop-loss", type=float, default=None, help="Fixed stop-loss percentage (e.g., 0.12 = 12%)")
    parser.add_argument("--stop-loss-atr", type=float, default=None, help="ATR multiplier for stop-loss")
    parser.add_argument("--trailing-trigger", type=float, default=None, help="Trailing stop trigger (e.g., 0.12 = +12%)")
    parser.add_argument("--trailing-level", type=float, default=None, help="Trailing stop level (e.g., 0.02 = -2%)")

    # Costs
    parser.add_argument("--costs", type=float, default=20.0, help="Round-trip costs in bps")
    parser.add_argument("--borrow-rate", type=float, default=0.06, help="Annual margin borrow rate")

    # Position sizing
    parser.add_argument("--per-trade-cap", type=float, default=0.20, help="Max allocation per trade")
    parser.add_argument("--max-positions", type=int, default=12, help="Max concurrent positions")
    parser.add_argument("--cap-per-quarter", type=int, default=12, help="Max entries per quarter")

    # Output
    parser.add_argument("--output-dir", default="out_v32", help="Output directory")
    parser.add_argument("--initial-cash", type=float, default=100000, help="Initial cash")

    args = parser.parse_args()

    # Load signals
    signals_path = Path(args.signals)
    if not signals_path.exists():
        print(f"Error: Signals file not found: {signals_path}")
        sys.exit(1)

    print(f"Loading signals from {signals_path}")
    signals = pd.read_csv(signals_path)
    print(f"  Total rows: {len(signals)}")

    # Filter to long trades only
    if "trade_long" in signals.columns:
        signals = signals[signals["trade_long"] == True].copy()
    print(f"  Long trades: {len(signals)}")

    # Load VIX data
    vix_path = Path(args.vix)
    vix_data = None
    if vix_path.exists():
        print(f"Loading VIX data from {vix_path}")
        vix_df = pd.read_csv(vix_path)
        # Handle both capitalized and lowercase column names
        date_col = "Date" if "Date" in vix_df.columns else "date"
        close_col = "Close" if "Close" in vix_df.columns else "close"
        vix_df[date_col] = pd.to_datetime(vix_df[date_col])
        vix_df = vix_df.set_index(date_col)[close_col]
        vix_df.index = pd.to_datetime(vix_df.index).normalize()
        vix_data = vix_df
        print(f"  VIX data: {len(vix_data)} days")
    else:
        print(f"Warning: VIX file not found: {vix_path}")
        print("  Using default VIX=15 (NORMAL regime)")

    # Build config
    config = BacktestConfigV32(
        initial_cash=args.initial_cash,
        target_gross_normal=args.gross_normal,
        target_gross_riskoff=args.gross_riskoff,
        target_gross_stress=args.gross_stress,
        vix_normal_threshold=args.vix_normal,
        vix_stress_threshold=args.vix_stress,
        stop_loss_pct=args.stop_loss,
        stop_loss_atr_mult=args.stop_loss_atr,
        trailing_stop_trigger_pct=args.trailing_trigger,
        trailing_stop_level_pct=args.trailing_level,
        commission_bps=args.costs / 2.0,  # Round-trip to per-side
        slippage_bps=args.costs / 2.0,
        annual_borrow_rate=args.borrow_rate,
        per_trade_cap=args.per_trade_cap,
        max_concurrent_positions=args.max_positions,
        cap_entries_per_quarter=args.cap_per_quarter,
    )

    print("\nConfiguration:")
    print(f"  Target Gross: NORMAL={config.target_gross_normal}, RISKOFF={config.target_gross_riskoff}, STRESS={config.target_gross_stress}")
    print(f"  VIX Thresholds: NORMAL<={config.vix_normal_threshold}, STRESS>{config.vix_stress_threshold}")
    print(f"  Stop-Loss: {config.stop_loss_pct*100 if config.stop_loss_pct else 'None'}%")
    print(f"  Costs: {args.costs} bps round-trip")
    print(f"  Borrow Rate: {config.annual_borrow_rate*100}% annual")

    # Run backtest
    print("\nRunning backtest...")
    price_provider = CSVPriceProvider(folder=Path(args.prices_folder))

    try:
        nav, trades, exposure, stats = run_backtest_v32(
            signals=signals,
            price_provider=price_provider,
            config=config,
            vix_data=vix_data,
        )
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(nav)
    trade_metrics = compute_trade_metrics(trades)
    metrics.update(trade_metrics)
    metrics["exposure_avg"] = exposure.mean()
    metrics["max_leverage"] = stats["max_leverage"]
    metrics["stop_loss_triggered"] = stats["stop_loss_triggered"]
    metrics["scheduled_exits"] = stats["scheduled_exits"]
    metrics["total_margin_interest"] = stats["total_margin_interest"]
    metrics["regime_trades"] = stats["regime_trades"]

    # Output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save NAV
    nav_df = nav.reset_index()
    nav_df.columns = ["date", "nav"]
    nav_df.to_csv(out_dir / "nav.csv", index=False)

    # Save trades
    if not trades.empty:
        trades.to_csv(out_dir / "trades.csv", index=False)

    # Save exposure
    exp_df = exposure.reset_index()
    exp_df.columns = ["date", "gross_exposure"]
    exp_df.to_csv(out_dir / "exposure.csv", index=False)

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (v3.2)")
    print("=" * 60)
    print(f"  Total Return: {metrics.get('total_return', 0)*100:.1f}%")
    print(f"  CAGR: {metrics.get('cagr', 0)*100:.1f}%")
    print(f"  Sharpe: {metrics.get('sharpe', 0):.2f}")
    print(f"  Sortino: {metrics.get('sortino', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Calmar: {metrics.get('calmar', 0):.2f}")
    print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  Trades: {metrics.get('n_trades', 0)}")
    print(f"  Avg Exposure: {metrics.get('exposure_avg', 0)*100:.1f}%")
    print(f"  Max Leverage: {metrics.get('max_leverage', 0):.2f}x")
    print(f"  Stop-Loss Triggered: {metrics.get('stop_loss_triggered', 0)}")
    print(f"  Margin Interest: ${metrics.get('total_margin_interest', 0):.2f}")
    print(f"  Regime Trades: {stats['regime_trades']}")
    print(f"\nOutput saved to: {out_dir}")

    # Check constraints
    print("\n" + "=" * 60)
    print("CONSTRAINT CHECK")
    print("=" * 60)
    mdd = metrics.get("max_drawdown", -1)
    sharpe = metrics.get("sharpe", 0)
    max_lev = metrics.get("max_leverage", 0)

    mdd_pass = mdd >= -0.20
    sharpe_pass = sharpe >= 1.0
    lev_pass = max_lev <= 2.0

    print(f"  MDD >= -20%: {mdd*100:.1f}% {'PASS' if mdd_pass else 'FAIL'}")
    print(f"  Sharpe >= 1.0: {sharpe:.2f} {'PASS' if sharpe_pass else 'FAIL'}")
    print(f"  Max Leverage <= 2.0x: {max_lev:.2f}x {'PASS' if lev_pass else 'FAIL'}")

    if mdd_pass and sharpe_pass and lev_pass:
        print("\n  ALL CONSTRAINTS PASSED")
    else:
        print("\n  SOME CONSTRAINTS FAILED")


if __name__ == "__main__":
    main()
