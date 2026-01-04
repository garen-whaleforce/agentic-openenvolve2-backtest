#!/usr/bin/env python3
"""
run_optimization_grid.py

Run 12-combo optimization grid to maximize Total Return:
- D6 regime dynamic: 3 options (A/B/C)
- Core score weight: 2 options (off/on)
- DD de-leveraging: 2 options (off/on)

Total: 3 x 2 x 2 = 12 combinations
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


@dataclass
class OptConfig:
    """Optimization configuration preset."""
    name: str
    # D6 regime dynamic weights
    d6_weight_normal: float
    d6_weight_riskoff: float
    d6_weight_stress: float
    # Score weight
    score_weight_k: Optional[float]
    # DD de-leveraging
    dd_delever_on: bool


def get_optimization_configs() -> List[OptConfig]:
    """Generate 12 optimization configurations."""
    configs = []

    # D6 regime dynamic options
    d6_options = {
        "A_Return": (1.0, 0.5, 0.0),  # Return-seeking: full in NORMAL, half in RISK_OFF
        "B_Balanced": (1.0, 0.33, 0.0),  # Balanced+: full in NORMAL, third in RISK_OFF
        "C_Defensive": (0.5, 0.0, 0.0),  # Defensive: half in NORMAL, none otherwise
    }

    # Score weight options
    score_options = {
        "EqWt": None,  # Equal weight (no score adjustment)
        "ScoreWt": 0.3,  # Score-weighted (±30% adjustment based on zscore)
    }

    # DD de-leveraging options
    dd_options = {
        "NoDD": False,
        "DDLev": True,
    }

    for d6_name, (d6_n, d6_r, d6_s) in d6_options.items():
        for score_name, score_k in score_options.items():
            for dd_name, dd_on in dd_options.items():
                name = f"{d6_name}_{score_name}_{dd_name}"
                configs.append(OptConfig(
                    name=name,
                    d6_weight_normal=d6_n,
                    d6_weight_riskoff=d6_r,
                    d6_weight_stress=d6_s,
                    score_weight_k=score_k,
                    dd_delever_on=dd_on,
                ))

    return configs


def build_backtest_config(opt: OptConfig) -> BacktestConfigV32:
    """Build BacktestConfigV32 from optimization config."""
    config = BacktestConfigV32(
        # Fixed parameters (from Balanced baseline)
        initial_cash=100000.0,
        cap_entries_per_quarter=24,
        max_concurrent_positions=24,
        sizing_positions=12,  # Core sizing denominator
        per_trade_cap=0.20,

        # Leverage by regime (200% leverage in NORMAL, matching baseline)
        target_gross_normal=2.0,
        target_gross_riskoff=1.0,
        target_gross_stress=0.0,
        vix_normal_threshold=22.0,
        vix_stress_threshold=28.0,

        # Satellite floor for D6
        satellite_floor=5.0,

        # Costs (realistic)
        commission_bps=10.0,
        slippage_bps=10.0,
        annual_borrow_rate=0.06,

        # Timing
        entry_lag_sessions=1,
        holding_sessions=30,

        # Sorting for score-weight
        sort_column="earnings_day_return",

        # D6 regime dynamic weights
        d6_weight_normal=opt.d6_weight_normal,
        d6_weight_riskoff=opt.d6_weight_riskoff,
        d6_weight_stress=opt.d6_weight_stress,

        # Score-weighted allocation
        score_weight_k=opt.score_weight_k,
        score_weight_min=0.7,
        score_weight_max=1.5,

        # DD de-leveraging (if enabled) - more relaxed thresholds for leveraged strategy
        dd_delever_threshold1=0.10 if opt.dd_delever_on else None,  # Start at -10%
        dd_delever_mult1=0.7,
        dd_delever_threshold2=0.15 if opt.dd_delever_on else None,  # Reduce more at -15%
        dd_delever_mult2=0.4,
        dd_delever_threshold3=0.20 if opt.dd_delever_on else None,  # Stop new entries at -20%
        dd_delever_mult3=0.0,
    )
    return config


def compute_metrics(nav: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
    """Compute performance metrics."""
    nav0 = nav.iloc[0]
    nav1 = nav.iloc[-1]
    total_ret = nav1 / nav0 - 1

    n_days = len(nav)
    cagr = (1 + total_ret) ** (252 / n_days) - 1

    daily_ret = nav.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0

    peak = nav.cummax()
    dd = nav / peak - 1
    mdd = dd.min()

    win_rate = (trades["net_ret"] > 0).mean() if len(trades) > 0 else 0
    avg_trade = trades["net_ret"].mean() if len(trades) > 0 else 0

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_trade": avg_trade,
    }


def main():
    # Load signals - use the full signals file with D7_CORE + D6_STRICT
    signals_path = Path(__file__).parent.parent / "long_only_signals_2017_2025_final.csv"

    if not signals_path.exists():
        # Fallback to alternative
        signals_path = Path(__file__).parent.parent / "signals_backtest_ready.csv"

    if not signals_path.exists():
        print(f"Error: Signals file not found at {signals_path}")
        sys.exit(1)

    print(f"Loading signals from: {signals_path}")
    signals = pd.read_csv(signals_path)
    print(f"Loaded {len(signals)} signals")

    # Add trade_long column if missing (all signals in this file are long trades)
    if "trade_long" not in signals.columns:
        signals["trade_long"] = True
        print("Added trade_long=True for all signals")

    # Filter to dates with available price data (up to 2024-10)
    if "reaction_date" in signals.columns:
        signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])
        signals = signals[signals["reaction_date"] <= "2024-10-31"]
        print(f"Filtered to signals up to 2024-10: {len(signals)} signals")

    # Show tier distribution
    if "trade_long_tier" in signals.columns:
        print(f"Tier distribution: {signals['trade_long_tier'].value_counts().to_dict()}")

    # Check if earnings_day_return exists
    if "earnings_day_return" not in signals.columns:
        print("Warning: earnings_day_return column not found, score-weighted allocation may not work")

    # Load VIX data
    vix_path = Path(__file__).parent.parent / "vix_daily.csv"
    vix_data = None
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path)
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix_data = vix_df.set_index("date")["close"]
        print(f"Loaded VIX data: {len(vix_data)} days")

    # Initialize price provider
    prices_path = Path(__file__).parent.parent / "prices"
    if not prices_path.exists():
        print(f"Error: Prices folder not found at {prices_path}")
        sys.exit(1)
    price_provider = CSVPriceProvider(folder=prices_path)

    # Get optimization configs
    opt_configs = get_optimization_configs()
    print(f"\nRunning {len(opt_configs)} optimization configurations...\n")

    results = []

    for i, opt in enumerate(opt_configs, 1):
        print(f"[{i}/{len(opt_configs)}] Running: {opt.name}")

        config = build_backtest_config(opt)

        try:
            nav, trades, exposure, stats = run_backtest_v32(
                signals=signals,
                price_provider=price_provider,
                config=config,
                vix_data=vix_data,
            )

            metrics = compute_metrics(nav, trades)
            metrics["name"] = opt.name
            metrics["d6_regime"] = f"N={opt.d6_weight_normal}/R={opt.d6_weight_riskoff}/S={opt.d6_weight_stress}"
            metrics["score_weight"] = "On" if opt.score_weight_k else "Off"
            metrics["dd_delever"] = "On" if opt.dd_delever_on else "Off"

            results.append(metrics)

            print(f"    Total Return: {metrics['total_return']*100:.1f}%, "
                  f"CAGR: {metrics['cagr']*100:.1f}%, "
                  f"Sharpe: {metrics['sharpe']:.2f}, "
                  f"MDD: {metrics['max_dd']*100:.1f}%")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("total_return", ascending=False)

    # Save results
    output_path = Path(__file__).parent / "optimization_grid_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("OPTIMIZATION GRID RESULTS (sorted by Total Return)")
    print("=" * 100)
    print()

    print(f"{'Rank':<4} {'Config':<30} {'TotalRet':>10} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8}")
    print("-" * 100)

    for i, row in results_df.iterrows():
        rank = list(results_df.index).index(i) + 1
        print(f"{rank:<4} {row['name']:<30} "
              f"{row['total_return']*100:>9.1f}% "
              f"{row['cagr']*100:>7.1f}% "
              f"{row['sharpe']:>8.2f} "
              f"{row['max_dd']*100:>7.1f}% "
              f"{row['win_rate']*100:>7.1f}%")

    print()

    # Best configurations
    print("=" * 100)
    print("TOP CONFIGURATIONS")
    print("=" * 100)

    best_return = results_df.iloc[0]
    print(f"\nBest Total Return: {best_return['name']}")
    print(f"  Total Return: {best_return['total_return']*100:.1f}%")
    print(f"  CAGR: {best_return['cagr']*100:.1f}%")
    print(f"  Sharpe: {best_return['sharpe']:.2f}")
    print(f"  MDD: {best_return['max_dd']*100:.1f}%")

    best_sharpe = results_df.loc[results_df["sharpe"].idxmax()]
    if best_sharpe["name"] != best_return["name"]:
        print(f"\nBest Sharpe: {best_sharpe['name']}")
        print(f"  Total Return: {best_sharpe['total_return']*100:.1f}%")
        print(f"  CAGR: {best_sharpe['cagr']*100:.1f}%")
        print(f"  Sharpe: {best_sharpe['sharpe']:.2f}")
        print(f"  MDD: {best_sharpe['max_dd']*100:.1f}%")

    best_mdd = results_df.loc[results_df["max_dd"].idxmax()]  # Less negative = better
    if best_mdd["name"] not in [best_return["name"], best_sharpe["name"]]:
        print(f"\nBest MDD: {best_mdd['name']}")
        print(f"  Total Return: {best_mdd['total_return']*100:.1f}%")
        print(f"  CAGR: {best_mdd['cagr']*100:.1f}%")
        print(f"  Sharpe: {best_mdd['sharpe']:.2f}")
        print(f"  MDD: {best_mdd['max_dd']*100:.1f}%")


if __name__ == "__main__":
    main()
