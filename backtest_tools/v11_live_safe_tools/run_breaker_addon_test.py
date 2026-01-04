#!/usr/bin/env python3
"""
run_breaker_addon_test.py

Run 6-combo test matrix for Portfolio-level Breaker and Winner Add-on:

1. Base (no breaker, no addon)
2. Breaker-1.0 (breaker target gross = 1.0)
3. Breaker-0.5 (breaker target gross = 0.5)
4. Breaker-1.0 + GrossUp (breaker 1.0, restore to 2.0 after cooldown)
5. Add-on only (no breaker, add-on enabled)
6. Breaker-1.0 + Add-on (both enabled)

Breaker specs:
- Trigger: SPY -3% OR VIX +20%
- Cooldown: 3 days
- No new entries during breaker

Add-on specs:
- Min hold: 5 sessions
- Trigger: +6% unrealized PnL
- Add-on: +0.33x original position
- Max 1 add-on per trade
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


@dataclass
class TestConfig:
    """Test configuration."""
    name: str
    breaker_spy_threshold: Optional[float] = None
    breaker_vix_threshold: Optional[float] = None
    breaker_target_gross: float = 1.0
    breaker_cooldown_days: int = 3
    addon_enabled: bool = False
    addon_min_hold_sessions: int = 5
    addon_trigger_pct: float = 0.06
    addon_mult: float = 0.33


def get_test_configs():
    """Generate test configurations.

    Original thresholds (SPY -3%, VIX +20%) triggered 52 times in 7.7 years - too frequent.
    Adjusted to more conservative thresholds:
    - SPY -4% (major single-day drop)
    - VIX +30% (significant vol spike)
    """
    return [
        TestConfig(
            name="1_Base",
        ),
        TestConfig(
            name="2_Breaker_1.0_SPY4",
            breaker_spy_threshold=0.04,  # More conservative: -4%
            breaker_vix_threshold=0.30,  # More conservative: +30%
            breaker_target_gross=1.0,
        ),
        TestConfig(
            name="3_Breaker_0.5_SPY4",
            breaker_spy_threshold=0.04,
            breaker_vix_threshold=0.30,
            breaker_target_gross=0.5,
        ),
        TestConfig(
            name="4_Breaker_1.0_SPY3",
            breaker_spy_threshold=0.03,  # Original: -3%
            breaker_vix_threshold=0.20,  # Original: +20%
            breaker_target_gross=1.0,
        ),
        TestConfig(
            name="5_Addon_Only",
            addon_enabled=True,
        ),
        TestConfig(
            name="6_Breaker_1.0_Addon_SPY4",
            breaker_spy_threshold=0.04,
            breaker_vix_threshold=0.30,
            breaker_target_gross=1.0,
            addon_enabled=True,
        ),
        # Additional tests with even more conservative thresholds
        TestConfig(
            name="7_Breaker_1.0_SPY5",
            breaker_spy_threshold=0.05,  # Very conservative: -5%
            breaker_vix_threshold=0.40,  # Very conservative: +40%
            breaker_target_gross=1.0,
        ),
        TestConfig(
            name="8_Addon_8pct",
            addon_enabled=True,
            addon_trigger_pct=0.08,  # Higher trigger: +8% unrealized
        ),
    ]


def build_backtest_config(test: TestConfig) -> BacktestConfigV32:
    """Build BacktestConfigV32 from test config."""
    return BacktestConfigV32(
        # Fixed parameters (from Return-Max baseline with 200% leverage)
        initial_cash=100000.0,
        cap_entries_per_quarter=24,
        max_concurrent_positions=24,
        sizing_positions=12,
        per_trade_cap=0.20,

        # Leverage by regime (200% in NORMAL)
        target_gross_normal=2.0,
        target_gross_riskoff=1.0,
        target_gross_stress=0.0,
        vix_normal_threshold=22.0,
        vix_stress_threshold=28.0,

        # Satellite floor
        satellite_floor=5.0,

        # Costs
        commission_bps=10.0,
        slippage_bps=10.0,
        annual_borrow_rate=0.06,

        # Timing
        entry_lag_sessions=1,
        holding_sessions=30,

        # Sorting
        sort_column="earnings_day_return",

        # Portfolio-level Breaker
        breaker_spy_threshold=test.breaker_spy_threshold,
        breaker_vix_threshold=test.breaker_vix_threshold,
        breaker_target_gross=test.breaker_target_gross,
        breaker_cooldown_days=test.breaker_cooldown_days,

        # Winner Add-on
        addon_enabled=test.addon_enabled,
        addon_min_hold_sessions=test.addon_min_hold_sessions,
        addon_trigger_pct=test.addon_trigger_pct,
        addon_mult=test.addon_mult,
    )


def compute_metrics(nav: pd.Series, trades: pd.DataFrame, stats: Dict) -> Dict:
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
        "breaker_count": stats.get("breaker_triggered_count", 0),
        "addon_count": stats.get("addon_count", 0),
    }


def download_spy_data(start_date: str = "2016-01-01", end_date: str = "2025-01-01") -> pd.Series:
    """Download SPY close prices from yfinance."""
    print("Downloading SPY data from yfinance...")
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        if spy.empty:
            print("Warning: Failed to download SPY data")
            return None
        spy_close = spy["Close"]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_close.index = pd.to_datetime(spy_close.index).normalize()
        print(f"Downloaded SPY data: {len(spy_close)} days")
        return spy_close
    except Exception as e:
        print(f"Warning: Failed to download SPY: {e}")
        return None


def main():
    # Load signals
    signals_path = Path(__file__).parent.parent / "long_only_signals_2017_2025_final.csv"
    if not signals_path.exists():
        signals_path = Path(__file__).parent.parent / "signals_backtest_ready.csv"
    if not signals_path.exists():
        print(f"Error: Signals file not found")
        sys.exit(1)

    print(f"Loading signals from: {signals_path}")
    signals = pd.read_csv(signals_path)
    print(f"Loaded {len(signals)} signals")

    if "trade_long" not in signals.columns:
        signals["trade_long"] = True

    if "reaction_date" in signals.columns:
        signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])
        signals = signals[signals["reaction_date"] <= "2024-10-31"]
        print(f"Filtered to signals up to 2024-10: {len(signals)} signals")

    # Load VIX data
    vix_path = Path(__file__).parent / "vix_data.csv"
    vix_data = None
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path)
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix_data = vix_df.set_index("date")["close"]
        print(f"Loaded VIX data: {len(vix_data)} days")
    else:
        print("Warning: VIX data not found, using default VIX=15")

    # Download SPY data
    spy_data = download_spy_data()

    # Initialize price provider
    prices_path = Path(__file__).parent.parent / "prices"
    if not prices_path.exists():
        print(f"Error: Prices folder not found at {prices_path}")
        sys.exit(1)
    price_provider = CSVPriceProvider(folder=prices_path)

    # Get test configs
    test_configs = get_test_configs()
    print(f"\nRunning {len(test_configs)} test configurations...\n")

    results = []

    for i, test in enumerate(test_configs, 1):
        print(f"[{i}/{len(test_configs)}] Running: {test.name}")

        config = build_backtest_config(test)

        try:
            nav, trades, exposure, stats = run_backtest_v32(
                signals=signals,
                price_provider=price_provider,
                config=config,
                vix_data=vix_data,
                spy_data=spy_data,
            )

            metrics = compute_metrics(nav, trades, stats)
            metrics["name"] = test.name
            metrics["breaker_dates"] = stats.get("breaker_triggered_dates", [])

            results.append(metrics)

            print(f"    Total Return: {metrics['total_return']*100:.1f}%, "
                  f"CAGR: {metrics['cagr']*100:.1f}%, "
                  f"Sharpe: {metrics['sharpe']:.2f}, "
                  f"MDD: {metrics['max_dd']*100:.1f}%, "
                  f"Breakers: {metrics['breaker_count']}, "
                  f"Add-ons: {metrics['addon_count']}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_path = Path(__file__).parent / "breaker_addon_test_results.csv"
    # Drop list columns for CSV
    csv_df = results_df.drop(columns=["breaker_dates"], errors="ignore")
    csv_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 120)
    print("BREAKER + ADD-ON TEST RESULTS")
    print("=" * 120)
    print()

    print(f"{'Name':<25} {'TotalRet':>10} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Trades':>7} {'WinRate':>8} {'Breakers':>9} {'Add-ons':>8}")
    print("-" * 120)

    for _, row in results_df.iterrows():
        print(f"{row['name']:<25} "
              f"{row['total_return']*100:>9.1f}% "
              f"{row['cagr']*100:>7.1f}% "
              f"{row['sharpe']:>8.2f} "
              f"{row['max_dd']*100:>7.1f}% "
              f"{row['n_trades']:>7.0f} "
              f"{row['win_rate']*100:>7.1f}% "
              f"{row['breaker_count']:>9.0f} "
              f"{row['addon_count']:>8.0f}")

    print()

    # Analysis
    print("=" * 120)
    print("ANALYSIS")
    print("=" * 120)

    base = results_df[results_df["name"] == "1_Base"].iloc[0] if len(results_df[results_df["name"] == "1_Base"]) > 0 else None

    if base is not None:
        print(f"\nBase Performance: {base['total_return']*100:.1f}% Total Return, {base['max_dd']*100:.1f}% MDD")
        print()

        for _, row in results_df.iterrows():
            if row["name"] == "1_Base":
                continue
            ret_delta = (row["total_return"] - base["total_return"]) * 100
            mdd_delta = (row["max_dd"] - base["max_dd"]) * 100
            print(f"{row['name']}:")
            print(f"  Return delta: {ret_delta:+.1f}% vs Base")
            print(f"  MDD delta: {mdd_delta:+.1f}% vs Base")
            if row["breaker_count"] > 0:
                print(f"  Breaker triggered: {row['breaker_count']} times")
                breaker_dates = row.get("breaker_dates", [])
                if breaker_dates:
                    print(f"  Breaker dates: {', '.join(breaker_dates[:5])}{'...' if len(breaker_dates) > 5 else ''}")
            if row["addon_count"] > 0:
                print(f"  Add-ons executed: {row['addon_count']} times")
            print()


if __name__ == "__main__":
    main()
