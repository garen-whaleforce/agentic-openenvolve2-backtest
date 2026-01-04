#!/usr/bin/env python3
"""
run_improved_breaker_addon_test.py

Test improved Breaker and Add-on variants:

Breaker improvements:
- "freeze" mode: Only block new entries, don't force-sell existing positions
- This avoids "selling at panic lows" problem

Add-on improvements:
- "pullback" mode: Only add on pullback from high, not at extension
- This avoids "buying at local tops" problem

Test matrix (2x2 + extras):
1. Base (no mechanism)
2. Freeze Only (cooldown=1)
3. Freeze Only (cooldown=3)
4. Pullback Add-on (pullback=2%)
5. Pullback Add-on (pullback=3%)
6. Freeze(1) + Pullback(3%)
7. Original Breaker reduce for comparison
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


@dataclass
class TestConfig:
    """Test configuration."""
    name: str
    # Breaker settings
    breaker_spy_threshold: Optional[float] = None
    breaker_vix_threshold: Optional[float] = None
    breaker_target_gross: float = 1.0
    breaker_cooldown_days: int = 3
    breaker_mode: str = "reduce"  # "reduce" or "freeze"
    # Add-on settings
    addon_enabled: bool = False
    addon_min_hold_sessions: int = 5
    addon_trigger_pct: float = 0.06
    addon_mult: float = 0.33
    addon_mode: str = "extension"  # "extension" or "pullback"
    addon_pullback_pct: float = 0.03


def get_test_configs():
    """Generate test configurations for improved variants."""
    return [
        # 1. Base
        TestConfig(name="1_Base"),

        # 2-3. Freeze Only (no force-sell, only block new entries)
        TestConfig(
            name="2_Freeze_1day",
            breaker_spy_threshold=0.04,
            breaker_vix_threshold=0.30,
            breaker_cooldown_days=1,
            breaker_mode="freeze",
        ),
        TestConfig(
            name="3_Freeze_3day",
            breaker_spy_threshold=0.04,
            breaker_vix_threshold=0.30,
            breaker_cooldown_days=3,
            breaker_mode="freeze",
        ),

        # 4-5. Pullback Add-on (add on pullback, not extension)
        TestConfig(
            name="4_Pullback_2pct",
            addon_enabled=True,
            addon_mode="pullback",
            addon_pullback_pct=0.02,
        ),
        TestConfig(
            name="5_Pullback_3pct",
            addon_enabled=True,
            addon_mode="pullback",
            addon_pullback_pct=0.03,
        ),

        # 6. Combined: Freeze + Pullback
        TestConfig(
            name="6_Freeze1_Pullback3",
            breaker_spy_threshold=0.04,
            breaker_vix_threshold=0.30,
            breaker_cooldown_days=1,
            breaker_mode="freeze",
            addon_enabled=True,
            addon_mode="pullback",
            addon_pullback_pct=0.03,
        ),

        # 7. Original reduce breaker for comparison
        TestConfig(
            name="7_Reduce_3day",
            breaker_spy_threshold=0.04,
            breaker_vix_threshold=0.30,
            breaker_cooldown_days=3,
            breaker_mode="reduce",
        ),

        # 8. Original extension add-on for comparison
        TestConfig(
            name="8_Extension_6pct",
            addon_enabled=True,
            addon_mode="extension",
            addon_trigger_pct=0.06,
        ),
    ]


def build_backtest_config(test: TestConfig) -> BacktestConfigV32:
    """Build BacktestConfigV32 from test config."""
    return BacktestConfigV32(
        # Fixed parameters
        initial_cash=100000.0,
        cap_entries_per_quarter=24,
        max_concurrent_positions=24,
        sizing_positions=12,
        per_trade_cap=0.20,

        # Leverage
        target_gross_normal=2.0,
        target_gross_riskoff=1.0,
        target_gross_stress=0.0,
        vix_normal_threshold=22.0,
        vix_stress_threshold=28.0,

        satellite_floor=5.0,

        # Costs
        commission_bps=10.0,
        slippage_bps=10.0,
        annual_borrow_rate=0.06,

        entry_lag_sessions=1,
        holding_sessions=30,
        sort_column="earnings_day_return",

        # Breaker
        breaker_spy_threshold=test.breaker_spy_threshold,
        breaker_vix_threshold=test.breaker_vix_threshold,
        breaker_target_gross=test.breaker_target_gross,
        breaker_cooldown_days=test.breaker_cooldown_days,
        breaker_mode=test.breaker_mode,

        # Add-on
        addon_enabled=test.addon_enabled,
        addon_min_hold_sessions=test.addon_min_hold_sessions,
        addon_trigger_pct=test.addon_trigger_pct,
        addon_mult=test.addon_mult,
        addon_mode=test.addon_mode,
        addon_pullback_pct=test.addon_pullback_pct,
    )


def compute_metrics(nav: pd.Series, trades: pd.DataFrame, stats: Dict) -> Dict:
    """Compute performance metrics."""
    nav0 = nav.iloc[0]
    nav1 = nav.iloc[-1]
    total_ret = nav1 / nav0 - 1
    terminal_mult = nav1 / nav0  # Terminal value multiplier

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
        "terminal_mult": terminal_mult,
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
    """Download SPY close prices."""
    print("Downloading SPY data...")
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        if spy.empty:
            return None
        spy_close = spy["Close"]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_close.index = pd.to_datetime(spy_close.index).normalize()
        print(f"Downloaded SPY: {len(spy_close)} days")
        return spy_close
    except Exception as e:
        print(f"Warning: SPY download failed: {e}")
        return None


def main():
    # Load signals
    signals_path = Path(__file__).parent.parent / "long_only_signals_2017_2025_final.csv"
    if not signals_path.exists():
        signals_path = Path(__file__).parent.parent / "signals_backtest_ready.csv"
    if not signals_path.exists():
        print("Error: Signals file not found")
        sys.exit(1)

    print(f"Loading signals: {signals_path}")
    signals = pd.read_csv(signals_path)
    print(f"Loaded {len(signals)} signals")

    if "trade_long" not in signals.columns:
        signals["trade_long"] = True

    if "reaction_date" in signals.columns:
        signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])
        # Use all available signals (2017-01 to 2025-08)
        print(f"Date range: {signals['reaction_date'].min().date()} to {signals['reaction_date'].max().date()}")

    # Load VIX
    vix_path = Path(__file__).parent / "vix_data.csv"
    vix_data = None
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path)
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix_data = vix_df.set_index("date")["close"]
        print(f"Loaded VIX: {len(vix_data)} days")

    # Load SPY
    spy_data = download_spy_data()

    # Price provider
    prices_path = Path(__file__).parent.parent / "prices"
    if not prices_path.exists():
        print(f"Error: Prices not found")
        sys.exit(1)
    price_provider = CSVPriceProvider(folder=prices_path)

    # Run tests
    test_configs = get_test_configs()
    print(f"\nRunning {len(test_configs)} configurations...\n")

    results = []

    for i, test in enumerate(test_configs, 1):
        print(f"[{i}/{len(test_configs)}] {test.name}")

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
            metrics["breaker_mode"] = test.breaker_mode
            metrics["addon_mode"] = test.addon_mode if test.addon_enabled else "N/A"

            results.append(metrics)

            print(f"    Return: {metrics['total_return']*100:.1f}% ({metrics['terminal_mult']:.2f}x), "
                  f"MDD: {metrics['max_dd']*100:.1f}%, "
                  f"Sharpe: {metrics['sharpe']:.2f}, "
                  f"Breakers: {metrics['breaker_count']}, "
                  f"Add-ons: {metrics['addon_count']}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Results
    results_df = pd.DataFrame(results)

    output_path = Path(__file__).parent / "improved_breaker_addon_results.csv"
    csv_df = results_df.drop(columns=["breaker_dates"], errors="ignore")
    csv_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 140)
    print("IMPROVED BREAKER + ADD-ON TEST RESULTS")
    print("=" * 140)

    print(f"\n{'Name':<25} {'Return':>10} {'Terminal':>9} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'BreakerMode':>12} {'AddonMode':>12} {'Brk#':>5} {'Add#':>5}")
    print("-" * 140)

    for _, row in results_df.iterrows():
        print(f"{row['name']:<25} "
              f"{row['total_return']*100:>9.1f}% "
              f"{row['terminal_mult']:>8.2f}x "
              f"{row['cagr']*100:>7.1f}% "
              f"{row['sharpe']:>8.2f} "
              f"{row['max_dd']*100:>7.1f}% "
              f"{row['breaker_mode']:>12} "
              f"{row['addon_mode']:>12} "
              f"{row['breaker_count']:>5.0f} "
              f"{row['addon_count']:>5.0f}")

    # Analysis
    print("\n" + "=" * 140)
    print("COMPARISON ANALYSIS")
    print("=" * 140)

    base = results_df[results_df["name"] == "1_Base"].iloc[0]
    print(f"\nBase: {base['total_return']*100:.1f}% ({base['terminal_mult']:.2f}x), MDD: {base['max_dd']*100:.1f}%\n")

    for _, row in results_df.iterrows():
        if row["name"] == "1_Base":
            continue
        terminal_delta = (row["terminal_mult"] / base["terminal_mult"] - 1) * 100
        mdd_delta = (row["max_dd"] - base["max_dd"]) * 100
        print(f"{row['name']}:")
        print(f"  Terminal: {terminal_delta:+.1f}% vs Base, MDD: {mdd_delta:+.2f}% vs Base")

    # Key comparisons
    print("\n" + "=" * 140)
    print("KEY COMPARISONS")
    print("=" * 140)

    # Freeze vs Reduce
    freeze_3 = results_df[results_df["name"] == "3_Freeze_3day"]
    reduce_3 = results_df[results_df["name"] == "7_Reduce_3day"]
    if len(freeze_3) > 0 and len(reduce_3) > 0:
        f3 = freeze_3.iloc[0]
        r3 = reduce_3.iloc[0]
        print(f"\nFreeze vs Reduce (3-day cooldown, SPY-4%/VIX+30%):")
        print(f"  Freeze: {f3['total_return']*100:.1f}% ({f3['terminal_mult']:.2f}x), MDD: {f3['max_dd']*100:.1f}%")
        print(f"  Reduce: {r3['total_return']*100:.1f}% ({r3['terminal_mult']:.2f}x), MDD: {r3['max_dd']*100:.1f}%")
        print(f"  -> Freeze saves {(f3['terminal_mult']/r3['terminal_mult']-1)*100:+.1f}% terminal value")

    # Pullback vs Extension
    pullback = results_df[results_df["name"] == "5_Pullback_3pct"]
    extension = results_df[results_df["name"] == "8_Extension_6pct"]
    if len(pullback) > 0 and len(extension) > 0:
        p = pullback.iloc[0]
        e = extension.iloc[0]
        print(f"\nPullback vs Extension Add-on:")
        print(f"  Pullback 3%: {p['total_return']*100:.1f}% ({p['terminal_mult']:.2f}x), MDD: {p['max_dd']*100:.1f}%, Add-ons: {p['addon_count']:.0f}")
        print(f"  Extension 6%: {e['total_return']*100:.1f}% ({e['terminal_mult']:.2f}x), MDD: {e['max_dd']*100:.1f}%, Add-ons: {e['addon_count']:.0f}")


if __name__ == "__main__":
    main()
