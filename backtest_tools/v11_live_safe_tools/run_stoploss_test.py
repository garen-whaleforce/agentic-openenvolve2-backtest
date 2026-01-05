#!/usr/bin/env python3
"""
run_stoploss_test.py

Minimal stop-loss test based on the analysis:
1. No stop (baseline)
2. Catastrophic stop = 25%
3. Catastrophic stop = 30%
4. Add-on leg stop = 10% (only cut add-on portion)

Using G250_All as the base configuration.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider

TRADING_DAYS = 252


@dataclass
class StopLossConfig:
    """Stop-loss test configuration."""
    name: str
    stop_loss_pct: Optional[float] = None  # Catastrophic stop (e.g., 0.25 = -25%)
    addon_stop_pct: Optional[float] = None  # Only for add-on leg (e.g., 0.10 = -10%)


def get_test_configs() -> List[StopLossConfig]:
    """Generate 4 stop-loss configurations."""
    return [
        StopLossConfig(name="1_NoStop"),
        StopLossConfig(name="2_CatStop_25pct", stop_loss_pct=0.25),
        StopLossConfig(name="3_CatStop_30pct", stop_loss_pct=0.30),
        StopLossConfig(name="4_AddonStop_10pct", addon_stop_pct=0.10),
    ]


def build_backtest_config(test: StopLossConfig) -> BacktestConfigV32:
    """Build BacktestConfigV32 from test config (G250_All base)."""
    return BacktestConfigV32(
        # G250_All base configuration
        initial_cash=100000.0,
        cap_entries_per_quarter=24,
        max_concurrent_positions=24,
        sizing_positions=12,
        per_trade_cap=0.25,  # 25% for G250

        # 250% leverage
        target_gross_normal=2.50,
        target_gross_riskoff=1.0,
        target_gross_stress=0.0,
        vix_normal_threshold=22.0,
        vix_stress_threshold=28.0,

        satellite_floor=5.0,

        # Costs
        commission_bps=10.0,
        slippage_bps=10.0,
        annual_borrow_rate=0.06,

        # Timing
        entry_lag_sessions=1,
        holding_sessions=30,
        sort_column="earnings_day_return",

        # Freeze 1-day breaker
        breaker_spy_threshold=0.04,
        breaker_vix_threshold=0.30,
        breaker_cooldown_days=1,
        breaker_mode="freeze",

        # Pullback 3% add-on
        addon_enabled=True,
        addon_mode="pullback",
        addon_trigger_pct=0.06,
        addon_pullback_pct=0.03,
        addon_mult=0.33,
        addon_min_hold_sessions=5,
        addon_max_per_trade=1,
        addon_d7_only=False,

        # Stop-loss settings
        stop_loss_pct=test.stop_loss_pct,
        addon_stop_pct=test.addon_stop_pct,
    )


def compute_metrics(nav: pd.Series, trades: pd.DataFrame, stats: Dict) -> Dict:
    """Compute performance metrics."""
    nav0 = nav.iloc[0]
    nav1 = nav.iloc[-1]
    total_ret = nav1 / nav0 - 1
    terminal_mult = nav1 / nav0

    n_days = len(nav)
    cagr = (1 + total_ret) ** (TRADING_DAYS / n_days) - 1

    daily_ret = nav.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = cagr / vol if vol > 0 else 0

    peak = nav.cummax()
    dd = nav / peak - 1
    mdd = dd.min()

    # Top trades analysis
    top10_contrib = 0.0
    top20_contrib = 0.0
    if len(trades) > 0 and "net_ret" in trades.columns:
        # Sort by net_ret descending
        sorted_trades = trades.sort_values("net_ret", ascending=False)
        top10 = sorted_trades.head(10)
        top20 = sorted_trades.head(20)
        top10_contrib = top10["net_ret"].sum()
        top20_contrib = top20["net_ret"].sum()

    return {
        "total_return": total_ret,
        "terminal_mult": terminal_mult,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "n_trades": len(trades),
        "stop_loss_count": stats.get("stop_loss_triggered", 0),
        "addon_count": stats.get("addon_count", 0),
        "top10_trades_contrib": top10_contrib,
        "top20_trades_contrib": top20_contrib,
    }


def download_spy_data(start_date: str = "2016-01-01", end_date: str = "2026-01-01") -> pd.Series:
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
        print("Error: Signals file not found")
        sys.exit(1)

    print(f"Loading signals: {signals_path}")
    signals = pd.read_csv(signals_path)
    print(f"Loaded {len(signals)} signals")

    if "trade_long" not in signals.columns:
        signals["trade_long"] = True

    signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])
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
        print("Error: Prices not found")
        sys.exit(1)
    price_provider = CSVPriceProvider(folder=prices_path)

    # Run tests
    test_configs = get_test_configs()
    print(f"\nRunning {len(test_configs)} stop-loss configurations...\n")

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
            metrics["stop_loss_pct"] = test.stop_loss_pct
            metrics["addon_stop_pct"] = test.addon_stop_pct
            results.append(metrics)

            print(f"    Return: {metrics['total_return']*100:.1f}% ({metrics['terminal_mult']:.2f}x), "
                  f"MDD: {metrics['max_dd']*100:.1f}%, "
                  f"Sharpe: {metrics['sharpe']:.2f}, "
                  f"Stop triggers: {metrics['stop_loss_count']}, "
                  f"Top10 contrib: {metrics['top10_trades_contrib']*100:.1f}%")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path(__file__).parent / "stoploss_test_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 140)
    print("STOP-LOSS TEST RESULTS (G250_All Base)")
    print("=" * 140)

    baseline = results_df[results_df["name"] == "1_NoStop"].iloc[0]

    print(f"\n{'Config':<25} {'Return':>10} {'Terminal':>9} {'MDD':>8} {'Sharpe':>8} {'Stops':>7} {'Top10':>10} {'Top20':>10}")
    print("-" * 140)

    for _, row in results_df.iterrows():
        terminal_delta = (row['terminal_mult'] / baseline['terminal_mult'] - 1) * 100
        mdd_delta = (row['max_dd'] - baseline['max_dd']) * 100

        print(f"{row['name']:<25} "
              f"{row['total_return']*100:>9.1f}% "
              f"{row['terminal_mult']:>8.2f}x "
              f"{row['max_dd']*100:>7.1f}% "
              f"{row['sharpe']:>8.2f} "
              f"{row['stop_loss_count']:>7.0f} "
              f"{row['top10_trades_contrib']*100:>9.1f}% "
              f"{row['top20_trades_contrib']*100:>9.1f}%")

    # Analysis
    print("\n" + "=" * 140)
    print("IMPACT ANALYSIS vs NO-STOP BASELINE")
    print("=" * 140)

    for _, row in results_df.iterrows():
        if row["name"] == "1_NoStop":
            continue

        terminal_delta = (row['terminal_mult'] / baseline['terminal_mult'] - 1) * 100
        mdd_delta = (row['max_dd'] - baseline['max_dd']) * 100
        top10_delta = (row['top10_trades_contrib'] - baseline['top10_trades_contrib']) * 100

        print(f"\n{row['name']}:")
        print(f"  Terminal: {terminal_delta:+.1f}% vs baseline")
        print(f"  MDD:      {mdd_delta:+.2f}% vs baseline (positive = worse)")
        print(f"  Top10:    {top10_delta:+.1f}% contribution change")
        print(f"  Stop triggers: {row['stop_loss_count']:.0f} times")

        # Verdict
        if terminal_delta < -5 and mdd_delta > -2:
            print(f"  Verdict: NOT RECOMMENDED (large return drop, minimal MDD improvement)")
        elif terminal_delta < -2 and mdd_delta > -1:
            print(f"  Verdict: MARGINAL (return drops more than MDD improves)")
        elif terminal_delta >= -1 and mdd_delta < -1:
            print(f"  Verdict: POTENTIALLY USEFUL (MDD improves, return stable)")
        else:
            print(f"  Verdict: NEEDS FURTHER ANALYSIS")

    # Key findings
    print("\n" + "=" * 140)
    print("KEY FINDINGS")
    print("=" * 140)

    best_terminal = results_df.loc[results_df['terminal_mult'].idxmax()]
    best_mdd = results_df.loc[results_df['max_dd'].idxmax()]
    best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]

    print(f"\nBest Terminal:  {best_terminal['name']} -> {best_terminal['terminal_mult']:.2f}x")
    print(f"Best MDD:       {best_mdd['name']} -> {best_mdd['max_dd']*100:.1f}%")
    print(f"Best Sharpe:    {best_sharpe['name']} -> {best_sharpe['sharpe']:.2f}")

    # Recommendation
    print("\n" + "=" * 140)
    print("RECOMMENDATION")
    print("=" * 140)

    if best_terminal['name'] == "1_NoStop":
        print("\nNO STOP-LOSS is the best option for maximizing returns.")
        print("The strategy's '30-day time stop' already handles tail risk effectively.")
        print("Adding stop-loss would likely clip the right tail (big winners) more than reduce left tail.")
    else:
        print(f"\nConsider using {best_terminal['name']} if:")
        print("- MDD improvement is meaningful")
        print("- Top trades contribution is not significantly reduced")


if __name__ == "__main__":
    main()
