#!/usr/bin/env python3
"""
Generate yearly performance comparison for key configurations:
1. Base (no mechanism)
2. Freeze 1-day
3. Pullback 3%
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

TRADING_DAYS = 252


def compute_yearly_metrics(nav: pd.Series) -> pd.DataFrame:
    """Compute yearly performance metrics."""
    nav = nav.copy()
    nav.index = pd.to_datetime(nav.index)
    nav = nav.sort_index()

    years = nav.index.year.unique()
    rows = []

    for year in sorted(years):
        year_nav = nav[nav.index.year == year]
        if len(year_nav) < 2:
            continue

        start_nav = year_nav.iloc[0]
        end_nav = year_nav.iloc[-1]
        year_return = end_nav / start_nav - 1.0
        n_days = len(year_nav)

        # Annualized return
        arr = (1 + year_return) ** (TRADING_DAYS / n_days) - 1.0 if n_days > 0 else np.nan

        # Volatility
        daily_ret = year_nav.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(TRADING_DAYS) if len(daily_ret) > 1 else np.nan

        # Sharpe
        sharpe = arr / vol if vol > 0 else np.nan

        # Max DD
        peak = year_nav.cummax()
        dd = year_nav / peak - 1.0
        mdd = dd.min()

        rows.append({
            "year": year,
            "total_return": year_return,
            "arr": arr,
            "volatility": vol,
            "sharpe": sharpe,
            "max_dd": mdd,
            "trading_days": n_days,
        })

    return pd.DataFrame(rows)


def run_config(name: str, signals: pd.DataFrame, price_provider, vix_data, spy_data, **kwargs) -> pd.Series:
    """Run a single configuration and return NAV."""
    config = BacktestConfigV32(
        initial_cash=100000.0,
        cap_entries_per_quarter=24,
        max_concurrent_positions=24,
        sizing_positions=12,
        per_trade_cap=0.20,
        target_gross_normal=2.0,
        target_gross_riskoff=1.0,
        target_gross_stress=0.0,
        vix_normal_threshold=22.0,
        vix_stress_threshold=28.0,
        satellite_floor=5.0,
        commission_bps=10.0,
        slippage_bps=10.0,
        annual_borrow_rate=0.06,
        entry_lag_sessions=1,
        holding_sessions=30,
        sort_column="earnings_day_return",
        **kwargs
    )

    nav, trades, exposure, stats = run_backtest_v32(
        signals=signals,
        price_provider=price_provider,
        config=config,
        vix_data=vix_data,
        spy_data=spy_data,
    )

    return nav, stats


def main():
    # Load data
    signals_path = Path(__file__).parent.parent / "long_only_signals_2017_2025_final.csv"
    signals = pd.read_csv(signals_path)
    if "trade_long" not in signals.columns:
        signals["trade_long"] = True
    signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])
    # Use all available signals (2017-01 to 2025-08)
    print(f"Date range: {signals['reaction_date'].min().date()} to {signals['reaction_date'].max().date()}")

    vix_path = Path(__file__).parent / "vix_data.csv"
    vix_df = pd.read_csv(vix_path)
    vix_df["date"] = pd.to_datetime(vix_df["date"])
    vix_data = vix_df.set_index("date")["close"]

    print("Downloading SPY data...")
    spy = yf.download("SPY", start="2016-01-01", end="2025-01-01", progress=False)
    spy_close = spy["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_data = spy_close
    spy_data.index = pd.to_datetime(spy_data.index).normalize()

    prices_path = Path(__file__).parent.parent / "prices"
    price_provider = CSVPriceProvider(folder=prices_path)

    # Run configurations
    configs = {
        "Base": {},
        "Freeze_1day": {
            "breaker_spy_threshold": 0.04,
            "breaker_vix_threshold": 0.30,
            "breaker_cooldown_days": 1,
            "breaker_mode": "freeze",
        },
        "Pullback_3pct": {
            "addon_enabled": True,
            "addon_mode": "pullback",
            "addon_trigger_pct": 0.06,
            "addon_pullback_pct": 0.03,
            "addon_mult": 0.33,
        },
    }

    all_yearly = {}
    all_overall = {}

    for name, kwargs in configs.items():
        print(f"\nRunning {name}...")
        nav, stats = run_config(name, signals, price_provider, vix_data, spy_data, **kwargs)

        yearly = compute_yearly_metrics(nav)
        yearly["config"] = name
        all_yearly[name] = yearly

        # Overall metrics
        total_ret = nav.iloc[-1] / nav.iloc[0] - 1
        n_days = len(nav)
        cagr = (1 + total_ret) ** (TRADING_DAYS / n_days) - 1
        daily_ret = nav.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
        sharpe = cagr / vol if vol > 0 else 0
        peak = nav.cummax()
        mdd = (nav / peak - 1).min()

        all_overall[name] = {
            "total_return": total_ret,
            "terminal_mult": nav.iloc[-1] / nav.iloc[0],
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "max_dd": mdd,
            "breaker_count": stats.get("breaker_triggered_count", 0),
            "addon_count": stats.get("addon_count", 0),
        }

    # Print results
    def pct(x): return f"{x*100:.2f}%"
    def num(x): return f"{x:.2f}"

    print("\n" + "=" * 150)
    print("YEARLY PERFORMANCE COMPARISON")
    print("=" * 150)

    for config_name in configs.keys():
        yearly = all_yearly[config_name]
        overall = all_overall[config_name]

        print(f"\n### {config_name}")
        print(f"Overall: {pct(overall['total_return'])} ({overall['terminal_mult']:.2f}x), "
              f"CAGR: {pct(overall['cagr'])}, Sharpe: {num(overall['sharpe'])}, "
              f"MDD: {pct(overall['max_dd'])}, Breakers: {overall['breaker_count']}, Add-ons: {overall['addon_count']}")
        print()
        print(f"{'Year':<6} {'Return':>10} {'ARR':>10} {'Vol':>10} {'Sharpe':>8} {'MDD':>10} {'Days':>6}")
        print("-" * 70)

        for _, row in yearly.iterrows():
            year_label = f"{int(row['year'])}"
            if row['trading_days'] < 250:
                year_label += "*"
            print(f"{year_label:<6} {pct(row['total_return']):>10} {pct(row['arr']):>10} "
                  f"{pct(row['volatility']):>10} {num(row['sharpe']):>8} {pct(row['max_dd']):>10} {int(row['trading_days']):>6}")

    # Side-by-side comparison table
    print("\n" + "=" * 150)
    print("SIDE-BY-SIDE YEARLY COMPARISON")
    print("=" * 150)

    years = sorted(all_yearly["Base"]["year"].unique())

    print(f"\n{'Year':<6} | {'Base Return':>12} {'Base MDD':>10} | {'Freeze Return':>14} {'Freeze MDD':>12} | {'Pullback Return':>16} {'Pullback MDD':>14}")
    print("-" * 130)

    for year in years:
        base_row = all_yearly["Base"][all_yearly["Base"]["year"] == year].iloc[0]
        freeze_row = all_yearly["Freeze_1day"][all_yearly["Freeze_1day"]["year"] == year].iloc[0]
        pullback_row = all_yearly["Pullback_3pct"][all_yearly["Pullback_3pct"]["year"] == year].iloc[0]

        print(f"{int(year):<6} | {pct(base_row['total_return']):>12} {pct(base_row['max_dd']):>10} | "
              f"{pct(freeze_row['total_return']):>14} {pct(freeze_row['max_dd']):>12} | "
              f"{pct(pullback_row['total_return']):>16} {pct(pullback_row['max_dd']):>14}")

    # Overall row
    print("-" * 130)
    print(f"{'Overall':<6} | {pct(all_overall['Base']['total_return']):>12} {pct(all_overall['Base']['max_dd']):>10} | "
          f"{pct(all_overall['Freeze_1day']['total_return']):>14} {pct(all_overall['Freeze_1day']['max_dd']):>12} | "
          f"{pct(all_overall['Pullback_3pct']['total_return']):>16} {pct(all_overall['Pullback_3pct']['max_dd']):>14}")

    # Save to markdown
    md_lines = ["# Yearly Performance Comparison\n"]
    md_lines.append("## Configuration Summary\n")
    md_lines.append("| Config | Total Return | Terminal | CAGR | Vol | Sharpe | MDD | Breakers | Add-ons |")
    md_lines.append("|--------|-------------|----------|------|-----|--------|-----|----------|---------|")
    for name, o in all_overall.items():
        md_lines.append(f"| {name} | {pct(o['total_return'])} | {o['terminal_mult']:.2f}x | {pct(o['cagr'])} | {pct(o['volatility'])} | {num(o['sharpe'])} | {pct(o['max_dd'])} | {o['breaker_count']} | {o['addon_count']} |")

    md_lines.append("\n## Yearly Breakdown\n")

    for config_name in configs.keys():
        yearly = all_yearly[config_name]
        md_lines.append(f"\n### {config_name}\n")
        md_lines.append("| Year | Return | ARR | Vol | Sharpe | MDD | Days |")
        md_lines.append("|------|--------|-----|-----|--------|-----|------|")
        for _, row in yearly.iterrows():
            md_lines.append(f"| {int(row['year'])} | {pct(row['total_return'])} | {pct(row['arr'])} | {pct(row['volatility'])} | {num(row['sharpe'])} | {pct(row['max_dd'])} | {int(row['trading_days'])} |")

    md_lines.append("\n## Side-by-Side Comparison\n")
    md_lines.append("| Year | Base Return | Base MDD | Freeze Return | Freeze MDD | Pullback Return | Pullback MDD |")
    md_lines.append("|------|-------------|----------|---------------|------------|-----------------|--------------|")
    for year in years:
        base_row = all_yearly["Base"][all_yearly["Base"]["year"] == year].iloc[0]
        freeze_row = all_yearly["Freeze_1day"][all_yearly["Freeze_1day"]["year"] == year].iloc[0]
        pullback_row = all_yearly["Pullback_3pct"][all_yearly["Pullback_3pct"]["year"] == year].iloc[0]
        md_lines.append(f"| {int(year)} | {pct(base_row['total_return'])} | {pct(base_row['max_dd'])} | {pct(freeze_row['total_return'])} | {pct(freeze_row['max_dd'])} | {pct(pullback_row['total_return'])} | {pct(pullback_row['max_dd'])} |")
    md_lines.append(f"| **Overall** | **{pct(all_overall['Base']['total_return'])}** | **{pct(all_overall['Base']['max_dd'])}** | **{pct(all_overall['Freeze_1day']['total_return'])}** | **{pct(all_overall['Freeze_1day']['max_dd'])}** | **{pct(all_overall['Pullback_3pct']['total_return'])}** | **{pct(all_overall['Pullback_3pct']['max_dd'])}** |")

    output_path = Path(__file__).parent / "yearly_comparison.md"
    output_path.write_text("\n".join(md_lines))
    print(f"\n\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
