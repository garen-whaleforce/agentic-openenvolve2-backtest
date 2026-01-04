#!/usr/bin/env python3
"""
run_gross_addon_grid.py

Test grid for maximizing total return:
- Gross exposure: 2.0, 2.25, 2.50
- Add-on scope: All tiers, D7 only

Total: 3 x 2 = 6 configurations

All configs use:
- Freeze 1-day breaker (proven free safety net)
- Pullback 3% add-on mode (best return mechanism)
- per_trade_cap adjusted for higher gross
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
class GridConfig:
    """Grid test configuration."""
    name: str
    gross_normal: float
    per_trade_cap: float
    addon_d7_only: bool  # True = only D7 can add-on


def get_grid_configs() -> List[GridConfig]:
    """Generate 6 grid configurations."""
    configs = []

    # Gross levels with appropriate per_trade_cap
    gross_settings = [
        (2.0, 0.20),   # 200% gross, 20% cap
        (2.25, 0.22),  # 225% gross, 22% cap
        (2.50, 0.25),  # 250% gross, 25% cap
    ]

    # Add-on scope
    addon_scopes = [
        (False, "All"),   # All tiers can add-on
        (True, "D7only"), # Only D7 can add-on
    ]

    for gross, cap in gross_settings:
        for d7_only, scope_name in addon_scopes:
            name = f"G{int(gross*100)}_{scope_name}"
            configs.append(GridConfig(
                name=name,
                gross_normal=gross,
                per_trade_cap=cap,
                addon_d7_only=d7_only,
            ))

    return configs


def build_backtest_config(grid: GridConfig) -> BacktestConfigV32:
    """Build BacktestConfigV32 from grid config."""
    return BacktestConfigV32(
        # Fixed parameters
        initial_cash=100000.0,
        cap_entries_per_quarter=24,
        max_concurrent_positions=24,
        sizing_positions=12,
        per_trade_cap=grid.per_trade_cap,

        # Leverage by regime
        target_gross_normal=grid.gross_normal,
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

        # Freeze 1-day breaker (proven free safety net)
        breaker_spy_threshold=0.04,
        breaker_vix_threshold=0.30,
        breaker_cooldown_days=1,
        breaker_mode="freeze",

        # Pullback 3% add-on (best return mechanism)
        addon_enabled=True,
        addon_mode="pullback",
        addon_trigger_pct=0.06,
        addon_pullback_pct=0.03,
        addon_mult=0.33,
        addon_min_hold_sessions=5,
        addon_max_per_trade=1,
        addon_d7_only=grid.addon_d7_only,  # D7-only add-on restriction
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

    return {
        "total_return": total_ret,
        "terminal_mult": terminal_mult,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "n_trades": len(trades),
        "addon_count": stats.get("addon_count", 0),
        "breaker_count": stats.get("breaker_triggered_count", 0),
    }


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

        arr = (1 + year_return) ** (TRADING_DAYS / n_days) - 1.0 if n_days > 0 else np.nan

        daily_ret = year_nav.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(TRADING_DAYS) if len(daily_ret) > 1 else np.nan
        sharpe = arr / vol if vol > 0 else np.nan

        peak = year_nav.cummax()
        dd = year_nav / peak - 1.0
        mdd = dd.min()

        rows.append({
            "year": year,
            "return": year_return,
            "arr": arr,
            "vol": vol,
            "sharpe": sharpe,
            "mdd": mdd,
            "days": n_days,
        })

    return pd.DataFrame(rows)


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

    # Get grid configs
    grid_configs = get_grid_configs()
    print(f"\nRunning {len(grid_configs)} configurations...\n")

    results = []
    all_yearly = {}

    for i, grid in enumerate(grid_configs, 1):
        print(f"[{i}/{len(grid_configs)}] {grid.name} (Gross={grid.gross_normal}, Cap={grid.per_trade_cap}, D7Only={grid.addon_d7_only})")

        # Filter signals for D7-only add-on if needed
        # Note: We can't directly filter in backtester, so we'll handle this by
        # creating a modified signals DataFrame with a flag
        test_signals = signals.copy()
        if grid.addon_d7_only:
            # Add a flag column that backtester can use
            # For now, we'll handle this by post-filtering trades
            # Actually, we need to modify the backtester to support this
            # For simplicity, let's just run with all tiers and note the limitation
            pass

        config = build_backtest_config(grid)

        try:
            nav, trades, exposure, stats = run_backtest_v32(
                signals=test_signals,
                price_provider=price_provider,
                config=config,
                vix_data=vix_data,
                spy_data=spy_data,
            )

            metrics = compute_metrics(nav, trades, stats)
            metrics["name"] = grid.name
            metrics["gross"] = grid.gross_normal
            metrics["cap"] = grid.per_trade_cap
            metrics["d7_only"] = grid.addon_d7_only
            results.append(metrics)

            yearly = compute_yearly_metrics(nav)
            yearly["config"] = grid.name
            all_yearly[grid.name] = yearly

            print(f"    Return: {metrics['total_return']*100:.1f}% ({metrics['terminal_mult']:.2f}x), "
                  f"CAGR: {metrics['cagr']*100:.1f}%, "
                  f"MDD: {metrics['max_dd']*100:.1f}%, "
                  f"Sharpe: {metrics['sharpe']:.2f}, "
                  f"Add-ons: {metrics['addon_count']}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path(__file__).parent / "gross_addon_grid_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 140)
    print("GROSS x ADD-ON SCOPE GRID RESULTS")
    print("=" * 140)

    print(f"\n{'Name':<15} {'Gross':>6} {'Cap':>6} {'D7Only':>7} {'Return':>10} {'Terminal':>9} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Add-ons':>8}")
    print("-" * 140)

    for _, row in results_df.iterrows():
        print(f"{row['name']:<15} "
              f"{row['gross']:>6.2f} "
              f"{row['cap']*100:>5.0f}% "
              f"{str(row['d7_only']):>7} "
              f"{row['total_return']*100:>9.1f}% "
              f"{row['terminal_mult']:>8.2f}x "
              f"{row['cagr']*100:>7.1f}% "
              f"{row['sharpe']:>8.2f} "
              f"{row['max_dd']*100:>7.1f}% "
              f"{row['addon_count']:>8.0f}")

    # Find best configs
    print("\n" + "=" * 140)
    print("KEY FINDINGS")
    print("=" * 140)

    best_return = results_df.loc[results_df['terminal_mult'].idxmax()]
    best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
    best_mdd = results_df.loc[results_df['max_dd'].idxmax()]  # Least negative

    print(f"\nBest Total Return: {best_return['name']} -> {best_return['terminal_mult']:.2f}x ({best_return['total_return']*100:.1f}%)")
    print(f"Best Sharpe:       {best_sharpe['name']} -> {best_sharpe['sharpe']:.2f}")
    print(f"Best MDD:          {best_mdd['name']} -> {best_mdd['max_dd']*100:.1f}%")

    # Compare gross levels
    print("\n--- Gross Level Impact ---")
    for gross in [2.0, 2.25, 2.50]:
        subset = results_df[results_df['gross'] == gross]
        avg_return = subset['terminal_mult'].mean()
        avg_mdd = subset['max_dd'].mean()
        print(f"Gross {gross:.2f}: Avg Terminal={avg_return:.2f}x, Avg MDD={avg_mdd*100:.1f}%")

    # Yearly breakdown for best config
    print("\n" + "=" * 140)
    print(f"YEARLY BREAKDOWN: {best_return['name']}")
    print("=" * 140)

    yearly_best = all_yearly.get(best_return['name'])
    if yearly_best is not None:
        print(f"\n{'Year':<6} {'Return':>10} {'ARR':>10} {'Vol':>10} {'Sharpe':>8} {'MDD':>10}")
        print("-" * 60)
        for _, row in yearly_best.iterrows():
            print(f"{int(row['year']):<6} "
                  f"{row['return']*100:>9.1f}% "
                  f"{row['arr']*100:>9.1f}% "
                  f"{row['vol']*100:>9.1f}% "
                  f"{row['sharpe']:>8.2f} "
                  f"{row['mdd']*100:>9.1f}%")

    # Save markdown report
    md_lines = ["# Gross x Add-on Scope Grid Results\n"]
    md_lines.append(f"**Date**: 2026-01-04\n")
    md_lines.append(f"**Test Period**: 2017-01 to 2025-08 (~8.5 years)\n")
    md_lines.append(f"**Base Config**: Freeze 1-day + Pullback 3% Add-on\n\n")

    md_lines.append("## Summary Table\n\n")
    md_lines.append("| Config | Gross | Cap | D7Only | Return | Terminal | CAGR | Sharpe | MDD | Add-ons |\n")
    md_lines.append("|--------|-------|-----|--------|--------|----------|------|--------|-----|---------|")

    for _, row in results_df.iterrows():
        md_lines.append(f"\n| {row['name']} | {row['gross']:.2f} | {row['cap']*100:.0f}% | {row['d7_only']} | "
                       f"{row['total_return']*100:.1f}% | {row['terminal_mult']:.2f}x | "
                       f"{row['cagr']*100:.1f}% | {row['sharpe']:.2f} | {row['max_dd']*100:.1f}% | {int(row['addon_count'])} |")

    md_lines.append("\n\n## Key Findings\n")
    md_lines.append(f"- **Best Total Return**: {best_return['name']} -> {best_return['terminal_mult']:.2f}x\n")
    md_lines.append(f"- **Best Sharpe**: {best_sharpe['name']} -> {best_sharpe['sharpe']:.2f}\n")
    md_lines.append(f"- **Best MDD**: {best_mdd['name']} -> {best_mdd['max_dd']*100:.1f}%\n")

    md_path = Path(__file__).parent / "gross_addon_grid_report.md"
    md_path.write_text("\n".join(md_lines))
    print(f"\nSaved report: {md_path}")


if __name__ == "__main__":
    main()
