#!/usr/bin/env python3
"""
d7_elite_analysis.py

Analysis of D7_CORE signals to find paths to Wilson LB >= 80%

Since eps_surprise and other gates are not available in current CSV,
we focus on:
1. risk_code gate analysis
2. Year/Quarter stratification
3. Sector analysis
4. Calculating what win rate is needed for Wilson LB >= 80%
"""
import pandas as pd
import numpy as np
from scipy import stats


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> tuple:
    """Calculate Wilson score confidence interval"""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = wins / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return (max(0, center - margin), min(1, center + margin))


def required_win_rate_for_wilson_lb(target_lb: float, n: int) -> float:
    """Calculate required win rate to achieve target Wilson LB"""
    # Binary search for required win rate
    low, high = target_lb, 1.0
    while high - low > 0.0001:
        mid = (low + high) / 2
        wins = int(mid * n)
        lb, _ = wilson_ci(wins, n)
        if lb >= target_lb:
            high = mid
        else:
            low = mid
    return high


def main():
    # Load data
    df = pd.read_csv('/Users/garen.lee/Coding/agentic-openenvolve2/EarningsCallAgenticRag/validation_runs/validation_2017Q1_2025Q3_12250_20260104_001322/validation_results.csv',
                     quotechar='"', on_bad_lines='skip')

    print(f"Total samples: {len(df)}")

    # Get all long trades
    all_long = df[df['trade_long'] == True].copy()
    print(f"All long trades: {len(all_long)}")

    # D7_CORE
    d7 = all_long[all_long['trade_long_tier'] == 'D7_CORE'].copy()
    print(f"D7_CORE trades: {len(d7)}")

    # D6_STRICT
    d6 = all_long[all_long['trade_long_tier'] == 'D6_STRICT'].copy()
    print(f"D6_STRICT trades: {len(d6)}")

    # Convert actual_return to numeric
    d7['actual_return'] = pd.to_numeric(d7['actual_return'], errors='coerce')
    d6['actual_return'] = pd.to_numeric(d6['actual_return'], errors='coerce')
    all_long['actual_return'] = pd.to_numeric(all_long['actual_return'], errors='coerce')

    print("\n" + "="*80)
    print("1. OVERALL SIGNAL PERFORMANCE")
    print("="*80)

    for name, data in [("All Long", all_long), ("D7_CORE", d7), ("D6_STRICT", d6)]:
        n = len(data)
        wins = (data['actual_return'] > 0).sum()
        win_rate = wins / n if n > 0 else 0
        lb, ub = wilson_ci(wins, n)
        avg_ret = data['actual_return'].mean()
        print(f"\n{name}:")
        print(f"  Trades: {n}")
        print(f"  Win Rate: {win_rate:.1%} ({wins}/{n})")
        print(f"  Wilson 95% CI: [{lb:.1%}, {ub:.1%}]")
        print(f"  Avg Return: {avg_ret:.2f}%")

    print("\n" + "="*80)
    print("2. REQUIRED WIN RATE FOR WILSON LB >= 80%")
    print("="*80)

    for n in [400, 500, 600, 700, 800]:
        req = required_win_rate_for_wilson_lb(0.80, n)
        print(f"  n={n}: Need win rate >= {req:.1%}")

    print("\n" + "="*80)
    print("3. D7_CORE BY RISK CODE")
    print("="*80)

    for risk in ['low', 'medium', 'high']:
        subset = d7[d7['risk_code'] == risk]
        n = len(subset)
        if n == 0:
            continue
        wins = (subset['actual_return'] > 0).sum()
        win_rate = wins / n
        lb, ub = wilson_ci(wins, n)
        avg_ret = subset['actual_return'].mean()
        print(f"\n{risk.upper()} risk:")
        print(f"  Trades: {n}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Wilson 95% CI: [{lb:.1%}, {ub:.1%}]")
        print(f"  Avg Return: {avg_ret:.2f}%")

    print("\n" + "="*80)
    print("4. D7_CORE BY YEAR")
    print("="*80)

    for year in sorted(d7['year'].unique()):
        subset = d7[d7['year'] == year]
        n = len(subset)
        wins = (subset['actual_return'] > 0).sum()
        win_rate = wins / n
        lb, ub = wilson_ci(wins, n)
        avg_ret = subset['actual_return'].mean()
        print(f"{year}: n={n:3d}, WR={win_rate:.1%}, LB={lb:.1%}, Avg={avg_ret:+.1f}%")

    print("\n" + "="*80)
    print("5. D7_CORE BY SECTOR (Top 10)")
    print("="*80)

    sector_stats = []
    for sector in d7['sector'].unique():
        subset = d7[d7['sector'] == sector]
        n = len(subset)
        if n < 10:
            continue
        wins = (subset['actual_return'] > 0).sum()
        win_rate = wins / n
        lb, ub = wilson_ci(wins, n)
        avg_ret = subset['actual_return'].mean()
        sector_stats.append({
            'sector': sector,
            'n': n,
            'win_rate': win_rate,
            'wilson_lb': lb,
            'avg_return': avg_ret
        })

    sector_df = pd.DataFrame(sector_stats).sort_values('win_rate', ascending=False)
    print(sector_df.to_string(index=False))

    print("\n" + "="*80)
    print("6. STRESS QUARTERS ANALYSIS")
    print("="*80)

    # 2020Q1 and 2022Q2 are known stress quarters
    for (year, quarter, name) in [(2020, 1, "2020Q1 (COVID)"), (2022, 2, "2022Q2 (Rate Hikes)")]:
        subset = d7[(d7['year'] == year) & (d7['quarter'] == quarter)]
        n = len(subset)
        if n == 0:
            print(f"{name}: No trades")
            continue
        wins = (subset['actual_return'] > 0).sum()
        win_rate = wins / n
        avg_ret = subset['actual_return'].mean()
        print(f"{name}: n={n}, WR={win_rate:.1%}, Avg={avg_ret:+.1f}%")

    # What if we exclude stress quarters?
    d7_no_stress = d7[~((d7['year']==2020) & (d7['quarter']==1)) &
                      ~((d7['year']==2022) & (d7['quarter']==2))]
    n = len(d7_no_stress)
    wins = (d7_no_stress['actual_return'] > 0).sum()
    win_rate = wins / n
    lb, ub = wilson_ci(wins, n)
    avg_ret = d7_no_stress['actual_return'].mean()
    print(f"\nD7_CORE excluding stress quarters:")
    print(f"  Trades: {n}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Wilson 95% CI: [{lb:.1%}, {ub:.1%}]")
    print(f"  Avg Return: {avg_ret:.2f}%")

    print("\n" + "="*80)
    print("7. PATH TO WILSON LB >= 80%")
    print("="*80)

    # Calculate what we need
    current_n = len(d7)
    current_wins = (d7['actual_return'] > 0).sum()
    current_wr = current_wins / current_n
    current_lb, _ = wilson_ci(current_wins, current_n)

    print(f"\nCurrent D7_CORE: n={current_n}, WR={current_wr:.1%}, LB={current_lb:.1%}")
    print(f"Gap to 80% LB: {0.80 - current_lb:.1%}")

    # Option 1: Add more samples with same win rate
    print("\nOption 1: Add more samples at 81.9% win rate")
    for target_n in [700, 800, 1000, 1200]:
        target_wins = int(target_n * current_wr)
        lb, _ = wilson_ci(target_wins, target_n)
        print(f"  n={target_n}: LB={lb:.1%} {'✅' if lb >= 0.80 else ''}")

    # Option 2: Improve win rate
    print("\nOption 2: Improve win rate (keep n=642)")
    for target_wr in [0.83, 0.84, 0.85, 0.86]:
        target_wins = int(current_n * target_wr)
        lb, _ = wilson_ci(target_wins, current_n)
        print(f"  WR={target_wr:.0%}: LB={lb:.1%} {'✅' if lb >= 0.80 else ''}")

    # Option 3: risk=low gate
    d7_low = d7[d7['risk_code'] == 'low']
    n_low = len(d7_low)
    wins_low = (d7_low['actual_return'] > 0).sum()
    wr_low = wins_low / n_low
    lb_low, _ = wilson_ci(wins_low, n_low)
    print(f"\nOption 3: D7_CORE + risk=low")
    print(f"  n={n_low}, WR={wr_low:.1%}, LB={lb_low:.1%}")
    print(f"  Status: {'✅ Achieves LB>=80%' if lb_low >= 0.80 else '❌ Does not achieve LB>=80%'}")

    # What win rate needed for risk=low to achieve LB>=80%?
    req_wr = required_win_rate_for_wilson_lb(0.80, n_low)
    print(f"  Required WR for LB>=80%: {req_wr:.1%}")
    print(f"  Current gap: {req_wr - wr_low:.1%}")


if __name__ == "__main__":
    main()
