#!/usr/bin/env python3
"""
d7_elite_grid_search.py

Grid search for D7_ELITE gate combinations to push Wilson LB >= 80%

Target: Find gate combinations that achieve:
- Win rate >= 82.5% (needed for Wilson LB >= 80% at n~500)
- Trades >= 400 (avoid over-filtering)
- Avg return maintained or improved

Gates tested (all observable at decision time):
1. risk_code == 'low' (vs any)
2. eps_surprise >= {0, 2%, 4%, 6%}
3. earnings_day_return >= {0, 1%, 2%, 3%}
4. pre_earnings_5d_return <= {5%, 10%, 15%, 20%, 999%}
"""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
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


def run_grid_search(df: pd.DataFrame) -> pd.DataFrame:
    """Run grid search over gate combinations"""

    # Filter to D7_CORE only and trade_long=True
    d7 = df[(df['trade_long_tier'] == 'D7_CORE') & (df['trade_long'] == True)].copy()
    print(f"D7_CORE long trades: {len(d7)}")

    # Ensure numeric columns
    for col in ['eps_surprise', 'earnings_day_return', 'pre_earnings_5d_return', 'actual_return_30d_pct']:
        if col in d7.columns:
            d7[col] = pd.to_numeric(d7[col], errors='coerce')

    # Check available columns
    print(f"\nAvailable columns: {d7.columns.tolist()[:20]}...")
    print(f"\nrisk_code distribution:")
    if 'risk_code' in d7.columns:
        print(d7['risk_code'].value_counts())

    # Define gate parameters
    risk_gates = ['any', 'low']
    eps_gates = [0, 0.02, 0.04, 0.06]  # eps_surprise thresholds
    day_ret_gates = [0, 0.01, 0.02, 0.03]  # earnings_day_return thresholds
    pre_ret_gates = [0.05, 0.10, 0.15, 0.20, 999]  # pre_earnings max thresholds

    results = []

    for risk, eps, day_ret, pre_ret in product(risk_gates, eps_gates, day_ret_gates, pre_ret_gates):
        # Apply gates
        mask = pd.Series([True] * len(d7), index=d7.index)

        # Risk gate
        if risk == 'low' and 'risk_code' in d7.columns:
            mask &= (d7['risk_code'] == 'low')

        # EPS surprise gate
        if eps > 0 and 'eps_surprise' in d7.columns:
            mask &= (d7['eps_surprise'] >= eps)

        # Earnings day return gate
        if day_ret > 0 and 'earnings_day_return' in d7.columns:
            mask &= (d7['earnings_day_return'] >= day_ret)

        # Pre-earnings return gate (avoid priced-in)
        if pre_ret < 999 and 'pre_earnings_5d_return' in d7.columns:
            mask &= (d7['pre_earnings_5d_return'] <= pre_ret)

        filtered = d7[mask]
        n_trades = len(filtered)

        if n_trades < 50:  # Skip if too few trades
            continue

        # Calculate metrics
        if 'correct' in filtered.columns:
            wins = filtered['correct'].sum()
        elif 'actual_return_30d_pct' in filtered.columns:
            wins = (filtered['actual_return_30d_pct'] > 0).sum()
        else:
            continue

        win_rate = wins / n_trades if n_trades > 0 else 0
        wilson_lb, wilson_ub = wilson_ci(wins, n_trades)

        # Avg return
        avg_return = filtered['actual_return_30d_pct'].mean() if 'actual_return_30d_pct' in filtered.columns else np.nan

        results.append({
            'risk_gate': risk,
            'eps_gate': eps,
            'day_ret_gate': day_ret,
            'pre_ret_gate': pre_ret,
            'n_trades': n_trades,
            'wins': wins,
            'win_rate': win_rate,
            'wilson_lb': wilson_lb,
            'wilson_ub': wilson_ub,
            'avg_return': avg_return,
            'lb_ge_80': wilson_lb >= 0.80,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['wilson_lb', 'n_trades'], ascending=[False, False])

    return results_df


def main():
    # Try multiple possible file locations
    possible_paths = [
        Path('/Users/garen.lee/Coding/agentic-openenvolve2/EarningsCallAgenticRag/validation_runs/validation_2017Q1_2025Q3_12250_20260104_001322/validation_results.csv'),
        Path('/Users/garen.lee/Desktop/validation_results_merged_final.csv'),
        Path('/Users/garen.lee/Desktop/validation_results.csv'),
        Path('validation_results.csv'),
    ]

    df = None
    for path in possible_paths:
        if path.exists():
            print(f"Loading: {path}")
            df = pd.read_csv(path)
            break

    if df is None:
        print("Error: Could not find validation results CSV")
        return

    print(f"Total rows: {len(df)}")

    # Run grid search
    results = run_grid_search(df)

    # Save results
    outpath = Path('d7_elite_grid_results.csv')
    results.to_csv(outpath, index=False)
    print(f"\nSaved {len(results)} configurations to {outpath}")

    # Show top results
    print("\n" + "="*80)
    print("TOP 20 CONFIGURATIONS (sorted by Wilson LB)")
    print("="*80)

    top20 = results.head(20)
    display_cols = ['risk_gate', 'eps_gate', 'day_ret_gate', 'pre_ret_gate',
                    'n_trades', 'win_rate', 'wilson_lb', 'avg_return', 'lb_ge_80']
    print(top20[display_cols].to_string(index=False))

    # Show configs that achieve Wilson LB >= 80%
    lb80 = results[results['wilson_lb'] >= 0.80]
    print(f"\n\nConfigurations with Wilson LB >= 80%: {len(lb80)}")
    if len(lb80) > 0:
        print(lb80[display_cols].to_string(index=False))

    # Show Pareto front (best Wilson LB for each trade count tier)
    print("\n\nPARETO FRONT (Best Wilson LB per trade tier)")
    print("-"*80)
    for min_trades in [600, 500, 400, 300, 200, 100]:
        tier = results[results['n_trades'] >= min_trades]
        if len(tier) > 0:
            best = tier.iloc[0]
            print(f"Trades >= {min_trades}: WinRate={best['win_rate']:.1%}, "
                  f"Wilson LB={best['wilson_lb']:.1%}, n={best['n_trades']}, "
                  f"gates: risk={best['risk_gate']}, eps>={best['eps_gate']:.0%}, "
                  f"day>={best['day_ret_gate']:.0%}, pre<={best['pre_ret_gate']:.0%}")


if __name__ == "__main__":
    main()
