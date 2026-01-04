#!/usr/bin/env python3
"""
Full Validation Runner for Long-only Strategy v1.1

Runs frozen v1.1 parameters across 4 time periods:
1. Backward Holdout: 2017-2018 (regime test)
2. Tune Set: 2019-2023 (parameter optimization period)
3. Validate Set: 2024 (out-of-sample validation)
4. Forward Holdout: 2025 Q1-Q3 (blind test extension)

Key Metrics:
- Win Rate with Wilson 95% CI lower bound
- Coverage with CV (stability)
- Tail Risk: p10, p05, max loss, profit factor
- Selection breakdown: STRICT/CAP/TOPUP
- Quarterly shortfall analysis

Usage:
    python run_full_validation.py --generate-data   # Generate test data for all periods
    python run_full_validation.py --analyze         # Analyze existing CSV files
    python run_full_validation.py --all             # Both generate and analyze

Author: Claude Code
Date: 2025-12-31
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
from scipy import stats


# =============================================================================
# FROZEN v1.1 PARAMETERS (DO NOT CHANGE DURING VALIDATION)
# =============================================================================
FROZEN_PARAMS = {
    'cap_pct': 8.0,
    'min_quota': 3,
    'score_floor_percentile': 80.0,
    'prerun_penalty_weight': 2.0,
    'prerun_threshold': 10.0,
    # Scoring weights
    'w_direction': 10.0,
    'w_positives': 3.0,
    'w_low_risk': 2.0,
    'w_eps': 5.0,
    'w_day_ret': 3.0,
    'veto_penalty': 20.0,
    'high_risk_penalty': 15.0,
    'medium_risk_penalty': 5.0,
}

# Time period definitions
TIME_PERIODS = {
    'backward_holdout': {
        'name': 'Backward Holdout',
        'years': (2017, 2018),
        'purpose': 'Regime Test (pre-tune)',
        'color': 'blue',
    },
    'tune': {
        'name': 'Tune Set',
        'years': (2019, 2023),
        'purpose': 'Parameter Optimization',
        'color': 'green',
    },
    'validate': {
        'name': 'Validate Set',
        'years': (2024, 2024),
        'purpose': 'Out-of-Sample Validation',
        'color': 'yellow',
    },
    'forward_holdout': {
        'name': 'Forward Holdout',
        'years': (2025, 2025),
        'purpose': 'Blind Test',
        'color': 'red',
    },
}


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = wins / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0, center - spread), min(1, center + spread))


def calculate_tail_risk(returns: pd.Series) -> Dict[str, float]:
    """Calculate tail risk metrics."""
    if len(returns) == 0:
        return {
            'p05': 0, 'p10': 0, 'max_loss': 0,
            'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
        }

    p05 = returns.quantile(0.05) if len(returns) >= 20 else returns.min()
    p10 = returns.quantile(0.10) if len(returns) >= 10 else returns.min()
    max_loss = returns.min()

    wins = returns[returns > 10]
    losses = returns[returns <= 10]

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    # Profit factor = gross profit / gross loss
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001  # Avoid div by zero
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'p05': p05,
        'p10': p10,
        'max_loss': max_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
    }


def load_and_process_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV and preprocess for analysis."""
    df = pd.read_csv(csv_path)

    # Convert boolean-like strings
    bool_cols = ['le_GuidanceRaised', 'le_DemandAcceleration', 'le_MarginExpansion',
                 'le_FCFImprovement', 'le_VisibilityImproving', 'le_GuidanceCut',
                 'le_DemandSoftness', 'le_MarginWeakness', 'le_CashBurn',
                 'le_VisibilityWorsening']

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'YES': True, 'NO': False, True: True, False: False})

    # Convert numeric columns
    numeric_cols = ['le_DirectionScore', 'le_HardPositivesCount', 'le_HardVetoCount',
                    'eps_surprise', 'earnings_day_return', 'actual_return_30d_pct',
                    'pre_earnings_5d_return']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate positives/vetoes
    pos_flags = ['le_GuidanceRaised', 'le_DemandAcceleration', 'le_MarginExpansion',
                 'le_FCFImprovement', 'le_VisibilityImproving']
    df['calc_positives'] = df[pos_flags].sum(axis=1)

    veto_flags = ['le_GuidanceCut', 'le_DemandSoftness', 'le_MarginWeakness',
                  'le_CashBurn', 'le_VisibilityWorsening']
    df['calc_vetoes'] = df[veto_flags].sum(axis=1)

    # Risk levels
    if 'le_PricedInRisk' in df.columns:
        df['is_high_risk'] = df['le_PricedInRisk'].str.lower() == 'high'
        df['is_low_risk'] = df['le_PricedInRisk'].str.lower() == 'low'
        df['is_medium_risk'] = df['le_PricedInRisk'].str.lower() == 'medium'
    else:
        df['is_high_risk'] = False
        df['is_low_risk'] = True
        df['is_medium_risk'] = False

    # Quarter string
    if 'year' in df.columns and 'quarter' in df.columns:
        df['quarter_str'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

    return df


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LongQualityScore v1.1 with frozen parameters."""
    df = df.copy()
    p = FROZEN_PARAMS

    # Positive components
    direction_score = df['le_DirectionScore'].fillna(0).astype(float) * p['w_direction']
    positives_score = df['calc_positives'].fillna(0).astype(float) * p['w_positives']
    low_risk_score = df['is_low_risk'].fillna(False).astype(int) * p['w_low_risk']

    eps_normalized = (df['eps_surprise'].fillna(0) / 0.20).clip(lower=-1, upper=1)
    eps_score = eps_normalized * p['w_eps']

    day_ret_normalized = (df['earnings_day_return'].fillna(0) / 5.0).clip(lower=-1, upper=1)
    day_ret_score = day_ret_normalized * p['w_day_ret']

    # Negative components
    veto_penalty_score = df['calc_vetoes'].fillna(0).astype(float) * p['veto_penalty']
    high_risk_penalty_score = df['is_high_risk'].fillna(False).astype(int) * p['high_risk_penalty']
    medium_risk_penalty_score = df['is_medium_risk'].fillna(False).astype(int) * p['medium_risk_penalty']

    # Pre-run penalty
    pre_run = df['pre_earnings_5d_return'].fillna(0) if 'pre_earnings_5d_return' in df.columns else 0
    prerun_excess = np.maximum(pre_run - p['prerun_threshold'], 0)
    prerun_penalty_score = prerun_excess * p['prerun_penalty_weight']

    df['long_quality_score'] = (
        direction_score + positives_score + low_risk_score +
        eps_score + day_ret_score -
        veto_penalty_score - high_risk_penalty_score -
        medium_risk_penalty_score - prerun_penalty_score
    )

    return df


def apply_threshold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply D7 CORE + D6 STRICT threshold strategy."""
    df = df.copy()

    if 'trade_long' in df.columns:
        df['strict_trade'] = df['trade_long'].fillna(False).astype(bool)
        return df

    df['strict_trade'] = False

    for idx, row in df.iterrows():
        direction = row.get('le_DirectionScore', 0) or 0
        vetoes = row.get('calc_vetoes', 0) or 0
        positives = row.get('calc_positives', 0) or 0
        eps_surprise = row.get('eps_surprise', 0) or 0
        day_return = row.get('earnings_day_return', 0) or 0
        is_high_risk = row.get('is_high_risk', False)

        # D7+ CORE
        if direction >= 7 and vetoes == 0 and not is_high_risk:
            if eps_surprise > 0 and day_return >= 1.5:
                df.loc[idx, 'strict_trade'] = True
                continue

        # D6 STRICT
        if direction == 6 and vetoes == 0:
            if day_return >= 1.0 and positives >= 1:
                df.loc[idx, 'strict_trade'] = True

    return df


def identify_relaxed_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Identify samples eligible for top-up."""
    df = df.copy()

    relaxed_mask = ~df['strict_trade']
    relaxed_mask &= (df['calc_vetoes'].fillna(0) == 0)
    relaxed_mask &= ~df['is_high_risk'].fillna(False)
    relaxed_mask &= (df['earnings_day_return'].fillna(-999) >= 0.5)
    relaxed_mask &= (df['eps_surprise'].fillna(-999) > 0)

    df['in_relaxed_pool'] = relaxed_mask
    return df


def apply_overlay_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Cap + Top-up overlay with frozen parameters."""
    df = df.copy()
    p = FROZEN_PARAMS

    df['overlay_selected'] = False
    df['selection_type'] = ''
    df['overlay_rank'] = 0

    quarters = df['quarter_str'].unique()

    for q in sorted(quarters):
        q_mask = df['quarter_str'] == q
        q_df = df[q_mask]
        n_samples = len(q_df)

        cap_n = max(p['min_quota'], int(n_samples * p['cap_pct'] / 100))

        strict_mask = q_mask & df['strict_trade']
        strict_indices = df[strict_mask].index.tolist()
        n_strict = len(strict_indices)

        # Score floor (percentile-based)
        q_scores = q_df['long_quality_score'].dropna()
        score_floor = np.percentile(q_scores, p['score_floor_percentile']) if len(q_scores) > 0 else 0

        selected_indices = []
        selection_types = {}

        # CAP
        if n_strict > cap_n:
            strict_df = df.loc[strict_indices].nlargest(cap_n, 'long_quality_score')
            for rank, idx in enumerate(strict_df.index, 1):
                selected_indices.append(idx)
                selection_types[idx] = 'CAP'
                df.loc[idx, 'overlay_rank'] = rank

        # Normal (use all strict)
        elif n_strict >= p['min_quota']:
            for rank, idx in enumerate(strict_indices, 1):
                selected_indices.append(idx)
                selection_types[idx] = 'STRICT'
                df.loc[idx, 'overlay_rank'] = rank

        # TOP-UP
        else:
            for rank, idx in enumerate(strict_indices, 1):
                selected_indices.append(idx)
                selection_types[idx] = 'STRICT'
                df.loc[idx, 'overlay_rank'] = rank

            remaining = p['min_quota'] - n_strict
            if remaining > 0:
                relaxed_mask = q_mask & df['in_relaxed_pool']
                relaxed_mask &= (df['long_quality_score'] >= score_floor)
                relaxed_df = df[relaxed_mask]

                if len(relaxed_df) > 0:
                    topup_df = relaxed_df.nlargest(remaining, 'long_quality_score')
                    for rank, idx in enumerate(topup_df.index, 1):
                        selected_indices.append(idx)
                        selection_types[idx] = 'TOPUP'
                        df.loc[idx, 'overlay_rank'] = n_strict + rank

        for idx in selected_indices:
            df.loc[idx, 'overlay_selected'] = True
            df.loc[idx, 'selection_type'] = selection_types[idx]

    return df


def analyze_period(df: pd.DataFrame, period_name: str) -> Dict[str, Any]:
    """Analyze a single time period comprehensively."""
    selected = df[df['overlay_selected'] == True]

    n_total = len(df)
    n_trades = len(selected)
    coverage = n_trades / n_total * 100 if n_total > 0 else 0

    if n_trades == 0:
        return {
            'period': period_name,
            'n_samples': n_total,
            'n_trades': 0,
            'coverage_pct': 0,
            'win_rate_pct': 0,
            'wilson_lower': 0,
            'wilson_upper': 0,
            'avg_return': 0,
            'strict_count': 0,
            'cap_count': 0,
            'topup_count': 0,
            'topup_win_rate': 0,
            'p05': 0,
            'p10': 0,
            'max_loss': 0,
            'profit_factor': 0,
        }

    returns = selected['actual_return_30d_pct']
    wins = (returns > 10).sum()
    win_rate = wins / n_trades * 100

    ci_lower, ci_upper = wilson_ci(wins, n_trades)
    tail_risk = calculate_tail_risk(returns)

    # Selection breakdown
    strict_count = len(selected[selected['selection_type'] == 'STRICT'])
    cap_count = len(selected[selected['selection_type'] == 'CAP'])
    topup_count = len(selected[selected['selection_type'] == 'TOPUP'])

    topup_df = selected[selected['selection_type'] == 'TOPUP']
    topup_wins = (topup_df['actual_return_30d_pct'] > 10).sum() if len(topup_df) > 0 else 0
    topup_win_rate = (topup_wins / len(topup_df) * 100) if len(topup_df) > 0 else 0

    return {
        'period': period_name,
        'n_samples': n_total,
        'n_trades': n_trades,
        'coverage_pct': coverage,
        'win_rate_pct': win_rate,
        'wins': wins,
        'losses': n_trades - wins,
        'wilson_lower': ci_lower * 100,
        'wilson_upper': ci_upper * 100,
        'avg_return': returns.mean(),
        'std_return': returns.std(),
        'strict_count': strict_count,
        'cap_count': cap_count,
        'topup_count': topup_count,
        'topup_win_rate': topup_win_rate,
        **tail_risk,
    }


def analyze_quarterly_shortfall(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze quarterly top-up shortfall."""
    p = FROZEN_PARAMS
    selected = df[df['overlay_selected'] == True]

    results = []
    for q in sorted(df['quarter_str'].unique()):
        q_df = df[df['quarter_str'] == q]
        q_selected = selected[selected['quarter_str'] == q]

        n_samples = len(q_df)
        n_trades = len(q_selected)
        n_strict = len(q_selected[q_selected['selection_type'] == 'STRICT'])
        n_cap = len(q_selected[q_selected['selection_type'] == 'CAP'])
        n_topup = len(q_selected[q_selected['selection_type'] == 'TOPUP'])

        # Shortfall = how many trades short of min_quota
        shortfall = max(0, p['min_quota'] - n_trades)

        # Win rate for this quarter
        if n_trades > 0:
            wins = (q_selected['actual_return_30d_pct'] > 10).sum()
            win_rate = wins / n_trades * 100
        else:
            win_rate = 0

        results.append({
            'quarter': q,
            'samples': n_samples,
            'trades': n_trades,
            'coverage': n_trades / n_samples * 100 if n_samples > 0 else 0,
            'strict': n_strict,
            'cap': n_cap,
            'topup': n_topup,
            'shortfall': shortfall,
            'win_rate': win_rate,
        })

    return pd.DataFrame(results)


def process_csv_file(csv_path: str, period_key: str) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Process a single CSV file and return metrics."""
    period_info = TIME_PERIODS[period_key]

    print(f"\n{'='*60}")
    print(f"Processing: {period_info['name']} ({period_info['purpose']})")
    print(f"{'='*60}")

    df = load_and_process_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    df = compute_quality_score(df)
    df = apply_threshold_strategy(df)
    df = identify_relaxed_pool(df)
    df = apply_overlay_strategy(df)

    metrics = analyze_period(df, period_info['name'])
    quarterly = analyze_quarterly_shortfall(df)

    return metrics, df, quarterly


def print_comprehensive_report(all_metrics: List[Dict], all_quarterly: List[pd.DataFrame]):
    """Print comprehensive validation report."""

    print("\n" + "=" * 80)
    print("FULL VALIDATION REPORT - Long-only Strategy v1.1 (Frozen Parameters)")
    print("=" * 80)

    print("\n[Frozen Parameters]")
    for key, val in FROZEN_PARAMS.items():
        print(f"  {key}: {val}")

    # Summary table
    print("\n" + "-" * 80)
    print("SUMMARY BY PERIOD")
    print("-" * 80)
    print(f"{'Period':<20} {'Samples':<10} {'Trades':<8} {'Coverage':<10} {'Win Rate':<12} {'Wilson LB':<12} {'Avg Ret':<10}")
    print("-" * 80)

    for m in all_metrics:
        print(f"{m['period']:<20} {m['n_samples']:<10} {m['n_trades']:<8} "
              f"{m['coverage_pct']:>6.1f}%    {m['win_rate_pct']:>6.1f}%      "
              f"{m['wilson_lower']:>6.1f}%      {m['avg_return']:>+7.2f}%")

    # Combined metrics
    total_samples = sum(m['n_samples'] for m in all_metrics)
    total_trades = sum(m['n_trades'] for m in all_metrics)
    total_wins = sum(m.get('wins', 0) for m in all_metrics)

    if total_trades > 0:
        overall_win_rate = total_wins / total_trades * 100
        overall_coverage = total_trades / total_samples * 100
        overall_wilson_lower, overall_wilson_upper = wilson_ci(total_wins, total_trades)

        print("-" * 80)
        print(f"{'OVERALL':<20} {total_samples:<10} {total_trades:<8} "
              f"{overall_coverage:>6.1f}%    {overall_win_rate:>6.1f}%      "
              f"{overall_wilson_lower*100:>6.1f}%")

    # Selection breakdown
    print("\n" + "-" * 80)
    print("SELECTION BREAKDOWN BY PERIOD")
    print("-" * 80)
    print(f"{'Period':<20} {'STRICT':<10} {'CAP':<10} {'TOPUP':<10} {'TOPUP WR':<12}")
    print("-" * 80)

    for m in all_metrics:
        topup_wr = f"{m['topup_win_rate']:.1f}%" if m['topup_count'] > 0 else "N/A"
        print(f"{m['period']:<20} {m['strict_count']:<10} {m['cap_count']:<10} "
              f"{m['topup_count']:<10} {topup_wr:<12}")

    # Tail risk
    print("\n" + "-" * 80)
    print("TAIL RISK METRICS BY PERIOD")
    print("-" * 80)
    print(f"{'Period':<20} {'P05':<10} {'P10':<10} {'Max Loss':<12} {'Profit Factor':<15}")
    print("-" * 80)

    for m in all_metrics:
        pf = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else "inf"
        print(f"{m['period']:<20} {m['p05']:>+7.1f}%  {m['p10']:>+7.1f}%  "
              f"{m['max_loss']:>+9.1f}%   {pf:<15}")

    # Quarterly shortfall analysis
    print("\n" + "-" * 80)
    print("QUARTERLY SHORTFALL ANALYSIS (Quarters with shortfall > 0)")
    print("-" * 80)

    all_quarterly_df = pd.concat(all_quarterly, ignore_index=True)
    shortfall_quarters = all_quarterly_df[all_quarterly_df['shortfall'] > 0]

    if len(shortfall_quarters) > 0:
        print(f"{'Quarter':<12} {'Samples':<10} {'Trades':<8} {'Shortfall':<10} {'Win Rate':<10}")
        print("-" * 50)
        for _, row in shortfall_quarters.iterrows():
            print(f"{row['quarter']:<12} {int(row['samples']):<10} {int(row['trades']):<8} "
                  f"{int(row['shortfall']):<10} {row['win_rate']:>6.1f}%")
    else:
        print("No quarters with shortfall (all met min_quota)")

    # Coverage stability
    print("\n" + "-" * 80)
    print("COVERAGE STABILITY BY PERIOD")
    print("-" * 80)

    for i, quarterly_df in enumerate(all_quarterly):
        period_name = all_metrics[i]['period']
        coverages = quarterly_df['coverage']
        cv = (coverages.std() / coverages.mean() * 100) if coverages.mean() > 0 else 100

        print(f"{period_name}:")
        print(f"  Coverage Range: {coverages.min():.1f}% - {coverages.max():.1f}%")
        print(f"  Coverage CV: {cv:.1f}%")

    # Final verdict
    print("\n" + "=" * 80)
    print("VALIDATION VERDICT")
    print("=" * 80)

    if total_trades > 0:
        # Check if Wilson lower bound >= 80%
        if overall_wilson_lower >= 0.80:
            print("✅ PASS: Wilson 95% lower bound >= 80%")
        elif overall_wilson_lower >= 0.75:
            print("⚠️ MARGINAL: Wilson lower bound 75-80%")
        else:
            print("❌ FAIL: Wilson lower bound < 75%")

        # Check coverage
        if 5 <= overall_coverage <= 10:
            print("✅ PASS: Coverage within 5-10% target")
        else:
            print(f"⚠️ Coverage {overall_coverage:.1f}% outside 5-10% target")


def main():
    parser = argparse.ArgumentParser(description="Full Validation Runner for v1.1 Strategy")
    parser.add_argument('--input-dir', type=str, default='.', help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for reports')

    # Individual period CSV files
    parser.add_argument('--backward', type=str, help='2017-2018 CSV file')
    parser.add_argument('--tune', type=str, help='2019-2023 CSV file')
    parser.add_argument('--validate', type=str, help='2024 CSV file')
    parser.add_argument('--forward', type=str, help='2025 CSV file')

    args = parser.parse_args()

    # Map period keys to CSV files
    csv_files = {}

    if args.backward:
        csv_files['backward_holdout'] = args.backward
    if args.tune:
        csv_files['tune'] = args.tune
    if args.validate:
        csv_files['validate'] = args.validate
    if args.forward:
        csv_files['forward_holdout'] = args.forward

    # Auto-detect if no files specified
    if not csv_files:
        input_dir = Path(args.input_dir)

        # Try to find matching files
        patterns = {
            'backward_holdout': ['*2017*2018*.csv', '*backward*.csv'],
            'tune': ['*2019*2023*.csv', '*tune*.csv'],
            'validate': ['*2024*.csv', '*validate*.csv'],
            'forward_holdout': ['*2025*.csv', '*forward*.csv', '*holdout*.csv'],
        }

        for period, pats in patterns.items():
            for pat in pats:
                matches = list(input_dir.glob(pat))
                if matches:
                    csv_files[period] = str(matches[0])
                    break

    if not csv_files:
        print("No CSV files found. Please specify files with --backward, --tune, --validate, --forward")
        print("\nExample:")
        print("  python run_full_validation.py --tune long_only_test_2019_2023.csv --validate long_only_test_2024.csv")
        return

    print(f"Found {len(csv_files)} period(s) to analyze:")
    for period, path in csv_files.items():
        print(f"  {TIME_PERIODS[period]['name']}: {path}")

    # Process each period
    all_metrics = []
    all_quarterly = []
    all_dfs = {}

    for period_key, csv_path in csv_files.items():
        if os.path.exists(csv_path):
            metrics, df, quarterly = process_csv_file(csv_path, period_key)
            all_metrics.append(metrics)
            all_quarterly.append(quarterly)
            all_dfs[period_key] = df
        else:
            print(f"Warning: File not found: {csv_path}")

    if all_metrics:
        print_comprehensive_report(all_metrics, all_quarterly)

        # Save combined results
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)

            # Save metrics summary
            pd.DataFrame(all_metrics).to_csv(output_dir / 'validation_summary.csv', index=False)

            # Save quarterly analysis
            pd.concat(all_quarterly).to_csv(output_dir / 'quarterly_analysis.csv', index=False)

            print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
