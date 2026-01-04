#!/usr/bin/env python3
"""
Ranking + Quota Strategy for Long-only Stock Selection.

This strategy addresses coverage instability by:
1. Computing LongQualityScore for all samples
2. Selecting top N or top X% per quarter
3. Ensuring consistent coverage regardless of market conditions

Key differences from threshold strategy:
- Threshold: "Does this stock meet criteria X, Y, Z?"
- Ranking: "Which stocks are the BEST this quarter?"

Usage:
    python ranking_quota_strategy.py --input long_only_tune_2019_2023.csv \
        --quota-pct 5 --min-quota 3

Author: Claude Code
Date: 2025-12-31
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from scipy import stats


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess test data."""
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
                    'eps_surprise', 'earnings_day_return', 'actual_return_30d_pct']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate positives count from individual flags
    pos_flags = ['le_GuidanceRaised', 'le_DemandAcceleration', 'le_MarginExpansion',
                 'le_FCFImprovement', 'le_VisibilityImproving']
    df['calc_positives'] = df[pos_flags].sum(axis=1)

    # Calculate vetoes count from individual flags
    veto_flags = ['le_GuidanceCut', 'le_DemandSoftness', 'le_MarginWeakness',
                  'le_CashBurn', 'le_VisibilityWorsening']
    df['calc_vetoes'] = df[veto_flags].sum(axis=1)

    # Risk level
    if 'le_PricedInRisk' in df.columns:
        df['is_high_risk'] = df['le_PricedInRisk'].str.lower() == 'high'
        df['is_low_risk'] = df['le_PricedInRisk'].str.lower() == 'low'
    else:
        df['is_high_risk'] = False
        df['is_low_risk'] = True

    # Create quarter string
    if 'year' in df.columns and 'quarter' in df.columns:
        df['quarter_str'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def compute_long_quality_score(
    df: pd.DataFrame,
    w_direction: float = 10.0,
    w_positives: float = 3.0,
    w_low_risk: float = 2.0,
    w_eps: float = 5.0,
    w_day_ret: float = 3.0,
    eps_cap: float = 0.20,  # Cap at 20% surprise
    day_ret_cap: float = 5.0,  # Cap at 5% day return
    veto_penalty: float = 20.0,  # Heavy penalty for vetoes
    high_risk_penalty: float = 15.0,  # Penalty for high risk
) -> pd.DataFrame:
    """
    Compute LongQualityScore for all samples.

    Formula:
        LongQualityScore = (
            DirectionScore * w_direction +
            HardPositivesCount * w_positives +
            (1 if LowRisk else 0) * w_low_risk +
            min(eps_surprise / eps_cap, 1) * w_eps +
            min(earnings_day_return / day_ret_cap, 1) * w_day_ret -
            HardVetoCount * veto_penalty -
            (high_risk_penalty if HighRisk else 0)
        )

    Score range: roughly -40 to +100
    """
    df = df.copy()

    # Component scores
    direction_score = df['le_DirectionScore'].fillna(0) * w_direction
    positives_score = df['calc_positives'].fillna(0) * w_positives
    low_risk_score = df['is_low_risk'].fillna(False).astype(int) * w_low_risk

    # EPS surprise component (normalized and capped)
    eps_normalized = (df['eps_surprise'].fillna(0) / eps_cap).clip(lower=-1, upper=1)
    eps_score = eps_normalized * w_eps

    # Earnings day return component (normalized and capped)
    day_ret_normalized = (df['earnings_day_return'].fillna(0) / day_ret_cap).clip(lower=-1, upper=1)
    day_ret_score = day_ret_normalized * w_day_ret

    # Penalties
    veto_penalty_score = df['calc_vetoes'].fillna(0) * veto_penalty
    high_risk_penalty_score = df['is_high_risk'].fillna(False).astype(int) * high_risk_penalty

    # Final score
    df['long_quality_score'] = (
        direction_score +
        positives_score +
        low_risk_score +
        eps_score +
        day_ret_score -
        veto_penalty_score -
        high_risk_penalty_score
    )

    # Also store component scores for analysis
    df['score_direction'] = direction_score
    df['score_positives'] = positives_score
    df['score_eps'] = eps_score
    df['score_day_ret'] = day_ret_score
    df['score_penalties'] = -(veto_penalty_score + high_risk_penalty_score)

    return df


def apply_quota_selection(
    df: pd.DataFrame,
    quota_pct: float = 5.0,
    min_quota: int = 3,
    max_quota: Optional[int] = None,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply per-quarter quota selection based on LongQualityScore.

    Args:
        df: DataFrame with long_quality_score column
        quota_pct: Percentage of samples to select per quarter (default 5%)
        min_quota: Minimum samples to select per quarter (default 3)
        max_quota: Maximum samples to select per quarter (optional)
        min_score: Minimum score threshold (optional, for hybrid approach)

    Returns:
        DataFrame with 'quota_selected' column indicating selected samples
    """
    df = df.copy()
    df['quota_selected'] = False
    df['quota_rank'] = 0

    quarters = df['quarter_str'].unique()

    for q in sorted(quarters):
        q_mask = df['quarter_str'] == q
        q_df = df[q_mask].copy()
        n_samples = len(q_df)

        # Calculate quota for this quarter
        quota = max(min_quota, int(n_samples * quota_pct / 100))
        if max_quota is not None:
            quota = min(quota, max_quota)

        # Apply minimum score threshold if specified
        if min_score is not None:
            eligible = q_df[q_df['long_quality_score'] >= min_score]
        else:
            eligible = q_df

        # Rank by score and select top N
        if len(eligible) > 0:
            ranked = eligible.nlargest(quota, 'long_quality_score')
            selected_indices = ranked.index

            # Update main dataframe
            df.loc[selected_indices, 'quota_selected'] = True

            # Assign ranks within quarter
            for rank, idx in enumerate(ranked.index, 1):
                df.loc[idx, 'quota_rank'] = rank

    return df


def evaluate_strategy(
    df: pd.DataFrame,
    selected_col: str = 'quota_selected',
) -> Dict[str, Any]:
    """Evaluate strategy performance on selected samples."""

    selected = df[df[selected_col] == True]

    if len(selected) == 0:
        return {
            'n_trades': 0,
            'coverage_pct': 0,
            'win_rate_pct': 0,
            'avg_return': 0,
            'std_return': 0,
        }

    n_trades = len(selected)
    coverage = n_trades / len(df) * 100

    # Win = return > 10%
    wins = (selected['actual_return_30d_pct'] > 10).sum()
    win_rate = wins / n_trades * 100

    avg_return = selected['actual_return_30d_pct'].mean()
    std_return = selected['actual_return_30d_pct'].std()

    # Winners and losers
    winners = selected[selected['actual_return_30d_pct'] > 10]
    losers = selected[selected['actual_return_30d_pct'] <= 10]

    winners_avg = winners['actual_return_30d_pct'].mean() if len(winners) > 0 else 0
    losers_avg = losers['actual_return_30d_pct'].mean() if len(losers) > 0 else 0

    # Wilson score confidence interval
    def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> tuple:
        if n == 0:
            return (0, 0)
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = wins / n
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        return (max(0, center - spread), min(1, center + spread))

    ci_lower, ci_upper = wilson_ci(wins, n_trades)

    return {
        'n_trades': n_trades,
        'coverage_pct': coverage,
        'win_rate_pct': win_rate,
        'wins': wins,
        'losses': n_trades - wins,
        'avg_return': avg_return,
        'std_return': std_return,
        'winners_avg': winners_avg,
        'losers_avg': losers_avg,
        'ci_lower': ci_lower * 100,
        'ci_upper': ci_upper * 100,
    }


def compare_strategies(
    df: pd.DataFrame,
    quota_pcts: List[float] = [3, 5, 7, 10],
    min_quotas: List[int] = [2, 3, 5],
) -> pd.DataFrame:
    """Compare different quota configurations."""

    results = []

    for quota_pct in quota_pcts:
        for min_quota in min_quotas:
            df_with_quota = apply_quota_selection(df, quota_pct=quota_pct, min_quota=min_quota)
            metrics = evaluate_strategy(df_with_quota, 'quota_selected')

            results.append({
                'quota_pct': quota_pct,
                'min_quota': min_quota,
                **metrics,
            })

    return pd.DataFrame(results)


def analyze_by_period(
    df: pd.DataFrame,
    selected_col: str = 'quota_selected',
) -> Dict[str, pd.DataFrame]:
    """Analyze strategy by year and quarter."""

    selected = df[df[selected_col] == True]

    # By year
    year_results = []
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        year_selected = selected[selected['year'] == year]

        n_samples = len(year_df)
        n_trades = len(year_selected)

        if n_trades > 0:
            wins = (year_selected['actual_return_30d_pct'] > 10).sum()
            year_results.append({
                'year': year,
                'samples': n_samples,
                'trades': n_trades,
                'coverage': n_trades / n_samples * 100,
                'win_rate': wins / n_trades * 100,
                'avg_return': year_selected['actual_return_30d_pct'].mean(),
            })
        else:
            year_results.append({
                'year': year,
                'samples': n_samples,
                'trades': 0,
                'coverage': 0,
                'win_rate': 0,
                'avg_return': 0,
            })

    # By quarter
    quarter_results = []
    for q in sorted(df['quarter_str'].unique()):
        q_df = df[df['quarter_str'] == q]
        q_selected = selected[selected['quarter_str'] == q]

        n_samples = len(q_df)
        n_trades = len(q_selected)

        if n_trades > 0:
            wins = (q_selected['actual_return_30d_pct'] > 10).sum()
            quarter_results.append({
                'quarter': q,
                'samples': n_samples,
                'trades': n_trades,
                'coverage': n_trades / n_samples * 100,
                'win_rate': wins / n_trades * 100,
                'avg_return': q_selected['actual_return_30d_pct'].mean(),
            })
        else:
            quarter_results.append({
                'quarter': q,
                'samples': n_samples,
                'trades': 0,
                'coverage': 0,
                'win_rate': 0,
                'avg_return': 0,
            })

    return {
        'by_year': pd.DataFrame(year_results),
        'by_quarter': pd.DataFrame(quarter_results),
    }


def print_report(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    period_analysis: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
):
    """Print comprehensive strategy report."""

    print("\n" + "=" * 80)
    print("RANKING + QUOTA STRATEGY REPORT")
    print("=" * 80)

    print("\n[Configuration]")
    for key, val in config.items():
        print(f"  {key}: {val}")

    print("\n[Overall Performance]")
    print(f"  Total Samples: {len(df)}")
    print(f"  Selected Trades: {metrics['n_trades']}")
    print(f"  Coverage: {metrics['coverage_pct']:.1f}%")
    print(f"  Win Rate: {metrics['win_rate_pct']:.1f}% ({metrics['wins']}W / {metrics['losses']}L)")
    print(f"  95% CI: [{metrics['ci_lower']:.1f}%, {metrics['ci_upper']:.1f}%]")
    print(f"  Avg Return: {metrics['avg_return']:+.2f}%")
    print(f"  Std Return: {metrics['std_return']:.2f}%")
    print(f"  Winners Avg: {metrics['winners_avg']:+.2f}%")
    print(f"  Losers Avg: {metrics['losers_avg']:+.2f}%")

    print("\n[Performance by Year]")
    year_df = period_analysis['by_year']
    print(f"  {'Year':<6} {'Samples':<8} {'Trades':<8} {'Coverage':<10} {'Win Rate':<10} {'Avg Ret':<10}")
    print("  " + "-" * 60)
    for _, row in year_df.iterrows():
        print(f"  {int(row['year']):<6} {int(row['samples']):<8} {int(row['trades']):<8} "
              f"{row['coverage']:>6.1f}%    {row['win_rate']:>6.1f}%    {row['avg_return']:>+7.2f}%")

    print("\n[Coverage Stability Check]")
    coverages = period_analysis['by_quarter']['coverage']
    print(f"  Min Coverage: {coverages.min():.1f}%")
    print(f"  Max Coverage: {coverages.max():.1f}%")
    print(f"  Std Coverage: {coverages.std():.2f}%")
    print(f"  Coverage Range: {coverages.max() - coverages.min():.1f}%")

    # Stability score (lower is better)
    stability_score = coverages.std() / coverages.mean() * 100 if coverages.mean() > 0 else 100
    print(f"  Stability Score (CV%): {stability_score:.1f}% {'(STABLE)' if stability_score < 30 else '(UNSTABLE)'}")


def grid_search_quota_params(
    df: pd.DataFrame,
    quota_pcts: List[float] = [3, 4, 5, 6, 7, 8, 10],
    min_quotas: List[int] = [2, 3, 4, 5],
    w_directions: List[float] = [8, 10, 12],
    w_positives_list: List[float] = [2, 3, 4],
    w_eps_list: List[float] = [3, 5, 7],
    w_day_ret_list: List[float] = [2, 3, 4],
) -> pd.DataFrame:
    """Grid search for optimal quota and scoring parameters."""

    results = []
    total = len(quota_pcts) * len(min_quotas) * len(w_directions) * len(w_positives_list) * len(w_eps_list) * len(w_day_ret_list)

    print(f"\nRunning grid search over {total} configurations...")

    combo = 0
    for quota_pct in quota_pcts:
        for min_quota in min_quotas:
            for w_dir in w_directions:
                for w_pos in w_positives_list:
                    for w_eps in w_eps_list:
                        for w_day in w_day_ret_list:
                            combo += 1

                            # Compute scores
                            df_scored = compute_long_quality_score(
                                df,
                                w_direction=w_dir,
                                w_positives=w_pos,
                                w_eps=w_eps,
                                w_day_ret=w_day,
                            )

                            # Apply quota
                            df_selected = apply_quota_selection(
                                df_scored,
                                quota_pct=quota_pct,
                                min_quota=min_quota,
                            )

                            # Evaluate
                            metrics = evaluate_strategy(df_selected, 'quota_selected')

                            # Coverage stability
                            period_analysis = analyze_by_period(df_selected, 'quota_selected')
                            coverages = period_analysis['by_quarter']['coverage']
                            coverage_std = coverages.std()
                            coverage_cv = (coverage_std / coverages.mean() * 100) if coverages.mean() > 0 else 100

                            results.append({
                                'quota_pct': quota_pct,
                                'min_quota': min_quota,
                                'w_direction': w_dir,
                                'w_positives': w_pos,
                                'w_eps': w_eps,
                                'w_day_ret': w_day,
                                'coverage_std': coverage_std,
                                'coverage_cv': coverage_cv,
                                **metrics,
                            })

    results_df = pd.DataFrame(results)
    print(f"Grid search complete. {len(results_df)} configurations tested.")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Ranking + Quota Strategy for Long-only Selection")
    parser.add_argument('--input', type=str, required=True, help='Input CSV from test run')
    parser.add_argument('--output', type=str, default=None, help='Output CSV with scores')
    parser.add_argument('--quota-pct', type=float, default=5.0, help='Quota percentage per quarter')
    parser.add_argument('--min-quota', type=int, default=3, help='Minimum quota per quarter')
    parser.add_argument('--grid-search', action='store_true', help='Run grid search')
    parser.add_argument('--compare-threshold', action='store_true', help='Compare with threshold strategy')

    args = parser.parse_args()

    # Load data
    df = load_data(args.input)

    if args.grid_search:
        # Run grid search
        results = grid_search_quota_params(df)

        # Filter by minimum requirements
        filtered = results[(results['n_trades'] >= 50) & (results['win_rate_pct'] >= 75)]

        if len(filtered) > 0:
            # Sort by composite score: win_rate - coverage instability
            filtered = filtered.copy()
            filtered['score'] = filtered['win_rate_pct'] - filtered['coverage_cv'] * 0.5
            top = filtered.nlargest(20, 'score')

            print("\n" + "=" * 80)
            print("TOP 20 CONFIGURATIONS (by composite score)")
            print("=" * 80)

            for i, (idx, row) in enumerate(top.iterrows(), 1):
                print(f"\n[Rank {i}] Score: {row['score']:.1f}")
                print(f"  Quota: {row['quota_pct']:.0f}%, min={int(row['min_quota'])}")
                print(f"  Weights: dir={row['w_direction']}, pos={row['w_positives']}, eps={row['w_eps']}, day={row['w_day_ret']}")
                print(f"  Results: {int(row['n_trades'])} trades ({row['coverage_pct']:.1f}% coverage)")
                print(f"           Win Rate: {row['win_rate_pct']:.1f}%")
                print(f"           Coverage CV: {row['coverage_cv']:.1f}% (stability)")

        # Save grid search results
        output_path = args.output or 'quota_grid_search_results.csv'
        results.to_csv(output_path, index=False)
        print(f"\nGrid search results saved to: {output_path}")

    else:
        # Single run with specified parameters
        config = {
            'quota_pct': args.quota_pct,
            'min_quota': args.min_quota,
            'w_direction': 10.0,
            'w_positives': 3.0,
            'w_eps': 5.0,
            'w_day_ret': 3.0,
        }

        # Compute scores
        df_scored = compute_long_quality_score(df)

        # Apply quota selection
        df_selected = apply_quota_selection(
            df_scored,
            quota_pct=args.quota_pct,
            min_quota=args.min_quota,
        )

        # Evaluate
        metrics = evaluate_strategy(df_selected, 'quota_selected')
        period_analysis = analyze_by_period(df_selected, 'quota_selected')

        # Print report
        print_report(df_selected, metrics, period_analysis, config)

        # Compare with threshold strategy if requested
        if args.compare_threshold and 'trade_long' in df.columns:
            print("\n" + "=" * 80)
            print("COMPARISON: QUOTA vs THRESHOLD STRATEGY")
            print("=" * 80)

            threshold_metrics = evaluate_strategy(df, 'trade_long')
            threshold_period = analyze_by_period(df, 'trade_long')

            threshold_coverages = threshold_period['by_quarter']['coverage']
            threshold_cv = (threshold_coverages.std() / threshold_coverages.mean() * 100) if threshold_coverages.mean() > 0 else 100

            quota_coverages = period_analysis['by_quarter']['coverage']
            quota_cv = (quota_coverages.std() / quota_coverages.mean() * 100) if quota_coverages.mean() > 0 else 100

            print(f"\n{'Metric':<20} {'Threshold':<15} {'Quota':<15} {'Better':<10}")
            print("-" * 60)
            print(f"{'Trades':<20} {threshold_metrics['n_trades']:<15} {metrics['n_trades']:<15}")
            print(f"{'Coverage':<20} {threshold_metrics['coverage_pct']:<14.1f}% {metrics['coverage_pct']:<14.1f}%")
            print(f"{'Win Rate':<20} {threshold_metrics['win_rate_pct']:<14.1f}% {metrics['win_rate_pct']:<14.1f}% {'Quota' if metrics['win_rate_pct'] > threshold_metrics['win_rate_pct'] else 'Threshold'}")
            print(f"{'Avg Return':<20} {threshold_metrics['avg_return']:<+14.2f}% {metrics['avg_return']:<+14.2f}% {'Quota' if metrics['avg_return'] > threshold_metrics['avg_return'] else 'Threshold'}")
            print(f"{'Coverage CV':<20} {threshold_cv:<14.1f}% {quota_cv:<14.1f}% {'Quota' if quota_cv < threshold_cv else 'Threshold'}")
            print(f"{'Coverage Range':<20} {threshold_coverages.max()-threshold_coverages.min():<14.1f}% {quota_coverages.max()-quota_coverages.min():<14.1f}%")

        # Save output
        if args.output:
            df_selected.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
