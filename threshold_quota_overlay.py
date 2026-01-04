#!/usr/bin/env python3
"""
Threshold-First + Quota Overlay Strategy (v1.1)

This strategy combines:
1. Threshold Strategy (D7 CORE + D6 STRICT) as the PRIMARY signal layer
2. Quota Overlay as a COVERAGE CONTROLLER (not signal replacement)

Key insight from v1.0 analysis:
- Pure quota in 2024: 73.7% win rate (BAD - diluted alpha)
- Threshold in 2024: 98.3% win rate (GOOD - high precision)
- Quota should NOT replace threshold, it should OVERLAY it

v1.1 Logic:
- Step 1: Run threshold strategy → strict_trades
- Step 2: Quota Overlay:
  (A) CAP: If strict_trades > cap_n → rank by score, keep top cap_n
  (B) TOP-UP: If strict_trades < min_quota → fill from relaxed_pool

Author: Claude Code
Date: 2025-12-31
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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
                    'eps_surprise', 'earnings_day_return', 'actual_return_30d_pct',
                    'pre_earnings_5d_return']
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
        df['is_medium_risk'] = df['le_PricedInRisk'].str.lower() == 'medium'
    else:
        df['is_high_risk'] = False
        df['is_low_risk'] = True
        df['is_medium_risk'] = False

    # Create quarter string
    if 'year' in df.columns and 'quarter' in df.columns:
        df['quarter_str'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def compute_long_quality_score_v11(
    df: pd.DataFrame,
    # Positive weights
    w_direction: float = 10.0,
    w_positives: float = 3.0,
    w_low_risk: float = 2.0,
    w_eps: float = 5.0,
    w_day_ret: float = 3.0,
    eps_cap: float = 0.20,
    day_ret_cap: float = 5.0,
    # Negative weights (v1.1 additions)
    veto_penalty: float = 20.0,
    high_risk_penalty: float = 15.0,
    medium_risk_penalty: float = 5.0,  # NEW: medium risk penalty
    prerun_penalty_weight: float = 2.0,  # NEW: pre-run penalty
    prerun_threshold: float = 10.0,  # Start penalizing above this
) -> pd.DataFrame:
    """
    Compute LongQualityScore v1.1 with improved negative risk factors.

    v1.1 additions:
    - pre_earnings_5d_return penalty (priced-in risk)
    - medium_risk penalty (not just high)

    Formula:
        LongQualityScore = (
            DirectionScore * w_direction +
            HardPositivesCount * w_positives +
            (1 if LowRisk else 0) * w_low_risk +
            min(eps_surprise / eps_cap, 1) * w_eps +
            min(earnings_day_return / day_ret_cap, 1) * w_day_ret -
            HardVetoCount * veto_penalty -
            (high_risk_penalty if HighRisk else 0) -
            (medium_risk_penalty if MediumRisk else 0) -
            max(pre_run - prerun_threshold, 0) * prerun_penalty_weight
        )
    """
    df = df.copy()

    # Positive components
    direction_score = df['le_DirectionScore'].fillna(0).astype(float) * w_direction
    positives_score = df['calc_positives'].fillna(0).astype(float) * w_positives
    low_risk_score = df['is_low_risk'].fillna(False).astype(int) * w_low_risk

    # EPS surprise component (normalized and capped)
    eps_normalized = (df['eps_surprise'].fillna(0) / eps_cap).clip(lower=-1, upper=1)
    eps_score = eps_normalized * w_eps

    # Earnings day return component (normalized and capped)
    day_ret_normalized = (df['earnings_day_return'].fillna(0) / day_ret_cap).clip(lower=-1, upper=1)
    day_ret_score = day_ret_normalized * w_day_ret

    # Negative components
    veto_penalty_score = df['calc_vetoes'].fillna(0).astype(float) * veto_penalty
    high_risk_penalty_score = df['is_high_risk'].fillna(False).astype(int) * high_risk_penalty
    medium_risk_penalty_score = df['is_medium_risk'].fillna(False).astype(int) * medium_risk_penalty

    # Pre-run penalty (v1.1 addition)
    pre_run = df['pre_earnings_5d_return'].fillna(0) if 'pre_earnings_5d_return' in df.columns else 0
    prerun_excess = np.maximum(pre_run - prerun_threshold, 0)
    prerun_penalty_score = prerun_excess * prerun_penalty_weight

    # Final score
    df['long_quality_score'] = (
        direction_score +
        positives_score +
        low_risk_score +
        eps_score +
        day_ret_score -
        veto_penalty_score -
        high_risk_penalty_score -
        medium_risk_penalty_score -
        prerun_penalty_score
    )

    # Store component scores for debugging
    df['score_direction'] = direction_score
    df['score_positives'] = positives_score
    df['score_eps'] = eps_score
    df['score_day_ret'] = day_ret_score
    df['score_penalties'] = -(veto_penalty_score + high_risk_penalty_score +
                              medium_risk_penalty_score + prerun_penalty_score)
    df['penalty_prerun'] = -prerun_penalty_score

    return df


def apply_threshold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the existing threshold strategy (D7 CORE + D6 STRICT).

    This uses the trade_long column from the original CSV if available,
    or recomputes it based on the known rules.
    """
    df = df.copy()

    # If trade_long already exists, use it
    if 'trade_long' in df.columns:
        df['strict_trade'] = df['trade_long'].fillna(False).astype(bool)
        return df

    # Otherwise, recompute based on threshold rules
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


def identify_relaxed_pool(
    df: pd.DataFrame,
    min_day_return: float = 0.5,
    require_eps_positive: bool = True,
) -> pd.DataFrame:
    """
    Identify samples eligible for top-up (relaxed pool).

    Relaxed pool requirements (still has hard gates):
    - computed_vetoes == 0
    - risk_code != "high"
    - eps_surprise > 0 (if required)
    - earnings_day_return >= min_day_return
    - NOT already a strict_trade
    """
    df = df.copy()

    # Start with all non-strict trades
    relaxed_mask = ~df['strict_trade']

    # Hard gates for relaxed pool
    relaxed_mask &= (df['calc_vetoes'].fillna(0) == 0)
    relaxed_mask &= ~df['is_high_risk'].fillna(False)
    relaxed_mask &= (df['earnings_day_return'].fillna(-999) >= min_day_return)

    if require_eps_positive:
        relaxed_mask &= (df['eps_surprise'].fillna(-999) > 0)

    df['in_relaxed_pool'] = relaxed_mask

    return df


def apply_overlay_strategy(
    df: pd.DataFrame,
    cap_pct: float = 8.0,
    min_quota: int = 3,
    score_floor_percentile: float = 80.0,
    use_absolute_floor: bool = False,
    absolute_floor: float = 50.0,
) -> pd.DataFrame:
    """
    Apply the quota overlay (Cap + Top-up) on top of threshold strategy.

    Args:
        df: DataFrame with strict_trade, in_relaxed_pool, and long_quality_score
        cap_pct: Maximum coverage percentage (for capping)
        min_quota: Minimum trades per quarter (for top-up)
        score_floor_percentile: Minimum percentile for top-up candidates
        use_absolute_floor: Use absolute score floor instead of percentile
        absolute_floor: Absolute score threshold for top-up

    Returns:
        DataFrame with overlay_selected and selection_type columns
    """
    df = df.copy()
    df['overlay_selected'] = False
    df['selection_type'] = ''  # STRICT, CAP, TOPUP, or empty
    df['overlay_rank'] = 0

    quarters = df['quarter_str'].unique()

    for q in sorted(quarters):
        q_mask = df['quarter_str'] == q
        q_df = df[q_mask]
        n_samples = len(q_df)

        # Calculate cap
        cap_n = max(min_quota, int(n_samples * cap_pct / 100))

        # Get strict trades for this quarter
        strict_mask = q_mask & df['strict_trade']
        strict_indices = df[strict_mask].index.tolist()
        n_strict = len(strict_indices)

        # Calculate score floor for this quarter (percentile-based)
        if use_absolute_floor:
            score_floor = absolute_floor
        else:
            q_scores = q_df['long_quality_score'].dropna()
            if len(q_scores) > 0:
                score_floor = np.percentile(q_scores, score_floor_percentile)
            else:
                score_floor = 0

        selected_indices = []
        selection_types = {}

        # CASE 1: Strict trades exceed cap → CAP (rank and keep top)
        if n_strict > cap_n:
            # Rank strict trades by score
            strict_df = df.loc[strict_indices].copy()
            strict_df = strict_df.nlargest(cap_n, 'long_quality_score')

            for rank, idx in enumerate(strict_df.index, 1):
                selected_indices.append(idx)
                selection_types[idx] = 'CAP'
                df.loc[idx, 'overlay_rank'] = rank

        # CASE 2: Strict trades within normal range → use all strict
        elif n_strict >= min_quota:
            for rank, idx in enumerate(strict_indices, 1):
                selected_indices.append(idx)
                selection_types[idx] = 'STRICT'
                df.loc[idx, 'overlay_rank'] = rank

        # CASE 3: Strict trades below min_quota → TOP-UP from relaxed pool
        else:
            # First, include all strict trades
            for rank, idx in enumerate(strict_indices, 1):
                selected_indices.append(idx)
                selection_types[idx] = 'STRICT'
                df.loc[idx, 'overlay_rank'] = rank

            # Then, top-up from relaxed pool
            remaining = min_quota - n_strict
            if remaining > 0:
                # Get relaxed pool candidates for this quarter
                relaxed_mask = q_mask & df['in_relaxed_pool']
                relaxed_mask &= (df['long_quality_score'] >= score_floor)  # Apply floor!

                relaxed_df = df[relaxed_mask].copy()

                if len(relaxed_df) > 0:
                    # Rank by score and take top N
                    topup_df = relaxed_df.nlargest(remaining, 'long_quality_score')

                    for rank, idx in enumerate(topup_df.index, 1):
                        selected_indices.append(idx)
                        selection_types[idx] = 'TOPUP'
                        df.loc[idx, 'overlay_rank'] = n_strict + rank

        # Update selection status
        for idx in selected_indices:
            df.loc[idx, 'overlay_selected'] = True
            df.loc[idx, 'selection_type'] = selection_types[idx]

    return df


def evaluate_strategy(
    df: pd.DataFrame,
    selected_col: str = 'overlay_selected',
) -> Dict[str, Any]:
    """Evaluate strategy performance."""

    selected = df[df[selected_col] == True]

    if len(selected) == 0:
        return {
            'n_trades': 0,
            'coverage_pct': 0,
            'win_rate_pct': 0,
            'avg_return': 0,
        }

    n_trades = len(selected)
    coverage = n_trades / len(df) * 100

    # Win = return > 10%
    wins = (selected['actual_return_30d_pct'] > 10).sum()
    win_rate = wins / n_trades * 100

    avg_return = selected['actual_return_30d_pct'].mean()
    std_return = selected['actual_return_30d_pct'].std() if n_trades > 1 else 0

    # Wilson confidence interval
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

    # Selection type breakdown
    strict_count = len(selected[selected['selection_type'] == 'STRICT'])
    cap_count = len(selected[selected['selection_type'] == 'CAP'])
    topup_count = len(selected[selected['selection_type'] == 'TOPUP'])

    # Top-up performance
    topup_df = selected[selected['selection_type'] == 'TOPUP']
    topup_wins = (topup_df['actual_return_30d_pct'] > 10).sum() if len(topup_df) > 0 else 0
    topup_win_rate = (topup_wins / len(topup_df) * 100) if len(topup_df) > 0 else 0

    return {
        'n_trades': n_trades,
        'coverage_pct': coverage,
        'win_rate_pct': win_rate,
        'wins': wins,
        'losses': n_trades - wins,
        'avg_return': avg_return,
        'std_return': std_return,
        'ci_lower': ci_lower * 100,
        'ci_upper': ci_upper * 100,
        'strict_count': strict_count,
        'cap_count': cap_count,
        'topup_count': topup_count,
        'topup_win_rate': topup_win_rate,
    }


def analyze_by_period(
    df: pd.DataFrame,
    selected_col: str = 'overlay_selected',
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
            strict = len(year_selected[year_selected['selection_type'] == 'STRICT'])
            cap = len(year_selected[year_selected['selection_type'] == 'CAP'])
            topup = len(year_selected[year_selected['selection_type'] == 'TOPUP'])

            year_results.append({
                'year': year,
                'samples': n_samples,
                'trades': n_trades,
                'coverage': n_trades / n_samples * 100,
                'win_rate': wins / n_trades * 100,
                'avg_return': year_selected['actual_return_30d_pct'].mean(),
                'strict': strict,
                'cap': cap,
                'topup': topup,
            })
        else:
            year_results.append({
                'year': year,
                'samples': n_samples,
                'trades': 0,
                'coverage': 0,
                'win_rate': 0,
                'avg_return': 0,
                'strict': 0,
                'cap': 0,
                'topup': 0,
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
    print("THRESHOLD-FIRST + QUOTA OVERLAY STRATEGY (v1.1)")
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

    print("\n[Selection Breakdown]")
    print(f"  STRICT (threshold passed): {metrics['strict_count']}")
    print(f"  CAP (threshold capped): {metrics['cap_count']}")
    print(f"  TOPUP (from relaxed pool): {metrics['topup_count']}")
    if metrics['topup_count'] > 0:
        print(f"  TOPUP Win Rate: {metrics['topup_win_rate']:.1f}%")

    print("\n[Performance by Year]")
    year_df = period_analysis['by_year']
    print(f"  {'Year':<6} {'Trades':<8} {'Coverage':<10} {'Win Rate':<10} {'Strict':<8} {'Cap':<6} {'TopUp':<6}")
    print("  " + "-" * 65)
    for _, row in year_df.iterrows():
        print(f"  {int(row['year']):<6} {int(row['trades']):<8} "
              f"{row['coverage']:>6.1f}%    {row['win_rate']:>6.1f}%    "
              f"{int(row['strict']):<8} {int(row['cap']):<6} {int(row['topup']):<6}")

    print("\n[Coverage Stability]")
    coverages = period_analysis['by_quarter']['coverage']
    print(f"  Min Coverage: {coverages.min():.1f}%")
    print(f"  Max Coverage: {coverages.max():.1f}%")
    print(f"  Coverage Range: {coverages.max() - coverages.min():.1f}%")
    cv = (coverages.std() / coverages.mean() * 100) if coverages.mean() > 0 else 100
    print(f"  Coverage CV: {cv:.1f}%")


def compare_strategies(
    df: pd.DataFrame,
    overlay_metrics: Dict[str, Any],
):
    """Compare overlay strategy with pure threshold."""

    print("\n" + "=" * 80)
    print("COMPARISON: v1.1 OVERLAY vs PURE THRESHOLD vs PURE QUOTA")
    print("=" * 80)

    # Pure threshold metrics
    threshold_selected = df[df['strict_trade'] == True]
    threshold_n = len(threshold_selected)
    threshold_wins = (threshold_selected['actual_return_30d_pct'] > 10).sum()
    threshold_win_rate = (threshold_wins / threshold_n * 100) if threshold_n > 0 else 0
    threshold_coverage = threshold_n / len(df) * 100

    print(f"\n{'Metric':<25} {'Threshold':<15} {'v1.1 Overlay':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Trades':<25} {threshold_n:<15} {overlay_metrics['n_trades']:<15} "
          f"{overlay_metrics['n_trades'] - threshold_n:+d}")
    print(f"{'Coverage':<25} {threshold_coverage:<14.1f}% {overlay_metrics['coverage_pct']:<14.1f}% "
          f"{overlay_metrics['coverage_pct'] - threshold_coverage:+.1f}%")
    print(f"{'Win Rate':<25} {threshold_win_rate:<14.1f}% {overlay_metrics['win_rate_pct']:<14.1f}% "
          f"{overlay_metrics['win_rate_pct'] - threshold_win_rate:+.1f}%")

    # Key insight
    print("\n[Key Insight]")
    if overlay_metrics['win_rate_pct'] >= threshold_win_rate - 2:
        print("  ✅ v1.1 maintains threshold win rate (within 2%)")
    else:
        print(f"  ⚠️ v1.1 reduced win rate by {threshold_win_rate - overlay_metrics['win_rate_pct']:.1f}%")

    if overlay_metrics['topup_count'] > 0:
        if overlay_metrics['topup_win_rate'] >= 70:
            print(f"  ✅ TOPUP quality is acceptable ({overlay_metrics['topup_win_rate']:.1f}% win rate)")
        else:
            print(f"  ⚠️ TOPUP quality is concerning ({overlay_metrics['topup_win_rate']:.1f}% win rate)")


def main():
    parser = argparse.ArgumentParser(description="Threshold-First + Quota Overlay Strategy v1.1")
    parser.add_argument('--input', type=str, required=True, help='Input CSV from test run')
    parser.add_argument('--output', type=str, default=None, help='Output CSV with selections')
    parser.add_argument('--cap-pct', type=float, default=8.0, help='Cap percentage per quarter')
    parser.add_argument('--min-quota', type=int, default=3, help='Minimum trades per quarter')
    parser.add_argument('--score-floor-pct', type=float, default=80.0, help='Score floor percentile for top-up')
    parser.add_argument('--prerun-penalty', type=float, default=2.0, help='Pre-run penalty weight')
    parser.add_argument('--prerun-threshold', type=float, default=10.0, help='Pre-run penalty threshold')

    args = parser.parse_args()

    config = {
        'cap_pct': args.cap_pct,
        'min_quota': args.min_quota,
        'score_floor_percentile': args.score_floor_pct,
        'prerun_penalty_weight': args.prerun_penalty,
        'prerun_threshold': args.prerun_threshold,
    }

    # Load data
    df = load_data(args.input)

    # Step 1: Compute quality scores (v1.1 with pre-run penalty)
    df = compute_long_quality_score_v11(
        df,
        prerun_penalty_weight=args.prerun_penalty,
        prerun_threshold=args.prerun_threshold,
    )

    # Step 2: Apply threshold strategy
    df = apply_threshold_strategy(df)

    # Step 3: Identify relaxed pool
    df = identify_relaxed_pool(df)

    # Step 4: Apply overlay strategy
    df = apply_overlay_strategy(
        df,
        cap_pct=args.cap_pct,
        min_quota=args.min_quota,
        score_floor_percentile=args.score_floor_pct,
    )

    # Evaluate
    metrics = evaluate_strategy(df, 'overlay_selected')
    period_analysis = analyze_by_period(df, 'overlay_selected')

    # Print report
    print_report(df, metrics, period_analysis, config)

    # Compare with pure threshold
    compare_strategies(df, metrics)

    # Save output
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
