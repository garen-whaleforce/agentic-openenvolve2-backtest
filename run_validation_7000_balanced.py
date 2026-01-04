#!/usr/bin/env python3
"""
2017-2025 Q3 Validation Script - 8750 samples (250/quarter) with lookahead fix v2.2

Production-grade validation runner with:
- Stratified sampling (250/quarter × 35 quarters = 8750 samples)
- Reproducible with seed (default: 42)
- Manifest support for exact reproducibility
- Run metadata for audit trail
- D7-only Wilson CI calculation
- 2022Q1 regime tagging
- Auto-generated supplementary statistics

Target: ~716 long trades (based on ~8.2% trade_long rate)

Usage:
    # Test run (5 samples)
    python run_validation_7000_balanced.py --test-only --save-manifest

    # Full run with manifest save
    python run_validation_7000_balanced.py --save-manifest --parallel 12

    # Resume from manifest
    python run_validation_7000_balanced.py --manifest path/to/manifest.csv --resume path/to/results.csv
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CRITICAL: Lookahead Protection Enforcement
# =============================================================================

def enforce_lookahead_protection():
    """Enforce lookahead protection settings. Fail if misconfigured."""
    # Check DISABLE_LEAKAGE_CHECK - must NOT be true
    if os.environ.get("DISABLE_LEAKAGE_CHECK", "").lower() == "true":
        raise RuntimeError(
            "FATAL: DISABLE_LEAKAGE_CHECK=true is set. "
            "This validation REQUIRES leakage checking. "
            "Unset this variable or set to false."
        )

    # Check LOOKAHEAD_ASSERTIONS - should be true
    if os.environ.get("LOOKAHEAD_ASSERTIONS", "").lower() != "true":
        print("=" * 70)
        print("WARNING: LOOKAHEAD_ASSERTIONS is not 'true'")
        print("This validation should run with LOOKAHEAD_ASSERTIONS=true")
        print("Setting it now...")
        print("=" * 70)
        os.environ["LOOKAHEAD_ASSERTIONS"] = "true"

    # Set required environment
    os.environ["HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS"] = "false"

    # Bump cache version if not set
    if not os.environ.get("CALL_CACHE_VERSION"):
        os.environ["CALL_CACHE_VERSION"] = "v2.2"

enforce_lookahead_protection()

from pg_client import get_cursor

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLES = 12250  # 35 quarters × 350 samples
SAMPLES_PER_QUARTER = 350
DEFAULT_CONCURRENCY = 10
TEST_SAMPLES = 5
START_YEAR = 2017
END_YEAR = 2025
END_QUARTER = 3  # Up to 2025 Q3
DEFAULT_SEED = 42

# =============================================================================
# Utility Functions
# =============================================================================

def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def get_env_snapshot() -> Dict[str, str]:
    """Get environment snapshot (keys only for secrets, values for non-secrets)."""
    secret_keys = {
        "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "NEO4J_PASSWORD", "DATABASE_URL", "PG_PASSWORD"
    }
    snapshot = {}
    for key in sorted(os.environ.keys()):
        if key.startswith(("LOOKAHEAD", "CALL_CACHE", "HISTORICAL_EARNINGS", "DISABLE_LEAKAGE")):
            snapshot[key] = os.environ[key]
        elif key in secret_keys:
            snapshot[key] = "[SET]" if os.environ.get(key) else "[NOT SET]"
    return snapshot


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Calculate Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0, 0.0

    from scipy import stats
    p = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * ((p * (1-p) + z**2 / (4*n)) / n) ** 0.5 / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return lower, upper, (upper - lower) / 2


# =============================================================================
# Sample Selection
# =============================================================================

def get_all_quarters() -> List[tuple]:
    """Get all quarter combinations from 2017Q1 to 2025Q3."""
    quarters = []
    for year in range(START_YEAR, END_YEAR + 1):
        max_q = END_QUARTER if year == END_YEAR else 4
        for q in range(1, max_q + 1):
            quarters.append((year, q))
    return quarters


def deterministic_hash(symbol: str, year: int, quarter: int, seed: int) -> str:
    """Generate deterministic hash for ordering."""
    key = f"{symbol}:{year}:Q{quarter}:seed{seed}"
    return hashlib.md5(key.encode()).hexdigest()


def get_stratified_samples(per_quarter: int = 200, seed: int = 42) -> List[Dict]:
    """Get stratified samples with deterministic ordering."""
    quarters = get_all_quarters()
    num_quarters = len(quarters)

    print(f"Stratified sampling: {per_quarter}/quarter × {num_quarters} quarters = {per_quarter * num_quarters} total")
    print(f"Seed: {seed}")

    all_samples = []
    quarter_counts = {}

    with get_cursor() as cur:
        if cur is None:
            raise RuntimeError("Database connection failed")

        for year, quarter in quarters:
            fetch_limit = per_quarter * 3

            cur.execute("""
                SELECT
                    et.symbol,
                    et.year,
                    et.quarter,
                    et.transcript_date_str,
                    pa.pct_change_t_plus_30 as actual_return,
                    c.name as company_name,
                    c.sector
                FROM earnings_transcripts et
                LEFT JOIN price_analysis pa ON et.id = pa.transcript_id
                LEFT JOIN companies c ON et.symbol = c.symbol
                WHERE et.year = %s
                    AND et.quarter = %s
                    AND et.transcript_date_str IS NOT NULL
                    AND pa.pct_change_t_plus_30 IS NOT NULL
                LIMIT %s
            """, (year, quarter, fetch_limit))

            quarter_samples = []
            for row in cur.fetchall():
                sample = {
                    "symbol": row["symbol"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "transcript_date": row["transcript_date_str"],
                    "actual_return": float(row["actual_return"]) if row["actual_return"] else None,
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "_hash": deterministic_hash(row["symbol"], year, quarter, seed)
                }
                quarter_samples.append(sample)

            quarter_samples.sort(key=lambda x: x["_hash"])
            selected = quarter_samples[:per_quarter]

            for s in selected:
                del s["_hash"]

            all_samples.extend(selected)
            quarter_key = f"{year}Q{quarter}"
            quarter_counts[quarter_key] = len(selected)

            if len(selected) < per_quarter:
                print(f"  Warning: {quarter_key} only has {len(selected)} samples")

    # Deterministic shuffle
    import random
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    # Add regime tags
    for s in all_samples:
        s["is_2022q1"] = (s["year"] == 2022 and s["quarter"] == 1)
        s["regime_tag"] = "2022Q1" if s["is_2022q1"] else ""

    print(f"Total samples: {len(all_samples)}")
    return all_samples


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load samples from manifest file."""
    df = pd.read_csv(manifest_path)
    samples = df.to_dict('records')

    # Add regime tags if missing
    for s in samples:
        if "is_2022q1" not in s:
            s["is_2022q1"] = (s["year"] == 2022 and s["quarter"] == 1)
        if "regime_tag" not in s:
            s["regime_tag"] = "2022Q1" if s["is_2022q1"] else ""

    print(f"Loaded {len(samples)} samples from manifest: {manifest_path}")
    return samples


def save_manifest(samples: List[Dict], output_path: str) -> str:
    """Save universe manifest."""
    manifest_data = []
    for s in samples:
        manifest_data.append({
            "symbol": s["symbol"],
            "year": s["year"],
            "quarter": s["quarter"],
            "transcript_date": s.get("transcript_date"),
            "sector": s.get("sector"),
            "company_name": s.get("company_name"),
            "actual_return": s.get("actual_return"),
            "is_2022q1": s.get("is_2022q1", False),
            "regime_tag": s.get("regime_tag", ""),
        })

    df = pd.DataFrame(manifest_data)
    df.to_csv(output_path, index=False)
    print(f"Manifest saved: {output_path}")
    return output_path


# =============================================================================
# Analysis Runner
# =============================================================================

async def analyze_sample(sample: Dict) -> Dict:
    """Run analysis on a single sample."""
    from analysis_engine import analyze_earnings_async

    start_time = time.time()
    symbol = sample["symbol"]
    year = sample["year"]
    quarter = sample["quarter"]
    transcript_date = sample.get("transcript_date")
    actual_return = sample.get("actual_return")

    try:
        result = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            skip_cache=True,  # Force fresh analysis for validation
        )

        elapsed = time.time() - start_time

        # Extract from agentic_result (the actual analysis output)
        ar = result.get("agentic_result", {}) or {}

        prediction = ar.get("prediction", "")
        confidence = ar.get("confidence", 0)
        direction_score = ar.get("le_DirectionScore")
        trade_long = ar.get("trade_long", False)
        trade_long_tier = ar.get("trade_long_tier", "")
        risk_code = ar.get("risk_code", "")
        long_block_reason = ar.get("long_block_reason", "")

        if actual_return is not None and prediction:
            pred_up = prediction.upper() == "UP"
            actual_up = actual_return > 0
            correct = pred_up == actual_up
        else:
            correct = None

        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "company_name": sample.get("company_name", ""),
            "sector": sample.get("sector", ""),
            "is_2022q1": sample.get("is_2022q1", False),
            "regime_tag": sample.get("regime_tag", ""),
            "success": True,
            "error": None,
            "time_seconds": elapsed,
            "prediction": prediction,
            "confidence": confidence,
            "direction_score": direction_score,
            "actual_return": actual_return,
            "correct": correct,
            "trade_long": trade_long,
            "trade_long_tier": trade_long_tier,
            "risk_code": risk_code,
            "long_block_reason": long_block_reason,
            "eps_surprise": ar.get("eps_surprise"),
            "earnings_day_return": ar.get("earnings_day_return"),
            "pre_earnings_5d_return": ar.get("pre_earnings_5d_return"),
            "computed_positives": ar.get("computed_positives"),
            "computed_vetoes": ar.get("computed_vetoes"),
            "le_DirectionScore": ar.get("le_DirectionScore"),
            "le_HardPositivesCount": ar.get("le_HardPositivesCount"),
            "le_HardVetoCount": ar.get("le_HardVetoCount"),
            "le_PricedInRisk": ar.get("le_PricedInRisk"),
            "le_GuidanceRaised": ar.get("le_GuidanceRaised"),
            "le_DemandAcceleration": ar.get("le_DemandAcceleration"),
            "le_MarginExpansion": ar.get("le_MarginExpansion"),
            "le_FCFImprovement": ar.get("le_FCFImprovement"),
            "le_VisibilityImproving": ar.get("le_VisibilityImproving"),
            "le_GuidanceCut": ar.get("le_GuidanceCut"),
            "le_DemandSoftness": ar.get("le_DemandSoftness"),
            "le_MarginWeakness": ar.get("le_MarginWeakness"),
            "le_CashBurn": ar.get("le_CashBurn"),
            "le_VisibilityWorsening": ar.get("le_VisibilityWorsening"),
            "input_tokens": ar.get("input_tokens"),
            "output_tokens": ar.get("output_tokens"),
            "cost_usd": ar.get("cost_usd"),
            "summary": ar.get("summary", "")[:500] if ar.get("summary") else "",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "company_name": sample.get("company_name", ""),
            "sector": sample.get("sector", ""),
            "is_2022q1": sample.get("is_2022q1", False),
            "regime_tag": sample.get("regime_tag", ""),
            "success": False,
            "error": str(e)[:200],
            "time_seconds": elapsed,
            "prediction": None,
            "confidence": None,
            "direction_score": None,
            "actual_return": actual_return,
            "correct": None,
            "trade_long": False,
            "trade_long_tier": None,
            "risk_code": None,
            "long_block_reason": None,
        }


async def run_batch(
    samples: List[Dict],
    concurrency: int,
    report_interval: int = 100,
    existing_results: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Run analysis with concurrency limit and progress reporting."""
    semaphore = asyncio.Semaphore(concurrency)
    results = existing_results or []
    completed = len(results)
    start_time = time.time()
    last_report_time = start_time

    # Skip already completed
    completed_keys = set()
    if existing_results:
        for r in existing_results:
            key = f"{r['symbol']}:{r['year']}:Q{r['quarter']}"
            completed_keys.add(key)

    samples_to_run = []
    for s in samples:
        key = f"{s['symbol']}:{s['year']}:Q{s['quarter']}"
        if key not in completed_keys:
            samples_to_run.append(s)

    if existing_results:
        print(f"Resuming: {len(existing_results)} completed, {len(samples_to_run)} remaining")

    async def run_with_semaphore(idx: int, sample: Dict) -> Dict:
        nonlocal completed, last_report_time
        async with semaphore:
            print(f"[{completed+1}/{len(samples)}] Starting {sample['symbol']} {sample['year']}Q{sample['quarter']}...")
            result = await analyze_sample(sample)
            completed += 1

            status = "OK" if result["success"] else "FAIL"
            pred = result.get("prediction", "N/A")
            correct = "Y" if result.get("correct") else ("N" if result.get("correct") is False else "?")
            trade_long = "LONG" if result.get("trade_long") else ""
            print(f"[{completed}/{len(samples)}] {sample['symbol']} Q{sample['quarter']}: {status} | {pred} | Correct={correct} | {trade_long}")

            current_time = time.time()
            if completed % report_interval == 0 or (current_time - last_report_time) >= 120:
                last_report_time = current_time
                print_progress(results + [result], completed, len(samples), start_time)

            return result

    tasks = [run_with_semaphore(i, s) for i, s in enumerate(samples_to_run)]
    new_results = await asyncio.gather(*tasks)

    return results + list(new_results)


def print_progress(results: List[Dict], completed: int, total: int, start_time: float):
    """Print progress summary."""
    elapsed = time.time() - start_time
    rate = completed / elapsed * 60 if elapsed > 0 else 0
    eta = (total - completed) / (completed / elapsed) if completed > 0 else 0

    successful = [r for r in results if r.get("success")]
    correct = [r for r in successful if r.get("correct")]
    long_trades = [r for r in successful if r.get("trade_long")]
    long_wins = [r for r in long_trades if r.get("actual_return", 0) > 0]

    print(f"\n{'='*70}")
    print(f"PROGRESS [{completed}/{total}] ({100*completed/total:.1f}%)")
    print(f"{'='*70}")
    print(f"Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min | Rate: {rate:.1f}/min")
    print(f"Success: {len(successful)} | Failed: {completed - len(successful)}")

    if successful:
        acc = len(correct) / len(successful) * 100
        print(f"Overall Accuracy: {acc:.1f}% ({len(correct)}/{len(successful)})")

    if long_trades:
        win_rate = len(long_wins) / len(long_trades) * 100
        avg_return = sum(r.get("actual_return", 0) for r in long_trades) / len(long_trades)
        print(f"Long-only: {len(long_trades)} trades | Win Rate: {win_rate:.1f}%")
        print(f"Long Avg Return: {avg_return:.2f}%")

        d7 = [r for r in long_trades if r.get("trade_long_tier") == "D7_CORE"]
        d6 = [r for r in long_trades if r.get("trade_long_tier") == "D6_STRICT"]
        if d7:
            d7_wins = len([r for r in d7 if r.get("actual_return", 0) > 0])
            print(f"  D7_CORE: {len(d7)} trades, {d7_wins} wins ({100*d7_wins/len(d7):.1f}%)")
        if d6:
            d6_wins = len([r for r in d6 if r.get("actual_return", 0) > 0])
            print(f"  D6_STRICT: {len(d6)} trades, {d6_wins} wins ({100*d6_wins/len(d6):.1f}%)")

    print(f"{'='*70}\n")


# =============================================================================
# Output Generation
# =============================================================================

def generate_supplementary_stats(results: List[Dict], output_dir: Path, seed: int) -> Dict[str, str]:
    """Generate comprehensive supplementary statistics."""
    files = {}

    df = pd.DataFrame(results)
    valid = df[df['success'] & df['actual_return'].notna()].copy()
    long_df = valid[valid['trade_long'] == True].copy()

    # 1. Main supplementary stats markdown
    md_lines = [
        f"# Validation 7000 Samples Supplementary Statistics",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Seed**: {seed}",
        f"**Version**: v2.2 (Lookahead Assertions Enabled)",
        f"",
        f"---",
        f"",
        f"## 1. Headline Statistics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Samples | {len(df)} |",
        f"| Success Rate | {100*df['success'].mean():.1f}% |",
        f"| Valid Samples | {len(valid)} |",
    ]

    if len(valid) > 0:
        correct = valid['correct'].sum()
        md_lines.append(f"| Overall Accuracy | {100*correct/len(valid):.1f}% ({correct}/{len(valid)}) |")

    md_lines.extend([
        f"| **Long Trades** | **{len(long_df)}** |",
    ])

    if len(long_df) > 0:
        long_wins = (long_df['actual_return'] > 0).sum()
        long_avg = long_df['actual_return'].mean()
        long_median = long_df['actual_return'].median()

        md_lines.extend([
            f"| **Long Win Rate** | **{100*long_wins/len(long_df):.1f}% ({long_wins}/{len(long_df)})** |",
            f"| **Long Avg Return** | **+{long_avg:.2f}%** |",
            f"| Long Median Return | +{long_median:.2f}% |",
        ])

        # Wilson CI - All longs
        lower, upper, half_width = wilson_ci(long_wins, len(long_df))
        md_lines.extend([
            f"",
            f"### Wilson 95% CI (All Long Trades)",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Win Rate | {100*long_wins/len(long_df):.1f}% |",
            f"| Wilson 95% CI | [{100*lower:.1f}%, {100*upper:.1f}%] |",
            f"| Half-width | {100*half_width:.1f}% |",
        ])

        # Wilson CI - D7 only
        d7_df = long_df[long_df['trade_long_tier'] == 'D7_CORE']
        if len(d7_df) > 0:
            d7_wins = (d7_df['actual_return'] > 0).sum()
            d7_lower, d7_upper, d7_hw = wilson_ci(d7_wins, len(d7_df))
            md_lines.extend([
                f"",
                f"### Wilson 95% CI (D7_CORE Only)",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| D7 Trades | {len(d7_df)} |",
                f"| D7 Win Rate | {100*d7_wins/len(d7_df):.1f}% |",
                f"| Wilson 95% CI | [{100*d7_lower:.1f}%, {100*d7_upper:.1f}%] |",
                f"| **Wilson LB >= 80%?** | **{'YES' if d7_lower >= 0.80 else 'NO'}** |",
            ])

        # Tier breakdown
        md_lines.extend([
            f"",
            f"### Tier Breakdown",
            f"",
            f"| Tier | Trades | Win Rate | Avg Return |",
            f"|------|--------|----------|------------|",
        ])

        for tier in ['D7_CORE', 'D6_STRICT']:
            tier_df = long_df[long_df['trade_long_tier'] == tier]
            if len(tier_df) > 0:
                tier_wins = (tier_df['actual_return'] > 0).sum()
                tier_avg = tier_df['actual_return'].mean()
                md_lines.append(f"| {tier} | {len(tier_df)} | {100*tier_wins/len(tier_df):.1f}% | +{tier_avg:.2f}% |")

        # 2022Q1 Regime Analysis
        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 2. 2022Q1 Regime Analysis",
            f"",
        ])

        q1_2022 = long_df[long_df['is_2022q1'] == True]
        non_q1_2022 = long_df[long_df['is_2022q1'] == False]

        md_lines.extend([
            f"| Regime | Trades | Win Rate | Avg Return |",
            f"|--------|--------|----------|------------|",
        ])

        if len(q1_2022) > 0:
            q1_wins = (q1_2022['actual_return'] > 0).sum()
            q1_avg = q1_2022['actual_return'].mean()
            md_lines.append(f"| **2022Q1** | {len(q1_2022)} | {100*q1_wins/len(q1_2022):.1f}% | {q1_avg:+.2f}% |")

        if len(non_q1_2022) > 0:
            non_wins = (non_q1_2022['actual_return'] > 0).sum()
            non_avg = non_q1_2022['actual_return'].mean()
            md_lines.append(f"| Other | {len(non_q1_2022)} | {100*non_wins/len(non_q1_2022):.1f}% | +{non_avg:.2f}% |")

        # Loss tail
        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Loss Tail Statistics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| P05 | {long_df['actual_return'].quantile(0.05):.2f}% |",
            f"| P10 | {long_df['actual_return'].quantile(0.10):.2f}% |",
            f"| Min (worst) | {long_df['actual_return'].min():.2f}% |",
            f"| Max (best) | {long_df['actual_return'].max():.2f}% |",
        ])

        wins_sum = long_df[long_df['actual_return'] > 0]['actual_return'].sum()
        losses_sum = abs(long_df[long_df['actual_return'] <= 0]['actual_return'].sum())
        pf = wins_sum / losses_sum if losses_sum > 0 else float('inf')
        md_lines.extend([
            f"| Profit Factor | {pf:.2f} |",
        ])

    md_lines.extend([
        f"",
        f"---",
        f"",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    md_path = output_dir / "supplementary_stats.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    files['supplementary_stats'] = str(md_path)

    # 2. Quarter stats CSV
    if len(long_df) > 0:
        quarterly = long_df.groupby(['year', 'quarter']).agg({
            'actual_return': ['count', lambda x: (x > 0).sum(), 'mean', 'sum', 'min', 'max']
        }).round(4)
        quarterly.columns = ['trades', 'wins', 'avg_return', 'total_return', 'min_return', 'max_return']
        quarterly['win_rate'] = (quarterly['wins'] / quarterly['trades']).round(4)
        quarterly = quarterly.reset_index()
        quarterly['yq'] = quarterly['year'].astype(str) + 'Q' + quarterly['quarter'].astype(str)

        qtr_path = output_dir / "long_quarter_stats.csv"
        quarterly.to_csv(qtr_path, index=False)
        files['quarter_stats'] = str(qtr_path)

    # 3. Year stats CSV
    if len(long_df) > 0:
        yearly = long_df.groupby('year').agg({
            'actual_return': ['count', lambda x: (x > 0).sum(), 'mean', 'sum', 'min', 'max']
        }).round(4)
        yearly.columns = ['trades', 'wins', 'avg_return', 'total_return', 'min_return', 'max_return']
        yearly['win_rate'] = (yearly['wins'] / yearly['trades']).round(4)
        yearly = yearly.reset_index()

        year_path = output_dir / "long_year_stats.csv"
        yearly.to_csv(year_path, index=False)
        files['year_stats'] = str(year_path)

    # 4. Sector×Tier for 2019 and 2022
    for year in [2019, 2022]:
        year_df = long_df[long_df['year'] == year]
        if len(year_df) > 0:
            grouped = year_df.groupby(['sector', 'trade_long_tier']).agg({
                'actual_return': ['count', lambda x: (x > 0).sum(), 'mean', 'sum', 'min']
            }).round(4)
            grouped.columns = ['trades', 'wins', 'avg_return', 'total_return', 'min_return']
            grouped['win_rate'] = (grouped['wins'] / grouped['trades']).round(4)
            grouped = grouped.reset_index()

            attr_path = output_dir / f"{year}_sector_tier_attribution.csv"
            grouped.to_csv(attr_path, index=False)
            files[f'{year}_attribution'] = str(attr_path)

    # 5. Worst losses
    if len(long_df) > 0:
        worst = long_df.nsmallest(25, 'actual_return')[
            ['symbol', 'year', 'quarter', 'sector', 'trade_long_tier', 'actual_return', 'is_2022q1']
        ]
        worst_path = output_dir / "long_worst_losses.csv"
        worst.to_csv(worst_path, index=False)
        files['worst_losses'] = str(worst_path)

    return files


def save_run_metadata(
    output_dir: Path,
    seed: int,
    samples_count: int,
    per_quarter: int,
    concurrency: int,
    manifest_path: Optional[str],
    results_path: str,
    stats_files: Dict[str, str],
) -> str:
    """Save run metadata for audit trail."""
    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "seed": seed,
        "samples_count": samples_count,
        "per_quarter": per_quarter,
        "concurrency": concurrency,
        "range": f"{START_YEAR}Q1-{END_YEAR}Q{END_QUARTER}",
        "manifest_path": manifest_path,
        "results_path": results_path,
        "stats_files": stats_files,
        "env_snapshot": get_env_snapshot(),
        "lookahead_protection": {
            "LOOKAHEAD_ASSERTIONS": os.environ.get("LOOKAHEAD_ASSERTIONS"),
            "DISABLE_LEAKAGE_CHECK": os.environ.get("DISABLE_LEAKAGE_CHECK"),
            "HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS": os.environ.get("HISTORICAL_EARNINGS_INCLUDE_POST_RETURNS"),
            "CALL_CACHE_VERSION": os.environ.get("CALL_CACHE_VERSION"),
        }
    }

    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {meta_path}")
    return str(meta_path)


def print_final_summary(results: List[Dict], seed: int):
    """Print comprehensive final summary."""
    df = pd.DataFrame(results)
    successful = df[df['success'] == True]
    valid = successful[successful['actual_return'].notna()]
    long_trades = valid[valid['trade_long'] == True]

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY (Seed: {seed})")
    print(f"{'='*70}")
    print(f"Total Samples: {len(df)}")
    print(f"Successful: {len(successful)}")

    if len(valid) > 0:
        correct = valid['correct'].sum()
        print(f"Overall Accuracy: {correct}/{len(valid)} ({100*correct/len(valid):.1f}%)")

    print(f"\nLong Trades: {len(long_trades)}")

    if len(long_trades) > 0:
        long_wins = (long_trades['actual_return'] > 0).sum()
        win_rate = 100 * long_wins / len(long_trades)
        avg_return = long_trades['actual_return'].mean()

        print(f"Long Win Rate: {long_wins}/{len(long_trades)} ({win_rate:.1f}%)")
        print(f"Long Avg Return: {avg_return:.2f}%")

        # Wilson CI
        lower, upper, _ = wilson_ci(long_wins, len(long_trades))
        print(f"\nWilson 95% CI: [{100*lower:.1f}%, {100*upper:.1f}%]")

        # D7 only
        d7 = long_trades[long_trades['trade_long_tier'] == 'D7_CORE']
        if len(d7) > 0:
            d7_wins = (d7['actual_return'] > 0).sum()
            d7_lower, d7_upper, _ = wilson_ci(d7_wins, len(d7))
            print(f"\nD7_CORE Only:")
            print(f"  Trades: {len(d7)}, Win Rate: {100*d7_wins/len(d7):.1f}%")
            print(f"  Wilson 95% CI: [{100*d7_lower:.1f}%, {100*d7_upper:.1f}%]")
            print(f"  Wilson LB >= 80%: {'YES' if d7_lower >= 0.80 else 'NO'}")

        # 2022Q1 regime
        q1_2022 = long_trades[long_trades['is_2022q1'] == True]
        if len(q1_2022) > 0:
            q1_wins = (q1_2022['actual_return'] > 0).sum()
            print(f"\n2022Q1 Regime:")
            print(f"  Trades: {len(q1_2022)}, Win Rate: {100*q1_wins/len(q1_2022):.1f}%")
            print(f"  Avg Return: {q1_2022['actual_return'].mean():.2f}%")

    print(f"{'='*70}")


async def main():
    parser = argparse.ArgumentParser(description="Run 7000 sample validation (stratified)")
    parser.add_argument("--test-only", action="store_true", help="Only run test samples")
    parser.add_argument("--samples", type=int, help="Override total samples")
    parser.add_argument("--per-quarter", type=int, default=SAMPLES_PER_QUARTER, help="Samples per quarter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--parallel", type=int, default=DEFAULT_CONCURRENCY, help="Concurrency")
    parser.add_argument("--report-interval", type=int, default=100, help="Progress report interval")
    parser.add_argument("--manifest", type=str, help="Load samples from manifest file")
    parser.add_argument("--save-manifest", action="store_true", help="Save universe manifest")
    parser.add_argument("--resume", type=str, help="Resume from existing results CSV")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"7000 SAMPLE STRATIFIED VALIDATION (v2.2)")
    print(f"{'='*70}")
    print(f"LOOKAHEAD_ASSERTIONS: {os.environ.get('LOOKAHEAD_ASSERTIONS')}")
    print(f"CALL_CACHE_VERSION: {os.environ.get('CALL_CACHE_VERSION')}")
    print(f"Seed: {args.seed}")
    print(f"Concurrency: {args.parallel}")
    print(f"{'='*70}\n")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"validation_runs/validation_{START_YEAR}Q1_{END_YEAR}Q{END_QUARTER}_{args.per_quarter * 35}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get samples
    if args.manifest:
        samples = load_manifest(args.manifest)
    else:
        samples = get_stratified_samples(per_quarter=args.per_quarter, seed=args.seed)

    # Save manifest if requested
    manifest_path = None
    if args.save_manifest:
        manifest_path = str(output_dir / f"universe_manifest_seed{args.seed}.csv")
        save_manifest(samples, manifest_path)

    # Load existing results for resume
    existing_results = None
    if args.resume:
        existing_df = pd.read_csv(args.resume)
        existing_results = existing_df.to_dict('records')
        print(f"Loaded {len(existing_results)} existing results for resume")

    # Limit samples
    if args.samples:
        samples = samples[:args.samples]

    # Test phase
    if args.test_only:
        print(f"\n--- TEST PHASE ({TEST_SAMPLES} samples) ---\n")
        test_results = await run_batch(samples[:TEST_SAMPLES], concurrency=TEST_SAMPLES, report_interval=1)

        test_success = sum(1 for r in test_results if r.get("success"))
        print(f"\nTest Results: {test_success}/{TEST_SAMPLES} successful")

        results_path = output_dir / "test_results.csv"
        pd.DataFrame(test_results).to_csv(results_path, index=False)
        print(f"Test results saved: {results_path}")
        return

    # Full run
    print(f"\n--- FULL RUN ({len(samples)} samples) ---\n")
    all_results = await run_batch(
        samples,
        concurrency=args.parallel,
        report_interval=args.report_interval,
        existing_results=existing_results,
    )

    # Save results
    results_path = output_dir / "validation_results.csv"
    pd.DataFrame(all_results).to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")

    # Generate supplementary stats
    stats_files = generate_supplementary_stats(all_results, output_dir, args.seed)

    # Save metadata
    save_run_metadata(
        output_dir=output_dir,
        seed=args.seed,
        samples_count=len(samples),
        per_quarter=args.per_quarter,
        concurrency=args.parallel,
        manifest_path=manifest_path,
        results_path=str(results_path),
        stats_files=stats_files,
    )

    # Print summary
    print_final_summary(all_results, args.seed)

    print(f"\n{'='*70}")
    print(f"OUTPUT FILES")
    print(f"{'='*70}")
    print(f"Results: {results_path}")
    if manifest_path:
        print(f"Manifest: {manifest_path}")
    for name, path in stats_files.items():
        print(f"{name}: {path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Change to EarningsCallAgenticRag directory
    script_dir = Path(__file__).parent
    rag_dir = script_dir / "EarningsCallAgenticRag"
    if rag_dir.exists():
        os.chdir(rag_dir)

    asyncio.run(main())
