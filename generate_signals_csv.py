#!/usr/bin/env python3
"""
Generate signals CSV from validation_results.csv for portfolio backtesting.

Takes validation results and adds reaction_date from earnings_transcripts table.
Output: signals CSV with columns: symbol, reaction_date, trade_long, trade_long_tier
"""
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pg_client import get_cursor

def get_reaction_dates_batch(symbols_quarters: list) -> dict:
    """
    Batch fetch reaction dates for multiple symbol/year/quarter combinations.
    Returns dict: {(symbol, year, quarter): reaction_date_str}
    """
    if not symbols_quarters:
        return {}

    results = {}
    with get_cursor() as cur:
        if cur is None:
            print("WARNING: Could not connect to database")
            return results

        try:
            # Build query for all combinations
            for symbol, year, quarter in symbols_quarters:
                cur.execute("""
                    SELECT transcript_date
                    FROM earnings_transcripts
                    WHERE UPPER(symbol) = %s AND year = %s AND quarter = %s
                """, (symbol.upper(), int(year), int(quarter)))
                row = cur.fetchone()
                if row and row.get("transcript_date"):
                    results[(symbol.upper(), int(year), int(quarter))] = str(row["transcript_date"])[:10]
        except Exception as e:
            print(f"Database error: {e}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate signals CSV from validation results")
    parser.add_argument("--input", "-i", required=True, help="Path to validation_results.csv")
    parser.add_argument("--output", "-o", required=True, help="Path to output signals CSV")
    parser.add_argument("--long-only", action="store_true", default=True,
                        help="Only include trade_long=True signals (default: True)")
    args = parser.parse_args()

    # Load validation results
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")

    # Filter to long signals only
    if args.long_only:
        df = df[df["trade_long"] == True].copy()
        print(f"  Long signals: {len(df)}")

    # Get unique symbol/year/quarter combinations
    combos = df[["symbol", "year", "quarter"]].drop_duplicates()
    combos_list = [(row["symbol"], row["year"], row["quarter"]) for _, row in combos.iterrows()]
    print(f"  Fetching reaction dates for {len(combos_list)} unique earnings events...")

    # Batch fetch reaction dates
    reaction_dates = get_reaction_dates_batch(combos_list)
    print(f"  Found {len(reaction_dates)} reaction dates in database")

    # Add reaction_date column
    def get_reaction_date(row):
        key = (row["symbol"].upper(), int(row["year"]), int(row["quarter"]))
        return reaction_dates.get(key)

    df["reaction_date"] = df.apply(get_reaction_date, axis=1)

    # Check for missing dates
    missing = df["reaction_date"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} rows missing reaction_date")
        # Show which ones are missing
        missing_df = df[df["reaction_date"].isna()][["symbol", "year", "quarter"]].drop_duplicates()
        print(f"  Missing: {missing_df.to_dict('records')[:10]}...")

    # Drop rows without reaction_date
    df = df.dropna(subset=["reaction_date"])
    print(f"  Final rows with reaction_date: {len(df)}")

    # Select and order columns for backtester
    output_cols = ["symbol", "reaction_date", "trade_long"]
    if "trade_long_tier" in df.columns:
        output_cols.append("trade_long_tier")
    if "confidence" in df.columns:
        output_cols.append("confidence")
    if "direction_score" in df.columns:
        output_cols.append("direction_score")
    if "actual_return" in df.columns:
        output_cols.append("actual_return")

    signals = df[output_cols].copy()

    # Sort by date
    signals = signals.sort_values("reaction_date")

    # Save
    signals.to_csv(args.output, index=False)
    print(f"\nSaved signals to {args.output}")
    print(f"  Total signals: {len(signals)}")

    # Summary stats
    if "trade_long_tier" in signals.columns:
        tier_counts = signals["trade_long_tier"].value_counts()
        print(f"\n  Tier breakdown:")
        for tier, count in tier_counts.items():
            print(f"    {tier}: {count}")


if __name__ == "__main__":
    main()
