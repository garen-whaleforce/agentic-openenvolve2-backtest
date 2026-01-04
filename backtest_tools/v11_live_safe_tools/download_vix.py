#!/usr/bin/env python3
"""
download_vix.py

Download VIX historical data from Yahoo Finance using yfinance.

Usage:
    python download_vix.py --output vix_data.csv --start 2017-01-01 --end 2025-12-31
"""
import argparse
from datetime import datetime
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("Please install yfinance: pip install yfinance")
    exit(1)

import pandas as pd


def download_vix(start: str, end: str, output: Path):
    """Download VIX data from Yahoo Finance"""
    print(f"Downloading VIX data from {start} to {end}...")

    # VIX ticker on Yahoo Finance
    vix = yf.Ticker("^VIX")
    df = vix.history(start=start, end=end)

    if df.empty:
        raise ValueError("No VIX data downloaded")

    # Normalize
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Keep only date and close
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
        df = df.drop(columns=["datetime"])

    # Select columns
    out_cols = ["date"]
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            out_cols.append(c)

    df = df[out_cols]

    # Save
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"VIX stats: mean={df['close'].mean():.2f}, max={df['close'].max():.2f}, min={df['close'].min():.2f}")

    # Show some high VIX dates
    high_vix = df[df["close"] > 30].copy()
    if len(high_vix) > 0:
        print(f"\nHigh VIX (>30) dates: {len(high_vix)} days")
        print(high_vix.head(20).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("vix_data.csv"))
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    download_vix(args.start, args.end, args.output)


if __name__ == "__main__":
    main()
