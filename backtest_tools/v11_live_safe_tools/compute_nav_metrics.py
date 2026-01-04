#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_nav_metrics.py

後處理腳本：從 nav.csv 計算：
- 總報酬率 (Total Return)
- 年化波動率 (Annual Volatility)
- 各年報酬 (Calendar Year Returns)
- 各年波動率
- 各年 MDD

不改動 backtester 邏輯，nav 已內含槓桿、借款利息、止損、成本等全部效果。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _infer_date_col(df: pd.DataFrame) -> str:
    """自動偵測日期欄位"""
    for c in ["date", "datetime", "timestamp", "time", "Date"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _infer_nav_col(df: pd.DataFrame) -> str:
    """自動偵測 NAV 欄位"""
    for c in ["nav", "equity", "portfolio_value", "total_value", "value", "NAV"]:
        if c in df.columns:
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("Cannot find numeric NAV column in nav.csv")
    return num_cols[-1]


def compute_nav_metrics(nav_csv: str, initial_cash: float = 100000.0) -> dict:
    """
    從 nav.csv 計算全部績效指標

    Returns:
        dict with total_return, cagr, annual_vol, sharpe, max_drawdown,
        annual_returns_table (DataFrame), etc.
    """
    df = pd.read_csv(nav_csv)

    date_col = _infer_date_col(df)
    nav_col = _infer_nav_col(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col])
    df[nav_col] = pd.to_numeric(df[nav_col], errors="coerce")
    df = df.dropna(subset=[nav_col])

    if len(df) < 5:
        raise ValueError("nav.csv too short to compute metrics")

    # Daily returns
    df["ret"] = df[nav_col].pct_change()
    df = df.dropna(subset=["ret"])

    # Use initial_cash as true start (first nav row is after first day)
    start_nav = initial_cash
    end_nav = float(df[nav_col].iloc[-1])
    total_return = end_nav / start_nav - 1.0

    # Annualized volatility
    ann_vol = float(df["ret"].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    # CAGR
    n_days = len(df["ret"])
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = float((end_nav / start_nav) ** (1.0 / years) - 1.0) if years > 0 else np.nan

    # Sharpe (assuming rf=0)
    sharpe = float(df["ret"].mean() / df["ret"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if df["ret"].std() > 0 else 0.0

    # Max Drawdown
    df["cummax"] = df[nav_col].cummax()
    df["drawdown"] = (df[nav_col] - df["cummax"]) / df["cummax"]
    max_dd = float(df["drawdown"].min())

    # Sortino (downside deviation)
    neg_rets = df["ret"][df["ret"] < 0]
    downside_std = float(neg_rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(neg_rets) > 1 else 0.0
    sortino = cagr / downside_std if downside_std > 0 else 0.0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Calendar year returns with volatility and MDD per year
    df["year"] = df[date_col].dt.year
    year_rows = []

    for y, g in df.groupby("year"):
        g = g.sort_values(date_col)

        # Year return: first nav to last nav of that year
        y_start = float(g[nav_col].iloc[0])
        y_end = float(g[nav_col].iloc[-1])
        y_ret = y_end / y_start - 1.0

        # Year volatility
        y_vol = float(g["ret"].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(g) > 2 else np.nan

        # Year MDD
        y_cummax = g[nav_col].cummax()
        y_dd = (g[nav_col] - y_cummax) / y_cummax
        y_mdd = float(y_dd.min())

        # Trading days in this year
        y_days = len(g)

        year_rows.append({
            "Year": int(y),
            "Return": y_ret,
            "Volatility": y_vol,
            "MDD": y_mdd,
            "Trading Days": y_days,
        })

    year_df = pd.DataFrame(year_rows).sort_values("Year")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "start_nav": start_nav,
        "end_nav": end_nav,
        "n_trading_days": int(n_days),
        "n_years": float(years),
        "annual_returns": {int(r["Year"]): float(r["Return"]) for r in year_rows},
        "annual_returns_table": year_df,
        "worst_year": year_df.loc[year_df["Return"].idxmin(), "Year"] if len(year_df) > 0 else None,
        "worst_year_return": float(year_df["Return"].min()) if len(year_df) > 0 else None,
        "best_year": year_df.loc[year_df["Return"].idxmax(), "Year"] if len(year_df) > 0 else None,
        "best_year_return": float(year_df["Return"].max()) if len(year_df) > 0 else None,
        "positive_years": int((year_df["Return"] > 0).sum()),
        "negative_years": int((year_df["Return"] <= 0).sum()),
    }


def print_summary(m: dict) -> None:
    """印出摘要報表"""
    print("\n" + "=" * 70)
    print("NAV METRICS SUMMARY")
    print("=" * 70)

    print(f"\n{'Overall Performance':^70}")
    print("-" * 70)
    print(f"  Total Return:     {m['total_return']*100:>10.2f}%")
    print(f"  CAGR:             {m['cagr']*100:>10.2f}%")
    print(f"  Annual Vol:       {m['annual_vol']*100:>10.2f}%")
    print(f"  Sharpe:           {m['sharpe']:>10.2f}")
    print(f"  Sortino:          {m['sortino']:>10.2f}")
    print(f"  Max Drawdown:     {m['max_drawdown']*100:>10.2f}%")
    print(f"  Calmar:           {m['calmar']:>10.2f}")

    print(f"\n{'Period Info':^70}")
    print("-" * 70)
    print(f"  Start NAV:        ${m['start_nav']:>12,.2f}")
    print(f"  End NAV:          ${m['end_nav']:>12,.2f}")
    print(f"  Trading Days:     {m['n_trading_days']:>10}")
    print(f"  Years:            {m['n_years']:>10.2f}")

    print(f"\n{'Year Analysis':^70}")
    print("-" * 70)
    print(f"  Positive Years:   {m['positive_years']:>10}")
    print(f"  Negative Years:   {m['negative_years']:>10}")
    print(f"  Best Year:        {m['best_year']} ({m['best_year_return']*100:+.1f}%)")
    print(f"  Worst Year:       {m['worst_year']} ({m['worst_year_return']*100:+.1f}%)")

    print(f"\n{'Annual Returns Table':^70}")
    print("-" * 70)
    year_df = m["annual_returns_table"].copy()
    year_df["Return"] = year_df["Return"].apply(lambda x: f"{x*100:+.2f}%")
    year_df["Volatility"] = year_df["Volatility"].apply(lambda x: f"{x*100:.2f}%")
    year_df["MDD"] = year_df["MDD"].apply(lambda x: f"{x*100:.2f}%")
    print(year_df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute NAV metrics from nav.csv")
    parser.add_argument("--nav", required=True, help="Path to nav.csv")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash (default: 100000)")
    parser.add_argument("--base-metrics", default=None, help="Optional existing metrics.json to enrich")
    parser.add_argument("--out-json", default=None, help="Output JSON path (default: same dir as nav.csv)")
    parser.add_argument("--out-annual-csv", default=None, help="Output annual returns CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress printed output")
    args = parser.parse_args()

    nav_path = Path(args.nav)
    if not nav_path.exists():
        print(f"Error: nav.csv not found: {nav_path}")
        return

    # Compute metrics
    m = compute_nav_metrics(str(nav_path), initial_cash=args.initial_cash)

    # Default output paths
    out_dir = nav_path.parent
    out_json = Path(args.out_json) if args.out_json else out_dir / "metrics_enriched.json"
    out_annual_csv = Path(args.out_annual_csv) if args.out_annual_csv else out_dir / "annual_returns.csv"

    # Merge with existing metrics.json if provided
    out_dict = {}
    if args.base_metrics:
        base_path = Path(args.base_metrics)
        if base_path.exists():
            out_dict = json.loads(base_path.read_text())

    # Update with computed metrics
    out_dict.update({
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "annual_vol": m["annual_vol"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "max_drawdown": m["max_drawdown"],
        "calmar": m["calmar"],
        "start_nav": m["start_nav"],
        "end_nav": m["end_nav"],
        "n_trading_days": m["n_trading_days"],
        "n_years": m["n_years"],
        "annual_returns": m["annual_returns"],
        "worst_year": m["worst_year"],
        "worst_year_return": m["worst_year_return"],
        "best_year": m["best_year"],
        "best_year_return": m["best_year_return"],
        "positive_years": m["positive_years"],
        "negative_years": m["negative_years"],
    })

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_dict = convert_for_json(out_dict)

    # Write outputs
    out_json.write_text(json.dumps(out_dict, indent=2, sort_keys=True), encoding="utf-8")
    m["annual_returns_table"].to_csv(out_annual_csv, index=False)

    if not args.quiet:
        print_summary(m)
        print(f"Wrote: {out_json}")
        print(f"Wrote: {out_annual_csv}")


if __name__ == "__main__":
    main()
