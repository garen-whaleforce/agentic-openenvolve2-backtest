#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_yearly_performance.py

從 nav.csv 計算每年度績效指標：
- Total Return（年度總報酬率）
- ARR（年化報酬率）
- 年化波動率
- Sharpe
- MDD（年度內最大回撤）
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def infer_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "datetime", "timestamp", "time"]:
        if c in df.columns:
            return c
    return df.columns[0]


def infer_nav_col(df: pd.DataFrame) -> str:
    for c in ["nav", "equity", "portfolio_value", "total_value", "value"]:
        if c in df.columns:
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("nav.csv 找不到 NAV 數值欄位（nav/equity/value 等）")
    return num_cols[-1]


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def sharpe_ratio(daily_ret: pd.Series, rf_annual: float = 0.0) -> float:
    rf_daily = rf_annual / TRADING_DAYS
    excess = daily_ret - rf_daily
    mu = excess.mean()
    sigma = daily_ret.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * mu / sigma)


def annualized_return(total_return: float, n_days: int) -> float:
    if n_days <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (TRADING_DAYS / n_days) - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", required=True, help="nav.csv 路徑（包含 date, nav）")
    ap.add_argument("--out-csv", default="yearly_performance.csv")
    ap.add_argument("--out-md", default="yearly_performance.md")
    ap.add_argument("--rf-annual", type=float, default=0.0, help="Sharpe 用的年化無風險利率（預設 0）")
    args = ap.parse_args()

    df = pd.read_csv(args.nav)
    date_col = infer_date_col(df)
    nav_col = infer_nav_col(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[nav_col] = pd.to_numeric(df[nav_col], errors="coerce")
    df = df.dropna(subset=[date_col, nav_col]).sort_values(date_col).drop_duplicates(subset=[date_col])

    df["ret"] = df[nav_col].pct_change()
    df = df.dropna(subset=["ret"])

    df["year"] = df[date_col].dt.year

    rows = []
    for y, g in df.groupby("year"):
        g = g.sort_values(date_col)

        start_nav = float(g[nav_col].iloc[0])
        end_nav = float(g[nav_col].iloc[-1])
        year_return = end_nav / start_nav - 1.0
        n_days = int(len(g["ret"]))

        vol = float(g["ret"].std(ddof=1) * np.sqrt(TRADING_DAYS)) if n_days > 2 else float("nan")
        sh = sharpe_ratio(g["ret"], rf_annual=args.rf_annual) if n_days > 2 else float("nan")
        mdd = max_drawdown(g[nav_col])
        arr = annualized_return(year_return, n_days)

        rows.append({
            "year": int(y),
            "total_return": year_return,
            "arr_annualized": arr,
            "ann_vol": vol,
            "sharpe": sh,
            "mdd": mdd,
            "trading_days": n_days,
            "start_nav": start_nav,
            "end_nav": end_nav,
        })

    out = pd.DataFrame(rows).sort_values("year")

    # Overall（全期間）
    nav0 = float(df[nav_col].iloc[0])
    nav1 = float(df[nav_col].iloc[-1])
    total = nav1 / nav0 - 1.0
    n_all = int(len(df["ret"]))
    cagr = float((1.0 + total) ** (TRADING_DAYS / n_all) - 1.0)
    vol_all = float(df["ret"].std(ddof=1) * np.sqrt(TRADING_DAYS))
    sh_all = sharpe_ratio(df["ret"], rf_annual=args.rf_annual)
    mdd_all = max_drawdown(df[nav_col])

    overall_row = pd.DataFrame([{
        "year": "OVERALL",
        "total_return": total,
        "arr_annualized": cagr,
        "ann_vol": vol_all,
        "sharpe": sh_all,
        "mdd": mdd_all,
        "trading_days": n_all,
        "start_nav": nav0,
        "end_nav": nav1,
    }])

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    out.to_csv(out_csv, index=False)

    # Markdown 輸出
    md = []
    md.append("# Yearly Performance\n")
    md.append("| Year | Total Return | ARR (Annualized) | Ann. Vol | Sharpe | MDD | Trading Days |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")

    def pct(x):
        return "—" if pd.isna(x) else f"{x*100:.2f}%"

    def num(x):
        return "—" if pd.isna(x) else f"{x:.2f}"

    for _, r in out.iterrows():
        md.append(
            f"| {r['year']} | {pct(r['total_return'])} | {pct(r['arr_annualized'])} | {pct(r['ann_vol'])} | {num(r['sharpe'])} | {pct(r['mdd'])} | {int(r['trading_days'])} |"
        )

    md.append("\n## Overall\n")
    r = overall_row.iloc[0]
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Total Return | {pct(r['total_return'])} |")
    md.append(f"| CAGR (ARR) | {pct(r['arr_annualized'])} |")
    md.append(f"| Annual Vol | {pct(r['ann_vol'])} |")
    md.append(f"| Sharpe | {num(r['sharpe'])} |")
    md.append(f"| Max Drawdown | {pct(r['mdd'])} |")
    md.append(f"| Trading Days | {int(r['trading_days'])} |")

    out_md.write_text("\n".join(md), encoding="utf-8")

    print("Wrote:", str(out_csv))
    print("Wrote:", str(out_md))


if __name__ == "__main__":
    main()
