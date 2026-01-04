#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_yearly_dashboard.py

完整年度歸因儀表板：
- 從 nav.csv 計算年度績效
- 從 exposure.csv 計算年度平均曝險
- 從 trades.csv 計算年度交易統計與 D6 占比
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TRADING_DAYS = 252


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", required=True, help="nav.csv 路徑")
    ap.add_argument("--exposure", default=None, help="exposure.csv 路徑（可選）")
    ap.add_argument("--trades", default=None, help="trades.csv 路徑（可選）")
    ap.add_argument("--out-csv", default="yearly_dashboard.csv")
    ap.add_argument("--out-md", default="yearly_dashboard.md")
    args = ap.parse_args()

    # 讀取 NAV
    nav_df = pd.read_csv(args.nav)
    date_col = [c for c in nav_df.columns if "date" in c.lower()][0]
    nav_col = [c for c in nav_df.columns if c.lower() in ["nav", "equity", "value"]][0]
    nav_df[date_col] = pd.to_datetime(nav_df[date_col])
    nav_df = nav_df.sort_values(date_col).drop_duplicates(subset=[date_col])
    nav_df["ret"] = nav_df[nav_col].pct_change()
    nav_df["year"] = nav_df[date_col].dt.year

    # 讀取 Exposure（可選）
    exposure_df = None
    if args.exposure and Path(args.exposure).exists():
        exposure_df = pd.read_csv(args.exposure)
        exp_date_col = [c for c in exposure_df.columns if "date" in c.lower()][0]
        exp_val_col = [c for c in exposure_df.columns if c.lower() in ["exposure", "gross", "gross_exposure"]][0]
        exposure_df[exp_date_col] = pd.to_datetime(exposure_df[exp_date_col])
        exposure_df["year"] = exposure_df[exp_date_col].dt.year

    # 讀取 Trades（可選）
    trades_df = None
    if args.trades and Path(args.trades).exists():
        trades_df = pd.read_csv(args.trades)
        if "entry_date" in trades_df.columns:
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
            trades_df["year"] = trades_df["entry_date"].dt.year

    # 計算年度統計
    rows = []
    for y, g in nav_df.groupby("year"):
        g = g.sort_values(date_col)
        start_nav = float(g[nav_col].iloc[0])
        end_nav = float(g[nav_col].iloc[-1])
        year_return = end_nav / start_nav - 1.0
        n_days = len(g)

        vol = float(g["ret"].dropna().std(ddof=1) * np.sqrt(TRADING_DAYS)) if n_days > 2 else np.nan
        sh = sharpe_ratio(g["ret"].dropna()) if n_days > 2 else np.nan
        mdd = max_drawdown(g[nav_col])
        arr = float((1 + year_return) ** (TRADING_DAYS / n_days) - 1.0) if n_days > 0 else np.nan

        row = {
            "year": int(y),
            "total_return": year_return,
            "arr_annualized": arr,
            "ann_vol": vol,
            "sharpe": sh,
            "mdd": mdd,
            "trading_days": n_days,
        }

        # 年度曝險
        if exposure_df is not None:
            exp_year = exposure_df[exposure_df["year"] == y]
            if len(exp_year) > 0:
                row["avg_exposure"] = float(exp_year[exp_val_col].mean())
            else:
                row["avg_exposure"] = np.nan
        else:
            row["avg_exposure"] = np.nan

        # 年度交易統計
        if trades_df is not None:
            trades_year = trades_df[trades_df["year"] == y]
            row["n_trades"] = len(trades_year)
            if len(trades_year) > 0 and "net_ret" in trades_year.columns:
                row["win_rate"] = float((trades_year["net_ret"] > 0).mean())
                row["avg_trade_ret"] = float(trades_year["net_ret"].mean())
            else:
                row["win_rate"] = np.nan
                row["avg_trade_ret"] = np.nan

            # D6 占比（如果有 tier 資訊）
            if "trade_long_tier" in trades_year.columns:
                d6_count = len(trades_year[trades_year["trade_long_tier"] == "D6_STRICT"])
                row["d6_trade_pct"] = d6_count / len(trades_year) if len(trades_year) > 0 else 0
            elif "is_satellite" in trades_year.columns:
                sat_count = trades_year["is_satellite"].sum()
                row["d6_trade_pct"] = sat_count / len(trades_year) if len(trades_year) > 0 else 0
            else:
                row["d6_trade_pct"] = np.nan
        else:
            row["n_trades"] = np.nan
            row["win_rate"] = np.nan
            row["avg_trade_ret"] = np.nan
            row["d6_trade_pct"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("year")

    # Overall 統計
    nav0 = float(nav_df[nav_col].iloc[0])
    nav1 = float(nav_df[nav_col].iloc[-1])
    total = nav1 / nav0 - 1.0
    n_all = len(nav_df)
    cagr = float((1.0 + total) ** (TRADING_DAYS / n_all) - 1.0)
    vol_all = float(nav_df["ret"].dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))
    sh_all = sharpe_ratio(nav_df["ret"].dropna())
    mdd_all = max_drawdown(nav_df[nav_col])

    # 輸出 CSV
    out.to_csv(args.out_csv, index=False)

    # 輸出 Markdown
    md = []
    md.append("# Yearly Performance Dashboard\n")

    def pct(x):
        return "—" if pd.isna(x) else f"{x*100:.2f}%"

    def num(x, decimals=2):
        return "—" if pd.isna(x) else f"{x:.{decimals}f}"

    # 主表
    md.append("## Performance by Year\n")
    md.append("| Year | Total Return | ARR | Vol | Sharpe | MDD | Exposure | Trades | Win Rate | Avg Trade | D6% |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in out.iterrows():
        year_label = f"{int(r['year'])}"
        if r["trading_days"] < 250:
            year_label += " (YTD)"
        md.append(
            f"| {year_label} | {pct(r['total_return'])} | {pct(r['arr_annualized'])} | "
            f"{pct(r['ann_vol'])} | {num(r['sharpe'])} | {pct(r['mdd'])} | "
            f"{pct(r.get('avg_exposure', np.nan))} | {int(r.get('n_trades', 0)) if not pd.isna(r.get('n_trades')) else '—'} | "
            f"{pct(r.get('win_rate', np.nan))} | {pct(r.get('avg_trade_ret', np.nan))} | "
            f"{pct(r.get('d6_trade_pct', np.nan))} |"
        )

    # Overall
    md.append("\n## Overall Statistics\n")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Total Return | {pct(total)} |")
    md.append(f"| CAGR | {pct(cagr)} |")
    md.append(f"| Annual Volatility | {pct(vol_all)} |")
    md.append(f"| Sharpe Ratio | {num(sh_all)} |")
    md.append(f"| Maximum Drawdown | {pct(mdd_all)} |")
    md.append(f"| Trading Days | {n_all} |")

    if trades_df is not None:
        total_trades = len(trades_df)
        total_win_rate = float((trades_df["net_ret"] > 0).mean()) if "net_ret" in trades_df.columns else np.nan
        total_avg_ret = float(trades_df["net_ret"].mean()) if "net_ret" in trades_df.columns else np.nan
        md.append(f"| Total Trades | {total_trades} |")
        md.append(f"| Overall Win Rate | {pct(total_win_rate)} |")
        md.append(f"| Avg Trade Return | {pct(total_avg_ret)} |")

    if exposure_df is not None:
        avg_exp = float(exposure_df[exp_val_col].mean())
        md.append(f"| Avg Gross Exposure | {pct(avg_exp)} |")

    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")


if __name__ == "__main__":
    main()
