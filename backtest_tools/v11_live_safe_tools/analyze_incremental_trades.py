#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_incremental_trades.py

分析增量交易的品質：比較 baseline (CAP=12) vs 解耦 (CAP=24, Sizing=12)
找出哪些交易是「新增」的，並評估其表現

用途：
1. 確認增量交易是否有正期望值
2. 決定是否需要衛星倉門檻 (satellite_floor)
3. 找出最佳的 satellite_floor 閾值
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


def run_and_get_trades(
    signals: pd.DataFrame,
    price_provider: CSVPriceProvider,
    vix_data: Optional[pd.Series],
    cap: int,
    max_pos: int,
    sizing_pos: Optional[int] = None,
    sort_column: Optional[str] = None,
    satellite_floor: Optional[float] = None,
) -> Tuple[pd.DataFrame, dict]:
    """執行回測並回傳交易記錄"""

    config = BacktestConfigV32(
        cap_entries_per_quarter=cap,
        max_concurrent_positions=max_pos,
        sizing_positions=sizing_pos,
        target_gross_normal=2.0,
        target_gross_riskoff=1.0,
        target_gross_stress=0.0,
        per_trade_cap=0.20,
        commission_bps=5,
        slippage_bps=5,
        annual_borrow_rate=0.06,
        sort_column=sort_column,
        satellite_floor=satellite_floor,
    )

    nav, trades, exposure, stats = run_backtest_v32(
        signals=signals,
        price_provider=price_provider,
        config=config,
        vix_data=vix_data,
    )

    # 計算指標
    start_nav = 100000.0
    end_nav = float(nav.iloc[-1])
    n_days = len(nav)
    years = n_days / 252
    cagr = (end_nav / start_nav) ** (1.0 / years) - 1.0

    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    mdd = float(dd.min())

    metrics = {
        "cagr": cagr,
        "mdd": mdd,
        "avg_exposure": float(exposure.mean()),
        "n_trades": len(trades),
    }

    return trades, metrics


def analyze_incremental(
    baseline_trades: pd.DataFrame,
    extended_trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    找出增量交易並分析其表現

    增量交易 = 在 extended 中成交但在 baseline 中未成交的交易
    """
    # 建立交易識別碼 (symbol + reaction_date)
    baseline_trades["trade_id"] = (
        baseline_trades["symbol"].astype(str) + "_" +
        baseline_trades["reaction_date"].astype(str)
    )
    extended_trades["trade_id"] = (
        extended_trades["symbol"].astype(str) + "_" +
        extended_trades["reaction_date"].astype(str)
    )

    baseline_ids = set(baseline_trades["trade_id"])
    extended_ids = set(extended_trades["trade_id"])

    # 增量交易 = 在 extended 但不在 baseline
    incremental_ids = extended_ids - baseline_ids
    incremental_trades = extended_trades[
        extended_trades["trade_id"].isin(incremental_ids)
    ].copy()

    # 共同交易
    common_ids = baseline_ids & extended_ids
    common_baseline = baseline_trades[baseline_trades["trade_id"].isin(common_ids)].copy()
    common_extended = extended_trades[extended_trades["trade_id"].isin(common_ids)].copy()

    return incremental_trades, common_baseline, common_extended


def print_comparison(
    baseline_trades: pd.DataFrame,
    extended_trades: pd.DataFrame,
    incremental_trades: pd.DataFrame,
    common_baseline: pd.DataFrame,
    common_extended: pd.DataFrame,
):
    """印出比較報告"""

    print("\n" + "=" * 80)
    print("增量交易歸因分析")
    print("=" * 80)

    print(f"\n交易數比較:")
    print(f"  Baseline 交易數:    {len(baseline_trades)}")
    print(f"  Extended 交易數:    {len(extended_trades)}")
    print(f"  增量交易數:         {len(incremental_trades)} (+{len(incremental_trades)})")
    print(f"  共同交易數:         {len(common_baseline)}")

    print("\n" + "-" * 80)
    print("報酬比較")
    print("-" * 80)

    def calc_stats(trades: pd.DataFrame, label: str):
        if trades.empty:
            print(f"  {label}: 無交易")
            return

        avg_ret = trades["net_ret"].mean() * 100
        med_ret = trades["net_ret"].median() * 100
        win_rate = (trades["net_ret"] > 0).mean() * 100
        avg_win = trades[trades["net_ret"] > 0]["net_ret"].mean() * 100 if (trades["net_ret"] > 0).any() else 0
        avg_loss = trades[trades["net_ret"] <= 0]["net_ret"].mean() * 100 if (trades["net_ret"] <= 0).any() else 0

        print(f"\n  {label}:")
        print(f"    平均報酬:     {avg_ret:+.2f}%")
        print(f"    中位數報酬:   {med_ret:+.2f}%")
        print(f"    勝率:         {win_rate:.1f}%")
        print(f"    平均獲利:     {avg_win:+.2f}%")
        print(f"    平均虧損:     {avg_loss:+.2f}%")

    calc_stats(baseline_trades, "Baseline (CAP=12)")
    calc_stats(extended_trades, "Extended (CAP=24, Sizing=12)")
    calc_stats(incremental_trades, "增量交易 (新增)")
    calc_stats(common_extended, "共同交易 (Extended 版本)")

    # 增量交易是否有正期望值
    if not incremental_trades.empty:
        incr_avg = incremental_trades["net_ret"].mean()
        baseline_avg = baseline_trades["net_ret"].mean()

        print("\n" + "-" * 80)
        print("診斷結論")
        print("-" * 80)

        if incr_avg > 0:
            print(f"\n  增量交易平均報酬 {incr_avg*100:+.2f}% > 0")
            print("  ✓ 增量交易有正期望值，CAP=24 有效")
        else:
            print(f"\n  增量交易平均報酬 {incr_avg*100:+.2f}% <= 0")
            print("  ✗ 增量交易可能稀釋整體表現")
            print("  建議：啟用 satellite_floor 過濾低品質增量交易")

        if incr_avg < baseline_avg:
            print(f"\n  增量交易 ({incr_avg*100:+.2f}%) < Baseline ({baseline_avg*100:+.2f}%)")
            print("  增量交易品質較低，建議使用衛星倉門檻")


def main():
    parser = argparse.ArgumentParser(description="增量交易歸因分析")
    parser.add_argument("--signals", required=True, help="訊號檔案路徑")
    parser.add_argument("--vix", default="vix_data.csv", help="VIX 資料檔案路徑")
    parser.add_argument("--prices-folder", default="../prices", help="價格資料夾路徑")
    parser.add_argument("--sort-column", default=None, help="排序欄位 (如 direction_score)")
    args = parser.parse_args()

    # 載入訊號
    signals_path = Path(args.signals)
    if not signals_path.exists():
        print(f"錯誤: 找不到訊號檔案: {signals_path}")
        return

    print(f"載入訊號: {signals_path}")
    signals = pd.read_csv(signals_path)

    # 只保留 trade_long=True
    if "trade_long" in signals.columns:
        signals = signals[signals["trade_long"] == True].copy()

    # 過濾掉 2024-11 之後的訊號
    if "reaction_date" in signals.columns:
        signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])
        cutoff = pd.Timestamp("2024-11-01")
        signals = signals[signals["reaction_date"] < cutoff].copy()

    print(f"  多頭訊號數: {len(signals)}")

    # 載入 VIX
    vix_path = Path(args.vix)
    vix_data = None
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path)
        date_col = "Date" if "Date" in vix_df.columns else "date"
        close_col = "Close" if "Close" in vix_df.columns else "close"
        vix_df[date_col] = pd.to_datetime(vix_df[date_col])
        vix_df = vix_df.set_index(date_col)[close_col]
        vix_df.index = pd.to_datetime(vix_df.index).normalize()
        vix_data = vix_df

    price_provider = CSVPriceProvider(folder=Path(args.prices_folder))

    # 執行 Baseline (CAP=12, MaxPos=12, Sizing=12)
    print("\n執行 Baseline (CAP=12, MaxPos=12)...")
    baseline_trades, baseline_metrics = run_and_get_trades(
        signals=signals,
        price_provider=price_provider,
        vix_data=vix_data,
        cap=12,
        max_pos=12,
        sizing_pos=None,
        sort_column=args.sort_column,
    )
    print(f"  CAGR: {baseline_metrics['cagr']*100:.2f}%, 交易數: {baseline_metrics['n_trades']}")

    # 執行 Extended (CAP=24, MaxPos=24, Sizing=12)
    print("\n執行 Extended (CAP=24, MaxPos=24, Sizing=12)...")
    extended_trades, extended_metrics = run_and_get_trades(
        signals=signals,
        price_provider=price_provider,
        vix_data=vix_data,
        cap=24,
        max_pos=24,
        sizing_pos=12,
        sort_column=args.sort_column,
    )
    print(f"  CAGR: {extended_metrics['cagr']*100:.2f}%, 交易數: {extended_metrics['n_trades']}")

    # 分析增量交易
    incremental, common_base, common_ext = analyze_incremental(baseline_trades, extended_trades)

    # 印出報告
    print_comparison(baseline_trades, extended_trades, incremental, common_base, common_ext)

    # 儲存增量交易明細
    if not incremental.empty:
        output_path = Path("incremental_trades.csv")
        incremental.to_csv(output_path, index=False)
        print(f"\n增量交易明細已儲存至: {output_path}")


if __name__ == "__main__":
    main()
