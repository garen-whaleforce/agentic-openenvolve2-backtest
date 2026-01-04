#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_satellite_floor_tests.py

執行衛星門檻 (satellite_floor) 配置測試
使用 earnings_day_return 作為排序欄位

測試目標：找到最佳配置來最大化 CAGR 同時控制風險
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


def run_config(
    signals: pd.DataFrame,
    price_provider: CSVPriceProvider,
    vix_data,
    config: BacktestConfigV32,
    desc: str,
) -> dict:
    """執行單一配置並回傳結果"""
    nav, trades, exposure, stats = run_backtest_v32(
        signals=signals,
        price_provider=price_provider,
        config=config,
        vix_data=vix_data,
    )

    # 計算 CAGR
    start_nav = 100000.0
    end_nav = float(nav.iloc[-1])
    n_days = len(nav)
    years = n_days / 252
    cagr = (end_nav / start_nav) ** (1.0 / years) - 1.0

    # MDD
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    mdd = float(dd.min())

    # 平均曝險
    avg_exposure = float(exposure.mean())

    # 交易統計
    n_trades = len(trades)
    win_rate = float((trades["net_ret"] > 0).mean()) if n_trades > 0 else np.nan
    avg_return = float(trades["net_ret"].mean()) if n_trades > 0 else np.nan

    # 衛星交易統計
    if "is_satellite" in trades.columns:
        sat_trades = trades[trades["is_satellite"] == True]
        n_satellite = len(sat_trades)
        sat_win_rate = float((sat_trades["net_ret"] > 0).mean()) if n_satellite > 0 else np.nan
        sat_avg_return = float(sat_trades["net_ret"].mean()) if n_satellite > 0 else np.nan
    else:
        n_satellite = 0
        sat_win_rate = np.nan
        sat_avg_return = np.nan

    # Sharpe ratio
    daily_returns = nav.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 else np.nan

    return {
        "description": desc,
        "cap": config.cap_entries_per_quarter,
        "max_pos": config.max_concurrent_positions,
        "sizing_pos": config.get_sizing_positions(),
        "sort_col": config.sort_column or "None",
        "sat_floor": config.satellite_floor if config.satellite_floor else "None",
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "avg_exposure": avg_exposure,
        "n_trades": n_trades,
        "n_satellite": n_satellite,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "sat_win_rate": sat_win_rate,
        "sat_avg_return": sat_avg_return,
        "end_nav": end_nav,
    }


def main():
    # 載入訊號
    signals_path = Path("../long_only_signals_2017_2025_final.csv")
    if not signals_path.exists():
        print(f"錯誤: 找不到訊號檔案: {signals_path}")
        return

    print(f"載入訊號: {signals_path}")
    signals = pd.read_csv(signals_path)

    # 只保留 D7_CORE
    signals = signals[signals["trade_long_tier"] == "D7_CORE"].copy()
    print(f"D7_CORE 訊號數: {len(signals)}")

    # 過濾日期範圍（避免價格資料缺失）
    signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])
    cutoff = pd.Timestamp("2024-11-01")
    signals = signals[signals["reaction_date"] < cutoff].copy()
    print(f"過濾 2024-11 前: {len(signals)} 筆")

    # 確保有 trade_long 欄位
    if "trade_long" not in signals.columns:
        signals["trade_long"] = True

    # 載入 VIX
    vix_path = Path("vix_data.csv")
    vix_data = None
    if vix_path.exists():
        print(f"載入 VIX 資料: {vix_path}")
        vix_df = pd.read_csv(vix_path)
        date_col = "Date" if "Date" in vix_df.columns else "date"
        close_col = "Close" if "Close" in vix_df.columns else "close"
        vix_df[date_col] = pd.to_datetime(vix_df[date_col])
        vix_df = vix_df.set_index(date_col)[close_col]
        vix_df.index = pd.to_datetime(vix_df.index).normalize()
        vix_data = vix_df

    # 建立 price provider
    price_provider = CSVPriceProvider(folder=Path("../prices"))

    # 測試配置
    print("\n" + "=" * 100)
    print("衛星門檻配置測試")
    print("=" * 100)

    configs = [
        # 1. Baseline: CAP=12, MaxPos=12, Sizing=12, 無排序
        (
            BacktestConfigV32(
                cap_entries_per_quarter=12,
                max_concurrent_positions=12,
                sizing_positions=None,
                target_gross_normal=2.0,
                target_gross_riskoff=1.0,
                target_gross_stress=0.0,
                per_trade_cap=0.20,
                commission_bps=5,
                slippage_bps=5,
                annual_borrow_rate=0.06,
            ),
            "Baseline CAP=12 MaxPos=12 無排序",
        ),
        # 2. CAP=12 + 排序 (earnings_day_return)
        (
            BacktestConfigV32(
                cap_entries_per_quarter=12,
                max_concurrent_positions=12,
                sizing_positions=None,
                sort_column="earnings_day_return",
                target_gross_normal=2.0,
                target_gross_riskoff=1.0,
                target_gross_stress=0.0,
                per_trade_cap=0.20,
                commission_bps=5,
                slippage_bps=5,
                annual_borrow_rate=0.06,
            ),
            "CAP=12 + 排序(earnings_day_return)",
        ),
        # 3. CAP=24, MaxPos=24, Sizing=12 (解耦) + 排序
        (
            BacktestConfigV32(
                cap_entries_per_quarter=24,
                max_concurrent_positions=24,
                sizing_positions=12,
                sort_column="earnings_day_return",
                target_gross_normal=2.0,
                target_gross_riskoff=1.0,
                target_gross_stress=0.0,
                per_trade_cap=0.20,
                commission_bps=5,
                slippage_bps=5,
                annual_borrow_rate=0.06,
            ),
            "CAP=24 Sizing=12 + 排序",
        ),
        # 4. CAP=24, Sizing=12, 排序 + 衛星門檻 3.0%
        (
            BacktestConfigV32(
                cap_entries_per_quarter=24,
                max_concurrent_positions=24,
                sizing_positions=12,
                sort_column="earnings_day_return",
                satellite_floor=3.0,
                target_gross_normal=2.0,
                target_gross_riskoff=1.0,
                target_gross_stress=0.0,
                per_trade_cap=0.20,
                commission_bps=5,
                slippage_bps=5,
                annual_borrow_rate=0.06,
            ),
            "CAP=24 Sizing=12 排序 Floor=3%",
        ),
        # 5. CAP=24, Sizing=12, 排序 + 衛星門檻 5.0%
        (
            BacktestConfigV32(
                cap_entries_per_quarter=24,
                max_concurrent_positions=24,
                sizing_positions=12,
                sort_column="earnings_day_return",
                satellite_floor=5.0,
                target_gross_normal=2.0,
                target_gross_riskoff=1.0,
                target_gross_stress=0.0,
                per_trade_cap=0.20,
                commission_bps=5,
                slippage_bps=5,
                annual_borrow_rate=0.06,
            ),
            "CAP=24 Sizing=12 排序 Floor=5%",
        ),
        # 6. CAP=24, Sizing=12, 排序 + 衛星門檻 7.0%
        (
            BacktestConfigV32(
                cap_entries_per_quarter=24,
                max_concurrent_positions=24,
                sizing_positions=12,
                sort_column="earnings_day_return",
                satellite_floor=7.0,
                target_gross_normal=2.0,
                target_gross_riskoff=1.0,
                target_gross_stress=0.0,
                per_trade_cap=0.20,
                commission_bps=5,
                slippage_bps=5,
                annual_borrow_rate=0.06,
            ),
            "CAP=24 Sizing=12 排序 Floor=7%",
        ),
        # 7. CAP=16, Sizing=12, 排序 + 衛星門檻 5.0% (微調)
        (
            BacktestConfigV32(
                cap_entries_per_quarter=16,
                max_concurrent_positions=16,
                sizing_positions=12,
                sort_column="earnings_day_return",
                satellite_floor=5.0,
                target_gross_normal=2.0,
                target_gross_riskoff=1.0,
                target_gross_stress=0.0,
                per_trade_cap=0.20,
                commission_bps=5,
                slippage_bps=5,
                annual_borrow_rate=0.06,
            ),
            "CAP=16 Sizing=12 排序 Floor=5%",
        ),
        # 8. CAP=20, Sizing=12, 排序 + 衛星門檻 4.0% (中間值)
        (
            BacktestConfigV32(
                cap_entries_per_quarter=20,
                max_concurrent_positions=20,
                sizing_positions=12,
                sort_column="earnings_day_return",
                satellite_floor=4.0,
                target_gross_normal=2.0,
                target_gross_riskoff=1.0,
                target_gross_stress=0.0,
                per_trade_cap=0.20,
                commission_bps=5,
                slippage_bps=5,
                annual_borrow_rate=0.06,
            ),
            "CAP=20 Sizing=12 排序 Floor=4%",
        ),
    ]

    results = []
    for config, desc in configs:
        print(f"\n測試: {desc}")
        try:
            r = run_config(
                signals=signals,
                price_provider=price_provider,
                vix_data=vix_data,
                config=config,
                desc=desc,
            )
            results.append(r)

            print(f"  CAGR:        {r['cagr']*100:>7.2f}%")
            print(f"  Sharpe:      {r['sharpe']:>7.2f}")
            print(f"  MDD:         {r['mdd']*100:>7.2f}%")
            print(f"  交易數:       {r['n_trades']:>5} (衛星: {r['n_satellite']})")
            print(f"  勝率:        {r['win_rate']*100:>7.1f}%")
            print(f"  平均報酬:     {r['avg_return']*100:>7.2f}%")
            if r['n_satellite'] > 0:
                print(f"  衛星勝率:     {r['sat_win_rate']*100:>7.1f}%")
                print(f"  衛星平均報酬: {r['sat_avg_return']*100:>7.2f}%")

        except Exception as e:
            print(f"  錯誤: {e}")
            import traceback
            traceback.print_exc()

    # 彙整報告
    print("\n" + "=" * 100)
    print("彙整比較")
    print("=" * 100)

    df = pd.DataFrame(results)

    # 格式化輸出
    print("\n| 配置 | CAP | MaxPos | Sizing | SortCol | Floor | CAGR | Sharpe | MDD | 交易數 | 衛星數 | 勝率 |")
    print("|------|-----|--------|--------|---------|-------|------|--------|-----|--------|--------|------|")

    for _, row in df.iterrows():
        floor_str = f"{row['sat_floor']}%" if row['sat_floor'] != "None" else "None"
        print(
            f"| {row['description'][:30]:30} | {row['cap']:>3} | {row['max_pos']:>6} | {row['sizing_pos']:>6} | "
            f"{row['sort_col']:>15} | {floor_str:>5} | {row['cagr']*100:.1f}% | {row['sharpe']:.2f} | "
            f"{row['mdd']*100:.1f}% | {row['n_trades']:>6} | {row['n_satellite']:>6} | {row['win_rate']*100:.1f}% |"
        )

    # 找出最佳配置
    print("\n" + "=" * 100)
    print("最佳配置分析")
    print("=" * 100)

    best_cagr = df.loc[df["cagr"].idxmax()]
    best_sharpe = df.loc[df["sharpe"].idxmax()]
    best_mdd = df.loc[df["mdd"].idxmax()]  # 最高 = 最小 MDD (更接近 0)

    print(f"\n最高 CAGR: {best_cagr['description']}")
    print(f"  CAGR: {best_cagr['cagr']*100:.2f}%, Sharpe: {best_cagr['sharpe']:.2f}, MDD: {best_cagr['mdd']*100:.1f}%")

    print(f"\n最高 Sharpe: {best_sharpe['description']}")
    print(f"  CAGR: {best_sharpe['cagr']*100:.2f}%, Sharpe: {best_sharpe['sharpe']:.2f}, MDD: {best_sharpe['mdd']*100:.1f}%")

    print(f"\n最小 MDD: {best_mdd['description']}")
    print(f"  CAGR: {best_mdd['cagr']*100:.2f}%, Sharpe: {best_mdd['sharpe']:.2f}, MDD: {best_mdd['mdd']*100:.1f}%")

    # Baseline vs Best 比較
    baseline = df[df["description"].str.contains("Baseline")].iloc[0]
    best = df.loc[df["cagr"].idxmax()]

    print("\n" + "-" * 50)
    print("Baseline vs Best CAGR 比較:")
    print("-" * 50)
    print(f"  CAGR:    {baseline['cagr']*100:.2f}% → {best['cagr']*100:.2f}% ({(best['cagr'] - baseline['cagr'])*100:+.2f}%)")
    print(f"  Sharpe:  {baseline['sharpe']:.2f} → {best['sharpe']:.2f} ({best['sharpe'] - baseline['sharpe']:+.2f})")
    print(f"  MDD:     {baseline['mdd']*100:.1f}% → {best['mdd']*100:.1f}%")
    print(f"  交易數:  {baseline['n_trades']} → {best['n_trades']} ({best['n_trades'] - baseline['n_trades']:+d})")

    # 儲存結果
    output_path = Path("satellite_floor_test_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n結果已儲存至: {output_path}")


if __name__ == "__main__":
    main()
