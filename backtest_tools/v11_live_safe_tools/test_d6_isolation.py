#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_d6_isolation.py

測試 D6 風險隔離策略
透過調整 D6 的權重乘數和排序懲罰來控制其風險貢獻
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


def run_with_tier_weights(
    signals: pd.DataFrame,
    price_provider: CSVPriceProvider,
    vix_data: pd.Series,
    d6_weight_mult: float = 1.0,
    d6_sort_penalty: Optional[float] = None,
) -> dict:
    """
    執行回測，並對 D6_STRICT 應用權重乘數或排序懲罰

    Args:
        d6_weight_mult: D6 訊號的權重乘數 (1.0 = 同等權重, 0.5 = 半權重)
        d6_sort_penalty: D6 排序分數懲罰乘數 (0.5 = 乘以 0.5 使其排後面)
    """
    # 複製訊號並添加權重調整欄位
    signals = signals.copy()

    # 根據 tier 設定權重乘數
    signals["weight_mult"] = 1.0
    signals.loc[signals["trade_long_tier"] == "D6_STRICT", "weight_mult"] = d6_weight_mult

    # 如果有 D6 排序懲罰，降低其 sort score
    if d6_sort_penalty is not None and d6_sort_penalty < 1.0:
        signals.loc[signals["trade_long_tier"] == "D6_STRICT", "earnings_day_return"] *= d6_sort_penalty

    config = BacktestConfigV32(
        cap_entries_per_quarter=24,
        max_concurrent_positions=24,
        sizing_positions=12,
        sort_column="earnings_day_return",
        satellite_floor=5.0,
        tier_weight_col="weight_mult",  # 使用新的 tier weight 功能
        target_gross_normal=2.0,
        target_gross_riskoff=1.0,
        target_gross_stress=0.0,
        per_trade_cap=0.20,
        commission_bps=10,
        slippage_bps=10,
        annual_borrow_rate=0.06,
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
    years = len(nav) / 252
    cagr = (end_nav / start_nav) ** (1/years) - 1

    # MDD
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    mdd = float(dd.min())

    # Sharpe
    daily_returns = nav.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    n_trades = len(trades)
    win_rate = float((trades["net_ret"] > 0).mean()) if n_trades > 0 else np.nan

    # 計算 D6 實際投資比例
    if not trades.empty and "invested_cash" in trades.columns:
        # 需要從原始 signals 找 tier 然後 match
        pass

    return {
        "d6_weight_mult": d6_weight_mult,
        "d6_sort_penalty": d6_sort_penalty if d6_sort_penalty else 1.0,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "n_trades": n_trades,
        "win_rate": win_rate,
    }


def main():
    # 載入訊號
    df = pd.read_csv("../long_only_signals_2017_2025_final.csv")
    df["reaction_date"] = pd.to_datetime(df["reaction_date"])
    cutoff = pd.Timestamp("2024-11-01")
    df = df[df["reaction_date"] < cutoff].copy()

    # D7 + D6
    d7_d6 = df[df["trade_long_tier"].isin(["D7_CORE", "D6_STRICT"])].copy()
    d7_d6["trade_long"] = True

    # 統計
    n_d7 = len(d7_d6[d7_d6["trade_long_tier"] == "D7_CORE"])
    n_d6 = len(d7_d6[d7_d6["trade_long_tier"] == "D6_STRICT"])
    print(f"訊號數: D7_CORE={n_d7}, D6_STRICT={n_d6}, 總計={len(d7_d6)}")

    # 載入 VIX
    vix_df = pd.read_csv("vix_data.csv")
    vix_df["date"] = pd.to_datetime(vix_df["date"])
    vix_df = vix_df.set_index("date")["close"]
    vix_df.index = pd.to_datetime(vix_df.index).normalize()

    # Price provider
    price_provider = CSVPriceProvider(folder=Path("../prices"))

    print("\n" + "=" * 80)
    print("D6 風險隔離測試 (10+10 bps 成本)")
    print("=" * 80)

    # 測試矩陣: (d6_weight_mult, d6_sort_penalty, 描述)
    test_configs = [
        (1.0, None, "D6=100% 權重 (基準)"),
        (0.67, None, "D6=67% 權重"),
        (0.5, None, "D6=50% 權重"),
        (0.33, None, "D6=33% 權重"),
        (1.0, 0.5, "D6 排序 x0.5"),
        (1.0, 0.3, "D6 排序 x0.3"),
        (0.5, 0.5, "D6=50% 權重 + 排序 x0.5"),
    ]

    results = []
    for weight, penalty, desc in test_configs:
        print(f"\n測試: {desc}")

        # 重新載入訊號（避免被上次修改影響）
        signals = d7_d6.copy()

        r = run_with_tier_weights(
            signals=signals,
            price_provider=price_provider,
            vix_data=vix_df,
            d6_weight_mult=weight,
            d6_sort_penalty=penalty,
        )
        r["description"] = desc
        results.append(r)

        print(f"  CAGR:   {r['cagr']*100:.2f}%")
        print(f"  Sharpe: {r['sharpe']:.2f}")
        print(f"  MDD:    {r['mdd']*100:.1f}%")
        print(f"  交易數: {r['n_trades']}")

    # 彙總
    print("\n" + "=" * 80)
    print("結果彙總")
    print("=" * 80)

    print("\n| 配置 | D6權重 | D6排序 | CAGR | Sharpe | MDD |")
    print("|------|--------|--------|------|--------|-----|")
    for r in results:
        penalty_str = f"x{r['d6_sort_penalty']:.1f}" if r['d6_sort_penalty'] < 1.0 else "x1.0"
        print(f"| {r['description'][:25]:25} | {r['d6_weight_mult']*100:.0f}% | {penalty_str:5} | "
              f"{r['cagr']*100:.1f}% | {r['sharpe']:.2f} | {r['mdd']*100:.1f}% |")

    # 找最佳
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    best_cagr = max(results, key=lambda x: x["cagr"])

    print(f"\n最高 Sharpe: {best_sharpe['description']}")
    print(f"  CAGR={best_sharpe['cagr']*100:.2f}%, Sharpe={best_sharpe['sharpe']:.2f}, MDD={best_sharpe['mdd']*100:.1f}%")

    print(f"\n最高 CAGR: {best_cagr['description']}")
    print(f"  CAGR={best_cagr['cagr']*100:.2f}%, Sharpe={best_cagr['sharpe']:.2f}, MDD={best_cagr['mdd']*100:.1f}%")

    # 儲存
    pd.DataFrame(results).to_csv("d6_isolation_results.csv", index=False)
    print("\n結果已儲存至: d6_isolation_results.csv")


if __name__ == "__main__":
    main()
