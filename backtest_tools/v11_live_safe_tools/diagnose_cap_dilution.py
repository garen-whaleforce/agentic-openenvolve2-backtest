#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_cap_dilution.py

診斷 CAP 稀釋問題：比較 CAP=12 vs CAP=24 的行為差異

分析內容：
1. 平均曝險 (gross exposure)
2. 平均每筆倉位權重 (position weight)
3. 訊號密度與實際入場數
4. (可選) Rank 分桶的報酬比較
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


def run_config(
    signals: pd.DataFrame,
    price_provider: CSVPriceProvider,
    vix_data: Optional[pd.Series],
    cap: int,
    max_pos: int,
    sizing_pos: Optional[int] = None,  # NEW: sizing_positions (decoupled)
    gross_normal: float = 2.0,
    stop_loss: Optional[float] = None,
) -> dict:
    """執行單一配置並回傳診斷資料"""

    config = BacktestConfigV32(
        cap_entries_per_quarter=cap,
        max_concurrent_positions=max_pos,
        sizing_positions=sizing_pos,  # NEW
        target_gross_normal=gross_normal,
        target_gross_riskoff=1.0,
        target_gross_stress=0.0,
        stop_loss_pct=stop_loss,
        per_trade_cap=0.20,
        commission_bps=5,
        slippage_bps=5,
        annual_borrow_rate=0.06,
    )

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

    # 平均曝險
    avg_exposure = float(exposure.mean())

    # 平均每筆倉位權重
    if not trades.empty and "invested_cash" in trades.columns:
        # 用進場時的投資金額 / 當時的 NAV (近似為 initial cash)
        avg_position_weight = float(trades["invested_cash"].mean() / start_nav)
    else:
        avg_position_weight = np.nan

    # 交易數
    n_trades = len(trades)

    # 勝率
    if not trades.empty:
        win_rate = float((trades["net_ret"] > 0).mean())
    else:
        win_rate = np.nan

    # MDD
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    mdd = float(dd.min())

    return {
        "cap": cap,
        "max_pos": max_pos,
        "sizing_pos": sizing_pos if sizing_pos else max_pos,  # NEW
        "gross_target": gross_normal,
        "cagr": cagr,
        "avg_exposure": avg_exposure,
        "avg_position_weight": avg_position_weight,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "mdd": mdd,
        "end_nav": end_nav,
    }


def main():
    parser = argparse.ArgumentParser(description="診斷 CAP 稀釋問題")
    parser.add_argument("--signals", required=True, help="訊號檔案路徑")
    parser.add_argument("--vix", default="vix_data.csv", help="VIX 資料檔案路徑")
    parser.add_argument("--prices-folder", default="../prices", help="價格資料夾路徑")
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

    # 過濾掉 2024-11 之後的訊號（避免價格資料缺失）
    if "reaction_date" in signals.columns:
        signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])
        cutoff = pd.Timestamp("2024-11-01")
        before_cutoff = len(signals)
        signals = signals[signals["reaction_date"] < cutoff].copy()
        print(f"  過濾 2024-11 前: {before_cutoff} → {len(signals)}")

    print(f"  多頭訊號數: {len(signals)}")

    # 載入 VIX
    vix_path = Path(args.vix)
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
    price_provider = CSVPriceProvider(folder=Path(args.prices_folder))

    # 測試配置組合
    print("\n" + "=" * 80)
    print("CAP 稀釋診斷測試")
    print("=" * 80)

    configs = [
        # (CAP, MaxPos, SizingPos, Gross, StopLoss, 描述)
        (12, 12, None, 2.0, None, "CAP=12, MaxPos=12, Sizing=12 (基準)"),
        (24, 24, None, 2.0, None, "CAP=24, MaxPos=24, Sizing=24 (等比放大-會稀釋)"),
        (24, 24, 12, 2.0, None, "CAP=24, MaxPos=24, Sizing=12 (解耦-新功能)"),
        (24, 12, None, 2.0, None, "CAP=24, MaxPos=12, Sizing=12 (舊解耦)"),
        (16, 16, 12, 2.0, None, "CAP=16, MaxPos=16, Sizing=12 (微調)"),
    ]

    results = []
    for cap, max_pos, sizing_pos, gross, stop, desc in configs:
        print(f"\n測試: {desc}")
        try:
            r = run_config(
                signals=signals,
                price_provider=price_provider,
                vix_data=vix_data,
                cap=cap,
                max_pos=max_pos,
                sizing_pos=sizing_pos,  # NEW
                gross_normal=gross,
                stop_loss=stop,
            )
            r["description"] = desc
            results.append(r)

            print(f"  CAGR:        {r['cagr']*100:>7.2f}%")
            print(f"  平均曝險:     {r['avg_exposure']*100:>7.2f}%")
            print(f"  平均倉位權重: {r['avg_position_weight']*100:>7.2f}%")
            print(f"  交易數:      {r['n_trades']:>7}")
            print(f"  勝率:        {r['win_rate']*100:>7.1f}%")
            print(f"  MDD:         {r['mdd']*100:>7.2f}%")

        except Exception as e:
            print(f"  錯誤: {e}")

    # 彙整報告
    print("\n" + "=" * 80)
    print("彙整比較")
    print("=" * 80)

    df = pd.DataFrame(results)

    # 格式化輸出
    print("\n| 配置 | CAP | MaxPos | Sizing | CAGR | 平均曝險 | 倉位權重 | 交易數 | 勝率 | MDD |")
    print("|------|-----|--------|--------|------|---------|---------|-------|------|-----|")

    for _, row in df.iterrows():
        print(f"| {row['description'][:25]} | {row['cap']} | {row['max_pos']} | {row['sizing_pos']} | "
              f"{row['cagr']*100:.1f}% | {row['avg_exposure']*100:.1f}% | "
              f"{row['avg_position_weight']*100:.1f}% | {row['n_trades']} | "
              f"{row['win_rate']*100:.1f}% | {row['mdd']*100:.1f}% |")

    # 診斷結論
    print("\n" + "=" * 80)
    print("診斷結論")
    print("=" * 80)

    if len(df) >= 2:
        base = df[df["cap"] == 12].iloc[0] if len(df[df["cap"] == 12]) > 0 else df.iloc[0]
        cap24 = df[df["cap"] == 24]

        if len(cap24) > 0:
            cap24_same = cap24[cap24["max_pos"] == 24]
            cap24_decoupled = cap24[cap24["max_pos"] == 12]

            if len(cap24_same) > 0:
                r24 = cap24_same.iloc[0]
                print(f"\n1. CAP=12 → CAP=24 (MaxPos 同步放大)")
                print(f"   CAGR 變化:     {base['cagr']*100:.2f}% → {r24['cagr']*100:.2f}% "
                      f"({(r24['cagr'] - base['cagr'])*100:+.2f}%)")
                print(f"   平均曝險變化:  {base['avg_exposure']*100:.1f}% → {r24['avg_exposure']*100:.1f}%")
                print(f"   倉位權重變化:  {base['avg_position_weight']*100:.1f}% → {r24['avg_position_weight']*100:.1f}%")

                if r24['avg_exposure'] < base['avg_exposure'] - 0.05:
                    print("\n   診斷: 【工程稀釋】 - 曝險下降，確認是 sizing 分母問題")
                elif r24['cagr'] < base['cagr'] and r24['avg_exposure'] >= base['avg_exposure']:
                    print("\n   診斷: 【品質稀釋】 - 曝險未降但 CAGR 下降，可能是弱訊號進入")
                else:
                    print("\n   診斷: 需要進一步分析")

            if len(cap24_decoupled) > 0:
                r24d = cap24_decoupled.iloc[0]
                print(f"\n2. CAP=24 但 MaxPos=12 (解耦測試)")
                print(f"   CAGR:          {r24d['cagr']*100:.2f}%")
                print(f"   平均曝險:       {r24d['avg_exposure']*100:.1f}%")
                print(f"   倉位權重:       {r24d['avg_position_weight']*100:.1f}%")

                if r24d['cagr'] > base['cagr']:
                    print("\n   解耦有效! CAP 可以提高但 MaxPos 保持較小")
                elif r24d['cagr'] < base['cagr']:
                    print("\n   解耦無效，問題可能在訊號品質而非工程")

    # 儲存結果
    output_path = Path("cap_dilution_diagnosis.csv")
    df.to_csv(output_path, index=False)
    print(f"\n結果已儲存至: {output_path}")


if __name__ == "__main__":
    main()
