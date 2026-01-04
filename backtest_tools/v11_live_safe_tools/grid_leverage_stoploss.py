#!/usr/bin/env python3
"""
grid_leverage_stoploss.py

Grid search for optimal leverage, stop-loss, and cost combinations.

Searches over:
- Target gross (normal): [1.0, 1.25, 1.5, 1.75]
- Target gross (riskoff): [0.4, 0.6, 0.8]
- Stop-loss: [None, 0.08, 0.10, 0.12, 0.15]
- Costs: [10, 20, 30, 50] bps

Constraints:
- Max Drawdown >= -20%
- Sharpe >= 1.0
- Max Leverage <= 2.0x

Output: grid_results.csv with all configurations sorted by CAGR
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtester_v32 import BacktestConfigV32, run_backtest_v32
from price_providers import CSVPriceProvider


def compute_metrics(nav: pd.Series, rf_annual: float = 0.0) -> dict:
    """計算績效指標"""
    if nav.empty or len(nav) < 2:
        return {}

    rets = nav.pct_change().dropna()
    if rets.empty:
        return {}

    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    n_days = len(rets)
    n_years = n_days / 252.0

    cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    ann_vol = rets.std() * np.sqrt(252)

    rf_daily = rf_annual / 252.0
    excess_rets = rets - rf_daily
    sharpe = (excess_rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0

    downside = rets[rets < rf_daily]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else 0.0
    sortino = (cagr - rf_annual) / downside_std if downside_std > 0 else 0.0

    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def compute_trade_metrics(trades: pd.DataFrame) -> dict:
    """計算交易指標"""
    if trades.empty:
        return {}

    n_trades = len(trades)
    wins = trades[trades["net_ret"] > 0]
    losses = trades[trades["net_ret"] <= 0]

    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = wins["net_ret"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["net_ret"].mean() if len(losses) > 0 else 0.0

    gross_profit = wins["net_ret"].sum()
    gross_loss = abs(losses["net_ret"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def run_single_config(
    signals: pd.DataFrame,
    price_provider: CSVPriceProvider,
    vix_data: Optional[pd.Series],
    gross_normal: float,
    gross_riskoff: float,
    gross_stress: float,
    stop_loss: Optional[float],
    costs_bps: float,
    borrow_rate: float,
) -> Optional[Dict]:
    """執行單一配置的回測"""
    config = BacktestConfigV32(
        initial_cash=100000.0,
        target_gross_normal=gross_normal,
        target_gross_riskoff=gross_riskoff,
        target_gross_stress=gross_stress,
        stop_loss_pct=stop_loss,
        commission_bps=costs_bps / 2.0,
        slippage_bps=costs_bps / 2.0,
        annual_borrow_rate=borrow_rate,
    )

    try:
        nav, trades, exposure, stats = run_backtest_v32(
            signals=signals,
            price_provider=price_provider,
            config=config,
            vix_data=vix_data,
        )
    except Exception as e:
        print(f"    錯誤: {e}")
        return None

    metrics = compute_metrics(nav)
    trade_metrics = compute_trade_metrics(trades)
    metrics.update(trade_metrics)
    metrics["exposure_avg"] = exposure.mean()
    metrics["max_leverage"] = stats["max_leverage"]
    metrics["stop_loss_triggered"] = stats["stop_loss_triggered"]
    metrics["margin_interest"] = stats["total_margin_interest"]

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Grid search for leverage and stop-loss")
    parser.add_argument("--signals", required=True, help="訊號檔案路徑")
    parser.add_argument("--vix", default="vix_data.csv", help="VIX 資料檔案路徑")
    parser.add_argument("--prices-folder", default="../prices", help="價格資料夾路徑")
    parser.add_argument("--output", default="grid_results.csv", help="輸出檔案")
    parser.add_argument("--borrow-rate", type=float, default=0.06, help="年化融資利率")
    parser.add_argument("--quick", action="store_true", help="快速模式 (較少配置)")
    args = parser.parse_args()

    # 載入訊號
    signals_path = Path(args.signals)
    if not signals_path.exists():
        print(f"錯誤: 找不到訊號檔案: {signals_path}")
        sys.exit(1)

    print(f"載入訊號: {signals_path}")
    signals = pd.read_csv(signals_path)
    if "trade_long" in signals.columns:
        signals = signals[signals["trade_long"] == True].copy()
    print(f"  多頭交易數: {len(signals)}")

    # 載入 VIX 資料
    vix_path = Path(args.vix)
    vix_data = None
    if vix_path.exists():
        print(f"載入 VIX 資料: {vix_path}")
        vix_df = pd.read_csv(vix_path)
        # Handle both capitalized and lowercase column names
        date_col = "Date" if "Date" in vix_df.columns else "date"
        close_col = "Close" if "Close" in vix_df.columns else "close"
        vix_df[date_col] = pd.to_datetime(vix_df[date_col])
        vix_df = vix_df.set_index(date_col)[close_col]
        vix_df.index = pd.to_datetime(vix_df.index).normalize()
        vix_data = vix_df
        print(f"  VIX 資料: {len(vix_data)} 天")
    else:
        print(f"警告: 找不到 VIX 檔案: {vix_path}")

    # 定義搜尋網格
    if args.quick:
        # 快速模式: 較少配置
        gross_normal_values = [1.0, 1.25, 1.5]
        gross_riskoff_values = [0.5, 0.7]
        stop_loss_values = [None, 0.10, 0.12]
        costs_values = [20, 30]
    else:
        # 完整網格
        gross_normal_values = [1.0, 1.15, 1.25, 1.35, 1.5, 1.75]
        gross_riskoff_values = [0.4, 0.5, 0.6, 0.7, 0.8]
        stop_loss_values = [None, 0.08, 0.10, 0.12, 0.15, 0.20]
        costs_values = [10, 20, 30, 50]

    gross_stress = 0.0  # 固定為 0

    total_configs = (
        len(gross_normal_values)
        * len(gross_riskoff_values)
        * len(stop_loss_values)
        * len(costs_values)
    )
    print(f"\n總配置數: {total_configs}")

    # 執行網格搜尋
    price_provider = CSVPriceProvider(folder=Path(args.prices_folder))
    results = []
    completed = 0

    for gross_n, gross_r, stop, costs in product(
        gross_normal_values, gross_riskoff_values, stop_loss_values, costs_values
    ):
        completed += 1
        stop_str = f"{stop*100:.0f}%" if stop else "None"
        print(f"[{completed}/{total_configs}] gross={gross_n}/{gross_r}, stop={stop_str}, costs={costs}bps")

        metrics = run_single_config(
            signals=signals,
            price_provider=price_provider,
            vix_data=vix_data,
            gross_normal=gross_n,
            gross_riskoff=gross_r,
            gross_stress=gross_stress,
            stop_loss=stop,
            costs_bps=costs,
            borrow_rate=args.borrow_rate,
        )

        if metrics is None:
            continue

        # 紀錄配置和結果
        result = {
            "gross_normal": gross_n,
            "gross_riskoff": gross_r,
            "gross_stress": gross_stress,
            "stop_loss": stop if stop else 0,
            "costs_bps": costs,
            "borrow_rate": args.borrow_rate,
            **metrics,
        }

        # 檢查限制條件
        mdd = metrics.get("max_drawdown", -1)
        sharpe = metrics.get("sharpe", 0)
        max_lev = metrics.get("max_leverage", 99)

        result["mdd_pass"] = mdd >= -0.20
        result["sharpe_pass"] = sharpe >= 1.0
        result["lev_pass"] = max_lev <= 2.0
        result["all_pass"] = result["mdd_pass"] and result["sharpe_pass"] and result["lev_pass"]

        results.append(result)

    # 儲存結果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("cagr", ascending=False)
    results_df.to_csv(args.output, index=False)
    print(f"\n結果已儲存到: {args.output}")

    # 顯示 Top 10 (所有配置)
    print("\n" + "=" * 80)
    print("TOP 10 配置 (依 CAGR 排序)")
    print("=" * 80)
    display_cols = [
        "gross_normal", "gross_riskoff", "stop_loss", "costs_bps",
        "cagr", "sharpe", "max_drawdown", "n_trades", "all_pass"
    ]
    top10 = results_df.head(10)[display_cols].copy()
    top10["cagr"] = top10["cagr"].apply(lambda x: f"{x*100:.1f}%")
    top10["max_drawdown"] = top10["max_drawdown"].apply(lambda x: f"{x*100:.1f}%")
    top10["sharpe"] = top10["sharpe"].apply(lambda x: f"{x:.2f}")
    top10["stop_loss"] = top10["stop_loss"].apply(lambda x: f"{x*100:.0f}%" if x > 0 else "None")
    print(top10.to_string(index=False))

    # 顯示 Top 10 (通過限制條件)
    passed = results_df[results_df["all_pass"] == True]
    print(f"\n通過所有限制條件的配置數: {len(passed)}")

    if len(passed) > 0:
        print("\n" + "=" * 80)
        print("TOP 10 通過限制條件的配置 (MDD >= -20%, Sharpe >= 1.0, Lev <= 2.0x)")
        print("=" * 80)
        top10_pass = passed.head(10)[display_cols].copy()
        top10_pass["cagr"] = top10_pass["cagr"].apply(lambda x: f"{x*100:.1f}%")
        top10_pass["max_drawdown"] = top10_pass["max_drawdown"].apply(lambda x: f"{x*100:.1f}%")
        top10_pass["sharpe"] = top10_pass["sharpe"].apply(lambda x: f"{x:.2f}")
        top10_pass["stop_loss"] = top10_pass["stop_loss"].apply(lambda x: f"{x*100:.0f}%" if x > 0 else "None")
        print(top10_pass.to_string(index=False))

        # 最佳配置
        best = passed.iloc[0]
        print("\n" + "=" * 80)
        print("最佳配置 (通過限制條件中 CAGR 最高)")
        print("=" * 80)
        print(f"  Target Gross Normal: {best['gross_normal']}")
        print(f"  Target Gross RiskOff: {best['gross_riskoff']}")
        print(f"  Stop-Loss: {best['stop_loss']*100:.0f}%" if best['stop_loss'] > 0 else "  Stop-Loss: None")
        print(f"  Costs: {best['costs_bps']:.0f} bps")
        print(f"  CAGR: {best['cagr']*100:.1f}%")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  Sortino: {best['sortino']:.2f}")
        print(f"  Max Drawdown: {best['max_drawdown']*100:.1f}%")
        print(f"  Win Rate: {best['win_rate']*100:.1f}%")
        print(f"  Trades: {best['n_trades']:.0f}")
    else:
        print("\n警告: 沒有配置通過所有限制條件!")


if __name__ == "__main__":
    main()
