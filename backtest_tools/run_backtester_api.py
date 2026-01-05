#!/usr/bin/env python3
"""
run_backtester_api.py

使用 Whaleforce Backtester API 進行回測
將本地 signals 轉換為 API 格式並執行回測

API 文檔: https://backtest.api.whaleforce.dev/docs
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import time

# Backtester API 設定
BASE_URL = "https://backtest.api.whaleforce.dev"
VERIFY_SSL = False  # 內部服務可能使用自簽證書

# 回測參數
INITIAL_CAPITAL = 100000.0
HOLDING_DAYS = 30  # 持倉天數


def load_signals(signals_path: str) -> pd.DataFrame:
    """載入信號檔案"""
    df = pd.read_csv(signals_path)
    df["reaction_date"] = pd.to_datetime(df["reaction_date"])
    return df


def signals_to_portfolios_rebalance(
    signals: pd.DataFrame,
    leverage: float = 2.25,
    per_trade_cap: float = 0.22,
    max_concurrent: int = 24,
) -> List[Dict[str, Any]]:
    """
    將信號轉換為 weighted_rebalance 策略的 portfolios 格式
    每天提供當前持倉的權重快照
    """
    portfolios = []
    signals = signals.sort_values("reaction_date")
    active_positions = []

    all_dates = pd.date_range(
        start=signals["reaction_date"].min(),
        end=signals["reaction_date"].max() + timedelta(days=60),
        freq="B"
    )

    for date in all_dates:
        active_positions = [
            (sym, entry, exit_)
            for sym, entry, exit_ in active_positions
            if exit_ > date
        ]

        day_signals = signals[signals["reaction_date"] == date]
        for _, row in day_signals.iterrows():
            symbol = row["symbol"]
            entry_date = row["reaction_date"]
            exit_date = entry_date + timedelta(days=HOLDING_DAYS)

            if symbol not in [s for s, _, _ in active_positions]:
                if len(active_positions) < max_concurrent:
                    active_positions.append((symbol, entry_date, exit_date))

        if active_positions:
            n_positions = len(active_positions)
            weight_per_position = min(leverage / n_positions, per_trade_cap)

            positions = [
                {"ticker": sym, "weight": weight_per_position, "action": "buy"}
                for sym, _, _ in active_positions
            ]

            portfolios.append({
                "date": date.strftime("%Y-%m-%dT00:00:00Z"),
                "positions": positions
            })

    return portfolios


def signals_to_portfolios_generalize(
    signals: pd.DataFrame,
    leverage: float = 2.25,
    per_trade_cap: float = 0.22,
    max_concurrent: int = 24,
) -> List[Dict[str, Any]]:
    """
    將信號轉換為 generalize 策略的 portfolios 格式

    generalize 策略特點:
    - 明確的 buy/sell 交易信號
    - 支援 stop-loss
    - 更接近實際交易邏輯

    格式: 只在進場和出場日發送信號
    """
    portfolios = []
    signals = signals.sort_values("reaction_date")

    # 追蹤持倉 {symbol: (entry_date, exit_date, weight)}
    active_positions = {}
    trade_events = []  # [(date, symbol, action, weight)]

    # 收集所有交易事件
    for _, row in signals.iterrows():
        symbol = row["symbol"]
        entry_date = row["reaction_date"]
        exit_date = entry_date + timedelta(days=HOLDING_DAYS)

        # 進場
        trade_events.append((entry_date, symbol, "buy", per_trade_cap))
        # 出場
        trade_events.append((exit_date, symbol, "sell", 0.0))

    # 按日期分組
    events_by_date = {}
    for date, symbol, action, weight in trade_events:
        date_str = date.strftime("%Y-%m-%dT00:00:00Z")
        if date_str not in events_by_date:
            events_by_date[date_str] = []
        events_by_date[date_str].append({
            "ticker": symbol,
            "weight": weight,
            "action": action
        })

    # 轉換為 portfolios 列表
    for date_str in sorted(events_by_date.keys()):
        portfolios.append({
            "date": date_str,
            "positions": events_by_date[date_str]
        })

    return portfolios


def submit_backtest(
    signals: pd.DataFrame,
    name: str = "Earnings Strategy",
    leverage: float = 2.25,
    per_trade_cap: float = 0.22,
    start_date: str = None,
    end_date: str = None,
) -> str:
    """
    提交回測請求，立即返回 backtest_id

    Returns:
        backtest_id: 用於之後查詢結果
    """
    # 轉換信號為 portfolios
    print("Converting signals to portfolios (generalize strategy)...")
    portfolios = signals_to_portfolios_generalize(
        signals,
        leverage=leverage,
        per_trade_cap=per_trade_cap
    )
    print(f"Generated {len(portfolios)} portfolio snapshots")

    if not portfolios:
        raise ValueError("No portfolios generated from signals")

    # 根據 portfolios 的日期範圍設定 start/end date
    portfolio_dates = [p["date"] for p in portfolios]
    first_date = min(portfolio_dates)
    last_date = max(portfolio_dates)

    if start_date is None:
        start_date = first_date
    if end_date is None:
        end_dt = datetime.strptime(last_date[:10], "%Y-%m-%d") + timedelta(days=60)
        end_date = end_dt.strftime("%Y-%m-%dT00:00:00Z")

    # 建立回測請求
    payload = {
        "start_date": start_date,
        "end_date": end_date,
        "interval": "1d",
        "initial_capital": INITIAL_CAPITAL,
        "base_currency": "USD",
        "strategy_name": "generalize",
        "name": name,
        "comment": f"Leverage: {leverage}x, Per-trade cap: {per_trade_cap*100}%",
        "leverage": leverage,
        "portfolios": portfolios,
        "costs": {
            "margin_cost": 0.06,
            "borrow_cost": 0.02
        },
        "stop_loss": {
            "long_loss": 0.15,
        }
    }

    print(f"\nSubmitting backtest: {name}")
    print(f"Date range: {start_date[:10]} to {end_date[:10]}")
    print(f"Leverage: {leverage}x")

    response = requests.post(
        f"{BASE_URL}/backtest/run",
        json=payload,
        verify=VERIFY_SSL
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        raise Exception(f"Backtest failed: {response.text}")

    result = response.json()
    backtest_id = result.get("backtest_id")
    print(f"Backtest ID: {backtest_id}")
    print("Backtest submitted. Use get_backtest_result() to check status.")

    return backtest_id


def get_backtest_status(backtest_id: str) -> Dict[str, Any]:
    """
    查詢回測狀態

    Returns:
        dict with 'state' (progressing/finish/failed) and metadata
    """
    response = requests.get(
        f"{BASE_URL}/backtest/metadata/{backtest_id}",
        verify=VERIFY_SSL
    )

    if response.status_code == 404:
        return {"state": "not_found", "backtest_id": backtest_id}

    if response.status_code != 200:
        raise Exception(f"Failed to get status: {response.text}")

    return response.json()


def get_backtest_result(backtest_id: str) -> Dict[str, Any]:
    """
    查詢回測結果

    Returns:
        dict with summary_metrics if complete, or status info if still running
    """
    # 先檢查狀態
    status = get_backtest_status(backtest_id)
    state = status.get("state", "unknown")

    if state == "not_found":
        return {"error": "Backtest not found", "backtest_id": backtest_id}

    if state == "progressing":
        return {
            "status": "running",
            "backtest_id": backtest_id,
            "message": "Backtest still in progress. Try again later."
        }

    if state == "failed":
        return {
            "status": "failed",
            "backtest_id": backtest_id,
            "message": "Backtest failed."
        }

    # 取得完整結果
    response = requests.get(
        f"{BASE_URL}/backtest/result/{backtest_id}",
        verify=VERIFY_SSL
    )

    if response.status_code != 200:
        return {"error": f"Failed to get result: {response.status_code}"}

    return response.json()


def check_multiple_backtests(backtest_ids: List[str]) -> pd.DataFrame:
    """
    批次查詢多個回測的狀態和結果

    Returns:
        DataFrame with backtest results
    """
    results = []

    for bid in backtest_ids:
        result = get_backtest_result(bid)

        if "error" in result or "status" in result:
            results.append({
                "backtest_id": bid,
                "state": result.get("status", result.get("error", "unknown")),
                "total_return_pct": None,
                "annualized_return_pct": None,
                "sharpe_ratio": None,
                "max_drawdown_pct": None,
            })
        else:
            metrics = result.get("summary_metrics", {})
            results.append({
                "backtest_id": bid,
                "state": "finish",
                "total_return_pct": metrics.get("total_return_pct"),
                "annualized_return_pct": metrics.get("annualized_return_pct"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                "final_portfolio_value": metrics.get("final_portfolio_value"),
            })

    return pd.DataFrame(results)


def run_spy_benchmark(start_date: str, end_date: str) -> Dict[str, Any]:
    """執行 SPY benchmark 回測"""
    payload = {
        "start_date": start_date,
        "end_date": end_date,
        "interval": "1d",
        "initial_capital": INITIAL_CAPITAL,
        "base_currency": "USD",
        "strategy_name": "weighted_rebalance",
        "name": "SPY Benchmark",
        "initial_portfolio": [{"ticker": "SPY", "weight": 1.0}]
    }

    response = requests.post(
        f"{BASE_URL}/backtest/run",
        json=payload,
        verify=VERIFY_SSL
    )

    if response.status_code != 200:
        raise Exception(f"SPY benchmark failed: {response.text}")

    backtest_id = response.json().get("backtest_id")
    print(f"SPY Backtest ID: {backtest_id}")

    # 輪詢等待結果
    print("Waiting for SPY results", end="", flush=True)
    max_wait = 60
    poll_interval = 3

    for i in range(0, max_wait, poll_interval):
        time.sleep(poll_interval)
        print(".", end="", flush=True)

        result_response = requests.get(
            f"{BASE_URL}/backtest/result/{backtest_id}",
            verify=VERIFY_SSL
        )

        if result_response.status_code == 200:
            result = result_response.json()
            if result.get("summary_metrics"):
                print(" Done!")
                return result

    print(" Timeout!")
    return {"summary_metrics": {}}


def print_results(result: Dict[str, Any], name: str = "Strategy"):
    """印出回測結果"""
    metrics = result.get("summary_metrics", {})

    print(f"\n{'='*60}")
    print(f" {name} Results")
    print(f"{'='*60}")
    print(f"Total Return:      {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Annualized Return: {metrics.get('annualized_return_pct', 0):.2f}%")
    print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Volatility:        {metrics.get('volatility_pct', 0):.2f}%")
    print(f"{'='*60}")


def submit_all_backtests() -> Dict[str, str]:
    """
    提交所有回測 (G200, G225, G250)，返回 backtest_ids

    Usage:
        ids = submit_all_backtests()
        # 等待 10 分鐘後...
        results = check_multiple_backtests(list(ids.values()))
    """
    signals_path = Path(__file__).parent / "long_only_signals_2017_2025_final.csv"

    if not signals_path.exists():
        print(f"Error: Signals file not found: {signals_path}")
        return {}

    print(f"Loading signals: {signals_path}")
    signals = load_signals(str(signals_path))
    print(f"Loaded {len(signals)} signals")
    print(f"Date range: {signals['reaction_date'].min().date()} to {signals['reaction_date'].max().date()}")

    configs = [
        ("G200", 2.0, 0.20),
        ("G225", 2.25, 0.22),
        ("G250", 2.5, 0.25),
    ]

    backtest_ids = {}

    for name, leverage, cap in configs:
        try:
            bid = submit_backtest(
                signals=signals,
                name=name,
                leverage=leverage,
                per_trade_cap=cap,
            )
            backtest_ids[name] = bid
            print()
        except Exception as e:
            print(f"Error submitting {name}: {e}")

    print("\n" + "=" * 60)
    print("All backtests submitted!")
    print("=" * 60)
    print("\nBacktest IDs:")
    for name, bid in backtest_ids.items():
        print(f"  {name}: {bid}")

    print("\nTo check results later, run:")
    print("  from run_backtester_api import check_multiple_backtests")
    print(f"  ids = {list(backtest_ids.values())}")
    print("  results = check_multiple_backtests(ids)")
    print("  print(results)")

    return backtest_ids


def main():
    """主程式 - 提交回測並等待結果"""
    import sys

    if len(sys.argv) > 1:
        # 如果有提供 backtest_id，查詢結果
        if sys.argv[1] == "check":
            if len(sys.argv) > 2:
                backtest_ids = sys.argv[2:]
                results = check_multiple_backtests(backtest_ids)
                print(results.to_string())
            else:
                print("Usage: python run_backtester_api.py check <backtest_id1> <backtest_id2> ...")
        elif sys.argv[1] == "status":
            if len(sys.argv) > 2:
                status = get_backtest_status(sys.argv[2])
                print(json.dumps(status, indent=2))
            else:
                print("Usage: python run_backtester_api.py status <backtest_id>")
        elif sys.argv[1] == "result":
            if len(sys.argv) > 2:
                result = get_backtest_result(sys.argv[2])
                if "summary_metrics" in result:
                    print_results(result, sys.argv[2][:12])
                else:
                    print(json.dumps(result, indent=2))
            else:
                print("Usage: python run_backtester_api.py result <backtest_id>")
        else:
            print("Unknown command. Available commands:")
            print("  python run_backtester_api.py          # Submit all backtests")
            print("  python run_backtester_api.py check <id1> <id2> ...  # Check multiple results")
            print("  python run_backtester_api.py status <id>  # Check status")
            print("  python run_backtester_api.py result <id>  # Get full result")
    else:
        # 預設：提交所有回測
        submit_all_backtests()


if __name__ == "__main__":
    main()
