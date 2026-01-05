#!/usr/bin/env python3
"""
Run G250 backtest using Whaleforce Backtester API (generalize strategy).

G250 Configuration:
- Leverage: 2.5x (250%)
- Per trade cap: 25%
- Holding period: 30 days
- Stop loss: 15%
- Entry lag: 1 day after signal
"""

import pandas as pd
import httpx
import json
from datetime import datetime, timedelta
from collections import defaultdict

# Whaleforce API endpoint
API_URL = "https://backtest.api.whaleforce.dev"

def load_signals(csv_path: str) -> pd.DataFrame:
    """Load signals from CSV."""
    df = pd.read_csv(csv_path)
    df['reaction_date'] = pd.to_datetime(df['reaction_date'])
    return df

def get_next_trading_day(date: datetime, days: int = 1) -> datetime:
    """Get the next trading day (skip weekends)."""
    result = date
    added = 0
    while added < days:
        result += timedelta(days=1)
        # Skip weekends (5=Saturday, 6=Sunday)
        if result.weekday() < 5:
            added += 1
    return result

def signals_to_portfolios_generalize(
    signals: pd.DataFrame,
    leverage: float = 2.5,
    per_trade_cap: float = 0.25,
    holding_days: int = 30,
    entry_lag: int = 1,
) -> list:
    """
    Convert signals to portfolio format for generalize strategy.

    Returns list of portfolios with format:
    [{"date": "YYYY-MM-DDTHH:MM:SS", "positions": [{"ticker": "XXX", "weight": 0.25}, ...]}, ...]
    """
    # Sort signals by reaction_date
    signals = signals.sort_values('reaction_date').reset_index(drop=True)

    # Trade events: (date, symbol, action, weight)
    trade_events = []

    for _, row in signals.iterrows():
        symbol = row['symbol']
        signal_date = row['reaction_date']

        # Entry date: signal_date + entry_lag trading days
        entry_date = get_next_trading_day(signal_date, entry_lag)

        # Exit date: entry_date + holding_days trading days
        exit_date = entry_date
        for _ in range(holding_days):
            exit_date = get_next_trading_day(exit_date, 1)

        # Add entry event
        trade_events.append((entry_date, symbol, "buy", per_trade_cap))

        # Add exit event
        trade_events.append((exit_date, symbol, "sell", 0.0))

    # Sort events by date
    trade_events.sort(key=lambda x: x[0])

    # Build daily portfolios
    current_positions = {}  # symbol -> weight
    portfolios = []

    # Group events by date
    events_by_date = defaultdict(list)
    for date, symbol, action, weight in trade_events:
        events_by_date[date].append((symbol, action, weight))

    # Process each date
    all_dates = sorted(events_by_date.keys())

    for date in all_dates:
        # Process exits first, then entries
        exits = [(s, a, w) for s, a, w in events_by_date[date] if a == "sell"]
        entries = [(s, a, w) for s, a, w in events_by_date[date] if a == "buy"]

        for symbol, action, weight in exits:
            if symbol in current_positions:
                del current_positions[symbol]

        for symbol, action, weight in entries:
            if symbol not in current_positions:
                current_positions[symbol] = weight

        # Create portfolio snapshot if we have positions
        if current_positions:
            # Normalize weights based on leverage
            n_positions = len(current_positions)
            target_weight = min(leverage / n_positions, per_trade_cap)

            positions = [
                {"ticker": sym, "weight": target_weight}  # Use "ticker" not "symbol"
                for sym in sorted(current_positions.keys())
            ]

            portfolios.append({
                "date": date.strftime("%Y-%m-%dT00:00:00"),  # ISO datetime format
                "positions": positions
            })

    return portfolios


def run_backtest(
    signals: pd.DataFrame,
    leverage: float = 2.5,
    per_trade_cap: float = 0.25,
    holding_days: int = 30,
    stop_loss: float = 0.15,
    initial_capital: float = 100000,
    start_date: str = "2017-01-01",
    end_date: str = "2025-11-19",
) -> dict:
    """Run backtest via Whaleforce API."""

    # Filter signals within date range
    signals = signals[
        (signals['reaction_date'] >= start_date) &
        (signals['reaction_date'] < end_date)
    ].copy()

    print(f"Filtered signals: {len(signals)} within {start_date} to {end_date}")

    # Convert to portfolios
    portfolios = signals_to_portfolios_generalize(
        signals,
        leverage=leverage,
        per_trade_cap=per_trade_cap,
        holding_days=holding_days,
    )

    print(f"Generated {len(portfolios)} portfolio snapshots")

    if not portfolios:
        print("No portfolios generated!")
        return {}

    # Get date range from portfolios
    portfolio_dates = [p["date"] for p in portfolios]
    bt_start = min(portfolio_dates)[:10]  # Get just YYYY-MM-DD part
    bt_end = max(portfolio_dates)[:10]    # Get just YYYY-MM-DD part

    # Add buffer to end date
    bt_end_dt = datetime.strptime(bt_end, "%Y-%m-%d") + timedelta(days=45)
    bt_end = bt_end_dt.strftime("%Y-%m-%d")

    print(f"Backtest period: {bt_start} to {bt_end}")

    # Prepare payload according to API schema
    payload = {
        "strategy_name": "generalize",
        "leverage": leverage,
        "initial_capital": initial_capital,
        "start_date": f"{bt_start}T00:00:00",  # ISO datetime format
        "end_date": f"{bt_end}T00:00:00",      # ISO datetime format
        "interval": "1d",
        "base_currency": "USD",
        "portfolios": portfolios,
        "costs": {
            "margin_cost": 0.06,  # 6% annual margin rate
            "borrow_cost": 0.02,  # 2% annual borrow cost
        },
        "stop_loss": {
            "long_loss": stop_loss  # 15% stop loss
        }
    }

    print(f"\nSubmitting backtest to {API_URL}...")
    print(f"  - Leverage: {leverage}x")
    print(f"  - Per trade cap: {per_trade_cap*100}%")
    print(f"  - Holding days: {holding_days}")
    print(f"  - Stop loss: {stop_loss*100}%")
    print(f"  - Initial capital: ${initial_capital:,.0f}")

    # Submit backtest
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{API_URL}/backtest/run", json=payload)

        if resp.status_code != 200:
            print(f"Error: {resp.status_code}")
            print(resp.text)
            return {}

        result = resp.json()

    return result


def print_results(result: dict):
    """Print backtest results."""
    if not result:
        print("No results to display")
        return

    print("\n" + "="*60)
    print("G250 BACKTEST RESULTS (Whaleforce API - Generalize)")
    print("="*60)

    if "backtest_id" in result:
        print(f"\nBacktest ID: {result['backtest_id']}")

    if "summary" in result:
        summary = result["summary"]
        print(f"\n{'Metric':<30} {'Value':>15}")
        print("-"*45)

        metrics = [
            ("Total Return", f"{summary.get('total_return', 0)*100:.1f}%"),
            ("CAGR", f"{summary.get('cagr', 0)*100:.2f}%"),
            ("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}"),
            ("Sortino Ratio", f"{summary.get('sortino_ratio', 0):.2f}"),
            ("Calmar Ratio", f"{summary.get('calmar_ratio', 0):.2f}"),
            ("Max Drawdown", f"{summary.get('max_drawdown', 0)*100:.2f}%"),
            ("Volatility (Ann.)", f"{summary.get('volatility', 0)*100:.2f}%"),
            ("Initial Capital", f"${summary.get('initial_capital', 0):,.0f}"),
            ("Final Value", f"${summary.get('final_value', 0):,.0f}"),
            ("Terminal Multiplier", f"{summary.get('final_value', 0)/summary.get('initial_capital', 1):.2f}x"),
        ]

        for name, value in metrics:
            print(f"{name:<30} {value:>15}")

    if "yearly_performance" in result:
        print(f"\n{'Year':<8} {'Return':>12} {'Sharpe':>10} {'MDD':>10}")
        print("-"*45)
        for year_data in result["yearly_performance"]:
            year = year_data.get("year", "N/A")
            ret = year_data.get("return", 0) * 100
            sharpe = year_data.get("sharpe", 0)
            mdd = year_data.get("max_drawdown", 0) * 100
            print(f"{year:<8} {ret:>11.1f}% {sharpe:>10.2f} {mdd:>9.1f}%")


def main():
    # Load signals
    signals_path = "/Users/garen.lee/Coding/agentic-openenvolve2-backtest/backtest_tools/long_only_signals_2017_2025_final.csv"

    print("Loading signals...")
    signals = load_signals(signals_path)
    print(f"Loaded {len(signals)} signals")
    print(f"Date range: {signals['reaction_date'].min()} to {signals['reaction_date'].max()}")

    # Run G250 backtest
    result = run_backtest(
        signals,
        leverage=2.5,
        per_trade_cap=0.25,
        holding_days=30,
        stop_loss=0.15,  # 15% stop loss
        initial_capital=100000,
        start_date="2017-01-01",
        end_date="2025-11-01",  # Before API end date limit
    )

    # Print results
    print_results(result)

    # Save full result
    if result:
        output_path = "/Users/garen.lee/Coding/agentic-openenvolve2-backtest/g250_backtest_result.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
