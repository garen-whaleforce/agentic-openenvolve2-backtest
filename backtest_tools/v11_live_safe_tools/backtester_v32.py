"""
backtester_v32.py

Enhanced backtester with leverage, stop-loss, and dynamic position sizing.

New features:
- Target gross exposure (leverage) by regime
- Fixed % or ATR-based stop-loss with conservative gap handling
- Margin borrowing with interest
- Integer share rounding
- Per-trade allocation cap

Based on v1.1-live-safe spec:
- Decision time: earnings reaction day close
- Entry: T+1 open (next trading session open)
- Exit: T+31 close or stop-loss trigger
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd

from calendar_utils import TradingCalendar
from price_providers import PriceProvider, PriceProviderError


@dataclass
class BacktestConfigV32:
    """Enhanced backtest configuration with leverage and stop-loss support."""

    # Calendar and timing
    calendar_name: str = "XNYS"
    entry_lag_sessions: int = 1
    holding_sessions: int = 30
    cap_entries_per_quarter: int = 12
    allow_multiple_positions_same_symbol: bool = False

    # Initial capital
    initial_cash: float = 100000.0

    # Leverage / Target Gross Exposure by regime
    # VIX thresholds: NORMAL <= vix_normal, RISK_OFF <= vix_stress, else STRESS
    target_gross_normal: float = 1.0  # 100% = no leverage
    target_gross_riskoff: float = 0.6
    target_gross_stress: float = 0.0
    vix_normal_threshold: float = 22.0
    vix_stress_threshold: float = 28.0

    # Position sizing
    max_concurrent_positions: int = 12
    sizing_positions: Optional[int] = None  # Sizing denominator (decoupled from max_concurrent)
    per_trade_cap: float = 0.20  # Max 20% of equity per trade
    min_allocation_cash: float = 100.0
    integer_shares: bool = True  # Round down to whole shares

    # Same-day sorting and satellite floor
    sort_column: Optional[str] = None  # e.g., "direction_score" - sort descending (best first)
    satellite_floor: Optional[float] = None  # Min score for positions beyond sizing_positions

    # Tier-based weight multiplier
    tier_weight_col: Optional[str] = None  # Column containing weight multiplier per signal (e.g., "weight_mult")

    # Regime-dynamic D6 weight (overrides tier_weight_col for satellites)
    # Satellites use this weight instead of tier_weight_col when in specified regime
    d6_weight_normal: Optional[float] = None  # D6 weight in NORMAL regime (e.g., 1.0)
    d6_weight_riskoff: Optional[float] = None  # D6 weight in RISK_OFF regime (e.g., 0.5)
    d6_weight_stress: Optional[float] = None  # D6 weight in STRESS regime (e.g., 0.0)

    # Core score-weighted allocation (applies to non-satellite positions)
    # Weight formula: base_weight * (1 + score_weight_k * zscore(sort_column))
    # Clipped to [score_weight_min, score_weight_max]
    score_weight_k: Optional[float] = None  # e.g., 0.3 means ±30% adjustment based on zscore
    score_weight_min: float = 0.7  # Minimum weight multiplier
    score_weight_max: float = 1.5  # Maximum weight multiplier

    # DD-based de-leveraging
    # Reduces new position size when portfolio is in drawdown
    dd_delever_threshold1: Optional[float] = None  # e.g., 0.06 = start reducing at -6% DD
    dd_delever_mult1: float = 0.7  # Multiply allocation by this when DD > threshold1
    dd_delever_threshold2: Optional[float] = None  # e.g., 0.09 = further reduce at -9% DD
    dd_delever_mult2: float = 0.4  # Multiply allocation by this when DD > threshold2
    dd_delever_threshold3: Optional[float] = None  # e.g., 0.12 = stop new entries at -12% DD
    dd_delever_mult3: float = 0.0  # Multiply allocation by 0 (no new entries)

    # Portfolio-level Breaker
    # Trigger: SPY 1d return <= -breaker_spy_threshold OR VIX 1d change >= +breaker_vix_threshold
    # Action modes:
    #   - "reduce": Reduce ALL existing positions to breaker_target_gross at next day open (original)
    #   - "freeze": Only freeze new entries, don't touch existing positions (recommended)
    # During breaker: No new entries allowed
    # Cooldown: breaker_cooldown_days trading sessions
    breaker_spy_threshold: Optional[float] = None  # e.g., 0.03 = trigger on SPY -3%
    breaker_vix_threshold: Optional[float] = None  # e.g., 0.20 = trigger on VIX +20%
    breaker_target_gross: float = 1.0  # Target gross during breaker (1.0 = 100%, 0.5 = 50%)
    breaker_cooldown_days: int = 3  # Days to stay in breaker mode
    breaker_mode: str = "reduce"  # "reduce" = sell to target gross, "freeze" = only block new entries

    # Winner Add-on
    # After min hold sessions, if conditions met, add addon_mult to position
    # Constraints: obey per_trade_cap, max_gross, max 1 add-on per trade, no add-on during breaker
    # Two modes:
    #   - "extension": Add when unrealized PnL >= addon_trigger_pct (追高加碼 - original)
    #   - "pullback": Add when (1) ever reached +addon_trigger_pct AND (2) pulled back addon_pullback_pct from high (回檔加碼 - recommended)
    addon_enabled: bool = False
    addon_min_hold_sessions: int = 5  # Min sessions before add-on eligible
    addon_trigger_pct: float = 0.06  # Unrealized PnL trigger (e.g., 0.06 = +6%)
    addon_mult: float = 0.33  # Add this fraction of original position (e.g., 0.33 = +33%)
    addon_max_per_trade: int = 1  # Max add-ons per trade
    addon_mode: str = "extension"  # "extension" = add at high, "pullback" = add on pullback from high
    addon_pullback_pct: float = 0.03  # For pullback mode: require this pullback from max_price (e.g., 0.03 = -3%)
    addon_d7_only: bool = False  # If True, only D7_CORE positions can receive add-ons

    # Costs (per side)
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    # Margin borrowing
    annual_borrow_rate: float = 0.06  # 6% annual rate for margin

    # Stop-loss
    stop_loss_pct: Optional[float] = None  # e.g., 0.12 = 12% hard stop
    stop_loss_atr_mult: Optional[float] = None  # e.g., 2.5 = 2.5x ATR20
    atr_lookback: int = 20
    addon_stop_pct: Optional[float] = None  # e.g., 0.10 = 10% stop ONLY for add-on leg (original position untouched)

    # Trailing stop (optional)
    trailing_stop_trigger_pct: Optional[float] = None  # e.g., 0.12 = activate after +12%
    trailing_stop_level_pct: Optional[float] = None  # e.g., 0.02 = trail at breakeven - 2%

    # Risk-free (annual)
    annual_rf_rate: float = 0.0

    # Price columns
    entry_price_col: str = "open"
    exit_price_col: str = "close"

    def daily_rf_rate(self) -> float:
        return float(self.annual_rf_rate) / 252.0

    def daily_borrow_rate(self) -> float:
        return float(self.annual_borrow_rate) / 252.0

    def get_sizing_positions(self) -> int:
        """Get sizing denominator (decoupled from max_concurrent_positions)."""
        if self.sizing_positions is not None:
            return self.sizing_positions
        return self.max_concurrent_positions


def _bps_to_frac(bps: float) -> float:
    return float(bps) / 10000.0


def _year_quarter(d: pd.Timestamp) -> Tuple[int, int]:
    return int(d.year), int((d.month - 1) // 3 + 1)


def _get_regime(vix: float, config: BacktestConfigV32) -> str:
    """Determine regime based on VIX value."""
    if vix <= config.vix_normal_threshold:
        return "NORMAL"
    elif vix <= config.vix_stress_threshold:
        return "RISK_OFF"
    else:
        return "STRESS"


def _get_target_gross(regime: str, config: BacktestConfigV32) -> float:
    """Get target gross exposure for given regime."""
    if regime == "NORMAL":
        return config.target_gross_normal
    elif regime == "RISK_OFF":
        return config.target_gross_riskoff
    else:
        return config.target_gross_stress


def _get_d6_weight_for_regime(regime: str, config: BacktestConfigV32) -> float:
    """Get D6 satellite weight multiplier for given regime.

    Returns the regime-specific D6 weight if configured, otherwise 1.0 (full weight).
    """
    if regime == "NORMAL" and config.d6_weight_normal is not None:
        return config.d6_weight_normal
    elif regime == "RISK_OFF" and config.d6_weight_riskoff is not None:
        return config.d6_weight_riskoff
    elif regime == "STRESS" and config.d6_weight_stress is not None:
        return config.d6_weight_stress
    return 1.0  # Default: full weight


def _get_dd_delever_mult(current_dd: float, config: BacktestConfigV32) -> float:
    """Get de-leveraging multiplier based on current drawdown.

    Args:
        current_dd: Current drawdown as positive fraction (e.g., 0.06 for -6%)
        config: Backtest configuration

    Returns:
        Multiplier to apply to new position allocations (0.0 to 1.0)
    """
    # Check thresholds from most severe to least severe
    if config.dd_delever_threshold3 is not None and current_dd >= config.dd_delever_threshold3:
        return config.dd_delever_mult3
    if config.dd_delever_threshold2 is not None and current_dd >= config.dd_delever_threshold2:
        return config.dd_delever_mult2
    if config.dd_delever_threshold1 is not None and current_dd >= config.dd_delever_threshold1:
        return config.dd_delever_mult1
    return 1.0  # No de-leveraging


def _compute_score_weight(
    score: float,
    scores_today: List[float],
    config: BacktestConfigV32,
) -> float:
    """Compute score-weighted allocation multiplier.

    Uses zscore of the signal's score relative to today's entries to adjust weight.

    Args:
        score: This signal's score
        scores_today: All scores for today's entries
        config: Backtest configuration

    Returns:
        Weight multiplier clipped to [score_weight_min, score_weight_max]
    """
    if config.score_weight_k is None or len(scores_today) < 2:
        return 1.0

    # Calculate zscore
    scores_arr = np.array(scores_today)
    mu = np.mean(scores_arr)
    sigma = np.std(scores_arr, ddof=1)

    if sigma == 0 or np.isnan(sigma):
        return 1.0

    zscore = (score - mu) / sigma

    # Apply weight adjustment
    weight = 1.0 + config.score_weight_k * zscore

    # Clip to bounds
    return float(np.clip(weight, config.score_weight_min, config.score_weight_max))


def prepare_signals(signals: pd.DataFrame, earnings_dates_path: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize signals DataFrame.

    If reaction_date is missing but year/quarter are present, will merge from earnings_dates.
    """
    df = signals.copy()

    # Handle trade_long first
    if "trade_long" not in df.columns:
        raise ValueError("Signals missing column: trade_long")

    if df["trade_long"].dtype == object:
        df["trade_long"] = df["trade_long"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    else:
        df["trade_long"] = df["trade_long"].astype(bool)

    df["symbol"] = df["symbol"].astype(str).str.upper()

    # If reaction_date is missing, try to merge from earnings_dates
    if "reaction_date" not in df.columns:
        if "year" not in df.columns or "quarter" not in df.columns:
            raise ValueError("Signals missing reaction_date and year/quarter columns")

        # Try default path
        if earnings_dates_path is None:
            from pathlib import Path
            possible_paths = [
                Path(__file__).parent.parent / "earnings_dates_from_db.csv",
                Path(__file__).parent / "earnings_dates_from_db.csv",
                Path("earnings_dates_from_db.csv"),
            ]
            for p in possible_paths:
                if p.exists():
                    earnings_dates_path = str(p)
                    break

        if earnings_dates_path is None or not Path(earnings_dates_path).exists():
            raise ValueError(f"reaction_date column missing and earnings_dates file not found")

        # Load earnings dates and merge
        dates_df = pd.read_csv(earnings_dates_path)
        dates_df["symbol"] = dates_df["symbol"].astype(str).str.upper()
        dates_df["year"] = dates_df["year"].astype(int)
        dates_df["quarter"] = dates_df["quarter"].astype(int)
        dates_df["reaction_date"] = pd.to_datetime(dates_df["reaction_date"])

        df["year"] = df["year"].astype(int)
        df["quarter"] = df["quarter"].astype(int)

        # Merge
        df = df.merge(
            dates_df[["symbol", "year", "quarter", "reaction_date"]],
            on=["symbol", "year", "quarter"],
            how="left"
        )

        # Report missing
        missing_dates = df["reaction_date"].isna().sum()
        if missing_dates > 0:
            print(f"Warning: {missing_dates} trades missing reaction_date after merge")
            df = df.dropna(subset=["reaction_date"])

    df["reaction_date"] = pd.to_datetime(df["reaction_date"], errors="coerce").dt.normalize()
    if df["reaction_date"].isna().any():
        print(f"Warning: Dropping {df['reaction_date'].isna().sum()} rows with invalid reaction_date")
        df = df.dropna(subset=["reaction_date"])

    # Default sort by date then symbol (deterministic)
    df = df.sort_values(["reaction_date", "symbol"]).reset_index(drop=True)
    return df


def _sort_same_day_entries(
    entries: List[Dict],
    sort_column: Optional[str],
) -> List[Dict]:
    """
    Sort same-day entries by quality score (descending) to ensure core weight goes to best signals.
    Falls back to symbol for deterministic ordering.
    """
    if sort_column is None:
        # Default: sort by symbol for deterministic ordering
        return sorted(entries, key=lambda x: x.get("symbol", ""))

    # Sort by score descending, then symbol for tie-breaking
    def sort_key(entry):
        score = entry.get(sort_column, 0)
        if score is None or (isinstance(score, float) and np.isnan(score)):
            score = -999  # Put NaN at the end
        return (-score, entry.get("symbol", ""))

    return sorted(entries, key=sort_key)


def run_backtest_v32(
    signals: pd.DataFrame,
    price_provider: PriceProvider,
    config: BacktestConfigV32,
    vix_data: Optional[pd.Series] = None,
    spy_data: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, Dict]:
    """
    Enhanced backtest with leverage, stop-loss, and dynamic sizing.

    Args:
        signals: DataFrame with symbol, reaction_date, trade_long
        price_provider: Price data provider
        config: BacktestConfigV32 configuration
        vix_data: Optional VIX series indexed by date (for regime detection and breaker)
        spy_data: Optional SPY close series indexed by date (for breaker trigger)

    Returns:
        nav: Daily NAV series
        trades_df: Trade ledger
        exposure: Daily gross exposure
        stats: Additional statistics dict
    """
    df = prepare_signals(signals)
    cal = TradingCalendar(config.calendar_name)

    # Accept trades based on CAP
    accepted_rows = []
    cap_counter: Dict[Tuple[int, int], int] = {}
    held_symbols_by_quarter: Dict[Tuple[int, int], set] = {}

    for _, r in df.iterrows():
        if not bool(r["trade_long"]):
            continue

        rd = pd.Timestamp(r["reaction_date"]).normalize()
        yq = _year_quarter(rd)
        cap_counter.setdefault(yq, 0)
        held_symbols_by_quarter.setdefault(yq, set())

        # Get VIX for regime-based cap
        vix_value = 15.0  # default if no VIX data
        if vix_data is not None and rd in vix_data.index:
            vix_value = vix_data.loc[rd]
        regime = _get_regime(vix_value, config)

        # Adjust cap based on regime
        if regime == "NORMAL":
            effective_cap = config.cap_entries_per_quarter
        elif regime == "RISK_OFF":
            effective_cap = max(1, config.cap_entries_per_quarter // 2)
        else:  # STRESS
            effective_cap = 0

        if cap_counter[yq] >= effective_cap:
            continue

        sym = str(r["symbol"]).upper()
        if (not config.allow_multiple_positions_same_symbol) and (sym in held_symbols_by_quarter[yq]):
            continue

        cap_counter[yq] += 1
        held_symbols_by_quarter[yq].add(sym)

        # Preserve all original columns plus computed ones
        row_dict = r.to_dict()
        row_dict.update({
            "symbol": sym,
            "reaction_date": rd,
            "vix_at_decision": vix_value,
            "regime": regime,
        })
        accepted_rows.append(row_dict)

    if not accepted_rows:
        raise ValueError("No accepted trades after CAP and regime filtering")

    accepted = pd.DataFrame(accepted_rows)
    accepted["entry_date"] = accepted["reaction_date"].apply(
        lambda d: cal.add_sessions(d, config.entry_lag_sessions)
    )
    accepted["exit_date"] = accepted["entry_date"].apply(
        lambda d: cal.add_sessions(d, config.holding_sessions)
    )

    # Price range
    start = accepted["entry_date"].min()
    end = accepted["exit_date"].max()
    symbols = sorted(accepted["symbol"].unique().tolist())

    # Load prices with buffer for ATR calculation
    atr_buffer = config.atr_lookback + 10
    price_start = cal.add_sessions(start, -atr_buffer) if config.stop_loss_atr_mult else start
    prices = price_provider.get_ohlc(symbols, start=price_start, end=end)

    # Pre-compute ATR if needed
    atr_cache: Dict[str, pd.Series] = {}
    if config.stop_loss_atr_mult:
        for sym in symbols:
            px = prices[sym]
            high = px["high"] if "high" in px.columns else px["close"]
            low = px["low"] if "low" in px.columns else px["close"]
            close = px["close"]
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_cache[sym] = tr.rolling(config.atr_lookback, min_periods=1).mean()

    # Build session index
    sessions = cal.sessions_in_range(start, end)
    nav = pd.Series(index=sessions, dtype=float)
    exposure = pd.Series(index=sessions, dtype=float)

    # Initialize portfolio
    cash = config.initial_cash
    positions: Dict[str, Dict] = {}
    trade_records: List[Dict] = []
    high_water_mark = config.initial_cash  # Track HWM for DD-based de-leveraging

    # Pre-index entries and scheduled exits
    entries_by_date: Dict[pd.Timestamp, List[Dict]] = {}
    scheduled_exits: Dict[pd.Timestamp, List[str]] = {}

    for _, r in accepted.iterrows():
        entries_by_date.setdefault(r["entry_date"], []).append(r.to_dict())
        scheduled_exits.setdefault(r["exit_date"], []).append(r["symbol"])

    commission = _bps_to_frac(config.commission_bps)
    slip = _bps_to_frac(config.slippage_bps)
    per_side_cost = commission + slip

    # Breaker state tracking
    breaker_active = False
    breaker_cooldown_remaining = 0
    breaker_triggered_dates: List[pd.Timestamp] = []

    # Statistics
    stats = {
        "stop_loss_triggered": 0,
        "addon_stop_triggered": 0,
        "scheduled_exits": 0,
        "total_margin_interest": 0.0,
        "max_leverage": 0.0,
        "regime_trades": {"NORMAL": 0, "RISK_OFF": 0, "STRESS": 0},
        "breaker_triggered_count": 0,
        "breaker_shares_sold": 0.0,
        "breaker_notional_reduced": 0.0,
        "addon_count": 0,
        "addon_notional_added": 0.0,
    }

    def get_px(sym: str, d: pd.Timestamp, col: str) -> float:
        dfp = prices[sym]
        if d not in dfp.index:
            raise PriceProviderError(f"Missing price for {sym} on {d.date()}")
        px = float(dfp.loc[d, col])
        if not np.isfinite(px) or px <= 0:
            raise PriceProviderError(f"Bad price for {sym} on {d.date()}: {col}={px}")
        return px

    def check_stop_loss(sym: str, pos: Dict, session: pd.Timestamp) -> Optional[float]:
        """
        Check if stop-loss is triggered. Returns exit price if triggered, None otherwise.
        Uses conservative gap handling: if Open <= stop, exit at Open.
        """
        if config.stop_loss_pct is None and config.stop_loss_atr_mult is None:
            return None

        entry_px = pos["entry_price"]

        # Calculate stop price
        if config.stop_loss_pct is not None:
            stop_price = entry_px * (1 - config.stop_loss_pct)
        elif config.stop_loss_atr_mult is not None:
            # ATR at entry date
            entry_date = pos["entry_date"]
            if sym in atr_cache and entry_date in atr_cache[sym].index:
                atr_val = atr_cache[sym].loc[entry_date]
                stop_price = entry_px - config.stop_loss_atr_mult * atr_val
            else:
                stop_price = entry_px * 0.85  # fallback 15% stop

        # Check trailing stop upgrade
        if config.trailing_stop_trigger_pct is not None:
            current_high = pos.get("max_price", entry_px)
            if current_high >= entry_px * (1 + config.trailing_stop_trigger_pct):
                # Upgrade stop to breakeven or trailing level
                if config.trailing_stop_level_pct is not None:
                    trail_stop = entry_px * (1 - config.trailing_stop_level_pct)
                else:
                    trail_stop = entry_px  # breakeven
                stop_price = max(stop_price, trail_stop)

        # Get OHLC for today
        try:
            px_open = get_px(sym, session, "open")
            px_low = get_px(sym, session, "low") if "low" in prices[sym].columns else px_open
        except PriceProviderError:
            return None

        # Conservative stop-loss execution
        if px_open <= stop_price:
            # Gap through stop - exit at open
            return px_open
        elif px_low <= stop_price:
            # Intraday stop - exit at stop price
            return stop_price

        return None

    def check_addon_stop_loss(sym: str, pos: Dict, session: pd.Timestamp) -> Optional[Tuple[float, float]]:
        """
        Check if add-on leg stop-loss is triggered.
        Only triggers if position has add-on shares and add-on PnL is below threshold.
        Returns (exit_price, addon_shares_to_sell) if triggered, None otherwise.
        Original position is NOT touched.
        """
        if config.addon_stop_pct is None:
            return None

        # Check if position has add-on shares
        addon_count = pos.get("addon_count", 0)
        if addon_count == 0:
            return None

        original_shares = pos.get("original_shares", pos["shares"])
        total_shares = pos["shares"]
        addon_shares = total_shares - original_shares

        if addon_shares <= 0:
            return None

        # Get add-on entry price (use max_price at add-on as proxy since we added on pullback)
        # For pullback add-on, entry is at the pullback price, not the original entry
        # We need to track addon_entry_price - for now use a conservative estimate
        addon_entry_price = pos.get("addon_entry_price", pos.get("max_price", pos["entry_price"]))

        # Get current price
        try:
            px_now = get_px(sym, session, config.exit_price_col)
        except PriceProviderError:
            return None

        # Calculate add-on PnL
        addon_pnl_pct = (px_now - addon_entry_price) / addon_entry_price if addon_entry_price > 0 else 0

        # Check if add-on stop triggered (e.g., -10% from addon entry)
        if addon_pnl_pct < -config.addon_stop_pct:
            return (px_now, addon_shares)

        return None

    # Main simulation loop
    for i, session in enumerate(sessions):
        session = pd.Timestamp(session).normalize()

        # Get VIX for today's regime
        vix_today = 15.0
        if vix_data is not None and session in vix_data.index:
            vix_today = vix_data.loc[session]
        regime_today = _get_regime(vix_today, config)
        target_gross = _get_target_gross(regime_today, config)

        # === BREAKER CHECK ===
        # Check if breaker should trigger based on previous day's market conditions
        breaker_triggered_today = False
        if i > 0 and (config.breaker_spy_threshold is not None or config.breaker_vix_threshold is not None):
            prev_session = pd.Timestamp(sessions[i-1]).normalize()

            # Check SPY return trigger
            if config.breaker_spy_threshold is not None and spy_data is not None:
                if session in spy_data.index and prev_session in spy_data.index:
                    spy_prev = spy_data.loc[prev_session]
                    spy_today = spy_data.loc[session]
                    if spy_prev > 0:
                        spy_return = (spy_today / spy_prev) - 1.0
                        if spy_return <= -config.breaker_spy_threshold:
                            breaker_triggered_today = True

            # Check VIX change trigger
            if not breaker_triggered_today and config.breaker_vix_threshold is not None and vix_data is not None:
                if session in vix_data.index and prev_session in vix_data.index:
                    vix_prev = vix_data.loc[prev_session]
                    if vix_prev > 0:
                        vix_change = (vix_today / vix_prev) - 1.0
                        if vix_change >= config.breaker_vix_threshold:
                            breaker_triggered_today = True

        # If breaker triggered, activate and set cooldown
        if breaker_triggered_today and not breaker_active:
            breaker_active = True
            breaker_cooldown_remaining = config.breaker_cooldown_days
            breaker_triggered_dates.append(session)
            stats["breaker_triggered_count"] += 1

        # === BREAKER EXECUTION ===
        # If breaker is active and mode is "reduce", reduce positions to target gross at open
        # If mode is "freeze", skip this section (only block new entries, handled later)
        if breaker_active and len(positions) > 0 and config.breaker_mode == "reduce":
            # Calculate current position value at open
            pos_val_open = 0.0
            for sym, pos in positions.items():
                try:
                    px_open = get_px(sym, session, "open")
                    pos_val_open += float(pos["shares"]) * px_open
                except:
                    pos_val_open += float(pos["shares"]) * float(pos["entry_price"])

            # Calculate equity at open (approximate with previous close cash + current positions)
            equity_open = cash + pos_val_open
            current_gross = pos_val_open / equity_open if equity_open > 0 else 0

            # If current gross exceeds breaker target, reduce all positions proportionally
            if current_gross > config.breaker_target_gross and pos_val_open > 0:
                reduction_ratio = config.breaker_target_gross / current_gross
                sell_ratio = 1.0 - reduction_ratio  # How much of each position to sell

                for sym in list(positions.keys()):
                    pos = positions[sym]
                    shares_current = float(pos["shares"])
                    shares_to_sell = int(shares_current * sell_ratio)

                    if shares_to_sell > 0:
                        try:
                            px_open = get_px(sym, session, "open")
                        except:
                            continue

                        proceeds = shares_to_sell * px_open
                        proceeds_net = proceeds * (1.0 - per_side_cost)
                        cash += proceeds_net

                        # Update position
                        pos["shares"] = shares_current - shares_to_sell
                        # Track invested_cash proportionally
                        pos["invested_cash"] = float(pos["invested_cash"]) * (1.0 - sell_ratio)

                        stats["breaker_shares_sold"] += shares_to_sell
                        stats["breaker_notional_reduced"] += proceeds

                        # If position fully sold, remove it
                        if pos["shares"] <= 0:
                            positions.pop(sym)

        # Decrement breaker cooldown
        if breaker_cooldown_remaining > 0:
            breaker_cooldown_remaining -= 1
            if breaker_cooldown_remaining == 0:
                breaker_active = False

        # Update max price for trailing stops
        for sym, pos in positions.items():
            try:
                px_high = get_px(sym, session, "high") if "high" in prices[sym].columns else get_px(sym, session, "close")
                pos["max_price"] = max(pos.get("max_price", pos["entry_price"]), px_high)
            except:
                pass

        # 1) Check stop-losses BEFORE scheduled exits
        stops_to_execute = []
        for sym, pos in list(positions.items()):
            stop_exit_px = check_stop_loss(sym, pos, session)
            if stop_exit_px is not None:
                stops_to_execute.append((sym, stop_exit_px))

        # Execute stop-losses
        for sym, exit_px in stops_to_execute:
            if sym not in positions:
                continue
            pos = positions.pop(sym)
            shares = float(pos["shares"])
            entry_px = float(pos["entry_price"])
            proceeds = shares * exit_px
            proceeds_net = proceeds * (1.0 - per_side_cost)
            cash += proceeds_net

            gross_ret = exit_px / entry_px - 1.0
            invested = float(pos["invested_cash"])
            net_ret = (proceeds_net / invested) - 1.0 if invested > 0 else np.nan

            trade_records.append({
                "symbol": sym,
                "reaction_date": pos["reaction_date"],
                "entry_date": pos["entry_date"],
                "exit_date": session,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "shares": shares,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "invested_cash": invested,
                "exit_reason": "stop_loss",
                "regime": pos.get("regime", "UNKNOWN"),
                "is_satellite": pos.get("is_satellite", False),
            })
            stats["stop_loss_triggered"] += 1

            # Remove from scheduled exits
            if pos["scheduled_exit_date"] in scheduled_exits:
                if sym in scheduled_exits[pos["scheduled_exit_date"]]:
                    scheduled_exits[pos["scheduled_exit_date"]].remove(sym)

        # 1.5) Check add-on leg stop-losses (only sell add-on shares, keep original position)
        addon_stops_to_execute = []
        for sym, pos in list(positions.items()):
            result = check_addon_stop_loss(sym, pos, session)
            if result is not None:
                addon_stops_to_execute.append((sym, result[0], result[1]))

        # Execute add-on stop-losses (partial exit)
        for sym, exit_px, addon_shares in addon_stops_to_execute:
            if sym not in positions:
                continue
            pos = positions[sym]

            # Sell only add-on shares
            proceeds = addon_shares * exit_px
            proceeds_net = proceeds * (1.0 - per_side_cost)
            cash += proceeds_net

            # Update position (keep original shares)
            original_shares = pos.get("original_shares", pos["shares"])
            pos["shares"] = original_shares  # Reset to original position
            pos["addon_count"] = 0  # Reset add-on count
            addon_entry_px = pos.get("addon_entry_price", exit_px)
            addon_invested = addon_shares * addon_entry_px * (1.0 + per_side_cost)
            pos["invested_cash"] = float(pos["invested_cash"]) - addon_invested

            stats["addon_stop_triggered"] += 1

        # 2) Execute scheduled exits at CLOSE
        if session in scheduled_exits:
            for sym in list(scheduled_exits[session]):
                if sym not in positions:
                    continue
                px_exit = get_px(sym, session, config.exit_price_col)
                pos = positions.pop(sym)
                shares = float(pos["shares"])
                entry_px = float(pos["entry_price"])
                proceeds = shares * px_exit
                proceeds_net = proceeds * (1.0 - per_side_cost)
                cash += proceeds_net

                gross_ret = px_exit / entry_px - 1.0
                invested = float(pos["invested_cash"])
                net_ret = (proceeds_net / invested) - 1.0 if invested > 0 else np.nan

                trade_records.append({
                    "symbol": sym,
                    "reaction_date": pos["reaction_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": session,
                    "entry_price": entry_px,
                    "exit_price": px_exit,
                    "shares": shares,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "invested_cash": invested,
                    "exit_reason": "scheduled",
                    "regime": pos.get("regime", "UNKNOWN"),
                    "is_satellite": pos.get("is_satellite", False),
                })
                stats["scheduled_exits"] += 1

        # 3) Calculate equity before entries
        if i == 0:
            equity = cash
        else:
            prev = pd.Timestamp(sessions[i-1]).normalize()
            pos_val = 0.0
            for sym, pos in positions.items():
                try:
                    px_prev = get_px(sym, prev, "close")
                    pos_val += float(pos["shares"]) * px_prev
                except:
                    pos_val += float(pos["shares"]) * float(pos["entry_price"])
            equity = cash + pos_val

        # Update high water mark for DD-based de-leveraging
        high_water_mark = max(high_water_mark, equity)
        current_dd = (high_water_mark - equity) / high_water_mark if high_water_mark > 0 else 0

        # === WINNER ADD-ON ===
        # Add to winning positions if: held >= min_hold, unrealized PnL >= trigger, not during breaker
        if config.addon_enabled and not breaker_active and len(positions) > 0:
            # Calculate current position values and gross for add-on constraints
            pos_val_for_addon = 0.0
            for sym, pos in positions.items():
                try:
                    px_now = get_px(sym, session, "open")
                except:
                    px_now = pos["entry_price"]
                pos_val_for_addon += float(pos["shares"]) * px_now

            current_gross_for_addon = pos_val_for_addon / equity if equity > 0 else 0

            for sym in list(positions.keys()):
                pos = positions[sym]

                # Check if already reached max add-ons for this trade
                addon_count = pos.get("addon_count", 0)
                if addon_count >= config.addon_max_per_trade:
                    continue

                # Check D7-only restriction
                if config.addon_d7_only:
                    tier = pos.get("tier", "UNKNOWN")
                    if not tier.startswith("D7"):
                        continue

                # Check minimum hold period
                entry_date = pos["entry_date"]
                sessions_held = 0
                for j in range(i + 1):
                    if pd.Timestamp(sessions[j]).normalize() >= entry_date:
                        sessions_held += 1
                if sessions_held < config.addon_min_hold_sessions:
                    continue

                # Check unrealized PnL and add-on mode
                try:
                    px_now = get_px(sym, session, "open")
                except:
                    continue

                entry_px = float(pos["entry_price"])
                max_price = float(pos.get("max_price", entry_px))
                unrealized_pnl = (px_now / entry_px) - 1.0
                max_pnl = (max_price / entry_px) - 1.0

                # Track if position ever reached trigger threshold
                if max_pnl >= config.addon_trigger_pct:
                    pos["ever_reached_trigger"] = True

                if config.addon_mode == "extension":
                    # Original mode: add when current PnL >= trigger (追高加碼)
                    if unrealized_pnl < config.addon_trigger_pct:
                        continue
                elif config.addon_mode == "pullback":
                    # Pullback mode: add when (1) ever reached trigger AND (2) pulled back from high (回檔加碼)
                    if not pos.get("ever_reached_trigger", False):
                        continue
                    # Check pullback from max_price
                    pullback_from_high = (max_price - px_now) / max_price if max_price > 0 else 0
                    if pullback_from_high < config.addon_pullback_pct:
                        continue
                    # Also require still in profit (don't add to losing positions)
                    if unrealized_pnl < 0:
                        continue
                else:
                    # Unknown mode, skip
                    continue

                # Calculate add-on size based on original position
                original_shares = float(pos.get("original_shares", pos["shares"]))
                addon_shares_target = original_shares * config.addon_mult

                if config.integer_shares:
                    addon_shares = math.floor(addon_shares_target)
                    if addon_shares < 1:
                        continue
                else:
                    addon_shares = addon_shares_target

                addon_notional = addon_shares * px_now

                # Check per_trade_cap constraint
                current_invested = float(pos["invested_cash"])
                max_per_trade = config.per_trade_cap * equity
                if current_invested + addon_notional > max_per_trade:
                    # Reduce addon to fit within cap
                    addon_notional = max_per_trade - current_invested
                    if addon_notional <= 0:
                        continue
                    addon_shares = addon_notional / px_now
                    if config.integer_shares:
                        addon_shares = math.floor(addon_shares)
                        if addon_shares < 1:
                            continue
                        addon_notional = addon_shares * px_now

                # Check max_gross constraint (use target_gross as ceiling)
                if current_gross_for_addon + (addon_notional / equity) > target_gross:
                    continue

                # Execute add-on
                addon_cost = addon_notional * (1.0 + per_side_cost)
                cash -= addon_cost

                pos["shares"] = float(pos["shares"]) + addon_shares
                pos["invested_cash"] = current_invested + addon_cost
                pos["addon_count"] = addon_count + 1
                pos["addon_entry_price"] = px_now  # Track add-on entry price for add-on stop-loss
                if "original_shares" not in pos:
                    pos["original_shares"] = float(pos["shares"]) - addon_shares

                stats["addon_count"] += 1
                stats["addon_notional_added"] += addon_notional

                # Update gross for next add-on check
                current_gross_for_addon += addon_notional / equity

        # 4) Execute entries at OPEN with dynamic sizing
        # Note: No new entries during breaker cooldown
        if session in entries_by_date and target_gross > 0 and not breaker_active:
            # Calculate current exposure
            pos_val_now = 0.0
            for sym, pos in positions.items():
                try:
                    px_now = get_px(sym, session, "open")
                except:
                    px_now = pos["entry_price"]
                pos_val_now += float(pos["shares"]) * px_now

            current_gross = pos_val_now / equity if equity > 0 else 0
            remaining_gross = max(0, target_gross - current_gross)
            remaining_slots = config.max_concurrent_positions - len(positions)

            # NEW: Use sizing_positions as denominator (decoupled from max_concurrent)
            sizing_slots = config.get_sizing_positions()

            # Get DD-based de-leveraging multiplier
            dd_delever_mult = _get_dd_delever_mult(current_dd, config)

            if remaining_slots > 0 and remaining_gross > 0 and dd_delever_mult > 0:
                entries_today = entries_by_date[session]

                # Sort same-day entries by quality score (best first for core weight)
                entries_today = _sort_same_day_entries(entries_today, config.sort_column)

                # Pre-compute scores for score-weighted allocation
                scores_today = []
                if config.score_weight_k is not None and config.sort_column is not None:
                    for e in entries_today:
                        s = e.get(config.sort_column, 0)
                        if s is not None and not (isinstance(s, float) and np.isnan(s)):
                            scores_today.append(float(s))

                # Allocation uses sizing_slots (fixed denominator) instead of remaining_slots
                # This prevents dilution when CAP/MaxPos increases but sizing stays constant
                alloc_per_slot = (target_gross * equity) / sizing_slots

                # Calculate remaining notional to prevent overshooting target gross
                target_notional = target_gross * equity
                current_notional = pos_val_now
                remaining_notional = max(0, target_notional - current_notional)

                # Track how many positions we've added today for satellite floor check
                positions_added_today = 0

                for r in entries_today:
                    if len(positions) >= config.max_concurrent_positions:
                        break

                    sym = r["symbol"]
                    if (not config.allow_multiple_positions_same_symbol) and (sym in positions):
                        continue

                    # Satellite floor check: if we're beyond core positions, require minimum score
                    current_position_count = len(positions) + positions_added_today
                    is_satellite = current_position_count >= sizing_slots
                    if is_satellite and config.satellite_floor is not None and config.sort_column is not None:
                        score = r.get(config.sort_column, 0)
                        if score is None or (isinstance(score, float) and np.isnan(score)):
                            score = 0
                        if score < config.satellite_floor:
                            continue  # Skip low-quality satellite positions

                    # Calculate allocation with multiple adjustments:
                    # 1. alloc_per_slot (sizing-based)
                    # 2. tier weight / regime-dynamic D6 weight for satellites
                    # 3. score-weighted adjustment for core positions
                    # 4. DD-based de-leveraging
                    # 5. per_trade_cap (single position cap)
                    # 6. remaining_notional (don't overshoot gross target)
                    base_alloc = alloc_per_slot

                    # Apply tier weight or regime-dynamic D6 weight
                    if is_satellite:
                        # For satellites: use regime-dynamic D6 weight if configured
                        regime_d6_weight = _get_d6_weight_for_regime(regime_today, config)
                        if regime_d6_weight < 1.0:
                            # Regime-dynamic takes precedence
                            base_alloc = alloc_per_slot * regime_d6_weight
                        elif config.tier_weight_col is not None:
                            # Fall back to tier_weight_col
                            weight_mult = r.get(config.tier_weight_col, 1.0)
                            if weight_mult is None or (isinstance(weight_mult, float) and np.isnan(weight_mult)):
                                weight_mult = 1.0
                            base_alloc = alloc_per_slot * weight_mult
                    else:
                        # For core positions: apply score-weighted allocation
                        if config.score_weight_k is not None and config.sort_column is not None:
                            score = r.get(config.sort_column, 0)
                            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                                score_weight = _compute_score_weight(float(score), scores_today, config)
                                base_alloc = alloc_per_slot * score_weight
                        # Also apply tier weight if configured (multiplicative)
                        if config.tier_weight_col is not None:
                            weight_mult = r.get(config.tier_weight_col, 1.0)
                            if weight_mult is None or (isinstance(weight_mult, float) and np.isnan(weight_mult)):
                                weight_mult = 1.0
                            base_alloc = base_alloc * weight_mult

                    # Apply DD-based de-leveraging
                    base_alloc = base_alloc * dd_delever_mult

                    invest = min(
                        base_alloc,
                        config.per_trade_cap * equity,
                        remaining_notional,
                    )
                    if invest < config.min_allocation_cash:
                        continue

                    px_entry = get_px(sym, session, config.entry_price_col)
                    invest_net = invest * (1.0 - per_side_cost)

                    if config.integer_shares:
                        shares = math.floor(invest_net / px_entry)
                        if shares < 1:
                            continue
                        actual_invest = shares * px_entry / (1.0 - per_side_cost)
                    else:
                        shares = invest_net / px_entry
                        actual_invest = invest

                    cash -= actual_invest
                    positions[sym] = {
                        "shares": shares,
                        "original_shares": shares,  # Track for add-on calculation
                        "entry_price": px_entry,
                        "entry_date": session,
                        "scheduled_exit_date": r["exit_date"],
                        "reaction_date": r["reaction_date"],
                        "invested_cash": actual_invest,
                        "max_price": px_entry,
                        "regime": r.get("regime", regime_today),
                        "is_satellite": is_satellite,  # Track if this is a satellite position
                        "addon_count": 0,  # Track number of add-ons
                        "tier": r.get("trade_long_tier", "UNKNOWN"),  # Track tier for D7-only add-on
                    }
                    stats["regime_trades"][r.get("regime", regime_today)] += 1
                    positions_added_today += 1

                    # Update remaining_notional after each entry to prevent overshoot
                    remaining_notional -= shares * px_entry

        # 5) Apply margin interest if cash is negative
        if cash < 0:
            margin_interest = abs(cash) * config.daily_borrow_rate()
            cash -= margin_interest
            stats["total_margin_interest"] += margin_interest

        # 6) Value NAV at CLOSE
        pos_val = 0.0
        for sym, pos in positions.items():
            px_close = get_px(sym, session, "close")
            pos_val += float(pos["shares"]) * px_close

        nav_close = cash + pos_val
        nav.loc[session] = nav_close

        gross_exposure = pos_val / nav_close if nav_close > 0 else 0
        exposure.loc[session] = gross_exposure
        stats["max_leverage"] = max(stats["max_leverage"], gross_exposure)

    # Add breaker dates to stats
    stats["breaker_triggered_dates"] = [d.strftime("%Y-%m-%d") for d in breaker_triggered_dates]

    # Create trades DataFrame
    trades_df = pd.DataFrame(trade_records)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
        trades_df["ret_pct"] = trades_df["net_ret"] * 100.0

    return nav, trades_df, exposure, stats
