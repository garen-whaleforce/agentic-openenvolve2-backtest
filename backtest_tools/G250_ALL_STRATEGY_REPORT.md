# G250_All Earnings Strategy - Complete Report

**Report Date**: 2026-01-04
**Backtest Period**: 2017-01-31 to 2025-08-21 (~8.5 years)
**Strategy Version**: v3.2 with Pullback Add-on + Freeze Breaker

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Return** | 1,243.1% |
| **Terminal Multiplier** | 13.43x |
| **CAGR** | 35.0% |
| **Sharpe Ratio** | 2.16 |
| **Max Drawdown** | -14.4% |
| **Total Trades** | 220 |
| **Win Rate** | ~65% (estimated) |

---

## Strategy Overview

### Core Concept
G250_All is a leveraged long-only earnings momentum strategy that:
1. Identifies stocks with strong positive earnings surprises
2. Enters positions the day after earnings announcement
3. Holds for 30 trading days
4. Uses 250% leverage in normal market conditions
5. Adds to winning positions on pullbacks

### Signal Selection Criteria

#### D7_CORE (Primary Tier - 76% of signals)
- EPS Surprise > 0
- Revenue Surprise > 0
- Earnings Day Return > 0 (positive gap)
- Plus 4 additional strict conditions

#### D6_STRICT (Satellite Tier - 24% of signals)
- 6 strict conditions
- Lower priority than D7_CORE
- Used to fill capacity when D7_CORE signals are scarce

---

## Position Sizing & Leverage

### Leverage by Market Regime

| VIX Level | Regime | Target Gross | Action |
|-----------|--------|--------------|--------|
| VIX < 22 | NORMAL | **250%** | Full leverage |
| VIX 22-28 | RISK_OFF | 100% | Reduce leverage |
| VIX > 28 | STRESS | 0% | No new positions |

### Position Limits

| Parameter | Value | Description |
|-----------|-------|-------------|
| Per Trade Cap | 25% | Max single position size |
| Sizing Positions | 12 | Denominator for base allocation |
| Max Concurrent | 24 | Maximum simultaneous positions |
| Entries Per Quarter | 24 | Quarterly entry limit |

### Base Allocation Formula
```
base_allocation = equity * target_gross / sizing_positions
                = equity * 2.50 / 12
                = equity * 20.8%

actual_allocation = min(base_allocation, equity * 25%)
```

---

## Entry & Exit Rules

### Entry Conditions
1. Signal meets D7_CORE or D6_STRICT criteria
2. Enter at **market open** on reaction_date + 1 trading day
3. VIX regime allows new positions (not STRESS)
4. Breaker not active
5. Position not already held

### Exit Conditions
1. **Scheduled Exit**: After 30 trading sessions (primary)
2. **No Stop-Loss**: Strategy does not use stop-loss orders

### Timing Parameters

| Parameter | Value |
|-----------|-------|
| Entry Lag | +1 trading day after earnings |
| Holding Period | 30 trading days (~6 weeks) |

---

## Risk Management Mechanisms

### 1. Freeze Breaker (Circuit Breaker)

**Purpose**: Prevent new entries during market panic without forced selling

| Parameter | Value | Description |
|-----------|-------|-------------|
| SPY Threshold | -4% | Trigger on SPY single-day drop >= 4% |
| VIX Threshold | +30% | Trigger on VIX single-day spike >= 30% |
| Cooldown Period | 1 day | Freeze duration |
| Mode | **freeze** | Only block new entries, don't sell existing |

**Trigger Logic**:
```
IF (SPY_return <= -4%) OR (VIX_change >= +30%):
    Activate Breaker
    Block new entries for 1 day
    DO NOT sell existing positions
```

**Historical Triggers**: 28 times in 8.5 years (~3.3 per year)

### 2. VIX Regime-Based Leverage

Automatically reduces leverage as VIX increases:
- VIX < 22: Full 250% leverage
- VIX 22-28: Reduced to 100%
- VIX > 28: No new positions (0%)

---

## Pullback Add-on Mechanism

**Purpose**: Add to winning positions at better prices after pullback

### Add-on Conditions

| Parameter | Value | Description |
|-----------|-------|-------------|
| Mode | **pullback** | Add on dip, not at highs |
| Trigger Threshold | +6% | Position must reach +6% unrealized gain |
| Pullback Required | -3% | Must pull back 3% from max price |
| Still Profitable | Required | Current price must still be above entry |
| Min Hold Sessions | 5 days | Must hold at least 5 days before eligible |
| Add-on Size | 33% | Add 33% of original position |
| Max Add-ons | 1 per trade | Only one add-on allowed |

### Add-on Logic
```
IF position.ever_reached_trigger (+6%):
    IF pullback_from_high >= 3%:
        IF current_price > entry_price:
            IF held >= 5 sessions:
                IF gross_exposure < target_gross:
                    ADD 33% to position
```

**Historical Add-ons**: 116 times in 220 trades (52.7% add-on rate)

---

## Transaction Costs

| Cost Type | Rate | Description |
|-----------|------|-------------|
| Commission | 10 bps (0.1%) | Per side |
| Slippage | 10 bps (0.1%) | Per side |
| **Total Per Side** | 20 bps | |
| **Round Trip** | 40 bps | Entry + Exit |
| Margin Rate | 6% annual | For borrowed capital |

---

## Performance Results

### Overall Performance (2017-2025)

| Metric | Value |
|--------|-------|
| Initial Capital | $100,000 |
| Final Value | $1,343,127 |
| Total Return | 1,243.1% |
| Terminal Multiplier | 13.43x |
| CAGR | 35.0% |
| Annualized Volatility | 16.2% |
| Sharpe Ratio | 2.16 |
| Max Drawdown | -14.4% |
| Total Trades | 220 |
| Add-on Count | 116 |
| Breaker Triggers | 28 |

### Yearly Performance Breakdown

| Year | Return | ARR | Volatility | Sharpe | MDD | Trades |
|------|--------|-----|------------|--------|-----|--------|
| 2017 | 46.9% | 52.1% | 13.6% | 3.83 | -3.8% | ~35 |
| 2018 | 11.8% | 11.9% | 22.0% | 0.54 | -12.3% | ~36 |
| 2019 | 5.7% | 5.7% | 3.4% | 1.68 | -1.2% | ~3 |
| 2020 | 8.7% | 8.7% | 10.4% | 0.83 | -5.1% | ~14 |
| 2021 | 81.2% | 81.2% | 20.8% | 3.90 | -8.2% | ~36 |
| 2022 | -6.5% | -6.5% | 11.8% | -0.55 | -12.7% | ~37 |
| 2023 | 143.6% | 145.3% | 24.6% | 5.91 | -8.0% | ~40 |
| 2024 | 64.0% | 64.0% | 16.9% | 3.80 | -7.3% | ~33 |
| 2025* | 6.2% | 8.3% | 4.7% | 1.77 | -3.3% | ~5 |

*2025 data through August only (~160 trading days)

### Performance Analysis by Year Type

#### Best Years
1. **2023**: +143.6% (Sharpe 5.91) - Strong earnings season, low VIX
2. **2021**: +81.2% (Sharpe 3.90) - Post-COVID recovery
3. **2024**: +64.0% (Sharpe 3.80) - AI/Tech earnings boom

#### Challenging Years
1. **2022**: -6.5% (only losing year) - Fed rate hikes, bear market
2. **2018**: +11.8% (Sharpe 0.54) - Q4 market selloff
3. **2020**: +8.7% (Sharpe 0.83) - COVID crash impact

---

## Sample Size Analysis

### Signal Distribution

| Year | Total Signals | D7_CORE | D6_STRICT | Avg/Quarter |
|------|---------------|---------|-----------|-------------|
| 2017 | 39 | 27 | 12 | 9.8 |
| 2018 | 40 | 25 | 15 | 10.0 |
| 2019 | 3 | 1 | 2 | 0.8 |
| 2020 | 16 | 14 | 2 | 4.0 |
| 2021 | 40 | 34 | 6 | 10.0 |
| 2022 | 41 | 30 | 11 | 10.2 |
| 2023 | 45 | 40 | 5 | 11.2 |
| 2024 | 37 | 25 | 12 | 9.2 |
| 2025 | 5 | 5 | 0 | 1.2 |
| **Total** | **266** | **201** | **65** | **7.4** |

### Execution Statistics

| Metric | Value |
|--------|-------|
| Total Signals | 266 |
| Executed Trades | 220 |
| Execution Rate | 82.7% |
| Avg Trades/Year | 25.9 |

### Statistical Significance Notes

- **220 trades** provides reasonable statistical significance for overall strategy evaluation
- Individual year results have wider confidence intervals due to smaller sample sizes
- **2019 warning**: Only 3 signals - results not statistically meaningful
- **2025 incomplete**: Only 5 signals through August

---

## Risk Factors & Limitations

### Strategy Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Leverage Risk** | 250% leverage amplifies losses 2.5x | VIX regime reduction |
| **Systemic Risk** | Market-wide selloffs hurt all positions | Freeze breaker |
| **Concentration** | Max 24 positions may cluster by sector | Quarterly entry limit |
| **No Stop-Loss** | Individual positions can have large losses | Short 30-day holding |
| **Margin Cost** | 6% annual borrowing cost | Factored into returns |

### Historical Drawdowns

| Period | Drawdown | Cause | Recovery |
|--------|----------|-------|----------|
| 2018 Q4 | -12.3% | Fed rate fears | 3 months |
| 2020 Mar | -5.1% | COVID crash | 2 months |
| 2022 Full Year | -12.7% | Fed hiking cycle | 8 months |

### Limitations

1. **Backtest Only**: Results are hypothetical, not live trading
2. **Sample Size**: 220 trades may not capture all market conditions
3. **Execution Assumptions**: Assumes perfect fills at open prices
4. **No Tax Consideration**: Returns are pre-tax
5. **Signal Quality**: D7/D6 criteria may degrade over time

---

## Configuration Summary (Python)

```python
BacktestConfigV32(
    # Capital
    initial_cash=100000.0,

    # Leverage
    target_gross_normal=2.50,      # 250%
    target_gross_riskoff=1.0,      # 100%
    target_gross_stress=0.0,       # 0%
    vix_normal_threshold=22.0,
    vix_stress_threshold=28.0,

    # Position Limits
    per_trade_cap=0.25,            # 25%
    max_concurrent_positions=24,
    sizing_positions=12,
    cap_entries_per_quarter=24,

    # Timing
    entry_lag_sessions=1,
    holding_sessions=30,

    # Costs
    commission_bps=10.0,
    slippage_bps=10.0,
    annual_borrow_rate=0.06,

    # Freeze Breaker
    breaker_spy_threshold=0.04,
    breaker_vix_threshold=0.30,
    breaker_cooldown_days=1,
    breaker_mode="freeze",

    # Pullback Add-on
    addon_enabled=True,
    addon_mode="pullback",
    addon_trigger_pct=0.06,
    addon_pullback_pct=0.03,
    addon_mult=0.33,
    addon_min_hold_sessions=5,
    addon_max_per_trade=1,
    addon_d7_only=False,
)
```

---

## Comparison with Alternative Configurations

| Config | Terminal | CAGR | Sharpe | MDD | Add-ons |
|--------|----------|------|--------|-----|---------|
| G200_All | 8.75x | 28.5% | 2.14 | -11.5% | 120 |
| G200_D7only | 8.75x | 28.5% | 2.17 | -10.9% | 94 |
| G225_All | 10.64x | 31.4% | 2.14 | -12.8% | 119 |
| G225_D7only | 10.62x | 31.4% | 2.17 | -12.3% | 93 |
| **G250_All** | **13.43x** | **35.0%** | **2.16** | **-14.4%** | **116** |
| G250_D7only | 13.43x | 35.0% | 2.19 | -14.1% | 90 |

### Key Observations

1. **Higher leverage = Higher returns but worse MDD**
   - G200: 8.75x terminal, -11.5% MDD
   - G250: 13.43x terminal, -14.4% MDD

2. **D7-only add-on slightly improves Sharpe** (2.19 vs 2.16)
   - Fewer add-ons (90 vs 116) but marginally better risk-adjusted returns

3. **G250_All chosen for maximum total return**
   - Accepts -14.4% MDD for 35% CAGR

---

## Daily Operation Workflow

```
Each Trading Day at Market Open:

1. CHECK VIX REGIME
   ├─ VIX < 22  → NORMAL  → Target 250% gross
   ├─ VIX 22-28 → RISK_OFF → Target 100% gross
   └─ VIX > 28  → STRESS   → No new entries

2. CHECK BREAKER STATUS
   ├─ Yesterday SPY dropped ≥4%? → Freeze 1 day
   └─ Yesterday VIX spiked ≥30%? → Freeze 1 day

3. PROCESS SCHEDULED EXITS
   └─ Close positions held ≥30 sessions

4. CHECK ADD-ON OPPORTUNITIES
   └─ For each position:
      ├─ Ever reached +6%? AND
      ├─ Pulled back ≥3% from high? AND
      ├─ Still profitable? AND
      ├─ Held ≥5 sessions? AND
      └─ Room under gross cap?
          → Add 33% to position

5. EXECUTE NEW ENTRIES
   └─ If breaker not active AND regime allows:
      ├─ Sort signals by earnings_day_return
      ├─ Allocate per sizing formula
      └─ Enter at market open
```

---

## Appendix: File References

| File | Description |
|------|-------------|
| `backtester_v32.py` | Core backtesting engine |
| `run_gross_addon_grid.py` | Grid test runner |
| `long_only_signals_2017_2025_final.csv` | Signal data |
| `price_providers.py` | Price data interface |
| `vix_data.csv` | VIX historical data |
| `gross_addon_grid_results.csv` | Grid test results |

---

*Report generated for NotebookLM analysis*
