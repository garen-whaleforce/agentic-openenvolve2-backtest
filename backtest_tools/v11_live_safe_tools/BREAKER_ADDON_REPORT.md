# Portfolio-level Breaker + Winner Add-on Test Report

**Date**: 2026-01-04
**Test Period**: 2017-02 to 2024-10 (~7.7 years)
**Base Strategy**: D7_CORE + D6_STRICT with 200% leverage (no VIX regime throttle)

---

## 1. Executive Summary

### Key Findings

| Mechanism | Impact on Return | Impact on MDD | Recommendation |
|-----------|-----------------|---------------|----------------|
| **Breaker (SPY -5%)** | -57.9% | +0.6% (better) | Consider for live |
| **Breaker (SPY -4%)** | -73.2% | +0.6% (better) | Secondary option |
| **Breaker (SPY -3%)** | -170.7% | +0.6% (better) | Too aggressive |
| **Add-on (+6%)** | -10.5% | -1.1% (worse) | Not recommended |
| **Add-on (+8%)** | -27.1% | -1.4% (worse) | Not recommended |

**Conclusion**: Both mechanisms reduce returns without meaningful MDD improvement. The breaker triggers too frequently even with conservative thresholds, and add-on increases drawdown risk.

---

## 2. Test Results

### 2.1 Full Results Table

| Config | Total Return | CAGR | Sharpe | MDD | Trades | Win Rate | Breakers | Add-ons |
|--------|-------------|------|--------|-----|--------|----------|----------|---------|
| **1_Base** | 629.1% | 28.8% | 2.22 | -10.5% | 218 | 77.1% | 0 | 0 |
| 2_Breaker_1.0_SPY4 | 555.9% | 27.0% | 2.14 | -10.0% | 215 | 76.7% | 21 | 0 |
| 3_Breaker_0.5_SPY4 | 496.8% | 25.5% | 2.11 | -10.1% | 216 | 76.9% | 21 | 0 |
| 4_Breaker_1.0_SPY3 | 458.4% | 24.5% | 2.07 | -9.9% | 210 | 76.2% | 52 | 0 |
| 5_Addon_Only | 618.6% | 28.5% | 2.09 | -11.7% | 216 | 75.5% | 0 | 147 |
| 6_Breaker+Addon_SPY4 | 550.3% | 26.9% | 2.02 | -11.4% | 214 | 75.2% | 21 | 145 |
| **7_Breaker_1.0_SPY5** | 571.3% | 27.4% | 2.16 | -9.9% | 217 | 77.0% | 13 | 0 |
| 8_Addon_8pct | 602.0% | 28.1% | 2.08 | -11.9% | 216 | 75.0% | 0 | 123 |

### 2.2 Breaker Trigger Analysis

| Threshold | Triggers | Avg/Year | Return Impact |
|-----------|----------|----------|---------------|
| SPY -3% OR VIX +20% | 52 | 6.8 | -170.7% |
| SPY -4% OR VIX +30% | 21 | 2.7 | -73.2% |
| SPY -5% OR VIX +40% | 13 | 1.7 | -57.9% |

**Breaker Trigger Dates (SPY -5%):**
- 2017-05-17, 2017-08-10
- 2018-02-05, 2018-10-10
- 2020-02-24, 2020-03-09, 2020-03-12, 2020-03-16, 2020-03-18
- 2020-06-11, 2022-05-09, 2022-05-18, 2022-06-13

### 2.3 Add-on Analysis

| Trigger | Add-ons | Avg/Year | Return Impact | MDD Impact |
|---------|---------|----------|---------------|------------|
| +6% unrealized | 147 | 19.1 | -10.5% | -1.1% worse |
| +8% unrealized | 123 | 16.0 | -27.1% | -1.4% worse |

**Problem**: Add-on increases position size in winners, but often these winners reverse, leading to larger losses. The net effect is worse drawdown without compensating return improvement.

---

## 3. Mechanism Analysis

### 3.1 Why Breaker Hurts Returns

1. **Timing mismatch**: Market drops often occur during earnings seasons when our signals are active. Breaker forces us to sell positions at the worst time.

2. **Recovery opportunity cost**: After breaker triggers, positions are reduced. Even though normal leverage resumes after 3-day cooldown, we miss the initial recovery bounce.

3. **Friction costs**: Each breaker event involves selling ~50% of positions, incurring additional commission and slippage.

4. **False positives**: Not all SPY -4% days lead to extended drawdowns. Many are single-day corrections followed by quick recoveries.

### 3.2 Why Add-on Hurts Drawdown

1. **Mean reversion**: Strong winners (+6%) often revert to mean. Adding more shares amplifies the reversal loss.

2. **Concentration risk**: Add-on increases position size beyond the original sizing formula, breaking the diversification benefit.

3. **Adverse selection**: Positions that trigger add-on criteria are often near local highs, not optimal entry points for additional capital.

---

## 4. Recommendations

### 4.1 For Live Trading

**Option A: Keep Base Strategy**
- 629.1% Total Return, -10.5% MDD
- Best risk-adjusted returns (Sharpe 2.22)
- Simple to execute

**Option B: Conservative Breaker (SPY -5% only)**
- 571.3% Total Return, -9.9% MDD
- Slightly better MDD protection
- Only 13 triggers over 7.7 years (~1.7/year)
- Useful as psychological safety net

### 4.2 Why NOT to Use These Mechanisms

1. **Breaker**: The 200% leverage strategy already has VIX-based regime throttling built in (though disabled in Return-Max config). Re-enabling VIX throttle achieves similar protection with better timing.

2. **Add-on**: The earnings strategy has a fixed 30-day holding period. Adding to winners mid-trade creates asymmetric risk without improving expected return.

### 4.3 Alternative Protection Methods

Instead of breaker/add-on, consider:

1. **Re-enable VIX regime throttle**: Target gross 1.0 when VIX > 22
2. **Tighten stop-loss**: Use 10-12% stop instead of 15%
3. **Reduce leverage**: Target gross 1.5 instead of 2.0

---

## 5. Configuration Reference

### Base (Recommended)
```python
target_gross_normal=2.0
target_gross_riskoff=1.0  # with VIX throttle
target_gross_stress=0.0
# No breaker
# No add-on
```

### Conservative Breaker (Optional)
```python
breaker_spy_threshold=0.05  # -5%
breaker_vix_threshold=0.40  # +40%
breaker_target_gross=1.0
breaker_cooldown_days=3
addon_enabled=False
```

---

## Appendix: Implementation Details

### Breaker Logic
1. Trigger check at session start (SPY return OR VIX change)
2. If triggered, reduce all positions proportionally to target gross at open
3. Block new entries for cooldown period
4. After cooldown, normal operation resumes

### Add-on Logic
1. Check positions held >= 5 sessions
2. If unrealized PnL >= trigger, add 0.33x of original position
3. Respect per_trade_cap and max_gross constraints
4. Max 1 add-on per trade

### Test Files
- Test runner: `run_breaker_addon_test.py`
- Results CSV: `breaker_addon_test_results.csv`
- Backtester: `backtester_v32.py`
