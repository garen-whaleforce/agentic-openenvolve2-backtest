# V3.2 Return-Max Strategy - Board Report

**Date:** 2025-01-04

**Period:** 2017-02 to 2024-10 (7.7 years)

---

## Executive Summary

| Metric | Return-Max | Baseline (w/ VIX) |
|---|---:|---:|
| **Total Return** | **1168%** | 629% |
| **CAGR** | **38.1%** | 27.6% |
| **Sharpe Ratio** | **1.94** | 2.19 |
| **Max Drawdown** | **-17.7%** | -9.9% |
| Win Rate | 76.9% | 76.9% |
| Total Trades | 251 | 218 |

### Key Finding

The Return-Max configuration (no VIX regime throttle) delivers **significantly higher returns** 
(1168% vs 629%) at the cost of **higher drawdowns** (-17.7% vs -9.9%). The difference is due to:

1. **Higher consistent leverage** - 200% gross exposure in all periods vs regime-adjusted
2. **More trades executed** - 251 vs 218 (no trades blocked by RISK_OFF/STRESS regimes)
3. **Higher volatility** - 17.4% vs 11.3% annual

---

## Configuration Comparison

| Parameter | Return-Max | VIX-Throttled |
|---|:---:|:---:|
| Target Gross (NORMAL) | 200% | 200% |
| Target Gross (RISK_OFF) | 200% | 100% |
| Target Gross (STRESS) | 200% | 0% |
| VIX Regime Throttle | OFF | ON |
| CAP/MaxPos/Sizing | 24/24/12 | 24/24/12 |
| Costs | 20 bps | 20 bps |

---

## Risk Assessment

### Drawdown Characteristics

| Episode | Period | Max DD | Duration | Recovery |
|---|:---|---:|---:|---:|
| 2022 Bear Market | May-Nov 2022 | -17.7% | 189d | Full |
| 2022 Year-End | Dec 2022-Mar 2023 | -14.3% | 119d | Full |
| 2018-2020 Range | Dec 2018-Jun 2020 | -14.2% | 549d | Full |

### Worst Single Trades

| Trade | Period | Return | Comment |
|---|:---|---:|:---|
| ABNB | May-Jun 2022 | -39% | Earnings gap + bear market |
| KIM | Apr-Jun 2022 | -26% | REIT sector weakness |
| AAPL | Jan-Mar 2020 | -24% | COVID crash |

---

## Year-by-Year Returns

| Year | Return | Sharpe | MDD | Assessment |
|---:|---:|---:|---:|:---|
| 2017 | +35.8% | 3.21 | -3.0% | Strong start |
| 2018 | +10.0% | 0.56 | -14.2% | Challenging |
| 2019 | +8.2% | 1.77 | -1.9% | Low activity |
| 2020 | +31.9% | 1.74 | -6.8% | COVID recovery |
| 2021 | +63.0% | 3.12 | -6.1% | Exceptional |
| 2022 | +18.7% | 0.78 | -17.7% | Bear market survivor |
| 2023 | +102.1% | 3.87 | -6.4% | Best year |
| 2024 | +53.2% | 3.20 | -5.9% | Strong YTD |

### Return Distribution

- **6/8 years** delivered >20% returns
- **Worst year (2019):** Still positive at +8.2%
- **Best year (2023):** +102% with Sharpe 3.87
- **Bear market (2022):** +18.7% despite MDD -17.7%

---

## Recommendations

### For Maximum Return (Risk Tolerant)

Use **Return-Max configuration** (no VIX throttle):
- Expected CAGR: ~38%
- Expected MDD: ~-18%
- Best for: Long-term compounding, can withstand drawdowns

### For Risk-Adjusted Return (Risk Averse)

Use **VIX-Throttled configuration**:
- Expected CAGR: ~27%
- Expected MDD: ~-10%
- Best for: Lower volatility preference, smoother equity curve

### Future Optimization Path

As suggested in the analysis:

1. **Portfolio-level breaker** - Quick de-leverage on market shocks (affects existing positions)
2. **Winner add-on** - Add to winning positions for higher effective leverage
3. **Cash overlay** - Deploy idle cash in T-bills or SPY

These paths offer higher ROI than the D6 regime/score-weight/DD-delever tested in this round.

---

*Report generated: 2025-01-04*
