# Total Return Optimization Grid Results

**Test Period:** 2017-02 to 2024-10

**Fixed Configuration:**
- CAP = 24, MaxPos = 24, Sizing = 12
- Target Gross: NORMAL=200%, RISK_OFF=100%, STRESS=0%
- Costs: 20 bps round-trip, Borrow: 6%
- Satellite Floor: 5.0 (earnings_day_return)

---

## Summary of Optimization Dimensions

### 1. D6 Regime Dynamic Weight
| Option | NORMAL | RISK_OFF | STRESS |
|---|:---:|:---:|:---:|
| A_Return | 100% | 50% | 0% |
| B_Balanced | 100% | 33% | 0% |
| C_Defensive | 50% | 0% | 0% |

### 2. Core Score-Weighted Allocation
- **EqWt**: Equal weight for all core positions
- **ScoreWt**: ±30% weight adjustment based on zscore of earnings_day_return

### 3. DD-Based De-Leveraging
- **NoDD**: No de-leveraging
- **DDLev**: Reduce allocation when DD > 10%/15%/20%

---

## Full Results

| Rank | Config | Total Return | CAGR | Sharpe | MDD | Win Rate |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | A_Return_EqWt_NoDD | 1168.2% | 38.1% | 2.19 | -17.7% | 76.9% |
| 2 | B_Balanced_EqWt_NoDD | 1168.2% | 38.1% | 2.19 | -17.7% | 76.9% |
| 3 | C_Defensive_EqWt_NoDD | 1166.7% | 38.1% | 2.19 | -17.7% | 76.8% |
| 4 | A_Return_ScoreWt_NoDD | 1159.4% | 38.0% | 2.17 | -18.9% | 76.8% |
| 5 | B_Balanced_ScoreWt_NoDD | 1159.4% | 38.0% | 2.17 | -18.9% | 76.8% |
| 6 | C_Defensive_ScoreWt_NoDD | 1158.1% | 38.0% | 2.17 | -18.9% | 76.8% |
| 7 | C_Defensive_ScoreWt_DDLev | 996.6% | 35.6% | 2.14 | -18.9% | 76.8% |
| 8 | C_Defensive_EqWt_DDLev | 994.1% | 35.6% | 2.14 | -17.7% | 76.9% |
| 9 | A_Return_ScoreWt_DDLev | 987.2% | 35.5% | 2.12 | -18.9% | 76.9% |
| 10 | B_Balanced_ScoreWt_DDLev | 987.2% | 35.5% | 2.12 | -18.9% | 76.9% |
| 11 | A_Return_EqWt_DDLev | 983.7% | 35.4% | 2.13 | -17.7% | 76.9% |
| 12 | B_Balanced_EqWt_DDLev | 983.7% | 35.4% | 2.13 | -17.7% | 76.9% |

---

## Key Insights

### 1. D6 Regime Dynamic - Minimal Impact
- All D6 regime options (A/B/C) produced nearly identical results
- This suggests D6 satellites are already rare in RISK_OFF/STRESS periods
- The satellite_floor=5.0 is already filtering most D6 signals

### 2. Score-Weighted Allocation - Slight Negative
- EqWt slightly outperforms ScoreWt (1168% vs 1159%)
- ScoreWt increases MDD (-18.9% vs -17.7%)
- Concentration in high-score signals may increase volatility

### 3. DD De-Leveraging - Trade-off
- DDLev reduces Total Return by ~15% (1168% → 984%)
- But MDD stays the same (-17.7%)
- DD de-leveraging kicks in too late to prevent the worst drawdowns

---

## Recommended Configurations

### For Maximum Return
**A_Return_EqWt_NoDD** or **B_Balanced_EqWt_NoDD**
- Total Return: 1168%
- CAGR: 38.1%
- Sharpe: 2.19
- MDD: -17.7%

### For Risk-Adjusted Return
**C_Defensive_EqWt_DDLev**
- Total Return: 994%
- CAGR: 35.6%
- Sharpe: 2.14
- MDD: -17.7%

---

*Generated: 2025-01-04*