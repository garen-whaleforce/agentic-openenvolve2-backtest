# Signals Analysis Summary (2017-2025)

This document summarizes the earnings signals analyzed by the v1.1 Live-Safe strategy.

## Overview

| Metric | Value |
|--------|-------|
| Total Signals Analyzed | 3,388 |
| Correct Predictions | 2,046 |
| Overall Prediction Accuracy | 60.59% |
| Long Trades Executed | 266 |

## Signal Distribution by Sector

| Sector | Signals | Percentage |
|--------|---------|------------|
| Technology | 714 | 21.1% |
| Industrials | 495 | 14.6% |
| Consumer Cyclical | 464 | 13.7% |
| Healthcare | 417 | 12.3% |
| Financial Services | 380 | 11.2% |
| Consumer Defensive | 209 | 6.2% |
| Energy | 182 | 5.4% |
| Communication Services | 158 | 4.7% |
| Basic Materials | 150 | 4.4% |
| Real Estate | 123 | 3.6% |
| Utilities | 96 | 2.8% |

## Long Trade Tier Breakdown

The strategy uses a tiered system for trade qualification:

| Tier | Trades | Description |
|------|--------|-------------|
| D7_CORE | 201 | Highest conviction - Direction Score ≥ 7, no vetoes |
| D6_STRICT | 65 | High conviction - Direction Score ≥ 6, strict criteria |

**Total Long Trades: 266**

## Signal Analysis Fields

Each signal contains the following key metrics:

### LLM Analysis Output
- **prediction**: UP or DOWN direction forecast
- **confidence**: Model confidence (0-1 scale)
- **direction_score**: Composite score (1-10) indicating conviction
- **summary**: Detailed analysis text

### Fundamental Indicators (Hard Positives)
- **GuidanceRaised**: Company raised forward guidance
- **DemandAcceleration**: Revenue/order growth accelerating
- **MarginExpansion**: Gross or operating margins improving
- **FCFImprovement**: Free cash flow improving
- **VisibilityImproving**: Forward visibility getting better

### Risk Indicators (Hard Vetoes)
- **GuidanceCut**: Company cut forward guidance
- **DemandSoftness**: Revenue/orders declining
- **MarginWeakness**: Margin compression
- **CashBurn**: Negative free cash flow trends
- **VisibilityWorsening**: Forward visibility deteriorating
- **PricedInRisk**: Large pre-earnings run-up (sell-the-news risk)

## Trade Qualification Logic

A signal qualifies for a long trade when:

### D7_CORE Tier (Highest Quality)
- Direction Score ≥ 7
- No hard vetoes
- Risk code = Low or Medium

### D6_STRICT Tier (High Quality)
- Direction Score ≥ 6
- ≥ 4 hard positives
- No hard vetoes
- No high risk flags

## Market Data Inputs

Each signal also includes:
- **eps_surprise**: Actual vs estimated EPS difference
- **earnings_day_return**: Stock return on earnings day
- **pre_earnings_5d_return**: 5-day return before earnings

## Strategy Performance by Signal Type

| Category | Win Rate |
|----------|----------|
| D7_CORE trades | ~86% |
| D6_STRICT trades | ~84% |
| All predictions | 60.59% |

The tiered filtering dramatically improves win rate from 60% (all predictions) to 85%+ (qualified trades only).

## Key Insights

1. **Technology sector** provides the most signals (21%), reflecting its earnings volatility
2. **Tiered qualification** is essential - only 7.8% of signals (266/3388) become trades
3. **Hard vetoes** effectively filter out poor opportunities
4. **Win rate improvement** from 60% to 85%+ demonstrates filter effectiveness
5. **Sector diversification** provides broad market exposure
