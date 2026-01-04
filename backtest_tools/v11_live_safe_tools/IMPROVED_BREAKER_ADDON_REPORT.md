# Improved Breaker + Add-on Test Report

**Date**: 2026-01-04
**Test Period**: 2017-01 to 2025-08 (~8.5 years)
**Base Strategy**: D7_CORE + D6_STRICT with 200% leverage

---

## Executive Summary

按照你的建議測試了改進版本：

| 變體 | 終值倍數 | vs Base | MDD | 評價 |
|------|---------|---------|-----|------|
| **Base** | 7.99x | - | -10.5% | 基準 |
| Freeze 1-day | 7.99x | +0.0% | -10.5% | **無損失** ✓ |
| Freeze 3-day | 7.65x | -4.2% | -10.5% | 小損失 |
| **Pullback 3%** | **8.75x** | **+9.6%** | -11.5% | **最佳回報** ✓ |
| Pullback 2% | 8.52x | +6.7% | -11.4% | 次佳 |
| Reduce 3-day | 7.18x | -10.0% | -10.0% | 損失大 |
| Extension 6% | 7.88x | -1.4% | -11.7% | MDD 更差 |

### 關鍵發現

1. **Freeze vs Reduce**: Freeze 模式比 Reduce 省下 +6.5% 終值（因為避免了「賣在恐慌低點」）

2. **Pullback vs Extension**: Pullback 3% 比 Extension 提升 +11.0% 終值（因為「回檔買」優於「追高買」）

3. **最佳組合**: **Pullback Add-on 3%** 達到 8.75x 終值（+9.6% vs Base），但 MDD 惡化 1%

---

## 測試結果

### 完整結果表

| Config | Total Return | Terminal | CAGR | Sharpe | MDD | Breakers | Add-ons |
|--------|-------------|----------|------|--------|-----|----------|---------|
| 1_Base | 698.6% | 7.99x | 27.1% | 2.17 | -10.5% | 0 | 0 |
| 2_Freeze_1day | 698.6% | 7.99x | 27.1% | 2.17 | -10.5% | 27 | 0 |
| 3_Freeze_3day | 665.2% | 7.65x | 26.5% | 2.13 | -10.5% | 23 | 0 |
| 4_Pullback_2pct | 752.4% | 8.52x | 28.1% | 2.11 | -11.4% | 0 | 141 |
| **5_Pullback_3pct** | **775.3%** | **8.75x** | **28.5%** | **2.14** | -11.5% | 0 | 120 |
| 6_Freeze1_Pullback3 | 775.3% | 8.75x | 28.5% | 2.14 | -11.5% | 27 | 120 |
| 7_Reduce_3day | 618.4% | 7.18x | 25.6% | 2.10 | -10.0% | 23 | 0 |
| 8_Extension_6pct | 687.6% | 7.88x | 26.9% | 2.04 | -11.7% | 0 | 152 |

---

## 機制分析

### Freeze Only Breaker

**設計改進**：
- 原版：觸發時強制減倉到 target gross（賣在恐慌點）
- 改進：只凍結新開倉，不動既有倉位

**結果**：
- Freeze 1-day: **完全無損**（7.99x = Base）
- Freeze 3-day: 損失 4.2%（因為錯過 3 天的新信號）
- Reduce 3-day: 損失 10.0%（因為恐慌點賣出）

**結論**：如果要用 breaker，**只用 Freeze 1-day**，可以得到心理安全感但不損失回報。

### Pullback Add-on

**設計改進**：
- 原版 (Extension)：浮盈 ≥6% 就加碼（追高）
- 改進 (Pullback)：曾經達到 +6%，且從高點回撤 ≥3% 才加碼（回檔買）

**結果**：
- Pullback 3%: +9.6% 終值，Add-ons: 120 次
- Pullback 2%: +6.7% 終值，Add-ons: 141 次
- Extension 6%: -1.4% 終值，Add-ons: 152 次

**為什麼 Pullback 有效**：
1. 回檔 3% 後買入，避開了區間高點
2. 仍要求浮盈 > 0（不買虧損倉位）
3. 較少的加倉次數（120 vs 152）= 更精選的進場點

**MDD 代價**：
- Pullback 的 MDD 從 -10.5% 惡化到 -11.5%（增加 1%）
- 這是因為加倉增加了曝險

---

## 建議配置

### Option A: 純粹主義（最低複雜度）

維持 Base，不加任何機制：
```python
# 無 breaker
# 無 add-on
```
- 終值: 7.99x
- MDD: -10.5%
- 簡單、可審計、已驗證

### Option B: 保守增強（推薦）

只加 Freeze 1-day breaker（心理安全網）：
```python
breaker_spy_threshold = 0.04
breaker_vix_threshold = 0.30
breaker_cooldown_days = 1
breaker_mode = "freeze"
# 無 add-on
```
- 終值: 7.99x（無損）
- MDD: -10.5%
- 大跌時不開新倉，但不強迫賣出

### Option C: 積極增強（接受 1% MDD 換 9.6% 回報）

加 Pullback Add-on 3%：
```python
addon_enabled = True
addon_mode = "pullback"
addon_trigger_pct = 0.06
addon_pullback_pct = 0.03
addon_mult = 0.33
# 無 breaker（或 freeze 1-day）
```
- 終值: 8.75x (+9.6%)
- MDD: -11.5% (-1%)
- Sharpe: 2.14（接近 Base 的 2.17）

---

## 實作細節

### Freeze Mode 程式碼位置

`backtester_v32.py` 第 564 行：
```python
# If mode is "freeze", skip position reduction (only block new entries)
if breaker_active and len(positions) > 0 and config.breaker_mode == "reduce":
    # ... reduce positions ...
```

### Pullback Mode 程式碼位置

`backtester_v32.py` 第 683-693 行：
```python
elif config.addon_mode == "pullback":
    # Pullback mode: add when (1) ever reached trigger AND (2) pulled back from high
    if not pos.get("ever_reached_trigger", False):
        continue
    pullback_from_high = (max_price - px_now) / max_price if max_price > 0 else 0
    if pullback_from_high < config.addon_pullback_pct:
        continue
    if unrealized_pnl < 0:  # Still require positive PnL
        continue
```

---

## 與原始版本對比

| 機制 | 原始版本 | 改進版本 | 改進效果 |
|------|---------|---------|---------|
| Breaker | Reduce（強制減倉）| Freeze（只停新倉）| +6.5% 終值 |
| Add-on | Extension（追高）| Pullback（回檔）| +11.0% 終值 |

---

## 結論

你的建議完全正確：

1. **Breaker 改成 Freeze**：避免「賣在恐慌低點」，終值損失從 -10% 降到 0%（1-day cooldown）

2. **Add-on 改成 Pullback**：避免「追高買入」，終值從 -1.4% 提升到 +9.6%

3. **最終建議**：
   - 保守：Base 或 Freeze 1-day
   - 積極：Pullback 3% Add-on（接受 1% MDD 代價換 9.6% 回報）

---

## 附錄：測試檔案

- 測試程式：`run_improved_breaker_addon_test.py`
- 結果 CSV：`improved_breaker_addon_results.csv`
- Backtester：`backtester_v32.py`（已更新支援 `breaker_mode` 和 `addon_mode`）
