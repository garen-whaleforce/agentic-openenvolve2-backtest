#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_scores_to_signals.py

合併 validation_results 中的 direction_score 和其他特徵到 signals 檔案
用於排序和衛星門檻功能
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="合併 direction_score 到訊號檔案")
    parser.add_argument("--signals", required=True, help="原始訊號檔案")
    parser.add_argument("--validation", required=True, help="驗證結果檔案 (有 direction_score)")
    parser.add_argument("--output", default=None, help="輸出檔案 (預設: signals_with_scores.csv)")
    args = parser.parse_args()

    signals_path = Path(args.signals)
    validation_path = Path(args.validation)

    print(f"載入訊號: {signals_path}")
    signals = pd.read_csv(signals_path)
    print(f"  訊號數: {len(signals)}")

    print(f"\n載入驗證結果: {validation_path}")
    validation = pd.read_csv(validation_path)
    print(f"  驗證記錄數: {len(validation)}")

    # 準備合併欄位
    # 訊號用 reaction_date，驗證結果用 year/quarter
    # 需要建立對應關係

    # 先處理訊號
    signals["symbol"] = signals["symbol"].astype(str).str.upper()
    if "reaction_date" in signals.columns:
        signals["reaction_date"] = pd.to_datetime(signals["reaction_date"])

    # 處理驗證結果
    validation["symbol"] = validation["symbol"].astype(str).str.upper()
    validation["year"] = validation["year"].astype(int)
    validation["quarter"] = validation["quarter"].astype(int)

    # 如果訊號有 year/quarter，直接用
    if "year" in signals.columns and "quarter" in signals.columns:
        signals["year"] = signals["year"].astype(int)
        signals["quarter"] = signals["quarter"].astype(int)
        merge_keys = ["symbol", "year", "quarter"]
    else:
        # 從 reaction_date 推算 year/quarter
        signals["year"] = signals["reaction_date"].dt.year
        signals["quarter"] = ((signals["reaction_date"].dt.month - 1) // 3 + 1).astype(int)
        merge_keys = ["symbol", "year", "quarter"]

    print(f"\n合併欄位: {merge_keys}")

    # 選擇要合併的欄位
    score_columns = [
        "direction_score",
        "le_DirectionScore",
        "le_HardPositivesCount",
        "le_HardVetoCount",
        "eps_surprise",
        "earnings_day_return",
        "pre_earnings_5d_return",
        "risk_code",
        "sector",
    ]

    # 只保留存在的欄位
    available_columns = [c for c in score_columns if c in validation.columns]
    print(f"合併欄位: {available_columns}")

    # 合併
    validation_subset = validation[merge_keys + available_columns].drop_duplicates(subset=merge_keys)

    merged = signals.merge(
        validation_subset,
        on=merge_keys,
        how="left"
    )

    # 檢查合併率
    matched = merged["direction_score"].notna().sum()
    total = len(merged)
    match_rate = matched / total * 100

    print(f"\n合併結果:")
    print(f"  總訊號數:     {total}")
    print(f"  成功合併:     {matched}")
    print(f"  合併率:       {match_rate:.1f}%")

    if match_rate < 90:
        print(f"\n  警告: 合併率低於 90%，可能有資料不匹配問題")

        # 找出未匹配的
        unmatched = merged[merged["direction_score"].isna()][["symbol", "year", "quarter", "reaction_date"]].head(10)
        print(f"\n  未匹配樣本 (前 10 筆):")
        print(unmatched.to_string())

    # 計算 direction_score 統計
    if matched > 0:
        scores = merged["direction_score"].dropna()
        print(f"\n  direction_score 統計:")
        print(f"    平均: {scores.mean():.2f}")
        print(f"    中位: {scores.median():.2f}")
        print(f"    P25:  {scores.quantile(0.25):.2f}")
        print(f"    P50:  {scores.quantile(0.50):.2f}")
        print(f"    P75:  {scores.quantile(0.75):.2f}")
        print(f"    P80:  {scores.quantile(0.80):.2f}")

    # 輸出
    output_path = Path(args.output) if args.output else signals_path.parent / "signals_with_scores.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n已儲存: {output_path}")


if __name__ == "__main__":
    main()
