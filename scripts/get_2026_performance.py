import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    json_path = os.path.join(project_root, "reports", "detailed_quant_stats.json")
    bench_path = os.path.join(project_root, "data", "csi1000_index.csv")

    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist. Please run scripts/run_backtest.py first.")
        return

    print("Loading detailed stats...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    strategy_data = data["30"]
    dates = pd.to_datetime(strategy_data["dates"])
    equity = pd.Series(strategy_data["equity_curve"], index=dates)

    print("Loading benchmark data...")
    bench_df = pd.read_csv(bench_path)
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_df = bench_df.set_index("date").sort_index()
    bench_series = bench_df["close"].reindex(dates).ffill()

    # Filter for 2026
    mask_2026 = dates.year == 2026
    dates_2026 = dates[mask_2026]
    
    if len(dates_2026) == 0:
        print("Error: No 2026 data found!")
        return

    equity_2026 = equity.loc[dates_2026]
    bench_2026 = bench_series.loc[dates_2026]

    # Calculate 2026 returns
    strat_start_val = equity_2026.iloc[0]
    strat_end_val = equity_2026.iloc[-1]
    strat_ret_2026 = (strat_end_val / strat_start_val) - 1

    bench_start_val = bench_2026.iloc[0]
    bench_end_val = bench_2026.iloc[-1]
    bench_ret_2026 = (bench_end_val / bench_start_val) - 1

    excess_ret_2026 = strat_ret_2026 - bench_ret_2026

    # Calculate 2026 Max Drawdown
    running_max_strat = equity_2026.cummax()
    dd_strat = (equity_2026 / running_max_strat) - 1
    max_dd_strat = dd_strat.min()

    running_max_bench = bench_2026.cummax()
    dd_bench = (bench_2026 / running_max_bench) - 1
    max_dd_bench = dd_bench.min()

    print("\n" + "="*50)
    print("2026 YTD PERFORMANCE SUMMARY (UP TO 2026-05-28)")
    print("="*50)
    print(f"Start Date: {dates_2026[0].strftime('%Y-%m-%d')}")
    print(f"End Date:   {dates_2026[-1].strftime('%Y-%m-%d')}")
    print("-"*50)
    print(f"Strategy Return:      {strat_ret_2026:.2%}")
    print(f"CSI 1000 Return:      {bench_ret_2026:.2%}")
    print(f"Excess Return (Alpha): {excess_ret_2026:.2%}")
    print("-"*50)
    print(f"Strategy Max DD:      {max_dd_strat:.2%}")
    print(f"CSI 1000 Max DD:      {max_dd_bench:.2%}")
    print("="*50)

if __name__ == "__main__":
    main()
