import os
import sys
import json
import pandas as pd
import quantstats_lumi as qs

# Get project root path
project_root = os.path.dirname(os.path.abspath(__file__))

def main():
    json_path = os.path.join(project_root, "reports", "detailed_quant_stats.json")
    bench_path = os.path.join(project_root, "data", "csi1000_index.csv")
    figures_dir = os.path.join(project_root, "figures")
    reports_dir = os.path.join(project_root, "reports")

    # Make sure output dirs exist
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist. Please run scripts/run_backtest.py first.")
        return

    print("Loading detailed quant stats JSON...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use the recommended portfolio size of 30
    strategy_data = data["30"]
    dates = pd.to_datetime(strategy_data["dates"])
    equity_curve = pd.Series(strategy_data["equity_curve"], index=dates)
    
    # Calculate daily returns for the strategy
    strategy_returns = equity_curve.pct_change().dropna()
    strategy_returns.name = "Strategy (30)"

    print("Loading benchmark CSI 1000 index data...")
    bench_df = pd.read_csv(bench_path)
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_df = bench_df.set_index("date").sort_index()
    
    # Calculate daily returns for the benchmark aligned with strategy dates
    bench_series = bench_df["close"].reindex(dates).ffill()
    bench_returns = bench_series.pct_change().dropna()
    bench_returns.name = "CSI 1000 Index"

    # Align dates
    common_idx = strategy_returns.index.intersection(bench_returns.index)
    strategy_returns = strategy_returns.loc[common_idx]
    bench_returns = bench_returns.loc[common_idx]

    print("Generating QuantStats HTML Tear Sheet...")
    report_html_path = os.path.join(reports_dir, "quantstats_report_30.html")
    qs.reports.html(
        strategy_returns, 
        benchmark=bench_returns, 
        output=report_html_path, 
        title="CSI 1000 GBDT Strategy (N=30) Performance Analysis"
    )
    print(f"HTML report saved to {report_html_path}")

    # Generate and save specific figures
    print("Plotting and saving figures...")
    
    # 1. Cumulative returns vs Benchmark (Overwrites media__cumulative_nav.png)
    nav_fig_path = os.path.join(figures_dir, "media__cumulative_nav.png")
    qs.plots.returns(strategy_returns, benchmark=bench_returns, savefig=nav_fig_path, show=False)
    print(f"Cumulative NAV plot saved to {nav_fig_path}")

    # 2. Monthly returns Heatmap (Overwrites media__monthly_heatmap.png)
    heatmap_fig_path = os.path.join(figures_dir, "media__monthly_heatmap.png")
    qs.plots.monthly_heatmap(strategy_returns, savefig=heatmap_fig_path, show=False)
    print(f"Monthly Heatmap saved to {heatmap_fig_path}")

    # 3. Drawdown Underwater Plot (Overwrites media__underwater_drawdown.png)
    underwater_fig_path = os.path.join(figures_dir, "media__underwater_drawdown.png")
    qs.plots.drawdown(strategy_returns, savefig=underwater_fig_path, show=False)
    print(f"Underwater Drawdown plot saved to {underwater_fig_path}")

    # 4. Yearly returns comparison (Overwrites media__yearly_excess.png)
    yearly_fig_path = os.path.join(figures_dir, "media__yearly_excess.png")
    qs.plots.yearly_returns(strategy_returns, benchmark=bench_returns, savefig=yearly_fig_path, show=False)
    print(f"Yearly Returns comparison plot saved to {yearly_fig_path}")

    print("QuantStats generation complete!")

if __name__ == "__main__":
    main()
