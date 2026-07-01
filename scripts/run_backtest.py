import os
import sys
import glob
import zipfile
import gc
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TmomAlpha.factors import calculate_factors
from TmomAlpha.portfolio_optimizer import optimize_portfolio_weights

def load_all_data(data_dir):
    print("Scanning data directory...")
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    print(f"Found {len(zip_files)} ZIP files. Loading daily data...")
    
    all_dfs = []
    for idx, f in enumerate(zip_files):
        symbol = os.path.basename(f).replace(".zip", "")
        try:
            with zipfile.ZipFile(f) as z:
                with z.open(f"{symbol}.csv") as csv_f:
                    df = pd.read_csv(csv_f, names=["date", "open", "high", "low", "close", "volume"])
                    df["symbol"] = symbol
                    all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {symbol}: {e}")
            
        if (idx + 1) % 200 == 0 or (idx + 1) == len(zip_files):
            print(f"Loaded {idx + 1}/{len(zip_files)} symbols...")
            
    if not all_dfs:
        return None
        
    big_df = pd.concat(all_dfs, ignore_index=True)
    big_df["date"] = pd.to_datetime(big_df["date"])
    
    # Scale back prices (since they are multiplied by 10000 in ZIP files)
    for col in ["open", "high", "low", "close"]:
        big_df[col] = big_df[col] / 10000.0
        
    big_df = big_df.set_index(["symbol", "date"]).sort_index()
    return big_df

def run_simulation(factors_df, labels, close_matrix, portfolio_size=30, min_expected_return=0.0):
    print(f"\n--- Running Simulation for N = {portfolio_size} ---")
    
    # Backtest dates: from 2018-01-02 to 2026-05-28
    all_dates = sorted(close_matrix.index)
    backtest_dates = [d for d in all_dates if d >= pd.Timestamp(2018, 1, 1) and d <= pd.Timestamp(2026, 5, 28)]
    
    # Load metadata JSON databases
    industry_map_path = os.path.join(project_root, "data", "stock_industry_map.json")
    shares_path = os.path.join(project_root, "data", "stock_shares_outstanding.json")
    
    with open(industry_map_path, "r", encoding="utf-8") as f:
        stock_industry_map = json.load(f)
    with open(shares_path, "r", encoding="utf-8") as f:
        stock_shares_outstanding = json.load(f)
    
    # Initial state
    cash = 10000000.0  # 10,000,000 RMB
    shares = {}  # {symbol: shares}
    
    # Track daily portfolio value
    daily_equity = []
    dates_tracked = []
    
    # Model tracking
    active_model = None
    last_year = 0
    last_month = 0
    
    # We will slice factors_df and labels inside the loop for training
    idx = pd.IndexSlice
    
    for idx_date, current_date in enumerate(backtest_dates):
        current_year = current_date.year
        current_month = current_date.month
        
        # 1. Update daily portfolio value (at the end of day close prices)
        portfolio_value = cash
        for symbol, qty in shares.items():
            if qty > 0:
                price = close_matrix.at[current_date, symbol]
                if pd.notna(price):
                     portfolio_value += qty * price
                     
        daily_equity.append(portfolio_value)
        dates_tracked.append(current_date)
        
        # 2. Check for Annual Retraining (triggered at the start of each new year in backtest)
        if current_year != last_year:
            # Training end date is the end of previous year
            train_end_date = pd.Timestamp(year=current_year - 1, month=12, day=31)
            
            # To strictly avoid look-ahead bias (data leakage) from overlapping 21-day forward labels,
            # we roll back the training end date by 21 trading days (approx. 30 calendar days).
            train_label_end_date = train_end_date - pd.Timedelta(days=30)
            
            # Retrieve training slice
            X_train = factors_df.loc[idx[:, :train_label_end_date], :]
            y_train = labels.loc[idx[:, :train_label_end_date]]
            train_data = pd.concat([X_train, y_train], axis=1).dropna()
            
            if len(train_data) >= 1000:
                print(f"[{current_date.strftime('%Y-%m-%d')}] Retraining LightGBM model up to {train_label_end_date.strftime('%Y-%m-%d')} (samples: {len(train_data)})...")
                X = train_data.drop(columns=["target"])
                y = train_data["target"]
                
                # Fit LightGBM with recommended hyperparams and GPU fallback
                active_model = lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.0421,
                    num_leaves=210,
                    max_depth=8,
                    reg_alpha=205.7,
                    reg_lambda=581.0,
                    colsample_bytree=0.888,
                    subsample=0.879,
                    random_state=42,
                    device='gpu',
                    n_jobs=12,
                    verbose=-1
                )
                try:
                    active_model.fit(X, y)
                except Exception as e:
                    print(f"LightGBM GPU training failed: {e}. Fallback to CPU.")
                    active_model = lgb.LGBMRegressor(
                        n_estimators=150,
                        learning_rate=0.0421,
                        num_leaves=210,
                        max_depth=8,
                        reg_alpha=205.7,
                        reg_lambda=581.0,
                        colsample_bytree=0.888,
                        subsample=0.879,
                        random_state=42,
                        device='cpu',
                        n_jobs=12,
                        verbose=-1
                    )
                    active_model.fit(X, y)
            else:
                print(f"[{current_date.strftime('%Y-%m-%d')}] Warning: Not enough data for retraining. Skipping.")
                
            last_year = current_year
            
        # 3. Check for Monthly Rebalancing (triggered at the start of each new month in backtest)
        if current_month != last_month:
            last_month = current_month
            
            if active_model is not None:
                # Get current factors for this date
                try:
                    current_factors = factors_df.xs(current_date, level="date")
                except KeyError:
                    # No factors for this date (non-trading day or missing date)
                    continue
                    
                if not current_factors.empty:
                    # Find symbols that are active and have valid close prices today
                    active_tickers = [sym for sym in current_factors.index if sym in close_matrix.columns and pd.notna(close_matrix.at[current_date, sym])]
                    
                    if active_tickers:
                        X_predict = current_factors.loc[active_tickers]
                        preds = active_model.predict(X_predict)
                        pred_series = pd.Series(preds, index=active_tickers)
                        
                        # Sort and filter (absolute momentum: predicted return > 0)
                        sorted_preds = pred_series.sort_values(ascending=False)
                        positive_preds = sorted_preds[sorted_preds > min_expected_return]
                        
                        # 计算过去 60 天的滚动收益率矩阵作为协方差收缩输入
                        history_prices = close_matrix.loc[:current_date].tail(60)
                        historical_returns = history_prices.pct_change().dropna(how='all')
                        
                        # 凸优化选股和权重求解 (使用全市场 active_tickers 进行 Barra 中性化，并修剪至 portfolio_size)
                        opt_weights = optimize_portfolio_weights(
                            predictions=pred_series,
                            current_factors=current_factors,
                            close_prices=close_matrix.loc[current_date],
                            float_shares_map=stock_shares_outstanding,
                            stock_industry_map=stock_industry_map,
                            historical_returns=historical_returns,
                            max_weight=0.05,
                            active_tickers=active_tickers,
                            max_portfolio_size=portfolio_size,
                            style_bound=None,
                            industry_bound=None
                        )
                        
                        if opt_weights is not None:
                            selected_tickers = list(opt_weights.keys())
                            weights = opt_weights
                        else:
                            selected_tickers = positive_preds.head(portfolio_size).index.tolist()
                            weights = {t: 1.0 / portfolio_size for t in selected_tickers}
                            
                        # Rebalance portfolio
                        # A. Liquidate symbols not in selected list
                        new_shares = {}
                        for sym in list(shares.keys()):
                            qty = shares[sym]
                            if qty > 0 and sym not in selected_tickers:
                                close_p = close_matrix.at[current_date, sym]
                                if pd.isna(close_p) or close_p <= 0:
                                    # Keep holding if suspended/unlisted
                                    new_shares[sym] = qty
                                    continue
                                trade_val = qty * close_p
                                fee = trade_val * 0.0015
                                cash += trade_val - fee
                                shares[sym] = 0.0
                                
                        # B. Allocate to selected symbols using computed weights (with 1% cash buffer for fees)
                        if selected_tickers:
                            for sym in selected_tickers:
                                close_p = close_matrix.at[current_date, sym]
                                if pd.isna(close_p) or close_p <= 0:
                                    # Keep holding if suspended
                                    if sym in shares:
                                        new_shares[sym] = shares[sym]
                                    continue
                                weight = weights[sym]
                                target_alloc = weight * portfolio_value * 0.99
                                target_qty = target_alloc / close_p
                                current_qty = shares.get(sym, 0.0)
                                
                                trade_qty = target_qty - current_qty
                                trade_val = trade_qty * close_p
                                fee = abs(trade_val) * 0.0015
                                
                                cash = cash - trade_val - fee
                                new_shares[sym] = target_qty
                                
                        # Update holding shares
                        shares = new_shares
                        
    # End of simulation
    # Calculate performance metrics
    equity_series = pd.Series(daily_equity, index=dates_tracked)
    
    total_days = len(equity_series)
    years = total_days / 244.0 # Chinese stock market has about 244 trading days per year
    
    cum_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    ann_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1 if years > 0 else 0
    
    daily_rets = equity_series.pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(244)
    
    # Sharpe (assumed 2% risk-free rate)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    running_max = equity_series.cummax()
    drawdowns = (equity_series / running_max) - 1
    max_dd = drawdowns.min()
    
    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    print(f"Results: Ann Return = {ann_ret:.2%}, Max DD = {max_dd:.2%}, Sharpe = {sharpe:.2f}, Calmar = {calmar:.2f}")
    
    return equity_series, {
        "cum_return": cum_ret,
        "annual_return": ann_ret,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar
    }

def main():
    data_dir = os.path.join(project_root, "data", "equity", "usa", "daily")
    big_df = load_all_data(data_dir)
    
    if big_df is None or len(big_df) == 0:
        print("Error: No data loaded.")
        return
        
    print(f"Total data shape: {big_df.shape}")
    
    # Compute factors using CLAUDE.md's orthogonalize=False constraint
    print("Calculating factors on the complete dataset (orthogonalize=False)...")
    factors_df = calculate_factors(big_df, orthogonalize=False)
    factors_df = factors_df.astype(np.float32)
    print(f"Factors computed. Shape: {factors_df.shape}")
    
    # Compute labels
    print("Calculating future return labels...")
    close_matrix = big_df["close"].unstack(level=0).ffill()
    future_ret = close_matrix.shift(-21) / close_matrix - 1
    labels = future_ret.stack().swaplevel(0, 1).sort_index().astype(np.float32)
    labels.name = "target"
    
    # Clean memory
    del big_df
    gc.collect()
    
    # We will simulate for N = 30 (our recommended portfolio size)
    n = 30
    eq, stats = run_simulation(factors_df, labels, close_matrix, portfolio_size=n)
    
    # Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(eq / 10000000.0, label=f"GBD Strategy (N=30, Sharpe={stats['sharpe']:.2f})")
    
    # Also load benchmark CSI 1000 index to plot for comparison
    bench_path = os.path.join(project_root, "data", "csi1000_index.csv")
    if os.path.exists(bench_path):
        try:
            bench_df = pd.read_csv(bench_path)
            bench_df["date"] = pd.to_datetime(bench_df["date"])
            bench_df = bench_df.set_index("date").sort_index()
            bench_series = bench_df["close"].reindex(eq.index).ffill()
            bench_norm = bench_series / bench_series.iloc[0]
            plt.plot(bench_norm, label="CSI 1000 Index", color="gray", alpha=0.7)
        except Exception as e:
            print(f"Failed to load benchmark: {e}")
            
    plt.title("CSI 1000 Multi-Factor GBDT Strategy (2018-2026)")
    plt.xlabel("Date")
    plt.ylabel("Normalized NAV (Base = 1.0)")
    plt.grid(True)
    plt.legend()
    
    # Save chart to our figures directory
    figures_dir = os.path.join(project_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "media__custom_factors_nav.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved equity curve plot to {plot_path}")
    
    # Write a detailed stats JSON for quantstats report generator
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    detailed_stats_path = os.path.join(reports_dir, "detailed_quant_stats.json")
    # Store standard structure
    export_data = {
        "30": {
            "dates": [d.strftime("%Y-%m-%d") for d in eq.index],
            "equity_curve": eq.tolist()
        }
    }
    with open(detailed_stats_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=4)
    print(f"Saved detailed stats to {detailed_stats_path}")
    
    # Write summary metrics JSON for Sharpe gate validation
    metrics_path = os.path.join(reports_dir, "backtest_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved backtest metrics JSON to {metrics_path}")
    
    # Print metrics
    print("\n" + "="*50)
    print("BACKTEST PERFORMANCE SUMMARY (WITH NEW CUSTOM FACTORS)")
    print("="*50)
    print(f"Annualized Return: {stats['annual_return']:.2%}")
    print(f"Max Drawdown:      {stats['max_dd']:.2%}")
    print(f"Sharpe Ratio:      {stats['sharpe']:.2f}")
    print(f"Calmar Ratio:      {stats['calmar']:.2f}")
    print("="*50)
    
if __name__ == "__main__":
    main()
